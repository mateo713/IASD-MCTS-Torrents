from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from core.battle_state import BattleState
from core.identifiers import extract_hp_chunk
from core.identifiers import normalize_stat_name
from core.identifiers import normalize_volatile_status_name
from core.models import ActionChoice, ActionType, BattlePhase, BattleSnapshot, OpponentConstraints
from infrastructure.showdown_parser import ParsedEvent
from engine.gen5_datasets import filter_candidate_sets_by_observed_damage


CHOICE_ITEMS: set[str] = {"choiceband", "choicespecs", "choicescarf"}


@lru_cache(maxsize=1)
def _load_moves_data() -> dict[str, dict[str, Any]]:
    data_path = Path(__file__).resolve().parents[3] / "foul-play" / "data" / "moves.json"
    try:
        with data_path.resolve().open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


@lru_cache(maxsize=1)
def _load_species_stats() -> dict[str, dict[str, Any]]:
    data_path = Path(__file__).resolve().parents[2] / "data" / "gen5" / "pokedex_stats.json"
    try:
        with data_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _move_volatile_status(move_data: dict[str, Any]) -> str:
    volatile_status = move_data.get("volatileStatus")
    if not volatile_status and isinstance(move_data.get("self"), dict):
        volatile_status = move_data["self"].get("volatileStatus")
    return normalize_volatile_status_name(str(volatile_status or ""))


def _species_base_speed(species_name: str | None) -> int:
    key = _normalize_species_key(species_name)
    if not key:
        return 0
    entry = _load_species_stats().get(key, {})
    base_stats = entry.get("base_stats", {})
    try:
        return int(base_stats.get("speed", 0) or 0)
    except Exception:
        return 0


class ShowdownStateTracker:
    """Room-scoped, deterministic tracker for observable battle state."""

    def __init__(self) -> None:
        self._battles: dict[str, BattleSnapshot] = {}
        # Pending switch mapping for events that arrive before we learn our side id.
        # Maps room_id -> {side_marker: species}
        self._pending_switches: dict[str, dict[str, str]] = {}
        # Per-room opponent knowledge keyed by normalized species id.
        self._opponent_constraints: dict[str, dict[str, OpponentConstraints]] = {}
        # Last user move metadata used to attribute explicit immunity outcomes.
        self._last_user_move_id: dict[str, str] = {}
        self._last_user_move_type: dict[str, str] = {}
        # Track crit tags that precede damage lines.
        self._pending_crit_target: dict[str, str] = {}
        # Track first mover per turn for speed-order inference.
        self._first_mover_by_turn: dict[str, tuple[int, bool]] = {}

    def get_or_create(self, room_id: str) -> BattleSnapshot:
        if room_id not in self._battles:
            self._battles[room_id] = BattleSnapshot(room_id=room_id, battle_state=BattleState(room_id=room_id))
        elif self._battles[room_id].battle_state is None:
            self._battles[room_id].battle_state = BattleState(room_id=room_id)
        return self._battles[room_id]

    def ingest(self, event: ParsedEvent) -> BattleSnapshot:
        battle = self.get_or_create(event.room_id)

        if event.event_type == "turn":
            battle.turn = int(event.payload.get("turn", battle.turn))
            battle.phase = BattlePhase.IN_BATTLE
            if battle.battle_state is not None:
                battle.battle_state.turn = battle.turn
                battle.battle_state.phase = battle.phase
                battle.battle_state.decrement_wishes()
                battle.battle_state.clear_turn_restriction_statuses()
        elif event.event_type == "request":
            request_json = event.payload.get("request", {})
            battle.last_request = request_json
            battle.request_id = request_json.get("rqid")
            previous_actions = list(battle.available_actions)
            new_actions = _extract_actions_from_request(request_json)
            if new_actions or not request_json.get("wait"):
                battle.available_actions = new_actions
            else:
                battle.available_actions = previous_actions
            _update_side_and_active_species_from_request(battle, request_json)
            _update_own_hp_fraction_from_request(battle, request_json)
            if battle.battle_state is not None:
                battle.battle_state.turn = battle.turn
                battle.battle_state.phase = battle.phase
                battle.battle_state.request_id = battle.request_id
                battle.battle_state.available_actions = list(battle.available_actions)
                battle.battle_state.update_from_request(request_json)
            _reconcile_active_slots_from_observed_species(battle)
            _sync_active_species_from_battle_state(battle)
            # Apply any pending switch lines we recorded earlier. These occur
            # when switch events arrived before we learned our `side.id` from
            # the request payload.
            pending = self._pending_switches.pop(battle.room_id, {})
            if pending:
                own_side_id = (battle.own_side_id or "").lower()
                for marker, species in pending.items():
                    if own_side_id and marker.startswith(own_side_id):
                        battle.active_species = species
                        if battle.battle_state is not None:
                            battle.battle_state.note_switch(
                                is_opponent=False,
                                ident=marker,
                                details=species,
                                condition=None,
                            )
                    else:
                        battle.opponent_active_species = species
                        if battle.battle_state is not None:
                            battle.battle_state.note_switch(
                                is_opponent=True,
                                ident=marker,
                                details=species,
                                condition=None,
                            )
                _reconcile_active_slots_from_observed_species(battle)
                _sync_active_species_from_battle_state(battle)
            if battle.battle_state is not None:
                own_active = battle.battle_state.active_own_pokemon
                if own_active is not None:
                    battle.active_species = own_active.display_name or own_active.species_id
                    battle.own_active_hp_fraction = own_active.hp_fraction()
            if battle.phase != BattlePhase.FINISHED:
                battle.phase = BattlePhase.IN_BATTLE
            if battle.battle_state is not None:
                battle.battle_state.phase = battle.phase
                battle.battle_state.available_actions = list(battle.available_actions)
            self._sync_active_opponent_constraints(event.room_id, battle)
        elif event.event_type in {"switch", "drag", "replace"}:
            _update_active_species_from_switch_line(battle, event.raw_line, self._pending_switches)
            _update_battle_state_from_switch_line(battle, event.raw_line)
            _reconcile_active_slots_from_observed_species(battle)
            _sync_active_species_from_battle_state(battle)
            self._sync_active_opponent_constraints(event.room_id, battle)
            if _is_opponent_switch_event(battle, event.raw_line):
                self._begin_new_opponent_stint(event.room_id, battle)
        elif event.event_type == "move":
            self._record_move(event.room_id, battle, event.raw_line)
        elif event.event_type in {"-item", "-enditem"}:
            self._record_opponent_item_event(event.room_id, battle, event.raw_line)
        elif event.event_type == "-damage":
            self._record_damage_event(event.room_id, battle, event.raw_line)
        elif event.event_type == "-crit":
            self._record_crit_event(event.room_id, battle, event.raw_line)
        elif event.event_type in {"-boost", "-unboost", "-setboost", "-clearboost", "-clearallboost"}:
            self._record_stat_boost_event(event.room_id, battle, event.raw_line)
        elif event.event_type in {"-status", "-curestatus"}:
            self._record_status_event(event.room_id, battle, event.raw_line)
        elif event.event_type in {"-sidestart", "-sideend", "-weather", "-fieldstart", "-fieldend", "-terrain"}:
            self._record_field_event(event.room_id, battle, event.raw_line)
        elif event.event_type in {"-immune", "-ability", "-activate", "-start", "-end", "-fail"}:
            self._record_opponent_ability_reveal(event.room_id, battle, event.raw_line)
        elif event.event_type == "-heal":
            self._record_heal_event(event.room_id, battle, event.raw_line)
        elif event.event_type in {"-mustrecharge", "-cant"}:
            # Showdown emits |-mustrecharge| or |-cant| lines when a Pokemon cannot act
            # (e.g. recharge after Hyper Beam or Truant). Record a volatile-status so
            # higher-level logic can avoid choosing illegal moves and reason about
            # forced inactivity.
            self._record_mustrecharge_or_cant_event(event.room_id, battle, event.raw_line)
        elif event.event_type in {"win", "tie"}:
            battle.phase = BattlePhase.FINISHED
            battle.winner = event.payload.get("winner")
            battle.available_actions = []
            if battle.battle_state is not None:
                battle.battle_state.phase = battle.phase
                battle.battle_state.winner = battle.winner
                battle.battle_state.available_actions = []

        return battle

    def _get_room_constraints(self, room_id: str) -> dict[str, OpponentConstraints]:
        return self._opponent_constraints.setdefault(room_id, {})

    def _sync_active_opponent_constraints(self, room_id: str, battle: BattleSnapshot) -> None:
        species_key = _normalize_species_key(battle.opponent_active_species)
        if not species_key:
            return
        room_constraints = self._get_room_constraints(room_id)
        constraints = room_constraints.setdefault(species_key, OpponentConstraints())
        battle.opponent_constraints = constraints

    def _sync_unburden_trigger(self, battle: BattleSnapshot) -> None:
        if battle.battle_state is None:
            return
        active_pokemon = battle.battle_state.active_opponent_pokemon
        battle.opponent_constraints.unburden_triggered = bool(
            active_pokemon is not None and active_pokemon.has_volatile_status("unburden")
        )

    def _begin_new_opponent_stint(self, room_id: str, battle: BattleSnapshot) -> None:
        self._sync_active_opponent_constraints(room_id, battle)
        constraints = battle.opponent_constraints
        constraints.active_stint_index += 1
        constraints.moves_used_since_switch_in.clear()
        constraints.speed_stage = 0
        constraints.observed_damage_fraction = None
        constraints.observed_damage_was_crit = False
        constraints.observed_damage_turn = None
        constraints.unburden_triggered = False
        self._pending_crit_target.pop(room_id, None)

    def _record_move(self, room_id: str, battle: BattleSnapshot, raw_line: str) -> None:
        parts = raw_line.split("|")
        if len(parts) < 4:
            return
        ident = parts[2].strip()
        move_name = parts[3].strip()
        if not move_name:
            return

        move_id = _normalize_move_id(move_name)
        move_data = _load_moves_data().get(move_id, {})
        move_type = _normalize_move_id(str(move_data.get("type", "")))

        opponent_move = _is_opponent_ident(battle, ident)
        if battle.battle_state is not None:
            battle.battle_state.note_move(
                is_opponent=opponent_move,
                ident=ident,
                move_id=move_id,
                name=move_data.get("name", move_name),
                category=str(move_data.get("category", "")) or None,
                move_type=move_type or None,
                base_power=int(move_data.get("basePower", 0) or 0) or None,
                turn=battle.turn,
            )
            volatile_status = _move_volatile_status(move_data)
            if volatile_status:
                battle.battle_state.note_volatile_status(
                    is_opponent=opponent_move,
                    ident=ident,
                    volatile_status=volatile_status,
                    active=True,
                    value=True,
                )
            if move_id == "wish":
                team = battle.battle_state.team_for_side(opponent_move)
                active_pokemon = team.active_pokemon
                if active_pokemon is not None and active_pokemon.max_hp > 0:
                    battle.battle_state.set_wish(
                        is_opponent=opponent_move,
                        turns_remaining=2,
                        hp_amount=active_pokemon.max_hp // 2,
                    )
        self._update_first_mover_and_infer_scarf(room_id, battle, opponent_move)

        if opponent_move:
            self._record_opponent_move(room_id, battle, move_id, move_data)
            return

        self._last_user_move_id[room_id] = move_id
        if move_type:
            self._last_user_move_type[room_id] = move_type

    def _update_first_mover_and_infer_scarf(self, room_id: str, battle: BattleSnapshot, opponent_move: bool) -> None:
        if battle.turn <= 0:
            return
        seen = self._first_mover_by_turn.get(room_id)
        if seen and seen[0] == battle.turn:
            return
        self._first_mover_by_turn[room_id] = (battle.turn, opponent_move)
        if not opponent_move:
            return

        self._sync_active_opponent_constraints(room_id, battle)
        constraints = battle.opponent_constraints
        if constraints.speed_stage > 0 or constraints.unburden_triggered or constraints.is_paralyzed:
            return
        if constraints.has_revealed_choice_conflict:
            return

        own_speed = _species_base_speed(battle.active_species)
        opponent_speed = _species_base_speed(battle.opponent_active_species)
        if own_speed <= 0 or opponent_speed <= 0:
            return

        # Conservative threshold: infer scarf only when turn order is strongly inconsistent
        # with base-speed expectations and no known speed modifiers explain it.
        if own_speed > opponent_speed:
            constraints.inferred_choice_scarf = True
            if "choicescarf" not in constraints.impossible_items:
                constraints.revealed_item = "choicescarf"

    def _record_opponent_move(
        self,
        room_id: str,
        battle: BattleSnapshot,
        move_id: str,
        move_data: dict[str, Any],
    ) -> None:
        self._sync_active_opponent_constraints(room_id, battle)
        battle.opponent_constraints.revealed_moves.add(move_id)
        battle.opponent_constraints.moves_used_since_switch_in.add(move_id)
        battle.opponent_constraints.last_opponent_move_id = move_id
        battle.opponent_constraints.observed_damage_was_crit = False

        # Showing at least two distinct moves in one active stint rules out choice lock.
        if len(battle.opponent_constraints.moves_used_since_switch_in) >= 2:
            battle.opponent_constraints.has_revealed_choice_conflict = True
            battle.opponent_constraints.impossible_items.update(CHOICE_ITEMS)
            if battle.opponent_constraints.revealed_item in CHOICE_ITEMS:
                battle.opponent_constraints.revealed_item = None

    def _record_opponent_item_event(self, room_id: str, battle: BattleSnapshot, raw_line: str) -> None:
        parts = raw_line.split("|")
        if len(parts) < 4:
            return
        ident = parts[2].strip()
        item_id = _normalize_move_id(parts[3].strip())
        if not item_id:
            return

        is_opponent = _is_opponent_ident(battle, ident)
        if battle.battle_state is not None:
            battle.battle_state.note_item(
                is_opponent=is_opponent,
                ident=ident,
                item=item_id,
                revealed=parts[1].strip().lower() == "-item",
                present=parts[1].strip().lower() == "-item",
            )

        if not is_opponent:
            return

        self._sync_active_opponent_constraints(room_id, battle)

        tag = parts[1].strip().lower()
        if tag == "-item":
            battle.opponent_constraints.revealed_item = item_id
            return

        # Losing an item makes that exact item impossible for future set samples.
        battle.opponent_constraints.impossible_items.add(item_id)
        if battle.opponent_constraints.revealed_item == item_id:
            battle.opponent_constraints.revealed_item = None
        self._sync_unburden_trigger(battle)

    def _record_field_event(self, room_id: str, battle: BattleSnapshot, raw_line: str) -> None:
        if battle.battle_state is None:
            return

        parts = raw_line.split("|")
        if len(parts) < 3:
            return

        tag = parts[1].strip().lower()
        if tag in {"-sidestart", "-sideend"}:
            if len(parts) < 4:
                return
            side_marker = parts[2].strip()
            condition = parts[3].strip()
            if not condition:
                return
            battle.battle_state.set_side_condition(
                is_opponent=_is_opponent_ident(battle, side_marker),
                condition=condition,
                active=tag == "-sidestart",
            )
            return

        if tag == "-weather":
            battle.battle_state.set_weather(parts[2].strip())
            return

        if tag == "-terrain":
            battle.battle_state.set_terrain(parts[2].strip())
            return

        if tag in {"-fieldstart", "-fieldend"}:
            effect_name = parts[2].strip()
            if not effect_name:
                return
            active = tag == "-fieldstart"
            battle.battle_state.set_pseudo_weather(effect_name, active=active)
            normalized_effect = normalize_volatile_status_name(effect_name)
            if normalized_effect == "trickroom":
                battle.battle_state.set_trick_room(active)

    def _record_crit_event(self, room_id: str, battle: BattleSnapshot, raw_line: str) -> None:
        ident = _extract_ident_from_protocol_line(raw_line)
        if not ident:
            return
        self._pending_crit_target[room_id] = ident

    def _record_damage_event(self, room_id: str, battle: BattleSnapshot, raw_line: str) -> None:
        parts = raw_line.split("|")
        if len(parts) < 4:
            return
        ident = parts[2].strip()
        is_opponent = _is_opponent_ident(battle, ident)
        hp_field = parts[3].strip()
        if "/" not in hp_field:
            return
        try:
            current_raw, max_raw = hp_field.split("/", 1)
            current_hp = int(current_raw)
            max_hp = int(max_raw)
        except ValueError:
            return
        if max_hp <= 0:
            return

        previous_hp_fraction = 1.0
        if battle.battle_state is not None:
            defender_before = battle.battle_state.active_opponent_pokemon if is_opponent else battle.battle_state.active_own_pokemon
            if defender_before is not None:
                hp_fraction = defender_before.hp_fraction()
                if hp_fraction is not None:
                    previous_hp_fraction = hp_fraction
            battle.battle_state.note_damage(
                is_opponent=is_opponent,
                ident=ident,
                current_hp=current_hp,
                max_hp=max_hp,
            )
        elif not is_opponent and battle.own_active_hp_fraction is not None:
            previous_hp_fraction = battle.own_active_hp_fraction

        new_hp_fraction = max(0.0, min(1.0, float(current_hp) / float(max_hp)))
        damage_fraction = max(0.0, previous_hp_fraction - new_hp_fraction)
        if not is_opponent:
            battle.own_active_hp_fraction = new_hp_fraction
        if damage_fraction <= 0:
            return

        source_move_id = (
            self._last_user_move_id.get(room_id)
            if is_opponent
            else battle.battle_state.last_own_move_id if battle.battle_state is not None else None
        )
        source_is_opponent = not is_opponent

        battle.opponent_constraints.observed_damage_fraction = damage_fraction
        battle.opponent_constraints.observed_damage_turn = battle.turn
        battle.opponent_constraints.observed_damage_source_move_id = source_move_id
        battle.opponent_constraints.observed_damage_source_is_opponent = source_is_opponent
        crit_target = self._pending_crit_target.get(room_id)
        battle.opponent_constraints.observed_damage_was_crit = bool(crit_target and crit_target == ident)
        self._pending_crit_target.pop(room_id, None)

        if battle.battle_state is None or battle.battle_state.active_opponent_pokemon is None or not source_move_id:
            return

        filtered_sets = filter_candidate_sets_by_observed_damage(
            snapshot=battle,
            candidate_sets=battle.battle_state.active_opponent_pokemon.candidate_sets,
            source_move_id=source_move_id,
            source_is_opponent=source_is_opponent,
            observed_damage_fraction=damage_fraction,
            observed_was_crit=battle.opponent_constraints.observed_damage_was_crit,
        )
        if filtered_sets:
            battle.battle_state.active_opponent_pokemon.candidate_sets = filtered_sets
            battle.battle_state.opponent_beliefs.hypotheses.clear()
            for candidate in filtered_sets:
                battle.battle_state.add_opponent_hypothesis(
                    candidate,
                    weight=float(candidate.get("count", 1) or 1),
                    source="damage_observation",
                )

    def _record_stat_boost_event(self, room_id: str, battle: BattleSnapshot, raw_line: str) -> None:
        parts = raw_line.split("|")
        if len(parts) < 3:
            return
        ident = parts[2].strip()
        is_opponent = _is_opponent_ident(battle, ident)
        tag = parts[1].strip().lower()
        stat_name = normalize_stat_name(parts[3].strip()) if len(parts) >= 4 else ""

        if tag == "-clearallboost":
            if battle.battle_state is not None:
                battle.battle_state.clear_all_boosts(is_opponent=is_opponent, ident=ident)
            if is_opponent:
                self._sync_active_opponent_constraints(room_id, battle)
                battle.opponent_constraints.speed_stage = 0
            return

        if not stat_name:
            return

        if tag == "-clearboost":
            if battle.battle_state is not None:
                battle.battle_state.clear_boost(is_opponent=is_opponent, ident=ident, stat_name=stat_name)
            if is_opponent and stat_name == "speed":
                self._sync_active_opponent_constraints(room_id, battle)
                battle.opponent_constraints.speed_stage = 0
            return

        try:
            stage_value = int(parts[4].strip()) if len(parts) >= 5 else 0
        except ValueError:
            return

        if tag == "-setboost":
            if battle.battle_state is not None:
                battle.battle_state.set_boost(is_opponent=is_opponent, ident=ident, stat_name=stat_name, stage=stage_value)
            if is_opponent and stat_name == "speed":
                self._sync_active_opponent_constraints(room_id, battle)
                battle.opponent_constraints.speed_stage = max(-6, min(6, stage_value))
            return

        delta = stage_value if tag == "-boost" else -stage_value

        if battle.battle_state is not None:
            battle.battle_state.note_boost(
                is_opponent=is_opponent,
                ident=ident,
                stat_name=stat_name,
                delta=delta,
            )

        if is_opponent and stat_name == "speed":
            self._sync_active_opponent_constraints(room_id, battle)
            updated = battle.opponent_constraints.speed_stage + delta
            battle.opponent_constraints.speed_stage = max(-6, min(6, updated))

    def _record_status_event(self, room_id: str, battle: BattleSnapshot, raw_line: str) -> None:
        parts = raw_line.split("|")
        if len(parts) < 4:
            return
        ident = parts[2].strip()
        is_opponent = _is_opponent_ident(battle, ident)
        if battle.battle_state is not None:
            battle.battle_state.note_status(
                is_opponent=is_opponent,
                ident=ident,
                status=parts[3].strip(),
                clear=parts[1].strip().lower() == "-curestatus",
            )
        if not is_opponent:
            return
        status = _normalize_move_id(parts[3].strip())
        self._sync_active_opponent_constraints(room_id, battle)
        tag = parts[1].strip().lower()
        if tag == "-status" and status == "par":
            battle.opponent_constraints.is_paralyzed = True
        elif tag == "-curestatus" and status == "par":
            battle.opponent_constraints.is_paralyzed = False

    def _record_volatile_status_event(self, room_id: str, battle: BattleSnapshot, raw_line: str) -> None:
        parts = raw_line.split("|")
        if len(parts) < 4:
            return
        tag = parts[1].strip().lower()
        if tag not in {"-start", "-end", "-activate"}:
            return
        value = parts[3].strip()
        if not value:
            return
        if re.search(r"(^|\W)ability:\s*", value, flags=re.IGNORECASE):
            return

        ident = parts[2].strip()
        volatile_status = normalize_volatile_status_name(value)
        if not volatile_status:
            return

        is_opponent = _is_opponent_ident(battle, ident)
        if battle.battle_state is not None:
            battle.battle_state.note_volatile_status(
                is_opponent=is_opponent,
                ident=ident,
                volatile_status=volatile_status,
                active=tag != "-end",
                value=True,
            )

        if is_opponent and volatile_status == "unburden":
            self._sync_unburden_trigger(battle)

    def _record_opponent_ability_reveal(self, room_id: str, battle: BattleSnapshot, raw_line: str) -> None:
        if raw_line.startswith("|-immune|"):
            self._record_opponent_immunity_event(room_id, battle, raw_line)

        self._record_volatile_status_event(room_id, battle, raw_line)

        ability_match = re.search(r"ability:\s*([^|\]]+)", raw_line, flags=re.IGNORECASE)
        if ability_match is None:
            parts = raw_line.split("|")
            if len(parts) < 4 or parts[1].strip().lower() != "-ability":
                return
            ability = _normalize_move_id(parts[3].strip())
        else:
            ability = _normalize_move_id(ability_match.group(1))
        if not ability:
            return

        ident = _extract_ident_from_protocol_line(raw_line)
        is_opponent = _is_opponent_ident(battle, ident) if ident else True

        if battle.battle_state is not None:
            battle.battle_state.note_ability(
                is_opponent=is_opponent,
                ident=ident,
                ability=ability,
            )

        if ident and not is_opponent:
            return

        self._sync_active_opponent_constraints(room_id, battle)
        self._sync_unburden_trigger(battle)
        battle.opponent_constraints.revealed_ability = ability

    def _record_opponent_immunity_event(self, room_id: str, battle: BattleSnapshot, raw_line: str) -> None:
        ident = _extract_ident_from_protocol_line(raw_line)
        if ident is None or not _is_opponent_ident(battle, ident):
            return

        if battle.battle_state is not None:
            battle.battle_state.note_immune(
                is_opponent=True,
                ident=ident,
                move_type=self._last_user_move_type.get(room_id),
            )
        self._sync_active_opponent_constraints(room_id, battle)
        move_type = self._last_user_move_type.get(room_id)
        if move_type:
            battle.opponent_constraints.observed_immune_move_types.add(move_type)

    def _record_mustrecharge_or_cant_event(self, room_id: str, battle: BattleSnapshot, raw_line: str) -> None:
        parts = raw_line.split("|")
        if len(parts) < 3:
            return
        ident = parts[2].strip()
        # Reason (e.g. 'truant' or empty) may appear in parts[3]
        reason = parts[3].strip() if len(parts) >= 4 else "mustrecharge"
        volatile_status = normalize_volatile_status_name(reason) or "mustrecharge"

        is_opponent = _is_opponent_ident(battle, ident)
        if battle.battle_state is not None:
            battle.battle_state.note_volatile_status(
                is_opponent=is_opponent,
                ident=ident,
                volatile_status=volatile_status,
                active=True,
                value=True,
            )

        # Update opponent constraints if relevant
        if is_opponent:
            self._sync_active_opponent_constraints(room_id, battle)

    def _record_heal_event(self, room_id: str, battle: BattleSnapshot, raw_line: str) -> None:
        parts = raw_line.split("|")
        if len(parts) < 4:
            return

        ident = parts[2].strip()
        condition = parts[3].strip()
        is_opponent = _is_opponent_ident(battle, ident)

        if battle.battle_state is not None:
            hp_chunk = extract_hp_chunk(condition)
            if hp_chunk is not None:
                current_hp, max_hp = hp_chunk
                battle.battle_state.note_damage(
                    is_opponent=is_opponent,
                    ident=ident,
                    current_hp=current_hp,
                    max_hp=max_hp,
                )

        if "[from] move: wish" in raw_line.lower() and battle.battle_state is not None:
            battle.battle_state.clear_wish(is_opponent=is_opponent)


def _is_opponent_switch_event(battle: BattleSnapshot, raw_line: str) -> bool:
    match = _SWITCH_LINE_RE.match(raw_line)
    if match is None:
        return False
    ident = match.group(2).strip()
    return _is_opponent_ident(battle, ident)


def _extract_actions_from_request(request_json: dict) -> list[ActionChoice]:
    if request_json.get("wait"):
        return []

    actions: list[ActionChoice] = []

    active_list = request_json.get("active", [])
    if active_list:
        active = active_list[0]
        for idx, move in enumerate(active.get("moves", []), start=1):
            if move.get("disabled"):
                continue
            move_id = _normalize_move_id(str(move.get("id") or move.get("move") or ""))
            if move_id != "struggle":
                try:
                    if int(move.get("pp")) <= 0:
                        continue
                except Exception:
                    pass
            if move.get("maybeDisabled"):
                continue
            command = f"/choose move {idx}"
            actions.append(
                ActionChoice(
                    action_type=ActionType.MOVE,
                    command=command,
                    label=move.get("move", move.get("id", f"move-{idx}")),
                )
            )

    side = request_json.get("side", {})
    for pkmn in side.get("pokemon", []):
        if pkmn.get("active"):
            continue
        condition = pkmn.get("condition", "")
        if "fnt" in condition:
            continue
        slot = pkmn.get("ident", "").split(":")[-1].strip()
        if slot:
            actions.append(
                ActionChoice(
                    action_type=ActionType.SWITCH,
                    command=f"/choose switch {slot}",
                    label=f"switch {slot}",
                )
            )

    if request_json.get("teamPreview"):
        actions.append(
            ActionChoice(
                action_type=ActionType.TEAM_PREVIEW,
                command="/team 123456",
                label="default team preview order",
            )
        )

    return actions


def _extract_species_from_details(details: str) -> str | None:
    if not details:
        return None
    species = details.split(",", 1)[0].strip()
    return species or None


def _normalize_species_key(species_name: str | None) -> str:
    if not species_name:
        return ""
    return re.sub(r"[^a-z0-9]", "", species_name.lower())


def _normalize_move_id(name: str | None) -> str:
    if not name:
        return ""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _extract_ident_from_protocol_line(raw_line: str) -> str | None:
    parts = raw_line.split("|")
    if len(parts) < 3:
        return None
    ident = parts[2].strip()
    return ident or None


def _is_opponent_ident(battle: BattleSnapshot, ident: str) -> bool:
    side_marker = ident.split(":", 1)[0].strip().lower()
    own_side_id = (battle.own_side_id or "").strip().lower()
    if own_side_id:
        return not side_marker.startswith(own_side_id)
    # If we don't yet know our side id, assume p2* is opponent by convention.
    if side_marker.startswith("p2"):
        return True
    if side_marker.startswith("p1"):
        return False
    return True


def _update_side_and_active_species_from_request(battle: BattleSnapshot, request_json: dict) -> None:
    side = request_json.get("side", {})
    side_id = side.get("id")
    if isinstance(side_id, str) and side_id:
        battle.own_side_id = side_id

    for pokemon in side.get("pokemon", []):
        if not pokemon.get("active"):
            continue
        details = str(pokemon.get("details", ""))
        species = _extract_species_from_details(details)
        if species:
            battle.active_species = species
            return


def _update_own_hp_fraction_from_request(battle: BattleSnapshot, request_json: dict) -> None:
    side = request_json.get("side", {})
    for pokemon in side.get("pokemon", []):
        if not pokemon.get("active"):
            continue
        condition = str(pokemon.get("condition", ""))
        hp_chunk = condition.split(" ", 1)[0]
        if "/" not in hp_chunk:
            return
        try:
            current_raw, max_raw = hp_chunk.split("/", 1)
            current_hp = int(current_raw)
            max_hp = int(max_raw)
        except ValueError:
            return
        if max_hp <= 0:
            return
        battle.own_active_hp_fraction = max(0.0, min(1.0, float(current_hp) / float(max_hp)))
        return


_SWITCH_LINE_RE = re.compile(r"^\|(switch|drag|replace)\|([^|]+)\|([^|]+)\|")


def _update_active_species_from_switch_line(
    battle: BattleSnapshot, raw_line: str, pending_switches: dict[str, dict[str, str]] | None = None
) -> None:
    match = _SWITCH_LINE_RE.match(raw_line)
    if match is None:
        return

    ident = match.group(2).strip()
    details = match.group(3).strip()
    species = _extract_species_from_details(details)
    if species is None:
        return

    side_marker = ident.split(":", 1)[0].strip().lower()
    own_side_id = (battle.own_side_id or "").lower()
    # If we already know which side belongs to us, assign directly.
    if own_side_id and side_marker.startswith(own_side_id):
        battle.active_species = species
        return
    if own_side_id:
        battle.opponent_active_species = species
        return
    # We don't yet know our side id (request may not have arrived). Record
    # the seen switch so it can be applied later once the request provides
    # the side identifier.
    # NOTE: This function will be called with a pending_switches mapping by
    # the tracker; if none is provided, fall back to assigning opponent.
    # Prefer the provided pending mapping from the tracker instance; fall
    # back to a module-level mapping if one exists, otherwise give up and
    # assign the opponent directly (conservative fallback).
    pending = pending_switches if pending_switches is not None else globals().get("_PENDING_SWITCHES")
    if pending is None:
        battle.opponent_active_species = species
        return
    room_pending = pending.setdefault(battle.room_id, {})
    room_pending[side_marker] = species


def _update_battle_state_from_switch_line(battle: BattleSnapshot, raw_line: str) -> None:
    if battle.battle_state is None or not battle.own_side_id:
        return
    parts = raw_line.split("|")
    if len(parts) < 5:
        return
    ident = parts[2].strip()
    details = parts[3].strip()
    condition = parts[4].strip()
    battle.battle_state.note_switch(
        is_opponent=_is_opponent_ident(battle, ident),
        ident=ident,
        details=details,
        condition=condition,
    )


def _sync_active_species_from_battle_state(battle: BattleSnapshot) -> None:
    battle_state = battle.battle_state
    if battle_state is None:
        return

    own_active = battle_state.active_own_pokemon
    if own_active is not None:
        battle.active_species = own_active.display_name or own_active.species_id
        battle.own_active_hp_fraction = own_active.hp_fraction()

    opponent_active = battle_state.active_opponent_pokemon
    if opponent_active is not None:
        battle.opponent_active_species = opponent_active.display_name or opponent_active.species_id


def _reconcile_active_slots_from_observed_species(battle: BattleSnapshot) -> None:
    battle_state = battle.battle_state
    if battle_state is None:
        return

    def _sync_team(team, observed_species: str | None) -> None:
        if not observed_species:
            return
        normalized_observed = _normalize_species_key(observed_species)
        if not normalized_observed:
            return
        active_pokemon = team.active_pokemon
        active_species = active_pokemon.display_name or active_pokemon.species_id if active_pokemon is not None else None
        if active_species and _normalize_species_key(active_species) == normalized_observed:
            return
        for index, pokemon in enumerate(team.pokemon):
            species_name = pokemon.display_name or pokemon.species_id
            if species_name and _normalize_species_key(species_name) == normalized_observed:
                team.set_active_slot(index)
                break

    _sync_team(battle_state.own_team, battle.active_species)
    _sync_team(battle_state.opponent_team, battle.opponent_active_species)
