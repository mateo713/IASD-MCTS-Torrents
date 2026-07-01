from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from core.battle_belief import BeliefPool
from core.identifiers import (
    extract_hp_chunk,
    extract_species_from_details,
    extract_status_from_condition,
    hp_fraction_from_condition,
    normalize_ability_id,
    normalize_identifier,
    normalize_item_id,
    normalize_move_id,
    normalize_stat_name,
    normalize_volatile_status_name,
    slot_index_from_ident,
)


_LAYERED_SIDE_CONDITIONS = {"spikes", "toxicspikes"}


def _stage_multiplier(stage: int) -> float:
    bounded = max(-6, min(6, int(stage)))
    if bounded >= 0:
        return (2.0 + bounded) / 2.0
    return 2.0 / (2.0 - bounded)


@dataclass(slots=True)
class MoveState:
    move_id: str = ""
    name: str | None = None
    category: str | None = None
    move_type: str | None = None
    base_power: int | None = None
    accuracy: float | None = None
    pp: int | None = None
    max_pp: int | None = None
    disabled: bool = False
    revealed: bool = False
    last_used_turn: int | None = None
    last_result: str | None = None


@dataclass(slots=True)
class PokemonState:
    species_id: str | None = None
    display_name: str | None = None
    slot: int | None = None
    level: int | None = None
    gender: str | None = None
    types: tuple[str, ...] = ()
    base_stats: dict[str, int] = field(default_factory=dict)
    stats: dict[str, int] = field(default_factory=dict)
    current_hp: int | None = None
    max_hp: int | None = None
    ability: str | None = None
    item: str | None = None
    status: str | None = None
    tera_type: str | None = None
    is_active: bool = False
    fainted: bool = False
    trapped: bool = False
    maybe_trapped: bool = False
    stat_boosts: dict[str, int] = field(default_factory=dict)
    volatile_statuses: dict[str, Any] = field(default_factory=dict)
    move_history: list[dict[str, Any]] = field(default_factory=list)
    moves: dict[str, MoveState] = field(default_factory=dict)
    revealed_moves: set[str] = field(default_factory=set)
    candidate_sets: list[dict[str, Any]] = field(default_factory=list)
    revealed_ability: bool = False
    revealed_item: bool = False
    last_seen_turn: int | None = None

    def hp_fraction(self) -> float | None:
        if self.current_hp is None or self.max_hp is None or self.max_hp <= 0:
            return None
        return max(0.0, min(1.0, float(self.current_hp) / float(self.max_hp)))

    def get_stat(self, stat_name: str) -> int | None:
        value = self.stats.get(normalize_stat_name(stat_name))
        if value is None:
            return None
        try:
            return int(value)
        except Exception:
            return None

    def set_stats(self, stats: dict[str, Any] | None) -> None:
        if not isinstance(stats, dict):
            return
        normalized_stats: dict[str, int] = {}
        for stat_name, value in stats.items():
            normalized_stat = normalize_stat_name(stat_name)
            if not normalized_stat:
                continue
            try:
                normalized_stats[normalized_stat] = int(value)
            except (TypeError, ValueError):
                continue
        if normalized_stats:
            self.stats = normalized_stats

    def effective_speed(self) -> int | None:
        return self.effective_stat("speed")

    def stat_stage(self, stat_name: str) -> int:
        normalized_stat = normalize_stat_name(stat_name)
        return int(self.stat_boosts.get(normalized_stat, 0))

    def set_stat_stage(self, stat_name: str, stage: int) -> None:
        normalized_stat = normalize_stat_name(stat_name)
        if not normalized_stat:
            return
        self.stat_boosts[normalized_stat] = max(-6, min(6, int(stage)))

    def clear_stat_stage(self, stat_name: str) -> None:
        normalized_stat = normalize_stat_name(stat_name)
        if normalized_stat in self.stat_boosts:
            self.stat_boosts[normalized_stat] = 0

    def clear_all_stat_stages(self) -> None:
        for stat_name in list(self.stat_boosts):
            self.stat_boosts[stat_name] = 0

    def add_volatile_status(self, volatile_name: str, value: Any = True) -> None:
        normalized_volatile = normalize_identifier(volatile_name)
        if not normalized_volatile:
            return
        self.volatile_statuses[normalized_volatile] = value

    def has_volatile_status(self, volatile_name: str) -> bool:
        normalized_volatile = normalize_identifier(volatile_name)
        if not normalized_volatile:
            return False
        return normalized_volatile in self.volatile_statuses

    def remove_volatile_status(self, volatile_name: str) -> None:
        normalized_volatile = normalize_identifier(volatile_name)
        if not normalized_volatile:
            return
        self.volatile_statuses.pop(normalized_volatile, None)

    def clear_volatile_statuses(self) -> None:
        immune_to = self.volatile_statuses.get("immune_to")
        self.volatile_statuses.clear()
        if immune_to is not None:
            self.volatile_statuses["immune_to"] = immune_to

    def clear_turn_restriction_statuses(self) -> None:
        self.remove_volatile_status("mustrecharge")
        self.remove_volatile_status("truant")

    def clear_transient_battle_effects(self) -> None:
        self.clear_all_stat_stages()
        self.clear_volatile_statuses()

    def sync_unburden_status(self) -> None:
        if normalize_identifier(self.ability) == "unburden" and self.item is None:
            self.add_volatile_status("unburden")
        else:
            self.remove_volatile_status("unburden")

    def stat_multiplier(self, stat_name: str) -> float:
        return _stage_multiplier(self.stat_stage(stat_name))

    def effective_stat(self, stat_name: str) -> int | float | None:
        normalized_stat = normalize_stat_name(stat_name)
        if normalized_stat in {"accuracy", "evasion"}:
            return self.stat_multiplier(normalized_stat)
        base_value = self.get_stat(normalized_stat)
        if base_value is None:
            return None
        return max(0, int(round(float(base_value) * self.stat_multiplier(normalized_stat))))

    def get_move(self, move_id: str) -> MoveState | None:
        return self.moves.get(normalize_move_id(move_id))

    def note_move(
        self,
        move_id: str,
        *,
        name: str | None = None,
        category: str | None = None,
        move_type: str | None = None,
        base_power: int | None = None,
        accuracy: float | None = None,
        pp: int | None = None,
        max_pp: int | None = None,
        turn: int | None = None,
        result: str | None = None,
        record_history: bool = True,
    ) -> MoveState:
        normalized_move_id = normalize_move_id(move_id)
        move = self.moves.get(normalized_move_id)
        if move is None:
            move = MoveState(move_id=normalized_move_id)
            self.moves[normalized_move_id] = move
        if name is not None:
            move.name = name
        if category is not None:
            move.category = category
        if move_type is not None:
            move.move_type = move_type
        if base_power is not None:
            move.base_power = base_power
        if accuracy is not None:
            move.accuracy = accuracy
        if pp is not None:
            try:
                move.pp = int(pp)
            except Exception:
                move.pp = None
        if max_pp is not None:
            try:
                move.max_pp = int(max_pp)
            except Exception:
                move.max_pp = None
        move.revealed = True
        if turn is not None:
            move.last_used_turn = turn
        if result is not None:
            move.last_result = result
        self.revealed_moves.add(normalized_move_id)
        if record_history:
            self.move_history.append(
                {
                    "move_id": normalized_move_id,
                    "name": name,
                    "turn": turn,
                    "result": result,
                }
            )
        return move

    def learn_species(self, species_name: str | None, *, data_dir: str | None = None) -> None:
        species_key = normalize_identifier(species_name)
        if not species_key:
            return

        self.species_id = species_key
        if species_name:
            self.display_name = species_name

        try:
            from engine.gen5_datasets import build_stat_profile, get_feasible_random_battle_sets, get_species_record
        except Exception:
            self.candidate_sets = []
            return

        record = get_species_record(species_name, data_dir)
        if record:
            base_stats = record.get("base_stats", {})
            if isinstance(base_stats, dict):
                self.base_stats = {normalize_stat_name(stat_name): int(value) for stat_name, value in base_stats.items()}

        candidate_sets = get_feasible_random_battle_sets(species_name, data_dir=data_dir)
        enriched_sets: list[dict[str, Any]] = []
        for candidate in candidate_sets:
            candidate_copy = dict(candidate)
            computed_stats = candidate_copy.get("computed_stats")
            stat_profile = candidate_copy.get("stat_profile")
            if not isinstance(computed_stats, dict) or not isinstance(stat_profile, dict):
                level = int(candidate_copy.get("level", 100) or 100)
                nature = normalize_identifier(str(candidate_copy.get("nature", "serious")))
                profile = build_stat_profile(species_name, level, nature)
                computed_stats = dict(profile.stats)
                stat_profile = {
                    "level": profile.level,
                    "nature": profile.nature,
                    "ivs": profile.ivs,
                    "evs": profile.evs,
                    "stats": dict(profile.stats),
                    "has_physical_attacks": profile.has_physical_attacks,
                }
            candidate_copy["computed_stats"] = dict(computed_stats)
            candidate_copy["stat_profile"] = dict(stat_profile)
            enriched_sets.append(candidate_copy)
        self.candidate_sets = enriched_sets

        representative_stats: dict[str, Any] | None = None
        if enriched_sets:
            representative_candidate = max(
                enriched_sets,
                key=lambda candidate: int(candidate.get("count", 1) or 1),
            )
            representative_stats = representative_candidate.get("computed_stats")
        if not isinstance(representative_stats, dict) or not representative_stats:
            try:
                level = int(self.level or 100)
                representative_stats = build_stat_profile(species_name, level, "serious").stats
            except Exception:
                representative_stats = {}
        self.set_stats(representative_stats)

    def filter_candidate_sets_by_observed_damage(
        self,
        *,
        snapshot: Any,
        source_move_id: str | None,
        source_is_opponent: bool,
        observed_damage_fraction: float,
        observed_was_crit: bool = False,
    ) -> list[dict[str, Any]]:
        if not self.candidate_sets or not source_move_id:
            return self.candidate_sets

        try:
            from engine.gen5_datasets import filter_candidate_sets_by_observed_damage
        except Exception:
            return self.candidate_sets

        filtered = filter_candidate_sets_by_observed_damage(
            snapshot=snapshot,
            candidate_sets=self.candidate_sets,
            source_move_id=source_move_id,
            source_is_opponent=source_is_opponent,
            observed_damage_fraction=observed_damage_fraction,
            observed_was_crit=observed_was_crit,
        )
        if filtered:
            self.candidate_sets = filtered
        return self.candidate_sets

    def update_from_condition(self, condition: str | None) -> None:
        hp_chunk = extract_hp_chunk(condition)
        if hp_chunk is not None:
            current_hp, max_hp = hp_chunk
            self.current_hp = current_hp
            self.max_hp = max_hp
            self.fainted = current_hp <= 0
        status = extract_status_from_condition(condition)
        if status is not None:
            self.status = status

    def note_item(self, item: str | None, revealed: bool = True, present: bool = True) -> None:
        normalized_item = normalize_item_id(item)
        if present:
            if normalized_item:
                self.item = normalized_item
                self.revealed_item = revealed
        else:
            self.item = None
            self.revealed_item = revealed
        self.sync_unburden_status()

    def note_ability(self, ability: str | None, revealed: bool = True) -> None:
        normalized_ability = normalize_ability_id(ability)
        if normalized_ability:
            self.ability = normalized_ability
            self.revealed_ability = revealed
        self.sync_unburden_status()

    def note_boost(self, stat_name: str, delta: int) -> None:
        normalized_stat = normalize_stat_name(stat_name)
        if not normalized_stat:
            return
        updated = self.stat_boosts.get(normalized_stat, 0) + delta
        self.stat_boosts[normalized_stat] = max(-6, min(6, updated))


@dataclass(slots=True)
class TeamState:
    side_id: str | None = None
    team_name: str | None = None
    pokemon: list[PokemonState] = field(default_factory=lambda: [PokemonState(slot=index) for index in range(6)])
    active_index: int = 0
    side_conditions: dict[str, Any] = field(default_factory=dict)

    def ensure_slot(self, slot_index: int) -> PokemonState:
        slot_index = max(0, min(5, slot_index))
        while len(self.pokemon) <= slot_index:
            self.pokemon.append(PokemonState(slot=len(self.pokemon)))
        pokemon = self.pokemon[slot_index]
        pokemon.slot = slot_index
        return pokemon

    def get_pokemon(self, slot_index: int) -> PokemonState | None:
        if slot_index < 0 or slot_index >= len(self.pokemon):
            return None
        return self.pokemon[slot_index]

    def find_slot_by_species(self, species_name: str | None) -> int | None:
        normalized_species = normalize_identifier(species_name)
        if not normalized_species:
            return None
        for index, pokemon in enumerate(self.pokemon):
            candidate_species = pokemon.display_name or pokemon.species_id
            if candidate_species and normalize_identifier(candidate_species) == normalized_species:
                return index
        return None

    def first_unassigned_slot(self) -> int | None:
        for index, pokemon in enumerate(self.pokemon):
            if pokemon.species_id is None and pokemon.display_name is None:
                return index
        return None

    def resolve_slot_for_species(self, species_name: str | None) -> int:
        matched_slot = self.find_slot_by_species(species_name)
        if matched_slot is not None:
            return matched_slot
        empty_slot = self.first_unassigned_slot()
        if empty_slot is not None:
            return empty_slot
        return self.active_index

    @property
    def active_pokemon(self) -> PokemonState | None:
        return self.get_pokemon(self.active_index)

    def iter_pokemon(self) -> Iterable[PokemonState]:
        return iter(self.pokemon)

    def set_active_slot(self, slot_index: int) -> PokemonState:
        self.active_index = max(0, min(5, slot_index))
        pokemon = self.ensure_slot(self.active_index)
        for index, member in enumerate(self.pokemon):
            member.is_active = index == self.active_index
        pokemon.is_active = True
        return pokemon

    def update_from_request(self, side_payload: dict[str, Any], *, turn: int | None = None, learn_species: bool = True) -> None:
        side_id = side_payload.get("id")
        if isinstance(side_id, str) and side_id:
            self.side_id = side_id

        if "sideConditions" in side_payload and isinstance(side_payload.get("sideConditions"), dict):
            self.side_conditions = dict(side_payload.get("sideConditions", {}))

        for index, entry in enumerate(side_payload.get("pokemon", [])[:6]):
            pokemon = self.ensure_slot(index)
            details = str(entry.get("details", ""))
            species = extract_species_from_details(details)
            if species:
                if learn_species:
                    pokemon.learn_species(species)
                pokemon.display_name = species
            pokemon.slot = index
            pokemon.is_active = bool(entry.get("active"))
            if pokemon.is_active:
                self.active_index = index
            pokemon.trapped = bool(entry.get("trapped"))
            pokemon.maybe_trapped = bool(entry.get("maybeTrapped"))
            pokemon.update_from_condition(str(entry.get("condition", "")))
            pokemon.note_item(entry.get("item"), revealed=bool(entry.get("item")))
            pokemon.note_ability(entry.get("ability"), revealed=bool(entry.get("ability")))
            raw_stats = entry.get("stats")
            if isinstance(raw_stats, dict) and raw_stats:
                pokemon.set_stats(raw_stats)
            elif not pokemon.stats and pokemon.candidate_sets:
                representative_candidate = max(
                    pokemon.candidate_sets,
                    key=lambda candidate: int(candidate.get("count", 1) or 1),
                )
                representative_stats = representative_candidate.get("computed_stats")
                if isinstance(representative_stats, dict):
                    pokemon.set_stats(representative_stats)
            moves = entry.get("moves", [])
            if isinstance(moves, list):
                for move_entry in moves:
                    if not isinstance(move_entry, dict):
                        continue
                    move_id = move_entry.get("id") or move_entry.get("move")
                    if not move_id:
                        continue
                    move_state = pokemon.note_move(
                        str(move_id),
                        name=move_entry.get("move") or move_entry.get("name"),
                        category=move_entry.get("category"),
                        move_type=move_entry.get("type"),
                        base_power=move_entry.get("basePower"),
                        accuracy=move_entry.get("accuracy"),
                        pp=move_entry.get("pp"),
                        max_pp=move_entry.get("maxpp") or move_entry.get("maxPP"),
                        turn=turn,
                        record_history=False,
                    )
                    move_state.disabled = bool(move_entry.get("disabled"))
            if turn is not None:
                pokemon.last_seen_turn = turn

        for index, pokemon in enumerate(self.pokemon):
            pokemon.is_active = index == self.active_index


@dataclass(slots=True)
class BattleFieldState:
    weather: str | None = None
    terrain: str | None = None
    trick_room: bool = False
    pseudo_weather: dict[str, Any] = field(default_factory=dict)
    side_conditions: dict[str, dict[str, Any]] = field(default_factory=lambda: {"own": {}, "opponent": {}})
    wish: dict[str, tuple[int, int]] = field(default_factory=lambda: {"own": (0, 0), "opponent": (0, 0)})


@dataclass(slots=True)
class BattleState:
    room_id: str
    turn: int = 0
    phase: Any = None
    request_id: int | None = None
    available_actions: list[Any] = field(default_factory=list)
    request_force_switch: list[bool] = field(default_factory=list)
    winner: str | None = None
    last_own_move_id: str | None = None
    last_opponent_move_id: str | None = None
    own_team: TeamState = field(default_factory=TeamState)
    opponent_team: TeamState = field(default_factory=TeamState)
    battlefield: BattleFieldState = field(default_factory=BattleFieldState)
    opponent_beliefs: BeliefPool = field(default_factory=BeliefPool)
    raw_request: dict[str, Any] | None = None
    own_side_id: str | None = None
    opponent_side_id: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def active_own_pokemon(self) -> PokemonState | None:
        return self.own_team.active_pokemon

    @property
    def active_opponent_pokemon(self) -> PokemonState | None:
        return self.opponent_team.active_pokemon

    @property
    def teams(self) -> tuple[TeamState, TeamState]:
        return self.own_team, self.opponent_team

    def team_for_side(self, is_opponent: bool) -> TeamState:
        return self.opponent_team if is_opponent else self.own_team

    def all_pokemon(self) -> list[PokemonState]:
        return [*self.own_team.pokemon, *self.opponent_team.pokemon]

    def sync_opponent_beliefs_from_active_pokemon(self) -> None:
        active_pokemon = self.active_opponent_pokemon
        self.opponent_beliefs.hypotheses.clear()
        if active_pokemon is None:
            return
        for candidate in active_pokemon.candidate_sets:
            self.add_opponent_hypothesis(
                candidate,
                weight=float(candidate.get("count", 1) or 1),
                source="species_learned",
            )

    def add_opponent_hypothesis(
        self,
        set_data: dict[str, Any],
        *,
        weight: float = 1.0,
        violations: int = 0,
        source: str | None = None,
        notes: dict[str, Any] | None = None,
    ) -> None:
        self.opponent_beliefs.add(
            set_data,
            weight=weight,
            violations=violations,
            source=source,
            notes=notes,
        )

    def sample_opponent_hypothesis(self, rng) -> dict[str, Any] | None:
        return self.opponent_beliefs.sample(rng)

    def best_opponent_hypotheses(self) -> list[dict[str, Any]]:
        return self.opponent_beliefs.best_by_violations()

    def update_from_request(self, request_json: dict[str, Any]) -> None:
        self.raw_request = request_json
        request_id = request_json.get("rqid")
        self.request_id = int(request_id) if isinstance(request_id, int | str) and str(request_id).isdigit() else request_id
        previous_actions = list(self.available_actions)
        self.available_actions = previous_actions
        force_switch = request_json.get("forceSwitch")
        if isinstance(force_switch, list):
            self.request_force_switch = [bool(value) for value in force_switch]
        else:
            self.request_force_switch = []
        active_list = request_json.get("active", [])
        if active_list:
            next_actions = request_json.get("available_actions", self.available_actions)
            if next_actions:
                self.available_actions = next_actions
        side_payload = request_json.get("side", {})
        if isinstance(side_payload, dict):
            self.own_team.update_from_request(side_payload, turn=self.turn, learn_species=True)
            self.own_side_id = self.own_team.side_id or self.own_side_id
        if active_list:
            active_entry = active_list[0] if isinstance(active_list, list) and active_list else None
            active_pokemon = self.active_own_pokemon
            if isinstance(active_entry, dict) and active_pokemon is not None:
                active_pokemon.trapped = bool(active_entry.get("trapped"))
                active_pokemon.maybe_trapped = bool(active_entry.get("maybeTrapped"))
                for move_entry in active_entry.get("moves", []) or []:
                    if not isinstance(move_entry, dict):
                        continue
                    move_id = move_entry.get("id") or move_entry.get("move")
                    if not move_id:
                        continue
                    move_state = active_pokemon.note_move(
                        str(move_id),
                        name=move_entry.get("move") or move_entry.get("name"),
                        category=move_entry.get("category"),
                        move_type=move_entry.get("type"),
                        base_power=move_entry.get("basePower"),
                        accuracy=move_entry.get("accuracy"),
                        pp=move_entry.get("pp"),
                        max_pp=move_entry.get("maxpp") or move_entry.get("maxPP"),
                        turn=self.turn,
                        record_history=False,
                    )
                    move_state.disabled = bool(move_entry.get("disabled")) or (move_state.pp is not None and move_state.pp <= 0)
        self._sync_legacy_indices()
        self._update_battlefield_from_request(request_json)

    def _update_battlefield_from_request(self, request_json: dict[str, Any]) -> None:
        if "weather" in request_json:
            self.set_weather(request_json.get("weather"))
        if "terrain" in request_json:
            self.set_terrain(request_json.get("terrain"))
        trick_room = request_json.get("trickRoom")
        if isinstance(trick_room, bool):
            self.set_trick_room(trick_room)
        pseudo_weather = request_json.get("pseudoWeather")
        if isinstance(pseudo_weather, dict):
            self.battlefield.pseudo_weather = dict(pseudo_weather)
        for key in ("sideConditions", "foeSideConditions"):
            value = request_json.get(key)
            if isinstance(value, dict):
                side_key = "own" if key == "sideConditions" else "opponent"
                existing = self.battlefield.side_conditions.get(side_key, {})
                merged: dict[str, Any] = {}
                for condition, condition_value in value.items():
                    normalized_condition = normalize_volatile_status_name(condition)
                    if not normalized_condition:
                        continue
                    if isinstance(condition_value, dict):
                        merged_value = dict(condition_value)
                        if normalized_condition in _LAYERED_SIDE_CONDITIONS and "layers" not in merged_value:
                            previous_value = existing.get(normalized_condition, {})
                            if isinstance(previous_value, dict) and "layers" in previous_value:
                                merged_value["layers"] = previous_value["layers"]
                        merged[normalized_condition] = merged_value
                    else:
                        merged[normalized_condition] = condition_value
                self.battlefield.side_conditions[side_key] = merged

    def set_weather(self, weather: str | None) -> None:
        normalized_weather = normalize_identifier(weather)
        if not normalized_weather or normalized_weather == "none":
            self.battlefield.weather = None
            return
        self.battlefield.weather = normalized_weather

    def set_terrain(self, terrain: str | None) -> None:
        normalized_terrain = normalize_identifier(terrain)
        if not normalized_terrain or normalized_terrain == "none":
            self.battlefield.terrain = None
            return
        self.battlefield.terrain = normalized_terrain

    def set_trick_room(self, active: bool) -> None:
        self.battlefield.trick_room = bool(active)

    def set_pseudo_weather(self, effect_name: str | None, *, active: bool = True) -> None:
        normalized_effect = normalize_volatile_status_name(effect_name)
        if not normalized_effect or normalized_effect == "none":
            return
        if active:
            self.battlefield.pseudo_weather[normalized_effect] = {"active": True}
        else:
            self.battlefield.pseudo_weather.pop(normalized_effect, None)

    def set_side_condition(self, *, is_opponent: bool, condition: str | None, active: bool = True) -> None:
        normalized_condition = normalize_volatile_status_name(condition)
        if not normalized_condition or normalized_condition == "none":
            return
        side_key = "opponent" if is_opponent else "own"
        side_conditions = self.battlefield.side_conditions.setdefault(side_key, {})
        if active:
            if normalized_condition in _LAYERED_SIDE_CONDITIONS:
                previous_value = side_conditions.get(normalized_condition, {})
                previous_layers = 0
                if isinstance(previous_value, dict):
                    try:
                        previous_layers = int(previous_value.get("layers", 0) or 0)
                    except Exception:
                        previous_layers = 0
                max_layers = 3 if normalized_condition == "spikes" else 2
                side_conditions[normalized_condition] = {
                    "active": True,
                    "layers": min(max_layers, previous_layers + 1),
                }
            else:
                side_conditions[normalized_condition] = {"active": True}
        else:
            side_conditions.pop(normalized_condition, None)

    def set_wish(self, *, is_opponent: bool, turns_remaining: int, hp_amount: int) -> None:
        side_key = "opponent" if is_opponent else "own"
        self.battlefield.wish[side_key] = (max(0, int(turns_remaining)), max(0, int(hp_amount)))

    def clear_wish(self, *, is_opponent: bool) -> None:
        side_key = "opponent" if is_opponent else "own"
        self.battlefield.wish[side_key] = (0, 0)

    def decrement_wishes(self) -> None:
        for side_key, value in list(self.battlefield.wish.items()):
            if not isinstance(value, tuple) or len(value) != 2:
                self.battlefield.wish[side_key] = (0, 0)
                continue
            turns_remaining, hp_amount = value
            if turns_remaining > 0:
                self.battlefield.wish[side_key] = (turns_remaining - 1, hp_amount)

    def note_switch(self, *, is_opponent: bool, ident: str | None, details: str | None, condition: str | None) -> PokemonState:
        team = self.team_for_side(is_opponent)
        previous_active = team.active_pokemon
        species = extract_species_from_details(details)
        slot_index = team.resolve_slot_for_species(species)
        pokemon = team.ensure_slot(slot_index)
        if species:
            pokemon.learn_species(species)
            pokemon.display_name = species
        pokemon.update_from_condition(condition)
        if previous_active is not None:
            if previous_active is not pokemon:
                previous_active.clear_transient_battle_effects()
        team.set_active_slot(slot_index)
        if is_opponent:
            self.sync_opponent_beliefs_from_active_pokemon()
        if is_opponent:
            self.opponent_side_id = ident.split(":", 1)[0].strip() if ident else self.opponent_side_id
        else:
            self.own_side_id = ident.split(":", 1)[0].strip() if ident else self.own_side_id
        self._sync_legacy_indices()
        return pokemon

    def note_move(
        self,
        *,
        is_opponent: bool,
        ident: str | None,
        move_id: str,
        name: str | None = None,
        category: str | None = None,
        move_type: str | None = None,
        base_power: int | None = None,
        accuracy: float | None = None,
        turn: int | None = None,
        result: str | None = None,
    ) -> MoveState:
        pokemon = self._active_pokemon_for_ident(is_opponent=is_opponent, ident=ident)
        move = pokemon.note_move(
            move_id,
            name=name,
            category=category,
            move_type=move_type,
            base_power=base_power,
            accuracy=accuracy,
            turn=turn,
            result=result,
        )
        if turn is not None:
            pokemon.last_seen_turn = turn
        if is_opponent:
            self.last_opponent_move_id = move_id
        else:
            self.last_own_move_id = move_id
        return move

    def note_damage(self, *, is_opponent: bool, ident: str | None, current_hp: int, max_hp: int) -> PokemonState:
        pokemon = self._active_pokemon_for_ident(is_opponent=is_opponent, ident=ident)
        pokemon.current_hp = current_hp
        pokemon.max_hp = max_hp
        pokemon.fainted = current_hp <= 0
        return pokemon

    def note_item(
        self,
        *,
        is_opponent: bool,
        ident: str | None,
        item: str | None,
        revealed: bool = True,
        present: bool = True,
    ) -> PokemonState:
        pokemon = self._active_pokemon_for_ident(is_opponent=is_opponent, ident=ident)
        pokemon.note_item(item, revealed=revealed, present=present)
        return pokemon

    def note_ability(self, *, is_opponent: bool, ident: str | None, ability: str | None, revealed: bool = True) -> PokemonState:
        pokemon = self._active_pokemon_for_ident(is_opponent=is_opponent, ident=ident)
        pokemon.note_ability(ability, revealed=revealed)
        return pokemon

    def note_status(
        self,
        *,
        is_opponent: bool,
        ident: str | None,
        status: str | None,
        clear: bool = False,
    ) -> PokemonState:
        pokemon = self._active_pokemon_for_ident(is_opponent=is_opponent, ident=ident)
        normalized_status = normalize_identifier(status)
        if clear or not normalized_status:
            pokemon.status = None
        else:
            pokemon.status = normalized_status
        return pokemon

    def note_volatile_status(
        self,
        *,
        is_opponent: bool,
        ident: str | None,
        volatile_status: str | None,
        active: bool = True,
        value: Any = True,
    ) -> PokemonState:
        pokemon = self._active_pokemon_for_ident(is_opponent=is_opponent, ident=ident)
        if active:
            pokemon.add_volatile_status(volatile_status, value=value)
        else:
            pokemon.remove_volatile_status(volatile_status)
        return pokemon

    def note_boost(self, *, is_opponent: bool, ident: str | None, stat_name: str, delta: int) -> PokemonState:
        pokemon = self._active_pokemon_for_ident(is_opponent=is_opponent, ident=ident)
        pokemon.note_boost(stat_name, delta)
        return pokemon

    def set_boost(self, *, is_opponent: bool, ident: str | None, stat_name: str, stage: int) -> PokemonState:
        pokemon = self._active_pokemon_for_ident(is_opponent=is_opponent, ident=ident)
        pokemon.set_stat_stage(stat_name, stage)
        return pokemon

    def clear_boost(self, *, is_opponent: bool, ident: str | None, stat_name: str) -> PokemonState:
        pokemon = self._active_pokemon_for_ident(is_opponent=is_opponent, ident=ident)
        pokemon.clear_stat_stage(stat_name)
        return pokemon

    def clear_all_boosts(self, *, is_opponent: bool, ident: str | None) -> PokemonState:
        pokemon = self._active_pokemon_for_ident(is_opponent=is_opponent, ident=ident)
        pokemon.clear_all_stat_stages()
        return pokemon

    def note_immune(self, *, is_opponent: bool, ident: str | None, move_type: str | None) -> PokemonState:
        pokemon = self._active_pokemon_for_ident(is_opponent=is_opponent, ident=ident)
        pokemon.volatile_statuses.setdefault("immune_to", set())
        immune_to = pokemon.volatile_statuses["immune_to"]
        if isinstance(immune_to, set):
            normalized_move_type = normalize_identifier(move_type)
            if normalized_move_type:
                immune_to.add(normalized_move_type)
        return pokemon

    def _active_pokemon_for_ident(self, *, is_opponent: bool, ident: str | None) -> PokemonState:
        team = self.team_for_side(is_opponent)
        active_pokemon = team.active_pokemon
        if active_pokemon is not None:
            return active_pokemon
        slot_index = slot_index_from_ident(ident)
        if slot_index is None:
            slot_index = team.active_index
        return team.ensure_slot(slot_index)

    def _sync_legacy_indices(self) -> None:
        self.own_team.active_index = max(0, min(5, self.own_team.active_index))
        self.opponent_team.active_index = max(0, min(5, self.opponent_team.active_index))

    def clear_turn_restriction_statuses(self) -> None:
        if self.active_own_pokemon is not None:
            self.active_own_pokemon.clear_turn_restriction_statuses()
        if self.active_opponent_pokemon is not None:
            self.active_opponent_pokemon.clear_turn_restriction_statuses()
