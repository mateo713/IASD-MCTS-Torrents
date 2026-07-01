from __future__ import annotations

import copy
import math
import random
import re
from dataclasses import dataclass
from typing import Any

from core.models import ActionChoice, BattleSnapshot
from core.identifiers import extract_status_from_condition, hp_fraction_from_condition
from engine.gen5_datasets import build_stat_profile, get_feasible_random_battle_sets
from engine.inference import (
    ABILITY_IMMUNITIES,
    _load_moves_data,
    estimate_expected_damage,
    get_move_data_from_action,
    get_move_id_from_action,
    normalize_entity_id,
    split_action_space,
    get_type_effectiveness_multiplier,
)
from engine.turn_analysis import effective_speed as pokemon_effective_speed
from engine.turn_analysis import hp_fraction as turn_hp_fraction
from engine.turn_analysis import stage_multiplier as turn_stage_multiplier
from strategies.base import Strategy
from strategies.base import fallback_action


class OneTurnExpectedDamageStrategy(Strategy):
    name = "rules_expected_damage"

    def choose_action(self, snapshot: BattleSnapshot, rng: random.Random) -> ActionChoice:
        if not snapshot.available_actions:
            return fallback_action(snapshot)

        moves, switches, others = split_action_space(snapshot)

        # If our active Pokemon is forced to recharge / truant (cannot act),
        # avoid selecting a move. Prefer switching when available, otherwise
        # fall back to other actions or the default fallback.
        battle_state = snapshot.battle_state
        if battle_state is not None and battle_state.active_own_pokemon is not None:
            active = battle_state.active_own_pokemon
            if active.has_volatile_status("mustrecharge") or active.has_volatile_status("truant"):
                if switches:
                    return rng.choice(switches)
                if others:
                    return rng.choice(others)
                return fallback_action(snapshot)

        if moves:
            scored_moves = [(move, estimate_expected_damage(snapshot, move)) for move in moves]
            best_score = max(score for _, score in scored_moves)
            if best_score > 0:
                return self._choose_near_tie_damage_move(snapshot, scored_moves)

            if switches:
                return rng.choice(switches)

            status_moves: list[ActionChoice] = []
            for move in moves:
                move_data = get_move_data_from_action(snapshot, move)
                category = str(move_data.get("category", "status")).lower()
                if category == "status":
                    status_moves.append(move)

            if status_moves:
                return rng.choice(status_moves)

            return rng.choice(moves)

        if switches:
            return rng.choice(switches)
        if others:
            return rng.choice(others)

        return fallback_action(snapshot)

    def _choose_near_tie_damage_move(
        self,
        snapshot: BattleSnapshot,
        scored_moves: list[tuple[ActionChoice, float]],
        near_tie_ratio: float = 0.2,
    ) -> ActionChoice:
        if not scored_moves:
            raise RuntimeError("Expected at least one scored move")

        best_score = max(score for _, score in scored_moves)
        near_tie_cutoff = best_score * (1.0 - max(0.0, min(0.9, near_tie_ratio)))
        near_tie_candidates = [
            (move, score)
            for move, score in scored_moves
            if score >= near_tie_cutoff
        ]
        if not near_tie_candidates:
            near_tie_candidates = scored_moves

        # Step 1: In near-ties, discard self stat-lowering attacks if possible.
        no_drop_candidates = [
            (move, score)
            for move, score in near_tie_candidates
            if self._self_drop_penalty(get_move_data_from_action(snapshot, move))[0] == 0.0
        ]
        if no_drop_candidates:
            near_tie_candidates = no_drop_candidates

        # Step 2: If exactly one candidate has recoil, discard that one.
        recoil_marks = [
            (move, score, self._recoil_fraction(get_move_data_from_action(snapshot, move)) > 0.0)
            for move, score in near_tie_candidates
        ]
        recoil_count = sum(1 for _, _, has_recoil in recoil_marks if has_recoil)
        if 0 < recoil_count < len(recoil_marks):
            near_tie_candidates = [(move, score) for move, score, has_recoil in recoil_marks if not has_recoil]

        # Step 3: pick the highest expected damage among remaining candidates.
        best_remaining = max(score for _, score in near_tie_candidates)
        best_exact = [(move, score) for move, score in near_tie_candidates if score == best_remaining]
        if len(best_exact) == 1:
            return best_exact[0][0]

        # Step 4: use accuracy only to break exact-equality ties.
        best_move = best_exact[0][0]
        best_accuracy = self._move_accuracy(get_move_data_from_action(snapshot, best_move))
        for move, _ in best_exact[1:]:
            candidate_accuracy = self._move_accuracy(get_move_data_from_action(snapshot, move))
            if candidate_accuracy > best_accuracy:
                best_move = move
                best_accuracy = candidate_accuracy
        return best_move

    def _self_drop_penalty(self, move_data: dict[str, Any]) -> tuple[float, float]:
        total_drop = 0.0

        def _accumulate_from_boosts(boosts: Any) -> None:
            nonlocal total_drop
            if not isinstance(boosts, dict):
                return
            for value in boosts.values():
                try:
                    numeric = float(value or 0)
                except Exception:
                    continue
                if numeric < 0:
                    total_drop += abs(numeric)

        _accumulate_from_boosts(move_data.get("boosts"))

        self_payload = move_data.get("self")
        if isinstance(self_payload, dict):
            _accumulate_from_boosts(self_payload.get("boosts"))

        self_boost_payload = move_data.get("selfBoost")
        if isinstance(self_boost_payload, dict):
            _accumulate_from_boosts(self_boost_payload.get("boosts"))

        has_drop = 1.0 if total_drop > 0 else 0.0
        return has_drop, total_drop

    def _recoil_fraction(self, move_data: dict[str, Any]) -> float:
        recoil = move_data.get("recoil")
        if isinstance(recoil, list) and len(recoil) == 2:
            try:
                return max(0.0, float(recoil[0]) / max(1.0, float(recoil[1])))
            except Exception:
                return 0.0
        return 0.0

    def _move_accuracy(self, move_data: dict[str, Any]) -> float:
        accuracy_raw = move_data.get("accuracy", 100)
        try:
            accuracy = 100.0 if isinstance(accuracy_raw, bool) else float(accuracy_raw or 100)
        except Exception:
            accuracy = 100.0
        return max(0.0, min(100.0, accuracy)) / 100.0


STATUS_MOVE_IDS: tuple[str, ...] = (
    "toxic",
    "willowisp",
    "thunderwave",
    "glare",
    "sleeppowder",
    "spore",
)
ENTRY_HAZARD_MOVE_IDS: tuple[str, ...] = (
    "stealthrock",
    "stickyweb",
    "toxicspikes",
    "spikes",
)
LAYERED_HAZARD_MOVE_IDS: tuple[str, ...] = (
    "spikes",
    "toxicspikes",
)
SCREEN_MOVE_IDS: tuple[str, ...] = ("reflect", "lightscreen")


@dataclass(slots=True)
class OpponentDamageSummary:
    expected_damage: float
    best_move_id: str
    best_move_category: str
    best_move_flinch_chance: float
    outspeeds: bool


def _stage_multiplier(stage: int) -> float:
    return turn_stage_multiplier(stage)


def _extract_active_condition(request: dict[str, Any]) -> str:
    side = request.get("side", {})
    for pokemon in side.get("pokemon", []):
        if pokemon.get("active"):
            return str(pokemon.get("condition", ""))
    return ""


def _extract_hp_fraction_from_condition(condition: str) -> float | None:
    return hp_fraction_from_condition(condition)


def _extract_status_from_condition(condition: str) -> str | None:
    return extract_status_from_condition(condition)


def _effective_species_speed(snapshot: BattleSnapshot, own: bool) -> float:
    battle_state = snapshot.battle_state
    if battle_state is not None:
        pokemon = battle_state.active_own_pokemon if own else battle_state.active_opponent_pokemon
        if pokemon is not None:
            return float(pokemon_effective_speed(pokemon, field=battle_state.battlefield))

    species = snapshot.active_species if own else snapshot.opponent_active_species
    if not species:
        return 0.0
    try:
        base_speed = float(build_stat_profile(species, level=100, nature="serious").stats.get("speed", 0))
    except Exception:
        base_speed = 0.0
    if not own:
        base_speed *= _stage_multiplier(snapshot.opponent_constraints.speed_stage)
        if snapshot.opponent_constraints.inferred_choice_scarf:
            base_speed *= 1.5
        if snapshot.opponent_constraints.is_paralyzed:
            base_speed *= 0.25
    return base_speed


def _extract_side_conditions(request: dict[str, Any], key_candidates: tuple[str, ...]) -> dict[str, Any]:
    for key in key_candidates:
        candidate = request.get(key)
        if isinstance(candidate, dict):
            return candidate
    side = request.get("side", {})
    for key in key_candidates:
        candidate = side.get(key)
        if isinstance(candidate, dict):
            return candidate
    return {}


def _extract_layers(side_conditions: dict[str, Any], condition_id: str) -> int:
    value = side_conditions.get(condition_id)
    if isinstance(value, dict):
        try:
            return int(value.get("layers", 1) or 1)
        except Exception:
            return 1
    if isinstance(value, (int, float)):
        return int(value)
    if value:
        return 1
    return 0


def _battlefield_side_conditions(snapshot: BattleSnapshot, own: bool) -> dict[str, Any]:
    battle_state = snapshot.battle_state
    if battle_state is not None:
        side_key = "own" if own else "opponent"
        side_conditions = battle_state.battlefield.side_conditions.get(side_key)
        if isinstance(side_conditions, dict):
            return side_conditions

    request = snapshot.last_request or {}
    if own:
        return _extract_side_conditions(request, ("sideConditions",))
    return _extract_side_conditions(request, ("foeSideConditions", "opponentSideConditions", "enemySideConditions"))


def _active_status(snapshot: BattleSnapshot, own: bool) -> str | None:
    battle_state = snapshot.battle_state
    if battle_state is not None:
        pokemon = battle_state.active_own_pokemon if own else battle_state.active_opponent_pokemon
        if pokemon is not None and pokemon.status:
            return pokemon.status

    request = snapshot.last_request or {}
    if own:
        condition = _extract_active_condition(request)
        return _extract_status_from_condition(condition)

    active_side = request.get("active", [])
    if active_side:
        active_entry = active_side[0]
        condition = str(active_entry.get("condition", ""))
        return _extract_status_from_condition(condition)
    return None


def _active_species(snapshot: BattleSnapshot, own: bool) -> str | None:
    species = snapshot.active_species if own else snapshot.opponent_active_species
    if species:
        return species

    battle_state = snapshot.battle_state
    if battle_state is None:
        return None

    pokemon = None
    try:
        pokemon = battle_state.active_own_pokemon if own else battle_state.active_opponent_pokemon
    except Exception:
        pokemon = None
    if pokemon is None:
        return None

    for attr_name in ("species_id", "id", "display_name", "name"):
        value = getattr(pokemon, attr_name, None)
        if value:
            return str(value).split(",", 1)[0].strip()
    return None


def _softmax_sample_index(values: list[float], rng: random.Random, temperature: float = 1.0) -> int:
    if not values:
        return 0
    capped_values = [max(-20.0, min(20.0, float(v))) for v in values]
    max_value = max(capped_values)
    exps = [math.exp((value - max_value) / max(0.05, temperature)) for value in capped_values]
    total = sum(exps)
    roll = rng.random() * total
    acc = 0.0
    for idx, value in enumerate(exps):
        acc += value
        if roll <= acc:
            return idx
    return len(values) - 1


class HeuristicStrategy(Strategy):
    name = "heuristic"
    _DUMMY_ZERO_DAMAGE_MOVE_ID = "splash"

    def choose_action(self, snapshot: BattleSnapshot, rng: random.Random) -> ActionChoice:
        """Choose an action with a simple deterministic-first heuristic decision tree.

        Protocol order:
        1. Prefer one-turn control moves (status, hazards, and screens) when they are useful.
           - Hazards use a decaying probability gate for Spikes: $p=1/2^{layers}$.
           - Reflect/Light Screen are selected from opponent feasible-set attack category counts.
           - Skip hazards when opponent is detected as offensively boosted.
           - If at least one feasible opponent set can both outspeed and KO us this turn, force switch logic.
        2. Compute a 2-turn stay-vs-switch score as damage ratio $r=\frac{dealt}{taken}$.
           - Stay score: both sides attack optimally for two turns with KO/order effects.
           - Switch score: include immediate switch-in damage plus next-turn optimal exchange.
           - Sample among stay and all legal switches with softmax over these ratios.
        3. If staying, select between setup, direct damage, or healing.
           - Immediate KO in our favor if available.
           - If we effectively have one move before KO, use best direct damage.
           - If outsped, prefer setup that grants speed control (always if also boosts offense/defense, else 50%).
           - Otherwise compare pre/post-setup KO-time ratios and pick best improving setup move.
           - Consider Recover/Roost-like healing (Wish explicitly ignored), including expected move failure
             from paralysis/flinch and end-of-turn residual effects.
           - Fall back to highest expected direct damage.

        Common-sense patches applied for missing runtime observability:
        - Opponent HP is approximated as full when exact value is unavailable in the snapshot.
        - Opponent offensive boosts are inferred only from request payload keys if present.
        - Hazard/screen-side state is read from best-effort request keys; unknown states are treated conservatively.
        - Switch-in next-turn offense is estimated from species feasible-set move quality when move list is unknown.
        """
        if not snapshot.available_actions:
            return fallback_action(snapshot)

        moves, switches, others = split_action_space(snapshot)

        # If our active Pokemon is forced to recharge / truant (cannot act),
        # avoid selecting a move. Prefer switching when available, otherwise
        # fall back to other actions or the default fallback.
        battle_state = snapshot.battle_state
        if battle_state is not None and battle_state.active_own_pokemon is not None:
            active = battle_state.active_own_pokemon
            if active.has_volatile_status("mustrecharge") or active.has_volatile_status("truant"):
                if switches:
                    switch_choice = self._choose_switch_from_ratios(snapshot, switches, rng, force_switch=True)
                    if switch_choice is not None:
                        return switch_choice
                    return rng.choice(switches)
                if others:
                    return rng.choice(others)
                return fallback_action(snapshot)

        if not moves:
            if switches:
                switch_choice = self._choose_switch_from_ratios(snapshot, switches, rng, force_switch=True)
                if switch_choice is not None:
                    return switch_choice
            if others:
                return rng.choice(others)
            return fallback_action(snapshot)

        move_scores = self._score_damaging_moves(snapshot, moves)
        best_damage_move = self._choose_best_scored_move(snapshot, move_scores, rng) if move_scores else None
        own_hp_fraction = self._own_hp_fraction(snapshot)
        status_or_field_move = self._choose_preferred_control_move(snapshot, moves, rng)

        if status_or_field_move is not None:
            if self._exists_outspeed_ko_set(snapshot):
                switch_choice = self._choose_switch_from_ratios(snapshot, switches, rng, force_switch=True)
                if switch_choice is not None:
                    return switch_choice
            else:
                move_id = get_move_id_from_action(snapshot, status_or_field_move)
                if move_id in ENTRY_HAZARD_MOVE_IDS and self._opponent_has_offensive_boost(snapshot):
                    pass
                else:
                    return status_or_field_move

        switch_choice = self._choose_switch_from_ratios(snapshot, switches, rng, force_switch=False)
        if switch_choice is not None:
            return switch_choice

        if best_damage_move is None:
            if status_or_field_move is not None:
                return status_or_field_move
            return rng.choice(moves)

        if self._can_ko_now(snapshot, move_scores):
            return best_damage_move

        own_hp_fraction = self._own_hp_fraction(snapshot)
        own_max_hp = max(1.0, self._estimate_own_max_hp(snapshot.active_species, snapshot=snapshot))
        opp_summary = self._estimate_opponent_best_damage(snapshot)
        taken_fraction = opp_summary.expected_damage / own_max_hp
        moves_before_ko = max(1, math.ceil(own_hp_fraction / max(1e-6, taken_fraction)))

        if moves_before_ko <= 1:
            return best_damage_move

        setup_choice = self._maybe_choose_speed_setup(snapshot, moves, opp_summary.outspeeds, rng)
        if setup_choice is not None:
            return setup_choice

        setup_choice = self._maybe_choose_ratio_improving_setup(snapshot, moves, move_scores, opp_summary, rng)
        if setup_choice is not None:
            return setup_choice

        heal_choice = self._maybe_choose_heal(snapshot, moves, move_scores, opp_summary, rng)
        if heal_choice is not None:
            return heal_choice

        return best_damage_move

    def _score_damaging_moves(self, snapshot: BattleSnapshot, moves: list[ActionChoice]) -> list[tuple[ActionChoice, float]]:
        scored: list[tuple[ActionChoice, float]] = []
        for move in moves:
            move_data = get_move_data_from_action(snapshot, move)
            category = str(move_data.get("category", "status")).lower()
            if category == "status":
                continue
            scored.append((move, estimate_expected_damage(snapshot, move)))
        return scored

    def _choose_best_scored_move(
        self,
        snapshot: BattleSnapshot,
        move_scores: list[tuple[ActionChoice, float]],
        rng: random.Random,
    ) -> ActionChoice:
        if not move_scores:
            raise RuntimeError("Heuristic strategy requires at least one move")
        return self._choose_near_tie_damage_move(snapshot, move_scores)

    def _choose_near_tie_damage_move(
        self,
        snapshot: BattleSnapshot,
        scored_moves: list[tuple[ActionChoice, float]],
        near_tie_ratio: float = 0.2,
    ) -> ActionChoice:
        if not scored_moves:
            raise RuntimeError("Expected at least one scored move")

        best_score = max(score for _, score in scored_moves)
        near_tie_cutoff = best_score * (1.0 - max(0.0, min(0.9, near_tie_ratio)))
        near_tie_candidates = [
            (move, score)
            for move, score in scored_moves
            if score >= near_tie_cutoff
        ]
        if not near_tie_candidates:
            near_tie_candidates = scored_moves

        # Step 1: In near-ties, discard self stat-lowering attacks if possible.
        no_drop_candidates = [
            (move, score)
            for move, score in near_tie_candidates
            if self._self_drop_penalty(get_move_data_from_action(snapshot, move))[0] == 0.0
        ]
        if no_drop_candidates:
            near_tie_candidates = no_drop_candidates

        # Step 2: If recoil/non-recoil are mixed, discard recoil options.
        recoil_marks = [
            (move, score, self._recoil_fraction(get_move_data_from_action(snapshot, move)) > 0.0)
            for move, score in near_tie_candidates
        ]
        recoil_count = sum(1 for _, _, has_recoil in recoil_marks if has_recoil)
        if 0 < recoil_count < len(recoil_marks):
            near_tie_candidates = [(move, score) for move, score, has_recoil in recoil_marks if not has_recoil]

        # Step 3: pick highest expected damage among remaining candidates.
        best_remaining = max(score for _, score in near_tie_candidates)
        best_exact = [(move, score) for move, score in near_tie_candidates if score == best_remaining]
        if len(best_exact) == 1:
            return best_exact[0][0]

        # Step 4: use accuracy only for exact score ties.
        best_move = best_exact[0][0]
        best_accuracy = self._move_accuracy(get_move_data_from_action(snapshot, best_move))
        for move, _ in best_exact[1:]:
            candidate_accuracy = self._move_accuracy(get_move_data_from_action(snapshot, move))
            if candidate_accuracy > best_accuracy:
                best_move = move
                best_accuracy = candidate_accuracy
        return best_move

    def _self_drop_penalty(self, move_data: dict[str, Any]) -> tuple[float, float]:
        total_drop = 0.0

        def _accumulate_from_boosts(boosts: Any) -> None:
            nonlocal total_drop
            if not isinstance(boosts, dict):
                return
            for value in boosts.values():
                try:
                    numeric = float(value or 0)
                except Exception:
                    continue
                if numeric < 0:
                    total_drop += abs(numeric)

        _accumulate_from_boosts(move_data.get("boosts"))

        self_payload = move_data.get("self")
        if isinstance(self_payload, dict):
            _accumulate_from_boosts(self_payload.get("boosts"))

        self_boost_payload = move_data.get("selfBoost")
        if isinstance(self_boost_payload, dict):
            _accumulate_from_boosts(self_boost_payload.get("boosts"))

        has_drop = 1.0 if total_drop > 0 else 0.0
        return has_drop, total_drop

    def _recoil_fraction(self, move_data: dict[str, Any]) -> float:
        recoil = move_data.get("recoil")
        if isinstance(recoil, list) and len(recoil) == 2:
            try:
                return max(0.0, float(recoil[0]) / max(1.0, float(recoil[1])))
            except Exception:
                return 0.0
        return 0.0

    def _move_accuracy(self, move_data: dict[str, Any]) -> float:
        accuracy_raw = move_data.get("accuracy", 100)
        try:
            accuracy = 100.0 if isinstance(accuracy_raw, bool) else float(accuracy_raw or 100)
        except Exception:
            accuracy = 100.0
        return max(0.0, min(100.0, accuracy)) / 100.0

    def _break_damage_tie(self, snapshot: BattleSnapshot, moves: list[ActionChoice]) -> ActionChoice:
        scored: list[tuple[tuple[float, float, float, int], ActionChoice]] = []
        for index, move in enumerate(moves):
            move_data = get_move_data_from_action(snapshot, move)
            accuracy_raw = move_data.get("accuracy", 100)
            accuracy = 100.0 if isinstance(accuracy_raw, bool) else float(accuracy_raw or 100)
            base_power = float(move_data.get("basePower", 0) or 0)
            recoil = move_data.get("recoil")
            recoil_fraction = 0.0
            if isinstance(recoil, list) and len(recoil) == 2:
                try:
                    recoil_fraction = max(0.0, float(recoil[0]) / max(1.0, float(recoil[1])))
                except Exception:
                    recoil_fraction = 0.0
            score = (accuracy, base_power, -recoil_fraction, -index)
            scored.append((score, move))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]

    def _choose_preferred_control_move(
        self,
        snapshot: BattleSnapshot,
        moves: list[ActionChoice],
        rng: random.Random,
    ) -> ActionChoice | None:
        candidate_by_id: dict[str, ActionChoice] = {}
        for move in moves:
            move_id = get_move_id_from_action(snapshot, move)
            if not move_id:
                continue
            candidate_by_id[move_id] = move

        screen_move = self._choose_screen_move(snapshot, candidate_by_id)
        if screen_move is not None:
            return screen_move

        for move_id in STATUS_MOVE_IDS:
            candidate = candidate_by_id.get(move_id)
            if candidate is not None and self._status_move_is_useful(snapshot, candidate):
                return candidate

        for move_id in ENTRY_HAZARD_MOVE_IDS:
            candidate = candidate_by_id.get(move_id)
            if candidate is None:
                continue
            if not self._hazard_move_is_useful(snapshot, move_id, rng):
                continue
            return candidate

        return None

    def _status_move_is_useful(self, snapshot: BattleSnapshot, move: ActionChoice) -> bool:
        move_data = get_move_data_from_action(snapshot, move)
        if not move_data:
            return False
        if str(move_data.get("category", "")).lower() != "status":
            return False
        move_id = get_move_id_from_action(snapshot, move)
        if move_id not in STATUS_MOVE_IDS:
            return False
        status = str(move_data.get("status", "")).lower()
        if status in {"brn", "psn", "tox", "par", "slp"}:
            opponent_status = _active_status(snapshot, own=False)
            if opponent_status is not None:
                return False
            if status == "brn" and "fire" in self._opponent_types(snapshot):
                return False
            if status == "par" and move_id == "thunderwave" and any(t in {"ground", "electric"} for t in self._opponent_types(snapshot)):
                return False 
            if status in {"psn", "tox"} and any(t in {"poison", "steel"} for t in self._opponent_types(snapshot)):
                return False
            return True
        return True

    def _hazard_move_is_useful(self, snapshot: BattleSnapshot, move_id: str, rng: random.Random) -> bool:
        foe_side_conditions = _battlefield_side_conditions(snapshot, own=False)
        if move_id == "spikes":
            layers = _extract_layers(foe_side_conditions, "spikes")
            if layers >= 3:
                return False
            probability = 1.0 / (2 ** max(0, layers))
            return rng.random() < probability
        if move_id == "toxicspikes":
            return _extract_layers(foe_side_conditions, "toxicspikes") < 2
        if move_id in {"stealthrock", "stickyweb"}:
            return _extract_layers(foe_side_conditions, move_id) < 1
        return True

    def _choose_screen_move(
        self,
        snapshot: BattleSnapshot,
        candidate_by_id: dict[str, ActionChoice],
    ) -> ActionChoice | None:
        reflect = candidate_by_id.get("reflect")
        light_screen = candidate_by_id.get("lightscreen")
        if reflect is None and light_screen is None:
            return None

        foe_side_conditions = _battlefield_side_conditions(snapshot, own=False)
        if reflect is not None and _extract_layers(foe_side_conditions, "reflect") >= 1:
            reflect = None
        if light_screen is not None and _extract_layers(foe_side_conditions, "lightscreen") >= 1:
            light_screen = None

        if reflect is None and light_screen is None:
            return None

        physical_count = 0
        special_count = 0
        feasible_sets = self._feasible_opponent_sets(snapshot)
        for set_entry in feasible_sets:
            weight = int(set_entry.get("count", 1) or 1)
            for move_id in set_entry.get("moves", []):
                norm_id = normalize_entity_id(str(move_id))
                move_data = self._move_data_by_id(snapshot, norm_id)
                category = str(move_data.get("category", "status")).lower()
                if category == "physical":
                    physical_count += weight
                elif category == "special":
                    special_count += weight

        prefer_reflect = physical_count >= special_count
        if prefer_reflect and reflect is not None:
            return reflect
        if (not prefer_reflect) and light_screen is not None:
            return light_screen
        return reflect or light_screen

    def _feasible_opponent_sets(self, snapshot: BattleSnapshot) -> list[dict[str, Any]]:
        opponent_species = _active_species(snapshot, own=False)
        sets = get_feasible_random_battle_sets(
            opponent_species,
            revealed_moves=sorted(snapshot.opponent_constraints.revealed_moves),
            revealed_ability=snapshot.opponent_constraints.revealed_ability,
            revealed_item=snapshot.opponent_constraints.revealed_item,
            impossible_abilities=snapshot.opponent_constraints.impossible_abilities,
            impossible_items=snapshot.opponent_constraints.impossible_items,
        )
        return sets if sets else [{}]

    def _move_data_by_id(self, snapshot: BattleSnapshot, move_id: str) -> dict[str, Any]:
        return _load_moves_data().get(move_id, {})

    def _exists_outspeed_ko_set(self, snapshot: BattleSnapshot) -> bool:
        own_max_hp = max(1.0, self._estimate_own_max_hp(snapshot.active_species, snapshot=snapshot))
        own_hp_fraction = self._own_hp_fraction(snapshot)
        for set_entry in self._feasible_opponent_sets(snapshot):
            summary = self._estimate_opponent_best_damage(snapshot, opponent_set_override=set_entry)
            if not summary.outspeeds:
                continue
            if summary.expected_damage >= own_max_hp * own_hp_fraction:
                return True
        return False

    def _choose_switch_from_ratios(
        self,
        snapshot: BattleSnapshot,
        switches: list[ActionChoice],
        rng: random.Random,
        force_switch: bool,
    ) -> ActionChoice | None:
        if not switches and force_switch:
            return None

        stay_ratio = self._two_turn_ratio_stay(snapshot)

        if force_switch:
            ratios: list[float] = []
            actions: list[ActionChoice | None] = []
            for switch_action in switches:
                ratio = self._two_turn_ratio_switch(snapshot, switch_action)
                ratios.append(ratio)
                actions.append(switch_action)
            if not ratios:
                return None
            best_index = max(range(len(ratios)), key=lambda idx: (ratios[idx], -idx))
            return actions[best_index]

        switch_ratios: list[float] = []
        best_switch_ratio = float("-inf")
        for switch_action in switches:
            ratio = self._two_turn_ratio_switch(snapshot, switch_action)
            switch_ratios.append(ratio)
            if ratio > best_switch_ratio:
                best_switch_ratio = ratio

        if not switch_ratios:
            return None

        # Switch whenever the best available pivot is at least as good as
        # staying, with a tiny epsilon to avoid churn on exact ties.
        if best_switch_ratio <= stay_ratio * 1.01:
            return None

        best_index = max(range(len(switch_ratios)), key=lambda idx: (switch_ratios[idx], -idx))
        return switches[best_index]

    def _two_turn_ratio_stay(self, snapshot: BattleSnapshot) -> float:
        own_damage = self._best_own_attack_damage(snapshot)
        opponent_summary = self._estimate_opponent_best_damage(snapshot)
        own_current_hp = self._current_hp_value(snapshot, own=True)
        opp_current_hp = self._current_hp_value(snapshot, own=False)
        return self._two_turn_ratio(
            own_damage,
            opponent_summary.expected_damage,
            own_moves_first=(not opponent_summary.outspeeds),
            own_current_hp=own_current_hp,
            opp_current_hp=opp_current_hp,
        )

    def _two_turn_ratio_switch(self, snapshot: BattleSnapshot, switch_action: ActionChoice) -> float:
        switched = self._snapshot_after_switch(snapshot, switch_action)
        incoming = self._estimate_opponent_best_damage(switched).expected_damage

        next_turn_own_damage = self._estimate_switch_in_offense(switched)
        next_turn_opponent = self._estimate_opponent_best_damage(switched)
        own_current_hp = self._current_hp_value(switched, own=True)
        opp_current_hp = self._current_hp_value(switched, own=False)
        incoming_fraction = max(0.0, incoming) / max(1e-6, own_current_hp)
        own_after_switch = max(0.0, own_current_hp - max(0.0, incoming))
        if own_after_switch <= 0.0:
            return 0.0

        own_speed = _effective_species_speed(switched, own=True)
        opp_speed = _effective_species_speed(switched, own=False)
        own_moves_first = own_speed >= opp_speed
        if snapshot.battle_state is not None and snapshot.battle_state.battlefield is not None and snapshot.battle_state.battlefield.trick_room:
            own_moves_first = own_speed <= opp_speed

        dealt_fraction, taken_fraction = self._single_exchange_fractions(
            dealt=next_turn_own_damage,
            taken=next_turn_opponent.expected_damage,
            own_moves_first=own_moves_first,
            own_current_hp=own_after_switch,
            opp_current_hp=opp_current_hp,
        )
        if dealt_fraction <= 1e-6:
            return 0.0

        switch_cost = incoming_fraction + taken_fraction
        if switch_cost <= 1e-6:
            return 1e6
        return dealt_fraction / switch_cost

    def _two_turn_ratio(
        self,
        dealt: float,
        taken: float,
        own_moves_first: bool,
        own_current_hp: float,
        opp_current_hp: float,
    ) -> float:
        own_dealt, own_taken = self._single_exchange_fractions(
            dealt=dealt,
            taken=taken,
            own_moves_first=own_moves_first,
            own_current_hp=own_current_hp,
            opp_current_hp=opp_current_hp,
        )
        if own_taken <= 1e-6:
            if own_dealt <= 1e-6:
                return 1.0
            return 1e6
        return own_dealt / own_taken

    def _single_exchange_fractions(
        self,
        *,
        dealt: float,
        taken: float,
        own_moves_first: bool,
        own_current_hp: float,
        opp_current_hp: float,
    ) -> tuple[float, float]:
        own_remaining = max(0.0, own_current_hp)
        opp_remaining = max(0.0, opp_current_hp)
        own_dealt = 0.0
        own_taken = 0.0

        if own_moves_first:
            if dealt > 0.0:
                attack = min(max(0.0, dealt), opp_remaining)
                own_dealt += attack / max(1e-6, opp_current_hp)
                opp_remaining -= attack
            if opp_remaining > 1e-6 and taken > 0.0:
                retaliation = min(max(0.0, taken), own_remaining)
                own_taken += retaliation / max(1e-6, own_current_hp)
        else:
            if taken > 0.0:
                retaliation = min(max(0.0, taken), own_remaining)
                own_taken += retaliation / max(1e-6, own_current_hp)
                own_remaining -= retaliation
            if own_remaining > 1e-6 and dealt > 0.0:
                attack = min(max(0.0, dealt), opp_remaining)
                own_dealt += attack / max(1e-6, opp_current_hp)

        return own_dealt, own_taken

    def _current_hp_value(self, snapshot: BattleSnapshot, own: bool) -> float:
        hp_fraction = self._own_hp_fraction(snapshot) if own else self._opponent_hp_fraction(snapshot)
        hp_fraction = max(0.0, min(1.0, hp_fraction))
        species = snapshot.active_species if own else snapshot.opponent_active_species
        max_hp = self._estimate_own_max_hp(species, snapshot=snapshot)
        return max(1.0, max_hp * hp_fraction)

    def _snapshot_after_switch(self, snapshot: BattleSnapshot, switch_action: ActionChoice) -> BattleSnapshot:
        request = copy.deepcopy(snapshot.last_request or {})
        side = request.get("side", {})
        team = side.get("pokemon", [])
        battle_state = copy.deepcopy(snapshot.battle_state) if snapshot.battle_state is not None else None

        switch_slot_match = re.search(r"(\d+)$", switch_action.command)
        switch_slot = int(switch_slot_match.group(1)) if switch_slot_match else 1
        switch_index = max(0, min(len(team) - 1, switch_slot - 1)) if team else 0

        switched_species = snapshot.active_species
        hp_fraction = snapshot.own_active_hp_fraction

        for idx, entry in enumerate(team):
            entry["active"] = idx == switch_index
            if idx == switch_index:
                details = str(entry.get("details", ""))
                switched_species = details.split(",", 1)[0].strip() if details else switched_species
                condition = str(entry.get("condition", ""))
                extracted = _extract_hp_fraction_from_condition(condition)
                if extracted is not None:
                    hp_fraction = extracted

        if battle_state is not None:
            try:
                battle_state.update_from_request(request)
            except Exception:
                pass

        switched = BattleSnapshot(
            room_id=snapshot.room_id,
            turn=snapshot.turn,
            phase=snapshot.phase,
            request_id=snapshot.request_id,
            available_actions=snapshot.available_actions,
            last_request=request,
            winner=snapshot.winner,
            own_side_id=snapshot.own_side_id,
            active_species=switched_species,
            opponent_active_species=snapshot.opponent_active_species,
            own_active_hp_fraction=hp_fraction,
            battle_state=battle_state,
            opponent_constraints=snapshot.opponent_constraints,
        )
        return switched

    def _best_own_attack_damage(self, snapshot: BattleSnapshot) -> float:
        moves, _, _ = split_action_space(snapshot)
        move_scores = self._score_damaging_moves(snapshot, moves)
        if not move_scores:
            return 0.0
        return max(score for _, score in move_scores)

    def _estimate_opponent_best_damage(
        self,
        snapshot: BattleSnapshot,
        opponent_set_override: dict[str, Any] | None = None,
    ) -> OpponentDamageSummary:
        feasible_sets = [opponent_set_override] if opponent_set_override is not None else self._feasible_opponent_sets(snapshot)
        if not feasible_sets:
            return OpponentDamageSummary(
                0.0,
                "",
                "physical",
                0.0,
                outspeeds=_effective_species_speed(snapshot, own=False) > _effective_species_speed(snapshot, own=True),
            )

        attacker_types = self._types_for_species(snapshot.opponent_active_species, snapshot=snapshot, own=False)
        defender_types = self._types_for_species(snapshot.active_species, snapshot=snapshot, own=True)
        defender_state = snapshot.battle_state.active_own_pokemon if snapshot.battle_state is not None else None
        defender_max_hp = 300.0
        if defender_state is not None and defender_state.max_hp is not None:
            try:
                defender_max_hp = max(1.0, float(defender_state.max_hp))
            except Exception:
                defender_max_hp = 300.0

        def _likely_ability(species_name: str | None, pokemon_state: Any | None) -> str:
            revealed = normalize_entity_id(getattr(pokemon_state, "ability", None))
            if revealed:
                return revealed
            if not species_name:
                return ""
            try:
                feasible_species_sets = get_feasible_random_battle_sets(species_name)
            except Exception:
                feasible_species_sets = []
            best_ability = ""
            best_count = -1
            for set_entry in feasible_species_sets:
                ability = normalize_entity_id(str(set_entry.get("ability", "") or ""))
                if not ability:
                    continue
                try:
                    count = int(set_entry.get("count", 1) or 1)
                except Exception:
                    count = 1
                if count > best_count:
                    best_count = count
                    best_ability = ability
            return best_ability

        defender_ability = _likely_ability(snapshot.active_species, defender_state)

        weighted_damage = 0.0
        weighted_physical = 0.0
        weighted_special = 0.0
        weighted_flinch = 0.0
        top_move = ""
        top_move_score = -1.0

        for candidate_set in feasible_sets:
            if not candidate_set:
                continue
            try:
                weight = float(int(candidate_set.get("count", 1) or 1))
            except Exception:
                weight = 1.0
            if weight <= 0:
                continue

            best_for_set = 0.0
            best_move_id = ""
            best_category = "physical"
            best_flinch = 0.0

            for move_id in candidate_set.get("moves", []):
                norm_move_id = normalize_entity_id(str(move_id))
                if not norm_move_id:
                    continue
                move_data = self._move_data_by_id(snapshot, norm_move_id)
                category = str(move_data.get("category", "status")).lower()
                if category == "status":
                    continue

                try:
                    base_power = float(move_data.get("basePower", 0) or 0)
                except Exception:
                    base_power = 0.0
                if base_power <= 0.0:
                    continue

                accuracy_raw = move_data.get("accuracy", 100)
                try:
                    accuracy = 100.0 if isinstance(accuracy_raw, bool) else float(accuracy_raw or 100)
                except Exception:
                    accuracy = 100.0
                accuracy_factor = max(0.0, min(100.0, accuracy)) / 100.0

                move_type_name = str(move_data.get("type", "typeless")).lower()
                stab = 1.5 if move_type_name in attacker_types else 1.0
                type_multiplier = get_type_effectiveness_multiplier(move_type_name, defender_types)
                if defender_ability in ABILITY_IMMUNITIES and move_type_name in ABILITY_IMMUNITIES[defender_ability]:
                    type_multiplier = 0.0
                expected = base_power * accuracy_factor * stab * type_multiplier

                if defender_state is not None:
                    try:
                        move_flags = move_data.get("flags", {}) or {}
                        contact_flag = bool(move_flags.get("contact")) if isinstance(move_flags, dict) else False
                    except Exception:
                        contact_flag = False
                    if contact_flag:
                        defender_item = normalize_entity_id(getattr(defender_state, "item", None))
                        if defender_ability in {"ironbarbs", "roughskin"} or defender_item == "rockyhelmet":
                            expected = max(0.0, expected - (defender_max_hp / 6.0))

                if expected > best_for_set:
                    best_for_set = expected
                    best_move_id = norm_move_id
                    best_category = category
                    best_flinch = self._flinch_chance(move_data)

            weighted_damage += best_for_set * weight
            if best_category == "special":
                weighted_special += weight
            else:
                weighted_physical += weight
            weighted_flinch += best_flinch * weight
            if best_for_set > top_move_score:
                top_move_score = best_for_set
                top_move = best_move_id

        total_weight = max(1.0, weighted_physical + weighted_special)
        expected_damage = weighted_damage / total_weight
        best_category = "special" if weighted_special > weighted_physical else "physical"
        outspeeds = _effective_species_speed(snapshot, own=False) > _effective_species_speed(snapshot, own=True)
        return OpponentDamageSummary(
            expected_damage=expected_damage,
            best_move_id=top_move,
            best_move_category=best_category,
            best_move_flinch_chance=(weighted_flinch / total_weight),
            outspeeds=outspeeds,
        )

    def _flinch_chance(self, move_data: dict[str, Any]) -> float:
        secondary = move_data.get("secondary")
        if isinstance(secondary, dict):
            volatile = str(secondary.get("volatileStatus", "")).lower()
            if volatile == "flinch":
                try:
                    return float(secondary.get("chance", 0.0) or 0.0) / 100.0
                except Exception:
                    return 0.0
        secondaries = move_data.get("secondaries")
        if isinstance(secondaries, list):
            for sec in secondaries:
                if not isinstance(sec, dict):
                    continue
                volatile = str(sec.get("volatileStatus", "")).lower()
                if volatile == "flinch":
                    try:
                        return float(sec.get("chance", 0.0) or 0.0) / 100.0
                    except Exception:
                        return 0.0
        return 0.0

    def _estimate_switch_in_offense(self, snapshot: BattleSnapshot) -> float:
        feasible_sets = get_feasible_random_battle_sets(snapshot.active_species)
        if not feasible_sets:
            return self._best_own_attack_damage(snapshot)

        weighted = 0.0
        total_weight = 0.0
        for set_entry in feasible_sets:
            weight = float(int(set_entry.get("count", 1) or 1))
            if weight <= 0:
                continue
            best = 0.0
            for move_id in set_entry.get("moves", []):
                norm_move_id = normalize_entity_id(str(move_id))
                if not norm_move_id:
                    continue
                move_data = self._move_data_by_id(snapshot, norm_move_id)
                category = str(move_data.get("category", "status")).lower()
                if category == "status":
                    continue
                power = float(move_data.get("basePower", 0) or 0)
                accuracy_raw = move_data.get("accuracy", 100)
                accuracy = 100.0 if isinstance(accuracy_raw, bool) else float(accuracy_raw or 100)
                best = max(best, power * max(0.0, min(1.0, accuracy / 100.0)))
            weighted += best * weight
            total_weight += weight

        if total_weight <= 0:
            return self._best_own_attack_damage(snapshot)
        return weighted / total_weight

    def _can_ko_now(self, snapshot: BattleSnapshot, move_scores: list[tuple[ActionChoice, float]]) -> bool:
        target_hp = self._current_hp_value(snapshot, own=False)
        return any(score >= target_hp for _, score in move_scores)

    def _opponent_hp_fraction(self, snapshot: BattleSnapshot) -> float:
        battle_state = snapshot.battle_state
        if battle_state is not None and battle_state.active_opponent_pokemon is not None:
            fraction = turn_hp_fraction(
                battle_state.active_opponent_pokemon.current_hp,
                battle_state.active_opponent_pokemon.max_hp,
            )
            if fraction is not None:
                return fraction
        return 1.0

    def _own_hp_fraction(self, snapshot: BattleSnapshot) -> float:
        battle_state = snapshot.battle_state
        if battle_state is not None and battle_state.active_own_pokemon is not None:
            fraction = turn_hp_fraction(
                battle_state.active_own_pokemon.current_hp,
                battle_state.active_own_pokemon.max_hp,
            )
            if fraction is not None:
                return fraction
        if snapshot.own_active_hp_fraction is not None:
            return max(0.0, min(1.0, snapshot.own_active_hp_fraction))
        condition = _extract_active_condition(snapshot.last_request or {})
        parsed = _extract_hp_fraction_from_condition(condition)
        if parsed is not None:
            return parsed
        return 1.0

    def _estimate_own_max_hp(self, species: str | None, *, snapshot: BattleSnapshot | None = None) -> float:
        if snapshot is not None and snapshot.battle_state is not None:
            active_own = snapshot.battle_state.active_own_pokemon
            if active_own is not None and active_own.max_hp is not None and active_own.max_hp > 0:
                return float(active_own.max_hp)
        if not species:
            return 300.0
        try:
            return float(build_stat_profile(species, level=100, nature="serious").stats.get("hp", 300))
        except Exception:
            return 300.0

    def _estimate_opponent_max_hp(self, species: str | None, *, snapshot: BattleSnapshot | None = None) -> float:
        if snapshot is not None and snapshot.battle_state is not None:
            active_opp = snapshot.battle_state.active_opponent_pokemon
            if active_opp is not None and active_opp.max_hp is not None and active_opp.max_hp > 0:
                return float(active_opp.max_hp)
        if not species:
            return 300.0
        feasible_sets = get_feasible_random_battle_sets(species)
        if not feasible_sets:
            return self._estimate_own_max_hp(species, snapshot=snapshot)
        weighted = 0.0
        total = 0.0
        for set_entry in feasible_sets:
            level = int(set_entry.get("level", 100) or 100)
            nature = str(set_entry.get("nature", "serious"))
            weight = float(int(set_entry.get("count", 1) or 1))
            try:
                hp_stat = float(build_stat_profile(species, level=level, nature=nature).stats.get("hp", 300))
            except Exception:
                hp_stat = 300.0
            weighted += hp_stat * max(1.0, weight)
            total += max(1.0, weight)
        return weighted / max(1.0, total)

    def _maybe_choose_speed_setup(
        self,
        snapshot: BattleSnapshot,
        moves: list[ActionChoice],
        outsped: bool,
        rng: random.Random,
    ) -> ActionChoice | None:
        if not outsped:
            return None

        own_speed = _effective_species_speed(snapshot, own=True)
        opp_speed = _effective_species_speed(snapshot, own=False)
        if own_speed >= opp_speed:
            return None

        for move in moves:
            move_data = get_move_data_from_action(snapshot, move)
            if str(move_data.get("category", "")).lower() != "status":
                continue
            boosts = move_data.get("boosts")
            if not isinstance(boosts, dict):
                continue
            speed_boost = int(boosts.get("speed", 0) or 0)
            if speed_boost <= 0:
                continue
            if own_speed * _stage_multiplier(speed_boost) <= opp_speed:
                continue
            has_other_boost = any(
                stat != "speed" and int(value or 0) > 0
                for stat, value in boosts.items()
            )
            if has_other_boost or rng.random() < 0.5:
                return move
        return None

    def _maybe_choose_ratio_improving_setup(
        self,
        snapshot: BattleSnapshot,
        moves: list[ActionChoice],
        move_scores: list[tuple[ActionChoice, float]],
        opp_summary: OpponentDamageSummary,
        rng: random.Random,
    ) -> ActionChoice | None:
        if not move_scores:
            return None

        own_current_hp = self._current_hp_value(snapshot, own=True)
        opp_current_hp = self._current_hp_value(snapshot, own=False)

        def _current_own_stage(stat_name: str) -> int:
            active = snapshot.battle_state.active_own_pokemon if snapshot.battle_state is not None else None
            if active is not None:
                try:
                    return max(-6, min(6, int(active.stat_stage(stat_name))))
                except Exception:
                    pass
                try:
                    return max(-6, min(6, int(active.stat_boosts.get(stat_name, 0) or 0)))
                except Exception:
                    pass
            return 0

        def _stage_gain(current_stage: int, delta_stage: int) -> float:
            if delta_stage == 0:
                return 1.0
            before = _stage_multiplier(current_stage)
            after = _stage_multiplier(max(-6, min(6, current_stage + int(delta_stage))))
            if before <= 1e-9:
                return 1.0
            return max(1e-9, after / before)

        def _effective_delta(current_stage: int, delta_stage: int) -> int:
            current = max(-6, min(6, int(current_stage)))
            projected = max(-6, min(6, current + int(delta_stage)))
            return projected - current

        base_own_damage = max(score for _, score in move_scores)
        base_t1 = own_current_hp / max(1e-6, opp_summary.expected_damage)
        base_t2 = opp_current_hp / max(1e-6, base_own_damage)
        base_ratio = base_t1 / max(1e-6, base_t2)

        best_setup: ActionChoice | None = None
        best_ratio = base_ratio

        # Require surviving at least two exchanges before attempting a setup,
        # otherwise setup is unlikely to pay off.
        try:
            moves_before_ko = max(1, math.ceil(own_current_hp / max(1e-6, opp_summary.expected_damage)))
        except Exception:
            moves_before_ko = 1
        if moves_before_ko < 2:
            return None

        # Precompute available damaging move categories to ensure setup affects them.
        available_physical = False
        available_special = False
        for _m in moves:
            _md = get_move_data_from_action(snapshot, _m)
            _cat = str(_md.get("category", "status")).lower()
            if _cat == "physical" and float(_md.get("basePower", 0) or 0) > 0:
                available_physical = True
            if _cat == "special" and float(_md.get("basePower", 0) or 0) > 0:
                available_special = True

        for move in moves:
            move_data = get_move_data_from_action(snapshot, move)
            if str(move_data.get("category", "")).lower() != "status":
                continue
            boosts = move_data.get("boosts")
            if not isinstance(boosts, dict):
                continue

            atk_stage = int(boosts.get("attack", 0) or 0)
            spa_stage = int(boosts.get("special-attack", 0) or 0)
            def_stage = int(boosts.get("defense", 0) or 0)
            spd_stage = int(boosts.get("special-defense", 0) or 0)

            # Skip setups that do not affect any available damaging moves.
            if atk_stage > 0 and not available_physical:
                continue
            if spa_stage > 0 and not available_special:
                continue

            speed_stage = int(boosts.get("speed", 0) or 0)

            current_atk_stage = _current_own_stage("attack")
            current_spa_stage = _current_own_stage("special-attack")
            current_def_stage = _current_own_stage("defense")
            current_spd_stage = _current_own_stage("special-defense")
            current_spe_stage = _current_own_stage("speed")

            effective_atk_delta = _effective_delta(current_atk_stage, atk_stage)
            effective_spa_delta = _effective_delta(current_spa_stage, spa_stage)
            effective_def_delta = _effective_delta(current_def_stage, def_stage)
            effective_spd_delta = _effective_delta(current_spd_stage, spd_stage)
            effective_spe_delta = _effective_delta(current_spe_stage, speed_stage)

            offensive_multiplier = max(
                _stage_gain(current_atk_stage, effective_atk_delta),
                _stage_gain(current_spa_stage, effective_spa_delta),
            )
            if offensive_multiplier < 1.0:
                offensive_multiplier = 1.0

            defensive_multiplier = (
                _stage_gain(current_def_stage, effective_def_delta)
                if opp_summary.best_move_category == "physical"
                else _stage_gain(current_spd_stage, effective_spd_delta)
            )
            if defensive_multiplier < 1.0:
                defensive_multiplier = 1.0

            setup_turn_taken = opp_summary.expected_damage
            current_own_moves_first = not opp_summary.outspeeds
            if current_own_moves_first and defensive_multiplier > 1.0:
                setup_turn_taken = setup_turn_taken / defensive_multiplier

            post_setup_own_hp = max(0.0, own_current_hp - max(0.0, setup_turn_taken))
            if post_setup_own_hp <= 0.0:
                continue

            post_own_moves_first = current_own_moves_first
            if effective_spe_delta > 0:
                own_speed = _effective_species_speed(snapshot, own=True)
                opp_speed = _effective_species_speed(snapshot, own=False)
                trick_room_active = bool(
                    snapshot.battle_state is not None
                    and snapshot.battle_state.battlefield is not None
                    and snapshot.battle_state.battlefield.trick_room
                )
                speed_gain = _stage_gain(current_spe_stage, effective_spe_delta)
                boosted_speed = own_speed * speed_gain
                post_own_moves_first = boosted_speed <= opp_speed if trick_room_active else boosted_speed >= opp_speed

            post_damage_taken = opp_summary.expected_damage / defensive_multiplier
            post_damage_dealt = base_own_damage * offensive_multiplier

            post_t1 = post_setup_own_hp / max(1e-6, post_damage_taken)
            post_t2 = opp_current_hp / max(1e-6, post_damage_dealt)
            post_ratio = post_t1 / max(1e-6, post_t2)
            if post_ratio > best_ratio:
                best_ratio = post_ratio
                best_setup = move
            #print(base_t1, base_t2, base_ratio, post_t1, post_t2, post_ratio, best_ratio)

        return best_setup


    def _maybe_choose_heal(
        self,
        snapshot: BattleSnapshot,
        moves: list[ActionChoice],
        move_scores: list[tuple[ActionChoice, float]],
        opp_summary: OpponentDamageSummary,
        rng: random.Random,
    ) -> ActionChoice | None:
        healing_moves: list[ActionChoice] = []
        for move in moves:
            move_id = get_move_id_from_action(snapshot, move)
            if move_id == "wish":
                continue
            move_data = get_move_data_from_action(snapshot, move)
            if self._is_reliable_heal(move_data):
                healing_moves.append(move)

        if not healing_moves:
            return None

        own_hp_fraction = self._own_hp_fraction(snapshot)
        if own_hp_fraction <= 0:
            return None

        best_heal: ActionChoice | None = None
        best_delta_hp = float("-inf")
        for move in healing_moves:
            expected_delta_hp = self._expected_heal_delta_hp(snapshot, move)
            if expected_delta_hp > best_delta_hp:
                best_delta_hp = expected_delta_hp
                best_heal = move

        if best_heal is None:
            return None

        if best_delta_hp > 0.0:
            return best_heal
        return None

    def _expected_heal_delta_hp(self, snapshot: BattleSnapshot, heal_move: ActionChoice) -> float:
        own_max_hp = max(1.0, self._estimate_own_max_hp(snapshot.active_species, snapshot=snapshot))
        own_hp_fraction = self._own_hp_fraction(snapshot)
        own_current_hp = max(0.0, min(own_max_hp, own_max_hp * own_hp_fraction))
        if own_current_hp <= 0.0:
            return float("-inf")

        move_data = get_move_data_from_action(snapshot, heal_move)
        heal_fraction = self._heal_fraction_for_move(snapshot, heal_move, move_data)
        heal_hp = max(0.0, heal_fraction * own_max_hp)
        after_turn_hp = self._expected_self_residual_fraction(snapshot) * own_max_hp

        own_speed = _effective_species_speed(snapshot, own=True)
        trick_room_active = bool(
            snapshot.battle_state is not None
            and snapshot.battle_state.battlefield is not None
            and snapshot.battle_state.battlefield.trick_room
        )

        weighted_sum = 0.0
        total_weight = 0.0
        for candidate_set in self._feasible_opponent_sets(snapshot):
            if candidate_set is None:
                continue
            try:
                weight = max(0.0, float(int(candidate_set.get("count", 1) or 1)))
            except Exception:
                weight = 0.0
            if weight <= 0.0:
                continue

            opponent_summary = self._estimate_opponent_best_damage(snapshot, opponent_set_override=candidate_set)
            damage_hp = max(0.0, float(opponent_summary.expected_damage))
            opponent_speed = self._opponent_speed_for_set(snapshot, candidate_set)
            own_first = (own_speed <= opponent_speed) if trick_room_active else (own_speed >= opponent_speed)

            delta_hp = self._simulate_heal_turn_delta_hp(
                current_hp=own_current_hp,
                max_hp=own_max_hp,
                heal_hp=heal_hp,
                damage_hp=damage_hp,
                after_turn_hp=after_turn_hp,
                own_first=own_first,
            )
            weighted_sum += delta_hp * weight
            total_weight += weight

        if total_weight <= 0.0:
            return float("-inf")
        return weighted_sum / total_weight

    def _simulate_heal_turn_delta_hp(
        self,
        *,
        current_hp: float,
        max_hp: float,
        heal_hp: float,
        damage_hp: float,
        after_turn_hp: float,
        own_first: bool,
    ) -> float:
        starting_hp = max(0.0, min(max_hp, current_hp))
        hp = starting_hp

        if own_first:
            hp = min(max_hp, hp + max(0.0, heal_hp))
            hp -= max(0.0, damage_hp)
            if hp <= 0.0:
                return -starting_hp
            hp = min(max_hp, hp + after_turn_hp)
            if hp <= 0.0:
                return -starting_hp
            return hp - starting_hp

        hp -= max(0.0, damage_hp)
        if hp <= 0.0:
            # If we are KOed before healing, treat the lost HP as negative healing.
            return -starting_hp

        hp = min(max_hp, hp + max(0.0, heal_hp))
        hp = min(max_hp, hp + after_turn_hp)
        if hp <= 0.0:
            return -starting_hp
        return hp - starting_hp

    def _weather_id(self, snapshot: BattleSnapshot) -> str:
        if snapshot.battle_state is not None and snapshot.battle_state.battlefield is not None:
            return normalize_entity_id(str(snapshot.battle_state.battlefield.weather or ""))
        request = snapshot.last_request or {}
        return normalize_entity_id(str(request.get("weather", "")))

    def _heal_fraction_for_move(
        self,
        snapshot: BattleSnapshot,
        move: ActionChoice,
        move_data: dict[str, Any],
    ) -> float:
        move_id = get_move_id_from_action(snapshot, move)
        weather = self._weather_id(snapshot)

        if move_id in {"morningsun", "moonlight", "synthesis"}:
            if weather in {"sunnyday", "desolateland"}:
                return 2.0 / 3.0
            if weather in {"raindance", "primordialsea", "sandstorm", "hail"}:
                return 1.0 / 4.0
            return 1.0 / 2.0

        if move_id == "shoreup":
            return (2.0 / 3.0) if weather == "sandstorm" else (1.0 / 2.0)

        return self._heal_fraction(move_data)

    def _opponent_speed_for_set(self, snapshot: BattleSnapshot, candidate_set: dict[str, Any]) -> float:
        try:
            computed_stats = candidate_set.get("computed_stats", {})
            speed = float(computed_stats.get("speed", 0) or 0)
        except Exception:
            speed = 0.0

        if speed <= 0.0:
            try:
                level = int(candidate_set.get("level", 100) or 100)
                nature = str(candidate_set.get("nature", "serious") or "serious")
                speed = float(build_stat_profile(snapshot.opponent_active_species, level=level, nature=nature).stats.get("speed", 0))
            except Exception:
                speed = 0.0

        item = normalize_entity_id(str(candidate_set.get("item", "") or ""))
        if item == "choicescarf":
            speed *= 1.5

        ability = normalize_entity_id(str(candidate_set.get("ability", "") or ""))
        weather = self._weather_id(snapshot)
        if ability == "chlorophyll" and weather in {"sunnyday", "desolateland"}:
            speed *= 2.0
        if ability == "swiftswim" and weather in {"raindance", "primordialsea"}:
            speed *= 2.0

        opponent_status = _active_status(snapshot, own=False)
        if opponent_status == "par":
            speed *= 0.25

        speed *= _stage_multiplier(snapshot.opponent_constraints.speed_stage)
        return speed

    def _is_reliable_heal(self, move_data: dict[str, Any]) -> bool:
        if not move_data:
            return False
        if str(move_data.get("category", "")).lower() != "status":
            return False
        flags = move_data.get("flags", {})
        if isinstance(flags, dict) and flags.get("heal"):
            return True
        return "heal" in move_data

    def _heal_fraction(self, move_data: dict[str, Any]) -> float:
        heal = move_data.get("heal")
        if isinstance(heal, list) and len(heal) == 2:
            try:
                return max(0.0, float(heal[0]) / max(1.0, float(heal[1])))
            except Exception:
                return 0.0
        return 0.0

    def _expected_action_success_factor(self, snapshot: BattleSnapshot, opp_summary: OpponentDamageSummary) -> float:
        battle_state = snapshot.battle_state
        if battle_state is not None and battle_state.active_own_pokemon is not None:
            status = battle_state.active_own_pokemon.status
        else:
            condition = _extract_active_condition(snapshot.last_request or {})
            status = _extract_status_from_condition(condition)
        paralysis_factor = 0.75 if status == "par" else 1.0
        flinch_block = opp_summary.best_move_flinch_chance if opp_summary.outspeeds else 0.0
        return max(0.0, paralysis_factor * (1.0 - flinch_block))

    def _expected_self_residual_fraction(self, snapshot: BattleSnapshot) -> float:
        battle_state = snapshot.battle_state
        if battle_state is not None and battle_state.active_own_pokemon is not None:
            active_pokemon = battle_state.active_own_pokemon
            status = active_pokemon.status
            item = active_pokemon.item or ""
            weather = battle_state.battlefield.weather or ""
            species_record_types = active_pokemon.types
        else:
            request = snapshot.last_request or {}
            condition = _extract_active_condition(request)
            status = _extract_status_from_condition(condition)
            side = request.get("side", {})
            active_entry = next((pk for pk in side.get("pokemon", []) if pk.get("active")), {})
            item = normalize_entity_id(str(active_entry.get("item", "")))
            weather = normalize_entity_id(str(request.get("weather", "")))
            species_record_types = self._types_for_species(snapshot.active_species, snapshot=snapshot, own=True)

        delta = 0.0
        if status in {"psn", "tox", "brn"}:
            delta -= 0.125

        if item == "leftovers":
            delta += 1.0 / 16.0

        rock_or_steel_or_ground = {"rock", "steel", "ground"}
        ice_type = {"ice"}
        if weather == "sandstorm" and not (set(species_record_types) & rock_or_steel_or_ground):
            delta -= 1.0 / 16.0
        if weather == "hail" and not (set(species_record_types) & ice_type):
            delta -= 1.0 / 16.0

        return delta

    def _expected_opponent_passive_damage_fraction(
        self,
        snapshot: BattleSnapshot,
        opp_summary: OpponentDamageSummary,
    ) -> float:
        request = snapshot.last_request or {}
        side = request.get("side", {})
        active_entry = next((pk for pk in side.get("pokemon", []) if pk.get("active")), {})
        own_item = normalize_entity_id(str(active_entry.get("item", "")))
        own_ability = normalize_entity_id(str(active_entry.get("ability", "")))

        passive = 0.0
        opponent_item = normalize_entity_id(snapshot.opponent_constraints.revealed_item)
        if opponent_item == "lifeorb":
            passive += 0.1

        if opp_summary.best_move_id:
            move_data = self._move_data_by_id(snapshot, opp_summary.best_move_id)
            recoil = move_data.get("recoil")
            if isinstance(recoil, list) and len(recoil) == 2:
                try:
                    passive += max(0.0, float(recoil[0]) / max(1.0, float(recoil[1])))
                except Exception:
                    pass
            if (
                opp_summary.best_move_category == "physical"
                and (own_item == "rockyhelmet" or own_ability in {"roughskin", "ironbarbs"})
            ):
                passive += 1.0 / 6.0

        return passive

    def _opponent_has_offensive_boost(self, snapshot: BattleSnapshot) -> bool:
        request = snapshot.last_request or {}
        for key in ("opponentBoosts", "foeBoosts", "enemyBoosts"):
            boosts = request.get(key)
            if not isinstance(boosts, dict):
                continue
            if any(int(boosts.get(stat, 0) or 0) > 0 for stat in ("attack", "special-attack", "special-defense")):
                return True
        return False

    def _opponent_types(self, snapshot: BattleSnapshot) -> tuple[str, ...]:
        return self._types_for_species(snapshot.opponent_active_species, snapshot=snapshot, own=False)

    def _types_for_species(
        self,
        species: str | None,
        *,
        snapshot: BattleSnapshot | None = None,
        own: bool = True,
    ) -> tuple[str, ...]:
        if snapshot is not None and snapshot.battle_state is not None:
            active = snapshot.battle_state.active_own_pokemon if own else snapshot.battle_state.active_opponent_pokemon
            if active is not None and active.types:
                return tuple(str(value).lower() for value in active.types if value)
        if not species:
            return tuple()
        try:
            from engine.inference import get_species_types

            return tuple(get_species_types(species))
        except Exception:
            return tuple()
