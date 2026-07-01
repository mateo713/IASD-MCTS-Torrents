from __future__ import annotations

import json
import re
import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

from core.models import ActionChoice, BattleSnapshot
from engine.gen5_datasets import get_feasible_random_battle_sets


ABILITY_IMMUNITIES: dict[str, set[str]] = {
    "stormdrain": {"water"},
    "waterabsorb": {"water"},
    "dryskin": {"water"},
    "flashfire": {"fire"},
    "voltabsorb": {"electric"},
    "lightningrod": {"electric"},
    "motordrive": {"electric"},
    "levitate": {"ground"},
    "sapsipper": {"grass"},
}


@lru_cache(maxsize=1)
def _load_moves_data() -> dict[str, dict[str, Any]]:
    data_path = Path(__file__).resolve().parents[3] / "foul-play" / "data" / "moves.json"
    try:
        with data_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    if isinstance(data, dict):
        return data
    return {}


@lru_cache(maxsize=1)
def _load_pokedex_data() -> dict[str, dict[str, Any]]:
    data_path = Path(__file__).resolve().parents[3] / "foul-play" / "data" / "pokedex.json"
    try:
        with data_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    if isinstance(data, dict):
        return data
    return {}


def normalize_entity_id(name: str | None) -> str:
    if not name:
        return ""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def split_action_space(snapshot: BattleSnapshot) -> tuple[list[ActionChoice], list[ActionChoice], list[ActionChoice]]:
    from core.models import ActionType

    moves: list[ActionChoice] = []
    switches: list[ActionChoice] = []
    others: list[ActionChoice] = []

    for action in snapshot.available_actions:
        if action.action_type == ActionType.MOVE:
            moves.append(action)
        elif action.action_type == ActionType.SWITCH:
            switches.append(action)
        else:
            others.append(action)

    return moves, switches, others


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


def _candidate_set_signature(candidate_set: dict[str, Any]) -> str:
    if not candidate_set:
        return "{}"
    payload = {
        "ability": str(candidate_set.get("ability", "")),
        "item": str(candidate_set.get("item", "")),
        "level": int(candidate_set.get("level", 0) or 0),
        "nature": str(candidate_set.get("nature", "")),
        "moves": tuple(sorted(str(move_id) for move_id in candidate_set.get("moves", []) if move_id)),
    }
    return json.dumps(payload, sort_keys=True)


def _candidate_set_weight(candidate_set: dict[str, Any], set_weights: dict[str, float] | None) -> float:
    key = _candidate_set_signature(candidate_set)
    if set_weights and key in set_weights:
        try:
            return max(0.0, float(set_weights[key]))
        except Exception:
            return 0.0
    try:
        return max(0.0, float(int(candidate_set.get("count", 1) or 1)))
    except Exception:
        return 0.0


def _deterministic_seed(parts: list[str]) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def estimate_expected_damage(
    snapshot: BattleSnapshot,
    move_action: ActionChoice,
    set_weights: dict[str, float] | None = None,
) -> float:
    move_data = get_move_data_from_action(snapshot, move_action)
    move_type = normalize_entity_id(str(move_data.get("type", "typeless")))
    move_type_name = str(move_data.get("type", "typeless")).lower()
    opponent_species = _active_species(snapshot, own=False)
    opponent_types = get_species_types(opponent_species)

    if move_data:
        category = str(move_data.get("category", "status")).lower()
        if category != "status":
            # Prefer explicit ability observed in the live battle state when available
            try:
                if snapshot.battle_state is not None and snapshot.battle_state.active_opponent_pokemon is not None:
                    raw_ability = getattr(snapshot.battle_state.active_opponent_pokemon, "ability", None) or getattr(snapshot.battle_state.active_opponent_pokemon, "base_ability", None)
                else:
                    raw_ability = snapshot.opponent_constraints.revealed_ability
            except Exception:
                raw_ability = snapshot.opponent_constraints.revealed_ability
            revealed_ability = normalize_entity_id(raw_ability)
            if (
                move_type in getattr(snapshot.opponent_constraints, "observed_immune_move_types", set())
                or (revealed_ability in ABILITY_IMMUNITIES and move_type in ABILITY_IMMUNITIES[revealed_ability])
                or get_type_effectiveness_multiplier(move_type_name, opponent_types) == 0.0
            ):
                return 0.0

    feasible_sets = get_feasible_random_battle_sets(
        opponent_species,
        revealed_moves=sorted(snapshot.opponent_constraints.revealed_moves),
        revealed_ability=snapshot.opponent_constraints.revealed_ability,
        revealed_item=snapshot.opponent_constraints.revealed_item,
        impossible_abilities=snapshot.opponent_constraints.impossible_abilities,
        impossible_items=snapshot.opponent_constraints.impossible_items,
    )

    # Prefer to use poke-engine's calculate_damage when available. Construct a
    # minimal poke-engine State from the battle snapshot and ask the engine for
    # damage rolls. If poke-engine is not available or conversion fails, fall
    # back to the lightweight heuristic.
    try:
        from engine.bridge import PokeEngineBridge, State as _State  # type: ignore
        from poke_engine import calculate_damage

        move_id = get_move_id_from_action(snapshot, move_action)

        if _State is not None:
            # Deterministic expectation over feasible opponent hypotheses.
            # We do not sample damage rolls because poke-engine calculate_damage
            # returns deterministic extrema vectors, not stochastic roll draws.
            bridge = PokeEngineBridge()

            weighted_sum = 0.0
            total_weight = 0.0
            hypotheses = feasible_sets if feasible_sets else [{}]
            for candidate_set in hypotheses:
                candidate = candidate_set if candidate_set else {}
                weight = _candidate_set_weight(candidate, set_weights)
                if weight <= 0:
                    continue

                seed_parts = [
                    str(snapshot.room_id),
                    str(snapshot.turn),
                    _candidate_set_signature(candidate),
                ]
                local_seed = _deterministic_seed(seed_parts)
                try:
                    import random as _random

                    state = bridge.snapshot_to_state(
                        snapshot,
                        rng=_random.Random(local_seed),
                        opponent_set_override=(candidate if candidate else None),
                    )
                except Exception:
                    state = None
                if state is None:
                    continue

                opp_move = "tackle"
                try:
                    s1_damage_first, _ = calculate_damage(state, move_id, opp_move, True)
                    s1_damage_second, _ = calculate_damage(state, move_id, opp_move, False)
                except Exception:
                    continue

                non_crit_first = float(s1_damage_first[0]) if s1_damage_first else 0.0
                non_crit_second = float(s1_damage_second[0]) if s1_damage_second else 0.0
                expected_for_set = (non_crit_first + non_crit_second) / 2.0
                weighted_sum += expected_for_set * weight
                total_weight += weight

            if total_weight > 0:
                return weighted_sum / total_weight
    except Exception:
        # Import or bridge conversion failed; fall back to heuristic
        pass

    # --- Fallback heuristic (previous implementation) ---
    if not move_data:
        return 0.0

    category = str(move_data.get("category", "status")).lower()
    if category == "status":
        return 0.0

    base_power = float(move_data.get("basePower", 0) or 0)
    if base_power <= 0:
        return 0.0

    accuracy_raw = move_data.get("accuracy", 100)
    accuracy = 100.0 if isinstance(accuracy_raw, bool) else float(accuracy_raw or 100)
    accuracy_factor = max(0.0, min(100.0, accuracy)) / 100.0
    own_types = get_species_types(snapshot.active_species)
    stab = 1.5 if move_type_name in own_types else 1.0
    type_multiplier = get_type_effectiveness_multiplier(move_type_name, opponent_types)

    # Basic expected-damage heuristic: base power * accuracy * STAB * type multiplier
    expected = base_power * accuracy_factor * stab * type_multiplier

    # Penalize contact moves when opponent has revealed passive contact abilities/items
    # (e.g. Iron Barbs / Rough Skin / Rocky Helmet). Use a conservative fixed
    # penalty approximating the expected self-damage (fraction of our max HP).
    try:
        move_flags = move_data.get("flags", {}) or {}
        contact_flag = bool(move_flags.get("contact")) if isinstance(move_flags, dict) else False
    except Exception:
        contact_flag = False

    contact_penalty_hp = 0.0
    if contact_flag:
        try:
            if snapshot.battle_state is not None and snapshot.battle_state.active_opponent_pokemon is not None:
                opp_ability = normalize_entity_id(getattr(snapshot.battle_state.active_opponent_pokemon, "ability", None) or getattr(snapshot.battle_state.active_opponent_pokemon, "base_ability", None))
                opp_item = normalize_entity_id(getattr(snapshot.battle_state.active_opponent_pokemon, "item", None))
            else:
                opp_ability = normalize_entity_id(snapshot.opponent_constraints.revealed_ability)
                opp_item = normalize_entity_id(snapshot.opponent_constraints.revealed_item)
        except Exception:
            opp_ability = ""
            opp_item = ""

        if opp_ability in {"ironbarbs", "roughskin"} or opp_item == "rockyhelmet":
            own_max_hp = None
            if snapshot.battle_state is not None and snapshot.battle_state.active_own_pokemon is not None:
                try:
                    own_max_hp = float(snapshot.battle_state.active_own_pokemon.max_hp or 0) or None
                except Exception:
                    own_max_hp = None
            if not own_max_hp:
                own_max_hp = 300.0
            # Use 1/6 as a conservative expected passive-damage fraction
            contact_penalty_hp = float(own_max_hp) * (1.0 / 6.0)

    adjusted = expected - contact_penalty_hp
    return max(0.0, adjusted)


def get_move_data_from_action(snapshot: BattleSnapshot, move_action: ActionChoice) -> dict[str, Any]:
    move_id = get_move_id_from_action(snapshot, move_action)
    if not move_id:
        return {}
    return _load_moves_data().get(move_id, {})


def get_move_id_from_action(snapshot: BattleSnapshot, move_action: ActionChoice) -> str:
    request = snapshot.last_request or {}
    active_list = request.get("active", [])
    if not active_list:
        return normalize_entity_id(move_action.label)

    command_parts = move_action.command.split()
    move_index = -1
    if len(command_parts) >= 3 and command_parts[0] == "/choose" and command_parts[1] == "move":
        try:
            move_index = int(command_parts[2]) - 1
        except ValueError:
            move_index = -1

    moves = active_list[0].get("moves", [])
    if move_index < 0 or move_index >= len(moves):
        return normalize_entity_id(move_action.label)

    chosen = moves[move_index]
    move_id = normalize_entity_id(str(chosen.get("id", "")))
    if move_id:
        return move_id
    return normalize_entity_id(str(chosen.get("move", move_action.label)))


def get_species_types(species_name: str | None) -> tuple[str, ...]:
    species_id = normalize_entity_id(species_name)
    if not species_id:
        return ("typeless",)

    entry = _load_pokedex_data().get(species_id, {})
    raw_types = entry.get("types", [])
    if not isinstance(raw_types, list) or not raw_types:
        return ("typeless",)

    types = [str(type_name).lower() for type_name in raw_types]
    if len(types) == 1:
        return (types[0],)
    return tuple(types[:2])


_TYPE_CHART: dict[str, dict[str, float]] = {
    "normal": {"rock": 0.5, "ghost": 0.0, "steel": 0.5},
    "fire": {
        "fire": 0.5,
        "water": 0.5,
        "grass": 2.0,
        "ice": 2.0,
        "bug": 2.0,
        "rock": 0.5,
        "dragon": 0.5,
        "steel": 2.0,
    },
    "water": {
        "fire": 2.0,
        "water": 0.5,
        "grass": 0.5,
        "ground": 2.0,
        "rock": 2.0,
        "dragon": 0.5,
    },
    "electric": {
        "water": 2.0,
        "electric": 0.5,
        "grass": 0.5,
        "ground": 0.0,
        "flying": 2.0,
        "dragon": 0.5,
    },
    "grass": {
        "fire": 0.5,
        "water": 2.0,
        "grass": 0.5,
        "poison": 0.5,
        "ground": 2.0,
        "flying": 0.5,
        "bug": 0.5,
        "rock": 2.0,
        "dragon": 0.5,
        "steel": 0.5,
    },
    "ice": {
        "fire": 0.5,
        "water": 0.5,
        "grass": 2.0,
        "ground": 2.0,
        "flying": 2.0,
        "dragon": 2.0,
        "steel": 0.5,
    },
    "fighting": {
        "normal": 2.0,
        "ice": 2.0,
        "poison": 0.5,
        "flying": 0.5,
        "psychic": 0.5,
        "bug": 0.5,
        "rock": 2.0,
        "ghost": 0.0,
        "dark": 2.0,
        "steel": 2.0,
        "fairy": 0.5,
    },
    "poison": {
        "grass": 2.0,
        "poison": 0.5,
        "ground": 0.5,
        "rock": 0.5,
        "ghost": 0.5,
        "steel": 0.0,
        "fairy": 2.0,
    },
    "ground": {
        "fire": 2.0,
        "electric": 2.0,
        "grass": 0.5,
        "poison": 2.0,
        "flying": 0.0,
        "bug": 0.5,
        "rock": 2.0,
        "steel": 2.0,
    },
    "flying": {
        "electric": 0.5,
        "grass": 2.0,
        "fighting": 2.0,
        "bug": 2.0,
        "rock": 0.5,
        "steel": 0.5,
    },
    "psychic": {
        "fighting": 2.0,
        "poison": 2.0,
        "psychic": 0.5,
        "dark": 0.0,
        "steel": 0.5,
    },
    "bug": {
        "fire": 0.5,
        "grass": 2.0,
        "fighting": 0.5,
        "poison": 0.5,
        "flying": 0.5,
        "psychic": 2.0,
        "ghost": 0.5,
        "dark": 2.0,
        "steel": 0.5,
        "fairy": 0.5,
    },
    "rock": {
        "fire": 2.0,
        "ice": 2.0,
        "fighting": 0.5,
        "ground": 0.5,
        "flying": 2.0,
        "bug": 2.0,
        "steel": 0.5,
    },
    "ghost": {"normal": 0.0, "psychic": 2.0, "ghost": 2.0, "dark": 0.5},
    "dragon": {"dragon": 2.0, "steel": 0.5, "fairy": 0.0},
    "dark": {"fighting": 0.5, "psychic": 2.0, "ghost": 2.0, "dark": 0.5, "fairy": 0.5},
    "steel": {
        "fire": 0.5,
        "water": 0.5,
        "electric": 0.5,
        "ice": 2.0,
        "rock": 2.0,
        "steel": 0.5,
        "fairy": 2.0,
    },
    "fairy": {"fire": 0.5, "fighting": 2.0, "poison": 0.5, "dragon": 2.0, "dark": 2.0, "steel": 0.5},
}


def get_type_effectiveness_multiplier(move_type: str, defender_types: tuple[str, ...]) -> float:
    if not defender_types:
        return 1.0
    normalized_move_type = move_type.lower()
    if normalized_move_type not in _TYPE_CHART:
        return 1.0

    multiplier = 1.0
    matchups = _TYPE_CHART[normalized_move_type]
    for defender_type in defender_types:
        normalized_defender_type = defender_type.lower()
        multiplier *= matchups.get(normalized_defender_type, 1.0)
    return multiplier
