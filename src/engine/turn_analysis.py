from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from core.battle_state import BattleFieldState, BattleState, PokemonState
from core.identifiers import extract_status_from_condition, hp_fraction_from_condition


@dataclass(slots=True)
class MoveRole:
    move_id: str
    category: str
    move_type: str
    base_power: int
    accuracy: float
    is_heal: bool = False
    is_status: bool = False
    is_hazard: bool = False
    is_screen: bool = False
    is_setup: bool = False


def stage_multiplier(stage: int) -> float:
    bounded = max(-6, min(6, int(stage)))
    if bounded >= 0:
        return (2.0 + bounded) / 2.0
    return 2.0 / (2.0 - bounded)


def hp_fraction(current_hp: int | None, max_hp: int | None) -> float | None:
    if current_hp is None or max_hp is None or max_hp <= 0:
        return None
    return max(0.0, min(1.0, float(current_hp) / float(max_hp)))


def pokemon_hp_fraction(pokemon: PokemonState | None) -> float | None:
    if pokemon is None:
        return None
    return pokemon.hp_fraction()


def pokemon_speed(pokemon: PokemonState | None) -> int | None:
    if pokemon is None:
        return None
    base_speed = pokemon.get_stat("speed")
    if base_speed is None:
        return None
    speed = float(base_speed)
    if pokemon.status == "par":
        speed *= 0.25
    speed *= stage_multiplier(pokemon.stat_stage("speed"))
    if pokemon.ability == "unburden" and pokemon.has_volatile_status("unburden"):
        speed *= 2.0
    return max(0, int(math.floor(speed)))


def move_role(move_data: dict[str, Any], move_id: str | None = None) -> MoveRole:
    normalized_move_id = (move_id or str(move_data.get("id", ""))).lower()
    category = str(move_data.get("category", "status")).lower()
    move_type = str(move_data.get("type", "typeless")).lower()
    base_power = int(move_data.get("basePower", 0) or 0)
    accuracy_raw = move_data.get("accuracy", 100)
    accuracy = 100.0 if isinstance(accuracy_raw, bool) else float(accuracy_raw or 100)
    flags = move_data.get("flags", {})
    if not isinstance(flags, dict):
        flags = {}
    side_conditions = str(move_data.get("side_conditions", "")).lower()
    boosts = move_data.get("boosts", {})
    if not isinstance(boosts, dict):
        boosts = {}

    return MoveRole(
        move_id=normalized_move_id,
        category=category,
        move_type=move_type,
        base_power=base_power,
        accuracy=accuracy,
        is_heal=bool(flags.get("heal") or "heal" in move_data),
        is_status=category == "status",
        is_hazard=side_conditions in {"spikes", "stealthrock", "stickyweb", "toxicspikes"},
        is_screen=side_conditions in {"reflect", "lightscreen"},
        is_setup=category == "status" and any(int(value or 0) != 0 for value in boosts.values()),
    )


def effective_speed(pokemon: PokemonState | None, *, field: BattleFieldState | None = None) -> float:
    if pokemon is None:
        return 0.0
    speed = pokemon_speed(pokemon)
    if speed is None:
        return 0.0
    effective = float(speed)
    if field is not None and field.weather == "sunnyday" and pokemon.ability == "chlorophyll":
        effective *= 2.0
    if field is not None and field.weather == "raindance" and pokemon.ability == "swiftswim":
        effective *= 2.0
    if pokemon.item == "choicescarf":
        effective *= 1.5
    if field is not None and field.trick_room:
        effective *= -1.0
    return effective


def can_ko_in_one_turn(attacker_damage: float, defender: PokemonState | None) -> bool:
    defender_hp = pokemon_hp_fraction(defender)
    if defender_hp is None:
        return False
    max_hp = defender.max_hp or 0
    return attacker_damage >= max_hp * defender_hp


def direct_damage_ratio(dealt: float, taken: float, *, own_moves_first: bool = True) -> float:
    own_two_turn = max(0.0, dealt) * 2.0
    opp_two_turn = max(0.0, taken) * 2.0
    if opp_two_turn <= 1e-6:
        return 1e6 if own_two_turn > 1e-6 else 1.0
    return own_two_turn / opp_two_turn


def parse_condition_snapshot(condition: str | None) -> dict[str, Any]:
    return {
        "hp_fraction": hp_fraction_from_condition(condition),
        "status": extract_status_from_condition(condition),
    }
