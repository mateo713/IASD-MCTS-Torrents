from __future__ import annotations

import json
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


_DEBUG_DIR = Path(__file__).resolve().parents[2] / "debug" / "no_move_inputs"


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        converted = [_jsonable(item) for item in value]
        return sorted(converted, key=lambda item: json.dumps(item, sort_keys=True, default=str))
    return str(value)


def _simple_attrs(obj: Any, attribute_names: list[str]) -> dict[str, Any]:
    return {attribute_name: _jsonable(getattr(obj, attribute_name, None)) for attribute_name in attribute_names}


def _summarize_move(move: Any) -> dict[str, Any]:
    return _simple_attrs(
        move,
        ["id", "disabled", "pp"],
    )


def _summarize_pokemon(pokemon: Any) -> dict[str, Any]:
    summary = _simple_attrs(
        pokemon,
        [
            "id",
            "level",
            "types",
            "base_types",
            "hp",
            "maxhp",
            "ability",
            "base_ability",
            "item",
            "nature",
            "evs",
            "attack",
            "defense",
            "special_attack",
            "special_defense",
            "speed",
            "status",
            "rest_turns",
            "sleep_turns",
            "weight_kg",
            "terastallized",
            "tera_type",
            "times_attacked",
            "stellar_boosted_types",
        ],
    )
    moves = getattr(pokemon, "moves", None)
    if isinstance(moves, list):
        summary["moves"] = [_summarize_move(move) for move in moves]
    return summary


def _summarize_side_conditions(side_conditions: Any) -> dict[str, Any]:
    return _simple_attrs(
        side_conditions,
        [
            "aurora_veil",
            "crafty_shield",
            "healing_wish",
            "light_screen",
            "lucky_chant",
            "lunar_dance",
            "mat_block",
            "mist",
            "protect",
            "quick_guard",
            "reflect",
            "safeguard",
            "spikes",
            "stealth_rock",
            "sticky_web",
            "tailwind",
            "toxic_count",
            "toxic_spikes",
            "wide_guard",
        ],
    )


def _summarize_volatile_durations(volatile_durations: Any) -> dict[str, Any]:
    return _simple_attrs(
        volatile_durations,
        ["confusion", "encore", "lockedmove", "slowstart", "taunt", "yawn"],
    )


def _summarize_side(side: Any) -> dict[str, Any]:
    summary = _simple_attrs(
        side,
        [
            "active_index",
            "baton_passing",
            "shed_tailing",
            "wish",
            "future_sight",
            "force_switch",
            "force_trapped",
            "slow_uturn_move",
            "volatile_statuses",
            "substitute_health",
            "attack_boost",
            "defense_boost",
            "special_attack_boost",
            "special_defense_boost",
            "speed_boost",
            "accuracy_boost",
            "evasion_boost",
            "last_used_move",
            "switch_out_move_second_saved_move",
        ],
    )
    summary["volatile_status_durations"] = _summarize_volatile_durations(getattr(side, "volatile_status_durations", None))
    summary["side_conditions"] = _summarize_side_conditions(getattr(side, "side_conditions", None))
    summary["pokemon"] = [_summarize_pokemon(pokemon) for pokemon in getattr(side, "pokemon", []) or []]
    return summary


def summarize_engine_state(state: Any | None) -> dict[str, Any] | None:
    if state is None:
        return None
    summary = _simple_attrs(
        state,
        [
            "weather",
            "weather_turns_remaining",
            "terrain",
            "terrain_turns_remaining",
            "trick_room",
            "trick_room_turns_remaining",
            "team_preview",
        ],
    )
    summary["state_string"] = state.to_string() if hasattr(state, "to_string") else None
    summary["side_one"] = _summarize_side(getattr(state, "side_one", None))
    summary["side_two"] = _summarize_side(getattr(state, "side_two", None))
    return summary


def _summarize_result(result: Any | None) -> Any:
    if result is None:
        return None
    if hasattr(result, "matrix") and hasattr(result, "depth_searched"):
        return {
            "kind": "iterative_deepening_expectiminimax",
            "side_one": _jsonable(getattr(result, "side_one", None)),
            "side_two": _jsonable(getattr(result, "side_two", None)),
            "matrix": _jsonable(getattr(result, "matrix", None)),
            "depth_searched": _jsonable(getattr(result, "depth_searched", None)),
        }
    if hasattr(result, "side_one") and hasattr(result, "side_two"):
        return {
            "kind": "mcts",
            "side_one": _jsonable(getattr(result, "side_one", None)),
            "side_two": _jsonable(getattr(result, "side_two", None)),
            "iteration_count": _jsonable(getattr(result, "iteration_count", None)),
        }
    return _jsonable(result)


def dump_no_move_context(
    *,
    strategy_name: str,
    snapshot: Any,
    engine_state: Any | None,
    selected_move: str,
    candidate_set: dict[str, Any] | None = None,
    result: Any | None = None,
) -> Path:
    _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    room_id = str(getattr(snapshot, "room_id", "unknown")).replace("/", "_").replace("\\", "_")
    turn = getattr(snapshot, "turn", "unknown")
    file_path = _DEBUG_DIR / f"{timestamp}_{strategy_name}_room-{room_id}_turn-{turn}.json"

    payload = {
        "strategy": strategy_name,
        "selected_move": selected_move,
        "snapshot": _jsonable(snapshot),
        "candidate_set": _jsonable(candidate_set),
        "engine_state": summarize_engine_state(engine_state),
        "search_result": _summarize_result(result),
    }

    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")

    return file_path


def dump_strategy_input_context(
    *,
    strategy_name: str,
    snapshot: Any,
    engine_state: Any | None,
    candidate_set: dict[str, Any] | None = None,
    result: Any | None = None,
    label: str = "request",
) -> Path:
    _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    room_id = str(getattr(snapshot, "room_id", "unknown")).replace("/", "_").replace("\\", "_")
    turn = getattr(snapshot, "turn", "unknown")
    file_path = _DEBUG_DIR / f"{timestamp}_{strategy_name}_{label}_room-{room_id}_turn-{turn}.json"

    payload = {
        "strategy": strategy_name,
        "label": label,
        "snapshot": _jsonable(snapshot),
        "candidate_set": _jsonable(candidate_set),
        "engine_state": summarize_engine_state(engine_state),
        "search_result": _summarize_result(result),
    }

    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")

    return file_path