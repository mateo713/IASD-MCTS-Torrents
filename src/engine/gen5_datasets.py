from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    from poke_engine import Move, Pokemon, Side, SideConditions, State, Terrain, VolatileStatusDurations, Weather
except Exception:  # pragma: no cover - depends on local environment
    Move = None
    Pokemon = None
    Side = None
    SideConditions = None
    State = None
    Terrain = None
    VolatileStatusDurations = None
    Weather = None

from core.models import ActionType, BattlePhase, BattleSnapshot
from core.identifiers import extract_status_from_condition

STAT_NAMES = ("hp", "attack", "defense", "special-attack", "special-defense", "speed")
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "gen5"
CHOICE_ITEMS = {"choiceband", "choicespecs", "choicescarf"}


NATURE_EFFECTS: dict[str, tuple[str | None, str | None]] = {
    "lonely": ("attack", "defense"),
    "adamant": ("attack", "special-attack"),
    "naughty": ("attack", "special-defense"),
    "brave": ("attack", "speed"),
    "bold": ("defense", "attack"),
    "impish": ("defense", "special-attack"),
    "lax": ("defense", "special-defense"),
    "relaxed": ("defense", "speed"),
    "modest": ("special-attack", "attack"),
    "mild": ("special-attack", "defense"),
    "rash": ("special-attack", "special-defense"),
    "quiet": ("special-attack", "speed"),
    "calm": ("special-defense", "attack"),
    "gentle": ("special-defense", "defense"),
    "careful": ("special-defense", "special-attack"),
    "sassy": ("special-defense", "speed"),
    "timid": ("speed", "attack"),
    "hasty": ("speed", "defense"),
    "jolly": ("speed", "special-attack"),
    "naive": ("speed", "special-defense"),
    "hardy": (None, None),
    "docile": (None, None),
    "serious": (None, None),
    "bashful": (None, None),
    "quirky": (None, None),
}


@dataclass(frozen=True, slots=True)
class BuiltPokemonStats:
    level: int
    nature: str
    ivs: tuple[int, int, int, int, int, int]
    evs: tuple[int, int, int, int, int, int]
    stats: dict[str, int]
    has_physical_attacks: bool


def normalize_name(name: str | None) -> str:
    if not name:
        return ""
    return (
        name.replace(" ", "")
        .replace("-", "")
        .replace(".", "")
        .replace("'", "")
        .replace("%", "")
        .replace("*", "")
        .replace(":", "")
        .replace("(", "")
        .replace(")", "")
        .strip()
        .lower()
        .encode("ascii", "ignore")
        .decode("utf-8")
    )


def _move_pps_from_state(pokemon_state: Any | None) -> dict[str, int | None]:
    if pokemon_state is None:
        return {}
    moves = getattr(pokemon_state, "moves", None)
    if not isinstance(moves, dict):
        return {}
    move_pps: dict[str, int | None] = {}
    for move_id, move in moves.items():
        normalized_move_id = normalize_name(str(move_id))
        if not normalized_move_id:
            continue
        pp_value = getattr(move, "pp", None)
        try:
            move_pps[normalized_move_id] = int(pp_value) if pp_value is not None else None
        except Exception:
            move_pps[normalized_move_id] = None
    return move_pps


def _move_pps_from_entry(entry: dict[str, Any]) -> dict[str, int | None]:
    moves = entry.get("moves", [])
    if not isinstance(moves, list):
        return {}
    move_pps: dict[str, int | None] = {}
    for move_value in moves:
        if not isinstance(move_value, dict):
            continue
        move_id = move_value.get("id") or move_value.get("move")
        normalized_move_id = normalize_name(str(move_id or ""))
        if not normalized_move_id:
            continue
        pp_value = move_value.get("pp")
        try:
            move_pps[normalized_move_id] = int(pp_value) if pp_value is not None else None
        except Exception:
            move_pps[normalized_move_id] = None
    return move_pps


@lru_cache(maxsize=1)
def _load_json(path: str) -> dict[str, Any]:
    data_path = Path(path)
    if not data_path.exists():
        return {}
    with data_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


@lru_cache(maxsize=1)
def load_species_stats(data_dir: str | None = None) -> dict[str, dict[str, Any]]:
    root = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    return _load_json(str(root / "pokedex_stats.json"))


@lru_cache(maxsize=1)
def load_species_abilities(data_dir: str | None = None) -> dict[str, dict[str, Any]]:
    root = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    return _load_json(str(root / "pokedex_abilities.json"))


@lru_cache(maxsize=1)
def load_random_battle_sets(data_dir: str | None = None) -> dict[str, list[dict[str, Any]]]:
    root = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    return _load_json(str(root / "random_battle_sets.json"))


def resolve_species_key(species_name: str | None) -> str:
    return normalize_name(species_name)


def get_species_record(species_name: str | None, data_dir: str | None = None) -> dict[str, Any]:
    key = resolve_species_key(species_name)
    stats = load_species_stats(data_dir)
    return stats.get(key, {})


def get_species_ability_options(species_name: str | None, data_dir: str | None = None) -> list[str]:
    key = resolve_species_key(species_name)
    abilities = load_species_abilities(data_dir)
    entry = abilities.get(key, {})
    ability_list = entry.get("abilities", [])
    return [str(ability) for ability in ability_list if str(ability).strip()]


def species_has_physical_attacks(species_name: str | None, data_dir: str | None = None) -> bool:
    key = resolve_species_key(species_name)
    species_stats = load_species_stats(data_dir)
    entry = species_stats.get(key, {})
    value = entry.get("has_physical_attacks")
    return bool(value)


def _apply_nature(stat_name: str, value: int, nature: str) -> int:
    boosted, reduced = NATURE_EFFECTS.get(nature.lower(), (None, None))
    if boosted == stat_name:
        return int(value * 1.1)
    if reduced == stat_name:
        return int(value * 0.9)
    return value


def _calc_standard_stat(base: int, iv: int, ev: int, level: int) -> int:
    return ((2 * base + iv + (ev // 4)) * level) // 100


def compute_stats_from_base_stats(
    base_stats: dict[str, int],
    level: int,
    nature: str,
    evs: tuple[int, int, int, int, int, int],
    ivs: tuple[int, int, int, int, int, int],
    species_name: str | None = None,
) -> dict[str, int]:
    species_key = resolve_species_key(species_name)
    is_shedinja = species_key in {"shedinja", "munja"}
    stats: dict[str, int] = {}

    if is_shedinja:
        stats["hp"] = 1
    else:
        stats["hp"] = _calc_standard_stat(base_stats["hp"], ivs[0], evs[0], level) + level + 10

    for index, stat_name in enumerate(STAT_NAMES[1:], start=1):
        base_value = int(base_stats[stat_name])
        stat_value = _calc_standard_stat(base_value, ivs[index], evs[index], level) + 5
        stats[stat_name] = _apply_nature(stat_name, stat_value, nature)

    return {name: int(value) for name, value in stats.items()}


def build_stat_profile(species_name: str | None, level: int, nature: str) -> BuiltPokemonStats:
    has_physical_attacks = species_has_physical_attacks(species_name)
    species_key = resolve_species_key(species_name)
    is_shedinja = species_key in {"shedinja", "munja"}

    if is_shedinja:
        evs = (0, 100, 100, 100, 100, 100)
        ivs = (31, 31, 31, 31, 31, 31)
    elif has_physical_attacks:
        evs = (84, 84, 84, 84, 84, 84)
        ivs = (31, 31, 31, 31, 31, 31)
    else:
        evs = (100, 0, 100, 100, 100, 100)
        ivs = (31, 0, 31, 31, 31, 31)

    base_stats = get_species_record(species_name).get("base_stats", {})
    stats = compute_stats_from_base_stats(base_stats, level, nature, evs, ivs, species_name=species_name)
    return BuiltPokemonStats(
        level=level,
        nature=nature,
        ivs=ivs,
        evs=evs,
        stats=stats,
        has_physical_attacks=has_physical_attacks,
    )

def _stat_profile_record(profile: BuiltPokemonStats) -> dict[str, Any]:
    return {
        "level": profile.level,
        "nature": profile.nature,
        "ivs": profile.ivs,
        "evs": profile.evs,
        "stats": dict(profile.stats),
        "has_physical_attacks": profile.has_physical_attacks,
    }

def _enrich_candidate_set_with_stats(species_name: str | None, candidate_set: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(candidate_set)
    computed_stats = enriched.get("computed_stats")
    stat_profile = enriched.get("stat_profile")
    if not isinstance(computed_stats, dict) or not isinstance(stat_profile, dict):
        level = int(enriched.get("level", 100) or 100)
        nature = normalize_name(str(enriched.get("nature", "serious")))
        profile = build_stat_profile(species_name, level, nature)
        computed_stats = dict(profile.stats)
        stat_profile = _stat_profile_record(profile)
    enriched["computed_stats"] = dict(computed_stats)
    enriched["stat_profile"] = dict(stat_profile)
    return enriched


def _engine_volatile_statuses_from_state(pokemon_state: Any | None) -> set[str] | None:
    if pokemon_state is None:
        return None
    volatile_statuses = getattr(pokemon_state, "volatile_statuses", None)
    if not isinstance(volatile_statuses, dict):
        return None
    normalized = {
        normalize_name(str(status_name))
        for status_name in volatile_statuses.keys()
        if status_name and normalize_name(str(status_name)) and status_name != "immune_to"
    }
    return normalized or None


def _engine_volatile_statuses_to_serialized_string(pokemon_state: Any | None) -> str:
    volatile_statuses = _engine_volatile_statuses_from_state(pokemon_state)
    if not volatile_statuses:
        return ""
    return ":".join(sorted(status.upper() for status in volatile_statuses))


def _engine_weather_from_state(battle_state: Any | None) -> Any:
    if battle_state is None:
        return Weather.NONE if Weather is not None else None
    weather_name = normalize_name(str(getattr(getattr(battle_state, "battlefield", None), "weather", "") or ""))
    if not weather_name:
        return Weather.NONE if Weather is not None else None
    if Weather is None:
        return weather_name
    weather_map = {
        "sunnyday": Weather.SUN,
        "raindance": Weather.RAIN,
        "sandstorm": Weather.SAND,
        "hail": Weather.HAIL,
    }
    return weather_map.get(weather_name, Weather.NONE)


def _engine_boost_from_state(pokemon_state: Any | None, stat_name: str) -> int:
    if pokemon_state is None:
        return 0
    boosts = getattr(pokemon_state, "stat_boosts", None)
    if not isinstance(boosts, dict):
        return 0
    value = boosts.get(stat_name)
    try:
        return int(value or 0)
    except Exception:
        return 0


def _force_trapped_from_state(pokemon_state: Any | None) -> bool:
    if pokemon_state is None:
        return False
    return bool(getattr(pokemon_state, "trapped", False) or getattr(pokemon_state, "maybe_trapped", False))


def _force_trapped_from_request(request_json: dict[str, Any]) -> bool:
    active_list = request_json.get("active", [])
    if not isinstance(active_list, list) or not active_list:
        return False
    active_entry = active_list[0]
    if not isinstance(active_entry, dict):
        return False
    return bool(active_entry.get("trapped") or active_entry.get("maybeTrapped"))


def _last_used_move_from_state(pokemon_state: Any | None) -> str | None:
    if pokemon_state is None:
        return None
    move_history = getattr(pokemon_state, "move_history", None)
    if not isinstance(move_history, list) or not move_history:
        return None
    for entry in reversed(move_history):
        if not isinstance(entry, dict):
            continue
        move_id = entry.get("move_id")
        if move_id:
            move_id = normalize_name(str(move_id))
            if move_id:
                return move_id
    return None


def _side_conditions_from_state(battle_state: Any | None, own_side: bool) -> Any:
    if SideConditions is None:
        return None

    side_conditions: dict[str, Any] = {}
    if battle_state is not None:
        battlefield = getattr(battle_state, "battlefield", None)
        if battlefield is not None:
            side_key = "own" if own_side else "opponent"
            candidate = getattr(battlefield, "side_conditions", {}).get(side_key)
            if isinstance(candidate, dict):
                side_conditions = candidate

    def _layers(condition_id: str, *, default: int = 0) -> int:
        value = side_conditions.get(condition_id)
        if isinstance(value, dict):
            try:
                if not value.get("active", True):
                    return 0
                return int(value.get("layers", 1) or 1)
            except Exception:
                return default
        if isinstance(value, (int, float)):
            return int(value)
        if value:
            return 1
        return default

    return SideConditions(
        spikes=_layers("spikes"),
        toxic_spikes=_layers("toxicspikes"),
        stealth_rock=_layers("stealthrock"),
        sticky_web=_layers("stickyweb"),
        tailwind=_layers("tailwind"),
        lucky_chant=_layers("luckychant"),
        lunar_dance=_layers("lunardance"),
        reflect=_layers("reflect"),
        light_screen=_layers("lightscreen"),
        aurora_veil=_layers("auroraveil"),
        crafty_shield=_layers("craftyshield"),
        safeguard=_layers("safeguard"),
        mist=_layers("mist"),
        protect=_layers("protect"),
        healing_wish=_layers("healingwish"),
        mat_block=_layers("matblock"),
        quick_guard=_layers("quickguard"),
        toxic_count=_layers("toxiccount"),
        wide_guard=_layers("wideguard"),
    )


def _wish_from_state(battle_state: Any | None, own_side: bool) -> tuple[int, int]:
    if battle_state is None:
        return (0, 0)
    battlefield = getattr(battle_state, "battlefield", None)
    if battlefield is None:
        return (0, 0)
    wish_map = getattr(battlefield, "wish", None)
    if not isinstance(wish_map, dict):
        return (0, 0)
    side_key = "own" if own_side else "opponent"
    wish_value = wish_map.get(side_key, (0, 0))
    if isinstance(wish_value, tuple) and len(wish_value) == 2:
        try:
            return int(wish_value[0]), int(wish_value[1])
        except Exception:
            return (0, 0)
    return (0, 0)


def _inject_volatile_statuses_into_state(state: Any, own_state: Any | None, opponent_state: Any | None) -> Any:
    own_statuses = _engine_volatile_statuses_to_serialized_string(own_state)
    opponent_statuses = _engine_volatile_statuses_to_serialized_string(opponent_state)
    if not own_statuses and not opponent_statuses:
        return state
    try:
        parts = state.to_string().split("=")
        if len(parts) > 36:
            if own_statuses:
                parts[8] = own_statuses
            if opponent_statuses:
                parts[36] = opponent_statuses
            return State.from_string("=".join(parts))
    except Exception:
        return state
    return state


def _weighted_choice(rng: random.Random, options: list[dict[str, Any]]) -> dict[str, Any]:
    if not options:
        return {}
    total = sum(int(option.get("count", 1)) for option in options)
    if total <= 0:
        return rng.choice(options)

    roll = rng.uniform(0, total)
    current = 0.0
    for option in options:
        current += int(option.get("count", 1))
        if roll <= current:
            return option
    return options[-1]


def choose_random_battle_set(
    species_name: str | None,
    revealed_moves: list[str] | None = None,
    revealed_ability: str | None = None,
    revealed_item: str | None = None,
    impossible_abilities: set[str] | None = None,
    impossible_items: set[str] | None = None,
    rng: random.Random | None = None,
    data_dir: str | None = None,
    uniform: bool = True,
) -> dict[str, Any]:
    rng = rng or random.Random()
    all_sets = get_feasible_random_battle_sets(
        species_name,
        revealed_moves=revealed_moves,
        revealed_ability=revealed_ability,
        revealed_item=revealed_item,
        impossible_abilities=impossible_abilities,
        impossible_items=impossible_items,
        data_dir=data_dir,
    )
    if not all_sets:
        return {}

    if uniform:
        return rng.choice(all_sets)
    return _weighted_choice(rng, all_sets)


def filter_candidate_sets_by_observed_damage(
    snapshot: BattleSnapshot,
    candidate_sets: list[dict[str, Any]],
    source_move_id: str | None,
    source_is_opponent: bool,
    observed_damage_fraction: float,
    observed_was_crit: bool = False,
) -> list[dict[str, Any]]:
    if not candidate_sets or not source_move_id:
        return candidate_sets

    try:
        from engine.bridge import PokeEngineBridge
        from poke_engine import calculate_damage
    except Exception:
        return candidate_sets

    bridge = PokeEngineBridge()
    filtered: list[dict[str, Any]] = []

    for index, candidate_set in enumerate(candidate_sets):
        try:
            state = bridge.snapshot_to_state(
                snapshot,
                rng=random.Random(f"{snapshot.room_id}:{snapshot.turn}:{index}"),
                opponent_set_override=candidate_set,
            )
        except Exception:
            state = None
        if state is None:
            continue

        try:
            if source_is_opponent:
                _, rolls = calculate_damage(state, "tackle", source_move_id, True)
                defender_side = getattr(state, "side_one", None)
            else:
                rolls, _ = calculate_damage(state, source_move_id, "tackle", True)
                defender_side = getattr(state, "side_two", None)
        except Exception:
            continue

        if defender_side is None:
            continue
        try:
            active_index = int(getattr(defender_side, "active_index", "0") or 0)
        except Exception:
            active_index = 0
        active_index = max(0, min(5, active_index))
        try:
            defender = defender_side.pokemon[active_index]
            defender_max_hp = float(getattr(defender, "maxhp", 0) or 0)
        except Exception:
            defender_max_hp = 0.0
        if defender_max_hp <= 0:
            continue

        normalized_rolls = [float(value) / defender_max_hp for value in rolls if isinstance(value, (int, float))]
        if not normalized_rolls:
            continue

        observed = observed_damage_fraction / (1.5 if observed_was_crit else 1.0)
        lower = max(0.0, observed * 0.94)
        upper = min(1.0, observed * 1.06)
        if max(normalized_rolls) >= lower and min(normalized_rolls) <= upper:
            filtered.append(candidate_set)

    return filtered if filtered else candidate_sets


def get_feasible_random_battle_sets(
    species_name: str | None,
    revealed_moves: list[str] | None = None,
    revealed_ability: str | None = None,
    revealed_item: str | None = None,
    impossible_abilities: set[str] | None = None,
    impossible_items: set[str] | None = None,
    data_dir: str | None = None,
) -> list[dict[str, Any]]:
    key = resolve_species_key(species_name)
    all_sets = list(load_random_battle_sets(data_dir).get(key, []))
    if not all_sets:
        return []

    normalized_revealed_moves = {normalize_name(move) for move in (revealed_moves or []) if move}
    normalized_revealed_ability = normalize_name(revealed_ability)
    normalized_revealed_item = normalize_name(revealed_item)
    normalized_impossible_abilities = {normalize_name(a) for a in (impossible_abilities or set()) if a}
    normalized_impossible_items = {normalize_name(i) for i in (impossible_items or set()) if i}

    feasible = all_sets
    if normalized_revealed_moves:
        feasible = [
            candidate
            for candidate in feasible
            if normalized_revealed_moves.issubset({normalize_name(move) for move in candidate.get("moves", [])})
        ]
    if normalized_revealed_ability:
        feasible = [
            candidate
            for candidate in feasible
            if normalize_name(str(candidate.get("ability", ""))) == normalized_revealed_ability
        ]
    if normalized_revealed_item:
        feasible = [
            candidate
            for candidate in feasible
            if normalize_name(str(candidate.get("item", ""))) == normalized_revealed_item
        ]
    if normalized_impossible_abilities:
        feasible = [
            candidate
            for candidate in feasible
            if normalize_name(str(candidate.get("ability", ""))) not in normalized_impossible_abilities
        ]
    if normalized_impossible_items:
        feasible = [
            candidate
            for candidate in feasible
            if normalize_name(str(candidate.get("item", ""))) not in normalized_impossible_items
        ]

    # If constraints are too strict (or data is incomplete), relax progressively
    # so callers can still build a valid state without crashing.
    if feasible:
        return [_enrich_candidate_set_with_stats(species_name, candidate) for candidate in feasible]
    if normalized_revealed_ability or normalized_revealed_item or normalized_revealed_moves:
        relaxed = all_sets
        if normalized_revealed_moves:
            relaxed = [
                candidate
                for candidate in relaxed
                if normalized_revealed_moves.issubset({normalize_name(move) for move in candidate.get("moves", [])})
            ]
        if normalized_revealed_ability:
            relaxed = [
                candidate
                for candidate in relaxed
                if normalize_name(str(candidate.get("ability", ""))) == normalized_revealed_ability
            ]
        if normalized_revealed_item:
            relaxed = [
                candidate
                for candidate in relaxed
                if normalize_name(str(candidate.get("item", ""))) == normalized_revealed_item
            ]
        if relaxed:
            return [_enrich_candidate_set_with_stats(species_name, candidate) for candidate in relaxed]

    # Final fail-safe for edge cases (e.g. Ditto/Illusion/noisy observations):
    # keep sets with the fewest total constraint violations.
    def _violation_score(candidate: dict[str, Any]) -> int:
        score = 0
        candidate_moves = {normalize_name(move) for move in candidate.get("moves", [])}
        candidate_ability = normalize_name(str(candidate.get("ability", "")))
        candidate_item = normalize_name(str(candidate.get("item", "")))

        if normalized_revealed_moves:
            score += len(normalized_revealed_moves.difference(candidate_moves))
        if normalized_revealed_ability and candidate_ability != normalized_revealed_ability:
            score += 2
        if normalized_revealed_item and candidate_item != normalized_revealed_item:
            score += 2
        if normalized_impossible_abilities and candidate_ability in normalized_impossible_abilities:
            score += 1
        if normalized_impossible_items and candidate_item in normalized_impossible_items:
            score += 1
        return score

    best_score = min(_violation_score(candidate) for candidate in all_sets)
    return [
        _enrich_candidate_set_with_stats(species_name, candidate)
        for candidate in all_sets
        if _violation_score(candidate) == best_score
    ]


def _make_move(move_id: str, disabled: bool = False, pp: int | None = None):
    if Move is None:
        return None
    try:
        return Move(id=normalize_name(move_id), pp=32 if pp is None else int(pp), disabled=disabled)
    except Exception:
        try:
            return Move(id=normalize_name(move_id), disabled=disabled)
        except Exception:
            return None


def _move_list_for_species(
    move_ids: list[str],
    *,
    locked_move_id: str | None = None,
    move_pps: dict[str, int | None] | None = None,
) -> list[Any]:
    normalized_locked_move_id = normalize_name(locked_move_id)
    return [
        move
        for move in (
            _make_move(
                move_id,
                disabled=bool(normalized_locked_move_id and normalize_name(move_id) != normalized_locked_move_id),
                pp=(move_pps or {}).get(normalize_name(move_id)),
            )
            for move_id in move_ids
        )
        if move is not None
    ]


def _team_state_entries(team_state: Any | None) -> list[dict[str, Any]]:
    if team_state is None:
        return []

    entries: list[dict[str, Any]] = []
    pokemon_list = getattr(team_state, "pokemon", None)
    if not isinstance(pokemon_list, list):
        return []

    for index, pokemon in enumerate(pokemon_list[:6]):
        if pokemon is None:
            continue
        details_name = getattr(pokemon, "display_name", None) or getattr(pokemon, "species_id", None) or ""
        current_hp = getattr(pokemon, "current_hp", None)
        max_hp = getattr(pokemon, "max_hp", None)
        condition = ""
        if current_hp is not None and max_hp is not None:
            condition = f"{int(current_hp)}/{int(max_hp)}"
        status = getattr(pokemon, "status", None)
        if status:
            condition = f"{condition} {status}".strip()
        moves = []
        for move in getattr(pokemon, "moves", {}).values():
            move_id = getattr(move, "move_id", None)
            if not move_id:
                continue
            current_pp = getattr(move, "pp", None)
            max_pp = getattr(move, "max_pp", None)
            try:
                current_pp_value = int(current_pp) if current_pp is not None else None
            except Exception:
                current_pp_value = None
            try:
                max_pp_value = int(max_pp) if max_pp is not None else None
            except Exception:
                max_pp_value = None
            moves.append(
                {
                    "id": move_id,
                    "move": move.name or move_id,
                    "disabled": bool(getattr(move, "disabled", False) or (current_pp_value is not None and current_pp_value <= 0)),
                    "pp": current_pp_value,
                    "maxpp": max_pp_value,
                }
            )
        entries.append(
            {
                "active": bool(getattr(pokemon, "is_active", False)),
                "details": details_name,
                "condition": condition,
                "ability": getattr(pokemon, "ability", None),
                "item": getattr(pokemon, "item", None),
                "moves": moves,
                "stats": dict(getattr(pokemon, "stats", {}) or {}),
                "species_name": details_name,
                "slot": index,
            }
        )

    return entries


def _active_pokemon_for_observed_species(team_state: Any | None, observed_species: str | None) -> Any | None:
    if team_state is None:
        return None
    normalized_observed = normalize_name(observed_species)
    if not normalized_observed:
        return getattr(team_state, "active_pokemon", None)
    pokemon_list = getattr(team_state, "pokemon", None)
    if not isinstance(pokemon_list, list):
        return getattr(team_state, "active_pokemon", None)
    for pokemon in pokemon_list:
        if pokemon is None:
            continue
        species_name = getattr(pokemon, "display_name", None) or getattr(pokemon, "species_id", None)
        if species_name and normalize_name(species_name) == normalized_observed:
            return pokemon
    return getattr(team_state, "active_pokemon", None)


def _align_team_active_slot(team_state: Any | None, observed_species: str | None) -> None:
    if team_state is None:
        return
    normalized_observed = normalize_name(observed_species)
    if not normalized_observed:
        return
    pokemon_list = getattr(team_state, "pokemon", None)
    if not isinstance(pokemon_list, list):
        return
    for index, pokemon in enumerate(pokemon_list):
        if pokemon is None:
            continue
        species_name = getattr(pokemon, "display_name", None) or getattr(pokemon, "species_id", None)
        if species_name and normalize_name(species_name) == normalized_observed:
            try:
                team_state.set_active_slot(index)
            except Exception:
                pass
            break


def _apply_status_to_pokemon(pokemon: Any, status: str | None) -> Any:
    normalized_status = _engine_status_token(status)
    if pokemon is not None and normalized_status:
        try:
            setattr(pokemon, "status", normalized_status)
        except Exception:
            pass
    return pokemon


def _apply_move_pp_to_pokemon(pokemon: Any, source_pokemon_state: Any | None) -> None:
    if pokemon is None or source_pokemon_state is None:
        return
    source_moves = getattr(source_pokemon_state, "moves", None)
    target_moves = getattr(pokemon, "moves", None)
    if not isinstance(source_moves, dict) or not isinstance(target_moves, (dict, list, tuple)):
        return
    source_move_list = list(source_moves.values())
    for index, source_move in enumerate(source_move_list):
        target_move = None
        if isinstance(target_moves, dict):
            source_move_id = getattr(source_move, "move_id", None)
            if source_move_id is not None:
                target_move = target_moves.get(normalize_name(str(source_move_id)))
        else:
            if index < len(target_moves):
                target_move = target_moves[index]
        if target_move is None:
            continue
        pp_value = getattr(source_move, "pp", None)
        if pp_value is not None:
            try:
                setattr(target_move, "pp", int(pp_value))
            except Exception:
                pass
        if hasattr(target_move, "disabled"):
            try:
                setattr(target_move, "disabled", bool(getattr(source_move, "disabled", False) or (pp_value is not None and int(pp_value) <= 0)))
            except Exception:
                pass


def _apply_move_pp_to_pokemon(pokemon: Any, source_pokemon_state: Any | None) -> None:
    if pokemon is None or source_pokemon_state is None:
        return
    source_moves = getattr(source_pokemon_state, "moves", None)
    target_moves = getattr(pokemon, "moves", None)
    if not isinstance(source_moves, dict) or not isinstance(target_moves, dict):
        return
    for move_id, source_move in source_moves.items():
        target_move = target_moves.get(normalize_name(str(move_id)))
        if target_move is None:
            continue
        pp_value = getattr(source_move, "pp", None)
        if pp_value is not None:
            try:
                setattr(target_move, "pp", int(pp_value))
            except Exception:
                pass


def _engine_last_used_move_token(move_id: str | None, move_ids: list[str] | None) -> str:
    normalized_move_id = normalize_name(move_id)
    if not normalized_move_id or not move_ids:
        return "move:none"
    for index, candidate_move_id in enumerate(move_ids[:4]):
        if normalize_name(candidate_move_id) == normalized_move_id:
            return f"move:{index}"
    return "move:none"


def _current_and_max_hp_from_condition(condition: str | None, fallback_max_hp: int) -> tuple[int, int]:
    if condition:
        hp_match = re.search(r"(\d+)\/(\d+)", condition)
        if hp_match:
            return int(hp_match.group(1)), int(hp_match.group(2))
        if "fnt" in condition.lower():
            return 0, fallback_max_hp
    return fallback_max_hp, fallback_max_hp


def _engine_status_token(status: str | None) -> str:
    normalized_status = normalize_name(status)
    return {
        "brn": "Burn",
        "frz": "Freeze",
        "par": "Paralyze",
        "psn": "Poison",
        "tox": "Toxic",
        "slp": "Sleep",
        "none": "None",
    }.get(normalized_status, normalized_status.capitalize() if normalized_status else "None")


def build_pokemon_from_species(
    species_name: str | None,
    revealed_moves: list[str] | None = None,
    revealed_ability: str | None = None,
    revealed_item: str | None = None,
    impossible_abilities: set[str] | None = None,
    impossible_items: set[str] | None = None,
    rng: random.Random | None = None,
    condition: str | None = None,
    data_dir: str | None = None,
    uniform_set_sampling: bool = True,
    selected_set_override: dict[str, Any] | None = None,
    volatile_statuses: set[str] | None = None,
    status: str | None = None,
    use_revealed_moves: bool = False,
    locked_move_id: str | None = None,
    move_pps: dict[str, int | None] | None = None,
    fallback_stats: dict[str, Any] | None = None,
):
    if Pokemon is None:
        return None

    rng = rng or random.Random()
    species_record = get_species_record(species_name, data_dir)
    if not species_record:
        species_key = resolve_species_key(species_name) or "none"
        item = normalize_name(str(revealed_item or "none"))
        move_ids = [normalize_name(move_id) for move_id in (revealed_moves or []) if move_id]
        try:
            normalized_volatile_statuses = {normalize_name(status) for status in (volatile_statuses or set()) if status}
        except Exception:
            normalized_volatile_statuses = None
        move_lock_id = (
            locked_move_id
            if locked_move_id and (item in CHOICE_ITEMS or (normalized_volatile_statuses and "lockedmove" in normalized_volatile_statuses))
            else None
        )
        py_moves = _move_list_for_species(move_ids, locked_move_id=move_lock_id, move_pps=move_pps)

        fallback_max_hp = None
        if isinstance(fallback_stats, dict):
            try:
                fallback_max_hp = int(fallback_stats.get("hp"))
            except (TypeError, ValueError):
                fallback_max_hp = None
        if fallback_max_hp is None or fallback_max_hp <= 0:
            fallback_max_hp = 100

        if condition:
            current_hp, max_hp = _current_and_max_hp_from_condition(condition, fallback_max_hp)
        else:
            current_hp = fallback_max_hp
            max_hp = fallback_max_hp

        ability = normalize_name(str(revealed_ability or "none"))
        item = normalize_name(str(revealed_item or "none"))
        normalized_status = _engine_status_token(status)
        try:
            return _apply_status_to_pokemon(Pokemon(
                id=species_key,
                hp=current_hp,
                maxhp=max_hp,
                ability=ability,
                base_ability=ability,
                item=item,
                moves=py_moves,
                status=normalized_status,
            ), normalized_status)
        except Exception:
            try:
                return _apply_status_to_pokemon(Pokemon(id=species_key, ability=ability, item=item, moves=py_moves), normalized_status)
            except Exception:
                pass
            return None

    selected_set = selected_set_override or choose_random_battle_set(
        species_name,
        revealed_moves=revealed_moves,
        revealed_ability=revealed_ability,
        revealed_item=revealed_item,
        impossible_abilities=impossible_abilities,
        impossible_items=impossible_items,
        rng=rng,
        data_dir=data_dir,
        uniform=uniform_set_sampling,
    )
    level = int(selected_set.get("level", 100) or 100)
    nature = normalize_name(str(selected_set.get("nature", "serious")))
    cached_profile = selected_set.get("stat_profile")
    cached_stats = selected_set.get("computed_stats")
    if isinstance(cached_profile, dict) and isinstance(cached_stats, dict):
        stat_profile = BuiltPokemonStats(
            level=int(cached_profile.get("level", level) or level),
            nature=str(cached_profile.get("nature", nature)),
            ivs=tuple(cached_profile.get("ivs", (31, 31, 31, 31, 31, 31))),
            evs=tuple(cached_profile.get("evs", (84, 84, 84, 84, 84, 84))),
            stats={str(name): int(value) for name, value in cached_stats.items()},
            has_physical_attacks=bool(cached_profile.get("has_physical_attacks", False)),
        )
    else:
        stat_profile = build_stat_profile(species_name, level, nature)

    base_stats = species_record.get("base_stats", {})
    types = species_record.get("types", ["typeless", "typeless"])
    types = [normalize_name(type_name) for type_name in types]
    if len(types) == 1:
        types.append("typeless")
    types = (types[0], types[1])

    move_ids = [normalize_name(move_id) for move_id in selected_set.get("moves", []) if move_id]
    if use_revealed_moves and revealed_moves:
        move_ids = [normalize_name(move_id) for move_id in revealed_moves if move_id]

    if condition:
        fallback_max_hp = stat_profile.stats["hp"]
        if isinstance(fallback_stats, dict):
            try:
                observed_max_hp = int(fallback_stats.get("hp"))
            except (TypeError, ValueError):
                observed_max_hp = None
            if observed_max_hp is not None and observed_max_hp > 0:
                fallback_max_hp = observed_max_hp
        current_hp, max_hp = _current_and_max_hp_from_condition(condition, fallback_max_hp)
    else:
        max_hp = stat_profile.stats["hp"]
        if isinstance(fallback_stats, dict):
            try:
                observed_max_hp = int(fallback_stats.get("hp"))
            except (TypeError, ValueError):
                observed_max_hp = None
            if observed_max_hp is not None and observed_max_hp > 0:
                max_hp = observed_max_hp
        current_hp = max_hp

    ability = normalize_name(str(selected_set.get("ability", "none")))
    item = normalize_name(str(selected_set.get("item", "none")))
    weight_kg = float(species_record.get("weight_kg", 0.0) or 0.0)
    tera_type = "typeless"
    ability_options = [normalize_name(a) for a in get_species_ability_options(species_name, data_dir)]
    if revealed_ability:
        ability = normalize_name(revealed_ability)
    elif ability_options and ability not in ability_options:
        ability = normalize_name(rng.choice(ability_options))
    if revealed_item:
        item = normalize_name(revealed_item)
    if status:
        status = _engine_status_token(status)

    try:
        normalized_volatile_statuses = {normalize_name(status) for status in (volatile_statuses or set()) if status}
    except Exception:
        normalized_volatile_statuses = None
    move_lock_id = (
        locked_move_id
        if locked_move_id and (item in CHOICE_ITEMS or (normalized_volatile_statuses and "lockedmove" in normalized_volatile_statuses))
        else None
    )
    py_moves = _move_list_for_species(move_ids, locked_move_id=move_lock_id, move_pps=move_pps)

    # Some builds of the poke_engine.Pokemon constructor may not accept all kwargs
    # (notably `volatile_statuses`). Try the full constructor first, then
    # fall back to progressively smaller argument sets if a TypeError is raised.
    try:
        return _apply_status_to_pokemon(Pokemon(
            id=resolve_species_key(species_name),
            level=level,
            types=types,
            base_types=types,
            hp=current_hp,
            maxhp=max_hp,
            ability=ability,
            base_ability=ability,
            item=item,
            nature=nature,
            evs=stat_profile.evs,
            attack=stat_profile.stats["attack"],
            defense=stat_profile.stats["defense"],
            special_attack=stat_profile.stats["special-attack"],
            special_defense=stat_profile.stats["special-defense"],
            speed=stat_profile.stats["speed"],
            status=_engine_status_token(status or "none"),
            rest_turns=0,
            sleep_turns=0,
            weight_kg=weight_kg,
            moves=py_moves,
            terastallized=False,
            tera_type=tera_type,
            volatile_statuses=normalized_volatile_statuses or None,
        ), status)
    except TypeError:
        try:
            # Retry without volatile_statuses kwarg for older/leaner bindings.
            return _apply_status_to_pokemon(Pokemon(
                id=resolve_species_key(species_name),
                level=level,
                types=types,
                base_types=types,
                hp=current_hp,
                maxhp=max_hp,
                ability=ability,
                base_ability=ability,
                item=item,
                nature=nature,
                evs=stat_profile.evs,
                attack=stat_profile.stats["attack"],
                defense=stat_profile.stats["defense"],
                special_attack=stat_profile.stats["special-attack"],
                special_defense=stat_profile.stats["special-defense"],
                speed=stat_profile.stats["speed"],
                status=_engine_status_token(status or "none"),
                rest_turns=0,
                sleep_turns=0,
                weight_kg=weight_kg,
                moves=py_moves,
                terastallized=False,
                tera_type=tera_type,
            ), status)
        except Exception:
            try:
                return _apply_status_to_pokemon(Pokemon(id=resolve_species_key(species_name), ability=ability, item=item, moves=py_moves), status)
            except Exception:
                return None


def build_poke_engine_state_from_snapshot(
    snapshot: BattleSnapshot,
    rng: random.Random | None = None,
    opponent_set_override: dict[str, Any] | None = None,
):
    if State is None or Side is None:
        return None
    rng = rng or random.Random(f"{snapshot.room_id}:{snapshot.turn}")
    request = snapshot.last_request or {}

    def _parse_level_from_details(details: str) -> int:
        level_match = re.search(r"L(\d+)", details or "")
        if level_match:
            try:
                return int(level_match.group(1))
            except Exception:
                return 100
        return 100

    def _extract_entry_move_ids(entry: dict[str, Any]) -> list[str]:
        move_ids: list[str] = []
        for move_value in entry.get("moves", []) or []:
            if isinstance(move_value, dict):
                move_id = move_value.get("id") or move_value.get("move")
            else:
                move_id = move_value
            normalized_move = normalize_name(str(move_id or ""))
            if normalized_move:
                move_ids.append(normalized_move)
        return move_ids

    def _build_own_set_override_from_entry(species_name: str | None, entry: dict[str, Any], condition: str) -> dict[str, Any]:
        details = str(entry.get("details", "") or "")
        level = _parse_level_from_details(details)
        species_record = get_species_record(species_name)
        if species_record:
            base_profile = build_stat_profile(species_name, level, "serious")
            stats = dict(base_profile.stats)
            profile_record = _stat_profile_record(base_profile)
        else:
            stats = {
                "hp": 100,
                "attack": 100,
                "defense": 100,
                "special-attack": 100,
                "special-defense": 100,
                "speed": 100,
            }
            profile_record = {
                "level": level,
                "nature": "serious",
                "ivs": (31, 31, 31, 31, 31, 31),
                "evs": (84, 84, 84, 84, 84, 84),
                "stats": dict(stats),
                "has_physical_attacks": False,
            }

        raw_stats = entry.get("stats", {})
        if isinstance(raw_stats, dict):
            stat_aliases = {
                "hp": "hp",
                "atk": "attack",
                "def": "defense",
                "spa": "special-attack",
                "spd": "special-defense",
                "spe": "speed",
            }
            for raw_name, normalized_name in stat_aliases.items():
                try:
                    if raw_name in raw_stats:
                        stats[normalized_name] = int(raw_stats[raw_name])
                except Exception:
                    continue

        hp_match = re.search(r"(\d+)\/(\d+)", condition or "")
        if hp_match:
            try:
                stats["hp"] = int(hp_match.group(2))
            except Exception:
                pass

        profile_record["level"] = level
        profile_record["nature"] = "serious"
        profile_record["stats"] = dict(stats)

        return {
            "level": level,
            "nature": "serious",
            "moves": _extract_entry_move_ids(entry),
            "ability": normalize_name(str(entry.get("ability", "none") or "none")),
            "item": normalize_name(str(entry.get("item", "none") or "none")),
            "computed_stats": dict(stats),
            "stat_profile": profile_record,
            "count": 1,
        }

    def _make_placeholder_side() -> Any:
        pokemon_list = []
        if Pokemon is not None:
            for _ in range(6):
                try:
                    pokemon_list.append(Pokemon.create_fainted())
                except Exception:
                    try:
                        pokemon_list.append(Pokemon(id="none"))
                    except Exception:
                        break
        while len(pokemon_list) < 6:
            pokemon_list.append(None)
        return pokemon_list

    def _build_side(
        team_entries: list[dict[str, Any]],
        active_species: str | None,
        team_state: Any | None = None,
        observed_moves: list[str] | None = None,
        available_move_ids: list[str] | None = None,
        active_pokemon_state: Any | None = None,
        is_own_side: bool = False,
        last_used_move_id: str | None = None,
        force_switch: bool = False,
        force_trapped: bool = False,
        wish: tuple[int, int] = (0, 0),
        side_conditions: Any | None = None,
    ) -> Any:
        pokemon_slots = _make_placeholder_side()
        active_index = 0
        seen_active = False
        active_volatile_statuses = _engine_volatile_statuses_from_state(active_pokemon_state)
        boost_kwargs = {
            "attack_boost": _engine_boost_from_state(active_pokemon_state, "attack"),
            "defense_boost": _engine_boost_from_state(active_pokemon_state, "defense"),
            "special_attack_boost": _engine_boost_from_state(active_pokemon_state, "special-attack"),
            "special_defense_boost": _engine_boost_from_state(active_pokemon_state, "special-defense"),
            "speed_boost": _engine_boost_from_state(active_pokemon_state, "speed"),
            "accuracy_boost": _engine_boost_from_state(active_pokemon_state, "accuracy"),
            "evasion_boost": _engine_boost_from_state(active_pokemon_state, "evasion"),
        }
        active_status = getattr(active_pokemon_state, "status", None)

        for index, entry in enumerate(team_entries[:6]):
            details = str(entry.get("details", ""))
            condition = str(entry.get("condition", ""))
            is_active = bool(entry.get("active"))
            slot_pokemon_state = active_pokemon_state if is_active else None
            species_name = str(entry.get("species_name", "")).strip() or (details.split(",", 1)[0].strip() if details else None)
            if not species_name and is_active:
                species_name = active_species
            entry_move_ids = _extract_entry_move_ids(entry)
            entry_status = extract_status_from_condition(condition)
            if is_active:
                active_index = index
                seen_active = True

            selected_set_override = None
            revealed_ability = None
            revealed_item = None
            revealed_moves = observed_moves if is_active else None
            if is_own_side:
                selected_set_override = _build_own_set_override_from_entry(species_name, entry, condition)
                revealed_ability = entry.get("ability")
                revealed_item = entry.get("item") or (slot_pokemon_state.item if is_active and slot_pokemon_state is not None else None)
                revealed_moves = observed_moves if is_active and observed_moves else entry_move_ids

            active_locked_move_id = last_used_move_id
            if is_own_side and is_active:
                normalized_current_moves = {
                    normalize_name(move_id)
                    for move_id in (available_move_ids or [])
                    if normalize_name(move_id)
                }
                if normalized_current_moves and normalize_name(active_locked_move_id) not in normalized_current_moves and not (
                    active_volatile_statuses is not None and "lockedmove" in active_volatile_statuses
                ):
                    active_locked_move_id = None

            slot_move_pps = _move_pps_from_entry(entry)
            if not slot_move_pps and slot_pokemon_state is not None:
                slot_move_pps = _move_pps_from_state(slot_pokemon_state)

            pokemon = build_pokemon_from_species(
                species_name,
                revealed_moves=revealed_moves,
                revealed_ability=revealed_ability,
                revealed_item=revealed_item,
                rng=rng,
                condition=condition,
                selected_set_override=selected_set_override,
                volatile_statuses=_engine_volatile_statuses_from_state(slot_pokemon_state) if slot_pokemon_state is not None else None,
                status=getattr(slot_pokemon_state, "status", None) if slot_pokemon_state is not None else entry_status,
                use_revealed_moves=is_own_side,
                locked_move_id=active_locked_move_id if is_active else None,
                move_pps=slot_move_pps,
                fallback_stats=getattr(slot_pokemon_state, "stats", None) if slot_pokemon_state is not None else entry.get("stats"),
            )
            if pokemon is not None:
                pokemon_slots[index] = pokemon

        if not seen_active and active_species:
            active_pokemon = build_pokemon_from_species(
                active_species,
                revealed_moves=observed_moves,
                rng=rng,
                volatile_statuses=active_volatile_statuses,
                status=active_status,
                use_revealed_moves=True,
                locked_move_id=last_used_move_id,
                move_pps=_move_pps_from_state(active_pokemon_state),
                fallback_stats=getattr(active_pokemon_state, "stats", None) if active_pokemon_state is not None else None,
            )
            if active_pokemon is not None:
                pokemon_slots[0] = active_pokemon
                active_index = 0

        while len(pokemon_slots) < 6:
            pokemon_slots.append(None)

        pokemon_slots = [pkmn if pkmn is not None else pokemon_slots[0] for pkmn in pokemon_slots[:6]]
        return Side(
            pokemon=pokemon_slots,
            side_conditions=side_conditions or SideConditions(),
            wish=wish,
            active_index=str(active_index),
            last_used_move=_engine_last_used_move_token(last_used_move_id, observed_moves if observed_moves else None),
            force_switch=force_switch,
            force_trapped=force_trapped,
            **boost_kwargs,
        )

    own_moves: list[str] = []
    active_list = request.get("active", [])
    if active_list:
        active = active_list[0]
        for move in active.get("moves", []):
            move_id = move.get("id") or move.get("move")
            if move_id:
                own_moves.append(str(move_id))
    own_request_force_trapped = _force_trapped_from_request(request)

    own_side_entries = request.get("side", {}).get("pokemon", [])
    battle_state = snapshot.battle_state
    own_team_state = getattr(battle_state, "own_team", None) if battle_state is not None else None
    opponent_team_state = getattr(battle_state, "opponent_team", None) if battle_state is not None else None
    _align_team_active_slot(own_team_state, snapshot.active_species)
    _align_team_active_slot(opponent_team_state, snapshot.opponent_active_species)
    own_state = own_team_state.active_pokemon if own_team_state is not None else None
    opponent_state = opponent_team_state.active_pokemon if opponent_team_state is not None else None
    opponent_team_entries = _team_state_entries(opponent_team_state) if battle_state is not None else []
    available_actions = list(snapshot.available_actions or [])
    request_force_switch = getattr(battle_state, "request_force_switch", None) if battle_state is not None else None
    own_request_force_switch = bool(request_force_switch and any(bool(value) for value in request_force_switch))
    own_force_switch = False
    own_state_hp = None
    if own_state is not None and getattr(own_state, "current_hp", None) is not None:
        try:
            own_state_hp = int(getattr(own_state, "current_hp", 0) or 0)
        except Exception:
            own_state_hp = None
    if own_request_force_switch:
        own_force_switch = True
    for entry in own_side_entries:
        if not isinstance(entry, dict) or not entry.get("active"):
            continue
        condition = str(entry.get("condition", ""))
        if "fnt" in condition.lower() and (own_state_hp is None or own_state_hp <= 0):
            print("force switch because fainted")
            own_force_switch = True
            break
    if not own_force_switch and own_state_hp is not None:
        own_force_switch = own_state_hp <= 0
        if own_force_switch:
            print("force switch because no HP left")
    own_last_used_move_id = _last_used_move_from_state(own_state)
    own_side = _build_side(
        own_side_entries,
        snapshot.active_species,
        own_team_state,
        own_moves,
        available_move_ids=own_moves,
        active_pokemon_state=own_state,
        is_own_side=True,
        last_used_move_id=own_last_used_move_id,
        force_switch=own_force_switch,
        force_trapped=own_request_force_trapped or _force_trapped_from_state(own_state),
        wish=_wish_from_state(battle_state, own_side=True),
        side_conditions=_side_conditions_from_state(battle_state, own_side=True),
    )

    opponent_constraints = snapshot.opponent_constraints
    if opponent_set_override is not None:
        feasible_opponent_sets = [opponent_set_override]
    else:
        feasible_opponent_sets = get_feasible_random_battle_sets(
            snapshot.opponent_active_species,
            revealed_moves=sorted(opponent_constraints.revealed_moves),
            revealed_ability=opponent_constraints.revealed_ability,
            revealed_item=opponent_constraints.revealed_item,
            impossible_abilities=opponent_constraints.impossible_abilities,
            impossible_items=opponent_constraints.impossible_items,
        )

    if opponent_set_override is None and feasible_opponent_sets and opponent_constraints.observed_damage_fraction is not None:
        source_move_id = (
            opponent_constraints.observed_damage_source_move_id or opponent_constraints.last_opponent_move_id
        )
        source_is_opponent = opponent_constraints.observed_damage_source_is_opponent
        if source_move_id and source_is_opponent is None:
            source_is_opponent = True
        if source_move_id and source_is_opponent is not None:
            filtered_sets = filter_candidate_sets_by_observed_damage(
                snapshot=snapshot,
                candidate_sets=feasible_opponent_sets,
                source_move_id=source_move_id,
                source_is_opponent=source_is_opponent,
                observed_damage_fraction=float(opponent_constraints.observed_damage_fraction),
                observed_was_crit=opponent_constraints.observed_damage_was_crit,
            )
            if filtered_sets:
                feasible_opponent_sets = filtered_sets

    selected_opponent_set: dict[str, Any] | None = None
    if feasible_opponent_sets:
        selected_opponent_set = rng.choice(feasible_opponent_sets)

    opponent_side = _build_side(
        opponent_team_entries,
        snapshot.opponent_active_species,
        opponent_team_state,
        observed_moves=sorted(opponent_constraints.revealed_moves),
        active_pokemon_state=opponent_state,
        is_own_side=False,
        last_used_move_id=_last_used_move_from_state(opponent_state),
        force_switch=bool(opponent_state is not None and getattr(opponent_state, "current_hp", None) is not None and int(getattr(opponent_state, "current_hp", 0) or 0) <= 0),
        force_trapped=_force_trapped_from_state(opponent_state),
        wish=_wish_from_state(battle_state, own_side=False),
        side_conditions=_side_conditions_from_state(battle_state, own_side=False),
    )

    state = State(
        side_one=own_side,
        side_two=opponent_side,
        weather=_engine_weather_from_state(battle_state),
        terrain=Terrain.NONE if Terrain is not None else None,
        trick_room=bool(getattr(getattr(battle_state, "battlefield", None), "trick_room", False)) if battle_state is not None else False,
    )
    state = _inject_volatile_statuses_into_state(state, own_state, opponent_state)
    return state
