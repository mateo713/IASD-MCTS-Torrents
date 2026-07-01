from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
FOUL_PLAY_ROOT = ROOT.parent / "foul-play"
FOUL_PLAY_DATA_DIR = FOUL_PLAY_ROOT / "data"
OUTPUT_DIR = ROOT / "data" / "gen5"
REMOTE_RANDBATS_URL = "https://pkmn.github.io/randbats/data/full/gen5randombattle.json"

STAT_KEYS = ("hp", "attack", "defense", "special-attack", "special-defense", "speed")


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


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected {path} to contain a JSON object")
    return data


def load_source_data() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    pokedex = load_json(FOUL_PLAY_DATA_DIR / "pokedex.json")
    moves = load_json(FOUL_PLAY_DATA_DIR / "moves.json")
    randbats = requests.get(REMOTE_RANDBATS_URL, timeout=30).json()
    if not isinstance(randbats, dict):
        raise TypeError("Remote randbats dataset was not a JSON object")
    return pokedex, moves, randbats


def choose_pokedex_entry(pokedex: dict[str, Any], species_key: str) -> tuple[str, dict[str, Any]] | None:
    if species_key in pokedex:
        return species_key, pokedex[species_key]

    for key, entry in pokedex.items():
        if normalize_name(entry.get("name", key)) == species_key:
            return key, entry

    return None


def parse_set_string(set_string: str) -> dict[str, Any]:
    parts = [part.strip() for part in set_string.split(",") if part.strip()]
    if len(parts) < 4:
        raise ValueError(f"Unexpected random battle set format: {set_string}")

    level = int(parts[0])
    item = normalize_name(parts[1])
    ability = normalize_name(parts[2])
    moves = [normalize_name(move) for move in parts[3:7]]
    return {
        "level": level,
        "item": item,
        "ability": ability,
        "moves": moves,
    }


def is_physical_move(move_id: str, moves: dict[str, Any]) -> bool:
    move_entry = moves.get(move_id, {})
    return str(move_entry.get("category", "")).lower() == "physical"


def build_datasets() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    pokedex, moves, randbats = load_source_data()
    stats_out: dict[str, Any] = {}
    abilities_out: dict[str, Any] = {}
    randbats_out: dict[str, list[dict[str, Any]]] = {}

    for species_name, set_map in randbats.items():
        species_key = normalize_name(species_name)
        resolved = choose_pokedex_entry(pokedex, species_key)
        if resolved is None:
            continue

        _, entry = resolved
        abilities = [normalize_name(ability_name) for ability_name in entry.get("abilities", {}).values()]
        base_stats_raw = entry.get("baseStats", {})
        base_stats = {
            "hp": int(base_stats_raw.get("hp", 0)),
            "attack": int(base_stats_raw.get("attack", 0)),
            "defense": int(base_stats_raw.get("defense", 0)),
            "special-attack": int(base_stats_raw.get("special-attack", 0)),
            "special-defense": int(base_stats_raw.get("special-defense", 0)),
            "speed": int(base_stats_raw.get("speed", 0)),
        }
        types = [normalize_name(type_name) for type_name in entry.get("types", [])]
        if len(types) == 1:
            types.append("typeless")
        elif not types:
            types = ["typeless", "typeless"]

        has_physical_attacks = False
        parsed_sets: list[dict[str, Any]] = []
        for set_string, count in set_map.items():
            parsed = parse_set_string(set_string)
            parsed["count"] = int(count)
            parsed_sets.append(parsed)
            for move_id in parsed["moves"]:
                if is_physical_move(move_id, moves):
                    has_physical_attacks = True

        parsed_sets.sort(key=lambda item: item["count"], reverse=True)

        stats_out[species_key] = {
            "num": int(entry.get("num", 0)),
            "name": normalize_name(entry.get("name", species_name)),
            "types": types,
            "base_stats": base_stats,
            "weight_kg": float(entry.get("weightkg", 0.0) or 0.0),
            "has_physical_attacks": has_physical_attacks,
        }
        abilities_out[species_key] = {
            "abilities": abilities,
            "most_likely_ability": abilities[0] if abilities else "none",
            "has_physical_attacks": has_physical_attacks,
        }
        randbats_out[species_key] = parsed_sets

    return stats_out, abilities_out, randbats_out


def write_datasets(stats_out: dict[str, Any], abilities_out: dict[str, Any], randbats_out: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "pokedex_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats_out, handle, indent=2, sort_keys=True)
    with (OUTPUT_DIR / "pokedex_abilities.json").open("w", encoding="utf-8") as handle:
        json.dump(abilities_out, handle, indent=2, sort_keys=True)
    with (OUTPUT_DIR / "random_battle_sets.json").open("w", encoding="utf-8") as handle:
        json.dump(randbats_out, handle, indent=2, sort_keys=True)


def main() -> int:
    stats_out, abilities_out, randbats_out = build_datasets()
    write_datasets(stats_out, abilities_out, randbats_out)
    print(f"Wrote datasets to {OUTPUT_DIR}")
    print(f"Species: {len(stats_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
