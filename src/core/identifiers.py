from __future__ import annotations

import re
import unicodedata

_IDENTIFIER_RE = re.compile(r"[^a-z0-9]")

_STAT_NAME_ALIASES = {
    "atk": "attack",
    "attack": "attack",
    "def": "defense",
    "defense": "defense",
    "spa": "special-attack",
    "spatk": "special-attack",
    "specialattack": "special-attack",
    "spc": "special-attack",
    "spdef": "special-defense",
    "specialdefense": "special-defense",
    "spd": "special-defense",
    "spe": "speed",
    "speed": "speed",
    "acc": "accuracy",
    "accuracy": "accuracy",
    "eva": "evasion",
    "evasion": "evasion",
}


def normalize_identifier(value: str | None) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKD", str(value))
    return _IDENTIFIER_RE.sub("", normalized.lower())


def normalize_stat_name(value: str | None) -> str:
    normalized = normalize_identifier(value)
    return _STAT_NAME_ALIASES.get(normalized, normalized)


def normalize_volatile_status_name(value: str | None) -> str:
    if not value:
        return ""
    raw_value = str(value).strip()
    if not raw_value:
        return ""
    if ":" in raw_value:
        raw_value = raw_value.rsplit(":", 1)[1].strip()
    return normalize_identifier(raw_value)


normalize_species_id = normalize_identifier
normalize_move_id = normalize_identifier
normalize_item_id = normalize_identifier
normalize_ability_id = normalize_identifier
normalize_slot_id = normalize_identifier


def extract_species_from_details(details: str | None) -> str | None:
    if not details:
        return None
    species = details.split(",", 1)[0].strip()
    return species or None


def extract_hp_chunk(condition: str | None) -> tuple[int, int] | None:
    if not condition:
        return None
    hp_chunk = str(condition).split(" ", 1)[0]
    if "/" not in hp_chunk:
        return None
    try:
        current_raw, max_raw = hp_chunk.split("/", 1)
        return int(current_raw), int(max_raw)
    except ValueError:
        return None


def hp_fraction_from_condition(condition: str | None) -> float | None:
    hp_chunk = extract_hp_chunk(condition)
    if hp_chunk is None:
        return None
    current_hp, max_hp = hp_chunk
    if max_hp <= 0:
        return None
    return max(0.0, min(1.0, float(current_hp) / float(max_hp)))


def extract_status_from_condition(condition: str | None) -> str | None:
    if not condition:
        return None
    parts = str(condition).split(" ", 1)
    if len(parts) < 2:
        return None
    token = parts[1].strip().lower()
    if token.startswith("tox"):
        return "tox"
    if token.startswith("psn"):
        return "psn"
    if token.startswith("brn"):
        return "brn"
    if token.startswith("par"):
        return "par"
    if token.startswith("slp"):
        return "slp"
    if token.startswith("frz"):
        return "frz"
    return None


def slot_index_from_ident(ident: str | None) -> int | None:
    if not ident:
        return None
    marker = str(ident).split(":", 1)[0].strip().lower()
    match = re.search(r"([a-f])$", marker)
    if match is not None:
        return ord(match.group(1)) - ord("a")
    match = re.search(r"(\d+)$", marker)
    if match is not None:
        try:
            return max(0, int(match.group(1)) - 1)
        except ValueError:
            return None
    return None
