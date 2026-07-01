from __future__ import annotations

from typing import Any


POKEMON_ALIVE = 30.0
POKEMON_HP = 100.0
USED_TERA = -75.0

POKEMON_ATTACK_BOOST = 30.0
POKEMON_DEFENSE_BOOST = 15.0
POKEMON_SPECIAL_ATTACK_BOOST = 30.0
POKEMON_SPECIAL_DEFENSE_BOOST = 15.0
POKEMON_SPEED_BOOST = 30.0

POKEMON_BOOST_MULTIPLIER_6 = 3.3
POKEMON_BOOST_MULTIPLIER_5 = 3.15
POKEMON_BOOST_MULTIPLIER_4 = 3.0
POKEMON_BOOST_MULTIPLIER_3 = 2.5
POKEMON_BOOST_MULTIPLIER_2 = 2.0
POKEMON_BOOST_MULTIPLIER_1 = 1.0
POKEMON_BOOST_MULTIPLIER_0 = 0.0
POKEMON_BOOST_MULTIPLIER_NEG_1 = -1.0
POKEMON_BOOST_MULTIPLIER_NEG_2 = -2.0
POKEMON_BOOST_MULTIPLIER_NEG_3 = -2.5
POKEMON_BOOST_MULTIPLIER_NEG_4 = -3.0
POKEMON_BOOST_MULTIPLIER_NEG_5 = -3.15
POKEMON_BOOST_MULTIPLIER_NEG_6 = -3.3

POKEMON_FROZEN = -40.0
POKEMON_ASLEEP = -25.0
POKEMON_PARALYZED = -25.0
POKEMON_TOXIC = -30.0
POKEMON_POISONED = -10.0
POKEMON_BURNED = -25.0

LEECH_SEED = -30.0
SUBSTITUTE = 40.0
CONFUSION = -20.0

REFLECT = 20.0
LIGHT_SCREEN = 20.0
AURORA_VEIL = 40.0
SAFE_GUARD = 5.0
TAILWIND = 7.0
HEALING_WISH = 30.0

STEALTH_ROCK = -10.0
SPIKES = -7.0
TOXIC_SPIKES = -7.0
STICKY_WEB = -25.0


def _normalized_key(value: Any) -> str:
    if value is None:
        return ""
    from engine.inference import normalize_entity_id

    return normalize_entity_id(str(value))


def _get_value(container: Any, attribute_name: str, default: Any = None) -> Any:
    if container is None:
        return default
    if isinstance(container, dict):
        return container.get(attribute_name, default)
    return getattr(container, attribute_name, default)


def _condition_count(container: Any, attribute_name: str) -> float:
    value = _get_value(container, attribute_name, 0)
    if isinstance(value, dict):
        for key in ("layers", "count", "value"):
            nested = value.get(key)
            if nested is not None:
                value = nested
                break
    try:
        return float(value or 0)
    except Exception:
        return 0.0


def _get_side_pokemon(side: Any) -> list[Any]:
    pokemon = _get_value(side, "pokemon", []) or []
    return list(pokemon)


def _pokemon_types(pokemon: Any) -> tuple[str, ...]:
    from engine.inference import get_species_types

    raw_types = _get_value(pokemon, "types", None) or _get_value(pokemon, "base_types", None)
    if isinstance(raw_types, (list, tuple)) and raw_types:
        normalized = [_normalized_key(type_name) for type_name in raw_types if _normalized_key(type_name)]
        if normalized:
            return tuple(normalized[:2])

    species_name = (
        _get_value(pokemon, "species_id", None)
        or _get_value(pokemon, "display_name", None)
        or _get_value(pokemon, "name", None)
        or _get_value(pokemon, "id", None)
    )
    return get_species_types(str(species_name) if species_name else None)


def _pokemon_ability(pokemon: Any) -> str:
    return _normalized_key(
        _get_value(pokemon, "ability", None)
        or _get_value(pokemon, "base_ability", None)
    )


def _pokemon_item(pokemon: Any) -> str:
    return _normalized_key(_get_value(pokemon, "item", None))


def _pokemon_status(pokemon: Any) -> str:
    return _normalized_key(_get_value(pokemon, "status", None))


def _pokemon_hp_score(pokemon: Any) -> float:
    hp = float(_get_value(pokemon, "hp", 0) or 0)
    maxhp = float(_get_value(pokemon, "maxhp", 0) or 0)
    score = 0.0
    if maxhp > 0:
        score += POKEMON_HP * hp / maxhp
    return score


def _move_category(move: Any) -> str:
    category = _get_value(_get_value(move, "choice", None), "category", None)
    if category is None:
        category = _get_value(move, "category", None)
    return _normalized_key(category)


def _pokemon_is_grounded(pokemon: Any) -> bool:
    grounded = _get_value(pokemon, "grounded", None)
    if grounded is not None:
        return bool(grounded)

    is_grounded = _get_value(pokemon, "is_grounded", None)
    if callable(is_grounded):
        try:
            return bool(is_grounded())
        except Exception:
            pass
    elif is_grounded is not None:
        return bool(is_grounded)

    ability = _pokemon_ability(pokemon)
    return "flying" not in _pokemon_types(pokemon) and ability != "levitate"


def _evaluate_boost_multiplier(boost: int) -> float:
    if boost == 6:
        return POKEMON_BOOST_MULTIPLIER_6
    if boost == 5:
        return POKEMON_BOOST_MULTIPLIER_5
    if boost == 4:
        return POKEMON_BOOST_MULTIPLIER_4
    if boost == 3:
        return POKEMON_BOOST_MULTIPLIER_3
    if boost == 2:
        return POKEMON_BOOST_MULTIPLIER_2
    if boost == 1:
        return POKEMON_BOOST_MULTIPLIER_1
    if boost == 0:
        return POKEMON_BOOST_MULTIPLIER_0
    if boost == -1:
        return POKEMON_BOOST_MULTIPLIER_NEG_1
    if boost == -2:
        return POKEMON_BOOST_MULTIPLIER_NEG_2
    if boost == -3:
        return POKEMON_BOOST_MULTIPLIER_NEG_3
    if boost == -4:
        return POKEMON_BOOST_MULTIPLIER_NEG_4
    if boost == -5:
        return POKEMON_BOOST_MULTIPLIER_NEG_5
    if boost == -6:
        return POKEMON_BOOST_MULTIPLIER_NEG_6
    raise ValueError(f"Invalid boost value: {boost}")


def _evaluate_poison(pokemon: Any, base_score: float) -> float:
    ability = _pokemon_ability(pokemon)
    if ability == "poisonheal":
        return 15.0
    if ability in {"guts", "marvelscale", "quickfeet", "toxicboost", "magicguard"}:
        return 10.0
    return base_score


def _evaluate_burned(pokemon: Any) -> float:
    ability = _pokemon_ability(pokemon)
    if ability in {"guts", "marvelscale", "quickfeet"}:
        return -2.0 * POKEMON_BURNED

    multiplier = 0.0
    for move in _get_value(pokemon, "moves", []) or []:
        if _move_category(move) == "physical":
            multiplier += 1.0

    if float(_get_value(pokemon, "special_attack", 0) or 0) > float(_get_value(pokemon, "attack", 0) or 0):
        multiplier /= 2.0

    return multiplier * POKEMON_BURNED


def _evaluate_hazards(pokemon: Any, side: Any) -> float:
    score = 0.0
    if _pokemon_item(pokemon) == "heavydutyboots":
        return score
    if _pokemon_ability(pokemon) == "magicguard":
        return score

    side_conditions = _get_value(side, "side_conditions", None)
    grounded = _pokemon_is_grounded(pokemon)
    score += _condition_count(side_conditions, "stealth_rock") * STEALTH_ROCK
    if grounded:
        score += _condition_count(side_conditions, "spikes") * SPIKES
        score += _condition_count(side_conditions, "toxic_spikes") * TOXIC_SPIKES
        score += _condition_count(side_conditions, "sticky_web") * STICKY_WEB
    return score


def _evaluate_pokemon(pokemon: Any) -> float:
    score = _pokemon_hp_score(pokemon)

    status = _pokemon_status(pokemon)
    if status == "burn":
        score += _evaluate_burned(pokemon)
    elif status == "freeze":
        score += POKEMON_FROZEN
    elif status == "sleep":
        score += POKEMON_ASLEEP
    elif status == "paralyze":
        score += POKEMON_PARALYZED
    elif status == "toxic":
        score += _evaluate_poison(pokemon, POKEMON_TOXIC)
    elif status == "poison":
        score += _evaluate_poison(pokemon, POKEMON_POISONED)

    if _pokemon_item(pokemon) != "none":
        score += 10.0

    if score < 0.0:
        score = 0.0

    return score + POKEMON_ALIVE


def _evaluate_active_modifiers(side: Any, score: float) -> float:
    volatile_statuses = _get_value(side, "volatile_statuses", None)
    volatile_status_names: set[str] = set()
    if isinstance(volatile_statuses, dict):
        volatile_status_names = {_normalized_key(status_name) for status_name in volatile_statuses.keys()}
    elif isinstance(volatile_statuses, (set, list, tuple)):
        volatile_status_names = {_normalized_key(status_name) for status_name in volatile_statuses}

    for status_name, delta in (("leechseed", LEECH_SEED), ("substitute", SUBSTITUTE), ("confusion", CONFUSION)):
        if status_name in volatile_status_names:
            score += delta

    score += _evaluate_boost_multiplier(int(_get_value(side, "attack_boost", 0) or 0)) * POKEMON_ATTACK_BOOST
    score += _evaluate_boost_multiplier(int(_get_value(side, "defense_boost", 0) or 0)) * POKEMON_DEFENSE_BOOST
    score += _evaluate_boost_multiplier(int(_get_value(side, "special_attack_boost", 0) or 0)) * POKEMON_SPECIAL_ATTACK_BOOST
    score += _evaluate_boost_multiplier(int(_get_value(side, "special_defense_boost", 0) or 0)) * POKEMON_SPECIAL_DEFENSE_BOOST
    score += _evaluate_boost_multiplier(int(_get_value(side, "speed_boost", 0) or 0)) * POKEMON_SPEED_BOOST
    return score


def _evaluate_side_conditions(side: Any, score: float) -> float:
    side_conditions = _get_value(side, "side_conditions", None)
    score += _condition_count(side_conditions, "reflect") * REFLECT
    score += _condition_count(side_conditions, "light_screen") * LIGHT_SCREEN
    score += _condition_count(side_conditions, "aurora_veil") * AURORA_VEIL
    score += _condition_count(side_conditions, "safeguard") * SAFE_GUARD
    score += _condition_count(side_conditions, "tailwind") * TAILWIND
    score += _condition_count(side_conditions, "healing_wish") * HEALING_WISH
    return score


def _evaluate_side(side: Any, *, sign: int) -> float:
    score = 0.0
    pokemon_list = _get_side_pokemon(side)
    used_tera = False
    active_index = int(_get_value(side, "active_index", 0) or 0)

    for index, pokemon in enumerate(pokemon_list):
        hp = float(_get_value(pokemon, "hp", 0) or 0)
        if hp > 0:
            score += _evaluate_pokemon(pokemon)
            score += _evaluate_hazards(pokemon, side)
            if index == active_index:
                score = _evaluate_active_modifiers(side, score)
        if bool(_get_value(pokemon, "terastallized", False)):
            used_tera = True

    if used_tera:
        score += USED_TERA

    score = _evaluate_side_conditions(side, score)
    return score


def evaluate_state(state: Any) -> float:
    score = 0.0
    score += _evaluate_side(_get_value(state, "side_one", None), sign=1)
    score -= _evaluate_side(_get_value(state, "side_two", None), sign=1)
    return score