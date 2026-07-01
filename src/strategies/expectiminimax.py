from __future__ import annotations

import logging
import math
import random
import time
from types import SimpleNamespace
from typing import Any

from core.models import ActionChoice, ActionType, BattleSnapshot
from engine.bridge import PokeEngineBridge
from engine.gen5_datasets import get_feasible_random_battle_sets
from engine.gen5_datasets import normalize_name
from engine.inference import get_move_id_from_action
from strategies.base import Strategy
from strategies.base import fallback_action
from strategies.evaluation import evaluate_state
from strategies.rules import HeuristicStrategy
from strategies.shared_search import TransitionBranch

try:
    from poke_engine import generate_instructions
except Exception:  # pragma: no cover - depends on local environment
    generate_instructions = None


logger = logging.getLogger(__name__)
_HEURISTIC_STRATEGY = HeuristicStrategy()


def _extract_species_from_details(details: str) -> str:
    if not details:
        return ""
    return details.split(",", 1)[0].strip()


def _switch_entries(snapshot: BattleSnapshot) -> list[dict[str, object]]:
    request = snapshot.last_request or {}
    side = request.get("side", {})
    pokemon = side.get("pokemon", [])
    if not isinstance(pokemon, list):
        return []

    entries: list[dict[str, object]] = []
    for entry in pokemon:
        if not isinstance(entry, dict):
            continue
        if entry.get("active"):
            continue
        condition = str(entry.get("condition", ""))
        if "fnt" in condition:
            continue
        entries.append(entry)
    return entries


def _choice_variants(selected_move: str) -> list[str]:
    target = normalize_name(selected_move)
    if not target:
        return []
    variants = [target]
    if target.startswith("switch"):
        stripped = target.removeprefix("switch")
        if stripped:
            variants.append(stripped)
    return variants


def _target_matches_value(target: str, value: str) -> bool:
    if not target or not value:
        return False
    if target == value:
        return True
    for family in ("hiddenpower", "return", "frustration"):
        if target.startswith(family) and value.startswith(family):
            return True
    return False


def _match_available_action(snapshot: BattleSnapshot, selected_move: str) -> ActionChoice | None:
    targets = _choice_variants(selected_move)
    if not targets:
        return None

    for action in snapshot.available_actions:
        command_id = normalize_name(action.command)
        label_id = normalize_name(action.label)
        if any(_target_matches_value(target, command_id) for target in targets) or any(
            _target_matches_value(target, label_id) for target in targets
        ):
            return action

    for action in snapshot.available_actions:
        if action.action_type == ActionType.MOVE:
            move_id = normalize_name(get_move_id_from_action(snapshot, action))
            if any(_target_matches_value(target, move_id) for target in targets):
                return action

    switch_actions = [action for action in snapshot.available_actions if action.action_type == ActionType.SWITCH]
    if not switch_actions:
        return None

    for action in switch_actions:
        command_id = normalize_name(action.command)
        label_id = normalize_name(action.label)
        if any(_target_matches_value(target, command_id) for target in targets) or any(
            _target_matches_value(target, label_id) for target in targets
        ):
            return action

    for index, entry in enumerate(_switch_entries(snapshot)):
        species = normalize_name(_extract_species_from_details(str(entry.get("details", ""))))
        ident = normalize_name(str(entry.get("ident", "")))
        if any(_target_matches_value(target, species) for target in targets) or any(
            _target_matches_value(target, ident) for target in targets
        ):
            if index < len(switch_actions):
                return switch_actions[index]
            break

    if len(snapshot.available_actions) == 1:
        return snapshot.available_actions[0]

    return None


def _candidate_set_weight(candidate_set: dict[str, object] | None) -> float:
    if not candidate_set:
        return 1.0
    try:
        return max(0.0, float(candidate_set.get("count", 1) or 1))
    except Exception:
        return 1.0


def _feasible_opponent_sets(snapshot: BattleSnapshot) -> list[dict[str, object]]:
    constraints = snapshot.opponent_constraints
    feasible_sets = get_feasible_random_battle_sets(
        snapshot.opponent_active_species,
        revealed_moves=sorted(constraints.revealed_moves),
        revealed_ability=constraints.revealed_ability,
        revealed_item=constraints.revealed_item,
        impossible_abilities=constraints.impossible_abilities,
        impossible_items=constraints.impossible_items,
    )
    return feasible_sets or [{}]


def _belief_sample_count(feasible_sets: list[dict[str, object]], duration_ms: int) -> int:
    if not feasible_sets:
        return 1
    budget = max(1, duration_ms // 20)
    return max(1, min(len(feasible_sets), budget))


def _sample_feasible_sets(
    feasible_sets: list[dict[str, object]],
    rng: random.Random,
    duration_ms: int,
) -> list[dict[str, object]]:
    if not feasible_sets:
        return [{}]

    sample_count = _belief_sample_count(feasible_sets, duration_ms)
    if len(feasible_sets) == 1:
        return [feasible_sets[0] for _ in range(sample_count)]

    weights = [_candidate_set_weight(candidate_set) for candidate_set in feasible_sets]
    if not any(weight > 0 for weight in weights):
        weights = None
    return list(rng.choices(feasible_sets, weights=weights, k=sample_count))


def _active_index(side: Any) -> int:
    try:
        return int(getattr(side, "active_index", 0) or 0)
    except Exception:
        return 0


def _active_pokemon(side: Any) -> Any | None:
    if side is None:
        return None
    pokemon = getattr(side, "pokemon", []) or []
    index = _active_index(side)
    if 0 <= index < len(pokemon):
        return pokemon[index]
    return None


def _iter_move_entries(pokemon: Any) -> list[Any]:
    moves = getattr(pokemon, "moves", []) or []
    if isinstance(moves, dict):
        return list(moves.values())
    return list(moves)


def _move_remaining_pp(move: Any) -> int | None:
    try:
        pp = getattr(move, "pp", None)
        if pp is None:
            return None
        return int(pp)
    except Exception:
        return None


def _legal_actions_for_side(state: Any, side: Any) -> list[str]:
    if side is None:
        return ["No Move"]

    pokemon = list(getattr(side, "pokemon", []) or [])
    active = _active_pokemon(side)
    active_index = _active_index(side)
    team_preview = bool(getattr(state, "team_preview", False))
    force_switch = bool(getattr(side, "force_switch", False))
    force_trapped = bool(getattr(side, "force_trapped", False))

    legal_actions: list[str] = []
    if not team_preview and not force_switch and active is not None:
        for move in _iter_move_entries(active):
            if bool(getattr(move, "disabled", False)):
                continue
            remaining_pp = _move_remaining_pp(move)
            if remaining_pp is not None and remaining_pp <= 0:
                continue
            move_id = normalize_name(str(getattr(move, "id", "") or ""))
            if move_id and move_id != "none":
                legal_actions.append(move_id)

    if force_trapped:
        return legal_actions or ["No Move"]

    for index, pokemon_slot in enumerate(pokemon):
        if index == active_index:
            continue
        if float(getattr(pokemon_slot, "hp", 0) or 0) <= 0:
            continue
        switch_id = normalize_name(str(getattr(pokemon_slot, "id", "") or ""))
        if switch_id:
            legal_actions.append(switch_id)

    return legal_actions or ["No Move"]


def _legal_action_provider(state: Any) -> tuple[list[str], list[str]]:
    return _legal_actions_for_side(state, getattr(state, "side_one", None)), _legal_actions_for_side(
        state, getattr(state, "side_two", None)
    )


def _engine_move_id(move_id: str) -> str:
    normalized = normalize_name(move_id)
    if normalized.startswith("hiddenpower"):
        return "hiddenpower"
    if normalized.startswith("return"):
        return "return"
    if normalized.startswith("frustration"):
        return "frustration"
    return normalized


def _transition_provider(state: Any, own_move: str, opp_move: str) -> list[TransitionBranch[Any]]:
    if generate_instructions is None:
        return [TransitionBranch(state=state, weight=1.0)]

    branches: list[TransitionBranch[Any]] = []
    own_move_id = _engine_move_id(own_move)
    opp_move_id = _engine_move_id(opp_move)
    try:
        instructions_iter = generate_instructions(state, own_move_id, opp_move_id)
    except Exception:
        return [TransitionBranch(state=state, weight=1.0)]

    for instructions in instructions_iter:
        next_state = state.apply_instructions(instructions)
        branches.append(TransitionBranch(state=next_state, weight=float(getattr(instructions, "percentage", 0.0) or 0.0)))

    return branches or [TransitionBranch(state=state, weight=1.0)]


def _search_value(state: Any, depth_remaining: int, deadline: float | None) -> float:
    if depth_remaining <= 0:
        return evaluate_state(state)
    if deadline is not None and time.monotonic() >= deadline:
        return evaluate_state(state)

    own_actions, opp_actions = _legal_action_provider(state)
    if not own_actions or not opp_actions or own_actions == ["No Move"] or opp_actions == ["No Move"]:
        return evaluate_state(state)

    best_score = float("-inf")
    for own_move in own_actions:
        if deadline is not None and time.monotonic() >= deadline:
            break
        worst_reply = float("inf")
        for opp_move in opp_actions:
            if deadline is not None and time.monotonic() >= deadline:
                break
            branches = _transition_provider(state, own_move, opp_move)
            branch_total = 0.0
            weight_total = 0.0
            for branch in branches:
                weight = max(0.0, float(getattr(branch, "weight", 0.0) or 0.0))
                if weight <= 0:
                    continue
                branch_total += weight * _search_value(branch.state, depth_remaining - 1, deadline)
                weight_total += weight
            score = branch_total / weight_total if weight_total > 0 else evaluate_state(state)
            worst_reply = min(worst_reply, score)
        best_score = max(best_score, worst_reply)

    return best_score if math.isfinite(best_score) else evaluate_state(state)


def iterative_deepening_expectiminimax(state: Any, duration_ms: int = 100) -> object:
    start = time.monotonic()
    deadline = start + max(0.0, duration_ms / 1000.0)
    own_actions, opp_actions = _legal_action_provider(state)

    if not own_actions or own_actions == ["No Move"]:
        return SimpleNamespace(side_one=["No Move"], side_two=["No Move"], matrix=[evaluate_state(state)], depth_searched=0)

    if not opp_actions:
        opp_actions = ["No Move"]

    final_matrix: list[float] = []
    final_depth = 0
    final_row_scores: dict[str, float] = {}

    depth = 0
    while time.monotonic() < deadline:
        depth += 1
        row_scores: dict[str, float] = {}
        matrix: list[float] = []
        for own_move in own_actions:
            if time.monotonic() >= deadline:
                break
            row_values: list[float] = []
            for opp_move in opp_actions:
                if time.monotonic() >= deadline:
                    break
                branches = _transition_provider(state, own_move, opp_move)
                branch_total = 0.0
                weight_total = 0.0
                for branch in branches:
                    weight = max(0.0, float(getattr(branch, "weight", 0.0) or 0.0))
                    if weight <= 0:
                        continue
                    branch_total += weight * _search_value(branch.state, depth - 1, deadline)
                    weight_total += weight
                score = branch_total / weight_total if weight_total > 0 else evaluate_state(state)
                row_values.append(score)
                matrix.append(score)
            row_scores[own_move] = min(row_values) if row_values else evaluate_state(state)

        final_matrix = matrix
        final_row_scores = row_scores
        final_depth = depth

        if time.monotonic() >= deadline:
            break

    return SimpleNamespace(
        side_one=[str(move) for move in own_actions],
        side_two=[str(move) for move in opp_actions],
        matrix=final_matrix,
        depth_searched=final_depth,
        row_scores=final_row_scores,
    )


def _move_scores_from_result(result: object) -> dict[str, float]:
    side_one = getattr(result, "side_one", []) or []
    side_two = getattr(result, "side_two", []) or []
    matrix = getattr(result, "matrix", []) or []
    if not side_one:
        return {}

    row_count = len(side_one)
    col_count = max(1, len(side_two))
    scores: dict[str, float] = {}
    index = 0

    for row_index in range(row_count):
        worst_score = float("inf")
        saw_finite_score = False
        for _ in range(col_count):
            if index >= len(matrix):
                break
            score = matrix[index]
            index += 1
            if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
                continue
            saw_finite_score = True
            worst_score = min(worst_score, float(score))

        if saw_finite_score:
            scores[str(side_one[row_index])] = worst_score

    if not scores and hasattr(result, "row_scores"):
        for move, score in getattr(result, "row_scores", {}).items():
            if isinstance(score, (int, float)) and math.isfinite(float(score)):
                scores[str(move)] = float(score)

    return scores


def _candidate_weighted_scores(scores_by_move: dict[str, float], sample_weight: float) -> dict[str, float]:
    return {move: score * sample_weight for move, score in scores_by_move.items()}


class ExpectiminimaxStrategy(Strategy):
    name = "expectiminimax"

    def __init__(self, duration_ms: int = 1000) -> None:
        self.duration_ms = max(1, int(duration_ms))
        self._bridge = PokeEngineBridge()

    def choose_action(self, snapshot: BattleSnapshot, rng: random.Random) -> ActionChoice:
        if not snapshot.available_actions:
            return fallback_action(snapshot)

        aggregated_scores: dict[str, float] = {}
        seen_command_to_action: dict[str, ActionChoice] = {}
        sampled_opponent_sets = _sample_feasible_sets(_feasible_opponent_sets(snapshot), rng, self.duration_ms)
        saw_nomove = False
        last_nomove_context: tuple[object, dict[str, object]] | None = None
        last_state: Any | None = None

        for candidate_set in sampled_opponent_sets:
            state = self._bridge.snapshot_to_state(snapshot, rng, opponent_set_override=candidate_set or None)
            if state is None:
                continue

            last_state = state
            try:
                result = iterative_deepening_expectiminimax(state, duration_ms=self.duration_ms)
            except Exception:
                continue

            move_scores = _move_scores_from_result(result)
            if not move_scores:
                continue

            sample_weight = 1 / len(sampled_opponent_sets)
            for selected_move, score in _candidate_weighted_scores(move_scores, sample_weight).items():
                matched_action = _match_available_action(snapshot, selected_move)
                if matched_action is None:
                    if normalize_name(selected_move) == "nomove":
                        saw_nomove = True
                        last_nomove_context = (result, candidate_set)
                        continue
                    raise RuntimeError(
                        f"Expectiminimax selected move {selected_move!r} but it does not match any available action"
                    )

                command = matched_action.command
                seen_command_to_action[command] = matched_action
                aggregated_scores[command] = aggregated_scores.get(command, 0.0) + score

        if not aggregated_scores:
            if saw_nomove:
                print("[expectiminimax] selected No Move; falling back to heuristic action", flush=True)
                return _HEURISTIC_STRATEGY.choose_action(snapshot, rng)
            return fallback_action(snapshot)

        best_score = max(aggregated_scores.values())
        best_commands = [command for command, score in aggregated_scores.items() if score == best_score]
        if len(best_commands) == 1:
            return seen_command_to_action[best_commands[0]]

        return seen_command_to_action[rng.choice(best_commands)]


ExpectiminimaxPlaceholderStrategy = ExpectiminimaxStrategy