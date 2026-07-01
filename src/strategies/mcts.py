from __future__ import annotations

import logging
import random
from types import SimpleNamespace

from core.models import ActionChoice, ActionType, BattleSnapshot
from engine.bridge import PokeEngineBridge
from engine.gen5_datasets import get_feasible_random_battle_sets
from engine.gen5_datasets import normalize_name
from engine.inference import get_move_id_from_action
from strategies.base import Strategy
from strategies.base import fallback_action
from strategies.evaluation import evaluate_state
from strategies.rules import HeuristicStrategy
from strategies.shared_search import PythonMctsConfig, PythonMctsSearch, TransitionBranch

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


def _candidate_set_weight(candidate_set: dict[str, object] | None) -> float:
    if not candidate_set:
        return 1.0
    try:
        return max(0.0, float(candidate_set.get("count", 1) or 1))
    except Exception:
        return 1.0


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


def _entry_average_score(entry: object) -> float:
    total_score = float(getattr(entry, "total_score"))
    visits = int(getattr(entry, "visits"))
    if visits <= 0:
        return float("-inf")
    return total_score / visits


def _top_entries(result: object, limit: int = 3) -> list[tuple[str, float, int]]:
    entries = []
    for entry in getattr(result, "side_one", []) or []:
        try:
            move_choice = str(getattr(entry, "move_choice"))
            score = _entry_average_score(entry)
            visits = int(getattr(entry, "visits"))
        except Exception:
            continue
        entries.append((move_choice, score, visits))
    entries.sort(key=lambda item: item[1], reverse=True)
    return entries[:limit]


def select_move_from_mcts_results(mcts_results: list[tuple[object, float, int]]) -> str:
    final_policy: dict[str, float] = {}
    for mcts_result, sample_chance, index in mcts_results:
        total_visits = int(getattr(mcts_result, "total_visits", 0) or 0)
        if total_visits <= 0:
            total_visits = sum(int(getattr(entry, "visits", 0) or 0) for entry in getattr(mcts_result, "side_one", []) or [])
        if total_visits <= 0:
            total_visits = 1
        this_policy = max(mcts_result.side_one, key=lambda x: x.visits)
        logger.info(
            "Policy {}: {} visited {}% avg_score={} sample_chance_multiplier={}".format(
                index,
                this_policy.move_choice,
                round(100 * this_policy.visits / total_visits, 2),
                round(this_policy.total_score / this_policy.visits, 3),
                round(sample_chance, 3),
            )
        )
        for s1_option in mcts_result.side_one:
            final_policy[s1_option.move_choice] = final_policy.get(
                s1_option.move_choice, 0.0
            ) + (sample_chance * (s1_option.visits / total_visits))

    final_policy = sorted(final_policy.items(), key=lambda x: x[1], reverse=True)
    highest_percentage = final_policy[0][1]
    final_policy = [i for i in final_policy if i[1] >= highest_percentage * 0.75]
    logger.info("Considered Choices:")
    for i, policy in enumerate(final_policy):
        logger.info(f"\t{round(policy[1] * 100, 3)}%: {policy[0]}")

    choice = random.choices(final_policy, weights=[p[1] for p in final_policy])[0]
    return choice[0]


def _build_search_result(best_action: str, search_summary: dict[str, object]) -> object:
    own_actions = []
    for entry in search_summary.get("own_actions", []):
        own_actions.append(
            SimpleNamespace(
                move_choice=str(entry.get("action", "")),
                total_score=float(entry.get("average_score", float("-inf"))) * max(1, int(entry.get("visits", 0) or 0)),
                visits=int(entry.get("visits", 0) or 0),
            )
        )
    return SimpleNamespace(
        best_action=best_action,
        total_visits=int(search_summary.get("iterations", 0) or 0),
        side_one=own_actions,
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


def _transition_provider(state, own_move: str, opp_move: str) -> list[TransitionBranch]:
    if generate_instructions is None:
        return [TransitionBranch(state=state, weight=1.0)]

    branches = []
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


def _active_pokemon(side) -> object | None:
    if side is None:
        return None
    pokemon = getattr(side, "pokemon", []) or []
    try:
        active_index = int(getattr(side, "active_index", 0) or 0)
    except Exception:
        active_index = 0
    if active_index < 0 or active_index >= len(pokemon):
        return None
    return pokemon[active_index]


def _iter_move_entries(pokemon: object) -> list[object]:
    moves = getattr(pokemon, "moves", []) or []
    if isinstance(moves, dict):
        return list(moves.values())
    return list(moves)


def _move_remaining_pp(move: object) -> int | None:
    try:
        pp = getattr(move, "pp", None)
        if pp is None:
            return None
        return int(pp)
    except Exception:
        return None


def _side_legal_actions(state, side) -> list[str]:
    if side is None:
        return []

    actions: list[str] = []
    active_pokemon = _active_pokemon(side)
    try:
        active_index = int(getattr(side, "active_index", 0) or 0)
    except Exception:
        active_index = 0
    force_switch = bool(getattr(side, "force_switch", False))
    force_trapped = bool(getattr(side, "force_trapped", False))
    team_preview = bool(getattr(state, "team_preview", False))

    if not force_switch and not team_preview and active_pokemon is not None:
        for move in _iter_move_entries(active_pokemon):
            if bool(getattr(move, "disabled", False)):
                continue
            remaining_pp = _move_remaining_pp(move)
            if remaining_pp is not None and remaining_pp <= 0:
                continue
            move_id = normalize_name(str(getattr(move, "id", "") or ""))
            if move_id and move_id != "none":
                actions.append(move_id)

    if force_trapped:
        return actions

    for index, pokemon in enumerate(getattr(side, "pokemon", []) or []):
        if active_pokemon is not None and index == active_index:
            continue
        if float(getattr(pokemon, "hp", 0) or 0) <= 0:
            continue
        switch_id = normalize_name(str(getattr(pokemon, "id", "") or ""))
        if switch_id:
            actions.append(switch_id)

    return actions


def _legal_action_provider(state) -> tuple[list[str], list[str]]:
    return _side_legal_actions(state, getattr(state, "side_one", None)), _side_legal_actions(state, getattr(state, "side_two", None))


def monte_carlo_tree_search(state, duration_ms: int = 100) -> object:
    search = PythonMctsSearch(
        root_state=state,
        legal_action_provider=_legal_action_provider,
        transition_provider=_transition_provider,
        state_evaluator=evaluate_state,
        config=PythonMctsConfig(duration_ms=duration_ms),
    )
    best_action, summary = search.run()
    return _build_search_result(best_action, summary)


class MctsStrategy(Strategy):
    name = "mcts"

    def __init__(self, duration_ms: int = 1000) -> None:
        self.duration_ms = max(1, int(duration_ms))
        self._bridge = PokeEngineBridge()

    def choose_action(self, snapshot: BattleSnapshot, rng: random.Random) -> ActionChoice:
        print(
            f"[mcts] room={snapshot.room_id} turn={snapshot.turn} phase={snapshot.phase} "
            f"actions={len(snapshot.available_actions)} own={snapshot.active_species} opp={snapshot.opponent_active_species}"
        )

        feasible_sets = _feasible_opponent_sets(snapshot)
        print(f"[mcts] feasible_sets={len(feasible_sets)}")
        sampled_opponent_sets = _sample_feasible_sets(feasible_sets, rng, self.duration_ms)
        print(f"[mcts] sampled_belief_worlds={len(sampled_opponent_sets)}")

        mcts_results: list[tuple[object, float, int]] = []
        for index, selected_opponent_set in enumerate(sampled_opponent_sets):
            print(
                f"[mcts] selected_opponent_set_count={selected_opponent_set.get('count', 1) if isinstance(selected_opponent_set, dict) else 'n/a'}"
            )

            state = self._bridge.snapshot_to_state(snapshot, rng, opponent_set_override=selected_opponent_set)
            if state is None:
                raise RuntimeError("snapshot_to_state returned None for MCTS")

            print(getattr(getattr(state, "side_one", None), "force_switch", None))

            print(
                f"[mcts] state_built side_one_active={getattr(getattr(state, 'side_one', None), 'active_index', None)} "
                f"side_two_active={getattr(getattr(state, 'side_two', None), 'active_index', None)} weather={getattr(state, 'weather', None)} "
                f"trick_room={getattr(state, 'trick_room', None)}"
            )

            print(f"[mcts] calling python_mcts duration_ms={self.duration_ms}")
            result = monte_carlo_tree_search(state, duration_ms=self.duration_ms)
            print(f"[mcts] search_done side_one_results={len(getattr(result, 'side_one', []) or [])}")
            if logging.getLogger().isEnabledFor(logging.INFO):
                print(f"[mcts] top_choices={_top_entries(result)}")

            mcts_results.append((result, 1 / len(sampled_opponent_sets), index))

        if not mcts_results:
            raise RuntimeError("monte_carlo_tree_search returned no side_one results")

        choice = select_move_from_mcts_results(mcts_results)
        matched_action = _match_available_action(snapshot, choice)
        if matched_action is None and normalize_name(choice) == "nomove":
            print("[mcts] selected No Move; falling back to heuristic action", flush=True)
            return _HEURISTIC_STRATEGY.choose_action(snapshot, rng)
        if matched_action is None:
            raise RuntimeError(
                f"MCTS selected move {choice!r} but it does not match any available action"
            )

        if logging.getLogger().isEnabledFor(logging.INFO) and matched_action.action_type == ActionType.SWITCH:
            print(
                f"[mcts] switch_choice_on_normal_turn best_move={choice!r} "
                f"force_switch={getattr(getattr(state, 'side_one', None), 'force_switch', None)}",
                flush=True,
            )

        print(f"[mcts] selected_action={matched_action.command} label={matched_action.label}")
        return matched_action


MctsPlaceholderStrategy = MctsStrategy