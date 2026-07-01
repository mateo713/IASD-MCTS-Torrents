from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
import time
from typing import Any, Callable, Generic, TypeVar

from strategies.evaluation import evaluate_state


StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")


@dataclass(slots=True)
class TransitionBranch(Generic[StateT]):
    state: StateT
    weight: float = 1.0


@dataclass(slots=True)
class MoveStats:
    total_score: float = 0.0
    visits: int = 0

    def average_score(self) -> float:
        if self.visits <= 0:
            return float("-inf")
        return self.total_score / self.visits


@dataclass(slots=True)
class PythonMctsNode(Generic[StateT, ActionT]):
    state: StateT
    depth: int = 0
    parent: PythonMctsNode[StateT, ActionT] | None = None
    parent_own_index: int | None = None
    parent_opp_index: int | None = None
    own_actions: list[ActionT] = field(default_factory=list)
    opp_actions: list[ActionT] = field(default_factory=list)
    own_stats: list[MoveStats] = field(default_factory=list)
    opp_stats: list[MoveStats] = field(default_factory=list)
    children: dict[tuple[int, int], list[PythonMctsNode[StateT, ActionT]]] = field(default_factory=dict)
    visits: int = 0

    def ensure_actions(
        self,
        legal_action_provider: Callable[[StateT], tuple[list[ActionT], list[ActionT]]],
    ) -> None:
        if self.own_actions and self.opp_actions:
            return
        own_actions, opp_actions = legal_action_provider(self.state)
        self.own_actions = list(own_actions)
        self.opp_actions = list(opp_actions)
        self.own_stats = [MoveStats() for _ in self.own_actions]
        self.opp_stats = [MoveStats() for _ in self.opp_actions]


LegalActionProvider = Callable[[StateT], tuple[list[ActionT], list[ActionT]]]
TransitionProvider = Callable[[StateT, ActionT, ActionT], list[TransitionBranch[StateT]]]
StateEvaluator = Callable[[StateT], float]


@dataclass(slots=True)
class PythonMctsConfig:
    duration_ms: int = 500
    max_depth: int = 4
    exploration_constant: float = 2.0
    min_branch_weight: float = 0.0

class PythonMctsSearch(Generic[StateT, ActionT]):
    def __init__(
        self,
        *,
        root_state: StateT,
        legal_action_provider: LegalActionProvider[StateT, ActionT],
        transition_provider: TransitionProvider[StateT, ActionT],
        state_evaluator: StateEvaluator[StateT],
        rng: random.Random | None = None,
        config: PythonMctsConfig | None = None,
    ) -> None:
        self.root = PythonMctsNode(root_state)
        self.legal_action_provider = legal_action_provider
        self.transition_provider = transition_provider
        self.state_evaluator = state_evaluator
        self.rng = rng or random.Random()
        self.config = config or PythonMctsConfig()
        self._root_value = self.state_evaluator(root_state)

    def run(self) -> tuple[ActionT, dict[str, Any]]:
        self.root.ensure_actions(self.legal_action_provider)
        if not self.root.own_actions:
            raise RuntimeError("PythonMctsSearch received no legal own actions")

        start = time.monotonic()
        deadline = start + max(0.0, self.config.duration_ms / 1000.0)
        iterations = 0
        while time.monotonic() < deadline:
            self._iteration()
            iterations += 1

        best_action = self._best_root_action()
        return best_action, self.root_policy_summary(iterations)

    def _iteration(self) -> None:
        node = self.root
        path: list[tuple[PythonMctsNode[StateT, ActionT], int, int]] = []

        while True:
            node.ensure_actions(self.legal_action_provider)
            if not node.own_actions or not node.opp_actions:
                break

            own_index = self._select_index(node.own_stats, node.visits)
            opp_index = self._select_index(node.opp_stats, node.visits)
            path.append((node, own_index, opp_index))

            child_key = (own_index, opp_index)
            child_list = node.children.get(child_key)
            if child_list:
                node = self._sample_branch(child_list)
                continue

            branches = self.transition_provider(
                node.state,
                node.own_actions[own_index],
                node.opp_actions[opp_index],
            )
            branches = [branch for branch in branches if branch.weight > self.config.min_branch_weight]
            if not branches:
                break

            child_list = []
            for branch in branches:
                child = PythonMctsNode(
                    state=branch.state,
                    depth=node.depth + 1,
                    parent=node,
                    parent_own_index=own_index,
                    parent_opp_index=opp_index,
                )
                child_list.append(child)

            node.children[child_key] = child_list
            node = self._sample_branch(child_list, branches)

            if node.depth >= self.config.max_depth:
                break

        score = self.state_evaluator(node.state)
        normalized_score = self._normalize_score(score)
        self._backpropagate(path, normalized_score)

    def _select_index(self, stats: list[MoveStats], parent_visits: int) -> int:
        for index, stat in enumerate(stats):
            if stat.visits == 0:
                return index

        best_index = 0
        best_value = float("-inf")
        exploration_scale = self.config.exploration_constant * math.sqrt(max(1.0, math.log(parent_visits + 1.0)))
        for index, stat in enumerate(stats):
            exploitation = stat.average_score()
            exploration = exploration_scale / math.sqrt(stat.visits)
            value = exploitation + exploration
            if value > best_value:
                best_value = value
                best_index = index
        return best_index

    def _sample_branch(
        self,
        branches: list[PythonMctsNode[StateT, ActionT]],
        raw_branches: list[TransitionBranch[StateT]] | None = None,
    ) -> PythonMctsNode[StateT, ActionT]:
        if not branches:
            raise RuntimeError("Cannot sample from an empty branch list")
        if raw_branches is None:
            weights = [1.0 for _ in branches]
        else:
            weights = [max(0.0, branch.weight) for branch in raw_branches]
            if not any(weight > 0 for weight in weights):
                weights = [1.0 for _ in branches]
        return self.rng.choices(branches, weights=weights, k=1)[0]

    def _backpropagate(self, path: list[tuple[PythonMctsNode[StateT, ActionT], int, int]], score: float) -> None:
        score = max(0.0, min(1.0, score))
        opponent_score = 1.0 - score
        for node, own_index, opp_index in path:
            node.visits += 1
            node.own_stats[own_index].visits += 1
            node.own_stats[own_index].total_score += score
            node.opp_stats[opp_index].visits += 1
            node.opp_stats[opp_index].total_score += opponent_score

    def _normalize_score(self, score: float) -> float:
        return 1.0 / (1.0 + math.exp(-0.0125 * (score - self._root_value)))

    def _best_root_action(self) -> ActionT:
        best_action = self.root.own_actions[0]
        best_score = float("-inf")
        for action, stat in zip(self.root.own_actions, self.root.own_stats, strict=False):
            score = stat.average_score()
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def root_policy_summary(self, iterations: int) -> dict[str, Any]:
        return {
            "iterations": iterations,
            "own_actions": [
                {
                    "action": action,
                    "visits": stat.visits,
                    "average_score": stat.average_score(),
                }
                for action, stat in zip(self.root.own_actions, self.root.own_stats, strict=False)
            ],
        }


def simple_state_evaluator(state: Any) -> float:
    """Best-effort Python port of poke-engine's genx evaluator."""

    return evaluate_state(state)
