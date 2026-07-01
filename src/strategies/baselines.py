from __future__ import annotations

import random

from core.models import ActionChoice, BattleSnapshot
from engine.inference import split_action_space
from strategies.base import Strategy
from strategies.base import fallback_action


class RandomUniformStrategy(Strategy):
    name = "random_uniform"

    def __init__(self, switch_probability: float = 0.2) -> None:
        self.switch_probability = min(1.0, max(0.0, switch_probability))

    def choose_action(self, snapshot: BattleSnapshot, rng: random.Random) -> ActionChoice:
        if not snapshot.available_actions:
            return fallback_action(snapshot)

        moves, switches, others = split_action_space(snapshot)

        if moves and switches:
            action_pool = switches if rng.random() < self.switch_probability else moves
            return rng.choice(action_pool)
        if moves:
            return rng.choice(moves)
        if switches:
            return rng.choice(switches)
        if others:
            return rng.choice(others)

        return fallback_action(snapshot)


class FirstLegalStrategy(Strategy):
    name = "first_legal"

    def choose_action(self, snapshot: BattleSnapshot, rng: random.Random) -> ActionChoice:
        if not snapshot.available_actions:
            return fallback_action(snapshot)

        moves, switches, others = split_action_space(snapshot)
        if moves:
            return moves[min(2, len(moves) - 1)]
        if switches:
            return switches[0]
        if others:
            return others[0]

        return fallback_action(snapshot)
