from __future__ import annotations

import random

from core.models import ActionChoice, BattleSnapshot
from engine.inference import split_action_space
from strategies.base import Strategy
from strategies.base import fallback_action


class ReasoningPlaceholderStrategy(Strategy):
    name = "reasoning_placeholder"

    def choose_action(self, snapshot: BattleSnapshot, rng: random.Random) -> ActionChoice:
        if not snapshot.available_actions:
            return fallback_action(snapshot)

        moves, switches, others = split_action_space(snapshot)
        if moves:
            return rng.choice(moves)
        if switches:
            return rng.choice(switches)
        if others:
            return others[0]
        return fallback_action(snapshot)
