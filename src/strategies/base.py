from __future__ import annotations

from abc import ABC, abstractmethod
import random
from typing import Callable

from core.models import ActionChoice, ActionType, BattleSnapshot


class Strategy(ABC):
    name: str = "strategy"

    @abstractmethod
    def choose_action(self, snapshot: BattleSnapshot, rng: random.Random) -> ActionChoice:
        raise NotImplementedError


def fallback_action(snapshot: BattleSnapshot) -> ActionChoice:
    if snapshot.available_actions:
        return snapshot.available_actions[0]
    return ActionChoice(ActionType.PASS, "/choose default", "default")


StrategyFactory = Callable[[], Strategy]
