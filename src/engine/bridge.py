from __future__ import annotations

import random
from typing import Any

try:
    from poke_engine import State
except Exception:  # pragma: no cover - depends on local environment
    State = None

from core.models import BattleSnapshot
from engine.gen5_datasets import build_poke_engine_state_from_snapshot


class PokeEngineBridge:
    """Bridge surface for converting tracked battle snapshots to poke-engine State."""

    def snapshot_to_state(
        self,
        snapshot: BattleSnapshot,
        rng: random.Random | None = None,
        opponent_set_override: dict[str, Any] | None = None,
    ) -> Any:
        if State is None:
            return None
        return build_poke_engine_state_from_snapshot(
            snapshot,
            rng=rng,
            opponent_set_override=opponent_set_override,
        )
