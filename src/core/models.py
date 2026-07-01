from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.battle_state import BattleState


class BattlePhase(str, Enum):
    TEAM_PREVIEW = "team_preview"
    IN_BATTLE = "in_battle"
    FINISHED = "finished"


class ActionType(str, Enum):
    MOVE = "move"
    SWITCH = "switch"
    TEAM_PREVIEW = "team_preview"
    PASS = "pass"


@dataclass(slots=True)
class ActionChoice:
    action_type: ActionType
    command: str
    label: str


@dataclass(slots=True)
class OpponentConstraints:
    impossible_items: set[str] = field(default_factory=set)
    impossible_abilities: set[str] = field(default_factory=set)
    revealed_ability: str | None = None
    revealed_item: str | None = None
    revealed_moves: set[str] = field(default_factory=set)
    moves_used_since_switch_in: set[str] = field(default_factory=set)
    observed_immune_move_types: set[str] = field(default_factory=set)
    last_opponent_move_id: str | None = None
    observed_damage_source_move_id: str | None = None
    observed_damage_source_is_opponent: bool | None = None
    observed_damage_fraction: float | None = None
    observed_damage_was_crit: bool = False
    observed_damage_turn: int | None = None
    speed_stage: int = 0
    is_paralyzed: bool = False
    unburden_triggered: bool = False
    inferred_choice_scarf: bool = False
    active_stint_index: int = 0
    has_revealed_choice_conflict: bool = False


@dataclass(slots=True)
class BattleSnapshot:
    room_id: str
    turn: int = 0
    phase: BattlePhase = BattlePhase.TEAM_PREVIEW
    request_id: int | None = None
    available_actions: list[ActionChoice] = field(default_factory=list)
    last_request: dict[str, Any] | None = None
    winner: str | None = None
    own_side_id: str | None = None
    active_species: str | None = None
    opponent_active_species: str | None = None
    own_active_hp_fraction: float | None = None
    battle_state: BattleState | None = None
    opponent_constraints: OpponentConstraints = field(default_factory=OpponentConstraints)

    @property
    def awaiting_action(self) -> bool:
        return self.phase != BattlePhase.FINISHED and bool(self.available_actions)
