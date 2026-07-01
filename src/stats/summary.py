from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class MatchSummary:
    battles: int = 0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    turns_total: int = 0

    @property
    def average_turns(self) -> float:
        return 0.0 if self.battles == 0 else self.turns_total / self.battles

    def as_dict(self) -> dict[str, float | int]:
        return {
            "battles": self.battles,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "average_turns": round(self.average_turns, 2),
        }
