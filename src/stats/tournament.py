from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class BattleOutcome:
    strategy_one: str
    strategy_two: str
    winner: str | None
    winner_strategy: str | None
    room_id: str
    turns: int

    def as_dict(self) -> dict[str, int | str | None]:
        return {
            "strategy_one": self.strategy_one,
            "strategy_two": self.strategy_two,
            "winner": self.winner,
            "winner_strategy": self.winner_strategy,
            "room_id": self.room_id,
            "turns": self.turns,
        }


@dataclass(slots=True)
class TournamentMatchupSummary:
    strategy_a: str
    strategy_b: str
    battles: int = 0
    wins_a: int = 0
    wins_b: int = 0
    ties: int = 0
    turns_total: int = 0

    @property
    def average_turns(self) -> float:
        return 0.0 if self.battles == 0 else self.turns_total / self.battles

    def record(self, outcome: BattleOutcome) -> None:
        self.battles += 1
        self.turns_total += outcome.turns

        if outcome.winner_strategy is None:
            self.ties += 1
        elif outcome.winner_strategy == self.strategy_a:
            self.wins_a += 1
        elif outcome.winner_strategy == self.strategy_b:
            self.wins_b += 1

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "strategy_a": self.strategy_a,
            "strategy_b": self.strategy_b,
            "battles": self.battles,
            "wins_a": self.wins_a,
            "wins_b": self.wins_b,
            "ties": self.ties,
            "average_turns": round(self.average_turns, 2),
        }


@dataclass(slots=True)
class TournamentSummary:
    battle_format: str
    matchups: list[TournamentMatchupSummary] = field(default_factory=list)

    @property
    def battles(self) -> int:
        return sum(matchup.battles for matchup in self.matchups)

    @property
    def wins(self) -> int:
        return sum(matchup.wins_a for matchup in self.matchups)

    @property
    def losses(self) -> int:
        return sum(matchup.wins_b for matchup in self.matchups)

    @property
    def ties(self) -> int:
        return sum(matchup.ties for matchup in self.matchups)

    @property
    def turns_total(self) -> int:
        return sum(matchup.turns_total for matchup in self.matchups)

    @property
    def average_turns(self) -> float:
        return 0.0 if self.battles == 0 else self.turns_total / self.battles

    def as_dict(self) -> dict[str, object]:
        return {
            "battle_format": self.battle_format,
            "battles": self.battles,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "average_turns": round(self.average_turns, 2),
            "matchups": [matchup.as_dict() for matchup in self.matchups],
        }
