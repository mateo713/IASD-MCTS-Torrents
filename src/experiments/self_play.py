from __future__ import annotations

from config import AppConfig
from stats.tournament import BattleOutcome, TournamentSummary
from strategies.base import Strategy


async def run_showdown_match(
    strategy_one_class: type[Strategy],
    strategy_two_class: type[Strategy],
    config: AppConfig | None = None,
    client_factory=None,
) -> BattleOutcome:
    from experiments.showdown import run_showdown_match as _run_showdown_match

    return await _run_showdown_match(
        strategy_one_class,
        strategy_two_class,
        config=config,
        client_factory=client_factory,
    )


async def run_round_robin_tournament(
    strategy_classes: list[type[Strategy]] | None = None,
    matches_per_pair: int = 10,
    config: AppConfig | None = None,
    client_factory=None,
) -> TournamentSummary:
    from experiments.showdown import run_round_robin_tournament as _run_round_robin_tournament

    return await _run_round_robin_tournament(
        strategy_classes=strategy_classes,
        matches_per_pair=matches_per_pair,
        config=config,
        client_factory=client_factory,
    )
