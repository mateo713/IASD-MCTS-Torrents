from __future__ import annotations

from config import AppConfig
from stats.summary import MatchSummary
from strategies.base import StrategyFactory


async def run_human_vs_bot(
    config: AppConfig,
    strategy_factory: StrategyFactory,
    team: str | None = None,
) -> MatchSummary:
    """Run one bot session so a human can challenge it from Showdown UI."""
    from experiments.showdown import run_human_session

    session_result = await run_human_session(config, strategy_factory, team=team)
    summary = MatchSummary(battles=1, turns_total=int(session_result.get("turns", 0) or 0))
    if session_result.get("winner") == config.username:
        summary.wins = 1
    elif session_result.get("winner"):
        summary.losses = 1
    else:
        summary.ties = 1
    return summary
