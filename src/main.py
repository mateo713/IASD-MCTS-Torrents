from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from config import AppConfig
from experiments.showdown import (
    get_human_session_username,
    resolve_strategy_selection,
    run_round_robin_tournament,
    run_showdown_match,
)
from strategies import get_strategy_factory


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pokemon-Battler MVP runner")
    parser.add_argument(
        "-m",
        "--mode",
        default="human",
        choices=["human", "match", "tournament"],
        help="Execution mode: human (Showdown UI), one bot match, or round robin",
    )
    parser.add_argument(
        "-u",
        "--username",
        type=str,
        default=None,
        help="Bot username used for human mode and as the base for bot-vs-bot sessions",
    )
    parser.add_argument(
        "-p",
        "--password",
        type=str,
        default=None,
        help="Optional Showdown password",
    )
    parser.add_argument(
        "-w",
        "--websocket-uri",
        type=str,
        default=None,
        help="Showdown websocket URI",
    )
    parser.add_argument(
        "-f",
        "--battle-format",
        type=str,
        default=None,
        help="Battle format, default gen5randombattle",
    )
    parser.add_argument(
        "-e",
        "--seed",
        type=int,
        default=None,
        help="Random seed used by the strategy RNG",
    )
    parser.add_argument(
        "-s",
        "--strategy",
        type=str,
        default="h",
        help="Strategy for human mode; accepts canonical or short names",
    )
    parser.add_argument(
        "-j",
        "--team",
        type=str,
        default=None,
        help="Optional packed team for human mode",
    )
    parser.add_argument(
        "-a",
        "--strategy-a",
        type=str,
        default="r",
        help="First strategy name for match mode",
    )
    parser.add_argument(
        "-b",
        "--strategy-b",
        type=str,
        default="f",
        help="Second strategy name for match mode",
    )
    parser.add_argument(
        "-o",
        "--team-one",
        type=str,
        default=None,
        help="Optional packed team for the first match participant",
    )
    parser.add_argument(
        "-q",
        "--team-two",
        type=str,
        default=None,
        help="Optional packed team for the second match participant",
    )
    parser.add_argument(
        "-l",
        "--strategies",
        type=str,
        default="",
        help="Comma-separated strategy names for tournament mode; default is all registered strategies",
    )
    parser.add_argument(
        "-n",
        "--matches-per-pair",
        type=int,
        default=10,
        help="Number of matches per pair in tournament mode",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Show verbose logging output",
    )
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plot tournament result matrix using matplotlib (default True)",
    )
    parser.add_argument(
        "-k",
        "--max-concurrent-matches",
        type=int,
        default=25,
        help="Maximum number of matches executed concurrently in tournament mode",
    )
    parser.add_argument(
        "-c",
        "--max-battles-per-bot",
        type=int,
        default=5,
        help="Maximum number of simultaneous battles any one strategy can run",
    )
    parser.add_argument(
        "--show-constraints",
        action="store_true",
        default=False,
        help="Human mode: print inferred opponent constraints over time",
    )
    return parser


async def _run_human_mode(config: AppConfig) -> None:
    from experiments.human_play import run_human_vs_bot

    bot_username = get_human_session_username(config)
    print(f"Bot is online as: {bot_username}", flush=True)
    print(f"Challenge format: {config.battle_format}", flush=True)
    print("From your Showdown account, run: /challenge <bot-name>, <format>", flush=True)

    summary = await run_human_vs_bot(config, get_strategy_factory(config.strategy_name), team=config.team)
    print(json.dumps(summary.as_dict(), indent=2, sort_keys=True), flush=True)


async def _run_match_mode(
    config: AppConfig,
    strategy_a_name: str | None,
    strategy_b_name: str | None,
    team_one: str | None,
    team_two: str | None,
) -> None:
    strategy_classes = resolve_strategy_selection([
        strategy_a_name,
        strategy_b_name,
    ])
    outcome = await run_showdown_match(
        strategy_classes[0],
        strategy_classes[1],
        config=config,
        team_one=team_one,
        team_two=team_two,
    )
    print(json.dumps(outcome.as_dict(), indent=2, sort_keys=True), flush=True)


async def _run_tournament_mode(
    config: AppConfig,
    strategy_names: str | None,
    matches_per_pair: int,
    max_concurrent_matches: int,
    max_battles_per_bot: int,
) -> object:
    selected = resolve_strategy_selection(
        [name.strip() for name in strategy_names.split(",") if name.strip()]
        if strategy_names
        else None
    )
    summary = await run_round_robin_tournament(
        strategy_factories=selected,
        matches_per_pair=matches_per_pair,
        config=config,
        max_parallel_matches=max_concurrent_matches,
        max_parallel_battles_per_bot=max_battles_per_bot,
    )
    return summary


def main() -> None:
    args = _build_parser().parse_args()
    config = AppConfig.from_env()
    # Configure logging only when verbose was requested. Otherwise keep the
    # process quiet and only emit the final JSON results to stdout.
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    else:
        logging.disable(logging.CRITICAL)
    config.strategy_name = args.strategy
    if args.username is not None:
        config.username = args.username
    if args.password is not None:
        config.password = args.password
    if args.websocket_uri is not None:
        config.websocket_uri = args.websocket_uri
    if args.battle_format is not None:
        config.battle_format = args.battle_format
    if args.seed is not None:
        config.rng_seed = args.seed
    config.show_constraints = bool(args.show_constraints)

    if args.mode == "human":
        asyncio.run(_run_human_mode(config))
        return

    if args.mode == "match":
        asyncio.run(_run_match_mode(config, args.strategy_a, args.strategy_b, args.team_one, args.team_two))
        return

    if args.mode == "tournament":
        summary = asyncio.run(
            _run_tournament_mode(
                config,
                args.strategies,
                args.matches_per_pair,
                args.max_concurrent_matches,
                args.max_battles_per_bot,
            )
        )
        print(json.dumps(summary.as_dict(), indent=2, sort_keys=True), flush=True)
        if args.plot:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
            except Exception as exc:  # pragma: no cover - best-effort reporting
                print(f"Plot requested but matplotlib not available: {exc}", flush=True)
            else:
                selected = resolve_strategy_selection(
                    [name.strip() for name in args.strategies.split(",") if name.strip()]
                    if args.strategies
                    else None
                )
                labels = [factory().name for factory in selected]
                n = len(labels)
                if n > 0:
                    mat = [[0] * n for _ in range(n)]
                    for m in summary.matchups:
                        if m.strategy_a in labels and m.strategy_b in labels:
                            i = labels.index(m.strategy_a)
                            j = labels.index(m.strategy_b)
                            mat[i][j] = m.wins_a
                            mat[j][i] = m.wins_b

                    maxval = max((max(row) for row in mat), default=0)
                    fig, ax = plt.subplots(figsize=(max(6, n), max(6, n)))
                    cax = ax.imshow(mat, interpolation="nearest", cmap="Blues")
                    ax.set_xticks(list(range(n)))
                    ax.set_yticks(list(range(n)))
                    ax.set_xticklabels(labels, rotation=45, ha="right")
                    ax.set_yticklabels(labels)
                    for i in range(n):
                        for j in range(n):
                            color = "white" if (maxval > 0 and mat[i][j] > maxval / 2) else "black"
                            ax.text(j, i, str(mat[i][j]), ha="center", va="center", color=color)
                    fig.colorbar(cax)
                    fig.tight_layout()
                    output_path = Path(__file__).resolve().parents[1] / "results" / "tournament_matrix.png"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    plt.savefig(output_path)
                    plt.close(fig)
                    print(f"Saved tournament heatmap to {output_path}", flush=True)
        return


if __name__ == "__main__":
    main()