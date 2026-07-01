from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from itertools import combinations
from typing import Callable

from config import AppConfig
from core.models import BattlePhase
from engine.bridge import PokeEngineBridge
from infrastructure.showdown_client import ShowdownClient, ShowdownCredentials
from infrastructure.showdown_parser import parse_showdown_message
from infrastructure.state_tracker import ShowdownStateTracker
from stats.tournament import BattleOutcome, TournamentMatchupSummary, TournamentSummary
from strategies import (
    ALL_STRATEGY_FACTORIES,
    Strategy,
    StrategyFactory,
    get_strategy_factory,
    get_strategy_short_name,
)
from strategies.no_move_debug import dump_strategy_input_context


_REQUEST_STATE_DUMP_SAMPLE_RATE = 0.05


def _build_sampled_engine_state(snapshot) -> object | None:
    try:
        bridge = PokeEngineBridge()
        dump_rng = random.Random(f"{snapshot.room_id}:{snapshot.turn}:sampled_dump")
        return bridge.snapshot_to_state(snapshot, rng=dump_rng)
    except Exception:
        return None


@dataclass(slots=True)
class _ParticipantContext:
    username: str
    strategy: Strategy
    client: ShowdownClient
    tracker: ShowdownStateTracker
    rng: random.Random
    accepted_challenge: bool = False


def _to_base36(value: int) -> str:
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    if value == 0:
        return "0"
    encoded = ""
    current = value
    while current > 0:
        current, remainder = divmod(current, 36)
        encoded = digits[remainder] + encoded
    return encoded


def _build_showdown_username(base_username: str, strategy_tag: str | None = None, side_tag: str | None = None) -> str:
    suffix_parts = [part for part in [strategy_tag, side_tag] if part]
    suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
    max_base_length = max(1, 18 - len(suffix))
    return f"{base_username[:max_base_length]}{suffix}"


def get_human_session_username(config: AppConfig) -> str:
    return _build_showdown_username(config.username or "mcts_bot")


async def _participant_loop(
    context: _ParticipantContext,
    finish_event: asyncio.Event,
    battle_format: str,
    result_holder: dict[str, object],
    accept_challenges: bool,
) -> None:
    while not finish_event.is_set():
        message = await context.client.recv()
        pending_choices: dict[str, tuple[str, int | None]] = {}
        for event in parse_showdown_message(message):
            if accept_challenges and not context.accepted_challenge:
                challenger = _maybe_accept_challenge(
                    event.raw_line, context.client.credentials.username, battle_format
                )
                if challenger is not None:
                    await context.client.accept_challenge(challenger)
                    context.accepted_challenge = True
                    continue

            snapshot = context.tracker.ingest(event)
            if event.event_type == "request" and logging.getLogger().isEnabledFor(logging.INFO):
                logging.info(_request_debug_line(snapshot))
            if event.event_type == "request" and context.rng.random() < _REQUEST_STATE_DUMP_SAMPLE_RATE:
                sampled_engine_state = _build_sampled_engine_state(snapshot)
                sampled_debug_path = dump_strategy_input_context(
                    strategy_name=context.strategy.name,
                    snapshot=snapshot,
                    engine_state=sampled_engine_state,
                    candidate_set=None,
                    result=None,
                    label="sampled_input",
                )
                logging.info(
                    "[request-sampled-dump] room=%s turn=%s path=%s",
                    snapshot.room_id,
                    snapshot.turn,
                    sampled_debug_path,
                )
            if event.event_type == "request" and logging.getLogger().isEnabledFor(logging.INFO):
                verbose_engine_state = _build_sampled_engine_state(snapshot)
                debug_path = dump_strategy_input_context(
                    strategy_name=context.strategy.name,
                    snapshot=snapshot,
                    engine_state=verbose_engine_state,
                    candidate_set=None,
                    result=None,
                    label="fed_input",
                )
                logging.info("[request-debug-dump] room=%s turn=%s path=%s", snapshot.room_id, snapshot.turn, debug_path)
            if event.event_type == "request" and snapshot.awaiting_action:
                pending_choices[snapshot.room_id] = (snapshot.room_id, snapshot.request_id)

            if snapshot.phase == BattlePhase.FINISHED:
                if snapshot.room_id and "outcome" not in result_holder:
                    result_holder["outcome"] = BattleOutcome(
                        strategy_one=str(result_holder.get("strategy_one", context.strategy.name)),
                        strategy_two=str(result_holder.get("strategy_two", context.strategy.name)),
                        winner=snapshot.winner,
                        winner_strategy=_winner_strategy_name(
                            snapshot.winner,
                            result_holder.get("username_one"),
                            result_holder.get("username_two"),
                            result_holder.get("strategy_one"),
                            result_holder.get("strategy_two"),
                        ),
                        room_id=snapshot.room_id,
                        turns=snapshot.turn,
                    )
                    result_holder["room_id"] = snapshot.room_id
                    result_holder["winner"] = snapshot.winner
                    result_holder["winner_strategy"] = result_holder["outcome"].winner_strategy
                    result_holder["turns"] = snapshot.turn
                finish_event.set()
                return

        for room_id, (_, request_id) in pending_choices.items():
            snapshot = context.tracker.get_or_create(room_id)
            if not snapshot.awaiting_action:
                continue
            try:
                selected_action = context.strategy.choose_action(snapshot, context.rng)
                await context.client.choose(
                    snapshot.room_id,
                    selected_action.command,
                    request_id,
                )
            except Exception:
                logging.exception(
                    "Strategy %s failed while choosing an action (room=%s, request_id=%s, turn=%s)",
                    context.strategy.name,
                    snapshot.room_id,
                    request_id,
                    snapshot.turn,
                )
                raise


def _maybe_accept_challenge(raw_line: str, username: str, battle_format: str) -> str | None:
    if not raw_line.startswith("|pm|"):
        return None

    parts = raw_line.split("|")
    if len(parts) < 5:
        return None

    sender = parts[2].strip()
    recipient = parts[3].strip().replace("!", "").replace("‽", "")
    challenge_text = parts[4].strip()

    if recipient != username:
        return None
    if not challenge_text.startswith("/challenge"):
        return None
    if battle_format not in challenge_text:
        return None

    return sender


def _winner_strategy_name(
    winner_username: str | None,
    username_one: str | None,
    username_two: str | None,
    strategy_one_name: str | None,
    strategy_two_name: str | None,
) -> str | None:
    if winner_username is None:
        return None
    if winner_username == username_one:
        return strategy_one_name
    if winner_username == username_two:
        return strategy_two_name
    return None


def _request_debug_line(snapshot) -> str:
    battle_state = snapshot.battle_state
    active_pokemon = None
    active_hp = None
    active_status = None
    request_force_switch = None
    if battle_state is not None:
        active_pokemon = battle_state.active_own_pokemon
        if active_pokemon is not None:
            active_hp = getattr(active_pokemon, "current_hp", None)
            active_status = getattr(active_pokemon, "status", None)
        request_force_switch = getattr(battle_state, "request_force_switch", None)
    return (
        "[request-debug] "
        f"room={snapshot.room_id} turn={snapshot.turn} awaiting={snapshot.awaiting_action} "
        f"req_force_switch={request_force_switch} active_hp={active_hp} active_status={active_status} "
        f"own_actions={len(snapshot.available_actions)} own_species={snapshot.active_species}"
    )


def _constraint_signature(snapshot) -> tuple[object, ...]:
    c = snapshot.opponent_constraints
    return (
        snapshot.turn,
        snapshot.opponent_active_species,
        c.revealed_ability,
        c.revealed_item,
        tuple(sorted(c.revealed_moves)),
        tuple(sorted(c.moves_used_since_switch_in)),
        tuple(sorted(c.impossible_items)),
        tuple(sorted(c.impossible_abilities)),
        tuple(sorted(c.observed_immune_move_types)),
        c.last_opponent_move_id,
        c.observed_damage_fraction,
        c.observed_damage_was_crit,
        c.speed_stage,
        c.is_paralyzed,
        c.unburden_triggered,
        c.inferred_choice_scarf,
        c.active_stint_index,
    )


def _format_constraint_line(snapshot) -> str:
    c = snapshot.opponent_constraints
    parts: list[str] = [
        f"turn={snapshot.turn}",
        f"opp={snapshot.opponent_active_species or 'unknown'}",
    ]
    if c.revealed_ability:
        parts.append(f"ability={c.revealed_ability}")
    if c.revealed_item:
        parts.append(f"item={c.revealed_item}")
    if c.inferred_choice_scarf and c.revealed_item != "choicescarf":
        parts.append("item~=choicescarf")
    if c.revealed_moves:
        parts.append(f"moves={','.join(sorted(c.revealed_moves))}")
    if c.observed_immune_move_types:
        parts.append(f"immune_to={','.join(sorted(c.observed_immune_move_types))}")
    if c.impossible_items:
        parts.append(f"impossible_items={','.join(sorted(c.impossible_items))}")
    if c.observed_damage_fraction is not None:
        pct = round(c.observed_damage_fraction * 100.0, 1)
        crit_label = " crit" if c.observed_damage_was_crit else ""
        parts.append(f"obs_dmg={pct}%{crit_label}")
    if c.speed_stage:
        parts.append(f"spe_stage={c.speed_stage:+d}")
    if c.is_paralyzed:
        parts.append("status=par")
    if c.unburden_triggered:
        parts.append("unburden=on")
    return "[constraints] " + " | ".join(parts)


async def _connect_and_login_with_retry(
    client: ShowdownClient,
    attempts: int = 5,
    initial_backoff_seconds: float = 1.0,
) -> None:
    last_error: Exception | None = None
    # Timeouts used to bound connect/login attempts so a hung server doesn't stall the
    # whole tournament. Add a small jitter to avoid synchronized retries.
    connect_timeout = 20.0
    login_timeout = 20.0
    for attempt in range(attempts):
        try:
            await asyncio.wait_for(client.connect(), timeout=connect_timeout)
            await asyncio.wait_for(client.login(), timeout=login_timeout)
            return
        except Exception as error:
            last_error = error
            try:
                await asyncio.wait_for(client.close(), timeout=3.0)
            except Exception:
                # best-effort close; ignore errors during cleanup
                pass
            if attempt == attempts - 1:
                break
            # exponential backoff with jitter to avoid retry storms
            backoff = initial_backoff_seconds * (2 ** attempt)
            jitter = random.uniform(0, initial_backoff_seconds)
            await asyncio.sleep(backoff + jitter)
    if last_error is not None:
        raise last_error
    raise RuntimeError("Connection/login failed without an explicit exception")


async def run_showdown_match(
    strategy_one_factory: StrategyFactory,
    strategy_two_factory: StrategyFactory,
    config: AppConfig | None = None,
    client_factory: Callable[[str, ShowdownCredentials], ShowdownClient] = ShowdownClient,
    team_one: str | None = None,
    team_two: str | None = None,
    session_tag: str | None = None,
) -> BattleOutcome:
    config = config or AppConfig.from_env()
    battle_format = config.battle_format

    strategy_one = strategy_one_factory()
    strategy_two = strategy_two_factory()
    base_username = config.username or "mcts_bot"
    username_one = _build_showdown_username(
        base_username,
        get_strategy_short_name(strategy_one_factory),
        f"{session_tag}a" if session_tag else "a",
    )
    username_two = _build_showdown_username(
        base_username,
        get_strategy_short_name(strategy_two_factory),
        f"{session_tag}b" if session_tag else "b",
    )

    client_one = client_factory(
        config.websocket_uri,
        ShowdownCredentials(username_one, config.password),
    )
    client_two = client_factory(
        config.websocket_uri,
        ShowdownCredentials(username_two, config.password),
    )

    # Per-match timeout (seconds). If a match doesn't complete within this time
    # we'll cancel participant tasks and return a timeout outcome. This prevents
    # individual stalled matches from freezing the whole tournament. Set very
    # high so long-running battles (100+ turns) are not cut short.
    # Default: 1 hour.
    DEFAULT_MATCH_TIMEOUT = 1800  # 86400 seconds
    try:
        logging.info("Starting match %s: %s vs %s", session_tag, username_one, username_two)
        await _connect_and_login_with_retry(client_one)
        await _connect_and_login_with_retry(client_two)

        if team_one is not None:
            await client_one.update_team(team_one)
        if team_two is not None:
            await client_two.update_team(team_two)

        context_one = _ParticipantContext(
            username=username_one,
            strategy=strategy_one,
            client=client_one,
            tracker=ShowdownStateTracker(),
            rng=random.Random(config.rng_seed),
        )
        context_two = _ParticipantContext(
            username=username_two,
            strategy=strategy_two,
            client=client_two,
            tracker=ShowdownStateTracker(),
            rng=random.Random(config.rng_seed + 1),
        )
        finish_event = asyncio.Event()
        result_holder: dict[str, object] = {
            "strategy_one": strategy_one.name,
            "strategy_two": strategy_two.name,
            "username_one": username_one,
            "username_two": username_two,
        }

        task_one = asyncio.create_task(
            _participant_loop(
                context_one,
                finish_event,
                battle_format,
                result_holder,
                accept_challenges=False,
            )
        )
        task_two = asyncio.create_task(
            _participant_loop(
                context_two,
                finish_event,
                battle_format,
                result_holder,
                accept_challenges=True,
            )
        )

        try:
            await client_one.challenge(username_two, battle_format)
            try:
                await asyncio.wait_for(finish_event.wait(), timeout=DEFAULT_MATCH_TIMEOUT)
            except asyncio.TimeoutError:
                logging.warning(
                    "Match timed out: %s vs %s (tag=%s)", username_one, username_two, session_tag
                )
                # Indicate timeout in result_holder and ensure participant loops exit
                result_holder["winner"] = None
                finish_event.set()
        finally:
            task_one.cancel()
            task_two.cancel()
            await asyncio.gather(task_one, task_two, return_exceptions=True)
            logging.info("Finished match %s: %s vs %s", session_tag, username_one, username_two)
    finally:
        # Close clients but don't block indefinitely on a hung close operation.
        try:
            await asyncio.wait_for(client_one.close(), timeout=5.0)
        except Exception:
            logging.exception("Error while closing client for %s", username_one)
        try:
            await asyncio.wait_for(client_two.close(), timeout=5.0)
        except Exception:
            logging.exception("Error while closing client for %s", username_two)

    outcome = result_holder.get("outcome")
    if isinstance(outcome, BattleOutcome):
        return outcome

    return BattleOutcome(
        strategy_one=strategy_one.name,
        strategy_two=strategy_two.name,
        winner=result_holder.get("winner") if isinstance(result_holder.get("winner"), str) else None,
        winner_strategy=result_holder.get("winner_strategy") if isinstance(result_holder.get("winner_strategy"), str) else None,
        room_id=str(result_holder.get("room_id", "")),
        turns=int(result_holder.get("turns", 0) or 0),
    )


async def run_human_session(
    config: AppConfig,
    strategy_factory: StrategyFactory,
    client_factory: Callable[[str, ShowdownCredentials], ShowdownClient] = ShowdownClient,
    team: str | None = None,
) -> dict[str, int | str]:
    client = client_factory(
        config.websocket_uri,
        ShowdownCredentials(get_human_session_username(config), config.password),
    )
    tracker = ShowdownStateTracker()
    rng = random.Random(config.rng_seed)
    finish_event = asyncio.Event()
    result: dict[str, int | str] = {"winner": "", "turns": 0, "room_id": ""}
    strategy = strategy_factory()
    last_signature: tuple[object, ...] | None = None

    await client.connect()
    await client.login()

    if team is not None:
        await client.update_team(team)

    try:
        while not finish_event.is_set():
            message = await client.recv()
            pending_choices: dict[str, tuple[str, int | None]] = {}
            for event in parse_showdown_message(message):
                challenge_sender = _maybe_accept_challenge(
                    event.raw_line, client.credentials.username, config.battle_format
                )
                if challenge_sender is not None:
                    await client.accept_challenge(challenge_sender)
                    continue

                snapshot = tracker.ingest(event)
                if config.show_constraints and snapshot.phase != BattlePhase.FINISHED:
                    signature = _constraint_signature(snapshot)
                    if signature != last_signature and snapshot.opponent_active_species:
                        print(_format_constraint_line(snapshot), flush=True)
                        last_signature = signature
                if event.event_type == "request" and logging.getLogger().isEnabledFor(logging.INFO):
                    logging.info(_request_debug_line(snapshot))
                if event.event_type == "request" and snapshot.awaiting_action:
                    pending_choices[snapshot.room_id] = (snapshot.room_id, snapshot.request_id)

                if snapshot.phase == BattlePhase.FINISHED:
                    result = {
                        "winner": snapshot.winner or "",
                        "turns": snapshot.turn,
                        "room_id": snapshot.room_id,
                    }
                    finish_event.set()
                    break
            if finish_event.is_set():
                break
            for room_id, (_, request_id) in pending_choices.items():
                snapshot = tracker.get_or_create(room_id)
                if not snapshot.awaiting_action:
                    continue
                try:
                    selected_action = strategy.choose_action(snapshot, rng)
                    await client.choose(
                        snapshot.room_id,
                        selected_action.command,
                        request_id,
                    )
                except Exception:
                    logging.exception(
                        "Strategy %s failed while choosing an action (room=%s, request_id=%s, turn=%s)",
                        strategy.name,
                        snapshot.room_id,
                        request_id,
                        snapshot.turn,
                    )
                    raise
    finally:
        await client.close()

    return result


async def run_round_robin_tournament(
    strategy_factories: list[StrategyFactory] | None = None,
    matches_per_pair: int = 10,
    config: AppConfig | None = None,
    client_factory: Callable[[str, ShowdownCredentials], ShowdownClient] = ShowdownClient,
    max_parallel_matches: int = 1,
    max_parallel_battles_per_bot: int = 1,
) -> TournamentSummary:
    config = config or AppConfig.from_env()
    selected_factories = strategy_factories or list(ALL_STRATEGY_FACTORIES)
    summary = TournamentSummary(battle_format=config.battle_format)

    max_parallel_matches = max(1, max_parallel_matches)
    max_parallel_battles_per_bot = max(1, max_parallel_battles_per_bot)

    matchup_by_pair: dict[tuple[StrategyFactory, StrategyFactory], TournamentMatchupSummary] = {}
    for strategy_a_factory, strategy_b_factory in combinations(selected_factories, 2):
        matchup = TournamentMatchupSummary(strategy_a=strategy_a_factory().name, strategy_b=strategy_b_factory().name)
        summary.matchups.append(matchup)
        matchup_by_pair[(strategy_a_factory, strategy_b_factory)] = matchup

    strategy_limits = {
        strategy_factory: asyncio.Semaphore(max_parallel_battles_per_bot)
        for strategy_factory in selected_factories
    }

    # Build the full list of jobs (pair, index, tag). We'll schedule workers equal
    # to `max_parallel_matches` to avoid creating thousands of tasks up-front which
    # can overwhelm the event loop and underlying resources.
    jobs: list[tuple[tuple[StrategyFactory, StrategyFactory], int, str]] = []
    match_counter = 0
    for pair in matchup_by_pair:
        for match_index in range(matches_per_pair):
            match_tag = _to_base36(match_counter)
            jobs.append((pair, match_index, match_tag))
            match_counter += 1

    total_jobs = len(jobs)
    if total_jobs == 0:
        return summary

    job_queue: asyncio.Queue[tuple[tuple[StrategyFactory, StrategyFactory], int, str]] = asyncio.Queue()
    for job in jobs:
        job_queue.put_nowait(job)

    async def _run_match_job(pair: tuple[StrategyFactory, StrategyFactory], match_index: int, match_tag: str) -> BattleOutcome:
        strategy_a_factory, strategy_b_factory = pair
        max_attempts = 3
        last_error: Exception | None = None
        for attempt in range(max_attempts):
            ordered = sorted((strategy_a_factory, strategy_b_factory), key=id)
            first_limit = strategy_limits[ordered[0]]
            second_limit = strategy_limits[ordered[1]]

            logging.debug(
                "Match %s waiting for strategy limits (pair=%s vs %s, index=%s, attempt=%s/%s)",
                match_tag,
                strategy_a_factory.__name__,
                strategy_b_factory.__name__,
                match_index,
                attempt + 1,
                max_attempts,
            )
            try:
                async with first_limit:
                    logging.debug("Match %s acquired first strategy slot", match_tag)
                    async with second_limit:
                        logging.debug("Match %s acquired second strategy slot; starting match", match_tag)
                        if match_index % 2 == 0:
                            return await run_showdown_match(
                                strategy_a_factory,
                                strategy_b_factory,
                                config=config,
                                client_factory=client_factory,
                                session_tag=match_tag,
                            )
                        return await run_showdown_match(
                            strategy_b_factory,
                            strategy_a_factory,
                            config=config,
                            client_factory=client_factory,
                            session_tag=match_tag,
                        )
            except TimeoutError as error:
                last_error = error
                if attempt == max_attempts - 1:
                    break
                backoff = 1.0 * (2 ** attempt)
                jitter = random.uniform(0.0, 1.0)
                logging.warning(
                    "Match %s timed out before completion; retrying after %.1fs (attempt %s/%s)",
                    match_tag,
                    backoff + jitter,
                    attempt + 1,
                    max_attempts,
                )
                await asyncio.sleep(backoff + jitter)

        if last_error is not None:
            logging.error(
                "Match %s failed after %s attempts; recording a timeout tie instead of aborting tournament",
                match_tag,
                max_attempts,
            )
            return BattleOutcome(
                strategy_one=strategy_a_factory().name,
                strategy_two=strategy_b_factory().name,
                winner=None,
                winner_strategy=None,
                room_id="",
                turns=0,
            )
        raise RuntimeError("Match job failed without an explicit error")

    # Worker coroutines that consume jobs and run matches. Number of concurrent
    # workers is bounded to `max_parallel_matches` to match previous semantics.
    workers: list[asyncio.Task[None]] = []
    num_workers = max(1, min(max_parallel_matches, total_jobs))

    results_counter = 0

    async def _worker(worker_id: int) -> None:
        nonlocal results_counter
        while True:
            try:
                pair, match_index, match_tag = await job_queue.get()
            except asyncio.CancelledError:
                break

            try:
                outcome = await _run_match_job(pair, match_index, match_tag)
                matchup_by_pair[pair].record(outcome)
                results_counter += 1
                if results_counter % 10 == 0 or results_counter == total_jobs:
                    logging.info("Tournament progress: %s/%s matches completed", results_counter, total_jobs)
            except Exception:
                logging.exception("Worker %s: match failed (pair=%s, idx=%s)", worker_id, pair, match_index)
                # Cancel remaining jobs and re-raise to abort tournament only for non-timeout failures.
                while not job_queue.empty():
                    try:
                        job_queue.get_nowait()
                        job_queue.task_done()
                    except Exception:
                        break
                raise
            finally:
                try:
                    job_queue.task_done()
                except Exception:
                    pass

    for i in range(num_workers):
        workers.append(asyncio.create_task(_worker(i)))

    try:
        await job_queue.join()
    except Exception:
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        raise

    for w in workers:
        w.cancel()
    await asyncio.gather(*workers, return_exceptions=True)

    return summary


def resolve_strategy_selection(strategy_names: list[str] | None = None) -> list[StrategyFactory]:
    if not strategy_names:
        return list(ALL_STRATEGY_FACTORIES)
    return [get_strategy_factory(strategy_name) for strategy_name in strategy_names]
