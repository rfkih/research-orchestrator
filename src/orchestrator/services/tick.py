"""``POST /tick`` orchestration — port of ``research-tick.sh``.

One tick = one iteration. The flow:

  1. Claim the next queue row (``FOR UPDATE SKIP LOCKED`` — concurrent
     ticks cannot collide).
  2. Derive the next sweep combo from ``sweep_config`` and the row's
     pre-increment ``iteration_number``.
  3. Look up the strategy's ``account_strategy_id``. Mismatch is a config
     error and the queue row is marked FAILED.
  4. Submit the backtest to the research JVM and poll ``backtest_run``
     until COMPLETED / FAILED.
  5. Pull run-level metrics, compute verdict, INSERT ``research_iteration_log``
     and pointer-update the queue row in a single transaction.
  6. Decide queue next-state per the bash decision table:
        NO_EDGE + early_stop=true        → COMPLETED / DISCARD
        SIGNIFICANT_EDGE + WF (deferred) → PENDING (continue) — Phase 4 wires WF
        INSUFFICIENT_EVIDENCE + budget   → PARKED
        otherwise                        → PENDING (continue)

Crash recovery: any exception after step 1 rolls the queue back to PENDING
with ``iteration_number`` decremented, mirroring the bash trap.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any
from uuid import UUID

import asyncpg
from fastapi.concurrency import run_in_threadpool

from redis.asyncio import Redis as AsyncRedis

from ..clients.jvm import JvmClient
from ..config import Settings
from ..errors import NextAction, OrchestratorError
from ..infra.db import Database
from ..logging import get_logger
from ..repo import (
    hypothesis_audit as audit_repo,
    iterations as iterations_repo,
    iterations_write,
    queue_write,
    trades as trades_repo,
)
from . import analyze, hedging_gate, paper_generator, portfolio, registry_sync, sweep
from .activity_logger import log_activity
from .tick_summary import compute_tick_summary

log = get_logger(__name__)

# Backtest poll-cap fallback. ``_poll_backtest_status`` uses this when a caller
# passes no explicit timeout (capacity / cpcv / cross_window / null_screen /
# walk_forward all do). The main /tick loop passes ``settings.poll_timeout_s``
# (env ``ORCH_POLL_TIMEOUT_S``, see config.py) so the operator can tune the cap
# for a slow link without a code change. Both default to 3600s: a full-window
# backtest over the remote VPS prod DB takes ~33 min, so the legacy 1800s cap
# falsely failed queues whose runs had actually completed.
_POLL_TIMEOUT_S = 3600
_POLL_INTERVAL_S = 5
# Stuck-RUNNING reaper buffer. The reaper resets RUNNING queue rows older than
# (poll cap + this buffer) back to PENDING — a generous margin over the poll
# cap so a healthy in-flight tick is never reset mid-run.
_STUCK_RUNNING_BUFFER_S = 5 * 60


def _external_declared_trials(sweep_config: dict[str, Any] | None) -> int:
    """Off-orchestrator trial count declared at enqueue (sweep_config
    ['external_trials']). Added into the DSR data-universe ``n_trials`` so
    hand-tuning / offline Python sweeps / direct JVM backtests the audit
    ledger can't see still deflate the Sharpe (2026-06-14 re-audit HOLE-1).
    Clamped ≥ 0 — can only raise multiplicity, so the gate only TIGHTENS.

    Coerce via float() so a JSONB-edited string like '15.7' truncates to 15
    (toward the gate-tightening direction) rather than being silently dropped
    to 0 (which would LOOSEN). sweep_config is operator-editable JSONB, so this
    is the last line of defense behind the API's int Field; genuinely
    uncoercible / non-finite junk drops to 0 with a warning."""
    raw = (sweep_config or {}).get("external_trials")
    if raw is None:
        return 0
    try:
        return max(0, int(float(raw)))
    except (TypeError, ValueError, OverflowError):
        log.warning("tick.external_trials_uncoercible", value=repr(raw))
        return 0


def _terminal(err: OrchestratorError) -> OrchestratorError:
    """Tag an OrchestratorError so the outer rollback skips it.

    Used when a code path has already written the queue row's next state
    itself (FAILED/PARKED/COMPLETED, or a deliberate PENDING-for-resume).
    The default ``except`` arm would otherwise call ``_rollback`` and
    clobber that with a second decrement.
    """
    err.queue_terminalized = True  # type: ignore[attr-defined]
    return err


class TickResult:
    """Plain container — the API layer renders this through Pydantic.

    ``pf`` and ``n_trades`` are optional surface metrics pulled out of the
    iteration's ``metrics_snapshot`` when available; they feed the
    ``summary.verdict_line`` so the runner agent can read a one-liner
    instead of the whole iteration JSON.
    """

    def __init__(
        self,
        *,
        outcome: str,
        queue_id: str | None = None,
        iteration_id: str | None = None,
        backtest_run_id: str | None = None,
        statistical_verdict: str | None = None,
        verdict: str | None = None,
        notes: list[str] | None = None,
        next_actions: list[dict[str, Any]] | None = None,
        pf: float | None = None,
        n_trades: int | None = None,
    ) -> None:
        self.outcome = outcome
        self.queue_id = queue_id
        self.iteration_id = iteration_id
        self.backtest_run_id = backtest_run_id
        self.statistical_verdict = statistical_verdict
        self.verdict = verdict
        self.notes = notes or []
        self.next_actions = next_actions or []
        self.pf = pf
        self.n_trades = n_trades

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "queue_id": self.queue_id,
            "iteration_id": self.iteration_id,
            "backtest_run_id": self.backtest_run_id,
            "statistical_verdict": self.statistical_verdict,
            "verdict": self.verdict,
            "notes": self.notes,
            "next_actions": self.next_actions,
            "summary": compute_tick_summary(
                outcome=self.outcome,
                statistical_verdict=self.statistical_verdict,
                verdict=self.verdict,
                next_actions=self.next_actions,
                pf=self.pf,
                n_trades=self.n_trades,
            ),
        }


async def _resolve_account_strategy_id(
    conn: asyncpg.Connection,
    strategy_code: str,
    account_id: UUID | None = None,
) -> str | None:
    """V54: when ``account_id`` is set (prod always sets it via
    ``ORCH_RESEARCH_ACCOUNT_ID``), pin the lookup to the research-agent's
    account so admin-owned rows for the same strategy_code can't be picked.
    When unset (local dev convenience), fall back to the legacy "first
    matching row" behaviour."""
    if account_id is not None:
        return await conn.fetchval(
            """
            SELECT account_strategy_id
            FROM account_strategy
            WHERE strategy_code = $1 AND account_id = $2 AND is_deleted = false
            LIMIT 1
            """,
            strategy_code,
            account_id,
        )
    return await conn.fetchval(
        """
        SELECT account_strategy_id
        FROM account_strategy
        WHERE strategy_code = $1 AND is_deleted = false
        LIMIT 1
        """,
        strategy_code,
    )


async def _resolve_account_strategy(
    conn: asyncpg.Connection,
    strategy_code: str,
    account_id: UUID | None = None,
) -> dict[str, Any] | None:
    """Like ``_resolve_account_strategy_id`` but also returns the strategy's
    allow_long/allow_short flags so researcher payloads match the production
    direction policy (instead of hardcoding both to True regardless of what
    the strategy actually supports). Result keys: ``account_strategy_id``,
    ``allow_long``, ``allow_short``.

    V54: ``account_id`` scopes the lookup to the research-agent account in
    prod; when unset, retains the legacy "first matching row" fallback for
    local dev."""
    if account_id is not None:
        row = await conn.fetchrow(
            """
            SELECT account_strategy_id, allow_long, allow_short
            FROM account_strategy
            WHERE strategy_code = $1 AND account_id = $2 AND is_deleted = false
            LIMIT 1
            """,
            strategy_code,
            account_id,
        )
    else:
        row = await conn.fetchrow(
            """
            SELECT account_strategy_id, allow_long, allow_short
            FROM account_strategy
            WHERE strategy_code = $1 AND is_deleted = false
            LIMIT 1
            """,
            strategy_code,
        )
    if row is None:
        return None
    return {
        "account_strategy_id": str(row["account_strategy_id"]),
        # Treat NULL as TRUE — matches the JVM's resolveAllowLong/Short fallback.
        "allow_long": True if row["allow_long"] is None else bool(row["allow_long"]),
        "allow_short": True if row["allow_short"] is None else bool(row["allow_short"]),
    }


async def _poll_backtest_status(
    db: Database, backtest_run_id: str, timeout_s: int = _POLL_TIMEOUT_S
) -> str:
    """Poll until terminal status. Returns the final status string.

    Holds a single connection for the full poll loop instead of acquiring
    one per probe — an hour-long poll at a 5s interval would otherwise burn
    ~720 pool acquisitions for nothing.
    """
    deadline = time.monotonic() + timeout_s
    last_status: str | None = None
    async with db.acquire() as conn:
        while time.monotonic() < deadline:
            row = await conn.fetchrow(
                "SELECT status FROM backtest_run WHERE backtest_run_id = $1",
                backtest_run_id,
            )
            last_status = (row["status"] if row else None) or "UNKNOWN"
            if last_status in ("COMPLETED", "FAILED", "CANCELLED", "ERROR"):
                return last_status
            await asyncio.sleep(_POLL_INTERVAL_S)
    return last_status or "TIMEOUT"


async def _fetch_run_metrics(conn: asyncpg.Connection, backtest_run_id: str) -> dict[str, Any]:
    row = await conn.fetchrow(
        """
        SELECT backtest_run_id, strategy_code, status, total_trades, total_wins,
               total_losses, win_rate, profit_factor, net_profit, return_pct,
               max_drawdown_pct, max_drawdown_amount, sharpe_ratio, sortino_ratio,
               avg_win, avg_loss, expectancy, initial_capital, ending_balance,
               asset, interval_name, start_time, end_time,
               avg_trade_return_pct, geometric_return_pct_at_alloc_90
        FROM backtest_run
        WHERE backtest_run_id = $1
        """,
        backtest_run_id,
    )
    if row is None:
        raise OrchestratorError(
            status_code=502,
            error_code="backtest_row_missing",
            message=f"Backtest reported COMPLETED but backtest_run row {backtest_run_id} not found.",
            retryable=False,
        )
    # Decimal/datetime → JSON-friendly. For the metrics_snapshot JSONB
    # column we serialise to native types up front.
    def _coerce(v: Any) -> Any:
        if isinstance(v, Decimal):
            return str(v)
        if isinstance(v, datetime):
            return v.isoformat()
        if isinstance(v, UUID):
            return str(v)
        return v

    return {k: _coerce(v) for k, v in dict(row).items()}


def _build_submit_payload(
    *,
    account_strategy_id: str,
    strategy_code: str,
    asset: str,
    interval_name: str,
    overrides: dict[str, Any],
    settings: Settings,  # noqa: ARG001 — reserved for prod-window/capital config
    allow_long: bool = True,
    allow_short: bool = True,
    start_time_override: str | None = None,
    ml_overrides: dict[str, Any] | None = None,
    universe: list[str] | None = None,
) -> dict[str, Any]:
    """Build the BacktestRunRequest payload for one sweep cell.

    Phase B (V100, 2026-05-19): ``ml_overrides`` carries optional ML
    gate overrides for this cell — gate enabled toggle, signal_name
    variant, shadow-mode toggle. None / empty means "no override; JVM
    uses account_strategy persisted values". The map is built by
    :func:`sweep.build_ml_override_maps` from the sentinel keys in
    the combo.
    """
    # Window + sizing match research-tick.sh lines 340-359. endTime is
    # "yesterday UTC midnight" so we never test on a partial bar; bash
    # had a hardcoded date that drifted with each rebuild, this doesn't.
    end_time = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%dT00:00:00")
    start_time = start_time_override or "2024-01-01T00:00:00"
    payload: dict[str, Any] = {
        "accountStrategyId": account_strategy_id,
        "strategyCode": strategy_code,
        "asset": asset,
        "interval": interval_name,
        "startTime": start_time,
        "endTime": end_time,
        "initialCapital": 1000,
        "riskPerTradePct": 2.0,
        "feeRate": 0.00075,
        "minNotional": 5,
        "minQty": 0.000001,
        "qtyStep": 0.000001,
        "maxOpenPositions": 1,
        # Mirror the production strategy's direction policy rather than
        # hardcoding both. A long-only strategy researched with shorts
        # enabled would surface fake P&L from a side it'll never trade.
        "allowLong": allow_long,
        "allowShort": allow_short,
        "strategyParamOverrides": {strategy_code: overrides},
        # Origin tag — surfaces as a "RESEARCHER" badge on the frontend run
        # list/detail so users can tell autonomous-agent runs apart from
        # their own. Backend defaults missing values to USER.
        "triggeredBy": "RESEARCHER",
    }
    if ml_overrides:
        # ml_overrides already has the JVM-side keys
        # (strategyMl*Overrides) → {strategy_code: value}. Spread into
        # the payload; absent keys are NOT added.
        payload.update(ml_overrides)
    if universe:
        # Cross-sectional run: the JVM routes to the XS coordinator. asset is
        # set to the first universe symbol so the NOT-NULL backtest_run.asset
        # column + @NotBlank validation stay satisfied; the XS path uses the
        # full universe list, not asset.
        payload["universe"] = universe
        payload["asset"] = universe[0]
    return payload


async def run_tick(
    *,
    db: Database,
    jvm: JvmClient,
    settings: Settings,
    agent_name: str,
    session_id: UUID | None = None,
    redis_client: AsyncRedis | None = None,
    track: str | None = None,
) -> TickResult:
    """One tick. Idempotent at the queue level — re-runs are safe because
    each tick claims a fresh row and atomically increments iteration_number.

    ``track`` (Phase 1): scope the claim to one research loop. None ⇒ the
    legacy global claim (picks any PENDING row); a value restricts to rows
    whose ``sweep_config['track']`` matches."""

    # ── Step 0: reap any RUNNING rows older than the poll cap ─────────
    # Covers SIGKILL-style crashes whose rollback handler never ran.
    async with db.acquire() as conn:
        async with conn.transaction():
            recovered = await queue_write.recover_stuck(
                conn, settings.poll_timeout_s + _STUCK_RUNNING_BUFFER_S
            )
    if recovered:
        log.warn("tick.recovered_stuck_rows", count=recovered)

    # ── Step 1: claim row ──────────────────────────────────────────────
    async with db.acquire() as conn:
        async with conn.transaction():
            claim = await queue_write.claim_next(conn, track=track)

    if claim is None:
        return TickResult(
            outcome="empty_queue",
            notes=["Queue has no PENDING/RUNNING rows. Enqueue with POST /queue."],
            next_actions=[{"kind": "call", "method": "POST", "path": "/queue"}],
        )

    queue_id = claim["queue_id"]
    strategy_code = claim["strategy_code"]
    interval_name = claim["interval_name"]
    instrument = claim["instrument"]
    sweep_config = claim["sweep_config"]
    prior_iter = int(claim["prior_iter"])
    iter_budget = int(claim["iter_budget"])
    early_stop = bool(claim["early_stop_on_no_edge"])
    started_at = claim["started_at"]
    active_run_id = claim.get("active_backtest_run_id")
    new_iter = prior_iter + 1

    log.info(
        "tick.claimed",
        queue_id=str(queue_id),
        strategy_code=strategy_code,
        iter=new_iter,
        budget=iter_budget,
    )

    try:
        return await _execute_after_claim(
            db=db,
            jvm=jvm,
            settings=settings,
            agent_name=agent_name,
            session_id=session_id,
            redis_client=redis_client,
            queue_id=queue_id,
            strategy_code=strategy_code,
            interval_name=interval_name,
            instrument=instrument,
            sweep_config=sweep_config,
            prior_iter=prior_iter,
            new_iter=new_iter,
            iter_budget=iter_budget,
            early_stop=early_stop,
            started_at=started_at,
            active_run_id=active_run_id,
        )
    except OrchestratorError as err:
        # If the failing path already moved the row to a terminal state
        # (FAILED/PARKED), do NOT roll back — that would reset to PENDING
        # and the row would be reclaimed and fail again forever.
        if not getattr(err, "queue_terminalized", False):
            await _rollback(db, queue_id, "Tick failed (domain error); rolled back to PENDING.")
        raise
    except BaseException as e:  # noqa: BLE001 — catches CancelledError (client disconnect) too
        log.error("tick.crashed", queue_id=str(queue_id), error=str(e))
        await _rollback(db, queue_id, f"Tick crashed: {type(e).__name__}; rolled back to PENDING.")
        raise


async def _rollback(db: Database, queue_id: Any, note: str) -> None:
    # expected_status='RUNNING': if the operator cancelled (PARKED) the row
    # while this tick was in flight, the rollback must not resurrect it.
    # active_backtest_run_id survives the rollback, so a run that is still
    # going on the JVM is resumed — not resubmitted — by the next tick.
    try:
        async with db.acquire() as conn:
            affected = await queue_write.rollback_iter(
                conn, queue_id, note, expected_status="RUNNING"
            )
        if affected == 0:
            log.warn(
                "tick.rollback_skipped_row_not_running",
                queue_id=str(queue_id),
            )
    except Exception as cleanup_err:  # noqa: BLE001
        log.warn("tick.rollback_failed", queue_id=str(queue_id), error=str(cleanup_err))


async def _execute_after_claim(
    *,
    db: Database,
    jvm: JvmClient,
    settings: Settings,
    agent_name: str,
    session_id: UUID | None,
    redis_client: AsyncRedis | None,
    queue_id: Any,
    strategy_code: str,
    interval_name: str,
    instrument: str,
    sweep_config: dict[str, Any],
    prior_iter: int,
    new_iter: int,
    iter_budget: int,
    early_stop: bool,
    started_at: Any,
    active_run_id: Any = None,
) -> TickResult:
    # ── Step 1.5: resume an in-flight run from a prior dead attempt ───
    # A prior tick that died mid-poll (client disconnect, poll-cap expiry,
    # crash, reaper recovery) leaves its run id on the queue row. Resuming
    # it — instead of re-deriving the combo and submitting a fresh run —
    # is what prevents duplicate JVM runs and duplicate audit rows.
    resumed = False
    resume_status: str | None = None
    combo: dict[str, Any] | None = None
    audit_id: Any = None
    if active_run_id is not None:
        async with db.acquire() as conn:
            resume_status = await conn.fetchval(
                "SELECT status FROM backtest_run WHERE backtest_run_id = $1",
                active_run_id,
            )
        if resume_status in ("RUNNING", "COMPLETED"):
            async with db.acquire() as conn:
                prior_audit = await audit_repo.latest_unattached_for_queue(
                    conn, queue_id
                )
                existing_iter_probe = await iterations_write.find_iteration_for_run(
                    conn, active_run_id
                )
            if prior_audit is not None or existing_iter_probe is not None:
                resumed = True
                if prior_audit is not None:
                    # The audit row written just before the original submit
                    # holds the authoritative combo (TPE sampling is not
                    # deterministic across processes — re-deriving diverges).
                    combo = prior_audit["params_snapshot"]
                    audit_id = prior_audit["audit_id"]
                log.info(
                    "tick.resuming_active_run",
                    queue_id=str(queue_id),
                    run_id=str(active_run_id),
                    run_status=resume_status,
                    adopting_iteration=existing_iter_probe is not None,
                )
            else:
                # Pointer with neither an unattached audit row nor a logged
                # iteration — inconsistent leftover. Fall through to a fresh
                # submit; the old run's result is orphaned (logged for ops).
                log.warn(
                    "tick.active_run_unresumable",
                    queue_id=str(queue_id),
                    run_id=str(active_run_id),
                )
        else:
            # FAILED/CANCELLED/ERROR or row missing — nothing to resume.
            resume_status = None

    if resumed:
        backtest_run_id = str(active_run_id)
        async with db.acquire() as conn:
            cumulative_trials = await audit_repo.count_data_universe_trials(
                conn, instrument, interval_name,
                external_declared=_external_declared_trials(sweep_config),
            ) + 1
    else:
        combo, audit_id, cumulative_trials, backtest_run_id = (
            await _derive_and_submit(
                db=db,
                jvm=jvm,
                settings=settings,
                agent_name=agent_name,
                session_id=session_id,
                redis_client=redis_client,
                queue_id=queue_id,
                strategy_code=strategy_code,
                interval_name=interval_name,
                instrument=instrument,
                sweep_config=sweep_config,
                prior_iter=prior_iter,
                new_iter=new_iter,
            )
        )
        if backtest_run_id is None:
            # Sweep exhausted — _derive_and_submit already parked the row.
            return TickResult(
                outcome="sweep_exhausted",
                queue_id=str(queue_id),
                notes=[str(combo)],
                next_actions=[
                    {"kind": "read_doc", "doc_anchor": "queue.sweep_exhausted"}
                ],
            )

    return await _poll_and_finish(
        db=db,
        settings=settings,
        agent_name=agent_name,
        session_id=session_id,
        redis_client=redis_client,
        queue_id=queue_id,
        strategy_code=strategy_code,
        interval_name=interval_name,
        instrument=instrument,
        sweep_config=sweep_config,
        new_iter=new_iter,
        iter_budget=iter_budget,
        early_stop=early_stop,
        started_at=started_at,
        backtest_run_id=backtest_run_id,
        combo=combo or {},
        audit_id=audit_id,
        cumulative_trials=cumulative_trials,
        skip_poll=(resumed and resume_status == "COMPLETED"),
    )


async def _derive_and_submit(
    *,
    db: Database,
    jvm: JvmClient,
    settings: Settings,
    agent_name: str,
    session_id: UUID | None,
    redis_client: AsyncRedis | None,
    queue_id: Any,
    strategy_code: str,
    interval_name: str,
    instrument: str,
    sweep_config: dict[str, Any],
    prior_iter: int,
    new_iter: int,
) -> tuple[dict[str, Any] | None, Any, int, str | None]:
    """Steps 2-4: derive combo, resolve account, audit, submit.

    Returns ``(combo, audit_id, cumulative_trials, backtest_run_id)``.
    ``backtest_run_id`` is None when the sweep is exhausted (row already
    parked; ``combo`` then carries the exhaustion note).
    """
    # ── Step 2: derive sweep combo ────────────────────────────────────
    # Grid: deterministic cross-product index. TPE: ask Optuna for the
    # next point given the queue's history of (params, pf) pairs.
    if sweep.is_tpe(sweep_config):
        async with db.acquire() as conn:
            past_trials = await iterations_repo.fetch_tpe_history_for_queue(
                conn, queue_id
            )
        try:
            combo = sweep.next_combo_tpe(
                sweep_config, past_trials, iter_index=prior_iter
            )
        except Exception as e:  # noqa: BLE001 — TPE config errors get a clean envelope
            note = (
                f"[{datetime.now(timezone.utc).isoformat()}] TPE sampler failed "
                f"at iter={prior_iter}: {type(e).__name__}: {e}."
            )
            async with db.acquire() as conn:
                await queue_write.fail_queue(
                    conn, queue_id, note, expected_status="RUNNING"
                )
            raise _terminal(OrchestratorError(
                status_code=500,
                error_code="tpe_sampler_failed",
                message=note,
                retryable=False,
                details={"queue_id": str(queue_id), "iter": prior_iter},
            )) from e
    else:
        combo = sweep.derive_combo(sweep_config, prior_iter)

    if combo is None:
        if sweep.is_tpe(sweep_config):
            exhausted_note = (
                f"TPE budget exhausted at iter={prior_iter}; "
                f"n_trials={sweep.tpe_trial_budget(sweep_config)}."
            )
        else:
            exhausted_note = (
                f"Sweep exhausted at iter={prior_iter}; "
                f"total combos={sweep.total_combos(sweep_config)}."
            )
        note = f"[{datetime.now(timezone.utc).isoformat()}] {exhausted_note}"
        async with db.acquire() as conn:
            await queue_write.park_exhausted(
                conn, queue_id, note, expected_status="RUNNING"
            )
        return note, None, 0, None

    # ── Step 3: resolve account_strategy_id ───────────────────────────
    async with db.acquire() as conn:
        as_row = await _resolve_account_strategy(
            conn, strategy_code, settings.research_account_id
        )
    if not as_row:
        note = (
            f"[{datetime.now(timezone.utc).isoformat()}] No account_strategy row "
            f"for strategy_code={strategy_code} (is_deleted=false)."
        )
        async with db.acquire() as conn:
            await queue_write.fail_queue(
                conn, queue_id, note, expected_status="RUNNING"
            )
        raise _terminal(OrchestratorError(
            status_code=412,
            error_code="account_strategy_missing",
            message=note,
            retryable=False,
            hint="Seed an account_strategy row for the strategy_code before enqueuing.",
            next_action=NextAction(kind="contact_human"),
            details={"queue_id": str(queue_id), "strategy_code": strategy_code},
        ))

    # ── Step 3.5: record hypothesis_audit row BEFORE the backtest ─────
    # Recording up-front means a crashed/timed-out tick still counts
    # toward trial multiplicity — overfitting risk is "did the loop ever
    # see this combo", not "did the run finish". We backfill verdicts
    # in step 5 once the iteration is logged.
    async with db.acquire() as conn:
        async with conn.transaction():
            audit_id = await audit_repo.insert_audit(
                conn,
                strategy_code=strategy_code,
                symbol=instrument,
                interval_name=interval_name,
                params_snapshot=combo,
                queue_id=queue_id,
                created_by=agent_name,
            )
            # +1 for the current in-flight trial — its audit row has
            # iteration_id=NULL until step 5 backfills, so the COUNT
            # query (filtered to completed rows only) excludes it.
            #
            # DSR multiplicity is the DATA-UNIVERSE trial count (every
            # COMPLETED trial on this symbol × interval, regardless of
            # strategy_code) — not the strategy_code-scoped count, which
            # under-counts multiplicity because it resets on every
            # archetype pivot (V93). ``instrument`` is the symbol here.
            cumulative_trials = await audit_repo.count_data_universe_trials(
                conn, instrument, interval_name,
                external_declared=_external_declared_trials(sweep_config),
            ) + 1

    # ── Step 4: submit backtest, poll for completion ──────────────────
    # Optional extended-window override carried on sweep_config. Used when
    # the operator backfills earlier history (e.g. ETH pre-2024) and wants
    # the sweep to exercise the extra bars instead of the default 2024-01-01
    # floor. None ⇒ default.
    window_cfg = sweep_config.get("backtest_window") or {}
    start_time_override = window_cfg.get("start_time")
    # Phase B (V100, 2026-05-19): split sentinel ML keys out of the
    # combo so they don't pollute strategyParamOverrides. The split
    # values land on the JVM's strategyMl*Overrides fields instead.
    # On non-ML sweeps both halves are dict + empty dict, no change.
    regular_combo, ml_combo = sweep.split_ml_overrides(combo)
    ml_override_payload = sweep.build_ml_override_maps(
        strategy_code=strategy_code,
        ml_overrides=ml_combo,
    )
    # 2026-05-31 fix: the VBO engine family implements its OWN in-engine
    # ML gate, reading the sentinels from the StrategySpec params via
    # VboTuning.from(spec) (s.paramBoolean("_ml_gate_enabled"), etc.) —
    # NOT via RiskGuardService's V100 strategyMl*Overrides path that
    # DCB-style engines use. For those engines the sentinels MUST also
    # remain in strategyParamOverrides or VboTuning.mlGateEnabled() is
    # always false and the gate silently fail-opens (bit-identical to
    # ALGO). The JVM's KNOWN_PARAM_KEYS for these codes already whitelists
    # the three sentinels. RiskGuardService-path engines ignore unknown
    # spec params, so re-adding them is harmless redundancy there.
    if strategy_code.upper() in sweep.IN_SPEC_ML_GATE_CODES and ml_combo:
        regular_combo = {**regular_combo, **ml_combo}
    payload = _build_submit_payload(
        account_strategy_id=as_row["account_strategy_id"],
        strategy_code=strategy_code,
        asset=instrument,
        interval_name=interval_name,
        overrides=regular_combo,
        settings=settings,
        allow_long=as_row["allow_long"],
        allow_short=as_row["allow_short"],
        start_time_override=start_time_override,
        ml_overrides=ml_override_payload,
        universe=(sweep_config or {}).get("universe"),
    )
    backtest_run_id = await jvm.submit_backtest(payload)
    log.info(
        "tick.submitted",
        queue_id=str(queue_id),
        run_id=backtest_run_id,
        combo=combo,
    )
    # Persist the in-flight run on the queue row IMMEDIATELY: from here on,
    # any death of this tick (disconnect, crash, poll cap) is recoverable —
    # the next claim resumes this run instead of double-submitting.
    async with db.acquire() as conn:
        await queue_write.set_active_run(conn, queue_id, backtest_run_id)

    # Activity log: TICK_DISPATCHED (fire-and-forget)
    try:
        async with db.acquire() as _log_conn:
            await log_activity(
                _log_conn,
                session_id=session_id,
                agent_name=agent_name,
                activity_type="TICK_DISPATCHED",
                title=f"Backtest dispatched for {strategy_code} iter {new_iter}",
                strategy_code=strategy_code,
                details={
                    "backtest_run_id": backtest_run_id,
                    "queue_id": str(queue_id),
                    "iter": new_iter,
                    "combo": combo,
                },
                related_id=queue_id if isinstance(queue_id, UUID) else None,
                related_type="queue",
                redis_client=redis_client,
            )
    except Exception as _act_exc:  # noqa: BLE001
        log.warning("tick.activity_log_failed", step="TICK_DISPATCHED", error=str(_act_exc))

    return combo, audit_id, cumulative_trials, backtest_run_id


async def _poll_and_finish(
    *,
    db: Database,
    settings: Settings,
    agent_name: str,
    session_id: UUID | None,
    redis_client: AsyncRedis | None,
    queue_id: Any,
    strategy_code: str,
    interval_name: str,
    instrument: str,
    sweep_config: dict[str, Any],
    new_iter: int,
    iter_budget: int,
    early_stop: bool,
    started_at: Any,
    backtest_run_id: str,
    combo: dict[str, Any],
    audit_id: Any,
    cumulative_trials: int,
    skip_poll: bool = False,
) -> TickResult:
    """Steps 4.5-6: poll to terminal, analyze, log, decide next state."""
    if skip_poll:
        final_status = "COMPLETED"
    else:
        final_status = await _poll_backtest_status(
            db, backtest_run_id, timeout_s=settings.poll_timeout_s
        )
    if final_status != "COMPLETED":
        ts = datetime.now(timezone.utc).isoformat()
        if final_status in ("FAILED", "CANCELLED", "ERROR"):
            # JVM-reported terminal failure — the run is genuinely dead.
            note = (
                f"[{ts}] Backtest terminal status="
                f"{final_status} (run_id={backtest_run_id})."
            )
            async with db.acquire() as conn:
                await queue_write.fail_queue(
                    conn, queue_id, note, expected_status="RUNNING"
                )
            raise _terminal(OrchestratorError(
                status_code=502,
                error_code="backtest_did_not_complete",
                message=note,
                retryable=False,
                hint=(
                    "JVM reported a terminal failure for this run. Inspect the "
                    "run, then POST /queue/{queue_id}/requeue to retry the cell."
                ),
                details={"backtest_run_id": backtest_run_id, "final_status": final_status},
            ))
        # Poll cap expired while the run is still going (RUNNING / UNKNOWN /
        # TIMEOUT) — an orchestrator-side latency condition, NOT a research
        # outcome. The historical bug terminalized the row to FAILED here,
        # orphaning a backtest that completed minutes later. Instead: hand
        # the row back to PENDING (active_backtest_run_id stays set) and
        # tell the caller to retry — the next tick resumes this same run.
        note = (
            f"[{ts}] Poll cap ({settings.poll_timeout_s}s) expired with run "
            f"status={final_status} (run_id={backtest_run_id}); row reset to "
            f"PENDING for resume — the run was NOT failed."
        )
        async with db.acquire() as conn:
            await queue_write.rollback_iter(
                conn, queue_id, note, expected_status="RUNNING"
            )
        raise _terminal(OrchestratorError(
            status_code=503,
            error_code="backtest_poll_cap_exceeded",
            message=note,
            retryable=True,
            hint=(
                "The backtest is still running on the JVM. POST /tick again "
                "to resume polling the same run — no duplicate is submitted."
            ),
            details={"backtest_run_id": backtest_run_id, "last_status": final_status},
        ))

    # ── Step 5: analyze + log iteration. If a prior attempt already logged
    # the iteration for this run (it died between commit and pointer
    # update), ADOPT that row instead of inserting a duplicate. ──────────
    async with db.acquire() as conn:
        existing_iter = await iterations_write.find_iteration_for_run(
            conn, backtest_run_id
        )
    if existing_iter is not None:
        iteration_id = existing_iter["iteration_id"]
        stat_v = existing_iter["statistical_verdict"]
        dec_v = existing_iter["verdict"]
        _ms = existing_iter.get("metrics_snapshot") or {}
        analysis = (_ms.get("analysis") or {}) if isinstance(_ms, dict) else {}
        log.info(
            "tick.adopted_existing_iteration",
            queue_id=str(queue_id),
            iteration_id=str(iteration_id),
            backtest_run_id=str(backtest_run_id),
        )
        await _attach_fenced(
            db,
            queue_id=queue_id,
            iteration_id=iteration_id,
            backtest_run_id=backtest_run_id,
            started_at=started_at,
        )
        return await _finish_iteration(
            db=db,
            agent_name=agent_name,
            session_id=session_id,
            redis_client=redis_client,
            queue_id=queue_id,
            strategy_code=strategy_code,
            sweep_config=sweep_config,
            new_iter=new_iter,
            iter_budget=iter_budget,
            early_stop=early_stop,
            backtest_run_id=backtest_run_id,
            iteration_id=iteration_id,
            stat_v=stat_v,
            dec_v=dec_v,
            analysis=analysis,
        )

    async with db.acquire() as conn:
        run_metrics = await _fetch_run_metrics(conn, backtest_run_id)
        trades = await trades_repo.fetch_trades(conn, backtest_run_id)

    # Re-fetch the raw run dict with native datetimes for analyze.py — the
    # _fetch_run_metrics path stringifies for JSONB. analyze_run wants
    # datetime objects to compute the annualisation factor.
    async with db.acquire() as conn:
        raw_run = await conn.fetchrow(
            "SELECT start_time, end_time, return_pct, max_drawdown_pct, "
            "       geometric_return_pct_at_alloc_90 "
            "FROM backtest_run WHERE backtest_run_id = $1",
            backtest_run_id,
        )
    raw_run_dict = dict(raw_run) if raw_run else {}

    # analyze_run runs a 2000-iter bootstrap on the PnL series + regime
    # stratification — pure Python, can be 100-500ms on a healthy
    # backtest. Offload to a threadpool so the event loop stays
    # responsive for the activity-publisher and concurrent /healthz
    # probes while it crunches.
    analysis = await run_in_threadpool(
        analyze.analyze_run,
        raw_run_dict,
        trades,
        cumulative_trials=cumulative_trials,
    )
    stat_v = analysis["statistical_verdict"]["verdict"]

    # ── Portfolio correlation gate (layered ON TOP of V11) ────────────
    # Only runs when V11 already cleared SIGNIFICANT_EDGE — never relaxes
    # V11, only further demotes. A candidate too correlated with the live
    # book (LSR/VCB/VBO) gets sent back to INSUFFICIENT_EVIDENCE; the
    # underlying signal may be real but it duplicates existing exposure.
    portfolio_corr_payload: dict[str, Any] | None = None
    if stat_v == "SIGNIFICANT_EDGE":
        gate = await portfolio.evaluate_portfolio_gate(
            db,
            run_metrics=run_metrics,
            trades=trades,
            pf_ci=analysis["pf_95_ci"],
        )
        portfolio_corr_payload = gate
        if gate.get("demote"):
            stat_v = "INSUFFICIENT_EVIDENCE"
            analysis = {
                **analysis,
                "statistical_verdict": {
                    "verdict": "INSUFFICIENT_EVIDENCE",
                    "reason": (
                        f"Demoted by portfolio gate: {gate.get('reason')}"
                    ),
                },
            }
            log.info(
                "tick.portfolio_gate_demoted",
                queue_id=str(queue_id),
                max_abs_corr=gate.get("max_abs_corr"),
                pf_lo_effective=gate.get("pf_lo_effective"),
            )

    sample_ok = analyze.sample_size_adequate(analysis["n_trades"])
    dec_v = analyze.decision_verdict(
        stat_v,
        analysis["n_trades"],
        analysis.get("annualized_geometric_return_pct_at_alloc_90"),
    )

    # ── Dual-track gate selection (Phase 2, 2026-06-07) ──────────────────
    # The queue row's track (sweep_config['track']) chooses the graduation
    # objective. track in (None, "trading") is the DEFAULT and MUST stay
    # byte-for-byte unchanged — V11 (stat_v) + V60 (dec_v, computed above)
    # decide the verdict, exactly as before. Only track == "hedging" diverges
    # (see _select_track_gate). V11/V60 constants and the trading path are
    # untouched.
    stat_v, dec_v, hedging_gate_payload = await _select_track_gate(
        db=db,
        track=(sweep_config or {}).get("track"),
        stat_v=stat_v,
        dec_v=dec_v,
        backtest_run_id=backtest_run_id,
        instrument=instrument,
        window_start=raw_run_dict.get("start_time"),
        window_end=raw_run_dict.get("end_time"),
    )

    # metrics_snapshot blends run-level metrics with the analyzer payload.
    metrics_snapshot = {**run_metrics, "analysis": analysis}
    if portfolio_corr_payload is not None:
        metrics_snapshot["portfolio_corr"] = portfolio_corr_payload
    if hedging_gate_payload is not None:
        # Stash the full hedging verdict for audit (mirrors portfolio_corr).
        metrics_snapshot["hedging_gate"] = hedging_gate_payload

    # Transaction A: commit the EVIDENCE (iteration + DSR + audit backfill)
    # before the fenced queue-pointer update. The previous single-transaction
    # shape rolled the iteration row back when the fence failed — destroying
    # a fully-completed backtest's audit trail and leaving the step-3.5
    # audit row uncounted by DSR multiplicity exactly when ticks raced.
    async with db.acquire() as conn:
        async with conn.transaction():
            log_iter_n = await iterations_write.next_iteration_number(conn, strategy_code)
            iteration_id = await iterations_write.insert_iteration(
                conn,
                strategy_code=strategy_code,
                iteration_number=log_iter_n,
                backtest_run_id=backtest_run_id,
                params_snapshot=combo,
                metrics_snapshot=metrics_snapshot,
                confidence_intervals={"pf_95": analysis["pf_95_ci"]},
                verdict=dec_v,
                statistical_verdict=stat_v,
                sample_size_adequate=sample_ok,
                git_commit_hash=None,
                code_change_summary=f"Sweep iter {log_iter_n} (orchestrator): {combo}",
                change_reasoning="Mechanical sweep iteration via POST /tick.",
                quant_audit_notes=(
                    "Logged by orchestrator. statistical_verdict via bootstrap PF CI "
                    "(n_resamples=2000); DSR per Bailey & Lopez de Prado 2014 with "
                    f"cumulative_trials={cumulative_trials} (real selection-bias "
                    f"multiplicity from hypothesis_audit); decision_verdict gates on "
                    f"annualized_geom_return_pct_at_alloc_90="
                    f"{analysis.get('annualized_geometric_return_pct_at_alloc_90')} "
                    f">= 10.0 (PASS threshold). Slippage at +20bps="
                    f"{analysis['slippage_haircut_pnl'].get('+20bps')} retained for "
                    f"audit but no longer gates the verdict. "
                    "Walk-forward (POST /walk-forward) gates final PASS verdict."
                ),
                created_by=agent_name,
            )
            # Persist DSR onto the JVM-owned backtest_run row so the backtest
            # leaderboard (which ranks on backtest_run.deflated_sharpe) sees it.
            # V134 added the column but never wired the writer — without this
            # the leaderboard's "deflated_sharpe IS NOT NULL" feed stays empty.
            await iterations_write.update_backtest_run_dsr(
                conn,
                backtest_run_id=backtest_run_id,
                deflated_sharpe=analysis.get("dsr"),
                dsr_n_trials=analysis.get("dsr_n_trials"),
            )
            # Backfill the audit row written in step 3.5 with the resolved
            # verdicts. The re-discovery gate keys on decision_verdict so
            # this update is what enables the gate to fire on later sweeps.
            if audit_id is not None:
                await audit_repo.update_audit_verdict(
                    conn,
                    audit_id=audit_id,
                    iteration_id=iteration_id,
                    statistical_verdict=stat_v,
                    decision_verdict=dec_v,
                )

    # Transaction B: fenced queue-pointer update.
    await _attach_fenced(
        db,
        queue_id=queue_id,
        iteration_id=iteration_id,
        backtest_run_id=backtest_run_id,
        started_at=started_at,
    )

    log.info(
        "tick.iteration_logged",
        queue_id=str(queue_id),
        iteration_id=str(iteration_id),
        statistical_verdict=stat_v,
        verdict=dec_v,
    )
    return await _finish_iteration(
        db=db,
        agent_name=agent_name,
        session_id=session_id,
        redis_client=redis_client,
        queue_id=queue_id,
        strategy_code=strategy_code,
        sweep_config=sweep_config,
        new_iter=new_iter,
        iter_budget=iter_budget,
        early_stop=early_stop,
        backtest_run_id=backtest_run_id,
        iteration_id=iteration_id,
        stat_v=stat_v,
        dec_v=dec_v,
        analysis=analysis,
    )


async def _attach_fenced(
    db: Database,
    *,
    queue_id: Any,
    iteration_id: Any,
    backtest_run_id: str,
    started_at: Any,
) -> None:
    """Fenced queue-pointer update (transaction B).

    Runs AFTER the iteration row is committed, so a fence mismatch loses
    nothing: the iteration is durable, and the next tick on this row adopts
    it via ``find_iteration_for_run`` instead of re-running the combo.
    """
    async with db.acquire() as conn:
        attached = await queue_write.attach_iteration(
            conn,
            queue_id,
            iteration_id=iteration_id,
            backtest_run_id=backtest_run_id,
            expected_started_at=started_at,
        )
    if attached == 0:
        # The reaper recovered this row mid-flight (started_at NULLed) or
        # another tick re-claimed it (fresh started_at). The iteration row
        # IS committed; only the queue pointer was withheld.
        raise _terminal(OrchestratorError(
            status_code=409,
            error_code="queue_row_reclaimed",
            message=(
                f"Queue row {queue_id} was recovered/reclaimed while this tick "
                f"was in flight (started_at fence mismatched). The iteration "
                f"row IS committed; the queue pointer was not updated. The "
                f"next tick on this row adopts the committed iteration — no "
                f"duplicate run will be submitted."
            ),
            retryable=True,
            details={
                "queue_id": str(queue_id),
                "iteration_id": str(iteration_id),
                "backtest_run_id": str(backtest_run_id),
            },
        ))


async def _sync_registry_best_effort(
    db: Database,
    *,
    strategy_code: str,
    backtest_run_id: str,
    statistical_verdict: str,
    agent_name: str,
) -> None:
    """Mirror this iteration's result into the Strategy Research Registry so it
    always reflects the latest research. BEST-EFFORT: its own connection, never
    raises, never affects the tick. Symbol/interval are read from the
    backtest_run; ``registry_sync`` leaves curated (auto_managed=FALSE) rows
    untouched and only creates/updates agent-owned rows."""
    try:
        async with db.acquire() as conn:
            br = await conn.fetchrow(
                "SELECT asset, interval_name FROM backtest_run WHERE backtest_run_id = $1::uuid",
                backtest_run_id,
            )
            if br is None:
                return
            outcome = await registry_sync.sync_from_research(
                conn,
                strategy_code=strategy_code,
                symbol=br["asset"],
                interval=br["interval_name"],
                statistical_verdict=statistical_verdict,
                agent=agent_name,
            )
        log.info("tick.registry_synced", strategy_code=strategy_code, outcome=outcome)
    except Exception as exc:  # noqa: BLE001
        log.warning("tick.registry_sync_failed", strategy_code=strategy_code, error=str(exc))


async def _finish_iteration(
    *,
    db: Database,
    agent_name: str,
    session_id: UUID | None,
    redis_client: AsyncRedis | None,
    queue_id: Any,
    strategy_code: str,
    sweep_config: dict[str, Any],
    new_iter: int,
    iter_budget: int,
    early_stop: bool,
    backtest_run_id: str,
    iteration_id: Any,
    stat_v: str,
    dec_v: str,
    analysis: dict[str, Any],
) -> TickResult:
    """Activity log + step 6 next-state + result assembly. Shared by the
    fresh-iteration path and the adopt-existing-iteration resume path."""
    # Activity log: ITERATION_COMPLETED (fire-and-forget)
    try:
        async with db.acquire() as _log_conn:
            await log_activity(
                _log_conn,
                session_id=session_id,
                agent_name=agent_name,
                activity_type="ITERATION_COMPLETED",
                title=(
                    f"{strategy_code} iter {new_iter}: "
                    f"{stat_v} / {dec_v}"
                ),
                strategy_code=strategy_code,
                details={
                    "statistical_verdict": stat_v,
                    "verdict": dec_v,
                    "backtest_run_id": backtest_run_id,
                    "iteration_id": str(iteration_id),
                    "queue_id": str(queue_id),
                    "iter": new_iter,
                    "n_trades": analysis.get("n_trades"),
                    "pf_95_ci": analysis.get("pf_95_ci"),
                },
                related_id=iteration_id if isinstance(iteration_id, UUID) else None,
                related_type="iteration",
                redis_client=redis_client,
            )
    except Exception as _act_exc:  # noqa: BLE001
        log.warning("tick.activity_log_failed", step="ITERATION_COMPLETED", error=str(_act_exc))

    # Mirror the result into the Strategy Research Registry (best-effort; never
    # affects the tick). Keeps the registry current on every research iteration.
    await _sync_registry_best_effort(
        db,
        strategy_code=strategy_code,
        backtest_run_id=backtest_run_id,
        statistical_verdict=stat_v,
        agent_name=agent_name,
    )

    # ── Step 6: queue next-state ──────────────────────────────────────
    next_actions = await _decide_next_state(
        db=db,
        queue_id=queue_id,
        new_iter=new_iter,
        iter_budget=iter_budget,
        early_stop=early_stop,
        statistical_verdict=stat_v,
        decision_verdict=dec_v,
        agent_name=agent_name,
        iteration_id=iteration_id,
        dsr=analysis.get("dsr") if isinstance(analysis, dict) else None,
        track=(sweep_config or {}).get("track"),
    )

    _pf_pt = analysis.get("pf_point_estimate") if isinstance(analysis, dict) else None
    _pf_f: float | None
    try:
        _pf_f = float(_pf_pt) if _pf_pt is not None else None
    except (TypeError, ValueError):
        _pf_f = None
    _n_trades = analysis.get("n_trades") if isinstance(analysis, dict) else None
    try:
        _n_trades_i = int(_n_trades) if _n_trades is not None else None
    except (TypeError, ValueError):
        _n_trades_i = None

    return TickResult(
        outcome="iterated",
        queue_id=str(queue_id),
        iteration_id=str(iteration_id),
        backtest_run_id=backtest_run_id,
        statistical_verdict=stat_v,
        verdict=dec_v,
        next_actions=next_actions,
        pf=_pf_f,
        n_trades=_n_trades_i,
    )


async def _select_track_gate(
    *,
    db: Database,
    track: str | None,
    stat_v: str,
    dec_v: str,
    backtest_run_id: Any,
    instrument: str,
    window_start: Any,
    window_end: Any,
) -> tuple[str, str, dict[str, Any] | None]:
    """Choose the graduation gate by the queue row's track (Phase 2, 2026-06-07).

    Returns ``(statistical_verdict, decision_verdict, hedging_gate_payload)``.

    track in (None, "trading") — the DEFAULT — returns the V11/V60-derived
    ``(stat_v, dec_v)`` UNCHANGED and ``None`` for the payload. The trading
    path is therefore byte-for-byte identical: when track is not "hedging" this
    helper is a pass-through and does NO database work.

    track == "hedging" is an operator-authorized, ADDITIVE objective — beats
    buy-hold on a risk-adjusted, equity-level basis — that REPLACES the
    trade-level V11 PASS requirement and the V60 >=10%/yr economic check for
    hedging rows ONLY. The hedging gate computes its own pass/fail, which we map
    onto the SAME (stat_v, dec_v) control-flow signals the trading path already
    feeds to ``_decide_next_state`` so downstream routing is reused verbatim:
        gate PASS → ("SIGNIFICANT_EDGE", "PASS")    → routes to /walk-forward
        gate FAIL → ("SIGNIFICANT_EDGE", "ITERATE") → the exact "edge not
                    sufficient" ITERATE/continue path the V60-miss case uses.
    Walk-forward ROBUST remains a hard downstream requirement either way.
    V11/V60 constants and the trading path's logic are NOT touched here.
    """
    if track != "hedging":
        return stat_v, dec_v, None

    # Benchmark the underlying on DAILY closes regardless of the strategy's bar
    # interval (buy-hold is an equity-level, daily-frequency baseline).
    #
    # Failure semantics (fixed 2026-06-12): an exception here is an INFRA
    # failure, not a gate verdict. The prior arm mapped it onto the same
    # ("SIGNIFICANT_EDGE", "ITERATE") signal a genuine gate FAIL produces —
    # writing a fabricated stat verdict into the append-only audit tables,
    # consuming an iteration of budget for a combo whose gate never ran, and
    # (on the last budgeted iteration) terminalizing the queue with a
    # research conclusion reached from a DB blip. Raise retryable instead:
    # the outer handler rolls the row back to PENDING and the next tick
    # resumes the SAME completed run (active_backtest_run_id) and re-runs
    # only the gate.
    try:
        async with db.acquire() as conn:
            payload = await hedging_gate.evaluate(
                conn,
                backtest_run_id=backtest_run_id,
                symbol=instrument,
                interval="1d",
                start=window_start,
                end=window_end,
            )
    except Exception as exc:
        log.exception(
            "tick.hedging_gate_error",
            instrument=instrument,
            backtest_run_id=str(backtest_run_id),
        )
        raise OrchestratorError(
            status_code=503,
            error_code="hedging_gate_error",
            message=(
                f"Hedging gate evaluation failed for run {backtest_run_id}: "
                f"{type(exc).__name__}: {exc}. The backtest itself completed; "
                f"retry the tick to re-evaluate the gate on the same run."
            ),
            retryable=True,
            details={"backtest_run_id": str(backtest_run_id)},
        ) from exc
    # Drive existing control flow off the equity-level verdict. V11/V60 are
    # intentionally bypassed for hedging rows.
    new_stat_v = "SIGNIFICANT_EDGE"
    new_dec_v = "PASS" if payload["passed"] else "ITERATE"
    return new_stat_v, new_dec_v, payload


async def _decide_next_state(
    *,
    db: Database,
    queue_id: Any,
    new_iter: int,
    iter_budget: int,
    early_stop: bool,
    statistical_verdict: str,
    decision_verdict: str,
    agent_name: str,
    iteration_id: Any = None,
    dsr: float | None = None,
    track: str | None = None,
) -> list[dict[str, Any]]:
    """Mirrors research-tick.sh lines 514-572. Returns agent next-action hints.

    When the queue reaches a terminal state (COMPLETED or PARKED), a research
    paper is auto-generated so the frontend shows an up-to-date paper without
    requiring a manual POST /papers/{queue_id}/generate.  Failures are logged
    but non-fatal — the tick result is not rolled back on paper-gen errors.
    """
    ts = datetime.now(timezone.utc).isoformat()
    if statistical_verdict == "NO_EDGE" and early_stop:
        note = f"[{ts}] Discarded at iter {new_iter}: stat verdict NO_EDGE + early_stop=true."
        async with db.acquire() as conn:
            affected = await queue_write.complete_queue(
                conn,
                queue_id,
                final_verdict="DISCARD",
                walk_forward_id=None,
                note=note,
                expected_status="RUNNING",
            )
        if affected == 0:
            return _transition_lost(queue_id, "COMPLETED/DISCARD")
        await _try_generate_paper(db, queue_id, agent_name, state="COMPLETED/DISCARD")
        return [{"kind": "call", "method": "GET", "path": "/journal?entry_type=ANTI_PATTERN"}]

    if statistical_verdict == "SIGNIFICANT_EDGE":
        # V60 economic gate (CLAUDE.md): SIGNIFICANT_EDGE is graduation-
        # eligible ONLY when the decision_verdict also PASSes the 10%/yr
        # annualized geometric-return threshold. A stat edge that misses the
        # economic bar is real but doesn't justify promotion → ITERATE:
        # keep sweeping for a higher-return cell rather than parking.
        #
        # BUGFIX 2026-06-02: previously this branch parked for ALL
        # SIGNIFICANT_EDGE regardless of decision_verdict. When V60 failed,
        # the row went PARKED but the tick still returned a CONTINUE-shaped
        # next_action, so a drain would call /tick again on what is now an
        # empty queue and terminate EMPTY_QUEUE — silently stranding a valid
        # stat-edge sweep mid-search.
        if decision_verdict == "PASS":
            # Walk-forward is a 6-fold × ~30min validation; running it inside
            # /tick would block the HTTP request for hours. Park the row and
            # direct the agent to POST /walk-forward, which (if ROBUST) flips
            # the queue to COMPLETED/PASS.
            note = (
                f"[{ts}] iter {new_iter} SIGNIFICANT_EDGE + V60 PASS — needs "
                f"walk-forward validation. POST /walk-forward to gate PASS."
            )
            async with db.acquire() as conn:
                affected = await queue_write.park_exhausted(
                    conn, queue_id, note, expected_status="RUNNING"
                )
            if affected == 0:
                return _transition_lost(queue_id, "PARKED/SIGNIFICANT_EDGE")
            await _try_generate_paper(
                db, queue_id, agent_name, state="PARKED/SIGNIFICANT_EDGE"
            )
            return [
                {"kind": "call", "method": "POST", "path": "/walk-forward",
                 "hint": "Body {strategy_code, queue_id} — gates PASS on stability_verdict=ROBUST."},
            ]
        # Stat edge real but economic return below 10%/yr — keep iterating
        # (unless the budget is spent) to look for a higher-return cell.
        if new_iter >= iter_budget:
            note = (
                f"[{ts}] iter {new_iter} SIGNIFICANT_EDGE but decision_verdict="
                f"{decision_verdict} (V60 economic gate not met) and iter_budget "
                f"exhausted. Completing as ITERATE_EXHAUSTED."
            )
            async with db.acquire() as conn:
                affected = await queue_write.complete_queue(
                    conn,
                    queue_id,
                    final_verdict="INSUFFICIENT_EVIDENCE",
                    walk_forward_id=None,
                    note=note,
                    expected_status="RUNNING",
                )
            if affected == 0:
                return _transition_lost(queue_id, "COMPLETED/ITERATE_EXHAUSTED")
            await _try_generate_paper(
                db, queue_id, agent_name, state="COMPLETED/ITERATE_EXHAUSTED"
            )
            # Signal Pool / House Book lane (Phase 0). A stat-real near-miss that
            # only failed the standalone 10%/yr bar is a candidate for the
            # managed pool — IF it adds uncorrelated return. Don't silently
            # discard it: when its DSR already clears the pool's validity bar,
            # route it to a pool-candidate walk-forward (the only remaining
            # validity check) instead of just enqueuing a fresh hypothesis. The
            # standalone V60 gate is untouched — this is the additive pool path.
            #
            # Fix 3 (2026-06-07): this trade-level pool lane gates on the
            # TRADE-LEVEL DSR, which is the WRONG objective for a hedging
            # candidate — the hedging gate intentionally bypasses trade-level
            # stats (it judges equity-level beats-buy-hold). A failing hedging
            # row arrives here as ("SIGNIFICANT_EDGE", "ITERATE") and could be
            # mis-routed into the trade-level pool on a coincidental DSR. Restrict
            # this branch to the trading track (None ⇒ legacy/default). Hedging
            # FAILs just continue/ITERATE below without touching this pool.
            actions: list[dict[str, Any]] = []
            if (
                track in (None, "trading")
                and dsr is not None
                and dsr >= analyze.DSR_SIGNIFICANCE_THRESHOLD
                and iteration_id is not None
            ):
                actions.append({
                    "kind": "call", "method": "POST", "path": "/walk-forward",
                    "hint": (
                        "Pool candidate — statistically real (DSR "
                        f"{dsr:.3f} ≥ {analyze.DSR_SIGNIFICANCE_THRESHOLD}) but "
                        "sub-10%/yr standalone. Body {strategy_code, interval_name, "
                        f"instrument, motivating_iteration_id: '{iteration_id}', "
                        "pool_candidate: true}. On stability_verdict=ROBUST the "
                        "response routes you to POST /pool/evaluate "
                        f"{{iteration_id: '{iteration_id}'}} to admit it to the "
                        "House Book (admitted only if it ADDS uncorrelated return)."
                    ),
                })
            actions.append({"kind": "call", "method": "POST", "path": "/queue"})
            return actions
        note = (
            f"[{ts}] iter {new_iter} SIGNIFICANT_EDGE but decision_verdict="
            f"{decision_verdict} (V60 economic gate not met); continuing sweep."
        )
        async with db.acquire() as conn:
            affected = await queue_write.reset_to_pending(
                conn, queue_id, note, expected_status="RUNNING"
            )
        if affected == 0:
            return _transition_lost(queue_id, "PENDING/continue")
        return [{"kind": "call", "method": "POST", "path": "/tick"}]

    if statistical_verdict in ("INSUFFICIENT_EVIDENCE", "NOT_TESTED"):
        if new_iter >= iter_budget:
            note = (
                f"[{ts}] Parked at iter {new_iter}: still INSUFFICIENT_EVIDENCE "
                f"after iter_budget={iter_budget}. Human review."
            )
            async with db.acquire() as conn:
                affected = await queue_write.complete_queue(
                    conn,
                    queue_id,
                    final_verdict="INSUFFICIENT_EVIDENCE",
                    walk_forward_id=None,
                    note=note,
                    expected_status="RUNNING",
                )
            if affected == 0:
                return _transition_lost(queue_id, "COMPLETED/INSUFFICIENT_EVIDENCE")
            await _try_generate_paper(db, queue_id, agent_name, state="COMPLETED/INSUFFICIENT_EVIDENCE")
            return [{"kind": "contact_human"}]
        note = f"[{ts}] iter {new_iter} INSUFFICIENT_EVIDENCE; continuing."
        async with db.acquire() as conn:
            affected = await queue_write.reset_to_pending(
                conn, queue_id, note, expected_status="RUNNING"
            )
        if affected == 0:
            return _transition_lost(queue_id, "PENDING/continue")
        return [{"kind": "call", "method": "POST", "path": "/tick"}]

    # NO_EDGE without early_stop, or unknown. The iter_budget applies here
    # too — before 2026-06-12 this branch never checked it, so a grid sweep
    # with early_stop_on_no_edge=false ran to grid exhaustion regardless of
    # budget, each NO_EDGE cell adding a trial to DSR multiplicity.
    if new_iter >= iter_budget:
        final = "DISCARD" if statistical_verdict == "NO_EDGE" else "INSUFFICIENT_EVIDENCE"
        note = (
            f"[{ts}] iter {new_iter} statistical_verdict={statistical_verdict}; "
            f"iter_budget={iter_budget} exhausted. Completing as {final}."
        )
        async with db.acquire() as conn:
            affected = await queue_write.complete_queue(
                conn,
                queue_id,
                final_verdict=final,
                walk_forward_id=None,
                note=note,
                expected_status="RUNNING",
            )
        if affected == 0:
            return _transition_lost(queue_id, f"COMPLETED/{final}")
        await _try_generate_paper(db, queue_id, agent_name, state=f"COMPLETED/{final}")
        return [{"kind": "call", "method": "GET", "path": "/journal?entry_type=ANTI_PATTERN"}]
    note = f"[{ts}] iter {new_iter} statistical_verdict={statistical_verdict}; continuing."
    async with db.acquire() as conn:
        affected = await queue_write.reset_to_pending(
            conn, queue_id, note, expected_status="RUNNING"
        )
    if affected == 0:
        return _transition_lost(queue_id, "PENDING/continue")
    return [{"kind": "call", "method": "POST", "path": "/tick"}]


def _transition_lost(queue_id: Any, intended: str) -> list[dict[str, Any]]:
    """Step-6 transition affected 0 rows: the operator PARKed/cancelled the
    row while the tick was in flight. The iteration evidence is committed;
    honour the cancel by NOT resurrecting the row."""
    log.warn(
        "tick.transition_skipped_row_not_running",
        queue_id=str(queue_id),
        intended_state=intended,
    )
    return [{
        "kind": "call", "method": "GET", "path": f"/queue/{queue_id}",
        "hint": (
            "Queue row left RUNNING-state machine mid-tick (operator cancel). "
            "Iteration was logged; row state was preserved."
        ),
    }]


async def _try_generate_paper(
    db: Database,
    queue_id: Any,
    agent_name: str,
    state: str,
) -> None:
    """Fire-and-forget paper generation at queue terminal state.

    Non-fatal: a failure here is logged but does not roll back the tick.
    The paper can always be regenerated manually via
    POST /papers/{queue_id}/generate.
    """
    try:
        async with db.acquire() as conn:
            await paper_generator.generate_paper(conn, queue_id, agent_name)
        log.info("tick.paper_generated", queue_id=str(queue_id), state=state)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "tick.paper_generation_failed",
            queue_id=str(queue_id),
            state=state,
            error=str(exc),
        )
