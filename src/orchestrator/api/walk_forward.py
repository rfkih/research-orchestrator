"""``POST /walk-forward`` — run a 6-fold walk-forward validation.

Long-running on purpose: 6 folds × up to 30min/poll = up to 3 hours.
Agents should call this deliberately after a tick verdict of
``SIGNIFICANT_EDGE`` (which parks the queue and emits a next_action
pointing here). The PASS gate in the iteration verdict requires
``stability_verdict = ROBUST`` from this endpoint.

Idempotency: ``Idempotency-Key`` honoured under the ``walk:`` namespace.
A retry of an in-flight call still triggers a new run (replay is by key,
not by request shape) — keep keys unique per logical attempt.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any
from uuid import UUID

import asyncpg
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from ..errors import NextAction, OrchestratorError
from ..repo import queue as queue_repo
from ..repo import reviews as reviews_repo
from ..repo import walk_forward as walk_forward_repo
from ..services.activity_logger import log_activity
from ..services.idempotency import cache_response, replay_cached_response
from ..services.walk_forward import (
    DEFAULT_FULL_START,
    LOW_FREQ_MIN_TOTAL_TRADES,
    is_low_frequency,
    run_walk_forward,
)
from .constants import REVIEW_VERDICTS_PASSING_GATE
from .deps import get_agent_name, get_db_conn

router = APIRouter(tags=["walk-forward"])


class WalkForwardRunDetailResponse(BaseModel):
    """Curator-relevant subset of a ``walk_forward_run`` row."""

    walk_forward_id: UUID
    strategy_code: str
    instrument: str
    interval_name: str
    stability_verdict: str
    n_folds: int
    fold_pf_mean: float | None = None
    fold_pf_std: float | None = None
    fold_pf_min: float | None = None
    fold_pf_max: float | None = None
    fold_pf_positive_pct: float | None = None
    fold_sharpe_mean: float | None = None
    fold_sharpe_std: float | None = None
    fold_return_mean: float | None = None
    fold_return_std: float | None = None
    total_trades_across_folds: int | None = None
    fold_results: list[Any] = Field(default_factory=list)
    motivating_iteration_id: UUID | None = None
    created_time: datetime | None = None
    created_by: str | None = None


class WalkForwardRequest(BaseModel):
    strategy_code: str = Field(..., min_length=1, max_length=60)
    interval_name: str = Field(..., description="5m|15m|1h|4h")
    instrument: str = Field("BTCUSDT", min_length=1, max_length=30)
    # 2018 floor (was 2024-01-01) so the default validation window spans real
    # bears; the service rejects any window with no drawdown (see
    # override_bear_coverage). BTC market_data is backfilled to 2017-08.
    full_start: date = Field(default_factory=lambda: DEFAULT_FULL_START)
    full_end: date | None = None
    train_months: int = Field(12, ge=1, le=60)
    test_months: int = Field(3, ge=1, le=24)
    step_months: int = Field(3, ge=1, le=24)
    n_folds: int = Field(6, ge=1, le=20)
    overrides: dict[str, Any] | None = None
    motivating_iteration_id: UUID | None = None
    # Paired-agent review gate (added 2026-05-05): /walk-forward refuses
    # without an APPROVED graduation review for motivating_iteration_id.
    # override_review_gate=true requires operator-level justification.
    override_review_gate: bool = False
    # Signal Pool / House Book lane (Phase 0, 2026-06-03). When true, this
    # walk-forward validates a sub-10% near-miss for *pool admission*, not
    # graduation to live capital. The graduation-review gate is skipped (pool
    # admission has its own gate in POST /pool/evaluate), the trade-frequency
    # guard still applies, and a ROBUST verdict routes to /pool/evaluate.
    pool_candidate: bool = False
    # Bear-coverage gate (2026-06-09). The service rejects a walk-forward whose
    # window contains no known bear regime — a bull-only OOS verdict is not
    # evidence. Set true ONLY for instruments whose history genuinely predates
    # every known bear, with a documented journal entry.
    override_bear_coverage: bool = False
    # DSR selection-bias honesty (2026-06-14 re-audit HOLE-1). Trials run
    # OUTSIDE the orchestrator for this (instrument, interval_name) family —
    # offline sweeps, hand-tuning, direct JVM backtests — that hypothesis_audit
    # cannot see. Added into the low-frequency walk-forward DSR ``n_trials`` so
    # the equity-curve significance bar reflects the TRUE multiplicity. Default
    # 0. Monotone: can only raise n_trials → the gate only TIGHTENS (V11-safe).
    # Set this ONLY for off-orchestrator trials NOT already declared on the
    # sweep that produced this candidate — re-declaring the same trials here
    # double-counts the family's multiplicity and may reject a real edge.
    external_trials: int = Field(0, ge=0, le=2000)


class WalkForwardResponse(BaseModel):
    walk_forward_id: str
    stability_verdict: str
    aggregate: dict[str, Any]
    fold_results: list[dict[str, Any]]
    next_actions: list[dict[str, Any]]


@router.post("/walk-forward", response_model=WalkForwardResponse)
async def post_walk_forward(
    body: WalkForwardRequest,
    request: Request,
    agent: str = Depends(get_agent_name),
) -> dict[str, Any]:
    # NO Depends(get_db_conn) here: a yield-dependency connection is held
    # until the response is sent, and this handler runs a multi-HOUR
    # walk-forward — pinning 1 of db_pool_max=10 connections idle for the
    # duration (and the dependency conn could be dead by the time the
    # post-run log_activity used it). Short-lived acquisitions only.
    db = request.app.state.db
    cached, idempotency_key = await replay_cached_response(request, agent, "walk")
    if cached is not None:
        return cached

    # Graduation review gate. Walk-forward consumes ~3h of JVM time and
    # is the formal step before promotion. Without an APPROVED reviewer
    # verdict on motivating_iteration_id, we don't burn that budget.
    if body.motivating_iteration_id is None and not body.override_review_gate:
        raise OrchestratorError(
            status_code=400,
            error_code="motivating_iteration_id_required",
            message=(
                "POST /walk-forward requires motivating_iteration_id (the "
                "iteration that triggered SIGNIFICANT_EDGE) so the review "
                "gate can find the matching APPROVED graduation verdict. "
                "Pass override_review_gate=true with documented justification "
                "to bypass."
            ),
            retryable=False,
            hint=(
                "Trigger /tick first, observe SIGNIFICANT_EDGE iteration, "
                "POST /reviews/request with target_kind='graduation', wait "
                "for APPROVED, then re-submit /walk-forward."
            ),
        )
    if (
        body.motivating_iteration_id is not None
        and not body.override_review_gate
        and not body.pool_candidate
    ):
        target_id = reviews_repo.graduation_target_id(body.motivating_iteration_id)
        async with db.acquire() as conn:
            latest = await reviews_repo.fetch_latest_verdict(conn, target_id)
        sd = (latest or {}).get("structured_data") or {}
        verdict = sd.get("verdict")
        if verdict not in REVIEW_VERDICTS_PASSING_GATE:
            raise OrchestratorError(
                status_code=409,
                error_code="graduation_review_required",
                message=(
                    f"No APPROVED graduation review for "
                    f"iteration_id={body.motivating_iteration_id}. "
                    f"Latest verdict: {verdict or 'NONE'}."
                ),
                retryable=False,
                hint=(
                    "Submit POST /reviews/request with target_kind='graduation' "
                    "{iteration_id}, wait for the reviewer to post APPROVED, "
                    "then re-submit /walk-forward."
                ),
                next_action=NextAction(
                    kind="call",
                    method="POST",
                    path="/reviews/request",
                ),
                details={
                    "target_id": target_id,
                    "latest_verdict": verdict,
                    "iteration_id": str(body.motivating_iteration_id),
                },
            )

    # Trade-frequency guard. Two tracks (frequency-aware, 2026-06-07):
    #   • Standard (high-turnover): fold-level PF estimates need ≥30 trades/fold,
    #     so require n_folds×30 in-sample trades.
    #   • Low-frequency (1d / structurally low-turnover): trade-count folds are
    #     inapplicable — the walk-forward validates on the OOS daily-equity
    #     return series instead — so only a small absolute floor applies. The
    #     statistical bars (DSR≥0.95, positive-fold≥60%) are unchanged; see
    #     services/walk_forward.low_freq_stability_verdict.
    _MIN_TRADES_PER_FOLD = 30
    if body.motivating_iteration_id is not None:
        async with db.acquire() as conn:
            iter_row = await conn.fetchrow(
                "SELECT metrics_snapshot FROM research_iteration_log WHERE iteration_id = $1",
                body.motivating_iteration_id,
            )
        if iter_row is not None:
            ms = iter_row["metrics_snapshot"] or {}
            n_trades = int(ms.get("total_trades") or 0)
            window_end = body.full_end or (
                datetime.now(timezone.utc).date() - timedelta(days=1)
            )
            window_days = (window_end - body.full_start).days
            if is_low_frequency(body.interval_name, n_trades, window_days):
                if n_trades < LOW_FREQ_MIN_TOTAL_TRADES:
                    raise OrchestratorError(
                        status_code=422,
                        error_code="insufficient_trade_frequency",
                        message=(
                            f"Iteration {body.motivating_iteration_id} is a "
                            f"low-frequency ({body.interval_name}) strategy with "
                            f"{n_trades} in-sample trades, below the low-frequency "
                            f"floor of {LOW_FREQ_MIN_TOTAL_TRADES}. Even validated on "
                            f"the equity curve, fewer trades can't establish a stable "
                            f"edge."
                        ),
                        retryable=False,
                        hint=(
                            "Extend the backtest window (e.g. full_start=2022-01-01) "
                            "so more trades and out-of-sample daily returns accumulate."
                        ),
                        details={
                            "n_trades": n_trades,
                            "validation_track": "LOW_FREQUENCY",
                            "required_trades": LOW_FREQ_MIN_TOTAL_TRADES,
                        },
                    )
            else:
                required = body.n_folds * _MIN_TRADES_PER_FOLD
                if n_trades < required:
                    raise OrchestratorError(
                        status_code=422,
                        error_code="insufficient_trade_frequency",
                        message=(
                            f"Iteration {body.motivating_iteration_id} has {n_trades} "
                            f"in-sample trades, below the {body.n_folds}-fold minimum of "
                            f"{required} ({_MIN_TRADES_PER_FOLD}/fold). Walk-forward fold "
                            f"estimates would be statistically unreliable at "
                            f"~{n_trades // body.n_folds} trades/fold."
                        ),
                        retryable=False,
                        hint=(
                            f"Options: (1) find params producing ≥{required} trades on the "
                            f"current window, (2) extend market_data backfill period so the "
                            f"backtest window grows, or (3) reduce n_folds to "
                            f"{max(1, n_trades // _MIN_TRADES_PER_FOLD)} "
                            f"(minimum {_MIN_TRADES_PER_FOLD} trades/fold)."
                        ),
                        details={
                            "n_trades": n_trades,
                            "n_folds": body.n_folds,
                            "required_trades": required,
                            "min_trades_per_fold": _MIN_TRADES_PER_FOLD,
                            "estimated_trades_per_fold": n_trades // body.n_folds,
                        },
                    )

    session_id = request.state.session_id
    redis_client = request.app.state.redis

    # Activity logs below: log_activity is itself fire-and-forget — it
    # swallows internal errors so a busted insert can't fail the request.
    async with db.acquire() as conn:
        await log_activity(
            conn,
            session_id=session_id,
            agent_name=agent,
            activity_type="WALK_FORWARD_SUBMITTED",
            title=f"Walk-forward submitted for {body.strategy_code}",
            strategy_code=body.strategy_code,
            details={
                "motivating_iteration_id": str(body.motivating_iteration_id) if body.motivating_iteration_id else None,
                "interval_name": body.interval_name,
                "instrument": body.instrument,
                "n_folds": body.n_folds,
                "train_months": body.train_months,
                "test_months": body.test_months,
            },
            related_id=body.motivating_iteration_id,
            related_type="iteration",
            redis_client=redis_client,
        )

    try:
        result = await run_walk_forward(
            db=request.app.state.db,
            jvm=request.app.state.jvm,
            settings=request.app.state.settings,
            agent_name=agent,
            strategy_code=body.strategy_code,
            interval_name=body.interval_name,
            instrument=body.instrument,
            full_start=body.full_start,
            full_end=body.full_end,
            train_months=body.train_months,
            test_months=body.test_months,
            step_months=body.step_months,
            n_folds=body.n_folds,
            overrides=body.overrides,
            motivating_iteration_id=body.motivating_iteration_id,
            require_bear_coverage=not body.override_bear_coverage,
            external_trials=body.external_trials,
        )
    except Exception as _wf_exc:
        async with db.acquire() as conn:
            await log_activity(
                conn,
                session_id=session_id,
                agent_name=agent,
                activity_type="WALK_FORWARD_RESULT",
                title=f"Walk-forward failed for {body.strategy_code}",
                strategy_code=body.strategy_code,
                details={"error": str(_wf_exc)},
                related_id=body.motivating_iteration_id,
                related_type="iteration",
                status="ERROR",
                redis_client=redis_client,
            )
        raise

    payload = result.to_dict()
    next_actions: list[dict[str, Any]] = []
    if result.stability_verdict == "ROBUST" and body.pool_candidate:
        # Pool lane: ROBUST clears the last validity check. Route to the
        # House Book admission gate (marginal-Sharpe contribution vs the
        # current pool) rather than the graduation rule.
        next_actions.append({
            "kind": "call", "method": "POST", "path": "/pool/evaluate",
            "hint": (
                "ROBUST — pool candidate passes walk-forward. POST /pool/evaluate "
                f"{{iteration_id: '{body.motivating_iteration_id}'}} to admit it to "
                "the House Book (admitted only if it ADDS uncorrelated return)."
            ),
        })
    elif result.stability_verdict == "ROBUST":
        next_actions.append({
            "kind": "note",
            "hint": "ROBUST — strategy passes walk-forward. Promote per graduation rule.",
        })
    elif result.stability_verdict in {"OVERFIT", "INCONSISTENT"}:
        next_actions.append({
            "kind": "call", "method": "POST", "path": "/queue",
            "hint": f"{result.stability_verdict} — re-design with tighter regularisation, then re-enqueue.",
        })
    else:  # NO_EDGE | INSUFFICIENT_EVIDENCE
        next_actions.append({
            "kind": "read_doc", "doc_anchor": "walk_forward.no_edge",
            "hint": f"{result.stability_verdict} — abandon or extend window.",
        })
    payload["next_actions"] = next_actions

    async with db.acquire() as conn:
        await log_activity(
            conn,
            session_id=session_id,
            agent_name=agent,
            activity_type="WALK_FORWARD_RESULT",
            title=(
                f"Walk-forward result for {body.strategy_code}: "
                f"{result.stability_verdict}"
            ),
            strategy_code=body.strategy_code,
            details={
                "stability_verdict": result.stability_verdict,
                "walk_forward_id": payload.get("walk_forward_id"),
                "motivating_iteration_id": str(body.motivating_iteration_id) if body.motivating_iteration_id else None,
                "aggregate": payload.get("aggregate"),
            },
            related_id=body.motivating_iteration_id,
            related_type="iteration",
            redis_client=redis_client,
        )

    await cache_response(request, agent, "walk", idempotency_key, payload)
    return payload


class WalkForwardBacklogResponse(BaseModel):
    count: int
    items: list[dict[str, Any]]
    next_actions: list[dict[str, Any]] = Field(default_factory=list)


@router.get("/walk-forward/backlog", response_model=WalkForwardBacklogResponse)
async def get_walk_forward_backlog(
    request: Request,
    track: str | None = None,
    limit: int = 100,
    agent: str = Depends(get_agent_name),
    conn: asyncpg.Connection = Depends(get_db_conn),
) -> WalkForwardBacklogResponse:
    """List PARKED SIGNIFICANT_EDGE candidates still awaiting walk-forward.

    Surfaces the out-of-sample-validation bottleneck (Finding 2): SIG_EDGE
    candidates park awaiting a deliberate ~3h walk-forward and silently
    accumulate. This makes that backlog drainable — the loop/operator reads it
    and drives each candidate through the (review → walk-forward) path instead
    of discovering the pile-up by accident.

    Deliberately a READ. Walk-forward submits real multi-hour JVM load and
    requires an APPROVED graduation review first, so auto-spawning runs from a
    GET would be wrong; the ``next_actions`` tell the caller how to drain it.
    """
    items = await queue_repo.list_walk_forward_backlog(
        conn, track=track, limit=max(1, min(limit, 500))
    )
    next_actions: list[dict[str, Any]] = []
    if items:
        next_actions.append({
            "kind": "note",
            "hint": (
                f"{len(items)} SIG_EDGE candidate(s) awaiting walk-forward. For "
                "each: ensure an APPROVED graduation review exists "
                "(POST /reviews/request target_kind='graduation'), then POST "
                "/walk-forward {motivating_iteration_id} — now bear-coverage "
                "enforced. Drain oldest-first to clear the backlog."
            ),
        })
    return WalkForwardBacklogResponse(
        count=len(items), items=items, next_actions=next_actions
    )


@router.get("/walk-forward/runs/{walk_forward_id}", response_model=WalkForwardRunDetailResponse)
async def get_walk_forward_run(
    walk_forward_id: UUID,
    request: Request,
    agent: str = Depends(get_agent_name),
    conn: asyncpg.Connection = Depends(get_db_conn),
) -> WalkForwardRunDetailResponse:
    """Return curator-relevant fields for a single ``walk_forward_run`` row.

    Used by the curator agent (Lens B) to read ``stability_verdict`` after
    a graduated iteration's walk-forward run has completed.
    """
    row = await walk_forward_repo.fetch_walk_forward_run(conn, walk_forward_id)
    if row is None:
        raise OrchestratorError(
            status_code=404,
            error_code="walk_forward_run_not_found",
            message=f"walk_forward_run {walk_forward_id} not found",
            retryable=False,
        )
    return WalkForwardRunDetailResponse(**row)
