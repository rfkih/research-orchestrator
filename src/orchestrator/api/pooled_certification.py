"""``POST /pooled-certification/*`` — orchestrator-side pooled-book
certification (operator methodology approval 2026-07-10).

Two long-running synchronous endpoints (platform style — the agent
backgrounds the HTTP call and polls DB-visible artifacts):

  * ``POST /pooled-certification/analyze`` — full-window coordinated
    single-symbol runs per sleeve, pooled trade-level scoring with the
    UNCHANGED V11 primitives, persisted as a ``research_iteration_log``
    row on the book strategy_code. Requires a pre-registered ACTIVE
    HYPOTHESIS journal row (``motivating_hypothesis_id``).
  * ``POST /pooled-certification/walk-forward`` — pooled-series rolling-
    fold validation. Gated on an APPROVED graduation review for the
    pooled iteration (same 409 contract as ``POST /walk-forward``).

See ``services/pooled_certification.py`` for the statistical contract and
the documented equal-slice sizing model.
"""

from __future__ import annotations

from datetime import date
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from ..errors import NextAction, OrchestratorError
from ..logging import get_logger
from ..repo import journal as journal_repo
from ..repo import reviews as reviews_repo
from ..services.activity_logger import log_activity
from ..services.idempotency import cache_response, replay_cached_response
from ..services.pooled_certification import (
    MAX_SLEEVES,
    MIN_SLEEVES,
    SleeveSpec,
    run_pooled_analyze,
    run_pooled_walk_forward,
)
from .constants import REVIEW_VERDICTS_PASSING_GATE, VALID_INTERVAL_NAMES
from .deps import get_agent_name

router = APIRouter(prefix="/pooled-certification", tags=["pooled-certification"])

log = get_logger(__name__)


class SleeveModel(BaseModel):
    strategy_code: str = Field(..., min_length=1, max_length=60)
    instrument: str = Field(..., min_length=1, max_length=30)
    interval_name: str = Field(..., description="5m|15m|1h|4h|1d")
    window_start: date
    overrides: dict[str, Any] = Field(default_factory=dict)


class PooledAnalyzeRequest(BaseModel):
    strategy_code: str = Field(..., min_length=1, max_length=60,
                               description="BOOK strategy_code, e.g. DCB_POOL")
    sleeves: list[SleeveModel] = Field(..., min_length=MIN_SLEEVES,
                                       max_length=MAX_SLEEVES)
    full_end: date | None = None
    # DSR selection-bias honesty: off-ledger trials for the POOLED surface
    # (e.g. offline pool-variant selection). Monotone — can only raise the
    # server-computed multiplicity, so the gate only TIGHTENS.
    external_trials: int = Field(0, ge=0, le=2000)
    motivating_hypothesis_id: UUID
    notes: str | None = Field(None, max_length=2000)


class PooledWalkForwardRequest(BaseModel):
    iteration_id: UUID
    train_months: int = Field(12, ge=1, le=60)
    test_months: int = Field(3, ge=1, le=24)
    step_months: int = Field(3, ge=1, le=24)
    n_folds: int = Field(18, ge=1, le=20)
    # Bear-coverage gate — same semantics as POST /walk-forward. Set true
    # ONLY for books whose sleeves' history genuinely predates every known
    # bear, with a documented journal entry.
    override_bear_coverage: bool = False


@router.post("/analyze")
async def post_pooled_analyze(
    body: PooledAnalyzeRequest,
    request: Request,
    agent: str = Depends(get_agent_name),
) -> dict[str, Any]:
    # No Depends(get_db_conn): this handler runs multiple full-window
    # backtests (minutes to an hour) — short-lived acquisitions only,
    # mirroring POST /walk-forward.
    db = request.app.state.db
    cached, idempotency_key = await replay_cached_response(
        request, agent, "pooled-analyze"
    )
    if cached is not None:
        return cached

    for s in body.sleeves:
        if s.interval_name not in VALID_INTERVAL_NAMES:
            raise OrchestratorError(
                status_code=400,
                error_code="invalid_interval",
                message=(
                    f"Sleeve {s.instrument} interval_name={s.interval_name!r} "
                    f"must be one of {sorted(VALID_INTERVAL_NAMES)}."
                ),
                retryable=False,
            )

    # Pre-registration gate: the pooled certification is an experiment and
    # must trace to an ACTIVE HYPOTHESIS row written BEFORE the compute —
    # the same discipline /queue enforces via the plan-review gate.
    async with db.acquire() as conn:
        hyp = await journal_repo.get_journal(conn, body.motivating_hypothesis_id)
    if hyp is None or (hyp.get("entry_type") or "").upper() != "HYPOTHESIS":
        raise OrchestratorError(
            status_code=412,
            error_code="pooled_hypothesis_missing",
            message=(
                f"motivating_hypothesis_id={body.motivating_hypothesis_id} is "
                "not a HYPOTHESIS journal row. Pre-register the hypothesis "
                "before running a pooled certification."
            ),
            retryable=False,
            next_action=NextAction(kind="call", method="POST", path="/journal"),
        )

    session_id = request.state.session_id
    redis_client = request.app.state.redis
    async with db.acquire() as conn:
        await log_activity(
            conn,
            session_id=session_id,
            agent_name=agent,
            activity_type="POOLED_CERT_ANALYZE_SUBMITTED",
            title=f"Pooled certification analyze submitted for {body.strategy_code}",
            strategy_code=body.strategy_code,
            details={
                "n_sleeves": len(body.sleeves),
                "sleeves": [
                    f"{s.instrument}/{s.interval_name}" for s in body.sleeves
                ],
                "motivating_hypothesis_id": str(body.motivating_hypothesis_id),
            },
            related_id=body.motivating_hypothesis_id,
            related_type="journal",
            redis_client=redis_client,
        )

    result = await run_pooled_analyze(
        db=db,
        jvm=request.app.state.jvm,
        settings=request.app.state.settings,
        agent_name=agent,
        book_strategy_code=body.strategy_code,
        sleeves=[
            SleeveSpec(
                strategy_code=s.strategy_code,
                instrument=s.instrument,
                interval_name=s.interval_name,
                window_start=s.window_start,
                overrides=s.overrides,
            )
            for s in body.sleeves
        ],
        full_end=body.full_end,
        external_trials=body.external_trials,
        motivating_hypothesis_id=body.motivating_hypothesis_id,
        notes=body.notes,
    )
    response = {
        **result,
        "next_actions": [
            {
                "kind": "call",
                "method": "POST",
                "path": "/reviews/request",
                "hint": (
                    "If statistical_verdict=SIGNIFICANT_EDGE: request a "
                    "graduation review for iteration_id, then "
                    "POST /pooled-certification/walk-forward."
                ),
            }
        ],
    }
    await cache_response(request, agent, "pooled-analyze", idempotency_key, response)
    return response


@router.post("/walk-forward")
async def post_pooled_walk_forward(
    body: PooledWalkForwardRequest,
    request: Request,
    agent: str = Depends(get_agent_name),
) -> dict[str, Any]:
    db = request.app.state.db
    cached, idempotency_key = await replay_cached_response(
        request, agent, "pooled-wf"
    )
    if cached is not None:
        return cached

    # Graduation-review gate — byte-identical contract to POST /walk-forward:
    # no APPROVED (or CONDITIONAL_APPROVAL) graduation verdict on the pooled
    # iteration means no multi-hour fold burn. No override parameter is
    # offered on the pooled path — reviewer verdict is authoritative.
    target_id = reviews_repo.graduation_target_id(body.iteration_id)
    async with db.acquire() as conn:
        latest = await reviews_repo.fetch_latest_verdict(conn, target_id)
    sd = (latest or {}).get("structured_data") or {}
    verdict = sd.get("verdict")
    if verdict not in REVIEW_VERDICTS_PASSING_GATE:
        raise OrchestratorError(
            status_code=409,
            error_code="graduation_review_required",
            message=(
                f"No APPROVED graduation review for pooled iteration_id="
                f"{body.iteration_id}. Latest verdict: {verdict or 'NONE'}."
            ),
            retryable=False,
            hint=(
                "Submit POST /reviews/request with target_kind='graduation' "
                "{iteration_id}, run /reviews/auto-run-checklist, then "
                "re-submit."
            ),
            next_action=NextAction(
                kind="call", method="POST", path="/reviews/request"
            ),
            details={
                "target_id": target_id,
                "latest_verdict": verdict,
                "iteration_id": str(body.iteration_id),
            },
        )

    session_id = request.state.session_id
    redis_client = request.app.state.redis
    async with db.acquire() as conn:
        await log_activity(
            conn,
            session_id=session_id,
            agent_name=agent,
            activity_type="POOLED_CERT_WF_SUBMITTED",
            title="Pooled certification walk-forward submitted",
            details={
                "iteration_id": str(body.iteration_id),
                "n_folds": body.n_folds,
                "train_months": body.train_months,
                "test_months": body.test_months,
            },
            related_id=body.iteration_id,
            related_type="iteration",
            redis_client=redis_client,
        )

    result = await run_pooled_walk_forward(
        db=db,
        jvm=request.app.state.jvm,
        settings=request.app.state.settings,
        agent_name=agent,
        iteration_id=body.iteration_id,
        train_months=body.train_months,
        test_months=body.test_months,
        step_months=body.step_months,
        n_folds=body.n_folds,
        require_bear_coverage=not body.override_bear_coverage,
    )
    await cache_response(request, agent, "pooled-wf", idempotency_key, result)
    return result
