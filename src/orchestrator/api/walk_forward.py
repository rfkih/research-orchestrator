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

import logging
from datetime import date
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from ..errors import NextAction, OrchestratorError
from ..repo import reviews as reviews_repo
from ..services.activity_logger import log_activity
from ..services.walk_forward import run_walk_forward
from .deps import get_agent_name, get_db_conn

logger = logging.getLogger(__name__)

router = APIRouter(tags=["walk-forward"])


class WalkForwardRequest(BaseModel):
    strategy_code: str = Field(..., min_length=1, max_length=60)
    interval_name: str = Field(..., description="5m|15m|1h|4h")
    instrument: str = Field("BTCUSDT", min_length=1, max_length=30)
    full_start: date = Field(default_factory=lambda: date(2024, 1, 1))
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
    conn=Depends(get_db_conn),
) -> dict[str, Any]:
    idempotency_key = getattr(request.state, "idempotency_key", None)
    store = request.app.state.idempotency
    if idempotency_key:
        cached = await store.get(agent, f"walk:{idempotency_key}")
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
    if body.motivating_iteration_id is not None and not body.override_review_gate:
        target_id = reviews_repo.graduation_target_id(body.motivating_iteration_id)
        latest = await reviews_repo.fetch_latest_verdict(conn, target_id)
        sd = (latest or {}).get("structured_data") or {}
        verdict = sd.get("verdict")
        if verdict not in ("APPROVED", "CONDITIONAL_APPROVAL"):
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

    session_id_str = request.headers.get("X-Session-Id")
    try:
        session_id: UUID | None = UUID(session_id_str) if session_id_str else None
    except ValueError:
        raise OrchestratorError(
            status_code=400,
            error_code="invalid_session_id",
            message=f"X-Session-Id is not a valid UUID: {session_id_str!r}",
            retryable=False,
        )
    redis_client = request.app.state.redis

    # Activity log: WALK_FORWARD_SUBMITTED (fire-and-forget)
    try:
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
    except Exception as _act_exc:  # noqa: BLE001
        logger.warning("Activity log insert failed (non-fatal): %s", _act_exc)

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
        )
    except Exception as _wf_exc:
        try:
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
        except Exception:
            pass
        raise

    payload = result.to_dict()
    next_actions: list[dict[str, Any]] = []
    if result.stability_verdict == "ROBUST":
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

    # Activity log: WALK_FORWARD_RESULT (fire-and-forget)
    try:
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
    except Exception as _act_exc:  # noqa: BLE001
        logger.warning("Activity log insert failed (non-fatal): %s", _act_exc)

    if idempotency_key:
        await store.put(agent, f"walk:{idempotency_key}", payload)
    return payload
