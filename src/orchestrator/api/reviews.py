"""``/reviews`` — paired-agent review protocol.

Researcher posts ``POST /reviews/request`` to enqueue a review.
Reviewer pulls from ``GET /reviews/pending``, runs the checklist
(``services/review.py``), and posts the verdict via ``POST /reviews``.
The orchestrator gates ``/queue`` and ``/walk-forward`` on the latest
APPROVED verdict for the relevant target_id.

target_id encoding lives in ``repo/reviews.py``:
  * ``plan:{strategy_code}:{axis_set_hash}:{hypothesis_id}``
  * ``graduation:{iteration_id}``
"""

from __future__ import annotations

import logging
from typing import Any, Literal
from uuid import UUID

import asyncpg
from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field

from ..errors import NextAction, OrchestratorError
from ..repo import reviews as reviews_repo
from ..services.activity_logger import log_activity
from .deps import get_agent_name, get_db_conn

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reviews", tags=["reviews"])

_VALID_TARGET_KIND = {"plan", "graduation"}
_VALID_VERDICT = {"APPROVED", "CONDITIONAL_APPROVAL", "REJECTED"}


# ── Request models ───────────────────────────────────────────────────


class PlanRequestPayload(BaseModel):
    strategy_code: str = Field(..., min_length=1, max_length=60)
    axis_names: list[str] = Field(..., min_length=1, max_length=10)
    hypothesis_id: str = Field(..., min_length=1, max_length=80)
    plan_path: str | None = Field(None, max_length=500)
    notes: str | None = Field(None, max_length=4000)


class GraduationRequestPayload(BaseModel):
    iteration_id: UUID
    strategy_code: str = Field(..., min_length=1, max_length=60)
    motivating_hypothesis_id: str | None = Field(None, max_length=80)
    notes: str | None = Field(None, max_length=4000)


class ReviewRequestBody(BaseModel):
    target_kind: Literal["plan", "graduation"]
    plan: PlanRequestPayload | None = None
    graduation: GraduationRequestPayload | None = None


class ReviewFindingItem(BaseModel):
    check_name: str
    severity: Literal["blocker", "warning", "info"]
    passed: bool
    finding: str
    details: dict[str, Any] = Field(default_factory=dict)


class ReviewVerdictBody(BaseModel):
    target_id: str = Field(..., min_length=1, max_length=200)
    target_kind: Literal["plan", "graduation"]
    verdict: Literal["APPROVED", "CONDITIONAL_APPROVAL", "REJECTED"]
    findings: list[ReviewFindingItem] = Field(default_factory=list)
    summary_reason: str = Field(..., min_length=1, max_length=2000)
    summary_n_blocker_fails: int = 0
    summary_n_warning_fails: int = 0
    motivating_request_id: str | None = Field(None, max_length=80)
    strategy_code: str | None = Field(None, max_length=60)


# ── POST /reviews/request ────────────────────────────────────────────


@router.post("/request", status_code=201)
async def submit_review_request(
    body: ReviewRequestBody,
    request: Request,
    conn: asyncpg.Connection = Depends(get_db_conn),
    agent: str = Depends(get_agent_name),
) -> dict[str, Any]:
    """Researcher submits a request for a review. Idempotent on
    Idempotency-Key — a retried request returns the original response."""
    idempotency_key = getattr(request.state, "idempotency_key", None)
    store = request.app.state.idempotency
    if idempotency_key:
        cached = await store.get(agent, f"review-request:{idempotency_key}")
        if cached is not None:
            return cached

    if body.target_kind == "plan":
        if body.plan is None:
            raise OrchestratorError(
                status_code=400,
                error_code="missing_plan_payload",
                message="target_kind='plan' requires a populated 'plan' object.",
                retryable=False,
            )
        target_id = reviews_repo.plan_target_id(
            body.plan.strategy_code,
            list(body.plan.axis_names),
            body.plan.hypothesis_id,
        )
        strategy_code = body.plan.strategy_code
        payload: dict[str, Any] = body.plan.model_dump()
    else:
        if body.graduation is None:
            raise OrchestratorError(
                status_code=400,
                error_code="missing_graduation_payload",
                message="target_kind='graduation' requires a populated 'graduation' object.",
                retryable=False,
            )
        target_id = reviews_repo.graduation_target_id(body.graduation.iteration_id)
        strategy_code = body.graduation.strategy_code
        payload = body.graduation.model_dump(mode="json")

    journal_id = await reviews_repo.insert_review_request(
        conn,
        target_id=target_id,
        target_kind=body.target_kind,
        strategy_code=strategy_code,
        payload=payload,
        requested_by=agent,
    )

    # Activity log: REVIEW_REQUESTED (fire-and-forget)
    try:
        redis_client = request.app.state.redis
        await log_activity(
            conn,
            session_id=request.state.session_id,
            agent_name=agent,
            activity_type="REVIEW_REQUESTED",
            title=f"Review requested: {body.target_kind} for {strategy_code}",
            strategy_code=strategy_code,
            details={
                "target_id": target_id,
                "target_kind": body.target_kind,
                "journal_id": str(journal_id) if journal_id else None,
            },
            redis_client=redis_client,
        )
    except Exception as _act_exc:  # noqa: BLE001
        logger.warning("Activity log insert failed (non-fatal): %s", _act_exc)

    response = {
        "journal_id": journal_id,
        "target_id": target_id,
        "target_kind": body.target_kind,
        "status": "PENDING",
        "next_actions": [
            {
                "kind": "call",
                "method": "GET",
                "path": "/reviews/pending",
                "hint": (
                    "Reviewer agent should pull this list, run the "
                    "checklist, and POST /reviews."
                ),
            },
            {
                "kind": "call",
                "method": "GET",
                "path": f"/reviews/by-target?target_id={target_id}",
                "hint": "Researcher polls here to read the verdict.",
            },
        ],
    }
    if idempotency_key:
        await store.put(agent, f"review-request:{idempotency_key}", response)
    return response


# ── POST /reviews ────────────────────────────────────────────────────


@router.post("", status_code=201)
async def submit_review_verdict(
    body: ReviewVerdictBody,
    request: Request,
    conn: asyncpg.Connection = Depends(get_db_conn),
    agent: str = Depends(get_agent_name),
) -> dict[str, Any]:
    """Reviewer posts a verdict against a target."""
    idempotency_key = getattr(request.state, "idempotency_key", None)
    store = request.app.state.idempotency
    if idempotency_key:
        cached = await store.get(agent, f"review-verdict:{idempotency_key}")
        if cached is not None:
            return cached

    findings = [f.model_dump() for f in body.findings]
    summary = {
        "reason": body.summary_reason,
        "n_blocker_fails": body.summary_n_blocker_fails,
        "n_warning_fails": body.summary_n_warning_fails,
    }
    journal_id = await reviews_repo.insert_review_verdict(
        conn,
        target_id=body.target_id,
        target_kind=body.target_kind,
        strategy_code=body.strategy_code,
        verdict=body.verdict,
        findings=findings,
        summary=summary,
        reviewer=agent,
        motivating_request_id=body.motivating_request_id,
    )

    # Activity log: REVIEW_RECEIVED (fire-and-forget)
    try:
        redis_client = request.app.state.redis
        await log_activity(
            conn,
            session_id=request.state.session_id,
            agent_name=agent,
            activity_type="REVIEW_RECEIVED",
            title=f"Review verdict: {body.verdict} for {body.target_kind} ({body.target_id[:60]})",
            strategy_code=body.strategy_code,
            details={
                "target_id": body.target_id,
                "target_kind": body.target_kind,
                "verdict": body.verdict,
                "journal_id": str(journal_id) if journal_id else None,
                "n_blocker_fails": body.summary_n_blocker_fails,
                "n_warning_fails": body.summary_n_warning_fails,
            },
            redis_client=redis_client,
        )
    except Exception as _act_exc:  # noqa: BLE001
        logger.warning("Activity log insert failed (non-fatal): %s", _act_exc)

    next_actions: list[dict[str, Any]] = []
    if body.target_kind == "plan":
        if body.verdict == "APPROVED":
            next_actions.append(
                {
                    "kind": "call",
                    "method": "POST",
                    "path": "/queue",
                    "hint": "Plan APPROVED — researcher may enqueue.",
                }
            )
        else:
            next_actions.append(
                {
                    "kind": "read_doc",
                    "doc_anchor": "review.findings_to_address",
                    "hint": "Researcher must address findings then re-request.",
                }
            )
    else:  # graduation
        if body.verdict == "APPROVED":
            next_actions.append(
                {
                    "kind": "call",
                    "method": "POST",
                    "path": "/walk-forward",
                    "hint": "Graduation APPROVED — researcher may /walk-forward.",
                }
            )
        else:
            next_actions.append(
                {
                    "kind": "read_doc",
                    "doc_anchor": "review.findings_to_address",
                    "hint": "Graduation blocked — researcher must journal and pivot.",
                }
            )

    response = {
        "journal_id": journal_id,
        "target_id": body.target_id,
        "target_kind": body.target_kind,
        "verdict": body.verdict,
        "next_actions": next_actions,
    }
    if idempotency_key:
        await store.put(agent, f"review-verdict:{idempotency_key}", response)
    return response


# ── GET /reviews/pending ─────────────────────────────────────────────


@router.get("/pending")
async def list_pending_reviews(
    limit: int = Query(50, ge=1, le=200),
    conn: asyncpg.Connection = Depends(get_db_conn),
) -> dict[str, Any]:
    """Reviewer's work queue. Oldest-first."""
    rows = await reviews_repo.fetch_pending_requests(conn, limit=limit)
    return {"items": rows, "count": len(rows)}


# ── GET /reviews/by-target ───────────────────────────────────────────


@router.get("/by-target")
async def get_review_by_target(
    target_id: str = Query(..., min_length=1, max_length=200),
    history: bool = Query(False, description="Return the full request+verdict trail."),
    conn: asyncpg.Connection = Depends(get_db_conn),
) -> dict[str, Any]:
    """Researcher polls this to read the latest verdict for a target.

    With ``history=true``, returns all request/verdict rows so the
    full audit trail is visible (useful for journaling re-submissions).
    """
    latest = await reviews_repo.fetch_latest_verdict(conn, target_id)
    out: dict[str, Any] = {"target_id": target_id, "latest_verdict": latest}
    if history:
        out["history"] = await reviews_repo.fetch_history_for_target(conn, target_id)
    if latest is None:
        out["next_actions"] = [
            {
                "kind": "retry",
                "wait_s": 30.0,
                "hint": (
                    "No verdict posted yet. Reviewer agent has not run, "
                    "or is still working."
                ),
            }
        ]
    return out
