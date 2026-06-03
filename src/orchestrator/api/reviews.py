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

from typing import Any, Literal
from uuid import UUID

import asyncpg
from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field

from ..errors import NextAction, OrchestratorError
from ..repo import reviews as reviews_repo
from ..services.activity_logger import log_activity
from ..services.idempotency import cache_response, replay_cached_response
from ..services.review import Severity
from ..services.review_runner import AUTO_REVIEWER_AGENT, auto_run_checklist
from .constants import REVIEW_VERDICTS_PASSING_GATE
from .deps import get_agent_name, get_db_conn

router = APIRouter(prefix="/reviews", tags=["reviews"])

# Allowed target_kind and verdict values live on the Pydantic Literal
# annotations below (ReviewRequestBody.target_kind, ReviewVerdictBody.verdict)
# — those drive both schema validation and the OpenAPI doc, so a duplicate
# frozenset here would just be drift waiting to happen.


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
    severity: Severity
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


@router.post("/request", status_code=201)
async def submit_review_request(
    body: ReviewRequestBody,
    request: Request,
    conn: asyncpg.Connection = Depends(get_db_conn),
    agent: str = Depends(get_agent_name),
) -> dict[str, Any]:
    """Researcher submits a request for a review. Idempotent on
    Idempotency-Key — a retried request returns the original response."""
    cached, idempotency_key = await replay_cached_response(
        request, agent, "review-request"
    )
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

    # Activity log: REVIEW_REQUESTED. log_activity swallows errors internally.
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
        redis_client=request.app.state.redis,
    )

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
    await cache_response(request, agent, "review-request", idempotency_key, response)
    return response


@router.post("", status_code=201)
async def submit_review_verdict(
    body: ReviewVerdictBody,
    request: Request,
    conn: asyncpg.Connection = Depends(get_db_conn),
    agent: str = Depends(get_agent_name),
) -> dict[str, Any]:
    """Reviewer posts a verdict against a target."""
    cached, idempotency_key = await replay_cached_response(
        request, agent, "review-verdict"
    )
    if cached is not None:
        return cached

    # Reviewer-distinctness: an agent may not post a verdict on a review it
    # requested itself. The /queue gate accepts any APPROVED verdict, and the
    # X-Agent-Name identity is self-asserted under a shared token, so without
    # this check the quant-researcher could self-approve its own plan/graduation
    # and clear the gate the paired-review loop exists to enforce. The
    # deterministic /reviews/auto-run-checklist path posts as AUTO_REVIEWER_AGENT
    # and bypasses this handler, so it is unaffected.
    requester = await reviews_repo.fetch_latest_request_requester(conn, body.target_id)
    if requester is not None and requester == agent:
        raise OrchestratorError(
            status_code=409,
            error_code="self_review_forbidden",
            message=(
                "An agent cannot review its own request. The verdict reviewer "
                f"({agent!r}) must differ from the agent that requested the "
                "review. Have a distinct reviewer agent post the verdict."
            ),
            retryable=False,
        )

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

    # Activity log: REVIEW_RECEIVED. log_activity swallows errors internally.
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
        redis_client=request.app.state.redis,
    )

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
    await cache_response(request, agent, "review-verdict", idempotency_key, response)
    return response


class AutoChecklistBody(BaseModel):
    """Input for ``POST /reviews/auto-run-checklist``.

    Researcher posts the same ``target_id`` it would normally hand to a
    spawned reviewer sub-agent. The orchestrator fetches the request
    row, runs the checklist server-side, persists the verdict via the
    normal ``insert_review_verdict`` path (which also closes the open
    request row), and returns the verdict shape callers already branch
    on for ``POST /reviews``.
    """

    target_id: str = Field(..., min_length=1, max_length=200)
    target_kind: Literal["plan", "graduation"]


@router.post("/auto-run-checklist", status_code=201)
async def auto_run_checklist_endpoint(
    body: AutoChecklistBody,
    request: Request,
    conn: asyncpg.Connection = Depends(get_db_conn),
    agent: str = Depends(get_agent_name),
) -> dict[str, Any]:
    """Run the reviewer checklist server-side and persist the verdict.

    This is the HTTP-driven equivalent of spawning a quant-reviewer
    sub-agent — the researcher no longer needs the ``Agent`` tool to
    clear the paired-review gate. The mechanical contract is identical:
    same pure-function checks from ``services/review.py``, same
    ``aggregate_verdict`` rule, same ``REVIEW_VERDICTS_PASSING_GATE``
    enforcement downstream.

    The recorded reviewer identity is ``research-orchestrator-auto-reviewer``
    regardless of the caller's ``X-Agent-Name`` so the audit trail makes
    the source of the verdict obvious; ``X-Agent-Name`` is still stamped
    on the activity-log row.
    """
    cached, idempotency_key = await replay_cached_response(
        request, agent, "review-auto-run"
    )
    if cached is not None:
        return cached

    aggregate = await auto_run_checklist(
        conn,
        target_id=body.target_id,
        target_kind=body.target_kind,
    )

    findings = list(aggregate["checks"])
    summary = {
        "reason": aggregate["reason"],
        "n_blocker_fails": aggregate["n_blocker_fails"],
        "n_warning_fails": aggregate["n_warning_fails"],
        "n_checks": aggregate["n_checks"],
        "auto_run": True,
    }
    journal_id = await reviews_repo.insert_review_verdict(
        conn,
        target_id=body.target_id,
        target_kind=body.target_kind,
        strategy_code=aggregate.get("strategy_code"),
        verdict=aggregate["verdict"],
        findings=findings,
        summary=summary,
        reviewer=AUTO_REVIEWER_AGENT,
        motivating_request_id=aggregate.get("motivating_request_id"),
    )

    await log_activity(
        conn,
        session_id=request.state.session_id,
        agent_name=agent,
        activity_type="REVIEW_RECEIVED",
        title=(
            f"Auto-reviewer verdict: {aggregate['verdict']} for "
            f"{body.target_kind} ({body.target_id[:60]})"
        ),
        strategy_code=aggregate.get("strategy_code"),
        details={
            "target_id": body.target_id,
            "target_kind": body.target_kind,
            "verdict": aggregate["verdict"],
            "journal_id": str(journal_id) if journal_id else None,
            "n_blocker_fails": aggregate["n_blocker_fails"],
            "n_warning_fails": aggregate["n_warning_fails"],
            "auto_run": True,
            "reviewer": AUTO_REVIEWER_AGENT,
        },
        redis_client=request.app.state.redis,
    )

    # Use the canonical gate constant — keeps next_actions in lockstep with
    # the actual /queue and /walk-forward gate logic. Re-deriving "APPROVED
    # OR CONDITIONAL_APPROVAL" here would drift the moment the constant changes.
    passes_gate = aggregate["verdict"] in REVIEW_VERDICTS_PASSING_GATE
    next_actions: list[dict[str, Any]] = []
    if body.target_kind == "plan":
        if passes_gate:
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
    else:
        if passes_gate:
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
        "verdict": aggregate["verdict"],
        "summary_reason": aggregate["reason"],
        "summary_n_blocker_fails": aggregate["n_blocker_fails"],
        "summary_n_warning_fails": aggregate["n_warning_fails"],
        "findings": findings,
        "reviewer": AUTO_REVIEWER_AGENT,
        "motivating_request_id": aggregate.get("motivating_request_id"),
        "next_actions": next_actions,
    }
    await cache_response(request, agent, "review-auto-run", idempotency_key, response)
    return response


@router.get("/pending")
async def list_pending_reviews(
    limit: int = Query(50, ge=1, le=200),
    conn: asyncpg.Connection = Depends(get_db_conn),
) -> dict[str, Any]:
    """Reviewer's work queue. Oldest-first."""
    rows = await reviews_repo.fetch_pending_requests(conn, limit=limit)
    return {"items": rows, "count": len(rows)}


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
