"""``/specialist-review/*`` — async checkpoint for the adversarial-
review gap (Path C, 2026-05-20).

The researcher cannot spawn sub-agents from inside its own sub-agent
context (smoke-tested 2026-05-20 — harness strips ``Agent`` from every
nested context regardless of the agent's ``tools:`` list). So we move
the qualitative skeptic / portfolio / ml-judge passes out of the
researcher session entirely:

  1. Researcher reaches step 9d (post-graduation-review APPROVED).
  2. Researcher POSTs one ``/specialist-review/request`` per applicable
     specialist (skeptic, portfolio, optionally ml-judge). All three at
     once — Option A1, confirmed by operator 2026-05-20.
  3. Researcher exits on the new terminal ``SPECIALIST_REVIEW_PENDING``.
  4. Operator's main session runs ``/run-pending-specialists`` slash
     command. The slash command:
       * GETs ``/specialist-review/pending`` for the work queue.
       * For each row, spawns ``Agent(subagent_type=specialist_name,
         prompt=request_payload + "end with VERDICT: <literal>")``.
       * Parses the VERDICT literal from the sub-agent's return.
       * POSTs ``/specialist-review/complete`` — verdict landed, request
         row PARKED, ``agent_decisions`` audit row written.
  5. Researcher's next session step 1a sees the verdicts via
     ``/agent/state.pending_specialist_reviews`` /
     ``completed_specialist_reviews_for_active_hypothesis`` and either:
       * any verdict is a veto → journal STRATEGY_OUTCOME, back to step 1
       * all verdicts CONCUR / CONCERN / ADD → continue at step 9 (walk-
         forward)
       * still some pending → exit again on SPECIALIST_REVIEW_PENDING

Same idempotency + activity-log + audit-row hygiene the existing
``/reviews`` endpoints use.
"""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

import asyncpg
from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field

from ..errors import OrchestratorError
from ..repo import agent_decisions as decisions_repo
from ..repo import specialist_reviews as specialist_reviews_repo
from ..services.activity_logger import log_activity
from ..services.idempotency import cache_response, replay_cached_response
from .deps import get_agent_name, get_db_conn

router = APIRouter(prefix="/specialist-review", tags=["specialist-review"])


SpecialistName = Literal[
    "quant-skeptic",
    "quant-portfolio-manager",
    "quant-ml-judge",
    "quant-curator",
]


# ── POST /specialist-review/request (researcher) ────────────────────


class SpecialistReviewRequestBody(BaseModel):
    specialist_name: SpecialistName
    iteration_id: UUID
    strategy_code: str = Field(..., min_length=1, max_length=60)
    motivating_hypothesis_id: UUID | None = None
    # Free-form payload — must carry everything the spawned sub-agent
    # needs to reach a verdict without re-querying the orchestrator
    # itself (the slash command forwards this verbatim into the spawn
    # prompt). Typical contents:
    #   - skeptic: {iteration_metrics, prescreen_response, plan_excerpt,
    #     hypothesis_summary}
    #   - portfolio: {iteration_metrics, correlations_response,
    #     optimize_response, candidate_proposed_weight}
    #   - ml-judge: {model_id, experiment_run_id, ml_prescreen_response,
    #     paired_delta_response}
    request_payload: dict[str, Any] = Field(default_factory=dict)


@router.post("/request", status_code=201)
async def submit_specialist_review_request(
    body: SpecialistReviewRequestBody,
    request: Request,
    conn: asyncpg.Connection = Depends(get_db_conn),
    agent: str = Depends(get_agent_name),
) -> dict[str, Any]:
    """Researcher writes a pending specialist-review row. Idempotent on
    Idempotency-Key — a retried request returns the original response."""
    cached, idempotency_key = await replay_cached_response(
        request, agent, "specialist-review-request"
    )
    if cached is not None:
        return cached

    journal_id, target_id = (
        await specialist_reviews_repo.insert_specialist_review_request(
            conn,
            specialist_name=body.specialist_name,
            iteration_id=body.iteration_id,
            strategy_code=body.strategy_code,
            motivating_hypothesis_id=(
                str(body.motivating_hypothesis_id)
                if body.motivating_hypothesis_id
                else None
            ),
            payload=body.request_payload,
            requested_by=agent,
        )
    )

    await log_activity(
        conn,
        session_id=request.state.session_id,
        agent_name=agent,
        activity_type="SPECIALIST_REVIEW_REQUESTED",
        title=(
            f"Specialist review requested: {body.specialist_name} "
            f"on {body.strategy_code} iteration {body.iteration_id}"
        ),
        strategy_code=body.strategy_code,
        details={
            "specialist_name": body.specialist_name,
            "iteration_id": str(body.iteration_id),
            "target_id": target_id,
            "journal_id": journal_id,
            "motivating_hypothesis_id": (
                str(body.motivating_hypothesis_id)
                if body.motivating_hypothesis_id
                else None
            ),
        },
        redis_client=request.app.state.redis,
    )

    response = {
        "journal_id": journal_id,
        "target_id": target_id,
        "specialist_name": body.specialist_name,
        "iteration_id": str(body.iteration_id),
        "status": "PENDING",
        "next_actions": [
            {
                "kind": "operator_run_command",
                "command": "/run-pending-specialists",
                "hint": (
                    "Operator runs this slash command from main session "
                    "to spawn the specialist sub-agent and post the verdict. "
                    "Researcher should exit on SPECIALIST_REVIEW_PENDING "
                    "terminal once all requests are written."
                ),
            },
        ],
    }
    await cache_response(
        request, agent, "specialist-review-request", idempotency_key, response
    )
    return response


# ── GET /specialist-review/pending (slash command) ──────────────────


@router.get("/pending")
async def list_pending_specialist_reviews(
    limit: int = Query(50, ge=1, le=200),
    conn: asyncpg.Connection = Depends(get_db_conn),
) -> dict[str, Any]:
    """Slash command's work queue. Oldest-first.

    Each row includes ``structured_data.request_payload`` — the slash
    command forwards this directly into the ``Agent`` spawn prompt.
    """
    rows = await specialist_reviews_repo.fetch_pending_specialist_requests(
        conn, limit=limit
    )
    return {"items": rows, "count": len(rows)}


# ── POST /specialist-review/complete (slash command) ────────────────


class SpecialistReviewCompleteBody(BaseModel):
    """The slash command's verdict-post.

    ``raw_response`` is the sub-agent's full return text (or a
    structured envelope if the slash command pre-parses). Stored for
    forensic audit. ``reasoning`` is a human-readable short summary
    (the slash command extracts this from the sub-agent's text).
    ``verdict`` is the strict literal that drove the parser; the repo
    layer validates it against ``VERDICT_LITERALS`` for the named
    specialist and 400s on mismatch.
    """

    target_id: str = Field(..., min_length=1, max_length=200)
    specialist_name: SpecialistName
    iteration_id: UUID
    strategy_code: str | None = Field(None, max_length=60)
    verdict: str = Field(..., min_length=1, max_length=40)
    reasoning: str = Field("", max_length=8000)
    raw_response: dict[str, Any] = Field(default_factory=dict)
    motivating_request_id: str | None = Field(None, max_length=80)
    latency_ms: int | None = Field(None, ge=0)


@router.post("/complete", status_code=201)
async def submit_specialist_review_complete(
    body: SpecialistReviewCompleteBody,
    request: Request,
    conn: asyncpg.Connection = Depends(get_db_conn),
    agent: str = Depends(get_agent_name),
) -> dict[str, Any]:
    """Slash command posts the spawned specialist's verdict.

    Atomic across {verdict journal row + park matching request row +
    agent_decisions audit row}: a single outer transaction wraps the
    repo's verdict-insert (which itself opens an inner transaction —
    asyncpg promotes nested ``async with conn.transaction()`` calls to
    savepoints). If the audit-row insert fails, the verdict + park are
    rolled back together — on retry with the same Idempotency-Key the
    slash command lands a clean single verdict instead of a duplicate.

    Activity log stays outside the transaction (fire-and-forget; it
    catches its own exceptions).

    Verdict literal validity is enforced at the repo layer — a bad
    literal raises ValueError which the orchestrator's exception handler
    maps to a 400.
    """
    cached, idempotency_key = await replay_cached_response(
        request, agent, "specialist-review-complete"
    )
    if cached is not None:
        return cached

    try:
        async with conn.transaction():
            verdict_journal_id = (
                await specialist_reviews_repo.insert_specialist_review_verdict(
                    conn,
                    target_id=body.target_id,
                    specialist_name=body.specialist_name,
                    iteration_id=body.iteration_id,
                    strategy_code=body.strategy_code,
                    verdict=body.verdict,
                    reasoning=body.reasoning,
                    raw_response=body.raw_response,
                    spawned_by=agent,
                    motivating_request_id=body.motivating_request_id,
                )
            )

            # Audit row written in the same transaction so failure rolls
            # back the verdict + park atomically. endpoint marker is
            # distinct from "subagent_spawned" (Path-A primary that
            # proved infeasible) and from "in_context_template" (old
            # Option D pattern).
            decision_id = await decisions_repo.insert_agent_decision(
                conn,
                specialist=body.specialist_name,
                endpoint="path_c_async_spawn",
                model_name="sonnet",  # all three specialists run on sonnet
                request_payload={
                    "target_id": body.target_id,
                    "iteration_id": str(body.iteration_id),
                    "motivating_request_id": body.motivating_request_id,
                },
                response_payload={
                    "verdict": body.verdict,
                    "reasoning": body.reasoning,
                    "raw_response": body.raw_response,
                    "verdict_journal_id": verdict_journal_id,
                },
                verdict=body.verdict,
                latency_ms=body.latency_ms,
                input_tokens=None,
                output_tokens=None,
                cache_read_input_tokens=None,
                cache_creation_input_tokens=None,
                status="ok",
                error_envelope=None,
                motivating_iteration_id=body.iteration_id,
                target_id=body.target_id,
                created_by=agent,
            )
    except ValueError as exc:
        # Bad verdict literal — surface as 400 so the slash command's
        # retry loop doesn't poison the journal with garbage. Raised
        # from inside the transaction; the rollback already happened
        # on context-exit.
        raise OrchestratorError(
            status_code=400,
            error_code="invalid_specialist_verdict",
            message=str(exc),
            retryable=False,
        ) from exc

    await log_activity(
        conn,
        session_id=request.state.session_id,
        agent_name=agent,
        activity_type="SPECIALIST_REVIEW_RECEIVED",
        title=(
            f"Specialist verdict: {body.specialist_name} {body.verdict} "
            f"— iter {body.iteration_id}"
        ),
        strategy_code=body.strategy_code,
        details={
            "specialist_name": body.specialist_name,
            "iteration_id": str(body.iteration_id),
            "target_id": body.target_id,
            "verdict": body.verdict,
            "verdict_journal_id": verdict_journal_id,
            "decision_id": str(decision_id),
        },
        redis_client=request.app.state.redis,
    )

    response = {
        "verdict_journal_id": verdict_journal_id,
        "decision_id": str(decision_id),
        "target_id": body.target_id,
        "specialist_name": body.specialist_name,
        "iteration_id": str(body.iteration_id),
        "verdict": body.verdict,
        "is_veto": body.verdict in (
            specialist_reviews_repo.VETO_VERDICTS.get(
                body.specialist_name, frozenset()
            )
        ),
    }
    await cache_response(
        request, agent, "specialist-review-complete", idempotency_key, response
    )
    return response


# ── GET /specialist-review/by-iteration (researcher resume) ─────────


@router.get("/by-iteration")
async def get_specialist_reviews_for_iteration(
    iteration_id: UUID = Query(...),
    conn: asyncpg.Connection = Depends(get_db_conn),
) -> dict[str, Any]:
    """All pending requests + verdicts for one iteration. Researcher's
    resume protocol (step 1a) reads this to decide whether to continue
    at step 9 (all verdicts in, no vetos) or exit again on
    SPECIALIST_REVIEW_PENDING (some still pending)."""
    pending = await specialist_reviews_repo.fetch_pending_for_iteration(
        conn, iteration_id
    )
    verdicts = await specialist_reviews_repo.fetch_verdicts_for_iteration(
        conn, iteration_id
    )

    # Latest-verdict-per-specialist view for fast resume branching.
    latest_by_specialist: dict[str, dict[str, Any]] = {}
    for v in verdicts:
        sd = v.get("structured_data") or {}
        name = sd.get("specialist_name")
        if name and name not in latest_by_specialist:
            # verdicts are ordered DESC by created_time, so first hit wins
            latest_by_specialist[name] = {
                "verdict": sd.get("verdict"),
                "is_veto": sd.get("is_veto", False),
                "reasoning": sd.get("reasoning"),
                "journal_id": str(v["journal_id"]),
                "created_time": v["created_time"],
            }

    any_veto = any(
        entry.get("is_veto") for entry in latest_by_specialist.values()
    )
    pending_specialists = sorted(
        {
            (p.get("structured_data") or {}).get("specialist_name")
            for p in pending
            if (p.get("structured_data") or {}).get("specialist_name")
        }
    )

    return {
        "iteration_id": str(iteration_id),
        "pending": pending,
        "verdicts": verdicts,
        "latest_by_specialist": latest_by_specialist,
        "any_veto": any_veto,
        "pending_specialists": pending_specialists,
        "n_pending": len(pending),
        "n_verdicts": len(verdicts),
    }
