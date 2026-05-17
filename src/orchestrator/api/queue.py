"""``/queue`` — research queue endpoints (read in Phase 2, write in Phase 3)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID

import asyncpg
from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field, model_validator

from ..errors import NextAction, OrchestratorError
from ..repo import hypothesis_audit as audit_repo
from ..repo import queue as queue_repo
from ..repo import queue_write
from ..repo import reviews as reviews_repo
from ..services.activity_logger import log_activity
from ..services.idempotency import cache_response, replay_cached_response
from .constants import REVIEW_VERDICTS_PASSING_GATE, VALID_INTERVAL_NAMES
from .deps import get_agent_name, get_db_conn
from .pagination import Page, decode_cursor, encode_cursor

router = APIRouter(prefix="/queue", tags=["queue"])

_VALID_STATUS = {"PENDING", "RUNNING", "PARKED", "COMPLETED", "FAILED"}


class SweepParam(BaseModel):
    """One sweep parameter. Shape depends on the parent ``strategy``:

      * ``grid``: provide ``values`` (list of allowed cells).
      * ``tpe``  + ``type=float|int``: provide ``low`` and ``high``.
      * ``tpe``  + ``type=choice``: provide ``values``.

    Validation is enforced at the ``SweepConfig`` level so a single error
    message can name the strategy that's incompatible with the params.
    """

    name: str = Field(..., min_length=1, max_length=80)
    # Grid mode + TPE-choice mode populate this:
    values: list[Any] | None = Field(None, max_length=50)
    # TPE mode populates these:
    type: Literal["float", "int", "choice"] | None = None
    low: str | None = Field(None, max_length=40)
    high: str | None = Field(None, max_length=40)


class SweepConfig(BaseModel):
    """Sweep specification. Backwards-compatible: existing callers that
    omit ``strategy`` get the original grid behaviour and need not pass
    ``n_trials``/``seed``.

    TPE (added 2026-05-05) is sample-efficient relative to a dense grid:
    n=20 TPE trials typically beat a 64-cell grid on a 4-dim search.
    Use it when param ranges are continuous and you don't want to commit
    to a coarse grid up front.
    """

    strategy: Literal["grid", "tpe"] = "grid"
    params: list[SweepParam] = Field(default_factory=list, max_length=10)
    # TPE only — ignored for grid (which derives length from cross-product):
    n_trials: int = Field(20, ge=4, le=200)
    seed: int = Field(42, ge=0)

    @model_validator(mode="after")
    def _check_param_shape(self) -> "SweepConfig":
        if self.strategy == "grid":
            for p in self.params:
                if p.values is None or len(p.values) == 0:
                    raise ValueError(
                        f"grid param {p.name!r} requires non-empty values"
                    )
        elif self.strategy == "tpe":
            for p in self.params:
                if p.type is None:
                    raise ValueError(
                        f"tpe param {p.name!r} requires type "
                        "(one of float|int|choice)"
                    )
                if p.type == "choice":
                    if not p.values:
                        raise ValueError(
                            f"tpe choice param {p.name!r} requires values"
                        )
                else:
                    if p.low is None or p.high is None:
                        raise ValueError(
                            f"tpe numeric param {p.name!r} requires low and high"
                        )
        return self


class EnqueueRequest(BaseModel):
    strategy_code: str = Field(..., min_length=1, max_length=60)
    interval_name: str = Field(..., description="5m|15m|1h|4h")
    instrument: str = Field("BTCUSDT", min_length=1, max_length=30)
    sweep_config: SweepConfig
    hypothesis: str | None = Field(None, max_length=4000)
    priority: int = Field(100, ge=1, le=999)
    iter_budget: int = Field(5, ge=1, le=200)
    early_stop_on_no_edge: bool = True
    require_walk_forward: bool = True
    # Tier 1 re-discovery gate: by default reject sweeps whose axis-set
    # already produced a DISCARD on this strategy. Set ``override_discard_gate``
    # to true ONLY when there's a documented reason (data backfill, bug fix
    # in entry signal, etc.) — gate exists to prevent the autonomous loop
    # from p-hacking by repeatedly re-testing the same dimensions.
    override_discard_gate: bool = False
    # Paired-agent review gate (added 2026-05-05): /queue refuses unless an
    # APPROVED plan review exists for (strategy_code, axis_names,
    # hypothesis_id). hypothesis_id is required when override_review_gate
    # is False — it identifies which pre-registered HYPOTHESIS the reviewer
    # endorsed. override_review_gate=true requires operator-level
    # justification (paste the journal entry id in the request).
    hypothesis_id: str | None = Field(None, max_length=80)
    override_review_gate: bool = False


class CancelRequest(BaseModel):
    reason: str = Field(..., min_length=1, max_length=500)


def resolve_iter_budget(
    sweep_config: SweepConfig, requested: int
) -> tuple[int, str | None]:
    """For TPE sweeps, iter_budget must be >= n_trials or the queue parks
    early (tick.py _decide_next_state parks at new_iter >= iter_budget).
    Auto-bump silently and return a note explaining the change.

    Pure function so it's testable without spinning up the handler.
    """
    if sweep_config.strategy == "tpe" and requested < sweep_config.n_trials:
        return sweep_config.n_trials, (
            f"iter_budget auto-bumped from {requested} to "
            f"{sweep_config.n_trials} to match sweep_config.n_trials "
            f"(TPE budget exhausts at n_trials, not iter_budget)."
        )
    return requested, None


@router.get("", response_model=Page)
async def list_queue(
    status: str | None = Query(None, description="PENDING|RUNNING|PARKED|COMPLETED|FAILED"),
    strategy_code: str | None = Query(None),
    cursor: str | None = Query(None, description="Opaque. From a prior next_cursor."),
    limit: int = Query(25, ge=1, le=200),
    conn: asyncpg.Connection = Depends(get_db_conn),
) -> Page:
    if status is not None and status not in _VALID_STATUS:
        raise OrchestratorError(
            status_code=400,
            error_code="bad_status_filter",
            message=f"status must be one of {sorted(_VALID_STATUS)}.",
            retryable=False,
        )
    after_t, after_id = (None, None)
    if cursor is not None:
        after_t, after_id = decode_cursor(cursor)

    rows = await queue_repo.list_queue(
        conn,
        status=status,
        strategy_code=strategy_code,
        after_created_time=after_t,
        after_id=after_id,
        limit=limit + 1,  # over-fetch by one to detect "has next"
    )
    has_more = len(rows) > limit
    items = rows[:limit]
    next_cursor: str | None = None
    if has_more:
        last = items[-1]
        next_cursor = encode_cursor(last["created_time"], last["queue_id"])

    next_actions: list[dict[str, Any]] = []
    if has_more:
        next_actions.append(
            {"kind": "call", "method": "GET", "path": f"/queue?cursor={next_cursor}"}
        )
    elif not items:
        next_actions.append({"kind": "read_doc", "doc_anchor": "queue.empty"})

    return Page(items=items, next_cursor=next_cursor, next_actions=next_actions)


@router.get("/{queue_id}")
async def get_queue(
    queue_id: UUID,
    conn: asyncpg.Connection = Depends(get_db_conn),
) -> dict[str, Any]:
    row = await queue_repo.get_queue(conn, queue_id)
    if row is None:
        raise OrchestratorError(
            status_code=404,
            error_code="queue_not_found",
            message=f"No research_queue row with queue_id={queue_id}.",
            retryable=False,
            next_action=NextAction(kind="call", method="GET", path="/queue"),
        )
    return row


@router.post("", status_code=201)
async def enqueue(
    body: EnqueueRequest,
    request: Request,
    conn: asyncpg.Connection = Depends(get_db_conn),
    agent: str = Depends(get_agent_name),
) -> dict[str, Any]:
    if body.interval_name not in VALID_INTERVAL_NAMES:
        raise OrchestratorError(
            status_code=400,
            error_code="bad_interval",
            message=f"interval_name must be one of {sorted(VALID_INTERVAL_NAMES)}.",
            retryable=False,
            hint="Backtest engine ticks 5m; finer granularities will silently miss bars.",
        )

    # Idempotency-Key replay: same agent + key returns the original response
    # so a retried enqueue doesn't multiply queue rows.
    cached, idempotency_key = await replay_cached_response(request, agent, "queue")
    if cached is not None:
        return cached

    sweep_dict = body.sweep_config.model_dump()
    effective_iter_budget, iter_budget_note = resolve_iter_budget(
        body.sweep_config, body.iter_budget
    )

    # Paired-agent review gate. Reviewer must have posted APPROVED for
    # the (strategy_code, axis_names, hypothesis_id) tuple before /queue
    # accepts the sweep. CONDITIONAL_APPROVAL also passes (researcher
    # acknowledges the warnings).
    param_names_for_review = [p["name"] for p in sweep_dict.get("params", [])]
    if param_names_for_review and not body.override_review_gate:
        if not body.hypothesis_id:
            raise OrchestratorError(
                status_code=400,
                error_code="hypothesis_id_required",
                message=(
                    "POST /queue requires hypothesis_id (the journal_id of "
                    "the pre-registered HYPOTHESIS) so the review gate can "
                    "find the matching APPROVED verdict. Pass "
                    "override_review_gate=true with documented justification "
                    "to bypass."
                ),
                retryable=False,
                hint=(
                    "Write a HYPOTHESIS journal entry first, request a plan "
                    "review (POST /reviews/request, target_kind='plan'), wait "
                    "for APPROVED, then re-submit /queue with hypothesis_id."
                ),
            )
        target_id = reviews_repo.plan_target_id(
            body.strategy_code, param_names_for_review, body.hypothesis_id
        )
        latest = await reviews_repo.fetch_latest_verdict(conn, target_id)
        sd = (latest or {}).get("structured_data") or {}
        verdict = sd.get("verdict")
        if verdict not in REVIEW_VERDICTS_PASSING_GATE:
            raise OrchestratorError(
                status_code=409,
                error_code="review_required",
                message=(
                    f"No APPROVED plan review on file for target_id={target_id}. "
                    f"Latest verdict: {verdict or 'NONE'}."
                ),
                retryable=False,
                hint=(
                    "Submit POST /reviews/request with target_kind='plan' and "
                    "wait for the reviewer agent to post APPROVED. The "
                    "researcher cannot self-approve."
                ),
                next_action=NextAction(
                    kind="call",
                    method="POST",
                    path="/reviews/request",
                ),
                details={
                    "target_id": target_id,
                    "latest_verdict": verdict,
                    "hypothesis_id": body.hypothesis_id,
                },
            )

    # Tier 1 re-discovery gate. Block sweeps that revisit a dimension
    # the agent has already discarded — without this, the autonomous
    # loop can quietly p-hack by re-running the same axis under a fresh
    # hypothesis line.
    param_names = [p["name"] for p in sweep_dict.get("params", [])]
    if param_names and not body.override_discard_gate:
        a_hash = audit_repo.axis_set_hash(param_names)
        prior = await audit_repo.axis_has_discard(conn, body.strategy_code, a_hash)
        if prior:
            raise OrchestratorError(
                status_code=409,
                error_code="axis_previously_discarded",
                message=(
                    f"strategy={body.strategy_code} has already produced a "
                    f"DISCARD verdict on this axis set "
                    f"({sorted(param_names)}) at "
                    f"{prior['created_time'].isoformat()}. Repeating it is a "
                    f"p-hacking risk."
                ),
                retryable=False,
                hint=(
                    "Pick a different axis combo, OR pass "
                    "override_discard_gate=true with a journal entry "
                    "explaining why the prior discard is no longer load-bearing "
                    "(e.g. data backfill, entry-signal bug fix)."
                ),
                next_action=NextAction(
                    kind="call",
                    method="GET",
                    path=f"/iterations/{prior['iteration_id']}",
                ),
                details={
                    "prior_audit_id": str(prior["audit_id"]),
                    "prior_iteration_id": str(prior["iteration_id"])
                        if prior["iteration_id"] else None,
                    "prior_params": prior["params_snapshot"],
                    "axis_set_hash": a_hash,
                },
            )

    row = await queue_write.insert_queue(
        conn,
        strategy_code=body.strategy_code,
        interval_name=body.interval_name,
        instrument=body.instrument,
        sweep_config=sweep_dict,
        hypothesis=body.hypothesis,
        priority=body.priority,
        iter_budget=effective_iter_budget,
        early_stop_on_no_edge=body.early_stop_on_no_edge,
        require_walk_forward=body.require_walk_forward,
        created_by=agent,
    )

    # Activity log: SWEEP_QUEUED. log_activity is itself fire-and-forget —
    # it swallows internal errors so a busted insert can't fail the request.
    await log_activity(
        conn,
        session_id=request.state.session_id,
        agent_name=agent,
        activity_type="SWEEP_QUEUED",
        title=f"Sweep queued for {body.strategy_code} ({body.interval_name})",
        strategy_code=body.strategy_code,
        details={
            "queue_id": str(row["queue_id"]),
            "strategy_code": body.strategy_code,
            "iter_budget": effective_iter_budget,
            "sweep_strategy": sweep_dict.get("strategy", "grid"),
            "interval_name": body.interval_name,
            "instrument": body.instrument,
            "hypothesis_id": body.hypothesis_id,
        },
        related_id=row["queue_id"] if isinstance(row.get("queue_id"), UUID) else None,
        related_type="queue",
        redis_client=request.app.state.redis,
    )

    response = {
        **row,
        "next_actions": [
            {"kind": "call", "method": "POST", "path": "/tick"},
            {"kind": "call", "method": "GET", "path": f"/queue/{row['queue_id']}"},
        ],
    }
    if iter_budget_note:
        response["notes"] = [iter_budget_note]
    await cache_response(request, agent, "queue", idempotency_key, response)
    return response


@router.post("/{queue_id}/cancel")
async def cancel_queue(
    queue_id: UUID,
    body: CancelRequest,
    conn: asyncpg.Connection = Depends(get_db_conn),
    agent: str = Depends(get_agent_name),
) -> dict[str, Any]:
    note = (
        f"[{datetime.now(timezone.utc).isoformat()}] Cancelled by {agent}: "
        f"{body.reason}"
    )
    affected = await queue_write.cancel_queue(conn, queue_id, note)
    if affected == 0:
        raise OrchestratorError(
            status_code=409,
            error_code="queue_not_cancellable",
            message=(
                f"No PENDING/RUNNING row with queue_id={queue_id}. Already "
                f"COMPLETED/PARKED/FAILED rows cannot be cancelled."
            ),
            retryable=False,
            next_action=NextAction(kind="call", method="GET", path=f"/queue/{queue_id}"),
        )
    row = await queue_repo.get_queue(conn, queue_id)
    return {"cancelled": True, "queue_id": str(queue_id), "row": row}
