"""``/queue`` — research queue endpoints (read in Phase 2, write in Phase 3)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import asyncpg
from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field

from ..errors import NextAction, OrchestratorError
from ..repo import hypothesis_audit as audit_repo
from ..repo import queue as queue_repo
from ..repo import queue_write
from .deps import get_agent_name, get_db_conn
from .pagination import Page, decode_cursor, encode_cursor

router = APIRouter(prefix="/queue", tags=["queue"])

_VALID_STATUS = {"PENDING", "RUNNING", "PARKED", "COMPLETED", "FAILED"}
_VALID_INTERVALS = {"5m", "15m", "1h", "4h"}


class SweepParam(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    values: list[Any] = Field(..., min_length=1, max_length=50)


class SweepConfig(BaseModel):
    """Same shape consumed by ``derive_combo`` in services/sweep.py.

    ``strategy`` is reserved for future non-grid search (e.g. Bayesian);
    Phase 3 only honours ``grid``. ``iter_budget`` lives on the parent
    EnqueueRequest and on the queue row itself — duplicating it here
    would just create a clobber path.
    """

    strategy: str = Field("grid", pattern=r"^grid$")
    params: list[SweepParam] = Field(default_factory=list, max_length=10)


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


class CancelRequest(BaseModel):
    reason: str = Field(..., min_length=1, max_length=500)


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
    if body.interval_name not in _VALID_INTERVALS:
        raise OrchestratorError(
            status_code=400,
            error_code="bad_interval",
            message=f"interval_name must be one of {sorted(_VALID_INTERVALS)}.",
            retryable=False,
            hint="Backtest engine ticks 5m; finer granularities will silently miss bars.",
        )

    # Idempotency-Key replay: same agent + key returns the original response
    # so a retried enqueue doesn't multiply queue rows.
    idempotency_key = getattr(request.state, "idempotency_key", None)
    store = request.app.state.idempotency
    if idempotency_key:
        cached = await store.get(agent, f"queue:{idempotency_key}")
        if cached is not None:
            return cached

    sweep_dict = body.sweep_config.model_dump()

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
        iter_budget=body.iter_budget,
        early_stop_on_no_edge=body.early_stop_on_no_edge,
        require_walk_forward=body.require_walk_forward,
        created_by=agent,
    )
    response = {
        **row,
        "next_actions": [
            {"kind": "call", "method": "POST", "path": "/tick"},
            {"kind": "call", "method": "GET", "path": f"/queue/{row['queue_id']}"},
        ],
    }
    if idempotency_key:
        await store.put(agent, f"queue:{idempotency_key}", response)
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
