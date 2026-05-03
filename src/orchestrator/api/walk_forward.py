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

from datetime import date
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from ..services.walk_forward import run_walk_forward
from .deps import get_agent_name

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
    idempotency_key = getattr(request.state, "idempotency_key", None)
    store = request.app.state.idempotency
    if idempotency_key:
        cached = await store.get(agent, f"walk:{idempotency_key}")
        if cached is not None:
            return cached

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

    if idempotency_key:
        await store.put(agent, f"walk:{idempotency_key}", payload)
    return payload
