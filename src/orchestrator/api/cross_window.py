"""``POST /cross-window`` — multi-window regime-stratified validation.

Phase 5. Stricter sibling of ``/walk-forward``: where walk-forward asks
"does the edge hold across rolling chronological folds?", cross-window
asks "does the edge hold across regime-labeled epochs (bull/bear/chop)?"
and demands ≥80% of windows be net-positive after a +20bps slippage
haircut. ROBUST_CROSS_WINDOW is the harder bar of the two.

Long-running on purpose: 6 windows × up to 30min/poll = up to 3 hours.
The agent calls this deliberately — typically after a ``/walk-forward``
returns ROBUST and before recommending the strategy to the operator
for promotion.

Idempotency: ``Idempotency-Key`` honoured under the ``cross:`` namespace.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from ..services.cross_window import ROBUST_PCT_THRESHOLD, run_cross_window
from .deps import get_agent_name

router = APIRouter(tags=["cross-window"])


class CrossWindowRequest(BaseModel):
    strategy_code: str = Field(..., min_length=1, max_length=60)
    interval_name: str = Field(..., description="5m|15m|1h|4h")
    instrument: str = Field("BTCUSDT", min_length=1, max_length=30)
    overrides: dict[str, Any] | None = None
    motivating_iteration_id: UUID | None = None
    motivating_walk_forward_id: UUID | None = None


class CrossWindowResponse(BaseModel):
    cross_window_id: str
    cross_window_verdict: str
    windows_catalog_version: str
    aggregate: dict[str, Any]
    window_results: list[dict[str, Any]]
    motivating_walk_forward_id: str | None = None
    motivating_iteration_id: str | None = None
    next_actions: list[dict[str, Any]]


@router.post("/cross-window", response_model=CrossWindowResponse)
async def post_cross_window(
    body: CrossWindowRequest,
    request: Request,
    agent: str = Depends(get_agent_name),
) -> dict[str, Any]:
    idempotency_key = getattr(request.state, "idempotency_key", None)
    store = request.app.state.idempotency
    if idempotency_key:
        cached = await store.get(agent, f"cross:{idempotency_key}")
        if cached is not None:
            return cached

    result = await run_cross_window(
        db=request.app.state.db,
        jvm=request.app.state.jvm,
        settings=request.app.state.settings,
        agent_name=agent,
        strategy_code=body.strategy_code,
        interval_name=body.interval_name,
        instrument=body.instrument,
        overrides=body.overrides,
        motivating_iteration_id=body.motivating_iteration_id,
        motivating_walk_forward_id=body.motivating_walk_forward_id,
    )

    payload = result.to_dict()
    next_actions: list[dict[str, Any]] = []
    if result.cross_window_verdict == "ROBUST_CROSS_WINDOW":
        next_actions.append({
            "kind": "note",
            "hint": (
                "ROBUST_CROSS_WINDOW — strategy holds across regimes. "
                "Journal as graduation candidate; operator promotes via "
                "/api/v1/strategy-promotion/{id}/promote."
            ),
        })
    elif result.cross_window_verdict == "INCONSISTENT_CROSS_WINDOW":
        pct = result.aggregate.get("pct_windows_net_positive")
        next_actions.append({
            "kind": "read_doc",
            "doc_anchor": "cross_window.inconsistent",
            "hint": (
                f"INCONSISTENT_CROSS_WINDOW — {pct}% windows net-positive "
                f"(bar = {ROBUST_PCT_THRESHOLD}%). Identify the failing "
                "regime tag and decide: regime-gate the strategy, or abandon."
            ),
        })
    elif result.cross_window_verdict == "NO_EDGE_CROSS_WINDOW":
        next_actions.append({
            "kind": "note",
            "hint": "NO_EDGE_CROSS_WINDOW — fails majority of regimes. Abandon.",
        })
    else:  # INSUFFICIENT_DATA_IN_WINDOW
        next_actions.append({
            "kind": "note",
            "hint": (
                "INSUFFICIENT_DATA_IN_WINDOW — too few trades per regime "
                "to judge. Loosen entry filter or drop the strategy."
            ),
        })
    payload["next_actions"] = next_actions

    if idempotency_key:
        await store.put(agent, f"cross:{idempotency_key}", payload)
    return payload
