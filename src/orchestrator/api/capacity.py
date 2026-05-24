"""``POST /capacity-sweep`` — scale a strategy across initial-capital
tiers to measure edge degradation.

Built 2026-05-20 under autonomous-mode operator delegation. Lifts
Scorecard #3 (capacity modeling) — see memory
``project_quant_grade_scorecard``.

Long-running: N scales × ~30min per scale. For the default
[100, 1k, 10k, 100k, 1M] this is ~2.5h total. Same idempotency
semantics as ``/walk-forward``: ``Idempotency-Key`` honoured under the
``capacity:`` namespace.

No new DB table for v1 — the per-scale results are the response body
(operator stores via journal or paste). A future ``capacity_sweep_run``
table would let the agent query historical sweeps without re-running.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from ..errors import OrchestratorError
from ..services.idempotency import cache_response, replay_cached_response
from ..services.capacity import run_capacity_sweep
from .deps import get_agent_name, get_db_conn

router = APIRouter(tags=["capacity"])


class CapacitySweepRequest(BaseModel):
    strategy_code: str = Field(..., min_length=1, max_length=60)
    interval_name: str = Field(..., description="5m|15m|1h|4h")
    instrument: str = Field("BTCUSDT", min_length=1, max_length=30)
    start_time: datetime = Field(
        ..., description="Backtest window start (UTC datetime)."
    )
    end_time: datetime = Field(
        ..., description="Backtest window end (UTC datetime)."
    )
    # Default ladder spans 4 orders of magnitude — enough to expose
    # most market-impact decay curves without burning excess JVM time.
    notional_scales: list[float] = Field(
        default_factory=lambda: [100.0, 1_000.0, 10_000.0, 100_000.0, 1_000_000.0],
        min_length=1,
        max_length=10,
        description="Initial-capital tiers in USD quote currency.",
    )
    overrides: dict[str, Any] | None = None
    motivating_iteration_id: UUID | None = None
    slippage_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=0.02,
        description=(
            "Per-fill entry slippage (fractional units, e.g. 0.0005 = 5 bps). "
            "Omit to use the service default. Capped at 2% to catch typos."
        ),
    )


class CapacitySweepResponse(BaseModel):
    capacity_sweep_id: str
    edge_decay_verdict: str
    max_viable_capital_usd: float | None
    motivating_iteration_id: str | None
    per_scale: list[dict[str, Any]]


@router.post("/capacity-sweep", response_model=CapacitySweepResponse)
async def post_capacity_sweep(
    body: CapacitySweepRequest,
    request: Request,
    agent: str = Depends(get_agent_name),
    conn=Depends(get_db_conn),  # noqa: ARG001 — ensures DB reachable + auth
) -> dict[str, Any]:
    cached, idempotency_key = await replay_cached_response(request, agent, "capacity")
    if cached is not None:
        return cached

    if body.end_time <= body.start_time:
        raise OrchestratorError(
            status_code=400,
            error_code="capacity_window_invalid",
            message="end_time must be strictly after start_time.",
            retryable=False,
        )

    kwargs: dict[str, Any] = {}
    if body.slippage_rate is not None:
        kwargs["slippage_rate"] = body.slippage_rate
    result = await run_capacity_sweep(
        db=request.app.state.db,
        jvm=request.app.state.jvm,
        settings=request.app.state.settings,
        agent_name=agent,
        strategy_code=body.strategy_code,
        interval_name=body.interval_name,
        instrument=body.instrument,
        start_time=body.start_time,
        end_time=body.end_time,
        notional_scales=body.notional_scales,
        overrides=body.overrides,
        motivating_iteration_id=body.motivating_iteration_id,
        **kwargs,
    )

    payload = result.to_dict()
    if idempotency_key is not None:
        await cache_response(request, agent, "capacity", idempotency_key, payload)
    return payload
