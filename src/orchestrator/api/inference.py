"""``/inference/*`` — proxy to the blackheart-inference sidecar.

Researcher-facing surface. The actual ML inference (artifact load,
feature fetch, predict, signal_history upsert) lives in
``blackheart-inference`` on ``:8000``. The orchestrator forwards the
researcher's call, stamps activity, honours Idempotency-Key.

Why proxy rather than direct? Two reasons:
  1. **Single auth surface.** The researcher already authenticates to
     ``:8082``; routing through here means no second token to manage.
  2. **Audit trail.** ``INFERENCE_RUN`` / ``INFERENCE_BACKFILL`` activity
     rows land in the same activity stream the researcher's other
     actions already write — single timeline, one filter query.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from ..services.activity_logger import log_activity
from ..services.idempotency import cache_response, replay_cached_response
from .deps import get_agent_name

router = APIRouter(prefix="/inference", tags=["inference"])


class RunBody(BaseModel):
    signal_id: UUID
    symbol: str = Field(..., min_length=1, max_length=20)
    ts: datetime
    interval_name: str = Field(..., description="5m|15m|1h|4h")
    source: Literal["stream", "catchup_scan", "historical_replay"] = "stream"


class BackfillBody(BaseModel):
    signal_id: UUID
    symbol: str = Field(..., min_length=1, max_length=20)
    interval_name: str = Field(..., description="5m|15m|1h|4h")
    start: datetime
    end: datetime
    source: Literal["catchup_scan", "historical_replay"] = "historical_replay"


@router.post("/run")
async def post_run(
    body: RunBody,
    request: Request,
    agent: str = Depends(get_agent_name),
) -> dict[str, Any]:
    """Single-bar inference. Honors Idempotency-Key for safe replay."""
    cached, idempotency_key = await replay_cached_response(
        request, agent, "inference-run"
    )
    if cached is not None:
        return cached

    payload = body.model_dump(mode="json")
    response = await request.app.state.inference.run(body=payload, agent_name=agent)

    try:
        async with request.app.state.db.acquire() as conn:
            await log_activity(
                conn,
                session_id=request.state.session_id,
                agent_name=agent,
                activity_type="INFERENCE_RUN",
                title=(
                    f"Inference run: signal={body.signal_id} symbol={body.symbol} "
                    f"ts={body.ts.isoformat()}"
                ),
                strategy_code=None,
                details={
                    "signal_id": str(body.signal_id),
                    "symbol": body.symbol,
                    "ts": body.ts.isoformat(),
                    "interval_name": body.interval_name,
                    "source": body.source,
                    "model_id": response.get("model_id"),
                    "value": response.get("value"),
                },
                redis_client=request.app.state.redis,
            )
    except Exception:  # noqa: BLE001
        pass

    await cache_response(request, agent, "inference-run", idempotency_key, response)
    return response


@router.post("/backfill")
async def post_backfill(
    body: BackfillBody,
    request: Request,
    agent: str = Depends(get_agent_name),
) -> dict[str, Any]:
    """Window inference. Use for paired-backtest preparation — populates
    signal_history across the historical range a backtest will replay."""
    cached, idempotency_key = await replay_cached_response(
        request, agent, "inference-backfill"
    )
    if cached is not None:
        return cached

    payload = body.model_dump(mode="json")
    response = await request.app.state.inference.backfill(body=payload, agent_name=agent)

    try:
        async with request.app.state.db.acquire() as conn:
            await log_activity(
                conn,
                session_id=request.state.session_id,
                agent_name=agent,
                activity_type="INFERENCE_BACKFILL",
                title=(
                    f"Inference backfill: signal={body.signal_id} symbol={body.symbol} "
                    f"[{body.start.isoformat()}, {body.end.isoformat()})"
                ),
                strategy_code=None,
                details={
                    "signal_id": str(body.signal_id),
                    "symbol": body.symbol,
                    "interval_name": body.interval_name,
                    "start": body.start.isoformat(),
                    "end": body.end.isoformat(),
                    "source": body.source,
                    "rows_in_window": response.get("rows_in_window"),
                    "rows_written": response.get("rows_written"),
                    "rows_skipped_missing_features": response.get(
                        "rows_skipped_missing_features"
                    ),
                    "model_id": response.get("model_id"),
                },
                redis_client=request.app.state.redis,
            )
    except Exception:  # noqa: BLE001
        pass

    await cache_response(
        request, agent, "inference-backfill", idempotency_key, response
    )
    return response
