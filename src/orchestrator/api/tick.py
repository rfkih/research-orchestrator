"""``POST /tick`` — run one research iteration end-to-end.

Synchronous on purpose: a backtest typically completes in 30-300s on the
research JVM, well within an HTTP request budget. The poll cap is 1800s.
For agents that don't want to hold a connection that long, the recommended
pattern is to call once per cron firing — not loop in-memory.

Idempotency: ``Idempotency-Key`` is honoured. Replays return the original
response. Without a key, two concurrent ticks are still safe (the DB-level
``SKIP LOCKED`` claim guarantees they pick different queue rows or one
gets ``empty_queue``).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from ..services.idempotency import cache_response, replay_cached_response
from ..services.tick import run_tick
from .deps import get_agent_name

router = APIRouter(tags=["tick"])


class TickResponse(BaseModel):
    """Loose schema — fields are ``None`` for empty-queue ticks."""

    outcome: str
    queue_id: str | None = None
    iteration_id: str | None = None
    backtest_run_id: str | None = None
    statistical_verdict: str | None = None
    verdict: str | None = None
    notes: list[str]
    next_actions: list[dict[str, Any]]


@router.post("/tick", response_model=TickResponse)
async def post_tick(
    request: Request,
    agent: str = Depends(get_agent_name),
) -> dict[str, Any]:
    cached, idempotency_key = await replay_cached_response(request, agent, "tick")
    if cached is not None:
        return cached

    result = await run_tick(
        db=request.app.state.db,
        jvm=request.app.state.jvm,
        settings=request.app.state.settings,
        agent_name=agent,
        session_id=request.state.session_id,
        redis_client=request.app.state.redis,
    )
    response = result.to_dict()
    await cache_response(request, agent, "tick", idempotency_key, response)
    return response
