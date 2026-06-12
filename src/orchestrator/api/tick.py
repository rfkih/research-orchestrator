"""``POST /tick`` — run one research iteration end-to-end.

Synchronous on purpose: a backtest typically completes in 30-300s on the
research JVM, well within an HTTP request budget. The poll cap is
``settings.poll_timeout_s`` (env ``ORCH_POLL_TIMEOUT_S``, default 3600s);
on expiry the row returns to PENDING with the run id persisted, so the
next tick RESUMES the same run (``backtest_poll_cap_exceeded``, retryable).
For agents that don't want to hold a connection that long, the recommended
pattern is to call once per cron firing — not loop in-memory.

Idempotency: ``Idempotency-Key`` is honoured. Replays return the original
response. Without a key, two concurrent ticks are still safe (the DB-level
``SKIP LOCKED`` claim guarantees they pick different queue rows or one
gets ``empty_queue``).
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from ..services.idempotency import cache_response, replay_cached_response
from ..services.tick import run_tick
from ..services.tick_drain import (
    DEFAULT_MAX_CONSECUTIVE_WAITS,
    DEFAULT_MAX_ITERS,
    DEFAULT_MAX_WALL_CLOCK_S,
    drain_ticks,
)
from .deps import get_agent_name

router = APIRouter(tags=["tick"])


class TickSummary(BaseModel):
    """One-liner distilled from the tick result.

    Designed for the quant-runner (Stage B, haiku-tier) which polls /tick
    repeatedly and needs a single field — ``next_action`` — to decide
    whether to keep ticking or escalate to the researcher.
    """

    verdict_line: str
    next_action: str  # CONTINUE | GRADUATE | PIVOT | EMPTY_QUEUE | WAIT | INFRA_FAIL
    decision_hint: str


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
    # Optional for back-compat: the idempotency cache (TTL 24h) may hold
    # pre-Phase-2 responses without this field. Live calls always populate
    # it via ``TickResult.to_dict()``; only stale cached replays from the
    # deploy-transition window can land here as None.
    summary: TickSummary | None = None


class TickBody(BaseModel):
    """Optional input for ``POST /tick``. ``track`` (Phase 1) scopes the
    queue claim to one research loop; omit for the legacy global queue."""

    track: Literal["trading", "hedging"] | None = None


@router.post("/tick", response_model=TickResponse)
async def post_tick(
    request: Request,
    body: TickBody | None = None,
    agent: str = Depends(get_agent_name),
) -> dict[str, Any]:
    cached, idempotency_key = await replay_cached_response(request, agent, "tick")
    if cached is not None:
        return cached

    params = body or TickBody()
    result = await run_tick(
        db=request.app.state.db,
        jvm=request.app.state.jvm,
        settings=request.app.state.settings,
        agent_name=agent,
        session_id=request.state.session_id,
        redis_client=request.app.state.redis,
        track=params.track,
    )
    response = result.to_dict()
    await cache_response(request, agent, "tick", idempotency_key, response)
    return response


class TickDrainBody(BaseModel):
    """Input for ``POST /tick/drain``.

    All fields are optional with runner-playbook-aligned defaults. The
    caps protect the orchestrator from a stuck sweep holding a worker
    indefinitely; callers expecting a long drain should re-call on
    ``MAX_ITERS_REACHED`` / ``MAX_WALL_CLOCK_REACHED`` rather than
    raising the caps unbounded.
    """

    max_iters: int = Field(DEFAULT_MAX_ITERS, ge=1, le=200)
    max_wall_clock_s: int = Field(DEFAULT_MAX_WALL_CLOCK_S, ge=60, le=21600)
    max_consecutive_waits: int = Field(DEFAULT_MAX_CONSECUTIVE_WAITS, ge=1, le=10)
    # Dual-track (Phase 1): scope the drain to one research loop. None ⇒ the
    # legacy global drain (claims any PENDING row).
    track: Literal["trading", "hedging"] | None = None


@router.post("/tick/drain")
async def post_tick_drain(
    request: Request,
    body: TickDrainBody | None = None,
    agent: str = Depends(get_agent_name),
) -> dict[str, Any]:
    """Drive ``run_tick`` until a terminal ``next_action`` or a cap fires.

    Returns the structured digest the quant-runner sub-agent used to
    return as text (Status / Iters / Verdicts / Last / Next). The
    researcher can switch on ``terminal_action`` directly:

      * ``GRADUATE``  → POST /reviews/auto-run-checklist target_kind='graduation'
      * ``PIVOT``     → pick next archetype/axis
      * ``EMPTY_QUEUE`` → POST /queue with a new sweep
      * ``INFRA_FAIL`` → journal INFRA_FAILURE
      * ``MAX_ITERS_REACHED`` / ``MAX_WALL_CLOCK_REACHED`` → re-call

    Idempotency: ``Idempotency-Key`` is honoured for the drain envelope.
    A replay of the same key returns the original digest WITHOUT
    re-driving the queue. Each underlying ``/tick`` call retains its
    own queue-claim semantics (``FOR UPDATE SKIP LOCKED``); concurrent
    drains never collide on the same row.
    """
    cached, idempotency_key = await replay_cached_response(
        request, agent, "tick-drain"
    )
    if cached is not None:
        return cached

    params = body or TickDrainBody()
    digest = await drain_ticks(
        db=request.app.state.db,
        jvm=request.app.state.jvm,
        settings=request.app.state.settings,
        agent_name=agent,
        session_id=request.state.session_id,
        track=params.track,
        redis_client=request.app.state.redis,
        max_iters=params.max_iters,
        max_wall_clock_s=params.max_wall_clock_s,
        max_consecutive_waits=params.max_consecutive_waits,
    )
    await cache_response(request, agent, "tick-drain", idempotency_key, digest)
    return digest
