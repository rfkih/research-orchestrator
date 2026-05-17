"""``POST /verdict-drift/scan`` — Phase 7.5 PROMOTED verdict-drift scan.

Compares the two most-recent ``cross_window_run`` rows for every
PROMOTED account_strategy and reports rank drops. Cheap (read-only,
two indexed lookups per strategy) — safe to call from a nightly cron
on the trading JVM.

Idempotent under ``vdrift:`` namespace. Repeat calls within the
idempotency TTL return the cached drift list rather than re-scanning,
which is fine — the underlying state changes only when the agent
inserts a new ``cross_window_run`` row.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from ..services.idempotency import cache_response, replay_cached_response
from ..services.verdict_drift import scan_promoted
from .deps import get_agent_name

router = APIRouter(tags=["verdict-drift"])


@router.post("/verdict-drift/scan")
async def post_verdict_drift_scan(
    request: Request,
    agent: str = Depends(get_agent_name),
) -> dict[str, Any]:
    cached, idempotency_key = await replay_cached_response(request, agent, "vdrift")
    if cached is not None:
        return cached

    drifts = await scan_promoted(db=request.app.state.db)
    payload = {"drifts": drifts, "count": len(drifts)}

    await cache_response(request, agent, "vdrift", idempotency_key, payload)
    return payload
