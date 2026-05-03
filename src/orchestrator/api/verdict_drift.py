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

from ..services.verdict_drift import scan_promoted
from .deps import get_agent_name

router = APIRouter(tags=["verdict-drift"])


@router.post("/verdict-drift/scan")
async def post_verdict_drift_scan(
    request: Request,
    agent: str = Depends(get_agent_name),
) -> dict[str, Any]:
    idempotency_key = getattr(request.state, "idempotency_key", None)
    store = request.app.state.idempotency
    if idempotency_key:
        cached = await store.get(agent, f"vdrift:{idempotency_key}")
        if cached is not None:
            return cached

    drifts = await scan_promoted(db=request.app.state.db)
    payload = {"drifts": drifts, "count": len(drifts)}

    if idempotency_key:
        await store.put(agent, f"vdrift:{idempotency_key}", payload)
    return payload
