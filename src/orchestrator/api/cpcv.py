"""``POST /cpcv`` — combinatorial-block path validation on a realized ledger.

A small-sample robustness *diagnostic*, not a gate. It runs ONE full-span
backtest, then resamples the realized trade ledger into ``C(n_blocks,
k_test)`` overlapping sub-paths to produce a PF/Sharpe distribution plus a
regime/year stratification. See ``services/cpcv.py`` for the methodology
and why this is the right adaptation for fixed-param rule strategies.

Unlike ``POST /walk-forward`` this has **no graduation-review gate**: it
is an ad-hoc analysis tool the agent/operator can run on any
``account_strategy`` at any time. It deliberately does NOT write a
``walk_forward_run`` row — ``cpcv_verdict`` is advisory and must never
leak into the curator's ``stability_verdict`` gate signal. Result is
returned and activity-logged only.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from ..services.activity_logger import log_activity
from ..services.cpcv import run_cpcv
from ..services.idempotency import cache_response, replay_cached_response
from .deps import get_agent_name, get_db_conn

router = APIRouter(tags=["cpcv"])


class CpcvRequest(BaseModel):
    strategy_code: str = Field(..., min_length=1, max_length=60)
    interval_name: str = Field(..., description="5m|15m|1h|4h")
    instrument: str = Field("BTCUSDT", min_length=1, max_length=30)
    full_start: date = Field(default_factory=lambda: date(2024, 1, 1))
    full_end: date | None = None
    n_blocks: int = Field(8, ge=2, le=20)
    k_test: int = Field(4, ge=1, le=19)
    n_trials: int = Field(1, ge=1, le=10000, description="DSR multiplicity")
    overrides: dict[str, Any] | None = None


class CpcvResponse(BaseModel):
    backtest_run_id: str | None
    cpcv_verdict: str | None
    analysis: dict[str, Any]
    next_actions: list[dict[str, Any]]


@router.post("/cpcv", response_model=CpcvResponse)
async def post_cpcv(
    body: CpcvRequest,
    request: Request,
    agent: str = Depends(get_agent_name),
    conn=Depends(get_db_conn),
) -> dict[str, Any]:
    cached, idempotency_key = await replay_cached_response(request, agent, "cpcv")
    if cached is not None:
        return cached

    session_id = request.state.session_id
    redis_client = request.app.state.redis

    await log_activity(
        conn,
        session_id=session_id,
        agent_name=agent,
        activity_type="CPCV_SUBMITTED",
        title=f"CPCV submitted for {body.strategy_code}",
        strategy_code=body.strategy_code,
        details={
            "interval_name": body.interval_name,
            "instrument": body.instrument,
            "n_blocks": body.n_blocks,
            "k_test": body.k_test,
        },
        redis_client=redis_client,
    )

    try:
        result = await run_cpcv(
            db=request.app.state.db,
            jvm=request.app.state.jvm,
            settings=request.app.state.settings,
            agent_name=agent,
            strategy_code=body.strategy_code,
            interval_name=body.interval_name,
            instrument=body.instrument,
            full_start=body.full_start,
            full_end=body.full_end,
            overrides=body.overrides,
            n_blocks=body.n_blocks,
            k_test=body.k_test,
            n_trials=body.n_trials,
        )
    except Exception as _exc:
        await log_activity(
            conn,
            session_id=session_id,
            agent_name=agent,
            activity_type="CPCV_RESULT",
            title=f"CPCV failed for {body.strategy_code}",
            strategy_code=body.strategy_code,
            details={"error": str(_exc)},
            status="ERROR",
            redis_client=redis_client,
        )
        raise

    payload = result.to_dict()
    verdict = payload.get("cpcv_verdict")
    next_actions: list[dict[str, Any]] = [{
        "kind": "note",
        "hint": (
            "CPCV is advisory only — it does NOT replace the V11/V60 gate "
            "or the walk-forward ROBUST cutoff. Use it to judge small-sample "
            "robustness and temporal stationarity, not to promote."
        ),
    }]
    payload["next_actions"] = next_actions

    await log_activity(
        conn,
        session_id=session_id,
        agent_name=agent,
        activity_type="CPCV_RESULT",
        title=f"CPCV result for {body.strategy_code}: {verdict}",
        strategy_code=body.strategy_code,
        details={
            "cpcv_verdict": verdict,
            "backtest_run_id": payload.get("backtest_run_id"),
            "paths": (payload.get("analysis") or {}).get("paths"),
            "pooled": (payload.get("analysis") or {}).get("pooled"),
        },
        redis_client=redis_client,
    )

    await cache_response(request, agent, "cpcv", idempotency_key, payload)
    return payload
