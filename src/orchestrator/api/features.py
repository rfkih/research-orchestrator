"""FastAPI routes exposing the M3 feature pipeline to the quant-researcher
agent.

Endpoints (per blueprint § 11):

* ``GET  /features``                              — registry listing
* ``GET  /features/{name}/v/{version}``           — spec + coverage
* ``GET  /features/{name}/v/{version}/sample``    — most recent N values
* ``GET  /features/runs``                         — recent compute_run rows
* ``GET  /features/runs/{run_id}``                — single compute_run audit
* ``POST /features/{name}/v/{version}/backfill``  — proxy to blackheart-ingest

``POST /features/register`` is deliberately NOT exposed yet. Today
features are registered via Flyway migration + a definitions.py update;
making the registry agent-writable would require either dynamic
transformer loading from the registry row (substantial refactor) or a
"registered-in-DB-but-not-in-code" status that's confusing to operate.
Deferred to M5 when the model-registration flow surfaces.
"""
from __future__ import annotations

import logging
from datetime import datetime
from uuid import UUID

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from orchestrator.repo import features as features_repo

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/features", tags=["features"])


@router.get("")
async def list_features(
    request: Request,
    family: str | None = Query(None, description="macro / positioning / flow / market_structure / sentiment / label / …"),
    status: str | None = Query(None, description="Registry status filter (default: any)"),
    label_direction: str | None = Query(None, description="forward / backward / null"),
    limit: int = Query(200, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """List registered features. Filters compose with AND."""
    async with request.app.state.db.acquire() as conn:
        rows = await features_repo.list_features(
            conn,
            family=family,
            status=status,
            label_direction=label_direction,
            limit=limit,
            offset=offset,
        )
        total = await features_repo.count_features(
            conn, family=family, status=status, label_direction=label_direction
        )
    return {"features": rows, "total": total, "limit": limit, "offset": offset}


@router.get("/runs")
async def list_runs(
    request: Request,
    feature_name: str | None = None,
    version: int | None = None,
    status: str | None = Query(None, description="pending / running / done / failed / cancelled"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List recent feature_compute_run rows. Ordered by started_at DESC."""
    async with request.app.state.db.acquire() as conn:
        runs = await features_repo.list_compute_runs(
            conn,
            feature_name=feature_name,
            version=version,
            status=status,
            limit=limit,
            offset=offset,
        )
    return {"runs": runs, "limit": limit, "offset": offset}


@router.get("/runs/{run_id}")
async def get_run(request: Request, run_id: UUID):
    """Single feature_compute_run audit row."""
    async with request.app.state.db.acquire() as conn:
        row = await features_repo.get_compute_run(conn, run_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"compute_run {run_id} not found")
    return row


@router.get("/{name}/v/{version}")
async def get_feature(request: Request, name: str, version: int):
    """Registry spec + coverage stats for one (name, version)."""
    async with request.app.state.db.acquire() as conn:
        spec = await features_repo.get_feature(conn, name, version)
        if spec is None:
            raise HTTPException(
                status_code=404,
                detail=f"feature '{name}' v{version} not in registry",
            )
        coverage = await features_repo.feature_coverage(conn, name, version)
    return {"spec": spec, "coverage": coverage}


@router.get("/{name}/v/{version}/sample")
async def feature_sample(
    request: Request,
    name: str,
    version: int,
    n: int = Query(100, ge=1, le=1000, description="Number of recent rows"),
    symbol: str | None = Query(None, description="Scope to one symbol (empty string for global)"),
    interval: str | None = Query(None, description="Scope to one interval"),
):
    """Most-recent N rows from ``feature_values`` for the given (name,
    version), DESC by ts. Use ``symbol``/``interval`` to scope per-bar
    features.
    """
    async with request.app.state.db.acquire() as conn:
        spec = await features_repo.get_feature(conn, name, version)
        if spec is None:
            raise HTTPException(
                status_code=404,
                detail=f"feature '{name}' v{version} not in registry",
            )
        rows = await features_repo.feature_sample(
            conn, name, version, n=n, symbol=symbol, interval=interval
        )
    return {
        "feature_name": name,
        "version": version,
        "n_requested": n,
        "n_returned": len(rows),
        "symbol_filter": symbol,
        "interval_filter": interval,
        "rows": rows,
    }


class FeatureBackfillRequest(BaseModel):
    start: datetime = Field(..., description="ISO datetime (UTC, naive). Window lower bound.")
    end: datetime = Field(..., description="ISO datetime (UTC, naive). Window upper bound.")
    symbol: str | None = Field(default=None, description="Required for per-bar features.")
    interval: str | None = Field(default=None, description="Required for per-bar features.")


@router.post("/{name}/v/{version}/backfill")
async def backfill_feature(
    request: Request,
    name: str,
    version: int,
    body: FeatureBackfillRequest,
):
    """Trigger one feature compute on the blackheart-ingest worker.

    Synchronous proxy — the orchestrator waits for the worker to finish
    (timeout governed by ``ORCH_INGEST_REQUEST_TIMEOUT_S``, default 600s).
    Returns ``{run_id, rows_written, status, duration_seconds, ...}``
    straight from the worker so the agent can immediately call
    ``GET /features/runs/{run_id}`` to inspect the audit row.

    Idempotency: honors ``Idempotency-Key`` — repeated POSTs with the same
    key get the cached response (~24h TTL via ``app.state.idempotency``).
    Success bodies AND deterministic 4xx errors (unknown feature, missing
    symbol/interval, etc.) are cached; transient failures (worker 5xx,
    httpx errors / timeouts) are NOT cached, so retries get a fresh
    attempt. Useful for cron-driven backfills that retry on transient
    failures.
    """
    agent = request.state.agent_name
    idempotency_key = request.state.idempotency_key
    store = request.app.state.idempotency
    cache_key = f"feature-backfill:{idempotency_key}" if idempotency_key else None
    if cache_key:
        cached = await store.get(agent, cache_key)
        if cached is not None:
            # Cached 4xx replays as the same HTTPException; cached success
            # replays as the response body.
            if isinstance(cached, dict) and cached.get("_idempotent_error"):
                raise HTTPException(
                    status_code=cached["status_code"],
                    detail=cached["detail"],
                )
            return cached

    # Validate the feature exists in the registry before bothering the
    # worker. Cheap precondition + clearer 404 than the worker's variant.
    async with request.app.state.db.acquire() as conn:
        spec = await features_repo.get_feature(conn, name, version)
    if spec is None:
        detail = f"feature '{name}' v{version} not in feature_registry"
        if cache_key:
            await store.put(
                agent,
                cache_key,
                {"_idempotent_error": True, "status_code": 404, "detail": detail},
            )
        raise HTTPException(status_code=404, detail=detail)

    settings = request.app.state.settings
    url = f"{settings.ingest_base_url.rstrip('/')}/compute/{name}/v/{version}"
    payload = {
        "start": body.start.isoformat(),
        "end": body.end.isoformat(),
        "symbol": body.symbol,
        "interval": body.interval,
    }
    logger.info(
        "feature backfill | agent=%s feature=%s v%d symbol=%s interval=%s window=%s..%s",
        agent, name, version, body.symbol, body.interval, body.start, body.end,
    )

    try:
        async with httpx.AsyncClient(timeout=settings.ingest_request_timeout_s) as client:
            resp = await client.post(url, json=payload)
    except httpx.HTTPError as e:
        # Transient — do NOT cache; let retries hit a fresh worker.
        raise HTTPException(
            status_code=503,
            detail=f"blackheart-ingest unreachable at {url}: {type(e).__name__}: {e}",
        ) from e

    if resp.status_code >= 400:
        # Forward the worker's error envelope so the agent sees the real cause
        # (404 unknown feature in Python tuple, 400 missing symbol/interval,
        # 502 compute failure).
        forwarded_detail: object
        try:
            forwarded_detail = resp.json()
        except Exception:  # noqa: BLE001
            forwarded_detail = resp.text
        # 4xx from worker is deterministic (validation/contract); cache so
        # retries don't hammer the worker. 5xx is transient (compute crash,
        # DB blip) and should be retried fresh.
        if cache_key and 400 <= resp.status_code < 500:
            await store.put(
                agent,
                cache_key,
                {
                    "_idempotent_error": True,
                    "status_code": resp.status_code,
                    "detail": forwarded_detail,
                },
            )
        raise HTTPException(
            status_code=resp.status_code,
            detail=forwarded_detail,
        )

    response = resp.json()
    if cache_key:
        await store.put(agent, cache_key, response)
    return response
