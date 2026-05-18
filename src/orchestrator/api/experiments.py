"""Experiment tracking write + read surface (R1.A of ML/research uplift).

Endpoints:

  POST   /experiments              — start a run, returns run_id
  PATCH  /experiments/{id}/params  — replace params snapshot (one-shot)
  PATCH  /experiments/{id}/tags    — set or merge tags
  PATCH  /experiments/{id}/metrics — append one or more metric rows
  POST   /experiments/{id}/finish  — transition to terminal status
  GET    /experiments              — list with filters
  GET    /experiments/{id}         — single run with per-fold metrics
  GET    /experiments/leaderboard  — top runs by named summary metric

V92 schema notes:
  * experiment_run.params, tags, summary_metrics, leakage_report are JSONB.
    The asyncpg codec at infra/db.py handles dict→jsonb both ways — handlers
    pass dicts directly.
  * experiment_metric is normalized; run-level scalars (fold_idx=NULL) are
    mirrored into experiment_run.summary_metrics by log_metric / log_metrics.
  * The CHECK constraint chk_experiment_metric_value_finite rejects NaN / Inf
    at DB level. Handlers also pre-filter so callers get a clean 422 instead
    of a 500.

Idempotency:
  * Every state-changing endpoint honours Idempotency-Key.
  * /finish has an additional natural guard: the underlying UPDATE matches
    WHERE status = 'running', so a second /finish on a terminal row returns
    None from the repo and surfaces as 409 here.
  * /metrics replay via Idempotency-Key skips the actual INSERT (preserves
    the cached response). Without a key, a retry CAN duplicate metric rows —
    same convention as models.py.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field, field_validator

from orchestrator.repo import experiments as experiments_repo

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/experiments", tags=["experiments"])


# ─────────────────────────────────────────────────────────────────────────
# Request shapes
# ─────────────────────────────────────────────────────────────────────────


class StartRunRequest(BaseModel):
    """Body for ``POST /experiments``. Mirrors the V92 spec_* columns plus
    an optional params + tags snapshot at start time.

    ``spec_name`` is the only required field — the rest map directly onto
    the ``ModelSpec`` shape in blackheart-train and are NULL'd in the
    registry when omitted. Callers commonly snapshot params after argparse
    resolution rather than at insert; that's fine — :func:`set_params`
    accepts the post-resolution dict.
    """

    spec_name: str = Field(..., min_length=1, max_length=120)
    spec_version: str | None = Field(default=None, max_length=40)
    spec_symbol: str | None = Field(default=None, max_length=20)
    spec_interval: str | None = Field(default=None, max_length=10)
    spec_horizon_bars: int | None = Field(default=None, ge=1, le=10_000)
    git_sha: str | None = Field(default=None, max_length=64)
    dataset_sha: str | None = Field(default=None, max_length=64)
    params: dict[str, Any] | None = None
    tags: dict[str, Any] | None = None


class SetParamsRequest(BaseModel):
    params: dict[str, Any]


class SetTagsRequest(BaseModel):
    tags: dict[str, Any]
    merge: bool = Field(
        default=True,
        description=(
            "True (default) → shallow-merge with the existing tags; False → "
            "replace wholesale. Merge is the common case — tags accumulate "
            "over a run's lifetime."
        ),
    )


class MetricEntry(BaseModel):
    """One row for ``PATCH /experiments/{id}/metrics``.

    ``fold_idx`` NULL = run-level scalar (gets mirrored into
    ``experiment_run.summary_metrics``). Per-fold metrics carry their fold
    index (0..N-1).

    ``value`` is validated against NaN/Inf at the Pydantic layer so the
    caller gets a 422 with a clear field path instead of a 500 from the
    DB CHECK constraint.
    """

    name: str = Field(..., min_length=1, max_length=80)
    value: float
    fold_idx: int | None = Field(default=None, ge=0, le=10_000)
    step: int | None = Field(default=None, ge=0)

    @field_validator("value")
    @classmethod
    def _reject_nonfinite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError(
                f"value must be finite (got {v!r}); NaN/Inf rejected by "
                "experiment_metric CHECK constraint — sanitize upstream."
            )
        return v


class LogMetricsRequest(BaseModel):
    metrics: list[MetricEntry] = Field(default_factory=list)


class FinishRunRequest(BaseModel):
    status: Literal["completed", "failed", "aborted"]
    content_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    leakage_report: dict[str, Any] | None = None
    error_message: str | None = None
    # R1 follow-up: dataset_sha is computed post-load, so it lands at
    # finish-time even though the column also exists on start_run. NULL
    # leaves the start-time value intact (COALESCE in the repo).
    dataset_sha: str | None = Field(default=None, max_length=64)


# ─────────────────────────────────────────────────────────────────────────
# Idempotency helper — same shape used across models.py
# ─────────────────────────────────────────────────────────────────────────


async def _replay_or_none(request: Request, cache_key: str | None) -> Any | None:
    """Return the cached response for ``cache_key``, raising HTTPException
    if the cached value was a serialized error. ``None`` means no cached
    entry — handler should execute the work.

    Bug-#8 fix (2026-05-17): the sentinel check is ``is True`` rather than
    truthy. The cache stores both success responses (arbitrary dicts) and
    error envelopes ({"_idempotent_error": True, "status_code": ...,
    "detail": ...}). A success response containing a top-level key
    ``_idempotent_error`` would have been misclassified by the prior
    ``cached.get("_idempotent_error")`` truthy check. ``is True``
    constrains the match to the exact sentinel shape this module writes.
    """
    if cache_key is None:
        return None
    cached = await request.app.state.idempotency.get(request.state.agent_name, cache_key)
    if cached is None:
        return None
    if isinstance(cached, dict) and cached.get("_idempotent_error") is True:
        raise HTTPException(status_code=cached["status_code"], detail=cached["detail"])
    return cached


async def _cache_response(request: Request, cache_key: str | None, response: Any) -> None:
    if cache_key is None:
        return
    await request.app.state.idempotency.put(
        request.state.agent_name, cache_key, response,
    )


async def _cache_error(
    request: Request,
    cache_key: str | None,
    *,
    status_code: int,
    detail: str,
) -> None:
    if cache_key is None:
        return
    await request.app.state.idempotency.put(
        request.state.agent_name, cache_key,
        {"_idempotent_error": True, "status_code": status_code, "detail": detail},
    )


# ─────────────────────────────────────────────────────────────────────────
# Write endpoints
# ─────────────────────────────────────────────────────────────────────────


@router.post("")
async def start_run(request: Request, body: StartRunRequest):
    """Open a new experiment_run with status='running'.

    Idempotent on ``Idempotency-Key`` — same key replays the original
    response (same run_id), which is the contract callers need when a
    network retry happens mid-startup.
    """
    agent = request.state.agent_name
    key = request.state.idempotency_key
    cache_key = f"experiments-start:{key}" if key else None

    replayed = await _replay_or_none(request, cache_key)
    if replayed is not None:
        return replayed

    async with request.app.state.db.acquire() as conn:
        row = await experiments_repo.insert_run(
            conn,
            spec_name=body.spec_name,
            spec_version=body.spec_version,
            spec_symbol=body.spec_symbol,
            spec_interval=body.spec_interval,
            spec_horizon_bars=body.spec_horizon_bars,
            git_sha=body.git_sha,
            dataset_sha=body.dataset_sha,
            params=body.params,
            tags=body.tags,
            created_by=f"orchestrator:{agent}",
        )

    response = {
        "run_id": str(row["run_id"]),
        "status": row["status"],
        "started_at": row["started_at"].isoformat() if row["started_at"] else None,
    }
    logger.info(
        "experiment started | agent=%s run_id=%s spec=%s symbol=%s",
        agent, row["run_id"], body.spec_name, body.spec_symbol,
    )
    await _cache_response(request, cache_key, response)
    return response


@router.patch("/{run_id}/params")
async def set_params(request: Request, run_id: UUID, body: SetParamsRequest):
    """Replace the run's params JSONB snapshot wholesale.

    Why replace, not merge: params describe the final hyperparameter
    configuration after argparse + spec resolution. Incremental merges
    would conflate config sources. For accumulating annotations use
    /tags.

    404 if run_id is unknown.
    """
    agent = request.state.agent_name
    key = request.state.idempotency_key
    cache_key = f"experiments-params:{run_id}:{key}" if key else None

    replayed = await _replay_or_none(request, cache_key)
    if replayed is not None:
        return replayed

    async with request.app.state.db.acquire() as conn:
        row = await experiments_repo.set_params(
            conn, run_id=run_id, params=body.params, actor=f"orchestrator:{agent}",
        )
    if row is None:
        detail = f"experiment_run {run_id} not found"
        await _cache_error(request, cache_key, status_code=404, detail=detail)
        raise HTTPException(status_code=404, detail=detail)

    response = {
        "run_id": str(row["run_id"]),
        "params": row["params"],
        "updated_time": row["updated_time"].isoformat() if row["updated_time"] else None,
    }
    await _cache_response(request, cache_key, response)
    return response


@router.patch("/{run_id}/tags")
async def set_tags(request: Request, run_id: UUID, body: SetTagsRequest):
    """Merge (default) or replace the run's tags JSONB blob.

    Merge uses Postgres ``||`` on JSONB — rightmost wins on collision, so
    the new ``tags`` argument overrides existing keys. Top-level only;
    nested merges require an explicit replace.

    404 if run_id is unknown.
    """
    agent = request.state.agent_name
    key = request.state.idempotency_key
    cache_key = f"experiments-tags:{run_id}:{key}" if key else None

    replayed = await _replay_or_none(request, cache_key)
    if replayed is not None:
        return replayed

    async with request.app.state.db.acquire() as conn:
        row = await experiments_repo.set_tags(
            conn, run_id=run_id, tags=body.tags, merge=body.merge,
            actor=f"orchestrator:{agent}",
        )
    if row is None:
        detail = f"experiment_run {run_id} not found"
        await _cache_error(request, cache_key, status_code=404, detail=detail)
        raise HTTPException(status_code=404, detail=detail)

    response = {
        "run_id": str(row["run_id"]),
        "tags": row["tags"],
        "updated_time": row["updated_time"].isoformat() if row["updated_time"] else None,
    }
    await _cache_response(request, cache_key, response)
    return response


@router.patch("/{run_id}/metrics")
async def log_metrics(request: Request, run_id: UUID, body: LogMetricsRequest):
    """Append one or more metric rows.

    Run-level scalars (``fold_idx=None``) are mirrored into
    ``experiment_run.summary_metrics`` so leaderboard reads stay JOIN-free.

    Idempotency: ``Idempotency-Key`` replay returns the cached response and
    SKIPS the INSERT. Without a key, a network retry will duplicate rows —
    that's the same convention as models.py and matches the "append-only"
    nature of metric logs.

    Empty ``metrics`` list is a no-op (returns inserted=0). Useful when a
    fold loop emits nothing for a given run.

    404 if run_id is unknown.
    """
    agent = request.state.agent_name
    key = request.state.idempotency_key
    cache_key = f"experiments-metrics:{run_id}:{key}" if key else None

    replayed = await _replay_or_none(request, cache_key)
    if replayed is not None:
        return replayed

    async with request.app.state.db.acquire() as conn:
        # Existence guard: the FK on experiment_metric.run_id would raise
        # ForeignKeyViolationError as a 500. We want a clean 404 instead.
        exists = await conn.fetchval(
            "SELECT 1 FROM experiment_run WHERE run_id = $1", run_id,
        )
        if not exists:
            detail = f"experiment_run {run_id} not found"
            await _cache_error(request, cache_key, status_code=404, detail=detail)
            raise HTTPException(status_code=404, detail=detail)

        inserted = await experiments_repo.log_metrics(
            conn, run_id=run_id,
            metrics=[m.model_dump() for m in body.metrics],
        )

    response = {"run_id": str(run_id), "inserted": inserted}
    await _cache_response(request, cache_key, response)
    return response


@router.post("/{run_id}/finish")
async def finish_run(request: Request, run_id: UUID, body: FinishRunRequest):
    """Transition the run to a terminal status (completed/failed/aborted).

    Optimistic guard inside the repo: only succeeds if current status is
    'running'. A second /finish call returns None from the repo → 409 here
    so callers can distinguish "race" from "unknown id" (404).

    ``content_sha256`` should be set when the run produced a registered
    model artifact — the V92 FK enforces referential integrity. NULL is
    correct for failed runs, --no-write smoke runs, and any path that
    didn't reach /models/register.

    Idempotent on ``Idempotency-Key`` — re-finishing the same run with the
    same key returns the cached response (the second /finish without a key
    would 409).
    """
    agent = request.state.agent_name
    key = request.state.idempotency_key
    cache_key = f"experiments-finish:{run_id}:{key}" if key else None

    replayed = await _replay_or_none(request, cache_key)
    if replayed is not None:
        return replayed

    async with request.app.state.db.acquire() as conn:
        row = await experiments_repo.finish_run(
            conn,
            run_id=run_id,
            status=body.status,
            content_sha256=body.content_sha256,
            leakage_report=body.leakage_report,
            error_message=body.error_message,
            dataset_sha=body.dataset_sha,
            actor=f"orchestrator:{agent}",
        )
        if row is None:
            # Either unknown id OR already terminal. Re-fetch to distinguish.
            existing = await conn.fetchval(
                "SELECT status FROM experiment_run WHERE run_id = $1", run_id,
            )
        else:
            existing = None

    if row is None:
        if existing is None:
            detail = f"experiment_run {run_id} not found"
            await _cache_error(request, cache_key, status_code=404, detail=detail)
            raise HTTPException(status_code=404, detail=detail)
        detail = (
            f"experiment_run {run_id} already terminal (status={existing!r}); "
            "use Idempotency-Key to replay the original /finish response."
        )
        await _cache_error(request, cache_key, status_code=409, detail=detail)
        raise HTTPException(status_code=409, detail=detail)

    response = {
        "run_id": str(row["run_id"]),
        "status": row["status"],
        "started_at": row["started_at"].isoformat() if row["started_at"] else None,
        "finished_at": row["finished_at"].isoformat() if row["finished_at"] else None,
        "duration_seconds": row["duration_seconds"],
        "content_sha256": row["content_sha256"],
        "dataset_sha": row["dataset_sha"],
        "summary_metrics": row["summary_metrics"],
    }
    logger.info(
        "experiment finished | agent=%s run_id=%s status=%s duration=%ss",
        agent, row["run_id"], row["status"], row["duration_seconds"],
    )
    await _cache_response(request, cache_key, response)
    return response


# ─────────────────────────────────────────────────────────────────────────
# Read endpoints
# ─────────────────────────────────────────────────────────────────────────


@router.get("/leaderboard")
async def get_leaderboard(
    request: Request,
    spec_name: str = Query(..., min_length=1, description="Spec to leaderboard."),
    metric: str = Query(..., min_length=1, description="Summary metric key (e.g. oof_dsr, oof_auc)."),
    order: Literal["asc", "desc"] = Query("desc"),
    status: str | None = Query("completed", description="Filter on run status. Pass ''/null to include all."),
    limit: int = Query(20, ge=1, le=500),
):
    """Top runs for ``spec_name`` ordered by a named summary metric.

    Reads from ``experiment_run.summary_metrics`` JSONB. Runs that didn't
    log the requested metric are filtered out. ``status`` default is
    ``completed`` so the leaderboard doesn't surface in-flight or failed
    runs.
    """
    # FastAPI's Query default doesn't natively turn '' into None — accept
    # both spellings so curl-friendly callers can pass ?status= for "all".
    effective_status = None if status in (None, "", "null") else status

    async with request.app.state.db.acquire() as conn:
        rows = await experiments_repo.leaderboard(
            conn,
            spec_name=spec_name, metric=metric, order=order,
            status=effective_status, limit=limit,
        )
    return {
        "spec_name": spec_name,
        "metric": metric,
        "order": order,
        "limit": limit,
        "rows": rows,
    }


@router.get("/{run_id}")
async def get_run(request: Request, run_id: UUID):
    """Full run detail including per-fold metric rows."""
    async with request.app.state.db.acquire() as conn:
        row = await experiments_repo.get_run(conn, run_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"experiment_run {run_id} not found")
    return row


@router.get("")
async def list_runs(
    request: Request,
    spec_name: str | None = Query(None),
    status: str | None = Query(None),
    symbol: str | None = Query(None),
    has_artifact: bool | None = Query(
        None,
        description=(
            "True → only runs with content_sha256 set (produced a registered model). "
            "False → only runs without (failed / --no-write). Null → both."
        ),
    ),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """List experiment_run headers with optional filters.

    Ordered by started_at DESC — newest first. Per-fold metrics are NOT
    included; use ``GET /experiments/{id}`` for that.
    """
    async with request.app.state.db.acquire() as conn:
        rows = await experiments_repo.list_runs(
            conn,
            spec_name=spec_name, status=status, symbol=symbol,
            has_artifact=has_artifact, limit=limit, offset=offset,
        )
        total = await experiments_repo.count_runs(
            conn,
            spec_name=spec_name, status=status, symbol=symbol,
            has_artifact=has_artifact,
        )
    return {"runs": rows, "total": total, "limit": limit, "offset": offset}
