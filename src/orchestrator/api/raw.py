"""Read-only routes exposing the ingestion plane to the agent.

* ``GET /raw/sources`` — schedule + health for each of the 7 ML sources
* ``GET /raw/{table}/coverage`` — per-(source, series_id) coverage with
  optional ``source``, ``series_id``, ``from``, ``to`` filters

These let the agent answer "which feeds are alive, how far back does
each go, what's the last successful tick?" without poking the trading
JVM directly.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from orchestrator.repo import raw as raw_repo

router = APIRouter(prefix="/raw", tags=["raw"])


@router.get("/sources")
async def list_sources(request: Request):
    """Per-source schedule + health. One row per
    (``ml_ingest_schedule.source``, ``symbol``) tuple. ``health_status``
    is ``healthy`` / ``degraded`` / ``failed`` / ``unknown``.
    """
    async with request.app.state.db.acquire() as conn:
        rows = await raw_repo.list_ingest_sources(conn)
    return {"sources": rows, "count": len(rows)}


@router.get("/{table}/coverage")
async def table_coverage(
    request: Request,
    table: str,
    source: str | None = Query(None, description="Filter to one ingestion source"),
    series_id: str | None = Query(None, description="Filter to one series_id"),
    from_ts: datetime | None = Query(
        None,
        alias="from",
        description="Lower bound on event_time (inclusive)",
    ),
    to_ts: datetime | None = Query(
        None,
        alias="to",
        description="Upper bound on event_time (inclusive)",
    ),
):
    """Coverage stats for one of the four raw-data partitioned tables.

    Returns one row per (source, series_id) within the optional window:
    row_count, first/last event_time, last ingestion_time. Rejects
    unknown table names (whitelist guard).
    """
    if not raw_repo.is_coverage_table(table):
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown raw table '{table}'. "
                "Allowed: macro_raw, onchain_raw, news_raw, social_raw."
            ),
        )
    async with request.app.state.db.acquire() as conn:
        rows = await raw_repo.raw_table_coverage(
            conn,
            table,
            source=source,
            series_id=series_id,
            from_ts=from_ts,
            to_ts=to_ts,
        )
    return {
        "table": table,
        "filters": {
            "source": source,
            "series_id": series_id,
            "from": from_ts.isoformat() if from_ts else None,
            "to": to_ts.isoformat() if to_ts else None,
        },
        "coverage": rows,
        "count": len(rows),
    }
