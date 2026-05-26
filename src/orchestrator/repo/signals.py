"""asyncpg queries against ``signal_definition`` + ``signal_history``.

Read-side surface for the frontend ML pages (V66 schema). The
orchestrator writes signal definitions implicitly via the training
pipeline (``models.py`` `register_model` cascades when a spec ships with
``signal_*`` metadata); this module is read + status-transition writes
only.

Schema reminders (V66):
* ``signal_definition`` — pk ``signal_id``; ``name`` is the UNIQUE handle
  the trading-JVM ``MLRegimeGateGuard`` looks up at runtime.
* ``signal_history`` — composite pk ``(signal_id, symbol, ts)``, partitioned
  by month (V66 partition setup; V90 hypertable conversion). ``source``
  is one of ``stream | catchup_scan | historical_replay``.

All bound-strategy joins resolve via ``account_strategy.ml_regime_signal_name``
which references ``signal_definition.name`` — the natural-key edge, not
the UUID. Mirrors how the trading-JVM ``MLRegimeGateGuard`` looks up
signals at runtime.
"""
from __future__ import annotations

from typing import Any
from uuid import UUID


# ── list ──────────────────────────────────────────────────────────────────
async def list_signals(
    conn,
    *,
    status: str | None,
    symbol: str | None,
    interval_name: str | None,
    q: str | None,
    limit: int,
    offset: int,
) -> list[dict[str, Any]]:
    """Page of signals with denormalised model + bound-strategy info."""
    where = []
    params: list[Any] = []
    if status:
        params.append(status)
        where.append(f"sd.status = ${len(params)}")
    if symbol:
        params.append(symbol)
        where.append(f"mr.symbol = ${len(params)}")
    if interval_name:
        params.append(interval_name)
        where.append(f"mr.interval = ${len(params)}")
    if q:
        params.append(f"%{q}%")
        where.append(f"sd.name ILIKE ${len(params)}")
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""
    params.extend([limit, offset])
    sql = f"""
        SELECT
            sd.signal_id, sd.name, sd.model_id, sd.status, sd.horizon,
            sd.description, sd.created_time, sd.updated_time,
            mr.family AS model_family, mr.purpose AS model_purpose,
            mr.symbol AS model_symbol, mr.interval AS model_interval,
            mr.metrics->>'gauntlet_verdict' AS gauntlet_verdict,
            (
                SELECT array_agg(DISTINCT ac.strategy_code ORDER BY ac.strategy_code)
                FROM account_strategy ac
                WHERE ac.ml_regime_signal_name = sd.name
                  AND ac.ml_regime_gate_enabled = TRUE
            ) AS bound_strategy_codes
        FROM signal_definition sd
        LEFT JOIN model_registry mr ON mr.id = sd.model_id
        {where_sql}
        ORDER BY sd.created_time DESC
        LIMIT ${len(params) - 1} OFFSET ${len(params)}
    """
    return [dict(r) for r in await conn.fetch(sql, *params)]


async def count_signals(
    conn,
    *,
    status: str | None,
    symbol: str | None,
    interval_name: str | None,
    q: str | None,
) -> int:
    where = []
    params: list[Any] = []
    if status:
        params.append(status); where.append(f"sd.status = ${len(params)}")
    if symbol:
        params.append(symbol); where.append(f"mr.symbol = ${len(params)}")
    if interval_name:
        params.append(interval_name); where.append(f"mr.interval = ${len(params)}")
    if q:
        params.append(f"%{q}%"); where.append(f"sd.name ILIKE ${len(params)}")
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""
    row = await conn.fetchrow(
        f"""
        SELECT COUNT(*) AS n
        FROM signal_definition sd
        LEFT JOIN model_registry mr ON mr.id = sd.model_id
        {where_sql}
        """,
        *params,
    )
    return int(row["n"]) if row else 0


# ── single ────────────────────────────────────────────────────────────────
async def get_signal(conn, signal_id: UUID) -> dict[str, Any] | None:
    row = await conn.fetchrow(
        """
        SELECT
            sd.signal_id, sd.name, sd.model_id, sd.status, sd.horizon,
            sd.description, sd.created_time, sd.updated_time,
            mr.family AS model_family, mr.purpose AS model_purpose,
            mr.symbol AS model_symbol, mr.interval AS model_interval,
            mr.metrics AS model_metrics,
            (
                SELECT array_agg(DISTINCT ac.strategy_code ORDER BY ac.strategy_code)
                FROM account_strategy ac
                WHERE ac.ml_regime_signal_name = sd.name
                  AND ac.ml_regime_gate_enabled = TRUE
            ) AS bound_strategy_codes
        FROM signal_definition sd
        LEFT JOIN model_registry mr ON mr.id = sd.model_id
        WHERE sd.signal_id = $1
        """,
        signal_id,
    )
    return dict(row) if row else None


async def get_signal_health_counters(conn, signal_id: UUID) -> dict[str, Any]:
    """Light-weight stats used by both the monitor rollup and the
    signal-detail Live tab. Single query, single round-trip."""
    row = await conn.fetchrow(
        """
        SELECT
            MAX(ts)                                                        AS last_fire_ts,
            COUNT(*) FILTER (WHERE ts > NOW() - INTERVAL '24 hours')        AS fires_24h,
            COUNT(*) FILTER (WHERE ts > NOW() - INTERVAL '7 days')          AS fires_7d
        FROM signal_history
        WHERE signal_id = $1
          AND ts        >  NOW() - INTERVAL '30 days'
        """,
        signal_id,
    )
    return {
        "last_fire_ts": row["last_fire_ts"] if row else None,
        "fires_24h":    int(row["fires_24h"]) if row else 0,
        "fires_7d":     int(row["fires_7d"])  if row else 0,
    }


# ── firings ───────────────────────────────────────────────────────────────
async def list_firings(
    conn,
    *,
    signal_id: UUID,
    since: Any | None,
    until: Any | None,
    source: str | None,
    limit: int,
    offset: int,
) -> list[dict[str, Any]]:
    where = ["sh.signal_id = $1"]
    params: list[Any] = [signal_id]
    if since:
        params.append(since); where.append(f"sh.ts >= ${len(params)}")
    if until:
        params.append(until); where.append(f"sh.ts <  ${len(params)}")
    if source:
        params.append(source); where.append(f"sh.source = ${len(params)}")
    params.extend([limit, offset])
    sql = f"""
        SELECT sh.signal_id, sh.symbol, sh.ts, sh.value, sh.confidence,
               sh.produced_at, sh.source, sh.meta, sh.created_time
        FROM signal_history sh
        WHERE {' AND '.join(where)}
        ORDER BY sh.ts DESC
        LIMIT ${len(params) - 1} OFFSET ${len(params)}
    """
    return [dict(r) for r in await conn.fetch(sql, *params)]


async def count_firings(
    conn,
    *,
    signal_id: UUID,
    since: Any | None,
    until: Any | None,
    source: str | None,
) -> int:
    where = ["sh.signal_id = $1"]
    params: list[Any] = [signal_id]
    if since:
        params.append(since); where.append(f"sh.ts >= ${len(params)}")
    if until:
        params.append(until); where.append(f"sh.ts <  ${len(params)}")
    if source:
        params.append(source); where.append(f"sh.source = ${len(params)}")
    row = await conn.fetchrow(
        f"SELECT COUNT(*) AS n FROM signal_history sh WHERE {' AND '.join(where)}",
        *params,
    )
    return int(row["n"]) if row else 0


# ── daily counts ──────────────────────────────────────────────────────────
async def daily_counts(conn, signal_id: UUID, days: int = 30) -> list[dict[str, Any]]:
    rows = await conn.fetch(
        """
        SELECT (ts AT TIME ZONE 'UTC')::date AS date,
               COUNT(*)                       AS fire_count
        FROM signal_history
        WHERE signal_id = $1
          AND ts        >  NOW() - ($2::int * INTERVAL '1 day')
        GROUP BY 1
        ORDER BY 1
        """,
        signal_id, days,
    )
    return [dict(r) for r in rows]


# ── status transition ─────────────────────────────────────────────────────
async def update_status(
    conn,
    *,
    signal_id: UUID,
    new_status: str,
    updated_by: str,
) -> dict[str, Any] | None:
    """Atomic status flip. Returns the new row, or ``None`` if the signal
    doesn't exist. State-transition validation lives in the API layer so
    the error envelope can carry a precise ``error_code``."""
    row = await conn.fetchrow(
        """
        UPDATE signal_definition
        SET status = $2, updated_time = NOW(), updated_by = $3
        WHERE signal_id = $1
        RETURNING signal_id, name, model_id, status, horizon, description,
                  created_time, updated_time
        """,
        signal_id, new_status, updated_by,
    )
    return dict(row) if row else None
