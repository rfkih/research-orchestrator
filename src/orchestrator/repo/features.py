"""asyncpg queries against feature_registry / feature_compute_run /
feature_values (V66 / V70 / V72 / V73 / V74 tables).

Read-only — orchestrator connects as ``blackheart_research`` which has
SELECT but not INSERT/UPDATE on these tables. Writes happen inside
``blackheart-ingest`` running as ``blackheart_trading``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID


def _to_naive_utc(dt: datetime) -> datetime:
    """feature_values.ts is TIMESTAMPTZ (asyncpg -> tz-aware), while
    market_data/feature_store use naive TIMESTAMP. Normalize to naive UTC
    so the /signal-screen point-in-time merge compares like with like
    (aware-vs-naive comparison raises TypeError)."""
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


async def list_features(
    conn,
    *,
    family: str | None = None,
    status: str | None = None,
    label_direction: str | None = None,
    limit: int = 200,
    offset: int = 0,
) -> list[dict]:
    """List feature_registry rows with optional filters. Ordered by
    family, name, version for stable agent enumeration.
    """
    wheres, params = ["TRUE"], []
    n = 1
    for val, col in [
        (family, "family"),
        (status, "status"),
        (label_direction, "label_direction"),
    ]:
        if val is not None:
            wheres.append(f"{col} = ${n}")
            params.append(val)
            n += 1
    params += [limit, offset]
    rows = await conn.fetch(
        f"""
        SELECT
            feature_name, version, family, owner,
            transformer_ref, inputs, output_dtype,
            symbols, intervals, pit_safe,
            ffill_policy, max_ffill_age_hours, backfill_strategy,
            label_for_model, label_direction, status,
            created_time, updated_time
        FROM feature_registry
        WHERE {" AND ".join(wheres)}
        ORDER BY family ASC, feature_name ASC, version ASC
        LIMIT ${n} OFFSET ${n + 1}
        """,
        *params,
    )
    return [dict(r) for r in rows]


async def count_features(
    conn,
    *,
    family: str | None = None,
    status: str | None = None,
    label_direction: str | None = None,
) -> int:
    wheres, params = ["TRUE"], []
    n = 1
    for val, col in [
        (family, "family"),
        (status, "status"),
        (label_direction, "label_direction"),
    ]:
        if val is not None:
            wheres.append(f"{col} = ${n}")
            params.append(val)
            n += 1
    row = await conn.fetchrow(
        f"SELECT COUNT(*) AS n FROM feature_registry WHERE {' AND '.join(wheres)}",
        *params,
    )
    return int(row["n"])


async def get_feature(conn, name: str, version: int) -> dict | None:
    row = await conn.fetchrow(
        """
        SELECT
            feature_name, version, family, owner,
            transformer_ref, inputs, output_dtype,
            symbols, intervals, pit_safe,
            ffill_policy, max_ffill_age_hours, backfill_strategy,
            label_for_model, label_direction, status,
            created_time, created_by, updated_time, updated_by
        FROM feature_registry
        WHERE feature_name = $1 AND version = $2
        """,
        name,
        version,
    )
    return dict(row) if row else None


async def feature_coverage(conn, name: str, version: int) -> dict:
    """Compute lightweight coverage stats for one feature's values plus
    the most-recent compute_run audit row.
    """
    cov = await conn.fetchrow(
        """
        SELECT
            COUNT(*) AS row_count,
            MIN(ts) AS first_ts,
            MAX(ts) AS last_ts,
            COUNT(DISTINCT NULLIF(symbol, '')) AS distinct_symbols,
            COUNT(DISTINCT NULLIF(interval, '')) AS distinct_intervals,
            MIN(value)::numeric AS min_value,
            AVG(value)::numeric AS mean_value,
            MAX(value)::numeric AS max_value
        FROM feature_values
        WHERE feature_name = $1 AND version = $2
        """,
        name,
        version,
    )
    # rank 14: per-(symbol, interval) breakdown. The aggregate row_count
    # above hides the common new-symbol gap — a feature with 44k BTC rows
    # and 0 BNB/1h rows reports distinct_symbols≥1 and looks "populated",
    # but a BNB/1h sweep finds nothing. This surface lets the pre-queue
    # coverage check see the actual per-surface depth.
    by_surface_rows = await conn.fetch(
        """
        SELECT NULLIF(symbol, '')   AS symbol,
               NULLIF(interval, '') AS interval,
               COUNT(*)             AS row_count,
               MIN(ts)              AS first_ts,
               MAX(ts)              AS last_ts
        FROM feature_values
        WHERE feature_name = $1 AND version = $2
        GROUP BY NULLIF(symbol, ''), NULLIF(interval, '')
        ORDER BY symbol NULLS FIRST, interval NULLS FIRST
        """,
        name,
        version,
    )
    latest_run = await conn.fetchrow(
        """
        SELECT
            run_id, symbol, interval, range_start, range_end,
            rows_written, status, started_at, finished_at, error_message
        FROM feature_compute_run
        WHERE feature_name = $1 AND version = $2
        ORDER BY started_at DESC NULLS LAST
        LIMIT 1
        """,
        name,
        version,
    )
    return {
        "row_count": int(cov["row_count"] or 0),
        "first_ts": cov["first_ts"],
        "last_ts": cov["last_ts"],
        "distinct_symbols": int(cov["distinct_symbols"] or 0),
        "distinct_intervals": int(cov["distinct_intervals"] or 0),
        "min_value": cov["min_value"],
        "mean_value": cov["mean_value"],
        "max_value": cov["max_value"],
        "by_surface": [
            {
                "symbol": r["symbol"],
                "interval": r["interval"],
                "row_count": int(r["row_count"] or 0),
                "first_ts": r["first_ts"],
                "last_ts": r["last_ts"],
            }
            for r in by_surface_rows
        ],
        "latest_run": dict(latest_run) if latest_run else None,
    }


async def feature_sample(
    conn,
    name: str,
    version: int,
    *,
    n: int = 100,
    symbol: str | None = None,
    interval: str | None = None,
) -> list[dict]:
    """Most-recent N rows for a (feature, version), optionally scoped by
    symbol+interval. Sorted DESC so the freshest values come first.
    """
    wheres = ["feature_name = $1", "version = $2"]
    params: list[Any] = [name, version]
    p = 3
    if symbol is not None:
        wheres.append(f"symbol = ${p}")
        params.append(symbol)
        p += 1
    if interval is not None:
        wheres.append(f"interval = ${p}")
        params.append(interval)
        p += 1
    params.append(n)
    rows = await conn.fetch(
        f"""
        SELECT ts, value, value_text, symbol, interval, compute_run_id
        FROM feature_values
        WHERE {" AND ".join(wheres)}
        ORDER BY ts DESC
        LIMIT ${p}
        """,
        *params,
    )
    return [dict(r) for r in rows]


async def get_compute_run(conn, run_id: UUID) -> dict | None:
    row = await conn.fetchrow(
        """
        SELECT
            run_id, feature_name, version, symbol, interval,
            range_start, range_end, rows_written, status,
            started_at, finished_at, error_message,
            created_time, created_by, updated_time, updated_by
        FROM feature_compute_run
        WHERE run_id = $1
        """,
        run_id,
    )
    return dict(row) if row else None


async def list_compute_runs(
    conn,
    *,
    feature_name: str | None = None,
    version: int | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    wheres, params = ["TRUE"], []
    n = 1
    if feature_name is not None:
        wheres.append(f"feature_name = ${n}")
        params.append(feature_name)
        n += 1
    if version is not None:
        wheres.append(f"version = ${n}")
        params.append(version)
        n += 1
    if status is not None:
        wheres.append(f"status = ${n}")
        params.append(status)
        n += 1
    params += [limit, offset]
    rows = await conn.fetch(
        f"""
        SELECT run_id, feature_name, version, symbol, interval,
               rows_written, status, started_at, finished_at, error_message
        FROM feature_compute_run
        WHERE {" AND ".join(wheres)}
        ORDER BY started_at DESC NULLS LAST
        LIMIT ${n} OFFSET ${n + 1}
        """,
        *params,
    )
    return [dict(r) for r in rows]


async def fetch_feature_series(
    conn,
    *,
    name: str,
    symbol: str,
    interval: str,
    start,
    end,
) -> tuple[list[tuple], dict]:
    """Ascending ``[(ts, value), ...]`` for one feature_values series, for
    the /signal-screen point-in-time alignment.

    Series resolution: feature_values rows are keyed (feature_name, version,
    symbol, interval, ts) where symbol/interval may be '' for symbol-less /
    bar-less (macro) series. We pin version to the MAX present for the name
    (latest definition), then try scopes from most to least specific:

      1. (symbol, interval)  -- true per-bar series
      2. (symbol, '')        -- per-symbol, not bar-aligned
      3. ('', interval)      -- symbol-less but bar-aligned
      4. ('', '')            -- macro / global series

    First non-empty scope wins. The ts window is widened backwards by the
    caller (start may be far before the bar window) so the point-in-time
    "latest value with ts <= bar start" join has a value for the first bars.
    NULL values are dropped. Returns (points, meta) where meta records the
    scope + version that resolved, for the screen's audit trail.
    """
    version = await conn.fetchval(
        "SELECT MAX(version) FROM feature_values WHERE feature_name = $1", name
    )
    if version is None:
        return [], {"resolved": False}

    scopes = [(symbol, interval), (symbol, ""), ("", interval), ("", "")]
    seen: set[tuple[str, str]] = set()
    for scope_symbol, scope_interval in scopes:
        if (scope_symbol, scope_interval) in seen:
            continue
        seen.add((scope_symbol, scope_interval))
        rows = await conn.fetch(
            """
            SELECT ts, value
            FROM feature_values
            WHERE feature_name = $1 AND version = $2
              AND symbol = $3 AND interval = $4
              AND ts >= $5 AND ts < $6
              AND value IS NOT NULL
            ORDER BY ts ASC
            """,
            name, version, scope_symbol, scope_interval, start, end,
        )
        if rows:
            return (
                [(_to_naive_utc(r["ts"]), float(r["value"])) for r in rows],
                {
                    "resolved": True,
                    "version": int(version),
                    "scope_symbol": scope_symbol,
                    "scope_interval": scope_interval,
                },
            )
    return [], {"resolved": False, "version": int(version)}
