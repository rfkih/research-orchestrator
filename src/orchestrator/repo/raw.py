"""Read-only queries against ingestion-plane tables:

* ``ml_ingest_schedule`` + ``ml_source_health`` (V67/V68) — the per-source
  cron schedule, last-pull status, cumulative counters.
* ``macro_raw`` / ``onchain_raw`` / ``news_raw`` / ``social_raw`` (V66) —
  per-source coverage by (series_id) for the time window the caller asks.

Coverage queries return COUNT/MIN/MAX without scanning the value column
so they remain cheap even over the full partition set.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any


# Tables exposed via /raw/{table}/coverage. Kept as a whitelist so the
# table name isn't user-controlled SQL injection bait.
_COVERAGE_TABLES: frozenset[str] = frozenset(
    {"macro_raw", "onchain_raw", "news_raw", "social_raw"}
)


def is_coverage_table(name: str) -> bool:
    return name in _COVERAGE_TABLES


async def list_ingest_sources(conn) -> list[dict]:
    """Join the schedule + health rows so the agent sees the full picture
    for each ingestion source in one query.
    """
    rows = await conn.fetch(
        """
        SELECT
            s.id              AS schedule_id,
            s.source,
            s.symbol,
            s.cron_expression,
            s.lookback_hours,
            s.enabled,
            s.config,
            s.last_run_at,
            s.last_success_at,
            s.last_error_message,
            s.next_run_at,
            h.health_status,
            h.last_pull_at,
            h.last_success_at AS health_last_success_at,
            h.last_failure_at,
            h.rows_inserted_total,
            h.rejected_pit_violations_total,
            h.consecutive_failures,
            h.errors_total,
            h.health_message
        FROM ml_ingest_schedule s
        LEFT JOIN ml_source_health h ON h.source = s.source
        ORDER BY s.source ASC, s.symbol ASC NULLS LAST
        """
    )
    return [dict(r) for r in rows]


async def raw_table_coverage(
    conn,
    table: str,
    *,
    source: str | None = None,
    series_id: str | None = None,
    from_ts: datetime | None = None,
    to_ts: datetime | None = None,
) -> list[dict]:
    """Per-(source, series_id) row count + min/max event_time within the
    optional window. Aggregates across all matching partitions.

    Table name must be in ``_COVERAGE_TABLES`` — callers should reject
    other values before invoking. Inline f-string is safe because the
    whitelist prevents arbitrary identifiers.
    """
    if not is_coverage_table(table):
        raise ValueError(
            f"Refusing coverage query on unknown table '{table}'. "
            f"Allowed: {sorted(_COVERAGE_TABLES)}"
        )

    wheres: list[str] = ["TRUE"]
    params: list[Any] = []
    n = 1
    if source is not None:
        wheres.append(f"source = ${n}")
        params.append(source)
        n += 1
    if series_id is not None:
        wheres.append(f"series_id = ${n}")
        params.append(series_id)
        n += 1
    if from_ts is not None:
        wheres.append(f"event_time >= ${n}")
        params.append(from_ts)
        n += 1
    if to_ts is not None:
        wheres.append(f"event_time <= ${n}")
        params.append(to_ts)
        n += 1

    rows = await conn.fetch(
        f"""
        SELECT
            source,
            series_id,
            COUNT(*)            AS row_count,
            MIN(event_time)     AS first_event_time,
            MAX(event_time)     AS last_event_time,
            MAX(ingestion_time) AS last_ingestion_time
        FROM {table}
        WHERE {" AND ".join(wheres)}
        GROUP BY source, series_id
        ORDER BY source, series_id
        """,
        *params,
    )
    return [dict(r) for r in rows]
