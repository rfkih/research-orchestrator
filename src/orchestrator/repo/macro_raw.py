"""Read macro time series (funding rates + Deribit DVOL) from macro_raw.

The factor-combination layer (Phase 3) sources its CARRY (funding) and VOL
(implied-vol / DVOL) factors from the JVM-owned ``macro_raw`` table. This is a
DML-only read mirroring ``repo/market_data.fetch_daily_closes`` — half-open
``[start, end)`` window, ascending, params bound.
"""
from __future__ import annotations
from datetime import datetime
import asyncpg


async def fetch_series(
    conn: asyncpg.Connection, *, series_id: str, symbol: str | None = None,
    start: datetime, end: datetime,
) -> list[tuple[datetime, float]]:
    """Ascending ``[(event_time, float(value)), ...]`` for one macro series.

    ``symbol`` is optional: single-series feeds (e.g. BTC DVOL) carry a NULL or
    fixed symbol, while per-symbol feeds (funding rates) require it. Half-open
    ``[start, end)``.
    """
    rows = await conn.fetch(
        """
        SELECT event_time, value
        FROM macro_raw
        WHERE series_id = $1
          AND ($2::text IS NULL OR symbol = $2)
          AND event_time >= $3 AND event_time < $4
        ORDER BY event_time ASC
        """,
        series_id, symbol, start, end,
    )
    return [(r["event_time"], float(r["value"])) for r in rows]
