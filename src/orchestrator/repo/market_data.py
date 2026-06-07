"""Read underlying price series from market_data for the buy-hold benchmark."""
from __future__ import annotations
from datetime import datetime
import asyncpg


async def fetch_daily_closes(
    conn: asyncpg.Connection, *, symbol: str, interval: str,
    start: datetime, end: datetime,
) -> list[tuple[datetime, float]]:
    """Ascending [(start_time, close_price), ...] for the buy-hold leg.
    Half-open [start, end). Mirrors the regime_analysis market_data read."""
    rows = await conn.fetch(
        """
        SELECT start_time, close_price
        FROM market_data
        WHERE symbol = $1 AND interval = $2
          AND start_time >= $3 AND start_time < $4
        ORDER BY start_time ASC
        """,
        symbol, interval, start, end,
    )
    return [(r["start_time"], float(r["close_price"])) for r in rows]
