"""Read-only access to the wide ``feature_store`` table (V1 baseline +
many column-add migrations; owned by the trading JVM).

The orchestrator never had a repo for this table before /signal-screen --
the JVM engines are its normal consumers. The screen needs exactly two
reads:

  * which numeric columns exist (so ``POST /signal-screen`` can resolve a
    ``signal`` name against the per-bar store before falling back to
    ``feature_values``), and
  * one (start_time, value) series per (symbol, interval, window).

SQL-injection note: ``feature_store`` columns arrive as user input (the
``signal`` field). We NEVER interpolate the raw request string -- the
column must first resolve against ``list_numeric_columns`` (server-side
``information_schema`` truth), then pass an identifier-shaped regex, and
only then is it quoted into the SELECT.
"""
from __future__ import annotations

import re
from datetime import datetime

import asyncpg

# Identifier guard for the dynamically-selected column. Resolution against
# information_schema happens first; this is defense-in-depth.
_IDENT_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")

# Key / bookkeeping columns that are never a screenable signal.
_META_COLUMNS = frozenset(
    {"id", "id_market_data", "symbol", "interval", "start_time", "end_time", "created_time"}
)

# information_schema data_types we accept as a signal series. Booleans are
# included (regime flags like is_bullish_breakout are legitimate 0/1
# signals); varchar regimes are not (no ordering).
_NUMERIC_TYPES = frozenset(
    {"numeric", "integer", "bigint", "smallint", "real", "double precision", "boolean"}
)


async def list_numeric_columns(conn: asyncpg.Connection) -> set[str]:
    """Set of feature_store column names usable as a screen signal."""
    rows = await conn.fetch(
        """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'feature_store'
        """
    )
    return {
        r["column_name"]
        for r in rows
        if r["data_type"] in _NUMERIC_TYPES and r["column_name"] not in _META_COLUMNS
    }


async def fetch_column_series(
    conn: asyncpg.Connection,
    *,
    column: str,
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
) -> list[tuple[datetime, float]]:
    """Ascending ``[(start_time, value), ...]`` for one feature_store column.

    Half-open ``[start, end)`` on ``start_time`` -- mirrors
    ``repo/market_data.fetch_daily_closes``. NULL values are dropped (a
    NULL bar is a coverage gap, not a zero). The value is cast to float;
    booleans become 0.0/1.0. The caller MUST have resolved ``column`` via
    :func:`list_numeric_columns` first; the regex below is a second line,
    not the gate.
    """
    if not _IDENT_RE.match(column):
        raise ValueError(f"feature_store column {column!r} is not identifier-shaped")
    rows = await conn.fetch(
        f"""
        SELECT start_time, "{column}" AS value
        FROM feature_store
        WHERE symbol = $1 AND interval = $2
          AND start_time >= $3 AND start_time < $4
          AND "{column}" IS NOT NULL
        ORDER BY start_time ASC
        """,
        symbol,
        interval,
        start,
        end,
    )
    return [(r["start_time"], float(r["value"])) for r in rows]
