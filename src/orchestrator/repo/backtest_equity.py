"""Read the JVM's per-day backtest equity series for the hedging gate.

The trading JVM records one ``backtest_equity_point`` row per
(backtest_run_id, equity_date) with the true mark-to-market ``total_equity``.
The hedging gate uses this real daily curve to measure strategy metrics for
buy-and-hold-style allocation strategies, instead of reconstructing equity from
sparse trade-exit P&L (which misses daily mark-to-market and produces garbage
for positions held across many days).
"""
from __future__ import annotations

from datetime import date
from uuid import UUID

import asyncpg


async def fetch_equity_points(
    conn: asyncpg.Connection, backtest_run_id: UUID
) -> list[tuple[date, float]]:
    """Ascending ``[(equity_date, total_equity), ...]`` for one backtest run.

    Mirrors the regime_analysis / market_data read style. Params bound. Returns
    an empty list for runs with no recorded equity points (legacy runs) — the
    hedging gate falls back to trade reconstruction in that case."""
    rows = await conn.fetch(
        """
        SELECT equity_date, total_equity
        FROM backtest_equity_point
        WHERE backtest_run_id = $1
        ORDER BY equity_date ASC
        """,
        backtest_run_id,
    )
    return [(r["equity_date"], float(r["total_equity"])) for r in rows]
