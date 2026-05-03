"""``backtest_trade`` reads — only the columns the analyzer needs.

The full trade row has 30+ columns. Pulling only what feeds analyze.py
keeps memory bounded for runs with thousands of trades.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import asyncpg


async def fetch_trades(conn: asyncpg.Connection, backtest_run_id: UUID) -> list[dict[str, Any]]:
    rows = await conn.fetch(
        """
        SELECT backtest_trade_id, side, status, exit_reason,
               realized_pnl_amount, realized_r_multiple,
               max_favorable_excursion_r AS mfe_r,
               max_adverse_excursion_r AS mae_r,
               bars_held, entry_time, exit_time,
               entry_adx, entry_rsi, entry_close_location_value AS entry_clv,
               entry_relative_volume20 AS entry_rvol,
               entry_trend_regime
        FROM backtest_trade
        WHERE backtest_run_id = $1
        ORDER BY entry_time ASC
        """,
        backtest_run_id,
    )
    return [dict(r) for r in rows]
