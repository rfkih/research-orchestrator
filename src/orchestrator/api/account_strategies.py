"""``GET /account-strategies/research`` — runnable-surface discovery.

The tick lifecycle resolves an ``account_strategy`` row before it can run a
sweep (tick.py step 3). When no row exists for a (strategy_code) on the
research account, the sweep fails — historically discovered only mid-drain
(see the 2026-06-02 DCB-on-BNB/XRP block, where an entire archetype was
silently unavailable after the VPS migration dropped its rows).

This read-only endpoint projects ``account_strategy`` filtered to the
research account so the researcher can see, at cold-boot, exactly which
(strategy_code, symbol, interval) tuples are runnable — and treat a missing
one as a DATA_WISHLIST item instead of burning a tick to discover the 412.

SELECT-only against a table the orchestrator already reads for tick
resolution — no write-boundary violation.
"""

from __future__ import annotations

from typing import Any

import asyncpg
from fastapi import APIRouter, Depends, Request

from .deps import get_agent_name, get_db_conn

router = APIRouter(prefix="/account-strategies", tags=["account-strategies"])


@router.get("/research")
async def list_research_account_strategies(
    request: Request,
    conn: asyncpg.Connection = Depends(get_db_conn),
    _: str = Depends(get_agent_name),
) -> dict[str, Any]:
    """List runnable (strategy_code, symbol, interval) tuples for the
    research account. When ``research_account_id`` is unset (local dev),
    returns all non-deleted rows (legacy first-match scope)."""
    account_id = request.app.state.settings.research_account_id
    if account_id is not None:
        rows = await conn.fetch(
            """
            SELECT strategy_code, symbol, interval_name, enabled, simulated
            FROM account_strategy
            WHERE account_id = $1 AND is_deleted = false
            ORDER BY strategy_code, symbol, interval_name
            """,
            account_id,
        )
    else:
        rows = await conn.fetch(
            """
            SELECT strategy_code, symbol, interval_name, enabled, simulated
            FROM account_strategy
            WHERE is_deleted = false
            ORDER BY strategy_code, symbol, interval_name
            """,
        )
    strategies = [
        {
            "strategy_code": r["strategy_code"],
            "symbol": r["symbol"],
            "interval_name": r["interval_name"],
            "enabled": r["enabled"],
            "simulated": r["simulated"],
        }
        for r in rows
    ]
    return {
        "research_account_id": str(account_id) if account_id else None,
        "count": len(strategies),
        "strategies": strategies,
        "note": (
            "Runnable surfaces for the research account. A (strategy_code, "
            "symbol, interval) you want to sweep but don't see here has no "
            "account_strategy row — POST /queue will 412 account_strategy_missing. "
            "Journal it as a DATA_WISHLIST and ask the operator to seed it."
        ),
    }
