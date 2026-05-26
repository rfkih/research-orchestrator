"""``GET /strategy-ranking`` — profitability leaderboard.

Returns strategies ranked by annualised geometric return (at 90% sizing)
across all completed backtest runs whose window is at least ``min_days``
long.  Each row carries the params snapshot so the operator can see
exactly which parameter set produced the result.
"""
from __future__ import annotations

import json
from typing import Any

import asyncpg
from fastapi import APIRouter, Depends, Query

from ..repo import rankings as rankings_repo
from .deps import get_db_conn

router = APIRouter(tags=["rankings"])


def _parse_snapshot(raw: str | dict | None) -> dict | None:
    """effective_params_snapshot is stored as TEXT (not JSONB) — parse if needed."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


@router.get("/strategy-ranking")
async def strategy_ranking(
    min_days: int = Query(
        365,
        ge=30,
        le=3650,
        description="Minimum backtest window in days. Default 365 (1 year).",
    ),
    limit: int = Query(50, ge=1, le=200),
    conn: asyncpg.Connection = Depends(get_db_conn),
) -> dict[str, Any]:
    rows = await rankings_repo.fetch_rankings(conn, min_days=min_days, limit=limit)

    items = []
    for r in rows:
        # Prefer effective_params_snapshot (merged defaults+overrides, V104+).
        # Fall back to params_snapshot from research_iteration_log for pre-V104 rows.
        eff = _parse_snapshot(r.get("effective_params_snapshot"))
        strategy_code = r["strategy_code"]
        params = eff.get(strategy_code) if eff is not None else None
        if params is None:
            params = r.get("params_snapshot") or {}

        items.append({
            "rank": r["rank"],
            "strategyCode": strategy_code,
            "symbol": r["symbol"],
            "intervalName": r["interval_name"],
            "verdict": r["verdict"],
            "statisticalVerdict": r["statistical_verdict"],
            "windowDays": r["window_days"],
            "startTime": r["start_time"],
            "endTime": r["end_time"],
            "initialCapital": float(r["initial_capital"]) if r["initial_capital"] is not None else None,
            "annualizedReturnPct": float(r["ann_return_pct"]) if r["ann_return_pct"] is not None else None,
            "sharpeAnnualized": float(r["sharpe_ann"]) if r["sharpe_ann"] is not None else None,
            "maxDrawdownPct": float(r["max_drawdown_pct"]) if r["max_drawdown_pct"] is not None else None,
            "nTrades": r["n_trades"],
            "profitFactor": float(r["profit_factor"]) if r["profit_factor"] is not None else None,
            "dsr": float(r["dsr"]) if r["dsr"] is not None else None,
            "psr": float(r["psr"]) if r["psr"] is not None else None,
            "winRate": float(r["win_rate"]) if r["win_rate"] is not None else None,
            "backtestRunId": r["backtest_run_id"],
            "params": params,
        })

    return {
        "rankings": items,
        "total": len(items),
        "minWindowDays": min_days,
    }
