"""Persist capacity-sweep summaries to ``capacity_sweep_result`` (V135).

Written by the research side (orchestrator runs the sweep); read by the
trading JVM's leaderboard CapacityFactor. One row per sweep run.
"""
from __future__ import annotations

import asyncpg


async def insert_capacity_sweep_result(
    conn: asyncpg.Connection,
    *,
    symbol: str,
    strategy_code: str,
    interval_name: str,
    judge_verdict: str,
    floor_capital_usd: float | None,
    max_viable_capital_usd: float | None,
    edge_at_floor_pct: float | None,
    edge_at_max_pct: float | None,
) -> str:
    """Insert one summary row; return its UUID as str.

    ``judge_verdict`` is NOT NULL on the table — Phase 2 writes a provisional
    verdict derived from the edge-decay classification + max-viable-vs-floor;
    the full ``quant-capacity-judge`` agent (Phase 3) may overwrite it later.
    """
    row_id = await conn.fetchval(
        """
        INSERT INTO capacity_sweep_result (
            id, symbol, strategy_code, interval_name, judge_verdict,
            floor_capital_usd, max_viable_capital_usd,
            edge_at_floor_pct, edge_at_max_pct, created_time
        ) VALUES (
            gen_random_uuid(), $1, $2, $3, $4, $5, $6, $7, $8, NOW()
        )
        RETURNING id
        """,
        symbol,
        strategy_code,
        interval_name,
        judge_verdict,
        floor_capital_usd,
        max_viable_capital_usd,
        edge_at_floor_pct,
        edge_at_max_pct,
    )
    return str(row_id)
