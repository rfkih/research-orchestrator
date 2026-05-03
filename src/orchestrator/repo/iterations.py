"""``research_iteration_log`` reads + leaderboard.

Iteration ranking matches ``leaderboard.sh``: pulls metrics out of the
JSONB ``metrics_snapshot`` and ``confidence_intervals`` columns, sorts by
the requested key. The agent uses this to decide whether to enqueue a
follow-up sweep or graduate a strategy.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

import asyncpg

from ..errors import OrchestratorError

_LEADERBOARD_SORTS = {
    "pf": "(metrics_snapshot->>'profit_factor')::numeric",
    "return_pct": "(metrics_snapshot->>'return_pct')::numeric",
    "sharpe": "(metrics_snapshot->>'sharpe_ratio')::numeric",
    "trade_count": "(metrics_snapshot->>'trade_count')::int",
    "created": "created_time",
}


async def list_iterations(
    conn: asyncpg.Connection,
    *,
    strategy_code: str | None,
    verdict: str | None,
    after_created_time: datetime | None,
    after_id: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    where = ["1=1"]
    args: list[Any] = []
    if strategy_code is not None:
        args.append(strategy_code)
        where.append(f"strategy_code = ${len(args)}")
    if verdict is not None:
        args.append(verdict)
        where.append(f"verdict = ${len(args)}")
    if after_created_time is not None and after_id is not None:
        # Iteration list orders newest-first, so cursor pagination uses
        # strict-less on the same composite.
        args.append(after_created_time)
        args.append(after_id)
        where.append(
            f"(created_time, iteration_id::text) < (${len(args) - 1}, ${len(args)})"
        )
    args.append(limit)
    sql = f"""
        SELECT iteration_id, strategy_code, iteration_number, backtest_run_id,
               params_snapshot, metrics_snapshot, verdict, statistical_verdict,
               confidence_intervals, sample_size_adequate,
               git_commit_hash, hypothesis_predicted, hypothesis_outcome,
               hypothesis_match, code_change_summary, change_reasoning,
               next_action, diagnostics, quant_audit_notes,
               created_time, created_by
        FROM research_iteration_log
        WHERE {' AND '.join(where)}
        ORDER BY created_time DESC, iteration_id::text DESC
        LIMIT ${len(args)}
    """
    rows = await conn.fetch(sql, *args)
    return [dict(r) for r in rows]


async def get_iteration(
    conn: asyncpg.Connection, iteration_id: UUID
) -> dict[str, Any] | None:
    row = await conn.fetchrow(
        """
        SELECT iteration_id, strategy_code, iteration_number, backtest_run_id,
               params_snapshot, metrics_snapshot, feature_buckets, notable_trades,
               verdict, statistical_verdict, confidence_intervals,
               effective_pf_after_slippage, sample_size_adequate,
               git_commit_hash, hypothesis_predicted, hypothesis_outcome,
               hypothesis_match, code_change_summary, change_reasoning,
               next_action, diagnostics, quant_audit_notes,
               created_time, created_by, updated_time, updated_by
        FROM research_iteration_log
        WHERE iteration_id = $1
        """,
        iteration_id,
    )
    return dict(row) if row else None


async def leaderboard(
    conn: asyncpg.Connection,
    *,
    strategy_code: str | None,
    verdict: str | None,
    significant_only: bool,
    sort: str,
    limit: int,
) -> list[dict[str, Any]]:
    if sort not in _LEADERBOARD_SORTS:
        raise OrchestratorError(
            status_code=400,
            error_code="bad_leaderboard_sort",
            message=f"sort must be one of {sorted(_LEADERBOARD_SORTS)}.",
            retryable=False,
            hint="Pass sort=pf for default profit-factor ranking.",
        )
    where = ["1=1"]
    args: list[Any] = []
    if strategy_code is not None:
        args.append(strategy_code)
        where.append(f"strategy_code = ${len(args)}")
    if verdict is not None:
        args.append(verdict)
        where.append(f"verdict = ${len(args)}")
    if significant_only:
        where.append("statistical_verdict = 'SIGNIFICANT_EDGE'")
    args.append(limit)
    sql = f"""
        SELECT strategy_code,
               iteration_id,
               iteration_number,
               verdict,
               statistical_verdict,
               (metrics_snapshot->>'profit_factor')::numeric           AS pf,
               (confidence_intervals->'pf_95'->>'low')::numeric        AS pf_lo,
               (confidence_intervals->'pf_95'->>'high')::numeric       AS pf_hi,
               (metrics_snapshot->>'return_pct')::numeric              AS return_pct,
               (metrics_snapshot->>'sharpe_ratio')::numeric            AS sharpe,
               (metrics_snapshot->>'trade_count')::int                 AS trade_count,
               sample_size_adequate,
               git_commit_hash,
               created_time
        FROM research_iteration_log
        WHERE {' AND '.join(where)}
        ORDER BY {_LEADERBOARD_SORTS[sort]} DESC NULLS LAST,
                 created_time DESC
        LIMIT ${len(args)}
    """
    rows = await conn.fetch(sql, *args)
    return [dict(r) for r in rows]
