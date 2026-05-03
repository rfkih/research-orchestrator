"""``research_iteration_log`` writes.

The iteration_number column is globally monotonic per ``strategy_code`` —
shared across every queue arc on that code. The bash second-pass audit
caught a bug where queue.iteration_number was reused, colliding when a
second sweep arc was queued. Use ``next_iteration_number`` here to
preserve that property.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import asyncpg


async def next_iteration_number(conn: asyncpg.Connection, strategy_code: str) -> int:
    """Compute the next per-strategy_code iteration number.

    Must be called inside a transaction. The advisory lock serialises
    concurrent inserts on the same strategy_code so two ticks don't read
    the same MAX and collide on UNIQUE(strategy_code, iteration_number) —
    the constraint would catch it, but only after we've already burned a
    backtest run on the JVM.
    """
    await conn.execute(
        "SELECT pg_advisory_xact_lock(hashtext($1)::bigint)", strategy_code
    )
    val = await conn.fetchval(
        """
        SELECT COALESCE(MAX(iteration_number), 0) + 1
        FROM research_iteration_log
        WHERE strategy_code = $1
        """,
        strategy_code,
    )
    return int(val)


async def insert_iteration(
    conn: asyncpg.Connection,
    *,
    strategy_code: str,
    iteration_number: int,
    backtest_run_id: UUID,
    params_snapshot: dict[str, Any],
    metrics_snapshot: dict[str, Any],
    confidence_intervals: dict[str, Any] | None,
    verdict: str,
    statistical_verdict: str,
    sample_size_adequate: bool,
    git_commit_hash: str | None,
    code_change_summary: str,
    change_reasoning: str,
    quant_audit_notes: str,
    created_by: str,
) -> UUID:
    iteration_id = await conn.fetchval(
        """
        INSERT INTO research_iteration_log (
            iteration_id, strategy_code, iteration_number, backtest_run_id,
            params_snapshot, code_change_summary, change_reasoning,
            metrics_snapshot, verdict, statistical_verdict,
            confidence_intervals, sample_size_adequate,
            git_commit_hash, quant_audit_notes,
            created_time, created_by, updated_time, updated_by
        ) VALUES (
            gen_random_uuid(), $1, $2, $3,
            $4, $5, $6,
            $7, $8, $9,
            $10, $11,
            $12, $13,
            NOW(), $14, NOW(), $14
        )
        RETURNING iteration_id
        """,
        strategy_code,
        iteration_number,
        backtest_run_id,
        params_snapshot,
        code_change_summary,
        change_reasoning,
        metrics_snapshot,
        verdict,
        statistical_verdict,
        confidence_intervals,
        sample_size_adequate,
        git_commit_hash,
        quant_audit_notes,
        created_by,
    )
    return iteration_id
