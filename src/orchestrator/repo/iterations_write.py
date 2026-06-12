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


async def find_iteration_for_run(
    conn: asyncpg.Connection, backtest_run_id: UUID | str
) -> dict[str, Any] | None:
    """Existing iteration row for a backtest run, or None.

    The resume path checks this before inserting: a tick that died between
    committing its iteration row and updating the queue pointer leaves a
    fully-logged iteration behind. The resuming tick must ADOPT that row
    (attach + decide next state) rather than insert a duplicate — there is
    no UNIQUE constraint on backtest_run_id to catch a double insert.
    """
    row = await conn.fetchrow(
        """
        SELECT iteration_id, statistical_verdict, verdict,
               sample_size_adequate, metrics_snapshot
        FROM research_iteration_log
        WHERE backtest_run_id = $1
        ORDER BY created_time DESC
        LIMIT 1
        """,
        backtest_run_id,
    )
    return dict(row) if row else None


async def update_backtest_run_dsr(
    conn: asyncpg.Connection,
    *,
    backtest_run_id: UUID,
    deflated_sharpe: float | None,
    dsr_n_trials: int | None,
) -> None:
    """Persist the analyzer's DSR onto the JVM-owned ``backtest_run`` row.

    V134 added ``backtest_run.deflated_sharpe`` / ``dsr_n_trials`` for the
    backtest leaderboard to rank on, but the writer was never wired — the DSR
    lived only in ``research_iteration_log.metrics_snapshot``, leaving the
    column NULL for every row and the leaderboard's
    ``deflated_sharpe IS NOT NULL`` feed empty. This closes that gap on the
    live path; existing rows are handled by the one-time backfill script.

    ``backtest_run`` is owned by the trading JVM (schema-wise); the
    orchestrator is a DML-only client, so a scoped single-row UPDATE is
    in-contract. No-op when ``deflated_sharpe`` is None (e.g. n<30 or a
    degenerate distribution) so we never overwrite with nulls spuriously.
    """
    if deflated_sharpe is None:
        return
    await conn.execute(
        """
        UPDATE backtest_run
           SET deflated_sharpe = $2,
               dsr_n_trials    = $3
         WHERE backtest_run_id = $1
        """,
        backtest_run_id,
        deflated_sharpe,
        dsr_n_trials,
    )


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
