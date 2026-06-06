"""``research_queue`` writes — enqueue, claim, status transitions.

The claim query mirrors ``research-tick.sh`` exactly: ``SELECT FOR UPDATE
SKIP LOCKED`` over PENDING/RUNNING rows ordered by ``(priority, created_time)``.
The atomic CTE pattern means concurrent ticks (cron drift, multiple
instances of /tick from different agents) cannot race on the same row.

A note on the bug fixed in the bash second-pass audit (2026-04-28):
``status='PENDING'`` is reset on continue branches. RUNNING is only the
in-flight state for a single tick — leaving it RUNNING permanently
deadlocks the row against future claims.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import asyncpg


async def find_account_strategy_id(
    conn: asyncpg.Connection,
    *,
    strategy_code: str,
    account_id: UUID | None,
) -> str | None:
    """Pre-queue existence check for an account_strategy row.

    Mirrors tick._resolve_account_strategy_id: when ``account_id`` is set
    (prod via ORCH_RESEARCH_ACCOUNT_ID), pin the lookup to the research
    account so admin rows can't satisfy the check. Returns the
    account_strategy_id, or None when no row exists — letting POST /queue
    reject the sweep up-front (412) instead of the first /tick failing
    mid-drain after a budget slot is spent.
    """
    if account_id is not None:
        return await conn.fetchval(
            """
            SELECT account_strategy_id
            FROM account_strategy
            WHERE strategy_code = $1 AND account_id = $2 AND is_deleted = false
            LIMIT 1
            """,
            strategy_code, account_id,
        )
    return await conn.fetchval(
        """
        SELECT account_strategy_id
        FROM account_strategy
        WHERE strategy_code = $1 AND is_deleted = false
        LIMIT 1
        """,
        strategy_code,
    )


async def insert_queue(
    conn: asyncpg.Connection,
    *,
    strategy_code: str,
    interval_name: str,
    instrument: str,
    sweep_config: dict[str, Any],
    hypothesis: str | None,
    priority: int,
    iter_budget: int,
    early_stop_on_no_edge: bool,
    require_walk_forward: bool,
    created_by: str,
    track: str | None = None,
) -> dict[str, Any]:
    # research_queue.queue_id is UUID NOT NULL with no DEFAULT (V13). Generate
    # in-SQL with gen_random_uuid() so we don't have to import uuid up here
    # and round-trip a Python UUID we'd otherwise immediately throw away.
    if track is not None:
        # Stamp the dual-track discriminator into the JSONB where the claim
        # query reads it. Build a NEW dict so the caller's sweep_config is
        # never mutated.
        sweep_config = {**sweep_config, "track": track}
    row = await conn.fetchrow(
        """
        INSERT INTO research_queue (
            queue_id,
            priority, strategy_code, interval_name, instrument,
            sweep_config, hypothesis, status, iteration_number,
            iter_budget, early_stop_on_no_edge, require_walk_forward,
            created_time, created_by
        ) VALUES (
            gen_random_uuid(),
            $1, $2, $3, $4, $5, $6, 'PENDING', 0, $7, $8, $9, NOW(), $10
        )
        RETURNING queue_id, priority, strategy_code, interval_name, instrument,
                  sweep_config, hypothesis, status, iteration_number, iter_budget,
                  early_stop_on_no_edge, require_walk_forward,
                  created_time, created_by
        """,
        priority,
        strategy_code,
        interval_name,
        instrument,
        sweep_config,
        hypothesis,
        iter_budget,
        early_stop_on_no_edge,
        require_walk_forward,
        created_by,
    )
    return dict(row)


async def claim_next(
    conn: asyncpg.Connection, *, track: str | None = None
) -> dict[str, Any] | None:
    """Atomically claim the next PENDING queue row.

    Returns ``started_at`` so callers can fence later writes against
    the reaper: if a slow tick comes back to write the iteration_log
    AFTER ``recover_stuck`` reset the row to PENDING and another tick
    re-claimed it (with a fresh ``started_at``), the original tick's
    fence will mismatch and ``attach_iteration`` will affect 0 rows.

    PENDING-only (was PENDING/RUNNING). The SKIP LOCKED row-lock only spans
    the claim transaction; once it commits, RUNNING rows are unlocked and a
    second tick would happily reclaim a row already in flight, double-firing
    the JVM. Crash recovery is handled by ``recover_stuck`` (resets RUNNING
    rows older than the poll cap back to PENDING) and ``rollback_iter``
    (called from the tick's exception handler).
    """
    row = await conn.fetchrow(
        """
        WITH claimed AS (
            SELECT queue_id, strategy_code, interval_name, instrument,
                   sweep_config, iteration_number AS prior_iter, iter_budget,
                   early_stop_on_no_edge, require_walk_forward
            FROM research_queue
            WHERE status = 'PENDING'
              AND ($1::text IS NULL OR sweep_config->>'track' = $1)
            ORDER BY priority ASC, created_time ASC
            LIMIT 1
            FOR UPDATE SKIP LOCKED
        )
        UPDATE research_queue rq
        SET status = 'RUNNING',
            iteration_number = rq.iteration_number + 1,
            started_at = NOW()
        FROM claimed
        WHERE rq.queue_id = claimed.queue_id
        RETURNING claimed.queue_id, claimed.strategy_code, claimed.interval_name,
                  claimed.instrument, claimed.sweep_config, claimed.prior_iter,
                  claimed.iter_budget, claimed.early_stop_on_no_edge,
                  claimed.require_walk_forward, rq.started_at
        """,
        track,
    )
    return dict(row) if row else None


async def recover_stuck(conn: asyncpg.Connection, max_age_seconds: int) -> int:
    """Reset RUNNING rows older than max_age_seconds back to PENDING.

    Covers the SIGKILL case: a tick whose process died before its rollback
    handler could run leaves a row in RUNNING permanently. We call this at
    the top of every tick; the cost is one indexed UPDATE.
    """
    result = await conn.execute(
        """
        UPDATE research_queue
        SET status = 'PENDING',
            iteration_number = GREATEST(iteration_number - 1, 0),
            notes = COALESCE(notes,'') || E'\n[' || NOW()::text ||
                    '] Recovered: RUNNING > ' || $1::text || 's; reset to PENDING.'
        WHERE status = 'RUNNING'
          AND started_at IS NOT NULL
          AND started_at < NOW() - make_interval(secs => $1)
        """,
        max_age_seconds,
    )
    return int(result.split()[-1])


async def rollback_iter(conn: asyncpg.Connection, queue_id: UUID, note: str) -> None:
    """Decrement iteration_number when a tick crashes mid-flight.

    Matches the bash ``cleanup`` trap behaviour. Status is reset to PENDING
    so the next tick can pick the row up cleanly.
    """
    await conn.execute(
        """
        UPDATE research_queue
        SET iteration_number = GREATEST(iteration_number - 1, 0),
            status = 'PENDING',
            notes = COALESCE(notes,'') || E'\n' || $2
        WHERE queue_id = $1
        """,
        queue_id,
        note,
    )


async def park_exhausted(conn: asyncpg.Connection, queue_id: UUID, note: str) -> None:
    await conn.execute(
        """
        UPDATE research_queue
        SET status = 'PARKED',
            completed_at = NOW(),
            notes = COALESCE(notes,'') || E'\n' || $2
        WHERE queue_id = $1
        """,
        queue_id,
        note,
    )


async def fail_queue(conn: asyncpg.Connection, queue_id: UUID, note: str) -> None:
    await conn.execute(
        """
        UPDATE research_queue
        SET status = 'FAILED',
            completed_at = NOW(),
            notes = COALESCE(notes,'') || E'\n' || $2
        WHERE queue_id = $1
        """,
        queue_id,
        note,
    )


async def complete_queue(
    conn: asyncpg.Connection,
    queue_id: UUID,
    *,
    final_verdict: str,
    walk_forward_id: UUID | None,
    note: str,
) -> None:
    await conn.execute(
        """
        UPDATE research_queue
        SET status = 'COMPLETED',
            final_verdict = $2,
            walk_forward_id = $3,
            completed_at = NOW(),
            notes = COALESCE(notes,'') || E'\n' || $4
        WHERE queue_id = $1
        """,
        queue_id,
        final_verdict,
        walk_forward_id,
        note,
    )


async def update_regime_label(
    conn: asyncpg.Connection,
    queue_id: UUID,
    regime_label: str | None,
    regime_analysis: dict | None,
) -> None:
    await conn.execute(
        """
        UPDATE research_queue
        SET regime_label    = $2,
            regime_analysis = $3
        WHERE queue_id = $1
        """,
        queue_id,
        regime_label,
        regime_analysis,
    )


async def reset_to_pending(conn: asyncpg.Connection, queue_id: UUID, note: str) -> None:
    await conn.execute(
        """
        UPDATE research_queue
        SET status = 'PENDING',
            notes = COALESCE(notes,'') || E'\n' || $2
        WHERE queue_id = $1
        """,
        queue_id,
        note,
    )


async def cancel_queue(conn: asyncpg.Connection, queue_id: UUID, note: str) -> int:
    """Mark a row PARKED with an operator-supplied cancel note. Returns
    affected row count so handlers can 404 cleanly."""
    result = await conn.execute(
        """
        UPDATE research_queue
        SET status = 'PARKED',
            completed_at = NOW(),
            notes = COALESCE(notes,'') || E'\n' || $2
        WHERE queue_id = $1
          AND status IN ('PENDING', 'RUNNING')
        """,
        queue_id,
        note,
    )
    # asyncpg returns "UPDATE n" — parse the count.
    return int(result.split()[-1])


async def attach_iteration(
    conn: asyncpg.Connection,
    queue_id: UUID,
    *,
    iteration_id: UUID,
    backtest_run_id: UUID,
    expected_started_at: Any | None = None,
) -> int:
    """Stamp the latest iteration / run pointers on the queue row.

    When ``expected_started_at`` is provided, the UPDATE is fenced with
    ``AND started_at = $4`` so a stale tick that came back after
    ``recover_stuck`` reset the row affects 0 rows — caller can detect
    by checking the returned count.

    Pairs with the iteration_log INSERT in the same transaction so the
    queue row never points at an iteration that doesn't exist yet.
    """
    if expected_started_at is None:
        result = await conn.execute(
            """
            UPDATE research_queue
            SET last_iteration_id = $2,
                last_run_id = $3
            WHERE queue_id = $1
            """,
            queue_id,
            iteration_id,
            backtest_run_id,
        )
    else:
        result = await conn.execute(
            """
            UPDATE research_queue
            SET last_iteration_id = $2,
                last_run_id = $3
            WHERE queue_id = $1
              AND started_at = $4
            """,
            queue_id,
            iteration_id,
            backtest_run_id,
            expected_started_at,
        )
    # asyncpg returns "UPDATE n" — parse the count.
    return int(result.split()[-1])
