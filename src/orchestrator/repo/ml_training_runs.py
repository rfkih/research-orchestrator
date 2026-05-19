"""``ml_training_runs`` DML.

Schema lives in Flyway V99. This is the orchestrator-side audit of
researcher-driven training subprocess spawns. Two distinct phases of
writes:

  1. ``insert_pending`` — at POST /ml/training-runs time, before we
     spawn anything. Returns the training_run_id so we can pass it
     into the subprocess as ``--triggered-by`` if blackheart-train ever
     learns to honor that flag (Phase B). For now it's just our handle.
  2. ``update_*`` — as the background task transitions the row:
     RUNNING (subprocess.pid known), then one of COMPLETED / FAILED /
     TIMED_OUT once the subprocess exits.

The reaper (``reap_orphaned``) runs once on lifespan startup. Any row
still in RUNNING from a prior process is ORPHANED — the subprocess may
still be running, but we lost the handle. Treat ORPHANED as a forensic
state, not a failure to retry.

Budget cap query (``count_active_in_window``) counts PENDING + RUNNING
+ COMPLETED. FAILED / TIMED_OUT / ORPHANED do not count — training
infra failures shouldn't punish the researcher's exploration budget.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

import asyncpg


VALID_STATUSES = frozenset(
    {"PENDING", "RUNNING", "COMPLETED", "FAILED", "TIMED_OUT", "ORPHANED"}
)
TERMINAL_STATUSES = frozenset({"COMPLETED", "FAILED", "TIMED_OUT", "ORPHANED"})
BUDGET_COUNTED_STATUSES = frozenset({"PENDING", "RUNNING", "COMPLETED"})


async def insert_pending(
    conn: asyncpg.Connection,
    *,
    agent_name: str,
    spec_name: str,
    hyperparams: dict[str, Any],
    cli_args: list[str],
    notes: str | None = None,
    motivating_hypothesis_id: UUID | None = None,
    created_by: str | None = None,
) -> UUID:
    """Insert a row in PENDING status. Returns training_run_id.

    ``hyperparams`` and ``cli_args`` are asyncpg-codec values — pass
    Python lists/dicts directly, do NOT json.dumps."""
    row = await conn.fetchrow(
        """
        INSERT INTO ml_training_runs
            (agent_name, spec_name, hyperparams, cli_args, status,
             notes, motivating_hypothesis_id, created_by, updated_by)
        VALUES ($1, $2, $3, $4, 'PENDING', $5, $6, $7, $7)
        RETURNING training_run_id
        """,
        agent_name, spec_name, hyperparams, cli_args,
        notes, motivating_hypothesis_id,
        created_by or agent_name,
    )
    return row["training_run_id"]


async def mark_running(
    conn: asyncpg.Connection,
    *,
    training_run_id: UUID,
    pid: int,
    updated_by: str,
) -> None:
    """Transition PENDING -> RUNNING once the subprocess is spawned and
    we have its pid. Idempotent at the SQL level (WHERE status='PENDING')
    so a duplicate background-task fire is a no-op rather than a crash."""
    await conn.execute(
        """
        UPDATE ml_training_runs
           SET status     = 'RUNNING',
               pid        = $2,
               started_at = NOW(),
               updated_time = NOW(),
               updated_by = $3
         WHERE training_run_id = $1
           AND status = 'PENDING'
        """,
        training_run_id, pid, updated_by,
    )


async def mark_terminal(
    conn: asyncpg.Connection,
    *,
    training_run_id: UUID,
    status: str,
    exit_code: int | None,
    duration_seconds: int | None,
    stdout_tail: str | None,
    stderr_tail: str | None,
    experiment_run_id: UUID | None,
    content_sha256: str | None,
    model_id: UUID | None,
    error_envelope: dict[str, Any] | None,
    updated_by: str,
) -> None:
    """Transition to a terminal status. Idempotent via WHERE status
    NOT IN (terminal set) — a duplicate fire after the first wins
    becomes a no-op rather than overwriting good data."""
    if status not in TERMINAL_STATUSES:
        raise ValueError(
            f"mark_terminal requires terminal status; got {status!r}. "
            f"Use mark_running for RUNNING transitions."
        )
    await conn.execute(
        """
        UPDATE ml_training_runs
           SET status            = $2,
               finished_at       = NOW(),
               exit_code         = $3,
               duration_seconds  = $4,
               stdout_tail       = $5,
               stderr_tail       = $6,
               experiment_run_id = $7,
               content_sha256    = $8,
               model_id          = $9,
               error_envelope    = $10,
               updated_time      = NOW(),
               updated_by        = $11
         WHERE training_run_id = $1
           AND status NOT IN ('COMPLETED', 'FAILED', 'TIMED_OUT', 'ORPHANED')
        """,
        training_run_id, status, exit_code, duration_seconds,
        stdout_tail, stderr_tail, experiment_run_id, content_sha256,
        model_id, error_envelope, updated_by,
    )


async def get_by_id(
    conn: asyncpg.Connection,
    *,
    training_run_id: UUID,
) -> dict[str, Any] | None:
    """Fetch a single row by id. Returns None when not found."""
    row = await conn.fetchrow(
        """
        SELECT training_run_id, agent_name, spec_name, hyperparams,
               cli_args, status, started_at, finished_at, exit_code,
               duration_seconds, pid, stdout_tail, stderr_tail,
               experiment_run_id, content_sha256, model_id,
               error_envelope, notes, motivating_hypothesis_id,
               created_time, created_by, updated_time, updated_by
          FROM ml_training_runs
         WHERE training_run_id = $1
        """,
        training_run_id,
    )
    return dict(row) if row else None


async def list_by_agent(
    conn: asyncpg.Connection,
    *,
    agent_name: str,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Recent training runs for one agent, newest first. Used by the
    researcher's resume protocol to pick up in-flight work after a
    session restart."""
    rows = await conn.fetch(
        """
        SELECT training_run_id, spec_name, status, started_at,
               finished_at, exit_code, model_id, content_sha256,
               error_envelope, notes, created_time
          FROM ml_training_runs
         WHERE agent_name = $1
         ORDER BY created_time DESC
         LIMIT $2
        """,
        agent_name, limit,
    )
    return [dict(r) for r in rows]


async def count_active_in_window(
    conn: asyncpg.Connection,
    *,
    agent_name: str,
    window_hours: int,
) -> int:
    """Count budget-affecting rows for one agent in the last
    ``window_hours``. Budget-counted = {PENDING, RUNNING, COMPLETED}.
    Used by the daily-cap enforcer."""
    row = await conn.fetchrow(
        """
        SELECT COUNT(*)::int AS n
          FROM ml_training_runs
         WHERE agent_name = $1
           AND status IN ('PENDING', 'RUNNING', 'COMPLETED')
           AND created_time > NOW() - make_interval(hours => $2)
        """,
        agent_name, window_hours,
    )
    return int(row["n"]) if row else 0


async def reap_orphaned(conn: asyncpg.Connection) -> int:
    """Lifespan-startup reaper. Any row still RUNNING from a prior
    process is now ORPHANED — the orchestrator restarted and we lost
    the subprocess handle. The subprocess MAY still be running; that's
    a forensic problem for the operator, not a correctness problem for
    the next training run. Returns the count of reaped rows."""
    rows = await conn.fetch(
        """
        UPDATE ml_training_runs
           SET status        = 'ORPHANED',
               finished_at   = NOW(),
               error_envelope = jsonb_build_object(
                   'error_code', 'orchestrator_restart',
                   'message', 'orchestrator process restarted while this '
                              'training run was RUNNING; subprocess handle lost'
               ),
               updated_time  = NOW(),
               updated_by    = 'reaper'
         WHERE status = 'RUNNING'
        RETURNING training_run_id
        """,
    )
    return len(rows)
