"""``research_queue`` reads.

Schema reference: ``V13__create_research_queue.sql``. Status enum:
PENDING, RUNNING, PARKED, COMPLETED, FAILED. The bash claim query uses
``ORDER BY priority ASC, created_time ASC`` + ``FOR UPDATE SKIP LOCKED``;
list endpoints here mirror the same ordering so the agent's view of "what's
next" matches what the claim loop will actually pick up (Phase 3).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

import asyncpg

# Full read shape — every column the agent / dashboard expect when reading
# a queue row. Keep list_queue and get_queue in lock-step by sourcing both
# from this constant; otherwise adding a column to V13 means remembering
# to touch two places, and the GET-by-id payload silently drifts from the
# list payload.
_QUEUE_READ_COLUMNS = (
    "queue_id, priority, strategy_code, interval_name, instrument, "
    "sweep_config, hypothesis, status, iteration_number, iter_budget, "
    "early_stop_on_no_edge, require_walk_forward, last_iteration_id, "
    "last_run_id, final_verdict, walk_forward_id, "
    "regime_label, regime_analysis, "
    "created_time, created_by, started_at, completed_at, notes"
)


async def list_queue(
    conn: asyncpg.Connection,
    *,
    status: str | None,
    strategy_code: str | None,
    final_verdict: str | None = None,
    after_created_time: datetime | None,
    after_id: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    where = ["1=1"]
    args: list[Any] = []
    if status is not None:
        args.append(status)
        where.append(f"status = ${len(args)}")
    if strategy_code is not None:
        args.append(strategy_code)
        where.append(f"strategy_code = ${len(args)}")
    if final_verdict is not None:
        # Server-side so callers (e.g. the walk-forward-candidate panel) don't
        # post-filter a truncated page and silently drop qualifying rows.
        args.append(final_verdict)
        where.append(f"final_verdict = ${len(args)}")
    if after_created_time is not None and after_id is not None:
        # Strict-greater on (created_time, queue_id) — matches the order-by
        # below so the next page is contiguous and stable under inserts.
        args.append(after_created_time)
        args.append(after_id)
        where.append(
            f"(created_time, queue_id::text) > (${len(args) - 1}, ${len(args)})"
        )
    args.append(limit)
    sql = f"""
        SELECT {_QUEUE_READ_COLUMNS}
        FROM research_queue
        WHERE {' AND '.join(where)}
        ORDER BY created_time ASC, queue_id::text ASC
        LIMIT ${len(args)}
    """
    rows = await conn.fetch(sql, *args)
    return [dict(r) for r in rows]


async def get_queue(conn: asyncpg.Connection, queue_id: UUID) -> dict[str, Any] | None:
    row = await conn.fetchrow(
        f"""
        SELECT {_QUEUE_READ_COLUMNS}
        FROM research_queue
        WHERE queue_id = $1
        """,
        queue_id,
    )
    return dict(row) if row else None
