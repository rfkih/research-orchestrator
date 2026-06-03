"""Review storage layer — backed by ``research_journal``.

Reviews live as journal rows discriminated by ``structured_data.kind``:

  * Review request: entry_type='IDEA_BACKLOG', kind='review_request'.
    The "to-do" semantic of IDEA_BACKLOG matches a request that's
    awaiting reviewer pickup.

  * Review verdict: entry_type='STRATEGY_OUTCOME', kind='review_verdict'.
    The verdict IS an outcome (of running the checklist on the target).

Both fit the V1 baseline ``research_journal.entry_type`` CHECK
constraint without a Flyway migration. We pay for that with a
discriminator-in-JSONB query pattern; the `idx_research_journal_structured_data`
GIN index handles it.

Target keying:
  * Plan reviews → target_id = ``plan:{strategy_code}:{axis_set_hash}:{hypothesis_id}``
  * Graduation reviews → target_id = ``graduation:{iteration_id}``

target_id is a string so a single index lookup serves both. Tests pin
the format so a future migration to a real table can copy verbatim.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import asyncpg

from ..repo.hypothesis_audit import axis_set_hash

# Discriminator strings — pinned in tests. Changing these means stale
# rows from old code will be invisible to the new code; treat as a
# semi-versioned schema.
KIND_REQUEST = "review_request"
KIND_VERDICT = "review_verdict"


def plan_target_id(
    strategy_code: str, axis_names: list[str], hypothesis_id: str | UUID
) -> str:
    """Stable identifier for a plan review's target.

    Same strategy + same axis-set + same hypothesis = same target.
    Re-requesting a review on the same target appends a new request
    row but the gate looks at latest VERDICT for the target, so
    duplicates are harmless.
    """
    a_hash = axis_set_hash(axis_names)
    return f"plan:{strategy_code}:{a_hash}:{hypothesis_id}"


def graduation_target_id(iteration_id: str | UUID) -> str:
    return f"graduation:{iteration_id}"


async def insert_review_request(
    conn: asyncpg.Connection,
    *,
    target_id: str,
    target_kind: str,
    strategy_code: str | None,
    payload: dict[str, Any],
    requested_by: str,
) -> str:
    """Researcher submits a review request. Returns the journal_id."""
    row = await conn.fetchrow(
        """
        INSERT INTO research_journal
          (journal_id, entry_type, strategy_code, title, content,
           structured_data, status, created_by, created_time)
        VALUES (gen_random_uuid(), 'IDEA_BACKLOG', $1, $2, $3,
                $4, 'ACTIVE', $5, NOW())
        RETURNING journal_id
        """,
        strategy_code,
        f"Review request: {target_kind} — {target_id}",
        f"Pending review on target_id={target_id}. Submitted by {requested_by}.",
        {
            "kind": KIND_REQUEST,
            "target_id": target_id,
            "target_kind": target_kind,
            "request_payload": payload,
            "requested_by": requested_by,
        },
        requested_by,
    )
    return str(row["journal_id"])


async def insert_review_verdict(
    conn: asyncpg.Connection,
    *,
    target_id: str,
    target_kind: str,
    strategy_code: str | None,
    verdict: str,
    findings: list[dict[str, Any]],
    summary: dict[str, Any],
    reviewer: str,
    motivating_request_id: str | None = None,
) -> str:
    """Reviewer submits a verdict. Returns the journal_id.

    ``verdict`` ∈ {APPROVED, CONDITIONAL_APPROVAL, REJECTED}.
    ``findings`` is the ``checks`` list from ``aggregate_verdict``.
    ``summary`` is the rest of the aggregate_verdict payload (n_checks,
    n_blocker_fails, n_warning_fails, reason).

    Marks any matching open request row as 'PARKED' in the same
    transaction so reviewer queue stays clean.
    """
    async with conn.transaction():
        row = await conn.fetchrow(
            """
            INSERT INTO research_journal
              (journal_id, entry_type, strategy_code, title, content,
               structured_data, status, created_by, created_time)
            VALUES (gen_random_uuid(), 'STRATEGY_OUTCOME', $1, $2, $3,
                    $4, 'ACTIVE', $5, NOW())
            RETURNING journal_id
            """,
            strategy_code,
            f"Review verdict: {target_kind} {verdict} — {target_id}",
            f"{summary.get('reason', verdict)}",
            {
                "kind": KIND_VERDICT,
                "target_id": target_id,
                "target_kind": target_kind,
                "verdict": verdict,
                "findings": findings,
                "summary": summary,
                "reviewer": reviewer,
                "motivating_request_id": motivating_request_id,
            },
            reviewer,
        )
        # Close any matching OPEN requests for this target so the work
        # queue clears. Multiple requests on the same target are valid
        # (re-submissions) but only the latest needs to stay open.
        await conn.execute(
            """
            UPDATE research_journal
               SET status = 'PARKED'
             WHERE entry_type = 'IDEA_BACKLOG'
               AND status = 'ACTIVE'
               AND structured_data->>'kind' = $1
               AND structured_data->>'target_id' = $2
            """,
            KIND_REQUEST,
            target_id,
        )
    return str(row["journal_id"])


async def fetch_pending_requests(
    conn: asyncpg.Connection, limit: int = 50
) -> list[dict[str, Any]]:
    """Reviewer's work queue — open requests ordered oldest-first.

    Oldest-first because the researcher is blocked waiting; FIFO
    minimises the queue's tail latency.
    """
    rows = await conn.fetch(
        """
        SELECT journal_id, strategy_code, title, content,
               structured_data, created_time, created_by
          FROM research_journal
         WHERE entry_type = 'IDEA_BACKLOG'
           AND status = 'ACTIVE'
           AND structured_data->>'kind' = $1
         ORDER BY created_time ASC
         LIMIT $2
        """,
        KIND_REQUEST,
        limit,
    )
    return [dict(r) for r in rows]


async def fetch_latest_verdict(
    conn: asyncpg.Connection, target_id: str
) -> dict[str, Any] | None:
    """Latest verdict for a target. Drives the orchestrator gates.

    Returns None if no verdict has been posted yet — gate denies.
    """
    row = await conn.fetchrow(
        """
        SELECT journal_id, strategy_code, title, content,
               structured_data, created_time, created_by
          FROM research_journal
         WHERE entry_type = 'STRATEGY_OUTCOME'
           AND status = 'ACTIVE'
           AND structured_data->>'kind' = $1
           AND structured_data->>'target_id' = $2
         ORDER BY created_time DESC
         LIMIT 1
        """,
        KIND_VERDICT,
        target_id,
    )
    return dict(row) if row else None


async def fetch_latest_request_requester(
    conn: asyncpg.Connection, target_id: str
) -> str | None:
    """The agent that submitted the most recent review *request* for a target.

    Used to enforce reviewer-distinctness in ``POST /reviews`` — an agent must
    not post a verdict on a review it requested itself. Any status (a prior
    verdict PARKs the request) so the most recent requester is found. Returns
    None when no request exists for the target.
    """
    row = await conn.fetchrow(
        """
        SELECT structured_data->>'requested_by' AS requested_by
          FROM research_journal
         WHERE entry_type = 'IDEA_BACKLOG'
           AND structured_data->>'kind' = $1
           AND structured_data->>'target_id' = $2
         ORDER BY created_time DESC
         LIMIT 1
        """,
        KIND_REQUEST,
        target_id,
    )
    return row["requested_by"] if row else None


async def fetch_history_for_target(
    conn: asyncpg.Connection, target_id: str, limit: int = 20
) -> list[dict[str, Any]]:
    """All requests + verdicts for a target — full audit trail.

    Newest-first so the agent can read the latest exchange first.
    """
    rows = await conn.fetch(
        """
        SELECT journal_id, entry_type, strategy_code, title, content,
               structured_data, status, created_time, created_by
          FROM research_journal
         WHERE structured_data->>'target_id' = $1
           AND structured_data->>'kind' IN ($2, $3)
         ORDER BY created_time DESC
         LIMIT $4
        """,
        target_id,
        KIND_REQUEST,
        KIND_VERDICT,
        limit,
    )
    return [dict(r) for r in rows]
