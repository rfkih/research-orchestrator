"""Specialist-review storage layer — backed by ``research_journal``.

Path C (2026-05-20) async-checkpoint contract. The qualitative skeptic /
portfolio / ml-judge passes can't run inside the researcher sub-agent
(harness strips ``Agent`` from nested contexts; see 2026-05-20 smoke
test journaled as ORCHESTRATOR_CHANGE). So the researcher writes a
"pending specialist review" row at step 9d, exits on the new terminal
``SPECIALIST_REVIEW_PENDING``, and the operator's main session — which
DOES have ``Agent`` — runs the ``/run-pending-specialists`` slash
command to spawn the specialists, parse verdicts, and post them back.

We piggyback on the same journal-row pattern the plan/graduation
reviewer system uses (``repo/reviews.py``):

  * Request: ``entry_type='IDEA_BACKLOG'`` + ``structured_data.kind=
    'specialist_review_request'``.
  * Verdict: ``entry_type='STRATEGY_OUTCOME'`` + ``structured_data.kind=
    'specialist_review_verdict'``.

This re-uses the V1-baseline CHECK constraint without a Flyway migration.

Target keying:
  * ``specialist:{specialist_name}:{iteration_id}`` —
    ``specialist_name`` ∈ {quant-skeptic, quant-portfolio-manager,
    quant-ml-judge}; ``iteration_id`` is the graduation candidate.

target_id is a string so the index used by reviews.py also works for us.
Tests pin the format so a future migration to a dedicated table can
copy verbatim.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import asyncpg


# Discriminator strings — pinned in tests. Changing these means stale
# rows from old code will be invisible to the new code; treat as a
# semi-versioned schema.
KIND_REQUEST = "specialist_review_request"
KIND_VERDICT = "specialist_review_verdict"

# Specialist identifiers the researcher and slash command both pin
# against. Adding a new specialist means adding the .md to
# .claude/agents/, updating this set, and teaching the slash command
# to spawn it.
SPECIALIST_NAMES = frozenset({
    "quant-skeptic",
    "quant-portfolio-manager",
    "quant-ml-judge",
})

# Verdict literals per specialist. Mismatched verdict → 400 from the
# /complete endpoint, so the slash command can fail fast.
VERDICT_LITERALS: dict[str, frozenset[str]] = {
    "quant-skeptic": frozenset({"CONCUR", "CONCERN", "OVERRIDE_REJECT"}),
    "quant-portfolio-manager": frozenset({"ADD", "CONCERN", "REJECT"}),
    "quant-ml-judge": frozenset({"CONCUR", "CONCERN", "OVERRIDE_REJECT"}),
}

# Vetos — verdicts that block graduation. The researcher's resume
# protocol branches on these.
VETO_VERDICTS: dict[str, frozenset[str]] = {
    "quant-skeptic": frozenset({"OVERRIDE_REJECT"}),
    "quant-portfolio-manager": frozenset({"REJECT"}),
    "quant-ml-judge": frozenset({"OVERRIDE_REJECT"}),
}


def specialist_target_id(specialist_name: str, iteration_id: str | UUID) -> str:
    """Stable identifier. One (specialist, iteration) → one target.

    Re-requesting a review on the same target appends a new request row
    but the gate reads ``fetch_latest_specialist_verdict_for_target``,
    so duplicates are harmless.
    """
    if specialist_name not in SPECIALIST_NAMES:
        raise ValueError(
            f"specialist_name={specialist_name!r} not in {sorted(SPECIALIST_NAMES)}"
        )
    return f"specialist:{specialist_name}:{iteration_id}"


async def insert_specialist_review_request(
    conn: asyncpg.Connection,
    *,
    specialist_name: str,
    iteration_id: str | UUID,
    strategy_code: str | None,
    motivating_hypothesis_id: str | None,
    payload: dict[str, Any],
    requested_by: str,
) -> tuple[str, str]:
    """Researcher submits a specialist-review request. Returns
    ``(journal_id, target_id)``.

    The ``payload`` carries everything the spawned specialist needs:
    iteration metrics, prescreen JSON, correlations / optimize envelopes
    (for portfolio), model_id + experiment_run_id (for ml-judge), etc.
    The slash command reads this directly and forwards it into the
    ``Agent`` spawn prompt — keeps the slash command stateless.
    """
    target_id = specialist_target_id(specialist_name, iteration_id)
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
        f"Specialist review request: {specialist_name} — iteration {iteration_id}",
        (
            f"Pending {specialist_name} audit on iteration_id={iteration_id}. "
            f"Submitted by {requested_by}. Run /run-pending-specialists to "
            f"spawn the audit."
        ),
        {
            "kind": KIND_REQUEST,
            "target_id": target_id,
            "specialist_name": specialist_name,
            "iteration_id": str(iteration_id),
            "motivating_hypothesis_id": (
                str(motivating_hypothesis_id) if motivating_hypothesis_id else None
            ),
            "request_payload": payload,
            "requested_by": requested_by,
        },
        requested_by,
    )
    return str(row["journal_id"]), target_id


async def insert_specialist_review_verdict(
    conn: asyncpg.Connection,
    *,
    target_id: str,
    specialist_name: str,
    iteration_id: str | UUID,
    strategy_code: str | None,
    verdict: str,
    reasoning: str,
    raw_response: dict[str, Any],
    spawned_by: str,
    motivating_request_id: str | None = None,
) -> str:
    """Slash command posts the specialist's verdict. Returns the
    journal_id of the verdict row.

    Verdict literal validity is enforced here against
    ``VERDICT_LITERALS`` so an upstream parse-bug can't silently insert
    a freeform string.

    Marks any matching open request row as 'PARKED' in the same
    transaction so the slash command's pending-queue stays clean.
    """
    allowed = VERDICT_LITERALS.get(specialist_name)
    if allowed is None:
        raise ValueError(
            f"specialist_name={specialist_name!r} not in {sorted(VERDICT_LITERALS)}"
        )
    if verdict not in allowed:
        raise ValueError(
            f"verdict={verdict!r} not in {sorted(allowed)} for "
            f"specialist={specialist_name!r}"
        )
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
            f"Specialist verdict: {specialist_name} {verdict} — iter {iteration_id}",
            reasoning[:4000] if reasoning else verdict,
            {
                "kind": KIND_VERDICT,
                "target_id": target_id,
                "specialist_name": specialist_name,
                "iteration_id": str(iteration_id),
                "verdict": verdict,
                "reasoning": reasoning,
                "raw_response": raw_response,
                "spawned_by": spawned_by,
                "motivating_request_id": motivating_request_id,
                "is_veto": verdict in VETO_VERDICTS.get(specialist_name, frozenset()),
            },
            spawned_by,
        )
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


async def fetch_pending_specialist_requests(
    conn: asyncpg.Connection, limit: int = 50
) -> list[dict[str, Any]]:
    """Slash command's work queue — open requests oldest-first.

    Oldest-first matches the reviews-repo pattern; FIFO minimises tail
    latency for the researcher waiting on its first request to clear.
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


async def fetch_latest_specialist_verdict_for_target(
    conn: asyncpg.Connection, target_id: str
) -> dict[str, Any] | None:
    """Latest verdict for a specialist target. Drives the researcher's
    resume protocol.

    Returns None if no verdict has been posted yet — researcher stays
    on the SPECIALIST_REVIEW_PENDING lockout for the next session.
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


async def fetch_verdicts_for_iteration(
    conn: asyncpg.Connection, iteration_id: str | UUID
) -> list[dict[str, Any]]:
    """All verdicts for one graduation candidate — researcher reads
    this on resume to decide whether all 3 (or 2) specialists have
    reported in.

    Newest-first; researcher takes the most recent verdict per
    specialist when there's been a re-request.
    """
    rows = await conn.fetch(
        """
        SELECT journal_id, strategy_code, title, content,
               structured_data, created_time, created_by
          FROM research_journal
         WHERE entry_type = 'STRATEGY_OUTCOME'
           AND status = 'ACTIVE'
           AND structured_data->>'kind' = $1
           AND structured_data->>'iteration_id' = $2
         ORDER BY created_time DESC
        """,
        KIND_VERDICT,
        str(iteration_id),
    )
    return [dict(r) for r in rows]


async def fetch_pending_for_iteration(
    conn: asyncpg.Connection, iteration_id: str | UUID
) -> list[dict[str, Any]]:
    """Open requests for one graduation candidate. Researcher reads
    this on resume to know which specialists still owe verdicts.
    """
    rows = await conn.fetch(
        """
        SELECT journal_id, strategy_code, title, content,
               structured_data, created_time, created_by
          FROM research_journal
         WHERE entry_type = 'IDEA_BACKLOG'
           AND status = 'ACTIVE'
           AND structured_data->>'kind' = $1
           AND structured_data->>'iteration_id' = $2
         ORDER BY created_time ASC
        """,
        KIND_REQUEST,
        str(iteration_id),
    )
    return [dict(r) for r in rows]
