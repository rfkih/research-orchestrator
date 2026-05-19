"""``agent_decisions`` DML.

Audit table schema lives in Flyway V97. Every specialist invocation
writes one row here regardless of outcome — success, parse error, API
error, timeout, refusal. The orchestrator is append-only; nothing
updates these rows after insert.

The two query patterns operators actually run live behind indexes
created in V97:

  * "last N decisions by specialist" — ``(specialist, created_time DESC)``
  * "all decisions for an iteration" — ``motivating_iteration_id``

Read helpers are intentionally absent for now: the auditing surface is
``GET /raw/agent_decisions/...`` (future) or psql directly. Writers
should never need to read this table inline; that's the contract.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import asyncpg


VALID_STATUSES = frozenset({"ok", "parse_error", "api_error", "timeout", "refused"})


async def insert_agent_decision(
    conn: asyncpg.Connection,
    *,
    specialist: str,
    endpoint: str,
    model_name: str | None,
    request_payload: dict[str, Any],
    response_payload: dict[str, Any] | None,
    verdict: str | None,
    latency_ms: int | None,
    input_tokens: int | None,
    output_tokens: int | None,
    cache_read_input_tokens: int | None,
    cache_creation_input_tokens: int | None,
    status: str,
    error_envelope: dict[str, Any] | None,
    motivating_iteration_id: UUID | None,
    target_id: str | None,
    created_by: str,
) -> UUID:
    """Append one audit row. Returns the decision_id.

    ``status`` is enforced both client-side (via VALID_STATUSES) and
    server-side (via the V97 CHECK constraint) — drift on one side is
    caught by the other.

    ``request_payload`` / ``response_payload`` / ``error_envelope`` are
    asyncpg-codec dicts; do NOT ``json.dumps`` first.
    """
    if status not in VALID_STATUSES:
        raise ValueError(
            f"agent_decisions.status={status!r} not in {sorted(VALID_STATUSES)}"
        )
    # V101 — mirror created_by onto updated_by since this table is
    # append-only; updated_time defaults to NOW() at the schema level.
    row = await conn.fetchrow(
        """
        INSERT INTO agent_decisions
            (specialist, endpoint, model_name, request_payload, response_payload,
             verdict, latency_ms, input_tokens, output_tokens,
             cache_read_input_tokens, cache_creation_input_tokens,
             status, error_envelope, motivating_iteration_id, target_id,
             created_by, updated_by)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,
                $15, $16, $16)
        RETURNING decision_id
        """,
        specialist, endpoint, model_name, request_payload, response_payload,
        verdict, latency_ms, input_tokens, output_tokens,
        cache_read_input_tokens, cache_creation_input_tokens,
        status, error_envelope, motivating_iteration_id, target_id,
        created_by,
    )
    return row["decision_id"]
