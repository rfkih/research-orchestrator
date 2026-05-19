"""Researcher-facing audit-row writer for ``agent_decisions``.

The Session 1 design had this writing happen server-side inside an
Anthropic SDK wrapper. Path A flipped the LLM transport to Claude Code
sub-agents (Max plan, no API key), so the writer moves into a normal
HTTP endpoint — ``POST /agent-decisions/log`` — that the researcher
calls after a specialist sub-agent returns its verdict.

The audit row schema (V97) is transport-agnostic. A row may originate
from a sub-agent verdict, a deterministic HTTP service (e.g.
``/skeptic-prescreen``), or a future API-based specialist — the
``specialist`` + ``endpoint`` columns carry the source.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import asyncpg

from ...repo import agent_decisions as decisions_repo


async def log_agent_decision(
    conn: asyncpg.Connection,
    *,
    specialist: str,
    endpoint: str,
    model_name: str | None,
    request_payload: dict[str, Any],
    response_payload: dict[str, Any] | None,
    verdict: str | None,
    latency_ms: int | None,
    status: str,
    error_envelope: dict[str, Any] | None,
    motivating_iteration_id: UUID | None,
    target_id: str | None,
    agent_name: str,
) -> UUID:
    """Thin wrapper around ``insert_agent_decision``.

    Exists so the API handler keeps a clean signature and the test
    surface has a single import target. Token-count fields default to
    None — for sub-agent-spawned specialists we don't have per-call
    telemetry the way an SDK client would; that's an acceptable trade
    for staying on Max plan.
    """
    return await decisions_repo.insert_agent_decision(
        conn,
        specialist=specialist,
        endpoint=endpoint,
        model_name=model_name,
        request_payload=request_payload,
        response_payload=response_payload,
        verdict=verdict,
        latency_ms=latency_ms,
        input_tokens=None,
        output_tokens=None,
        cache_read_input_tokens=None,
        cache_creation_input_tokens=None,
        status=status,
        error_envelope=error_envelope,
        motivating_iteration_id=motivating_iteration_id,
        target_id=target_id,
        created_by=agent_name,
    )
