"""``/journal`` — agent's hypothesis / outcome / anti-pattern log.

Backed by ``research_journal`` (V10). The agent reads this on cold-boot to
recover prior lessons (anti-patterns, falsified hypotheses) so it doesn't
re-run a sweep that's already been ruled out.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

import asyncpg
from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field

from ..errors import NextAction, OrchestratorError
from ..repo import journal as journal_repo
from ..services.idempotency import cache_response, replay_cached_response
from .deps import get_agent_name, get_db_conn
from .pagination import Page, decode_cursor_ext, encode_cursor_ext

router = APIRouter(prefix="/journal", tags=["journal"])

_VALID_TYPE = {
    "STRATEGY_OUTCOME",
    "CROSS_STRATEGY_FINDING",
    "HYPOTHESIS",
    "IDEA_BACKLOG",
    "ANTI_PATTERN",
    "RUN_SUMMARY",
}
_VALID_STATUS = {"ACTIVE", "STALE", "FALSIFIED", "PARKED"}
_VALID_SORT_BY = {"created_time", "strategy_code", "status", "entry_type"}
_VALID_SORT_DIR = {"asc", "desc"}


class JournalCreateRequest(BaseModel):
    """Mirrors the V10 ``research_journal`` columns the agent is allowed to set.

    ``journal_id`` / ``created_time`` / ``updated_time`` / ``created_by`` /
    ``updated_by`` are server-assigned (gen_random_uuid + NOW + the
    X-Agent-Name header), so they're absent from the request body.
    """

    entry_type: str = Field(..., min_length=1, max_length=40)
    title: str = Field(..., min_length=1, max_length=300)
    content: str = Field(..., min_length=1)
    strategy_code: str | None = Field(None, max_length=60)
    interval_name: str | None = Field(None, max_length=20)
    instrument: str | None = Field(None, max_length=30)
    structured_data: dict[str, Any] = Field(default_factory=dict)
    status: str = Field("ACTIVE", min_length=1, max_length=20)
    iteration_id_refs: list[UUID] | None = None
    related_journal_ids: list[UUID] | None = None


@router.get("", response_model=Page)
async def list_journal(
    entry_type: str | None = Query(None),
    status: str | None = Query(None),
    strategy_code: str | None = Query(None),
    search: str | None = Query(
        None, description="Free-text search over title + content (English tsvector)."
    ),
    sort_by: str = Query("created_time", description=f"Sort field. One of {sorted(_VALID_SORT_BY)}."),
    sort_dir: str = Query("desc", description="Sort direction: asc or desc."),
    cursor: str | None = Query(None),
    limit: int = Query(25, ge=1, le=200),
    conn: asyncpg.Connection = Depends(get_db_conn),
) -> Page:
    if entry_type is not None and entry_type not in _VALID_TYPE:
        raise OrchestratorError(
            status_code=400,
            error_code="bad_entry_type",
            message=f"entry_type must be one of {sorted(_VALID_TYPE)}.",
            retryable=False,
        )
    if status is not None and status not in _VALID_STATUS:
        raise OrchestratorError(
            status_code=400,
            error_code="bad_journal_status",
            message=f"status must be one of {sorted(_VALID_STATUS)}.",
            retryable=False,
        )
    if sort_by not in _VALID_SORT_BY:
        raise OrchestratorError(
            status_code=400,
            error_code="bad_sort_by",
            message=f"sort_by must be one of {sorted(_VALID_SORT_BY)}.",
            retryable=False,
        )
    if sort_dir not in _VALID_SORT_DIR:
        raise OrchestratorError(
            status_code=400,
            error_code="bad_sort_dir",
            message="sort_dir must be 'asc' or 'desc'.",
            retryable=False,
        )

    after_value: Any = None
    after_created_time: datetime | None = None
    after_id: str | None = None
    if cursor is not None:
        decoded = decode_cursor_ext(cursor)
        # Cursor is authoritative — ignore query-string sort params when paginating.
        sort_by = decoded.get("sb", "created_time")
        sort_dir = decoded.get("sd", "desc")
        after_id = decoded.get("id")
        raw_v = decoded.get("v")
        if sort_by == "created_time":
            after_value = datetime.fromisoformat(raw_v) if raw_v else None
            after_created_time = after_value
        else:
            after_value = raw_v
            raw_t = decoded.get("t")
            after_created_time = datetime.fromisoformat(raw_t) if raw_t else None

    rows = await journal_repo.list_journal(
        conn,
        entry_type=entry_type,
        status=status,
        strategy_code=strategy_code,
        search=search,
        sort_by=sort_by,
        sort_dir=sort_dir,
        after_value=after_value,
        after_created_time=after_created_time,
        after_id=after_id,
        limit=limit + 1,
    )
    has_more = len(rows) > limit
    items = rows[:limit]
    next_cursor: str | None = None
    if has_more:
        last = items[-1]
        if sort_by == "created_time":
            payload: dict[str, Any] = {
                "sb": sort_by,
                "sd": sort_dir,
                "v": last["created_time"].isoformat(),
                "id": str(last["journal_id"]),
            }
        else:
            payload = {
                "sb": sort_by,
                "sd": sort_dir,
                "v": last.get(sort_by),
                "t": last["created_time"].isoformat(),
                "id": str(last["journal_id"]),
            }
        next_cursor = encode_cursor_ext(payload)

    next_actions: list[dict[str, Any]] = []
    if has_more:
        next_actions.append(
            {"kind": "call", "method": "GET", "path": f"/journal?cursor={next_cursor}"}
        )
    return Page(items=items, next_cursor=next_cursor, next_actions=next_actions)


@router.get("/{journal_id}")
async def get_journal(
    journal_id: UUID,
    conn: asyncpg.Connection = Depends(get_db_conn),
) -> dict[str, Any]:
    row = await journal_repo.get_journal(conn, journal_id)
    if row is None:
        raise OrchestratorError(
            status_code=404,
            error_code="journal_not_found",
            message=f"No research_journal row with journal_id={journal_id}.",
            retryable=False,
            next_action=NextAction(kind="call", method="GET", path="/journal"),
        )
    return row


@router.post("", status_code=201)
async def create_journal(
    body: JournalCreateRequest,
    request: Request,
    conn: asyncpg.Connection = Depends(get_db_conn),
    agent: str = Depends(get_agent_name),
) -> dict[str, Any]:
    if body.entry_type not in _VALID_TYPE:
        raise OrchestratorError(
            status_code=400,
            error_code="bad_entry_type",
            message=f"entry_type must be one of {sorted(_VALID_TYPE)}.",
            retryable=False,
        )
    if body.status not in _VALID_STATUS:
        raise OrchestratorError(
            status_code=400,
            error_code="bad_journal_status",
            message=f"status must be one of {sorted(_VALID_STATUS)}.",
            retryable=False,
        )

    # Idempotency-Key replay — same agent + key returns the cached response so
    # a retried pre-registration of a hypothesis can't double-write the row.
    cached, idempotency_key = await replay_cached_response(request, agent, "journal")
    if cached is not None:
        return cached

    row = await journal_repo.insert_journal(
        conn,
        entry_type=body.entry_type,
        title=body.title,
        content=body.content,
        strategy_code=body.strategy_code,
        interval_name=body.interval_name,
        instrument=body.instrument,
        structured_data=body.structured_data,
        status=body.status,
        iteration_id_refs=body.iteration_id_refs,
        related_journal_ids=body.related_journal_ids,
        created_by=agent,
    )
    response = {
        **row,
        "next_actions": [
            {"kind": "call", "method": "GET", "path": f"/journal/{row['journal_id']}"},
        ],
    }
    await cache_response(request, agent, "journal", idempotency_key, response)
    return response
