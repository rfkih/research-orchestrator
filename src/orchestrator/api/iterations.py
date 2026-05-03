"""``/iterations`` and ``/leaderboard`` — research result reads."""

from __future__ import annotations

from typing import Any
from uuid import UUID

import asyncpg
from fastapi import APIRouter, Depends, Query

from ..errors import NextAction, OrchestratorError
from ..repo import iterations as it_repo
from .deps import get_db_conn
from .pagination import Page, decode_cursor, encode_cursor

router = APIRouter(tags=["iterations"])

_VALID_VERDICT = {"PASS", "ITERATE", "DISCARD", "FAILED"}


@router.get("/iterations", response_model=Page)
async def list_iterations(
    strategy_code: str | None = Query(None),
    verdict: str | None = Query(None, description="PASS|ITERATE|DISCARD|FAILED"),
    cursor: str | None = Query(None),
    limit: int = Query(25, ge=1, le=200),
    conn: asyncpg.Connection = Depends(get_db_conn),
) -> Page:
    if verdict is not None and verdict not in _VALID_VERDICT:
        raise OrchestratorError(
            status_code=400,
            error_code="bad_verdict_filter",
            message=f"verdict must be one of {sorted(_VALID_VERDICT)}.",
            retryable=False,
        )
    after_t, after_id = (None, None)
    if cursor is not None:
        after_t, after_id = decode_cursor(cursor)

    rows = await it_repo.list_iterations(
        conn,
        strategy_code=strategy_code,
        verdict=verdict,
        after_created_time=after_t,
        after_id=after_id,
        limit=limit + 1,
    )
    has_more = len(rows) > limit
    items = rows[:limit]
    next_cursor: str | None = None
    if has_more:
        last = items[-1]
        next_cursor = encode_cursor(last["created_time"], last["iteration_id"])

    next_actions: list[dict[str, Any]] = []
    if has_more:
        next_actions.append(
            {"kind": "call", "method": "GET", "path": f"/iterations?cursor={next_cursor}"}
        )
    return Page(items=items, next_cursor=next_cursor, next_actions=next_actions)


@router.get("/iterations/{iteration_id}")
async def get_iteration(
    iteration_id: UUID,
    conn: asyncpg.Connection = Depends(get_db_conn),
) -> dict[str, Any]:
    row = await it_repo.get_iteration(conn, iteration_id)
    if row is None:
        raise OrchestratorError(
            status_code=404,
            error_code="iteration_not_found",
            message=f"No research_iteration_log row with iteration_id={iteration_id}.",
            retryable=False,
            next_action=NextAction(kind="call", method="GET", path="/leaderboard"),
        )
    return row


@router.get("/leaderboard")
async def leaderboard(
    strategy_code: str | None = Query(None),
    verdict: str | None = Query(None),
    significant_only: bool = Query(
        False, description="If true, filter to statistical_verdict=SIGNIFICANT_EDGE."
    ),
    sort: str = Query("pf", description="pf|return_pct|sharpe|trade_count|created"),
    limit: int = Query(25, ge=1, le=200),
    conn: asyncpg.Connection = Depends(get_db_conn),
) -> dict[str, Any]:
    rows = await it_repo.leaderboard(
        conn,
        strategy_code=strategy_code,
        verdict=verdict,
        significant_only=significant_only,
        sort=sort,
        limit=limit,
    )
    next_actions: list[dict[str, Any]] = []
    if rows:
        top = rows[0]
        next_actions.append(
            {
                "kind": "call",
                "method": "GET",
                "path": f"/iterations/{top['iteration_id']}",
            }
        )
    return {"items": rows, "next_actions": next_actions}
