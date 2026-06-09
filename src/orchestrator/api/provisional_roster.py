"""``GET /provisional-roster`` — fill the live trading book to the floor.

The graduation gate is (deliberately) strict, so the live book can run short of
tradeable strategies. This endpoint keeps a MINIMUM of
``MIN_ACTIVE_TRADING_STRATEGIES`` by ranking the best near-miss TRADING
strategies (balanced composite: annualised return × overfit-quality) and
returning the top picks needed to fill the gap, each flagged ``provisional``.

The JVM is the source of truth for how many trading strategies are currently
active (it owns the live book), so it passes ``active_count`` and the
``exclude`` list of already-live strategy_codes; the orchestrator owns "which
near-miss is best to add". The JVM's auto-activation companion consumes the
``fills`` and live-promotes them at full size with a PROVISIONAL marker + a
V102 floor-exception (the fills are NOT full graduates).

Read-only. This NEVER mutates the live book itself — promotion happens JVM-side.
"""

from __future__ import annotations

from typing import Any

import asyncpg
from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field

from ..repo import provisional_roster as roster_repo
from ..services.provisional_roster import (
    MIN_ACTIVE_TRADING_STRATEGIES,
    roster_gap,
    select_roster_fill,
)
from .deps import get_agent_name, get_db_conn

router = APIRouter(tags=["provisional-roster"])


class ProvisionalRosterResponse(BaseModel):
    floor: int
    active_count: int
    gap: int
    fills: list[dict[str, Any]] = Field(default_factory=list)
    next_actions: list[dict[str, Any]] = Field(default_factory=list)


@router.get("/provisional-roster", response_model=ProvisionalRosterResponse)
async def get_provisional_roster(
    request: Request,
    active_count: int = Query(
        ..., ge=0, description="Currently-active TRADING strategies (JVM-supplied)."
    ),
    exclude: list[str] = Query(
        default_factory=list,
        description="strategy_codes already live, to exclude from fills.",
    ),
    floor: int = Query(MIN_ACTIVE_TRADING_STRATEGIES, ge=0, le=50),
    agent: str = Depends(get_agent_name),
    conn: asyncpg.Connection = Depends(get_db_conn),
) -> ProvisionalRosterResponse:
    gap = roster_gap(active_count, floor)
    if gap <= 0:
        # Roster already at/above floor — nothing to provision.
        return ProvisionalRosterResponse(
            floor=floor, active_count=active_count, gap=0, fills=[], next_actions=[]
        )

    candidates = await roster_repo.list_provisional_candidates(conn)
    fills = select_roster_fill(
        active_count,
        candidates,
        floor=floor,
        exclude_strategy_codes=frozenset(exclude),
    )
    next_actions: list[dict[str, Any]] = []
    if fills:
        next_actions.append({
            "kind": "note",
            "hint": (
                f"{len(fills)} of {gap} provisional slot(s) fillable. JVM: "
                "auto-activate these at full size with a PROVISIONAL marker + "
                "V102 floor-exception; auto-demote when a real qualifier "
                "graduates or the roster exceeds the floor."
            ),
        })
    elif gap > 0:
        next_actions.append({
            "kind": "note",
            "hint": (
                f"Roster {gap} below floor but NO eligible near-miss exists "
                "(no trading-track SIGNIFICANT_EDGE with positive return). Keep "
                "researching — the floor cannot be filled with noise."
            ),
        })
    return ProvisionalRosterResponse(
        floor=floor,
        active_count=active_count,
        gap=gap,
        fills=fills,
        next_actions=next_actions,
    )
