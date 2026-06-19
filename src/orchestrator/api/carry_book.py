"""Levered carry-book approval lane (PR #2).

A NEW, ISOLATED orchestrator gate that certifies a multi-leg carry-book
backtest on BOOK-level metrics (Sharpe / maxDD / net CAGR / leg-count /
leverage / positive-every-year) instead of the single-asset trade-count / DSR
gate. Companion to PR #1 (trading-engine V188 ``PORTFOLIO_CARRY`` mode).

Two endpoints:

* ``POST /carry-book/gate`` — certify a COMPLETED ``PORTFOLIO_CARRY``
  ``backtest_run`` by id. Reads its equity curve + trade rows + carry columns,
  computes the book metrics, and returns the structured verdict.
* ``GET  /carry-book/preview`` — stateless what-if: given raw per-leg
  daily-return series + a leverage + a flat borrow model, build a synthetic
  book curve and return the SAME verdict shape without a stored run. Useful for
  sizing a book before committing a backtest.

Auth + deps mirror ``api/pool.py``: the shared ``X-Orch-Token`` is validated
globally by ``AuthMiddleware``; routers depend on ``get_agent_name`` /
``get_db_conn``. This lane is ADDITIVE — it never touches the frozen V11/V60
single-asset gate or analyze.py's constants.
"""
from __future__ import annotations

from typing import Any
from uuid import UUID

import asyncpg
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from ..services import carry_book_gate
from ..services.carry_book_gate import DEFAULT_LEVERAGE_CAP
from ..services.idempotency import cache_response, replay_cached_response
from .deps import get_agent_name, get_db_conn

router = APIRouter(prefix="/carry-book", tags=["carry-book"])


class GateBody(BaseModel):
    """Certify a stored PORTFOLIO_CARRY run. ``leverage_cap`` defaults to the
    module constant; ``require_basis_coverage`` hard-fails any basis-unmodeled
    leg when true."""

    backtest_run_id: UUID
    leverage_cap: float = Field(DEFAULT_LEVERAGE_CAP, gt=0.0, le=20.0)
    require_basis_coverage: bool = False


class PreviewLeg(BaseModel):
    """One leg's per-day return series for the stateless what-if."""

    symbol: str = Field(..., min_length=1, max_length=30)
    daily_returns: list[float] = Field(..., min_length=2)


class PreviewBody(BaseModel):
    """Stateless what-if input. ``legs`` are the per-leg daily-return series;
    ``leverage`` scales the equal-notional book; ``borrow_cost_bps_per_day`` is
    a flat per-day financing drag applied at the book level."""

    legs: list[PreviewLeg] = Field(..., min_length=1)
    leverage: float = Field(1.0, gt=0.0, le=20.0)
    borrow_cost_bps_per_day: float = Field(0.0, ge=0.0, le=500.0)
    leverage_cap: float = Field(DEFAULT_LEVERAGE_CAP, gt=0.0, le=20.0)
    require_basis_coverage: bool = False


@router.post("/gate")
async def gate(
    body: GateBody,
    request: Request,
    conn: asyncpg.Connection = Depends(get_db_conn),
    agent: str = Depends(get_agent_name),
) -> dict[str, Any]:
    """Book-level PASS/FAIL on a completed PORTFOLIO_CARRY backtest run.

    Idempotent per Idempotency-Key (the verdict is a pure function of the stored
    run + thresholds, so replay is safe)."""
    cached, idem_key = await replay_cached_response(request, agent, "carry_book_gate")
    if cached is not None:
        return cached

    verdict = await carry_book_gate.evaluate(
        conn,
        backtest_run_id=body.backtest_run_id,
        leverage_cap=body.leverage_cap,
        require_basis_coverage=body.require_basis_coverage,
    )
    await cache_response(request, agent, "carry_book_gate", idem_key, verdict)
    return verdict


@router.api_route("/preview", methods=["GET", "POST"])
async def preview(
    body: PreviewBody,
    _: Request,
    __: str = Depends(get_agent_name),
) -> dict[str, Any]:
    """Stateless what-if: same verdict shape as ``/gate`` from raw per-leg
    daily-return series + leverage + borrow model. No DB read, no stored run.

    Exposed on GET (per the PR #2 contract — a read-only preview) AND POST,
    because the structured per-leg return series is too large/nested for a
    query string and many HTTP clients drop a GET request body. Either verb
    carries the JSON ``PreviewBody``; the handler is byte-identical and
    READ-ONLY — it neither persists nor mutates anything."""
    leg_returns = {leg.symbol: list(leg.daily_returns) for leg in body.legs}
    return carry_book_gate.preview_from_leg_returns(
        leg_returns=leg_returns,
        leverage=body.leverage,
        borrow_cost_bps_per_day=body.borrow_cost_bps_per_day,
        leverage_cap=body.leverage_cap,
        require_basis_coverage=body.require_basis_coverage,
    )
