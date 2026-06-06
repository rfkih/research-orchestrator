"""``POST /combination/evaluate`` — factor-neutral combination-book admission.

Phase 3 (spec §8b/§8c, Components 6+7). EVALUATION ONLY — this endpoint runs the
4-factor neutralization + the residual-fed admission math on a candidate and
returns the verdict. It does NOT auto-promote / auto-persist a member: the
operator/curator path is unchanged. (Pass ``persist:true`` to opt INTO writing a
``signal_combination`` member when — and only when — the candidate admits.)

Pipeline (mirrors the Phase 2a hedging-gate equity path for the candidate
series, then layers the factor model on top):

1. Resolve the candidate's ``backtest_run_id`` (+ ``strategy_code``) — directly
   from the body, or via ``iteration_id`` → ``research_iteration_log``.
2. Build the candidate's calendar-daily equity-return series from its
   ``backtest_trade`` rows (``portfolio.daily_returns_from_trades`` +
   ``pool._calendar_fill`` — the exact reuse Phase 2a's ``hedging_gate.evaluate``
   uses), keyed back to dates for the OLS.
3. ``factor_model.build_factors`` over the window → MARKET/MOMENTUM/CARRY/VOL.
4. ``factor_model.neutralize`` → idiosyncratic alpha, betas, residuals,
   alpha t-stat, DISGUISED_BETA.
5. If ``disguised_beta`` (or thin data) → return early ``admitted=False,
   disguised_beta=True`` (no IC / no admit — the raw edge is pure beta).
6. Else compute the IC and run ``combination_book.admit`` over the candidate's
   RESIDUAL series vs the existing combination members' residuals.

**IC signal source (documented pragmatic proxy).** An iteration does not expose
a separate per-bar signal series here, so — per the plan — the IC is computed
from the candidate's factor-neutral RESIDUAL returns vs the NEXT-day MARKET
factor return (forward market returns). i.e. "does this signal's idiosyncratic
edge rank-predict the next day's market move?" If the body supplies explicit
``signal``/``fwd_returns`` arrays, those are used verbatim instead (lets a caller
drive a real signal/forward series when it has one).
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any
from uuid import UUID

import asyncpg
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field, model_validator

from ..errors import OrchestratorError
from ..repo import trades as trades_repo
from ..services import combination_book, factor_model, portfolio
from ..services import pool as pool_svc
from ..services.idempotency import cache_response, replay_cached_response
from .deps import get_agent_name, get_db_conn

router = APIRouter(prefix="/combination", tags=["combination"])

_MARKET_FACTOR = "MARKET"


class EvaluateBody(BaseModel):
    # Resolve the candidate by either an iteration_id (→ run + strategy_code via
    # research_iteration_log) OR a direct backtest_run_id.
    iteration_id: UUID | None = None
    backtest_run_id: UUID | None = None
    strategy_code: str | None = Field(None, max_length=60)
    symbol: str = Field(..., min_length=1, max_length=30)
    interval: str = Field(..., min_length=1, max_length=10)
    start: datetime
    end: datetime
    expected_sign: str = Field("+", pattern=r"^[+\-0]$")
    track: str | None = None
    # Optional explicit IC inputs (override the residual-vs-forward-market proxy).
    signal: list[float] | None = None
    fwd_returns: list[float] | None = None
    # Evaluation-only by default. Set true to persist a member iff it admits.
    persist: bool = False

    @model_validator(mode="after")
    def _need_a_candidate_ref(self) -> "EvaluateBody":
        if self.iteration_id is None and self.backtest_run_id is None:
            raise ValueError("provide iteration_id or backtest_run_id")
        return self


def _neut_summary(neut: dict[str, Any]) -> dict[str, Any]:
    """The serialisable neutralization header (drops the bulky residual dict)."""
    return {
        "alpha": neut.get("alpha"),
        "alpha_tstat": neut.get("alpha_tstat"),
        "betas": neut.get("betas", {}),
        "disguised_beta": bool(neut.get("disguised_beta")),
        "n_obs": neut.get("n_obs"),
        "reason": neut.get("reason"),
    }


def _ic_from_residuals_vs_forward_market(
    residuals: dict[date, float], factors: dict[str, dict[date, float]]
) -> dict[str, Any]:
    """Pragmatic IC proxy: rank-correlate today's residual return against the
    NEXT day's MARKET factor return. Paired on consecutive residual dates that
    both have a following market observation. Returns the
    ``combination_book.information_coefficient`` verdict; if the MARKET factor is
    absent or the pairing is too thin, returns a not-significant sentinel."""
    market = factors.get(_MARKET_FACTOR)
    if not market or not residuals:
        return {"ic": 0.0, "significant": False, "sign": "0",
                "source": "residual_vs_fwd_market", "n_pairs": 0}
    res_dates = sorted(residuals)
    sig: list[float] = []
    fwd: list[float] = []
    for cur, nxt in zip(res_dates, res_dates[1:]):
        if nxt in market:  # forward market return realised on the next residual day
            sig.append(residuals[cur])
            fwd.append(market[nxt])
    ic = combination_book.information_coefficient(sig, fwd)
    ic["source"] = "residual_vs_fwd_market"
    ic["n_pairs"] = len(sig)
    return ic


async def _members_residuals(
    conn: asyncpg.Connection, *, track: str | None
) -> dict[str, dict[date, float]]:
    """Existing combination members' stored residual series, keyed by surface.

    Members store their residual snapshot under
    ``admission_metrics->'residuals'`` (date-string → float), written by
    ``add_member`` when ``persist=true``. Members lacking a stored residual
    series (e.g. an externally-inserted row) are skipped — the marginal-Sharpe
    math simply sees fewer comparators, never crashes. Empty book → ``{}`` (the
    first admit has no incumbents to be uncorrelated with)."""
    out: dict[str, dict[date, float]] = {}
    for m in await combination_book.list_members(conn, track=track):
        stored = (m.get("admission_metrics") or {}).get("residuals")
        if not isinstance(stored, dict):
            continue
        series: dict[date, float] = {}
        for k, v in stored.items():
            try:
                series[date.fromisoformat(k)] = float(v)
            except (ValueError, TypeError):
                continue
        if series:
            out[m["surface"]] = series
    return out


@router.post("/evaluate")
async def evaluate(
    body: EvaluateBody,
    request: Request,
    conn: asyncpg.Connection = Depends(get_db_conn),
    agent: str = Depends(get_agent_name),
) -> dict[str, Any]:
    cached, idem_key = await replay_cached_response(request, agent, "combination_evaluate")
    if cached is not None:
        return cached

    # 1. Resolve candidate run + strategy_code.
    backtest_run_id = body.backtest_run_id
    strategy_code = body.strategy_code
    if body.iteration_id is not None:
        # Resolve iteration → run + strategy_code with a minimal scoped query
        # (only the two columns we need; avoids pulling the analyzer's wide
        # column set that ``iterations.get_iteration`` selects).
        it = await conn.fetchrow(
            "SELECT backtest_run_id, strategy_code "
            "FROM research_iteration_log WHERE iteration_id = $1",
            body.iteration_id,
        )
        if it is None:
            raise OrchestratorError(
                status_code=404,
                error_code="iteration_not_found",
                message=f"No research_iteration_log row for {body.iteration_id}.",
                retryable=False,
                hint="Pass a valid iteration_id, or a direct backtest_run_id.",
            )
        backtest_run_id = backtest_run_id or it["backtest_run_id"]
        strategy_code = strategy_code or it["strategy_code"]
    if backtest_run_id is None:
        raise OrchestratorError(
            status_code=422,
            error_code="missing_backtest_run",
            message="iteration has no backtest_run_id and none was supplied.",
            retryable=False,
            hint="Supply backtest_run_id explicitly.",
        )

    # 2. Candidate calendar-daily equity returns (same path as hedging_gate).
    trades = await trades_repo.fetch_trades(conn, backtest_run_id)
    sparse = portfolio.daily_returns_from_trades(trades)
    filled = pool_svc._calendar_fill(sparse, body.start.date(), body.end.date())
    # Re-key the calendar-filled list back onto dates for the date-aligned OLS.
    candidate_returns: dict[date, float] = {}
    d = body.start.date()
    for r in filled:
        candidate_returns[d] = r
        d = date.fromordinal(d.toordinal() + 1)

    # 3. Factors over the window.
    factors = await factor_model.build_factors(conn, body.start, body.end)

    # 4. Neutralize.
    neut = factor_model.neutralize(candidate_returns, factors)
    neut_summary = _neut_summary(neut)
    base_response: dict[str, Any] = {
        "backtest_run_id": str(backtest_run_id),
        "strategy_code": strategy_code,
        "symbol": body.symbol,
        "interval": body.interval,
        "track": body.track,
        "expected_sign": body.expected_sign,
        "neutralization": neut_summary,
        "factors_used": sorted(factors),
    }

    # 5. Short-circuit: disguised beta (or thin data) — no IC / no admit.
    if neut.get("disguised_beta") or neut.get("reason") == "insufficient_obs":
        response = {
            **base_response,
            "admitted": False,
            "disguised_beta": bool(neut.get("disguised_beta")),
            "admission": {"admitted": False, "reasons": {}},
            "ic": None,
            "reason": neut.get("reason") or "disguised_beta",
        }
        await cache_response(request, agent, "combination_evaluate", idem_key, response)
        return response

    # 6. IC + residual-fed admission.
    residuals = neut.get("residuals", {})
    if body.signal is not None and body.fwd_returns is not None:
        ic = combination_book.information_coefficient(body.signal, body.fwd_returns)
        ic["source"] = "body"
        ic["n_pairs"] = min(len(body.signal), len(body.fwd_returns))
    else:
        ic = _ic_from_residuals_vs_forward_market(residuals, factors)

    members_residuals = await _members_residuals(conn, track=body.track)
    admission = combination_book.admit(
        candidate_residuals=residuals,
        members_residuals=members_residuals,
        ic=ic,
        alpha_tstat=neut.get("alpha_tstat"),
        expected_sign=body.expected_sign,
    )

    persisted: dict[str, Any] | None = None
    if body.persist and admission["admitted"]:
        if body.iteration_id is None:
            persisted = {"persisted": False, "reason": "persist_requires_iteration_id"}
        else:
            residual_metrics = {
                "alpha": neut.get("alpha"),
                "alpha_tstat": neut.get("alpha_tstat"),
                "betas": neut.get("betas", {}),
                "ic": ic.get("ic"),
                "ic_significant": ic.get("significant"),
                "marginal": admission["marginal"].get("marginal"),
                "max_abs_corr": admission["marginal"].get("max_abs_corr"),
                # Stored residual series (date-string keys) so later admits can
                # measure marginal-Sharpe vs this member.
                "residuals": {k.isoformat(): v for k, v in residuals.items()},
            }
            pool_id = await combination_book.add_member(
                conn,
                iteration_id=body.iteration_id,
                strategy_code=strategy_code or "UNKNOWN",
                symbol=body.symbol,
                interval_name=body.interval,
                track=body.track,
                residual_metrics=residual_metrics,
                created_by=agent,
            )
            persisted = {
                "persisted": pool_id is not None,
                "pool_id": str(pool_id) if pool_id is not None else None,
                "reason": "admitted" if pool_id is not None else "already_member",
            }

    response = {
        **base_response,
        "admitted": admission["admitted"],
        "disguised_beta": False,
        "ic": ic,
        "admission": admission,
    }
    if persisted is not None:
        response["persistence"] = persisted
    await cache_response(request, agent, "combination_evaluate", idem_key, response)
    return response
