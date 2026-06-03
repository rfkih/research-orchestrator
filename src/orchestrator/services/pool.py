"""Signal Pool / House Book — admission + marginal-contribution math.

Phase 0 (2026-06-03). A candidate that is statistically real but failed the
standalone 10%/yr gate is admitted to the pool iff it ADDS uncorrelated return
— i.e. inserting it (at HRP weights) raises the pool's annualised Sharpe by
more than ``theta``. Reuses the existing optimizer + return-series plumbing;
adds no new rigor and changes none of the standalone gate.

Uniform treatment: the marginal test is computed against whatever is currently
``status='active'`` in ``signal_pool`` — no privileged/protected baseline.
"""
from __future__ import annotations

import math
from datetime import date
from typing import Any
from uuid import UUID

import asyncpg

from ..repo import trades as trades_repo
from .portfolio import daily_returns_from_trades
from .specialists.portfolio_math import (
    hrp_weights,
    realised_vol,
    spearman_corr_matrix,
)

# Trading days per year for Sharpe annualisation — matches the rest of the
# platform (BacktestMetricsService / analyze.py use 252).
_TRADING_DAYS = 252
# Minimum marginal Sharpe uplift to admit — a small positive margin so noise
# doesn't pad the book. Overridable per call.
DEFAULT_THETA = 0.02
# DSR significance bar — identical to analyze.DSR_SIGNIFICANCE_THRESHOLD.
DSR_THRESHOLD = 0.95
_MIN_OBS = 5


def _combined_series(
    weights: dict[str, float],
    series_by_code: dict[str, dict[date, float]],
) -> dict[date, float]:
    """Portfolio daily-return series = Σ weight_c · return_c(date), over the
    union of dates (a member missing a date contributes 0 that day)."""
    out: dict[date, float] = {}
    for code, w in weights.items():
        s = series_by_code.get(code)
        if not s:
            continue
        for d, r in s.items():
            out[d] = out.get(d, 0.0) + w * r
    return out


def annualised_sharpe(series: dict[date, float]) -> float:
    """Annualised Sharpe of a daily-return series. 0.0 when too few points
    or zero variance (can't distinguish signal from a flat line)."""
    vals = list(series.values())
    n = len(vals)
    if n < _MIN_OBS:
        return 0.0
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    sd = math.sqrt(var)
    if sd == 0:
        return 0.0
    return (mean / sd) * math.sqrt(_TRADING_DAYS)


def _hrp_for(series_by_code: dict[str, dict[date, float]]) -> dict[str, float]:
    """HRP weights over the given return series. Falls back to equal weight
    when there are <2 series (HRP needs a pair to cluster)."""
    codes = sorted(series_by_code.keys())
    if not codes:
        return {}
    if len(codes) == 1:
        return {codes[0]: 1.0}
    ordered, corr = spearman_corr_matrix(series_by_code, min_overlap=_MIN_OBS)
    vol = []
    for c in ordered:
        v = realised_vol(series_by_code[c], min_obs=_MIN_OBS)
        vol.append(v if (v is not None and v > 0) else 1e-9)
    import numpy as np

    return hrp_weights(ordered, corr, np.asarray(vol, dtype=float))


def marginal_sharpe_contribution(
    candidate_returns: dict[date, float],
    members_returns: dict[str, dict[date, float]],
    *,
    candidate_key: str = "__candidate__",
) -> dict[str, Any]:
    """Sharpe of the HRP pool before vs after adding the candidate.

    Returns {marginal, sharpe_before, sharpe_after, max_abs_corr, n_overlap}.
    Empty pool → marginal = candidate's own standalone Sharpe.
    """
    # Before: pool as-is.
    if members_returns:
        w_before = _hrp_for(members_returns)
        sharpe_before = annualised_sharpe(_combined_series(w_before, members_returns))
    else:
        sharpe_before = 0.0

    # After: pool + candidate.
    after_series = dict(members_returns)
    after_series[candidate_key] = candidate_returns
    w_after = _hrp_for(after_series)
    sharpe_after = annualised_sharpe(_combined_series(w_after, after_series))

    # Max |corr| of the candidate vs any member (transparency; the marginal
    # test already penalises redundancy).
    max_abs_corr: float | None = None
    n_overlap = 0
    if members_returns:
        _, corr = spearman_corr_matrix(after_series, min_overlap=_MIN_OBS)
        codes = sorted(after_series.keys())
        ci = codes.index(candidate_key)
        import numpy as np

        row = corr[ci]
        others = [abs(row[j]) for j in range(len(codes)) if j != ci and not np.isnan(row[j])]
        if others:
            max_abs_corr = float(max(others))
        cand_dates = set(candidate_returns)
        for c, s in members_returns.items():
            n_overlap = max(n_overlap, len(cand_dates & set(s)))

    return {
        "marginal": round(sharpe_after - sharpe_before, 6),
        "sharpe_before": round(sharpe_before, 6),
        "sharpe_after": round(sharpe_after, 6),
        "max_abs_corr": None if max_abs_corr is None else round(max_abs_corr, 4),
        "n_overlap": n_overlap,
    }


# ── DB-backed admission ──────────────────────────────────────────────


async def _candidate_context(
    conn: asyncpg.Connection, iteration_id: UUID
) -> dict[str, Any] | None:
    """Pull the candidate's identity, validity flags, and backtest_run_id."""
    row = await conn.fetchrow(
        """
        SELECT il.iteration_id, il.strategy_code, il.backtest_run_id,
               il.statistical_verdict,
               il.metrics_snapshot,
               br.asset          AS symbol,
               br.interval_name  AS interval_name,
               br.deflated_sharpe AS br_dsr
          FROM research_iteration_log il
          JOIN backtest_run br ON br.backtest_run_id = il.backtest_run_id
         WHERE il.iteration_id = $1
        """,
        iteration_id,
    )
    if row is None:
        return None
    metrics = row["metrics_snapshot"] or {}
    analysis = metrics.get("analysis") if isinstance(metrics, dict) else None
    dsr = None
    if isinstance(analysis, dict) and analysis.get("dsr") is not None:
        dsr = float(analysis["dsr"])
    elif row["br_dsr"] is not None:
        dsr = float(row["br_dsr"])
    return {
        "iteration_id": row["iteration_id"],
        "strategy_code": row["strategy_code"],
        "symbol": row["symbol"],
        "interval_name": row["interval_name"],
        "backtest_run_id": row["backtest_run_id"],
        "statistical_verdict": row["statistical_verdict"],
        "dsr": dsr,
        "metrics": metrics,
    }


async def _is_walk_forward_robust(conn: asyncpg.Connection, iteration_id: UUID) -> bool:
    val = await conn.fetchval(
        """
        SELECT 1 FROM walk_forward_run
         WHERE motivating_iteration_id = $1 AND stability_verdict = 'ROBUST'
         LIMIT 1
        """,
        iteration_id,
    )
    return val is not None


async def _returns_for_run(conn: asyncpg.Connection, backtest_run_id: UUID) -> dict[date, float]:
    trades = await trades_repo.fetch_trades(conn, backtest_run_id)
    ic = await conn.fetchval(
        "SELECT initial_capital FROM backtest_run WHERE backtest_run_id = $1",
        backtest_run_id,
    )
    return daily_returns_from_trades(trades, float(ic or 100.0))


async def _active_members_returns(
    conn: asyncpg.Connection, *, exclude_iteration: UUID | None = None
) -> dict[str, dict[date, float]]:
    """Latest COMPLETED backtest return series for each active pool member,
    keyed by a stable member key (strategy_code:symbol:interval)."""
    members = await conn.fetch(
        "SELECT strategy_code, symbol, interval_name, iteration_id "
        "FROM signal_pool WHERE status = 'active'"
    )
    out: dict[str, dict[date, float]] = {}
    for m in members:
        if exclude_iteration is not None and m["iteration_id"] == exclude_iteration:
            continue
        run_id = await conn.fetchval(
            """
            SELECT backtest_run_id FROM backtest_run
             WHERE strategy_code = $1 AND asset = $2 AND interval_name = $3
               AND status = 'COMPLETED'
             ORDER BY created_time DESC LIMIT 1
            """,
            m["strategy_code"], m["symbol"], m["interval_name"],
        )
        if run_id is None:
            continue
        series = await _returns_for_run(conn, run_id)
        if series:
            key = f"{m['strategy_code']}:{m['symbol']}:{m['interval_name']}"
            out[key] = series
    return out


async def evaluate_admission(
    conn: asyncpg.Connection, iteration_id: UUID, *, theta: float = DEFAULT_THETA
) -> dict[str, Any]:
    """Decide whether ``iteration_id`` should join the pool.

    Returns {admit, reason, ctx, contribution}. Does NOT write — the API layer
    inserts the signal_pool row on admit (so it can be idempotent + audited).
    """
    ctx = await _candidate_context(conn, iteration_id)
    if ctx is None:
        return {"admit": False, "reason": "iteration_not_found", "ctx": None}

    # Validity bar — UNCHANGED rigor (only the standalone-10% gate is dropped).
    if ctx["statistical_verdict"] != "SIGNIFICANT_EDGE":
        return {"admit": False, "reason": "not_significant_edge", "ctx": ctx}
    if ctx["dsr"] is None or ctx["dsr"] < DSR_THRESHOLD:
        return {"admit": False, "reason": "dsr_below_threshold", "ctx": ctx}
    if not await _is_walk_forward_robust(conn, iteration_id):
        return {"admit": False, "reason": "walk_forward_not_robust", "ctx": ctx}

    candidate_returns = await _returns_for_run(conn, ctx["backtest_run_id"])
    if len(candidate_returns) < _MIN_OBS:
        return {"admit": False, "reason": "insufficient_return_history", "ctx": ctx}

    members = await _active_members_returns(conn, exclude_iteration=iteration_id)
    contribution = marginal_sharpe_contribution(candidate_returns, members)

    admit = contribution["marginal"] > theta
    reason = "admitted" if admit else "marginal_below_theta"
    return {
        "admit": admit,
        "reason": reason,
        "theta": theta,
        "ctx": ctx,
        "contribution": contribution,
    }
