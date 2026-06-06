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
from datetime import date, timedelta
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


def _calendar_fill(series: dict[date, float], start: date, end: date) -> list[float]:
    """Daily list over [start, end] inclusive; a day with no return is 0.0.

    Turns the sparse exit-day series from ``daily_returns_from_trades`` into a
    true calendar-daily grid so the √252 Sharpe annualisation is valid and
    members with different trade frequencies are comparable. (The remaining
    approximation — exit-day bucketing vs true mark-to-market — is documented
    in ``portfolio.daily_returns_from_trades``.)"""
    out: list[float] = []
    d = start
    while d <= end:
        out.append(series.get(d, 0.0))
        d += timedelta(days=1)
    return out


def _common_window(series_list: list[dict[date, float]]) -> tuple[date, date] | None:
    """Overlapping [start, end] across all non-empty series (intersection of
    spans). None when they don't overlap — combining over a union and
    zero-filling the non-overlap would treat 'no data' as 'flat' and bias the
    combined Sharpe."""
    spans = [(min(s), max(s)) for s in series_list if s]
    if not spans:
        return None
    start = max(s for s, _ in spans)
    end = min(e for _, e in spans)
    return (start, end) if start <= end else None


def _combined_series(
    weights: dict[str, float],
    series_by_code: dict[str, dict[date, float]],
    window: tuple[date, date],
) -> dict[date, float]:
    """Portfolio daily-return series over the common ``window`` =
    Σ weight_c · return_c(day), calendar-daily. Within the overlap a member
    missing a day is a genuine flat day, so 0-fill is correct here."""
    start, end = window
    out: dict[date, float] = {}
    d = start
    while d <= end:
        tot = 0.0
        for code, w in weights.items():
            s = series_by_code.get(code)
            if s:
                tot += w * s.get(d, 0.0)
        out[d] = tot
        d += timedelta(days=1)
    return out


def annualised_sharpe(series: dict[date, float]) -> float:
    """Annualised Sharpe of a return series, computed on a CALENDAR-DAILY grid
    (missing days = 0) so the √252 annualisation is valid. 0.0 when fewer than
    ``_MIN_OBS`` actual return-days or zero variance."""
    if len(series) < _MIN_OBS:
        return 0.0
    days = sorted(series)
    vals = _calendar_fill(series, days[0], days[-1])
    n = len(vals)
    if n < 2:
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
    """Sharpe of the HRP pool before vs after adding the candidate, measured
    apples-to-apples on the dates the series actually OVERLAP.

    Returns {marginal, sharpe_before, sharpe_after, max_abs_corr, n_overlap,
    reason}. Empty pool → marginal = candidate's own standalone Sharpe.
    Insufficient overlap with the existing pool → marginal 0.0 (we do not
    manufacture diversification out of non-overlapping windows).
    """
    after_series = dict(members_returns)
    after_series[candidate_key] = candidate_returns

    # Diagnostics: candidate's max |corr| + max date-overlap vs any member.
    max_abs_corr: float | None = None
    n_overlap = 0
    if members_returns:
        cand_dates = set(candidate_returns)
        for s in members_returns.values():
            n_overlap = max(n_overlap, len(cand_dates & set(s)))
        _, corr = spearman_corr_matrix(after_series, min_overlap=_MIN_OBS)
        codes = sorted(after_series.keys())
        ci = codes.index(candidate_key)
        import numpy as np

        row = corr[ci]
        others = [abs(row[j]) for j in range(len(codes)) if j != ci and not np.isnan(row[j])]
        if others:
            max_abs_corr = float(max(others))

    def _result(marginal: float, sb: float, sa: float, reason: str) -> dict[str, Any]:
        return {
            "marginal": round(marginal, 6),
            "sharpe_before": round(sb, 6),
            "sharpe_after": round(sa, 6),
            "max_abs_corr": None if max_abs_corr is None else round(max_abs_corr, 4),
            "n_overlap": n_overlap,
            "reason": reason,
        }

    # Empty pool: the candidate stands alone on its own window.
    if not members_returns:
        sa = annualised_sharpe(candidate_returns)
        return _result(sa, 0.0, sa, "standalone")

    # Non-empty pool: require a real date-overlap so before/after are measured
    # on the SAME window (else the marginal is comparing different periods).
    window = _common_window(list(after_series.values()))
    if n_overlap < _MIN_OBS or window is None:
        return _result(0.0, 0.0, 0.0, "insufficient_overlap")

    w_before = _hrp_for(members_returns)
    sharpe_before = annualised_sharpe(_combined_series(w_before, members_returns, window))
    w_after = _hrp_for(after_series)
    sharpe_after = annualised_sharpe(_combined_series(w_after, after_series, window))
    return _result(sharpe_after - sharpe_before, sharpe_before, sharpe_after, "ok")


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
    """Return series for each active pool member, keyed by member key
    (strategy_code:symbol:interval).

    Pinned to the backtest the member was ADMITTED on (its
    signal_pool.iteration_id → research_iteration_log.backtest_run_id), NOT the
    latest backtest for the surface — so a later, unrelated sweep on the same
    (code,symbol,interval) can't silently swap a member's evidence and shift
    its weight."""
    members = await conn.fetch(
        "SELECT strategy_code, symbol, interval_name, iteration_id "
        "FROM signal_pool WHERE status = 'active' AND kind = 'signal_pool'"
    )
    out: dict[str, dict[date, float]] = {}
    for m in members:
        if exclude_iteration is not None and m["iteration_id"] == exclude_iteration:
            continue
        run_id = await conn.fetchval(
            "SELECT backtest_run_id FROM research_iteration_log WHERE iteration_id = $1",
            m["iteration_id"],
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


# ── Phase 1-B: book risk guard + eviction lifecycle ──────────────────


def apply_per_symbol_cap(
    weights: dict[str, float],
    symbol_by_key: dict[str, str],
    cap: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Cap aggregate book weight on any single symbol at ``cap``, water-filling
    the freed weight to under-cap members. Returns (weights, diagnostics).

    Infeasible compositions (n_symbols × cap < 1) can't honour the cap while
    summing to 1, so we return the input unchanged + ``capped=False`` rather
    than silently under-deploying the book.

    Algorithm: water-fill at the SYMBOL level with pinning — a symbol that would
    exceed its fair share is pinned at exactly ``cap`` and removed from the free
    pool (never re-inflates), then the remaining budget is split among the free
    symbols. Converges in ≤ n_symbols passes and guarantees every symbol ≤ cap.
    Each symbol's weight is then split among its members by their input share."""
    eps = 1e-9
    if not weights:
        return {}, {"capped": False, "reason": "empty"}
    symbols = {symbol_by_key[k] for k in weights}
    if len(symbols) * cap < 1.0 - eps:
        return dict(weights), {
            "capped": False, "reason": "infeasible_cap",
            "n_symbols": len(symbols), "cap": cap,
        }
    # Symbol-level base shares (input weights sum to ~1).
    sym_base: dict[str, float] = {}
    for k, val in weights.items():
        sym_base[symbol_by_key[k]] = sym_base.get(symbol_by_key[k], 0.0) + val

    pinned: set[str] = set()
    sym_w: dict[str, float] = {}
    for _ in range(len(symbols) + 1):
        free = [s for s in sym_base if s not in pinned]
        free_budget = max(0.0, 1.0 - cap * len(pinned))
        free_total = sum(sym_base[s] for s in free) or 1.0
        newly = [s for s in free if free_budget * sym_base[s] / free_total > cap + eps]
        if not newly:
            for s in sym_base:
                sym_w[s] = cap if s in pinned else free_budget * sym_base[s] / free_total
            break
        pinned.update(newly)

    # Split each symbol's weight among its members by within-symbol share.
    out: dict[str, float] = {}
    for s, w_s in sym_w.items():
        members_s = [k for k in weights if symbol_by_key[k] == s]
        tot = sum(weights[k] for k in members_s) or 1.0
        for k in members_s:
            out[k] = w_s * weights[k] / tot
    per_symbol = {s: round(sym_w.get(s, 0.0), 5) for s in symbols}
    return out, {"capped": bool(pinned), "cap": cap, "per_symbol": per_symbol}


def _greedy_evictions(
    series_by_key: dict[str, dict[date, float]], evict_theta: float
) -> list[dict[str, Any]]:
    """Greedily flag redundant members ONE AT A TIME: find the member whose
    marginal vs the rest is lowest and ≤ θ, evict it, then re-evaluate the
    survivors. Stops when none qualify or one member remains.

    One-at-a-time is essential: a mutually-redundant pair (A≈B) would BOTH show
    ~0 marginal in a single pass and both be evicted, wiping a signal the book
    should keep. After A is removed, B's marginal vs the rest jumps positive and
    B survives — exactly the intended behaviour."""
    remaining = dict(series_by_key)
    evicted: list[dict[str, Any]] = []
    while len(remaining) > 1:
        worst_key: str | None = None
        worst_marg: float | None = None
        for k, s in remaining.items():
            others = {kk: ss for kk, ss in remaining.items() if kk != k}
            m = marginal_sharpe_contribution(s, others)["marginal"]
            if m <= evict_theta and (worst_marg is None or m < worst_marg):
                worst_key, worst_marg = k, m
        if worst_key is None:
            break
        evicted.append({"key": worst_key, "marginal": worst_marg})
        del remaining[worst_key]
    return evicted


async def evaluate_members_for_eviction(
    conn: asyncpg.Connection, *, evict_theta: float = 0.0
) -> dict[str, Any]:
    """Decide which active members to evict (greedy, one-at-a-time). Returns
    {evict: [{pool_id, key, marginal}], verdicts: [...]} where ``evict`` is the
    ordered eviction list and ``verdicts`` is the per-member first-pass view.
    Pure of writes — the API layer applies + journals."""
    members = await conn.fetch(
        "SELECT pool_id, strategy_code, symbol, interval_name "
        "FROM signal_pool WHERE status = 'active' AND kind = 'signal_pool'"
    )
    series = await _active_members_returns(conn)
    pool_id_by_key = {
        f"{m['strategy_code']}:{m['symbol']}:{m['interval_name']}": m["pool_id"]
        for m in members
    }

    # First-pass per-member view (diagnostics).
    verdicts: list[dict[str, Any]] = []
    for key, pid in pool_id_by_key.items():
        cand = series.get(key)
        if cand is None:
            verdicts.append({"key": key, "marginal": None, "reason": "no_series"})
            continue
        others = {k: s for k, s in series.items() if k != key}
        c = marginal_sharpe_contribution(cand, others)
        verdicts.append({"key": key, "marginal": c["marginal"], "reason": c.get("reason")})

    evict = [
        {"pool_id": pool_id_by_key.get(e["key"]), "key": e["key"], "marginal": e["marginal"]}
        for e in _greedy_evictions(series, evict_theta)
    ]
    return {"evict": evict, "verdicts": verdicts}


# ── Phase 1-C: book performance + attribution ────────────────────────


async def book_performance(conn: asyncpg.Connection) -> dict[str, Any]:
    """Combined HRP-weighted book performance vs each member standalone, plus
    per-member attribution and the combined equity curve. The headline metric
    is ``combined_beats_best_member`` — the diversification payoff."""
    series = await _active_members_returns(conn)
    if not series:
        return {"n_members": 0, "members": [], "combined_sharpe": 0.0,
                "best_member_sharpe": 0.0, "combined_beats_best_member": False,
                "equity_curve": [], "window": None}
    w = _hrp_for(series)
    window = _common_window(list(series.values()))

    def _on_window(s: dict[date, float]) -> dict[date, float]:
        # Restrict a member to the common window so its standalone Sharpe is
        # measured on the SAME period as the combined book (apples-to-apples).
        if window is None:
            return s
        start, end = window
        return {d: r for d, r in s.items() if start <= d <= end}

    members_out = []
    best = 0.0
    for key, s in series.items():
        sa = annualised_sharpe(_on_window(s))
        best = max(best, sa)
        weight = round(w.get(key, 0.0), 5)
        mean_ret = sum(s.values()) / len(s) if s else 0.0
        members_out.append({
            "surface": key,
            "weight": weight,
            "standalone_sharpe": round(sa, 4),
            "mean_daily_return": round(mean_ret, 6),
            "contribution": round(weight * mean_ret, 6),
        })
    combined_sharpe = 0.0
    equity: list[dict[str, Any]] = []
    if window is not None:
        combined = _combined_series(w, series, window)
        combined_sharpe = annualised_sharpe(combined)
        equity_val = 1.0  # geometric compounding of daily returns
        for d in sorted(combined):
            equity_val *= 1.0 + combined[d]
            equity.append({"date": d.isoformat(), "cum_return": round(equity_val - 1.0, 6)})
    return {
        "n_members": len(series),
        "combined_sharpe": round(combined_sharpe, 4),
        "best_member_sharpe": round(best, 4),
        "combined_beats_best_member": combined_sharpe > best,
        "members": sorted(members_out, key=lambda r: r["surface"]),
        "equity_curve": equity,
        "window": None if window is None
        else {"start": window[0].isoformat(), "end": window[1].isoformat()},
    }
