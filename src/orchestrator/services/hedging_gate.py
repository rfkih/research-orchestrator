"""Hedging-track gate: beats-buy-hold on a risk-adjusted, equity-level basis.

Separate, additive objective for track=hedging. Does NOT touch V11/V60, which
stay frozen for track=trading. Operator-authorized 2026-06-07.
"""
from __future__ import annotations
from datetime import date, datetime, timedelta
from typing import Any
from uuid import UUID
import math
import random

import asyncpg

from ..repo import backtest_equity
from ..repo import market_data
from ..repo import trades as trades_repo
from . import portfolio, pool

# ── Pinned gate constants (operator-tunable; pin-tested below) ──────────
TOL_CAGR_PCT: float = 5.0       # hedge may give up at most this much CAGR vs buy-hold
THETA_SHARPE: float = 0.25      # a "material" Sharpe improvement
THETA_DD_PCT: float = 5.0       # a "material" maxDD reduction (percentage points)
BOOTSTRAP_REPS: int = 1000
CI_LEVEL: float = 0.95
RNG_SEED: int = 42
TRADING_DAYS: int = 252
MIN_STACK_OBS: int = 30      # min daily stack-return obs for the BTC-stack significance test


def _returns(values: list[float]) -> list[float]:
    out = []
    for a, b in zip(values, values[1:]):
        if a:
            out.append(b / a - 1.0)
    return out


def _max_drawdown_pct(equity: list[float]) -> float:
    peak = equity[0] if equity else 0.0
    mdd = 0.0
    for v in equity:
        peak = max(peak, v)
        if peak > 0:
            mdd = max(mdd, (peak - v) / peak)
    return mdd * 100.0


def _sharpe(rets: list[float]) -> float | None:
    if len(rets) < 2:
        return None
    mu = sum(rets) / len(rets)
    var = sum((r - mu) ** 2 for r in rets) / (len(rets) - 1)
    sd = math.sqrt(var)
    if sd == 0:
        return None
    return (mu / sd) * math.sqrt(TRADING_DAYS)


# The buy-hold benchmark is ALWAYS evaluated on DAILY bars: _sharpe annualises
# with sqrt(252) and CAGR divides by len/365, both of which assume one
# observation per day. Fetching the benchmark at a non-daily interval would make
# those annualisations silently wrong.
_BENCHMARK_INTERVAL = "1d"


def buy_hold_metrics(closes: list[tuple[date, float]]) -> dict[str, Any]:
    """CAGR%, annualised Sharpe, maxDD% of holding the underlying.

    Precondition: ``closes`` MUST be a DAILY series (one bar per calendar day).
    The Sharpe (sqrt(252)) and CAGR (len/365) annualisations assume daily bars;
    passing intraday closes yields wrong annualised metrics. Callers fetch the
    benchmark at ``_BENCHMARK_INTERVAL`` ("1d") regardless of the strategy's own
    interval."""
    px = [c for _, c in closes]
    if len(px) < 2:
        return {"cagr_pct": None, "sharpe": None, "max_drawdown_pct": None}
    rets = _returns(px)
    years = max(len(px) / 365.0, 1e-9)
    cagr_pct = ((px[-1] / px[0]) ** (1.0 / years) - 1.0) * 100.0
    return {
        "cagr_pct": cagr_pct,
        "sharpe": _sharpe(rets),
        "max_drawdown_pct": _max_drawdown_pct(px),
    }


def beats_buy_hold_risk_adj(*, strat: dict[str, Any], bench: dict[str, Any]) -> dict[str, Any]:
    """Deterministic: PASS iff CAGR floor holds AND a material risk improvement.
    floor:  strat.cagr >= bench.cagr - TOL_CAGR_PCT
    edge:   (strat.sharpe - bench.sharpe) >= THETA_SHARPE
            OR (bench.maxDD - strat.maxDD) >= THETA_DD_PCT
    """
    def _n(x): return None if x is None else float(x)
    s_cagr, b_cagr = _n(strat.get("cagr_pct")), _n(bench.get("cagr_pct"))
    s_sh, b_sh = _n(strat.get("sharpe")), _n(bench.get("sharpe"))
    s_dd, b_dd = _n(strat.get("max_drawdown_pct")), _n(bench.get("max_drawdown_pct"))
    if None in (s_cagr, b_cagr, s_sh, b_sh, s_dd, b_dd):
        return {"passed": False, "reason": "insufficient metrics for hedging gate"}
    floor_ok = s_cagr >= b_cagr - TOL_CAGR_PCT
    sharpe_gain = s_sh - b_sh
    dd_cut = b_dd - s_dd
    material = (sharpe_gain >= THETA_SHARPE) or (dd_cut >= THETA_DD_PCT)
    passed = bool(floor_ok and material)
    return {
        "passed": passed,
        "floor_ok": floor_ok,
        "sharpe_gain": round(sharpe_gain, 4),
        "dd_cut_pct": round(dd_cut, 4),
        "reason": (
            "beats buy-hold risk-adjusted" if passed
            else ("CAGR floor breached" if not floor_ok else "no material risk improvement")
        ),
    }


def improvement_significant(
    *, strat_returns: list[float], bench_returns: list[float]
) -> dict[str, Any]:
    """Stationary block bootstrap of the paired daily-return series.

    Replaces V11 (trade-level) for track=hedging. The strategy and benchmark
    daily returns are paired index-aligned over the common window; each
    bootstrap resample draws contiguous blocks (with wraparound) at matching
    positions in BOTH series, then computes ``sharpe(strat) - sharpe(bench)``.
    The improvement is "significant" iff the ``CI_LEVEL`` lower percentile of
    those paired differences is strictly > 0.

    Deterministic: uses ``random.Random(RNG_SEED)`` (never global ``random``)
    so the same inputs always yield the same CI. Block length is the standard
    n**(1/3) rule. Fails closed (not significant) on thin data (< 30 paired
    obs) with ``reason="insufficient_overlap"``.
    """
    n = min(len(strat_returns), len(bench_returns))
    if n < 30:
        return {
            "sharpe_improvement_significant": False,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "n_obs": n,
            "reason": "insufficient_overlap",
        }

    strat = list(strat_returns[:n])
    bench = list(bench_returns[:n])
    block = max(1, round(n ** (1.0 / 3.0)))
    rng = random.Random(RNG_SEED)

    # Bootstrap-local Sharpe: reuses _sharpe but resolves its degenerate
    # zero-variance case (which returns None) so the paired difference stays
    # defined. A constant return stream is the limiting best/worst risk-adjusted
    # case — positive mean → +inf Sharpe (capped), negative → -inf, flat → 0.
    # _sharpe itself is left untouched (decision-gate metrics rely on its
    # None-on-zero-variance contract).
    _CONST_SHARPE_CAP = 1e6

    def _bs_sharpe(rets: list[float]) -> float | None:
        s = _sharpe(rets)
        if s is not None:
            return s
        if len(rets) < 2:
            return None
        mu = sum(rets) / len(rets)
        if mu > 0:
            return _CONST_SHARPE_CAP
        if mu < 0:
            return -_CONST_SHARPE_CAP
        return 0.0

    diffs: list[float] = []
    for _ in range(BOOTSTRAP_REPS):
        s_sample: list[float] = []
        b_sample: list[float] = []
        while len(s_sample) < n:
            start = rng.randrange(n)
            for k in range(block):
                if len(s_sample) >= n:
                    break
                idx = (start + k) % n
                s_sample.append(strat[idx])
                b_sample.append(bench[idx])
        s_sh = _bs_sharpe(s_sample)
        b_sh = _bs_sharpe(b_sample)
        if s_sh is None or b_sh is None:
            continue
        diffs.append(s_sh - b_sh)

    if not diffs:
        return {
            "sharpe_improvement_significant": False,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "n_obs": n,
            "reason": "no finite bootstrap resamples",
        }

    diffs.sort()
    alpha = (1.0 - CI_LEVEL) / 2.0
    lo_idx = int(alpha * len(diffs))
    hi_idx = max(int((1.0 - alpha) * len(diffs)) - 1, 0)
    ci_low = diffs[lo_idx]
    ci_high = diffs[hi_idx]
    return {
        "sharpe_improvement_significant": bool(ci_low > 0.0),
        "ci_low": round(ci_low, 6),
        "ci_high": round(ci_high, 6),
        "n_obs": n,
    }


def _align_returns_by_date(
    *,
    strat_returns: list[float],
    strat_start: date,
    closes: list[tuple[date, float]],
) -> tuple[list[float], list[float]]:
    """Phase-align the strat and benchmark daily-return series on calendar date.

    The two series are built on DIFFERENT day-0 conventions, so a naive
    ``[:common]`` length-truncation pairs day *i* of strat with day *i+1* of
    bench (a 1-day phase offset):

      * ``strat_returns`` is a calendar grid over ``[start, end]`` inclusive,
        so ``strat_returns[i]`` is the return realised ON calendar date
        ``strat_start + i`` (``strat_returns[0]`` is the start date itself —
        a leading 0/level entry from ``pool._calendar_fill``).
      * ``bench`` comes from ``_returns(closes)``, which DROPS day-0: the k-th
        benchmark return ``closes[k+1]/closes[k] - 1`` is realised ON date
        ``closes[k+1]``.

    To pair the SAME calendar day's returns we key each series by the date its
    return is realised on, intersect the date sets, and emit values in
    sorted-date order. This is robust to gaps/holidays in either series, not
    just the uniform 1-day offset.
    """
    strat_by_date: dict[date, float] = {}
    d = strat_start
    for r in strat_returns:
        strat_by_date[d] = r
        d += timedelta(days=1)

    bench_by_date: dict[date, float] = {}
    for (_, prev_px), (cur_date, cur_px) in zip(closes, closes[1:]):
        if prev_px:
            # close[k] -> close[k+1] return is realised on close[k+1]'s date.
            cd = cur_date.date() if isinstance(cur_date, datetime) else cur_date
            bench_by_date[cd] = cur_px / prev_px - 1.0

    common_dates = sorted(set(strat_by_date) & set(bench_by_date))
    strat_aligned = [strat_by_date[cd] for cd in common_dates]
    bench_aligned = [bench_by_date[cd] for cd in common_dates]
    return strat_aligned, bench_aligned


def _equity_curve(returns: list[float]) -> list[float]:
    """Cumulative-compounded equity curve from a daily-return series, starting
    at 1.0. ``[r0, r1, ...]`` → ``[1.0, 1+r0, (1+r0)(1+r1), ...]``."""
    equity = [1.0]
    for r in returns:
        equity.append(equity[-1] * (1.0 + r))
    return equity


def _strat_metrics(daily_returns: list[float]) -> dict[str, Any]:
    """CAGR%, annualised Sharpe, maxDD% from a calendar-daily strat return
    series. Sharpe is computed on the daily returns; maxDD on the compounded
    equity curve; CAGR from first/last equity over the window length (in
    calendar days, 365-day year — matching ``buy_hold_metrics``)."""
    if len(daily_returns) < 2:
        return {"cagr_pct": None, "sharpe": None, "max_drawdown_pct": None}
    equity = _equity_curve(daily_returns)
    years = max(len(daily_returns) / 365.0, 1e-9)
    e0, e1 = equity[0], equity[-1]
    cagr_pct: float | None
    if e0 > 0 and e1 > 0:
        cagr_pct = ((e1 / e0) ** (1.0 / years) - 1.0) * 100.0
    else:
        cagr_pct = None
    return {
        "cagr_pct": cagr_pct,
        "sharpe": _sharpe(daily_returns),
        "max_drawdown_pct": _max_drawdown_pct(equity),
    }


def _strat_metrics_from_equity(equity: list[float], dates: list[date]) -> dict[str, Any]:
    """CAGR%, annualised Sharpe, maxDD% from a REAL per-day total_equity series.

    Unlike ``_strat_metrics`` (which compounds a return series off a normalised
    1.0 base), maxDD here is the TRUE peak-to-trough of the recorded
    ``total_equity`` and CAGR is first/last equity over the calendar span
    (first→last date, 365-day year — matching ``buy_hold_metrics``). Sharpe is
    the annualised mean/sd of the per-day pct-change returns."""
    if len(equity) < 2:
        return {"cagr_pct": None, "sharpe": None, "max_drawdown_pct": None}
    rets = _returns(equity)
    # Calendar span first→last date inclusive (≥1 day); fall back to obs count
    # if dates are degenerate. 365-day year, same convention as buy_hold_metrics.
    span_days = max((dates[-1] - dates[0]).days, 1)
    years = max(span_days / 365.0, 1e-9)
    e0, e1 = equity[0], equity[-1]
    cagr_pct: float | None
    if e0 > 0 and e1 > 0:
        cagr_pct = ((e1 / e0) ** (1.0 / years) - 1.0) * 100.0
    else:
        cagr_pct = None
    return {
        "cagr_pct": cagr_pct,
        "sharpe": _sharpe(rets),
        "max_drawdown_pct": _max_drawdown_pct(equity),
    }


def _strat_returns_by_date_from_equity(
    points: list[tuple[date, float]]
) -> dict[date, float]:
    """Pct-change strat returns keyed by the date the return is realised on.

    ``points`` is ascending ``[(equity_date, total_equity), ...]``. The return
    for the step ending on ``points[k]`` is realised on ``points[k]``'s date —
    the same day-0-dropping convention the benchmark uses in
    ``_align_returns_by_date``, so the two series pair on the SAME calendar
    day."""
    out: dict[date, float] = {}
    for (_, prev_eq), (cur_date, cur_eq) in zip(points, points[1:]):
        if prev_eq:
            cd = cur_date.date() if isinstance(cur_date, datetime) else cur_date
            out[cd] = cur_eq / prev_eq - 1.0
    return out


def _align_strat_dict_to_bench(
    *,
    strat_by_date: dict[date, float],
    closes: list[tuple[date, float]],
) -> tuple[list[float], list[float]]:
    """Intersect a date-keyed strat-return map with the benchmark daily returns,
    emitting paired values in sorted-date order. Sibling of
    ``_align_returns_by_date`` for the equity-point path (where strat returns
    are already keyed by realised date rather than a contiguous calendar grid)."""
    bench_by_date: dict[date, float] = {}
    for (_, prev_px), (cur_date, cur_px) in zip(closes, closes[1:]):
        if prev_px:
            cd = cur_date.date() if isinstance(cur_date, datetime) else cur_date
            bench_by_date[cd] = cur_px / prev_px - 1.0
    common_dates = sorted(set(strat_by_date) & set(bench_by_date))
    strat_aligned = [strat_by_date[cd] for cd in common_dates]
    bench_aligned = [bench_by_date[cd] for cd in common_dates]
    return strat_aligned, bench_aligned


# ── BTC-denominated ("stack sats") lens ─────────────────────────────────────
#
# ADDITIVE/informational only. Measures whether the strategy grows the user's
# BITCOIN stack (total_equity / price) vs simply HOLDING the same BTC (a flat
# 1.0 baseline). It does NOT alter the gate's `passed` value, which stays the
# USDT-denominated beats_buy_hold_risk_adj + improvement_significant decision.
#
# Methodology guards (deliberate, do NOT "fix"):
#   * NO "beats buy-hold in BTC" return criterion. The terminal stack ratio vs
#     hold equals the USDT outperformance (price cancels) → redundant. The lens
#     reports ABSOLUTE stack growth vs the flat 1.0 baseline only.
#   * stack_maxdd is the ABSOLUTE peak-to-trough of the stack curve, NOT a
#     comparison to buy-hold (whose stack DD is 0 by construction).
#   * Significance is ONE-SAMPLE (the stack grew > 0), NOT a paired-vs-bench test.


def btc_stack_series(
    equity_by_date: dict[date, float], close_by_date: dict[date, float]
) -> list[tuple[date, float]]:
    """Aligned, normalized-to-1.0 BTC-stack curve, ascending by date.

    The strategy's BTC stack on day t = ``total_equity(t) / price(t)`` (how much
    BTC the equity could hold). We intersect the equity and close series BY DATE,
    skip any day with a missing or ≤0 price, then normalize so the first stack
    value = 1.0. Returns [] if fewer than 2 aligned points survive."""
    common = sorted(set(equity_by_date) & set(close_by_date))
    raw: list[tuple[date, float]] = []
    for d in common:
        px = close_by_date[d]
        if px is None or px <= 0:
            continue
        raw.append((d, equity_by_date[d] / px))
    if len(raw) < 2:
        return []
    base = raw[0][1]
    if base <= 0:
        return []
    return [(d, s / base) for d, s in raw]


def btc_stack_metrics(stack_curve: list[tuple[date, float]]) -> dict[str, Any]:
    """Absolute growth metrics of the normalized BTC-stack curve.

    ``final_stack_x`` = terminal stack multiple (last value; first is 1.0 by
    normalization). ``stack_cagr_pct`` annualises last/first over the CALENDAR
    window (first→last date, 365-day year — matching the V60 fix). ``stack_maxdd_pct``
    is the ABSOLUTE peak-to-trough drawdown of the stack curve (NOT vs buy-hold)."""
    if len(stack_curve) < 2:
        return {
            "final_stack_x": None,
            "stack_cagr_pct": None,
            "stack_maxdd_pct": None,
            "n_obs": len(stack_curve),
        }
    dates = [d for d, _ in stack_curve]
    stacks = [s for _, s in stack_curve]
    first, last = stacks[0], stacks[-1]
    span_days = max((dates[-1] - dates[0]).days, 1)
    years = max(span_days / 365.0, 1e-9)
    cagr_pct: float | None
    if first > 0 and last > 0:
        cagr_pct = ((last / first) ** (1.0 / years) - 1.0) * 100.0
    else:
        cagr_pct = None
    return {
        "final_stack_x": last / first if first else None,
        "stack_cagr_pct": cagr_pct,
        "stack_maxdd_pct": _max_drawdown_pct(stacks),
        "n_obs": len(stack_curve),
    }


def stack_growth_significant(stack_returns: list[float]) -> dict[str, Any]:
    """ONE-SAMPLE seeded block bootstrap of ABSOLUTE BTC-stack growth.

    The stack daily-return series is heavily autocorrelated (long ~zero runs
    while fully in BTC, inverse-price clusters while in cash), so we use the SAME
    block-bootstrap style as ``improvement_significant``: block length
    ``max(1, round(n**(1/3)))``, ``random.Random(RNG_SEED)``. Each resample's
    terminal cumulative growth is ``∏(1+r) − 1``; we take the ``CI_LEVEL``
    percentile CI. ``stack_growth_significant`` is True iff ``ci_low > 0`` (the
    stack genuinely grew, not noise). Fails closed below ``MIN_STACK_OBS``."""
    n = len(stack_returns)
    if n < MIN_STACK_OBS:
        return {
            "stack_growth_significant": False,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "n_obs": n,
            "reason": "insufficient_obs",
        }

    series = list(stack_returns[:n])
    block = max(1, round(n ** (1.0 / 3.0)))
    rng = random.Random(RNG_SEED)

    growths: list[float] = []
    for _ in range(BOOTSTRAP_REPS):
        sample: list[float] = []
        while len(sample) < n:
            start = rng.randrange(n)
            for k in range(block):
                if len(sample) >= n:
                    break
                sample.append(series[(start + k) % n])
        cum = 1.0
        for r in sample:
            cum *= 1.0 + r
        growths.append(cum - 1.0)

    growths.sort()
    alpha = (1.0 - CI_LEVEL) / 2.0
    lo_idx = int(alpha * len(growths))
    hi_idx = max(int((1.0 - alpha) * len(growths)) - 1, 0)
    ci_low = growths[lo_idx]
    ci_high = growths[hi_idx]
    return {
        "stack_growth_significant": bool(ci_low > 0.0),
        "ci_low": round(ci_low, 6),
        "ci_high": round(ci_high, 6),
        "n_obs": n,
    }


async def evaluate(
    conn: asyncpg.Connection,
    *,
    backtest_run_id: UUID,
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
) -> dict[str, Any]:
    """Assemble the full hedging verdict for a backtest run.

    Equity-level beats-buy-hold gate for ``track=hedging`` (replaces the
    trade-level V11 + V60-economic gates for hedging ONLY). Composes:

    1. Buy-hold benchmark metrics from the underlying's daily closes.
    2. The strategy's calendar-daily equity-return series. PREFERRED source is
       the JVM's REAL per-day ``backtest_equity_point.total_equity`` curve
       (true mark-to-market — correct for buy-and-hold-style allocation
       strategies that hold a position across many days). Only when that series
       is absent (legacy runs, <2 rows) does it FALL BACK to reconstructing the
       curve from sparse ``backtest_trade`` exit-day P&L. The source used is
       reported in ``strat_equity_source`` ("equity_points" | "trades").
    3. The deterministic ``beats_buy_hold_risk_adj`` decision gate AND the
       bootstrap ``improvement_significant`` significance test.

    PASS iff the decision gate passes AND the Sharpe improvement is
    significant. Walk-forward ROBUST is still required downstream (unchanged).

    Additionally attaches an ADDITIVE/informational ``btc_lens`` (BTC-stack
    "stack sats" view: absolute stack growth, CAGR, absolute maxDD, and a
    one-sample significance test) that does NOT influence ``passed``. On the
    trade-fallback path or with <2 date-aligned stack points it is
    ``{"available": False, "reason": ...}``.

    Returns
    ``{passed, decision, significance, benchmark, strategy, strat_equity_source,
    btc_lens}``.
    """
    # 1. Buy-hold benchmark from the underlying's daily closes. The benchmark is
    #    ALWAYS daily — buy_hold_metrics annualises with sqrt(252)/365-day-year,
    #    which assume one bar per day — so force "1d" REGARDLESS of the strategy's
    #    own ``interval`` (the live caller passes "1d" today, but this keeps the
    #    interval-agnostic signature defensive).
    closes = await market_data.fetch_daily_closes(
        conn, symbol=symbol, interval=_BENCHMARK_INTERVAL, start=start, end=end
    )
    bench = buy_hold_metrics(closes)

    # 2. Strategy daily equity-return series. PREFER the JVM's real per-day
    #    total_equity curve (true mark-to-market — correct for allocation
    #    strategies that hold a position across many days, where reconstructing
    #    from sparse trade-exit P&L misses daily MTM and produces garbage). Only
    #    when that series is absent (legacy runs without equity points, <2 rows)
    #    do we fall back to the trade-reconstruction path so nothing regresses.
    # Close prices keyed by calendar date — reused for the buy-hold alignment
    # AND the BTC-stack lens below. Built once here from the closes already
    # fetched (no extra DB round-trip).
    close_by_date: dict[date, float] = {}
    for cd, px in closes:
        d = cd.date() if isinstance(cd, datetime) else cd
        close_by_date[d] = px

    btc_lens: dict[str, Any]
    points = await backtest_equity.fetch_equity_points(conn, backtest_run_id)
    if len(points) >= 2:
        strat_equity_source = "equity_points"
        eq_dates = [
            (d.date() if isinstance(d, datetime) else d) for d, _ in points
        ]
        eq_values = [v for _, v in points]
        strat = _strat_metrics_from_equity(eq_values, eq_dates)
        # 3. Benchmark daily returns PHASE-aligned to the strat series on
        #    calendar date. Strat returns are keyed by realised date (pct-change
        #    of total_equity); _align_strat_dict_to_bench intersects with the
        #    same-day-keyed benchmark returns. Both drop day-0, so the SAME
        #    calendar day's returns are paired.
        strat_by_date = _strat_returns_by_date_from_equity(points)
        strat_aligned, bench_aligned = _align_strat_dict_to_bench(
            strat_by_date=strat_by_date,
            closes=[(d, px) for d, px in closes],
        )

        # ── BTC-stack ("stack sats") lens — ADDITIVE/informational only. ──────
        # Stack(t) = total_equity(t) / price(t), aligned by date & normalized to
        # 1.0. We report ABSOLUTE growth vs the flat 1.0 hold baseline (final_x,
        # CAGR, absolute maxDD) + a ONE-SAMPLE significance test that the stack
        # grew > 0. This does NOT influence `passed` (the USDT gate above).
        equity_by_date: dict[date, float] = dict(zip(eq_dates, eq_values))
        stack_curve = btc_stack_series(equity_by_date, close_by_date)
        if len(stack_curve) >= 2:
            stack_metrics = btc_stack_metrics(stack_curve)
            stack_rets = _returns([s for _, s in stack_curve])
            stack_sig = stack_growth_significant(stack_rets)
            btc_lens = {
                "final_stack_x": stack_metrics["final_stack_x"],
                "stack_cagr_pct": stack_metrics["stack_cagr_pct"],
                "stack_maxdd_pct": stack_metrics["stack_maxdd_pct"],
                "stack_growth_significant": stack_sig["stack_growth_significant"],
                "ci_low": stack_sig["ci_low"],
                "ci_high": stack_sig["ci_high"],
                "n_obs": stack_metrics["n_obs"],
                "buy_hold_stack_is_flat": True,
            }
        else:
            btc_lens = {
                "available": False,
                "reason": "fewer than 2 date-aligned stack points",
            }
    else:
        strat_equity_source = "trades"
        # The trade-fallback path reconstructs a normalized return curve, not a
        # real per-day total_equity series, so there is no meaningful BTC stack
        # to denominate by price. The lens is unavailable here by design.
        btc_lens = {
            "available": False,
            "reason": "no equity points (trade-reconstruction path)",
        }
        trades = await trades_repo.fetch_trades(conn, backtest_run_id)
        sparse = portfolio.daily_returns_from_trades(trades)
        strat_returns = pool._calendar_fill(sparse, start.date(), end.date())
        strat = _strat_metrics(strat_returns)
        # Benchmark daily returns PHASE-aligned on calendar date. The two series
        # use different day-0 conventions (strat day-0 = the start date's level
        # entry; bench drops day-0), so a plain length-truncation would pair
        # strat day i with bench day i+1. _align_returns_by_date keys both by the
        # date each return is realised on and intersects, so the SAME calendar
        # day's returns are paired. See _align_returns_by_date for the offset
        # derivation.
        strat_aligned, bench_aligned = _align_returns_by_date(
            strat_returns=strat_returns,
            strat_start=start.date(),
            closes=[(d, px) for d, px in closes],
        )

    # 4-5. Decision gate + significance.
    decision = beats_buy_hold_risk_adj(strat=strat, bench=bench)
    sig = improvement_significant(
        strat_returns=strat_aligned, bench_returns=bench_aligned
    )

    passed = bool(decision.get("passed") and sig["sharpe_improvement_significant"])
    return {
        "passed": passed,
        "decision": decision,
        "significance": sig,
        "benchmark": bench,
        "strategy": strat,
        "strat_equity_source": strat_equity_source,
        "btc_lens": btc_lens,
    }
