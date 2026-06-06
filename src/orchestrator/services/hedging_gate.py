"""Hedging-track gate: beats-buy-hold on a risk-adjusted, equity-level basis.

Separate, additive objective for track=hedging. Does NOT touch V11/V60, which
stay frozen for track=trading. Operator-authorized 2026-06-07.
"""
from __future__ import annotations
from datetime import date, datetime
from typing import Any
from uuid import UUID
import math
import random

import asyncpg

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


def buy_hold_metrics(closes: list[tuple[date, float]]) -> dict[str, Any]:
    """CAGR%, annualised Sharpe, maxDD% of holding the underlying."""
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
    2. The strategy's calendar-daily equity-return series from its
       ``backtest_trade`` rows (binned by exit-day, calendar-filled).
    3. The deterministic ``beats_buy_hold_risk_adj`` decision gate AND the
       bootstrap ``improvement_significant`` significance test.

    PASS iff the decision gate passes AND the Sharpe improvement is
    significant. Walk-forward ROBUST is still required downstream (unchanged).

    Returns ``{passed, decision, significance, benchmark, strategy}``.
    """
    # 1. Buy-hold benchmark from the underlying's daily closes.
    closes = await market_data.fetch_daily_closes(
        conn, symbol=symbol, interval=interval, start=start, end=end
    )
    bench = buy_hold_metrics(closes)

    # 2. Strategy daily equity-return series from its trades.
    trades = await trades_repo.fetch_trades(conn, backtest_run_id)
    sparse = portfolio.daily_returns_from_trades(trades)
    strat_returns = pool._calendar_fill(sparse, start.date(), end.date())
    strat = _strat_metrics(strat_returns)

    # 3. Benchmark daily returns from the close prices, aligned to the strat
    #    series over the common window length before bootstrapping.
    bench_returns = _returns([c for _, c in closes])
    common = min(len(strat_returns), len(bench_returns))
    strat_aligned = strat_returns[:common]
    bench_aligned = bench_returns[:common]

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
    }
