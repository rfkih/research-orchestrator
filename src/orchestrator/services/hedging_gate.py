"""Hedging-track gate: beats-buy-hold on a risk-adjusted, equity-level basis.

Separate, additive objective for track=hedging. Does NOT touch V11/V60, which
stay frozen for track=trading. Operator-authorized 2026-06-07.
"""
from __future__ import annotations
from datetime import date
from typing import Any
import math
import random

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
