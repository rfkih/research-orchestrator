"""Hedging-track gate: beats-buy-hold on a risk-adjusted, equity-level basis.

Separate, additive objective for track=hedging. Does NOT touch V11/V60, which
stay frozen for track=trading. Operator-authorized 2026-06-07.
"""
from __future__ import annotations
from datetime import date
from typing import Any
import math

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
