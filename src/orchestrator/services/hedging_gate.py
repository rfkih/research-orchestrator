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
