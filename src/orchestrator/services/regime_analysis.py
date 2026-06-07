"""Regime analysis — breaks down backtest trade performance by market regime.

Primary source: ``backtest_trade.entry_trend_regime`` (already computed by the
strategy engine per bar). Fallback: SMA-200 + ATR-percentile classification
from ``market_data`` when ``entry_trend_regime`` is sparse (< 50 % populated).

Returns a ``RegimeAnalysis`` dict that the caller can store on ``research_queue``
and surface to the researcher agent for the "don't-give-up" regime-retry decision.
"""

from __future__ import annotations

import logging
import math
import statistics
from datetime import timezone as dt_tz
from typing import Any
from uuid import UUID

import asyncpg

logger = logging.getLogger(__name__)

# Minimum trades in a regime bucket to consider it statistically meaningful.
_MIN_REGIME_TRADES = 10
# PF threshold above which a regime is "promising" (agent may re-queue).
PROMISING_PF_THRESHOLD = 1.15

# Sentinel used in breakdown dicts: no-loss regimes get PF=this value so
# _find_best still selects them correctly, while has_no_losses=True preserves
# the qualitative distinction (all-win regime vs very high PF regime).
_PF_NO_LOSS_SENTINEL = 99.0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def analyze(
    conn: asyncpg.Connection,
    backtest_run_id: UUID,
    instrument: str,
    interval_name: str,
) -> dict[str, Any]:
    """Compute per-regime metrics for the given backtest run.

    Returns:
        {
            "best_regime":    str | None,
            "best_pf":        float | None,
            "is_promising":   bool,
            "breakdown":      {regime: {n, pf, win_rate, mean_return_pct, no_losses}},
            "regime_source":  "entry_trend_regime" | "market_data_sma200_atr",
            "n_total_trades": int,
            "n_classified":   int,
        }
    """
    trades = await _fetch_trades(conn, backtest_run_id)
    if not trades:
        return _empty_result()

    classified, source = _classify_from_trades(trades)

    if classified is None:
        classified = await _classify_from_market_data(conn, trades, instrument, interval_name)
        source = "market_data_sma200_atr"

    if classified is None:
        return _empty_result()

    breakdown = _compute_breakdown(classified)
    best_regime, best_pf = _find_best(breakdown)

    return {
        "best_regime": best_regime,
        "best_pf": best_pf,
        "is_promising": best_pf is not None and best_pf >= PROMISING_PF_THRESHOLD,
        "breakdown": breakdown,
        "regime_source": source,
        "n_total_trades": len(trades),
        "n_classified": len(classified),
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

async def _fetch_trades(conn: asyncpg.Connection, backtest_run_id: UUID) -> list[dict]:
    rows = await conn.fetch(
        """
        SELECT entry_time,
               realized_pnl_amount,
               notional_size,
               entry_trend_regime
        FROM backtest_trade
        WHERE backtest_run_id = $1
        ORDER BY entry_time ASC
        """,
        backtest_run_id,
    )
    return [dict(r) for r in rows]


def _classify_from_trades(
    trades: list[dict],
) -> tuple[list[dict] | None, str]:
    """Use engine-stored ``entry_trend_regime`` if ≥ 50 % of trades have it."""
    populated = [t for t in trades if t.get("entry_trend_regime") is not None]
    if len(populated) < len(trades) * 0.5:
        return None, "entry_trend_regime"
    return [
        {
            "pnl": t["realized_pnl_amount"],
            "notional": t["notional_size"],
            "regime": str(t["entry_trend_regime"]),
        }
        for t in populated
    ], "entry_trend_regime"


async def _classify_from_market_data(
    conn: asyncpg.Connection,
    trades: list[dict],
    instrument: str,
    interval_name: str,
) -> list[dict] | None:
    """Derive regime per trade from SMA-200 (trend) + ATR-percentile (vol).

    Fixes applied vs v1:
    - O(n) rolling ATR: ATRs pre-computed once; percentile window is a slice.
    - Off-by-one fixed: percentile window now includes bar i (i+1 in slice end).
    - Timezone-safe: datetime subtraction wrapped via _tz_diff_s().
    """
    try:
        # NOTE: real market_data columns are start_time/close_price/high_price/
        # low_price (Flyway V1 baseline). They are aliased to the names the rest
        # of this function reads (open_time/close/hl_range). A schema mismatch
        # here was previously swallowed by the broad except below and surfaced as
        # a benign "market_data unavailable" log → regime classification silently
        # no-op'd in prod.
        rows = await conn.fetch(
            """
            SELECT start_time AS open_time,
                   close_price AS close,
                   (high_price - low_price) AS hl_range
            FROM market_data
            WHERE symbol   = $1
              AND interval  = $2
            ORDER BY start_time ASC
            """,
            instrument,
            interval_name,
        )
    except Exception:
        logger.warning("regime_analysis: market_data unavailable for %s %s", instrument, interval_name)
        return None

    if len(rows) < 210:
        return None

    closes = [float(r["close"]) for r in rows]
    hl_ranges = [float(r["hl_range"]) for r in rows]
    times = [r["open_time"] for r in rows]

    atr_window = 14
    pct_window = 100

    # Pre-compute rolling ATR once — O(n) instead of O(n × pct_window × atr_window).
    atrs: list[float] = []
    for i in range(len(hl_ranges)):
        start = max(0, i - atr_window)
        atrs.append(statistics.mean(hl_ranges[start:i]) if i > 0 else hl_ranges[0])

    # Compute SMA-200 and ATR percentile per bar.
    bar_regimes: dict = {}
    for i in range(200, len(rows)):
        sma200 = statistics.mean(closes[i - 200:i])
        atr = atrs[i]

        # ATR percentile window: [i-pct_window, i] inclusive.
        # Fix: use i+1 as slice end so bar i IS included in the window.
        pct_start = max(atr_window, i - pct_window)
        atr_series = atrs[pct_start:i + 1]
        atr_pct = sum(1 for a in atr_series if a <= atr) / len(atr_series) if atr_series else 0.5

        trend = "BULL" if closes[i] > sma200 else "BEAR"
        vol = "HIGH_VOL" if atr_pct >= 0.5 else "LOW_VOL"
        bar_regimes[times[i]] = f"{trend}_{vol}"

    # Match each trade's entry_time to the closest bar.
    result = []
    for trade in trades:
        et = trade["entry_time"]
        regime = bar_regimes.get(et)
        if regime is None:
            # Nearest bar within 2 hours — timezone-safe comparison.
            candidates = [t for t in bar_regimes if _tz_diff_s(t, et) < 7200]
            nearest = min(candidates, key=lambda t: _tz_diff_s(t, et), default=None)
            regime = bar_regimes.get(nearest) if nearest else "UNKNOWN"
        result.append({
            "pnl": trade["realized_pnl_amount"],
            "notional": trade["notional_size"],
            "regime": regime,
        })

    return result if result else None


def _tz_diff_s(t1: Any, t2: Any) -> float:
    """Absolute second difference between two datetimes, handling naive/aware mismatch."""
    try:
        return abs((t1 - t2).total_seconds())
    except TypeError:
        if getattr(t1, "tzinfo", None) is None:
            t1 = t1.replace(tzinfo=dt_tz.utc)
        if getattr(t2, "tzinfo", None) is None:
            t2 = t2.replace(tzinfo=dt_tz.utc)
        return abs((t1 - t2).total_seconds())


def _compute_breakdown(classified: list[dict]) -> dict[str, dict]:
    """Aggregate PF / win_rate / mean_return per regime label.

    Stores ``no_losses: true`` when every trade in the bucket was profitable
    so callers can distinguish "zero-loss regime" from a merely high-PF regime.
    PF is capped at _PF_NO_LOSS_SENTINEL (99) for no-loss cases so _find_best
    can still do numeric comparison, but has_no_losses preserves the signal.
    """
    buckets: dict[str, list[dict]] = {}
    for t in classified:
        buckets.setdefault(t["regime"], []).append(t)

    breakdown = {}
    for regime, group in buckets.items():
        wins = sum(1 for t in group if float(t["pnl"]) > 0)
        gross_profit = sum(float(t["pnl"]) for t in group if float(t["pnl"]) > 0)
        gross_loss = abs(sum(float(t["pnl"]) for t in group if float(t["pnl"]) < 0))
        no_losses = gross_loss == 0

        if gross_loss > 0:
            pf = gross_profit / gross_loss
        elif gross_profit > 0:
            pf = _PF_NO_LOSS_SENTINEL
        else:
            pf = 0.0

        notional_sum = sum(float(t["notional"]) for t in group if t.get("notional"))
        mean_return = (
            sum(float(t["pnl"]) for t in group) / notional_sum * 100
            if notional_sum > 0 else 0.0
        )
        breakdown[regime] = {
            "n_trades": len(group),
            "win_rate": round(wins / len(group), 4) if group else 0.0,
            "profit_factor": round(min(pf, _PF_NO_LOSS_SENTINEL), 4),
            "no_losses": no_losses,
            "mean_return_pct": round(mean_return, 4),
        }
    return breakdown


def _find_best(breakdown: dict[str, dict]) -> tuple[str | None, float | None]:
    candidates = {
        regime: data
        for regime, data in breakdown.items()
        if data["n_trades"] >= _MIN_REGIME_TRADES
    }
    if not candidates:
        return None, None
    best = max(candidates, key=lambda r: candidates[r]["profit_factor"])
    return best, candidates[best]["profit_factor"]


def _empty_result() -> dict[str, Any]:
    return {
        "best_regime": None,
        "best_pf": None,
        "is_promising": False,
        "breakdown": {},
        "regime_source": "no_trades",
        "n_total_trades": 0,
        "n_classified": 0,
    }
