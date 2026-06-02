"""Combinatorial-block path validation over a realized trade ledger.

This is the small-sample companion to ``services/walk_forward.py``. The
standard walk-forward runs one fixed parameter set across ~6 sequential
rolling out-of-sample windows and judges stability from those 6 noisy
fold points — which is exactly why frequency-starved strategies keep
landing on ``INSUFFICIENT_EVIDENCE``: split a thin ledger 6 ways and each
fold is dominated by one or two trades.

We instead run **one** full-span backtest, take the realized
``backtest_trade`` PnL series, partition it into ``n_blocks`` contiguous
equal-calendar time-blocks, and evaluate every combination of
``k_test`` blocks — ``C(n_blocks, k_test)`` paths. That yields a
*distribution* of profit-factor / Sharpe across dozens-to-hundreds of
overlapping sub-paths from the same data, plus a regime/year
stratification that surfaces temporal non-stationarity directly.

Why this and not textbook CPCV (López de Prado)? Canonical CPCV is for
*supervised* models: it trains a learner on N−k groups, tests on k, and
computes PBO from the in-sample-vs-out-of-sample rank across
combinations, with purge/embargo to kill label leakage. A fixed-param
rule strategy fits no model and selects no params per fold, so canonical
PBO is not computable and label-overlap purging does not apply (single
open position ⇒ non-overlapping trades ⇒ no inter-block leakage). The
faithful adaptation for a fixed-param strategy is this combinatorial
*block* evaluation of the realized ledger.

**This is an advisory diagnostic.** It does NOT touch, replace, or
loosen the V11 statistical gate or the V60 economic gate or the
walk-forward ROBUST cutoff. ``cpcv_verdict`` is reported alongside —
never in place of — ``stability_verdict``. Using a wide CI here as an
excuse to pass a candidate the real gate rejects would be fraud.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from itertools import combinations
from typing import Any
from uuid import UUID

from . import analyze

# --- Advisory verdict thresholds (diagnostic only — NOT the V11/V60 gate) ---
CPCV_PATH_POSITIVE_ROBUST_PCT = 80.0   # ≥ this fraction of paths with PF>1 → robust leg
CPCV_PATH_POSITIVE_FRAGILE_PCT = 60.0  # < this fraction of paths with PF>1 → fragile
CPCV_MEDIAN_PF_FLOOR = 1.10            # median path PF must clear this for robust leg
CPCV_WORST_PERIOD_PF_FLOOR = 0.80      # no calendar year may fall below this for robust leg
CPCV_MIN_TRADES = 40                   # below this the ledger is too thin even for CPCV
CPCV_MIN_PERIOD_TRADES = 15            # ignore year/regime strata thinner than this in the verdict

_DEFAULT_N_BLOCKS = 8
_DEFAULT_K_TEST = 4


# --------------------------------------------------------------------------- #
# Pure functions — fully unit-testable on a synthetic ledger.
# --------------------------------------------------------------------------- #
def _pnl(trade: dict[str, Any]) -> float | None:
    """Realized PnL for a closed trade, or None for open/unrealized rows."""
    return analyze._f(trade.get("realized_pnl_amount"))


def closed_trades(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Trades with a realized PnL and a usable entry_time, sorted by entry."""
    out = [
        t for t in trades
        if _pnl(t) is not None and isinstance(t.get("entry_time"), datetime)
    ]
    out.sort(key=lambda t: t["entry_time"])
    return out


def assign_blocks(
    trades: list[dict[str, Any]], n_blocks: int
) -> list[list[dict[str, Any]]]:
    """Partition the entry-time span into ``n_blocks`` equal-calendar
    contiguous blocks and bucket each trade by its entry_time.

    Equal *calendar* width (not equal trade-count) is deliberate: it maps
    each block to a slice of market time, so the combinatorial paths
    sample different regime mixes — which is what makes the distribution
    informative about temporal robustness. Empty blocks are kept (an
    empty stretch of calendar is itself signal) but contribute no trades.
    """
    if n_blocks < 2:
        raise ValueError("n_blocks must be >= 2")
    ts = closed_trades(trades)
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(n_blocks)]
    if not ts:
        return buckets
    t0 = ts[0]["entry_time"]
    t1 = ts[-1]["entry_time"]
    span = (t1 - t0).total_seconds()
    if span <= 0:
        # All trades share one instant — drop them all into block 0.
        buckets[0] = ts
        return buckets
    for t in ts:
        frac = (t["entry_time"] - t0).total_seconds() / span
        idx = min(int(frac * n_blocks), n_blocks - 1)
        buckets[idx].append(t)
    return buckets


def path_metrics(pnls: list[float]) -> dict[str, Any] | None:
    """Scale-free metrics for one combinatorial path. None if PF is
    uncomputable (n<2 or no losers)."""
    pf = analyze.profit_factor(pnls)
    if pf is None or pf >= analyze.PF_INFINITY_SENTINEL:
        return None
    sr = analyze.sharpe_ratio(pnls, ann_factor=None)
    return {
        "n": len(pnls),
        "pf": round(pf, 4),
        "sharpe_per_trade": round(sr, 4) if sr is not None else None,
        "sum_pnl": round(sum(pnls), 4),
        "mean_pnl": round(sum(pnls) / len(pnls), 4) if pnls else None,
    }


def combinatorial_paths(
    blocks: list[list[dict[str, Any]]], k_test: int
) -> list[dict[str, Any]]:
    """Evaluate every C(n_blocks, k_test) combination of test blocks.

    Each path pools the trades of its ``k_test`` blocks and computes
    ``path_metrics``. Paths whose pooled ledger is too thin for a PF
    (n<2 / no losers) are recorded with ``metrics=None`` so the count of
    *uncomputable* paths is itself visible.
    """
    n_blocks = len(blocks)
    if not (1 <= k_test < n_blocks):
        raise ValueError("k_test must satisfy 1 <= k_test < n_blocks")
    paths: list[dict[str, Any]] = []
    for combo in combinations(range(n_blocks), k_test):
        pnls = [
            p for idx in combo for t in blocks[idx]
            if (p := _pnl(t)) is not None
        ]
        paths.append({"blocks": list(combo), "metrics": path_metrics(pnls)})
    return paths


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated percentile (q in [0,1]) on a pre-sorted list."""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def summarize_paths(paths: list[dict[str, Any]]) -> dict[str, Any]:
    """Collapse the path set into a PF/Sharpe distribution summary."""
    pfs = sorted(
        p["metrics"]["pf"] for p in paths
        if p.get("metrics") and p["metrics"].get("pf") is not None
    )
    srs = [
        p["metrics"]["sharpe_per_trade"] for p in paths
        if p.get("metrics") and p["metrics"].get("sharpe_per_trade") is not None
    ]
    n_paths = len(paths)
    n_computable = len(pfs)
    pf_dist: dict[str, Any] | None = None
    if pfs:
        pf_dist = {
            "median": round(_percentile(pfs, 0.5), 4),
            "p5": round(_percentile(pfs, 0.05), 4),
            "p25": round(_percentile(pfs, 0.25), 4),
            "p75": round(_percentile(pfs, 0.75), 4),
            "p95": round(_percentile(pfs, 0.95), 4),
            "min": round(pfs[0], 4),
            "max": round(pfs[-1], 4),
        }
    path_positive_pct = (
        round(100 * sum(1 for v in pfs if v > 1.0) / n_computable, 2)
        if n_computable else 0.0
    )
    sharpe_positive_pct = (
        round(100 * sum(1 for v in srs if v > 0.0) / len(srs), 2)
        if srs else 0.0
    )
    return {
        "n_paths": n_paths,
        "n_paths_computable": n_computable,
        "pf": pf_dist,
        "path_positive_pct": path_positive_pct,
        "sharpe_positive_pct": sharpe_positive_pct,
    }


def _pf_by_key(
    trades: list[dict[str, Any]], keyfn
) -> dict[str, dict[str, Any]]:
    """Group closed trades by ``keyfn(trade)`` and report PF + n + pnl
    per stratum. PF (not just summed PnL) is what the gate cares about,
    so the verdict reasons over this rather than analyze.regime_stratify."""
    buckets: dict[str, list[float]] = {}
    for t in trades:
        p = _pnl(t)
        if p is None:
            continue
        key = keyfn(t)
        if key is None:
            continue
        buckets.setdefault(key, []).append(p)
    out: dict[str, dict[str, Any]] = {}
    for key, pnls in buckets.items():
        pf = analyze.profit_factor(pnls)
        out[key] = {
            "n": len(pnls),
            "pf": round(pf, 4) if pf is not None else None,
            "pnl": round(sum(pnls), 4),
            "win_rate": round(sum(1 for p in pnls if p > 0) / len(pnls), 4),
        }
    return out


def _year_key(t: dict[str, Any]) -> str | None:
    et = t.get("entry_time")
    return str(et.year) if isinstance(et, datetime) else None


def _regime_key(t: dict[str, Any]) -> str:
    return t.get("entry_trend_regime") or "UNKNOWN"


def cpcv_advisory_verdict(
    summary: dict[str, Any],
    pooled_pf_ci: dict[str, Any],
    by_year: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Advisory robustness call — NEVER the gate.

    Returns ``{verdict, reasons}`` where verdict ∈
    {ROBUST_ADVISORY, MIXED, FRAGILE}. The reasons list spells out which
    leg drove the call so a human (or the curator) can audit it.
    """
    reasons: list[str] = []
    pos = summary.get("path_positive_pct") or 0.0
    pf = summary.get("pf") or {}
    med_pf = pf.get("median")
    pf_low = pooled_pf_ci.get("low")

    # Worst calendar year among strata thick enough to trust.
    worst_year_pf: float | None = None
    worst_year: str | None = None
    for yr, st in by_year.items():
        if st["n"] < CPCV_MIN_PERIOD_TRADES or st["pf"] is None:
            continue
        if worst_year_pf is None or st["pf"] < worst_year_pf:
            worst_year_pf = st["pf"]
            worst_year = yr

    # FRAGILE legs (any one trips it).
    if pos < CPCV_PATH_POSITIVE_FRAGILE_PCT:
        reasons.append(
            f"only {pos}% of paths PF>1 (<{CPCV_PATH_POSITIVE_FRAGILE_PCT}%)"
        )
    if med_pf is None or med_pf < 1.0:
        reasons.append(f"median path PF {med_pf} < 1.0")
    if reasons:
        return {"verdict": "FRAGILE", "reasons": reasons}

    # ROBUST_ADVISORY requires all legs.
    robust = True
    if pos < CPCV_PATH_POSITIVE_ROBUST_PCT:
        robust = False
        reasons.append(
            f"path-positivity {pos}% < {CPCV_PATH_POSITIVE_ROBUST_PCT}%"
        )
    if med_pf < CPCV_MEDIAN_PF_FLOOR:
        robust = False
        reasons.append(f"median path PF {med_pf} < {CPCV_MEDIAN_PF_FLOOR}")
    if pf_low is None or pf_low <= 1.0:
        robust = False
        reasons.append(f"pooled bootstrap PF 95% CI low {pf_low} not > 1.0")
    if worst_year_pf is not None and worst_year_pf < CPCV_WORST_PERIOD_PF_FLOOR:
        robust = False
        reasons.append(
            f"worst year {worst_year} PF {worst_year_pf} "
            f"< {CPCV_WORST_PERIOD_PF_FLOOR} (non-stationary)"
        )
    if robust:
        return {
            "verdict": "ROBUST_ADVISORY",
            "reasons": ["all advisory legs cleared"],
        }
    return {"verdict": "MIXED", "reasons": reasons}


def analyze_ledger(
    trades: list[dict[str, Any]],
    *,
    n_blocks: int = _DEFAULT_N_BLOCKS,
    k_test: int = _DEFAULT_K_TEST,
    n_trials: int = 1,
) -> dict[str, Any]:
    """Run the full combinatorial-block analysis on a realized ledger.

    Pure function: no DB, no JVM, no network — everything downstream of
    fetching the ``backtest_trade`` rows. This is the unit-test surface.
    """
    closed = closed_trades(trades)
    n = len(closed)
    pnls = [p for t in closed if (p := _pnl(t)) is not None]

    if n < CPCV_MIN_TRADES:
        return {
            "n_trades": n,
            "note": (
                f"ledger has {n} closed trades (< {CPCV_MIN_TRADES}); too "
                f"thin for combinatorial-block inference. CPCV does not "
                f"manufacture data — extend the window or pool symbols."
            ),
            "cpcv_verdict": "INSUFFICIENT_DATA",
        }

    # Clamp k_test into the valid range for the chosen block count.
    k_test = max(1, min(k_test, n_blocks - 1))
    blocks = assign_blocks(closed, n_blocks)
    paths = combinatorial_paths(blocks, k_test)
    summary = summarize_paths(paths)

    pooled_pf_ci = analyze.bootstrap_pf_ci(pnls)
    sr_per_sample = analyze.sharpe_ratio(pnls, ann_factor=None)
    skew = analyze.skewness(pnls)
    kurt = analyze.excess_kurtosis(pnls)
    pooled_dsr = analyze.deflated_sharpe_ratio(
        sr=sr_per_sample,
        n_obs=n,
        skew=skew,
        kurt=kurt,
        n_trials=n_trials,
        pnls=pnls,
    )

    by_year = _pf_by_key(closed, _year_key)
    by_regime = _pf_by_key(closed, _regime_key)
    advisory = cpcv_advisory_verdict(summary, pooled_pf_ci, by_year)

    return {
        "n_trades": n,
        "config": {"n_blocks": n_blocks, "k_test": k_test, "n_trials": n_trials},
        "block_trade_counts": [len(b) for b in blocks],
        "paths": summary,
        "pooled": {
            "pf": round(analyze.profit_factor(pnls), 4)
            if analyze.profit_factor(pnls) is not None else None,
            "pf_ci_95": pooled_pf_ci,
            "sharpe_per_trade": round(sr_per_sample, 4)
            if sr_per_sample is not None else None,
            "dsr": round(pooled_dsr, 4) if pooled_dsr is not None else None,
            "skew": round(skew, 4),
            "excess_kurtosis": round(kurt, 4),
            "total_pnl": round(sum(pnls), 4),
        },
        "stratify": {
            "by_year_pf": by_year,
            "by_regime_pf": by_regime,
            "pnl_view": analyze.regime_stratify(closed),
        },
        "cpcv_verdict": advisory["verdict"],
        "cpcv_reasons": advisory["reasons"],
    }


# --------------------------------------------------------------------------- #
# Async driver — one full-span backtest, then the pure analysis above.
# --------------------------------------------------------------------------- #
class CpcvResult:
    def __init__(self, *, backtest_run_id: UUID | None, analysis: dict[str, Any]) -> None:
        self.backtest_run_id = backtest_run_id
        self.analysis = analysis

    def to_dict(self) -> dict[str, Any]:
        return {
            "backtest_run_id": str(self.backtest_run_id) if self.backtest_run_id else None,
            "cpcv_verdict": self.analysis.get("cpcv_verdict"),
            "analysis": self.analysis,
        }


async def run_cpcv(
    *,
    db,
    jvm,
    settings,
    agent_name: str,
    strategy_code: str,
    interval_name: str,
    instrument: str = "BTCUSDT",
    full_start: date = date(2024, 1, 1),
    full_end: date | None = None,
    overrides: dict[str, Any] | None = None,
    n_blocks: int = _DEFAULT_N_BLOCKS,
    k_test: int = _DEFAULT_K_TEST,
    n_trials: int = 1,
) -> CpcvResult:
    """Submit ONE full-span backtest, fetch its trade ledger, and run the
    combinatorial-block analysis. ~1 JVM backtest instead of C(N,k)."""
    # Imported here (not at module top) to keep the pure functions above
    # importable without dragging in the JVM/DB/sweep machinery — and to
    # mirror walk_forward.py's reuse of tick's helpers.
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz

    from ..errors import OrchestratorError
    from ..repo import trades as trades_repo
    from .tick import (
        _fetch_run_metrics,
        _poll_backtest_status,
        _resolve_account_strategy,
    )
    from .walk_forward import _build_payload

    if full_end is None:
        full_end = (_dt.now(_tz.utc) - _td(days=1)).date()

    async with db.acquire() as conn:
        as_row = await _resolve_account_strategy(
            conn, strategy_code, settings.research_account_id
        )
    if not as_row:
        raise OrchestratorError(
            status_code=412,
            error_code="account_strategy_missing",
            message=f"No account_strategy row for strategy_code={strategy_code}.",
            retryable=False,
            hint="Seed an account_strategy row before requesting CPCV.",
        )

    payload = _build_payload(
        account_strategy_id=as_row["account_strategy_id"],
        strategy_code=strategy_code,
        asset=instrument,
        interval_name=interval_name,
        test_start=full_start,
        test_end=full_end,
        overrides=overrides,
        allow_long=as_row["allow_long"],
        allow_short=as_row["allow_short"],
    )

    run_id = await jvm.submit_backtest(payload)
    final_status = await _poll_backtest_status(db, run_id)
    if final_status != "COMPLETED":
        raise OrchestratorError(
            status_code=502,
            error_code="cpcv_backtest_failed",
            message=f"Full-span backtest for CPCV ended {final_status}.",
            retryable=True,
            details={"run_id": str(run_id), "status": final_status},
        )

    async with db.acquire() as conn:
        run_metrics = await _fetch_run_metrics(conn, run_id)
        trade_rows = await trades_repo.fetch_trades(conn, run_id)

    analysis = analyze_ledger(
        trade_rows, n_blocks=n_blocks, k_test=k_test, n_trials=n_trials
    )
    analysis["run_window"] = {
        "start": full_start.isoformat(),
        "end": full_end.isoformat(),
    }
    analysis["run_total_trades"] = run_metrics.get("total_trades")
    return CpcvResult(backtest_run_id=run_id, analysis=analysis)
