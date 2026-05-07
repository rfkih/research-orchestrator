"""Portfolio-aware correlation gate.

Layered ON TOP of (never replacing) the V11 statistical contract: a
candidate that already cleared n>=100, PF 95% CI lower>1.0, and the
+20bps slippage check still has to clear a *correlation* gate before
graduating to ``SIGNIFICANT_EDGE``.

Why: a 7%/yr strategy with rho=0.1 to the existing book (LSR/VCB/VBO)
is more valuable than a 12%/yr at rho=0.7. Capital-allocation math
says so. The 10%/yr standalone bar treats every candidate as if you
had no book — wrong objective for marginal additions.

Discount rule (pinned, operator-controlled — see PORTFOLIO_DISCOUNT_K):

    pf_lo_effective = pf_lo_raw × (1 - K × |max_corr|)

A candidate at pf_lo=1.05 with max_corr=0.6 has pf_lo_effective=0.735
and gets demoted. The same candidate at max_corr=0.1 stays at 0.998 —
still demoted, but only just; the operator sees a marginal correlation
gate firing rather than a runaway edge.

Demotion writes the candidate's stat verdict back to
``INSUFFICIENT_EVIDENCE`` (never DISCARD; the underlying signal is
real, it's just redundant). The portfolio context is stashed on
``metrics_snapshot.portfolio_corr`` for journal forensics.

This module is pure-functions-only at the boundary of analyze. The
async fetcher (``fetch_book_baselines``) reaches into ``backtest_run``
and ``backtest_trade`` and caches results module-level for 24h so the
hot tick path doesn't re-query.
"""

from __future__ import annotations

import math
import time
from datetime import date, datetime
from typing import Any

import asyncpg

from ..infra.db import Database
from ..logging import get_logger
from ..repo import trades as trades_repo

log = get_logger(__name__)

# Operator-controlled constants. Changing these widens or narrows the
# correlation gate. Pin tests in tests/test_portfolio.py guard against
# accidental drift; document any intentional change as an
# ORCHESTRATOR_CHANGE journal entry citing rationale.
PROTECTED_STRATEGY_CODES: tuple[str, ...] = ("LSR", "VCB", "VBO")
PORTFOLIO_DISCOUNT_K: float = 0.5
MIN_OVERLAP_DAYS_FOR_CORR: int = 5
BASELINE_CACHE_TTL_S: int = 24 * 3600
# Reject baseline backtest_run rows older than this. A year-old
# experimental run with non-production params would otherwise become the
# correlation baseline and the gate would fire on stale signal. 30 days
# matches the typical research cycle for re-validating the protected book.
BASELINE_MAX_AGE_DAYS: int = 30


# ── Pure functions ────────────────────────────────────────────────────


def daily_returns_from_trades(
    trades: list[dict[str, Any]], initial_capital: float = 100.0
) -> dict[date, float]:
    """Bin trade pnls by exit-day (entry-day fallback), normalize to
    fraction-of-capital.

    Why exit_time over entry_time: PnL is realised at exit. A trade
    entered on day X and held until day X+10 should not allocate its
    full PnL to day X — that biases correlation toward "did we both
    enter on the same day" instead of "did we both make money on the
    same day". Exit-day bucketing is closer to the realised cash flow.

    Known approximation: for accurate portfolio correlation a true
    daily mark-to-market series (from the JVM's equity curve) is the
    proper input. Exit-day bucketing still over-concentrates pnl on
    one day; mark-to-market would spread the unrealised gain across
    the holding period. We use exit-day for now because the equity
    curve isn't exposed via a single DB query — upgrade path: extend
    backtest_run with a daily_equity JSONB or pull from a new view.

    initial_capital normalisation matters because the candidate and
    the baseline strategies may use different account sizes;
    correlating raw pnl would mix scale with shape.
    """
    if initial_capital <= 0:
        initial_capital = 100.0
    by_day: dict[date, float] = {}
    for t in trades:
        ts = t.get("exit_time")
        if not isinstance(ts, datetime):
            ts = t.get("entry_time")
        if not isinstance(ts, datetime):
            continue
        try:
            pnl = float(t.get("realized_pnl_amount") or 0.0)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(pnl):
            continue
        d = ts.date()
        by_day[d] = by_day.get(d, 0.0) + pnl / initial_capital
    return by_day


def _pearson_on_lists(xa: list[float], xb: list[float]) -> float | None:
    """Pearson correlation on two equal-length numeric lists. Returns
    None if either has zero variance."""
    n = len(xa)
    if n < 2:
        return None
    mean_a = sum(xa) / n
    mean_b = sum(xb) / n
    num = sum((xa[i] - mean_a) * (xb[i] - mean_b) for i in range(n))
    den_a = math.sqrt(sum((xa[i] - mean_a) ** 2 for i in range(n)))
    den_b = math.sqrt(sum((xb[i] - mean_b) ** 2 for i in range(n)))
    if den_a == 0 or den_b == 0:
        return None
    return num / (den_a * den_b)


def pearson_corr(
    a: dict[date, float], b: dict[date, float], min_overlap: int = MIN_OVERLAP_DAYS_FOR_CORR
) -> float | None:
    """Pearson correlation on the date-overlap. Returns None if the
    overlap is too thin or either series is constant on the overlap."""
    overlap = sorted(set(a) & set(b))
    if len(overlap) < min_overlap:
        return None
    return _pearson_on_lists([a[d] for d in overlap], [b[d] for d in overlap])


def _ranks_with_average_ties(values: list[float]) -> list[float]:
    """Average-rank assignment for ties. Pure stdlib; matches
    scipy.stats.rankdata(method='average') closely enough for our needs."""
    indexed = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and values[indexed[j + 1]] == values[indexed[i]]:
            j += 1
        # Positions i..j are tied; assign each the average of (i+1)..(j+1).
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1
    return ranks


def spearman_corr(
    a: dict[date, float], b: dict[date, float], min_overlap: int = MIN_OVERLAP_DAYS_FOR_CORR
) -> float | None:
    """Spearman rank correlation on the date-overlap. More robust to
    outliers than Pearson — a single 5-sigma day shifts ranks by one,
    not by the value's full magnitude. We use Spearman as the binding
    correlation for the gate decision; Pearson is exposed alongside for
    forensics so the operator can see when they diverge.
    """
    overlap = sorted(set(a) & set(b))
    if len(overlap) < min_overlap:
        return None
    xa = [a[d] for d in overlap]
    xb = [b[d] for d in overlap]
    return _pearson_on_lists(_ranks_with_average_ties(xa), _ranks_with_average_ties(xb))


def max_abs_correlation(
    candidate_returns: dict[date, float],
    baselines: dict[str, dict[date, float]],
    method: str = "spearman",
) -> tuple[float | None, dict[str, float | None]]:
    """Compute per-baseline correlation; return (max_abs, per_code).

    ``method`` defaults to ``spearman`` (robust). Pass ``pearson`` for
    the linear estimator. ``per_code`` is exposed so the journal entry
    names which protected strategy is the binding constraint.
    """
    fn = spearman_corr if method == "spearman" else pearson_corr
    per_code: dict[str, float | None] = {}
    for code, baseline_returns in baselines.items():
        per_code[code] = fn(candidate_returns, baseline_returns)
    finite = [abs(c) for c in per_code.values() if c is not None]
    return (max(finite) if finite else None, per_code)


def effective_pf_lo(
    pf_lo: float, max_abs_corr: float, k: float = PORTFOLIO_DISCOUNT_K
) -> float:
    """Discount the lower CI bound by correlation with the book.

    rho=0      → unchanged.
    rho=1      → halved (default K=0.5).
    rho<0      → unchanged (the absolute value is what matters: a
                 perfectly anti-correlated strategy is also a hedge,
                 but for graduation we only penalize redundancy).
    """
    capped = max(0.0, min(1.0, abs(max_abs_corr)))
    return pf_lo * (1.0 - k * capped)


def apply_portfolio_gate(
    pf_ci: dict[str, Any],
    max_abs_corr: float | None,
    per_code_corr: dict[str, float | None] | None = None,
    k: float = PORTFOLIO_DISCOUNT_K,
    method: str = "spearman",
    pearson_per_code: dict[str, float | None] | None = None,
) -> dict[str, Any]:
    """Decide whether the correlation gate demotes a SIGNIFICANT_EDGE.

    ``max_abs_corr`` and ``per_code_corr`` are the *binding* correlation
    (Spearman by default; robust to outliers). ``pearson_per_code`` is
    the parallel Pearson series — exposed for forensics so the operator
    can see when robust and linear disagree.

    Returns a structured payload that the caller stashes on
    metrics_snapshot.portfolio_corr — and reads `demote=True` to
    rewrite the statistical_verdict back to INSUFFICIENT_EVIDENCE.
    """
    def _round_per_code(d: dict[str, float | None] | None) -> dict[str, float | None]:
        return {
            c: (round(v, 4) if v is not None else None) for c, v in (d or {}).items()
        }

    if max_abs_corr is None:
        return {
            "applied": False,
            "demote": False,
            "method": method,
            "reason": (
                "No baseline correlation available — either the protected "
                "book has no recent backtest_run rows on file or the "
                "overlap with the candidate window is below "
                f"{MIN_OVERLAP_DAYS_FOR_CORR} days. Gate skipped."
            ),
            "per_code_corr": _round_per_code(per_code_corr),
            "pearson_per_code": _round_per_code(pearson_per_code),
            "k": k,
        }
    pf_lo = pf_ci.get("low")
    if pf_lo is None:
        return {
            "applied": False,
            "demote": False,
            "method": method,
            "reason": "PF 95% CI lower bound unavailable.",
            "per_code_corr": _round_per_code(per_code_corr),
            "pearson_per_code": _round_per_code(pearson_per_code),
            "max_abs_corr": round(max_abs_corr, 4),
            "k": k,
        }
    try:
        pf_lo_f = float(pf_lo)
    except (TypeError, ValueError):
        return {
            "applied": False,
            "demote": False,
            "method": method,
            "reason": "PF lower bound not coercible to float.",
            "per_code_corr": _round_per_code(per_code_corr),
            "pearson_per_code": _round_per_code(pearson_per_code),
            "max_abs_corr": round(max_abs_corr, 4),
            "k": k,
        }
    eff = effective_pf_lo(pf_lo_f, max_abs_corr, k)
    return {
        "applied": True,
        "demote": eff <= 1.0,
        "method": method,
        "pf_lo_raw": round(pf_lo_f, 4),
        "pf_lo_effective": round(eff, 4),
        "max_abs_corr": round(max_abs_corr, 4),
        "per_code_corr": _round_per_code(per_code_corr),
        "pearson_per_code": _round_per_code(pearson_per_code),
        "k": k,
        "reason": (
            f"pf_lo {round(pf_lo_f, 4)} × (1 - {k}·|{round(max_abs_corr, 4)}|) "
            f"= {round(eff, 4)}; "
            f"{'demoted (effective <= 1.0)' if eff <= 1.0 else 'cleared (effective > 1.0)'}."
        ),
    }


# ── Async fetcher ─────────────────────────────────────────────────────


# Module-level cache. Recreated on process restart; stale-but-recent
# entries are explicitly tolerated up to BASELINE_CACHE_TTL_S because
# the protected strategies' performance shifts on a multi-week timescale,
# not per-tick. clear_baseline_cache() exists for tests.
_BASELINE_CACHE: dict[str, dict[date, float]] | None = None
_BASELINE_CACHE_T: float = 0.0


def clear_baseline_cache() -> None:
    """Invalidate the module-level cache. Intended for tests."""
    global _BASELINE_CACHE, _BASELINE_CACHE_T
    _BASELINE_CACHE = None
    _BASELINE_CACHE_T = 0.0


async def _fetch_one_baseline(
    conn: asyncpg.Connection, strategy_code: str
) -> dict[date, float] | None:
    """Read the most recent COMPLETED backtest_run for a protected
    strategy WITHIN ``BASELINE_MAX_AGE_DAYS``, bin its trades to daily
    returns. None if no recent run exists — the gate will then skip
    cleanly rather than correlating against stale or experimental params.
    """
    row = await conn.fetchrow(
        """
        SELECT backtest_run_id, initial_capital
          FROM backtest_run
         WHERE strategy_code = $1
           AND status = 'COMPLETED'
           AND created_time > NOW() - make_interval(days => $2)
         ORDER BY created_time DESC
         LIMIT 1
        """,
        strategy_code,
        BASELINE_MAX_AGE_DAYS,
    )
    if row is None:
        return None
    run_id = row["backtest_run_id"]
    initial_capital_raw = row["initial_capital"]
    try:
        initial_capital = float(initial_capital_raw or 100.0)
    except (TypeError, ValueError):
        initial_capital = 100.0
    trades = await trades_repo.fetch_trades(conn, run_id)
    return daily_returns_from_trades(trades, initial_capital)


async def evaluate_portfolio_gate(
    db: Database,
    *,
    run_metrics: dict[str, Any],
    trades: list[dict[str, Any]],
    pf_ci: dict[str, Any],
) -> dict[str, Any]:
    """Compute the portfolio correlation gate end-to-end.

    Reads ``initial_capital`` from ``run_metrics`` (orchestrator's
    ``_fetch_run_metrics`` produces snake_case from Postgres so the key
    is ``initial_capital``, not ``initialCapital`` — this function pins
    that contract). Bins candidate trades to daily returns, fetches the
    cached protected-book baselines, computes Spearman + Pearson against
    each, applies ``apply_portfolio_gate`` to decide whether to demote.

    Returning a dict (vs raising) lets the caller stash the full payload
    on metrics_snapshot.portfolio_corr regardless of demote outcome —
    forensics need the per-code corr breakdown either way.
    """
    try:
        initial_capital = float(run_metrics.get("initial_capital") or 100.0)
    except (TypeError, ValueError):
        initial_capital = 100.0
    candidate_returns = daily_returns_from_trades(trades, initial_capital)
    baselines = await fetch_book_baselines(db)
    spearman_max, spearman_per_code = max_abs_correlation(
        candidate_returns, baselines, method="spearman"
    )
    _, pearson_per_code = max_abs_correlation(
        candidate_returns, baselines, method="pearson"
    )
    return apply_portfolio_gate(
        pf_ci,
        max_abs_corr=spearman_max,
        per_code_corr=spearman_per_code,
        pearson_per_code=pearson_per_code,
        method="spearman",
    )


async def fetch_book_baselines(
    db: Database,
    codes: tuple[str, ...] = PROTECTED_STRATEGY_CODES,
    now_fn: Any = None,
) -> dict[str, dict[date, float]]:
    """Get the most-recent daily-return series for each protected code.

    Cached in process for 24h. Codes for which no backtest_run exists are
    silently omitted from the return — the gate will then have nothing
    to correlate against and apply_portfolio_gate will skip cleanly.
    """
    global _BASELINE_CACHE, _BASELINE_CACHE_T
    now = (now_fn or time.monotonic)()
    if _BASELINE_CACHE is not None and (now - _BASELINE_CACHE_T) < BASELINE_CACHE_TTL_S:
        return _BASELINE_CACHE
    out: dict[str, dict[date, float]] = {}
    async with db.acquire() as conn:
        for code in codes:
            try:
                series = await _fetch_one_baseline(conn, code)
            except Exception as e:  # noqa: BLE001
                log.warn("portfolio.baseline_fetch_failed", code=code, error=str(e))
                continue
            if series is not None and len(series) >= MIN_OVERLAP_DAYS_FOR_CORR:
                out[code] = series
    _BASELINE_CACHE = out
    _BASELINE_CACHE_T = now
    log.info("portfolio.baselines_loaded", codes=list(out.keys()))
    return out
