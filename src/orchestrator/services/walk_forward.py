"""Walk-forward orchestration — port of ``research/scripts/walk-forward.sh``.

Splits the standard window into N rolling folds, runs a backtest per
fold's TEST period, aggregates fold-level metrics, and persists a single
``walk_forward_run`` row. The stability verdict mirrors the bash logic:

  total trades < 100             → INSUFFICIENT_EVIDENCE
  fold_pf_mean < 1.0             → NO_EDGE
  pf_positive_pct < 60           → OVERFIT (mean>1) | NO_EDGE (mean<=1)
  pf_std > 1.5                   → INCONSISTENT
  otherwise                      → ROBUST

Each fold's backtest is submitted to the JVM and polled to completion,
mirroring tick.py. With 6 folds × up to 30min, a full walk-forward can
take 3 hours — too long for an HTTP request from the cron tick. We
expose this as ``POST /walk-forward`` so the agent calls it deliberately
(or a separate cron does), and gate the PASS verdict on ROBUST.
"""

from __future__ import annotations

import statistics
from datetime import date, datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from ..clients.jvm import JvmClient
from ..config import Settings
from ..errors import OrchestratorError
from ..infra.db import Database
from ..logging import get_logger
from ..repo import hypothesis_audit as audit_repo
from ..repo import walk_forward as wf_repo
from . import sweep
from .analyze import (
    DAYS_PER_YEAR,
    deflated_sharpe_ratio,
    excess_kurtosis,
    sharpe_ratio,
    skewness,
)
from .tick import _fetch_run_metrics, _poll_backtest_status, _resolve_account_strategy

log = get_logger(__name__)

# ── Frequency-aware validation track (2026-06-07) ─────────────────────────────
# Low-turnover strategies (e.g. 1d trend hedges) cannot produce the trade
# counts the trade-level fold statistics require — a daily long/flat hedge
# trades a handful of times per year by design. Penalising it on trade count
# is a mis-calibration, not a real failure. The low-frequency track validates
# such strategies on the OUT-OF-SAMPLE daily-equity return series instead.
#
# HONESTY GUARANTEE: every statistical *bar* is identical to the standard
# (high-frequency) path — the DSR significance bar stays at the V11 0.90, the
# positive-fold consistency bar stays at 60%. ONLY the unit of observation
# changes: daily equity returns (n≈bars) instead of trades, and calendar-fold
# return instead of per-fold profit factor. No threshold is loosened; a
# high-turnover strategy is routed to the existing path and is unaffected.
LOW_FREQ_MAX_TRADES_PER_YEAR = 52   # ≥ this annualised trade rate → standard path
LOW_FREQ_MIN_TOTAL_TRADES = 8       # floor: guards against 1–2 lucky entries
LOW_FREQ_MIN_DAILY_OBS = 30         # DSR needs ≥30 return observations to be valid
LOW_FREQ_DSR_BAR = 0.90             # 0.90 — operator methodology decision 2026-06-15 (was 0.95); == V11 DSR bar, measured on the equity curve
LOW_FREQ_FOLD_POSITIVE_PCT = 60.0   # == high-freq pf_positive_pct consistency bar
LOW_FREQ_RETURN_CV_MAX = 2.5        # fold-return coeff. of variation → INCONSISTENT
_LOW_FREQ_INTERVALS = {"1d", "3d", "1w"}  # structurally low-turnover bar sizes

# ── Bear-coverage enforcement (2026-06-09) ────────────────────────────────────
# The walk-forward verdict is the gate's only OUT-OF-SAMPLE evidence. If the
# validation window contains no bear regime, a ROBUST verdict only proves the
# candidate survives a bull — which a long/flat trend hedge cannot fail by
# design. Every candidate then looks "robust but data-bound to one cycle".
# BTC market_data is backfilled to 2017-08, so the bears below ARE available;
# the old default window (2024-01-01) simply excluded them. We require the
# validation window to overlap a known bear by at least MIN_BEAR_OVERLAP_DAYS,
# with an operator override for instruments whose history genuinely predates
# none of them. This is a STRICTER gate — it never loosens an existing bar.
#
# Pinned, well-established crypto drawdowns (operator-tunable, pin-tested):
KNOWN_BEAR_REGIMES: tuple[tuple[date, date], ...] = (
    (date(2018, 1, 17), date(2018, 12, 15)),   # 2018 post-ICO bear (~ -84%)
    (date(2020, 2, 19), date(2020, 3, 23)),    # COVID liquidation crash (~ -50%)
    (date(2021, 11, 10), date(2022, 12, 31)),  # 2021-22 cycle bear (~ -77%)
)
MIN_BEAR_OVERLAP_DAYS = 45  # window must include ≥ this much of a single bear
# Replaces the old 2024-01-01 default. 2018 floor captures all three bears for
# BTC/ETH; instruments with shorter history still cover the 2020/2022 bears and
# their pre-history folds error out harmlessly (skipped by aggregate_folds).
DEFAULT_FULL_START = date(2018, 1, 1)


def _overlap_days(a_start: date, a_end: date, b_start: date, b_end: date) -> int:
    """Inclusive day-count of the intersection of two date ranges (0 if none)."""
    lo = max(a_start, b_start)
    hi = min(a_end, b_end)
    return max(0, (hi - lo).days)


def window_covers_bear(
    full_start: date, full_end: date, *, min_overlap_days: int = MIN_BEAR_OVERLAP_DAYS
) -> bool:
    """True iff ``[full_start, full_end]`` overlaps any KNOWN_BEAR_REGIME by at
    least ``min_overlap_days``. Clipping the final week of a bear does not count
    — the window must contain a *material* slice of a drawdown for an OOS
    verdict on it to mean anything."""
    return any(
        _overlap_days(full_start, full_end, b_start, b_end) >= min_overlap_days
        for b_start, b_end in KNOWN_BEAR_REGIMES
    )


def annualized_trade_rate(n_trades: int, window_days: float) -> float:
    """Trades per 365-day year over a window. ``inf`` for a zero-length
    window so a degenerate window never qualifies as low-frequency by
    accident."""
    if window_days <= 0:
        return float("inf")
    return n_trades * DAYS_PER_YEAR / window_days


def is_low_frequency(interval_name: str, n_trades: int, window_days: float) -> bool:
    """True when a strategy should be validated on the equity-curve track.

    Qualifies if the bar size is structurally low-turnover (≥ daily) OR the
    realised annualised trade rate is below ``LOW_FREQ_MAX_TRADES_PER_YEAR``.
    """
    if interval_name in _LOW_FREQ_INTERVALS:
        return True
    return annualized_trade_rate(n_trades, window_days) < LOW_FREQ_MAX_TRADES_PER_YEAR


def _lag1_autocorr(returns: list[float]) -> float:
    """Lag-1 autocorrelation of a return series. 0.0 for degenerate input."""
    n = len(returns)
    if n < 3:
        return 0.0
    mean = sum(returns) / n
    den = sum((r - mean) ** 2 for r in returns)
    if den <= 0:
        return 0.0
    num = sum((returns[i] - mean) * (returns[i - 1] - mean) for i in range(1, n))
    return num / den


def effective_sample_size(returns: list[float]) -> float:
    """Autocorrelation-adjusted effective sample size (Lo 2002).

    The deflated-Sharpe standard error assumes i.i.d. observations. A 1d
    long/flat hedge's daily returns are NOT independent — flat stretches are
    runs of ~zero and held stretches are serially correlated — so the raw
    day count would overstate the true number of independent bets and inflate
    significance. We discount by lag-1 autocorrelation ρ via the AR(1)
    variance-of-the-mean factor ``n_eff = n·(1-ρ)/(1+ρ)``. Negative ρ is not
    credited (clamped to 0 → no inflation above n); ρ is capped at 0.99 so
    n_eff can't collapse to zero. This is what keeps the equity-curve DSR
    honest rather than a rubber stamp.
    """
    n = len(returns)
    if n < 3:
        return float(n)
    rho = min(max(_lag1_autocorr(returns), 0.0), 0.99)
    return n * (1.0 - rho) / (1.0 + rho)


def _add_months(d: date, months: int) -> date:
    """Calendar-month addition without dateutil. Clamps day-of-month to
    the new month's max (matches ``relativedelta`` behaviour)."""
    total = d.year * 12 + (d.month - 1) + months
    year, month = divmod(total, 12)
    month += 1
    # Days in target month
    if month == 12:
        next_first = date(year + 1, 1, 1)
    else:
        next_first = date(year, month + 1, 1)
    last_day = (next_first - timedelta(days=1)).day
    return date(year, month, min(d.day, last_day))


def build_folds(
    full_start: date,
    full_end: date,
    train_months: int,
    test_months: int,
    step_months: int,
    n_folds: int,
) -> list[tuple[date, date]]:
    """Return ``[(test_start, test_end), ...]``. Stops when test_end
    overruns full_end. Bash counts folds where every test_end fits."""
    folds: list[tuple[date, date]] = []
    for i in range(n_folds):
        train_start = _add_months(full_start, i * step_months)
        test_start = _add_months(train_start, train_months)
        test_end = _add_months(test_start, test_months)
        if test_end > full_end:
            break
        folds.append((test_start, test_end))
    return folds


def aggregate_folds(fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    metric_data: dict[str, list[float]] = {
        "pf": [], "return_pct": [], "sharpe": [], "trades": [], "wr": [],
    }
    for f in fold_results:
        if "metrics" not in f or f.get("error"):
            continue
        for k in metric_data:
            v = f["metrics"].get(k)
            if v is not None:
                metric_data[k].append(float(v))

    def stats(key: str) -> dict[str, float] | None:
        vals = metric_data[key]
        if not vals:
            return None
        return {
            "mean": round(statistics.fmean(vals), 4),
            "std": round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0,
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
        }

    n_completed = len([f for f in fold_results if "metrics" in f and not f.get("error")])
    pf_pos = (
        round(100 * len([v for v in metric_data["pf"] if v > 1.0])
              / max(len(metric_data["pf"]), 1), 2)
    )
    return {
        "n_folds_completed": n_completed,
        "pf": stats("pf"),
        "return_pct": stats("return_pct"),
        "sharpe": stats("sharpe"),
        "pf_positive_pct": pf_pos,
        "total_trades_across_folds": int(sum(metric_data["trades"])),
    }


def stability_verdict(agg: dict[str, Any]) -> str:
    pf = agg.get("pf") or {}
    pf_mean = pf.get("mean") or 0.0
    pf_pos = agg.get("pf_positive_pct") or 0.0
    pf_std = pf.get("std") or 0.0
    n = agg.get("total_trades_across_folds") or 0

    if n < 100:
        return "INSUFFICIENT_EVIDENCE"
    if pf_mean < 1.0:
        return "NO_EDGE"
    if pf_pos < 60:
        return "OVERFIT" if pf_mean > 1.0 else "NO_EDGE"
    if pf_std > 1.5:
        return "INCONSISTENT"
    return "ROBUST"


def low_freq_stability_verdict(
    *,
    oos_daily_returns: list[float],
    fold_returns_pct: list[float],
    total_trades: int,
    n_trials: int,
) -> tuple[str, dict[str, Any]]:
    """Equity-curve stability verdict for low-turnover strategies.

    Mirrors the *role* of ``stability_verdict`` (a consistency gate) but on
    the out-of-sample daily-equity return series, and additionally
    establishes significance via the deflated Sharpe — for a low-frequency
    strategy the trade-level DSR at iteration time was invalid (n_trades<30),
    so the equity curve (n≈daily bars) is where significance is honestly
    measured. Returns ``(verdict, metrics)`` where ``verdict`` is one of the
    CHECK-constrained set {ROBUST, INCONSISTENT, INSUFFICIENT_EVIDENCE,
    OVERFIT, NO_EDGE} so it persists into ``walk_forward_run`` unchanged.

    Significance/consistency BARS are identical to the standard path
    (DSR≥0.90, positive-fold≥60%); only the observation unit differs.
    """
    n_obs = len(oos_daily_returns)
    n_folds = len(fold_returns_pct)
    fold_pos_pct = (
        round(100.0 * len([r for r in fold_returns_pct if r > 0]) / n_folds, 2)
        if n_folds
        else 0.0
    )
    fold_mean = statistics.fmean(fold_returns_pct) if fold_returns_pct else 0.0
    fold_std = statistics.pstdev(fold_returns_pct) if n_folds > 1 else 0.0

    sr = sharpe_ratio(oos_daily_returns, ann_factor=DAYS_PER_YEAR)
    skew = skewness(oos_daily_returns)
    kurt = excess_kurtosis(oos_daily_returns)  # excess — matches analyze_run
    # Autocorrelation-adjusted effective N (Lo 2002): daily returns of a
    # low-turnover hedge are serially dependent, so the DSR standard error
    # must use the effective independent-observation count, not the raw day
    # count. This is the difference between an honest gate and a rubber stamp.
    n_eff = effective_sample_size(oos_daily_returns)
    # DSR takes the PER-OBSERVATION Sharpe — SR* (= z_α/√n) and the variance
    # formula both live at observation frequency (Bailey & LdP 2014). The
    # 2026-06-09 fix (86651af) corrected this in analyze_run but missed this
    # call site: passing the √365-annualised ``sr`` inflated the DSR z-score
    # ~19×, letting i.i.d. noise with annualised Sharpe 0.19 score DSR 0.967
    # and clear the 0.90 bar as ROBUST. ``sr`` (annualised) stays reported.
    sr_per_obs = sharpe_ratio(oos_daily_returns, None)
    dsr = (
        deflated_sharpe_ratio(
            sr_per_obs, int(n_eff), skew, kurt, n_trials, pnls=oos_daily_returns
        )
        if sr_per_obs is not None and n_eff >= LOW_FREQ_MIN_DAILY_OBS
        else None
    )

    metrics: dict[str, Any] = {
        "validation_track": "LOW_FREQUENCY",
        "oos_n_obs": n_obs,
        "oos_n_eff": round(n_eff, 1),
        "oos_autocorr_lag1": round(_lag1_autocorr(oos_daily_returns), 4),
        "oos_sharpe": round(sr, 4) if sr is not None else None,
        "oos_sharpe_per_obs": round(sr_per_obs, 6) if sr_per_obs is not None else None,
        "dsr": round(dsr, 4) if dsr is not None else None,
        "dsr_n_trials": int(max(1, n_trials)),
        "fold_return_positive_pct": fold_pos_pct,
        "fold_return_mean": round(fold_mean, 4),
        "fold_return_std": round(fold_std, 4),
        "total_trades": total_trades,
        "dsr_bar": LOW_FREQ_DSR_BAR,
    }

    # 1. Not enough evidence to judge — too few trades or too few daily obs.
    if total_trades < LOW_FREQ_MIN_TOTAL_TRADES or n_obs < LOW_FREQ_MIN_DAILY_OBS:
        return "INSUFFICIENT_EVIDENCE", metrics
    # 2. No directional edge across the out-of-sample folds.
    if fold_mean <= 0:
        return "NO_EDGE", metrics
    # 3. Edge concentrated in too few calendar periods → overfit risk.
    if fold_pos_pct < LOW_FREQ_FOLD_POSITIVE_PCT:
        return "OVERFIT", metrics
    # 4. Significance not established on the equity curve (same V11 0.90 bar).
    if dsr is None or dsr < LOW_FREQ_DSR_BAR:
        return "INSUFFICIENT_EVIDENCE", metrics
    # 5. One fold carries the whole result — unstable across the cycle.
    if fold_mean > 0 and (fold_std / fold_mean) > LOW_FREQ_RETURN_CV_MAX:
        return "INCONSISTENT", metrics
    # 6. Significant on the equity curve AND consistent across calendar folds.
    return "ROBUST", metrics


def _build_payload(
    *,
    account_strategy_id: str,
    strategy_code: str,
    asset: str,
    interval_name: str,
    test_start: date,
    test_end: date,
    overrides: dict[str, Any] | None,
    allow_long: bool = True,
    allow_short: bool = True,
) -> dict[str, Any]:
    """Same shape as tick._build_submit_payload, with the per-fold window.

    Walks the ``overrides`` dict and splits out ML sentinels (``_ml_*``)
    using the same ``sweep.split_ml_overrides`` + ``build_ml_override_maps``
    pipeline as tick.py. Without this split, HYBRID walk-forwards would
    pass the sentinels as strategy params and the ML gate would not fire
    — making each fold ALGO-mode (silently divergent from the in-sample
    iteration that motivated the walk-forward).
    """
    regular, ml = sweep.split_ml_overrides(overrides or {})
    ml_override_payload = sweep.build_ml_override_maps(
        strategy_code=strategy_code,
        ml_overrides=ml,
    )
    payload: dict[str, Any] = {
        "accountStrategyId": account_strategy_id,
        "strategyCode": strategy_code,
        "asset": asset,
        "interval": interval_name,
        "startTime": f"{test_start.isoformat()}T00:00:00",
        "endTime": f"{test_end.isoformat()}T00:00:00",
        "initialCapital": 1000,
        "riskPerTradePct": 2.0,
        "feeRate": 0.00075,
        "minNotional": 5,
        "minQty": 0.000001,
        "qtyStep": 0.000001,
        "maxOpenPositions": 1,
        # Mirror the production strategy's direction policy — see tick.py.
        "allowLong": allow_long,
        "allowShort": allow_short,
        "strategyParamOverrides": {strategy_code: regular},
        "triggeredBy": "RESEARCHER",
    }
    if ml_override_payload:
        payload.update(ml_override_payload)
    return payload


class WalkForwardResult:
    def __init__(
        self,
        *,
        walk_forward_id: UUID,
        stability_verdict: str,
        aggregate: dict[str, Any],
        fold_results: list[dict[str, Any]],
    ) -> None:
        self.walk_forward_id = walk_forward_id
        self.stability_verdict = stability_verdict
        self.aggregate = aggregate
        self.fold_results = fold_results

    def to_dict(self) -> dict[str, Any]:
        return {
            "walk_forward_id": str(self.walk_forward_id),
            "stability_verdict": self.stability_verdict,
            "aggregate": self.aggregate,
            "fold_results": self.fold_results,
        }


async def run_walk_forward(
    *,
    db: Database,
    jvm: JvmClient,
    settings: Settings,  # noqa: ARG001 — reserved for prod-window/capital config
    agent_name: str,
    strategy_code: str,
    interval_name: str,
    instrument: str = "BTCUSDT",
    full_start: date = DEFAULT_FULL_START,
    full_end: date | None = None,
    train_months: int = 12,
    test_months: int = 3,
    step_months: int = 3,
    n_folds: int = 6,
    overrides: dict[str, Any] | None = None,
    motivating_iteration_id: UUID | None = None,
    require_bear_coverage: bool = True,
    external_trials: int = 0,
) -> WalkForwardResult:
    if full_end is None:
        full_end = (datetime.now(timezone.utc) - timedelta(days=1)).date()

    # Overlapping OOS test windows double-count daily returns in the
    # concatenated low-frequency series (inflating n and DSR) — the Lo-2002
    # n_eff adjustment corrects autocorrelation, not duplication. Refuse the
    # configuration outright rather than silently producing inflated stats.
    if step_months < test_months:
        raise OrchestratorError(
            status_code=400,
            error_code="walk_forward_overlapping_folds",
            message=(
                f"step_months={step_months} < test_months={test_months} produces "
                f"overlapping OOS test windows; overlapping days would be "
                f"double-counted in the validation series."
            ),
            retryable=False,
            hint="Use step_months >= test_months (default 3/3).",
        )

    # Bear-coverage gate. A walk-forward is the gate's only OOS evidence; if its
    # window has no drawdown, ROBUST proves nothing for a trend hedge. Reject
    # bull-only windows unless the operator explicitly overrides (e.g. an
    # instrument whose history predates every known bear). STRICTER, never looser.
    if require_bear_coverage and not window_covers_bear(full_start, full_end):
        raise OrchestratorError(
            status_code=422,
            error_code="validation_window_excludes_bear",
            message=(
                f"Walk-forward window [{full_start.isoformat()}, "
                f"{full_end.isoformat()}] contains no known bear regime "
                f"(≥{MIN_BEAR_OVERLAP_DAYS}d overlap). A ROBUST verdict on a "
                f"bull-only window is not out-of-sample evidence — every "
                f"candidate survives a bull by construction."
            ),
            retryable=False,
            hint=(
                "Extend full_start to include a drawdown — BTC data goes back "
                "to 2017-08, so e.g. full_start=2018-01-01 covers the 2018, "
                "2020 and 2021-22 bears. Pass override_bear_coverage=true only "
                "for instruments whose history genuinely predates every bear, "
                "with a documented journal entry."
            ),
            details={
                "full_start": full_start.isoformat(),
                "full_end": full_end.isoformat(),
                "known_bears": [
                    [b0.isoformat(), b1.isoformat()] for b0, b1 in KNOWN_BEAR_REGIMES
                ],
                "min_overlap_days": MIN_BEAR_OVERLAP_DAYS,
            },
        )

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
            hint="Seed an account_strategy row before requesting walk-forward.",
        )
    as_id = as_row["account_strategy_id"]

    folds = build_folds(full_start, full_end, train_months, test_months, step_months, n_folds)
    if not folds:
        raise OrchestratorError(
            status_code=400,
            error_code="walk_forward_no_folds",
            message="No folds fit in the requested window.",
            retryable=False,
            details={
                "full_start": full_start.isoformat(),
                "full_end": full_end.isoformat(),
                "train_months": train_months,
                "test_months": test_months,
                "step_months": step_months,
                "n_folds": n_folds,
            },
        )

    fold_results: list[dict[str, Any]] = []
    for idx, (test_start, test_end) in enumerate(folds, start=1):
        log.info(
            "walk_forward.fold_start",
            strategy_code=strategy_code,
            fold=idx,
            n_folds=len(folds),
            test_start=test_start.isoformat(),
            test_end=test_end.isoformat(),
        )
        payload = _build_payload(
            account_strategy_id=as_id,
            strategy_code=strategy_code,
            asset=instrument,
            interval_name=interval_name,
            test_start=test_start,
            test_end=test_end,
            overrides=overrides,
            allow_long=as_row["allow_long"],
            allow_short=as_row["allow_short"],
        )
        try:
            run_id = await jvm.submit_backtest(payload)
        except OrchestratorError as e:
            fold_results.append({
                "fold": idx,
                "test_start": test_start.isoformat(),
                "test_end": test_end.isoformat(),
                "run_id": None,
                "error": e.envelope.error_code,
            })
            continue

        final_status = await _poll_backtest_status(
            db, run_id, timeout_s=settings.poll_timeout_s
        )
        if final_status != "COMPLETED":
            # Distinguish a JVM-reported failure from a poll-cap expiry on a
            # still-RUNNING fold — the latter is an orchestrator latency
            # condition; lumping them together silently degraded the fold
            # set (and biased the verdict) on exactly the long-window
            # strategies most likely to hit the cap.
            capped = final_status not in ("FAILED", "CANCELLED", "ERROR")
            fold_results.append({
                "fold": idx,
                "test_start": test_start.isoformat(),
                "test_end": test_end.isoformat(),
                "run_id": run_id,
                "error": (
                    f"poll_cap_exceeded status={final_status}"
                    if capped
                    else f"status={final_status}"
                ),
                "poll_capped": capped,
            })
            continue

        async with db.acquire() as conn:
            metrics = await _fetch_run_metrics(conn, run_id)
        fold_results.append({
            "fold": idx,
            "test_start": test_start.isoformat(),
            "test_end": test_end.isoformat(),
            "run_id": run_id,
            "metrics": {
                "pf": metrics.get("profit_factor"),
                "return_pct": metrics.get("return_pct"),
                "sharpe": metrics.get("sharpe_ratio"),
                "wr": metrics.get("win_rate"),
                "trades": metrics.get("total_trades"),
                "max_dd_pct": metrics.get("max_drawdown_pct"),
                "net_profit": metrics.get("net_profit"),
            },
        })

    agg = aggregate_folds(fold_results)
    agg["n_folds_poll_capped"] = len(
        [f for f in fold_results if f.get("poll_capped")]
    )
    total_trades = agg.get("total_trades_across_folds") or 0
    # Routing rate denominator = the TEST days the trades actually occurred
    # in (completed folds only). Dividing by the full window — which includes
    # the train prefix and, since the 2018 bear-coverage floor, ~6 years of
    # calendar the folds never test — understated the rate ~5×, misrouting
    # strategies trading up to ~290/yr into the low-frequency track and past
    # its trade-level gates.
    test_window_days = sum(
        (
            date.fromisoformat(f["test_end"]) - date.fromisoformat(f["test_start"])
        ).days
        for f in fold_results
        if f.get("metrics") and not f.get("error")
    )

    # Frequency-aware routing. A low-turnover strategy (e.g. a 1d trend
    # hedge) is validated on the OOS daily-equity return series rather than
    # trade-count fold statistics it can't satisfy by design. Same bars,
    # different observation unit — see low_freq_stability_verdict.
    low_freq = is_low_frequency(interval_name, total_trades, test_window_days)
    if low_freq:
        oos_daily_returns: list[float] = []
        fold_returns_pct: list[float] = []
        async with db.acquire() as conn:
            for f in fold_results:
                m = f.get("metrics")
                if not m or f.get("error"):
                    continue
                if m.get("return_pct") is not None:
                    fold_returns_pct.append(float(m["return_pct"]))
                run_id = f.get("run_id")
                if run_id:
                    oos_daily_returns.extend(
                        await wf_repo.fetch_fold_daily_returns(conn, run_id)
                    )
            n_trials = await audit_repo.count_data_universe_trials(
                conn, instrument, interval_name,
                external_declared=external_trials,
            )
        verdict, lf_metrics = low_freq_stability_verdict(
            oos_daily_returns=oos_daily_returns,
            fold_returns_pct=fold_returns_pct,
            total_trades=total_trades,
            n_trials=n_trials,
        )
        agg["validation_track"] = "LOW_FREQUENCY"
        agg["low_freq"] = lf_metrics
        # Persist the equity-curve evidence alongside the per-fold rows so
        # the curator/operator can audit the verdict (fold_results is jsonb).
        fold_results.append({"validation_summary": lf_metrics})
    else:
        verdict = stability_verdict(agg)
        agg["validation_track"] = "STANDARD"

    pf = agg.get("pf") or {}
    ret = agg.get("return_pct") or {}
    sharpe = agg.get("sharpe") or {}

    async with db.acquire() as conn:
        wf_id = await wf_repo.insert_walk_forward(
            conn,
            strategy_code=strategy_code,
            interval_name=interval_name,
            instrument=instrument,
            full_start=datetime.combine(full_start, datetime.min.time()),
            full_end=datetime.combine(full_end, datetime.min.time()),
            train_months=train_months,
            test_months=test_months,
            n_folds=len(folds),
            fold_pf_mean=pf.get("mean"),
            fold_pf_std=pf.get("std"),
            fold_pf_min=pf.get("min"),
            fold_pf_max=pf.get("max"),
            fold_pf_positive_pct=agg.get("pf_positive_pct"),
            fold_return_mean=ret.get("mean"),
            fold_return_std=ret.get("std"),
            fold_sharpe_mean=sharpe.get("mean"),
            fold_sharpe_std=sharpe.get("std"),
            total_trades_across_folds=agg.get("total_trades_across_folds") or 0,
            stability_verdict=verdict,
            motivating_iteration_id=motivating_iteration_id,
            fold_results=fold_results,
            git_commit_hash=None,
            notes=None,
            created_by=agent_name,
        )

    log.info(
        "walk_forward.complete",
        strategy_code=strategy_code,
        walk_forward_id=str(wf_id),
        verdict=verdict,
        n_folds=len(folds),
    )
    return WalkForwardResult(
        walk_forward_id=wf_id,
        stability_verdict=verdict,
        aggregate=agg,
        fold_results=fold_results,
    )
