"""Frequency-aware walk-forward track (2026-06-07).

Low-turnover strategies (1d trend hedges) can't satisfy the trade-count
fold gate by design, so they are validated on the out-of-sample daily-equity
return series instead. These tests pin the qualification logic, every verdict
branch, and the operator-owned threshold constants.

HONESTY INVARIANT under test: the statistical bars are IDENTICAL to the
standard path (DSR≥0.95, positive-fold≥60%); only the observation unit
changes. The pin tests fail loudly if a knob drifts.
"""

from __future__ import annotations

import random

from orchestrator.services import walk_forward as wf


# ── Threshold pin tests (operator-owned; drift must surface) ──────────────────


def test_low_freq_constants_are_pinned():
    assert wf.LOW_FREQ_MAX_TRADES_PER_YEAR == 52
    assert wf.LOW_FREQ_MIN_TOTAL_TRADES == 8
    assert wf.LOW_FREQ_MIN_DAILY_OBS == 30
    assert wf.LOW_FREQ_DSR_BAR == 0.95  # == V11 significance bar
    assert wf.LOW_FREQ_FOLD_POSITIVE_PCT == 60.0  # == high-freq consistency bar
    assert wf.LOW_FREQ_RETURN_CV_MAX == 2.5
    assert wf._LOW_FREQ_INTERVALS == {"1d", "3d", "1w"}


def test_dsr_bar_matches_v11():
    """The low-freq significance bar must equal the V11 PSR/DSR bar (0.95).
    This is the contract guarantee that we did not loosen significance."""
    from orchestrator.services import analyze

    # The V11 quant-grade bar referenced in analyze.probabilistic_sharpe_ratio
    # docstring is 0.95; the low-freq path reuses it verbatim.
    assert wf.LOW_FREQ_DSR_BAR == 0.95
    assert analyze.MIN_TRADES_FOR_SIG == 100  # standard-path floor unchanged


# ── Classification ────────────────────────────────────────────────────────────


def test_annualized_trade_rate_basic():
    assert wf.annualized_trade_rate(52, 365) == 52.0
    assert wf.annualized_trade_rate(10, 730) == 5.0


def test_annualized_trade_rate_zero_window_is_inf():
    assert wf.annualized_trade_rate(5, 0) == float("inf")


def test_1d_interval_is_always_low_frequency():
    # Even a high realised trade count on a daily bar routes to low-freq.
    assert wf.is_low_frequency("1d", n_trades=500, window_days=365) is True


def test_coarser_than_daily_is_low_frequency():
    assert wf.is_low_frequency("1w", n_trades=3, window_days=365) is True


def test_intraday_low_turnover_is_low_frequency():
    # 1h bar but only ~20 trades/yr → still structurally low-turnover.
    assert wf.is_low_frequency("1h", n_trades=20, window_days=365) is True


def test_intraday_high_turnover_uses_standard_path():
    assert wf.is_low_frequency("1h", n_trades=300, window_days=365) is False


# ── Verdict branches ──────────────────────────────────────────────────────────


def _gauss_returns(n: int, mean: float, std: float, seed: int = 7) -> list[float]:
    """i.i.d. Gaussian daily returns with EXACT sample mean = ``mean``.

    De-meaning removes sample-mean drift so the realised Sharpe is
    deterministic (the DSR bar at n_eff≈400 is only ~0.16 annualised, so an
    un-centered mean-0 draw would still clear it). i.i.d. → lag-1
    autocorrelation ≈ 0 so the Lo(2002) effective-N ≈ n."""
    rng = random.Random(seed)
    raw = [rng.gauss(0.0, std) for _ in range(n)]
    m = sum(raw) / n
    return [r - m + mean for r in raw]


def _strong_returns(n: int = 400) -> list[float]:
    """High-Sharpe daily series: mean/std = 0.1 → annualised Sharpe ≈ 1.9."""
    return _gauss_returns(n, mean=0.10, std=1.0)


def _weak_returns(n: int = 400) -> list[float]:
    """Exactly zero-drift noise → annualised Sharpe ≈ 0 → DSR ≪ 0.95."""
    return _gauss_returns(n, mean=0.0, std=1.0)


def test_insufficient_when_too_few_trades():
    verdict, m = wf.low_freq_stability_verdict(
        oos_daily_returns=_strong_returns(),
        fold_returns_pct=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        total_trades=5,  # < LOW_FREQ_MIN_TOTAL_TRADES
        n_trials=1,
    )
    assert verdict == "INSUFFICIENT_EVIDENCE"
    assert m["validation_track"] == "LOW_FREQUENCY"


def test_insufficient_when_too_few_daily_obs():
    verdict, _ = wf.low_freq_stability_verdict(
        oos_daily_returns=_strong_returns(10),  # < 30 obs
        fold_returns_pct=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        total_trades=40,
        n_trials=1,
    )
    assert verdict == "INSUFFICIENT_EVIDENCE"


def test_no_edge_when_fold_mean_not_positive():
    verdict, _ = wf.low_freq_stability_verdict(
        oos_daily_returns=_strong_returns(),
        fold_returns_pct=[-1.0, -2.0, 0.5, -0.5, -1.0, 0.2],  # mean < 0
        total_trades=40,
        n_trials=1,
    )
    assert verdict == "NO_EDGE"


def test_overfit_when_too_few_positive_folds():
    # mean > 0 but only 2/6 folds positive (< 60%) → edge concentrated.
    verdict, _ = wf.low_freq_stability_verdict(
        oos_daily_returns=_strong_returns(),
        fold_returns_pct=[10.0, 8.0, -1.0, -1.0, -1.0, -1.0],
        total_trades=40,
        n_trials=1,
    )
    assert verdict == "OVERFIT"


def test_insufficient_when_dsr_below_bar():
    # Folds consistent + positive, but the equity curve's deflated Sharpe
    # can't clear 0.95 → significance not established.
    verdict, m = wf.low_freq_stability_verdict(
        oos_daily_returns=_weak_returns(),
        fold_returns_pct=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        total_trades=40,
        n_trials=1,
    )
    assert verdict == "INSUFFICIENT_EVIDENCE"
    assert m["dsr"] is None or m["dsr"] < wf.LOW_FREQ_DSR_BAR


def test_inconsistent_when_one_fold_dominates():
    # Significant equity curve, all folds positive, but one fold carries
    # the whole result (CV > 2.5) → unstable across the cycle.
    verdict, _ = wf.low_freq_stability_verdict(
        oos_daily_returns=_strong_returns(),
        fold_returns_pct=[0.001] * 8 + [50.0],
        total_trades=40,
        n_trials=1,
    )
    assert verdict == "INCONSISTENT"


def test_robust_when_significant_and_consistent():
    verdict, m = wf.low_freq_stability_verdict(
        oos_daily_returns=_strong_returns(),
        fold_returns_pct=[5.0, 8.0, 4.0, 12.0, 3.0, 6.0],
        total_trades=40,
        n_trials=1,
    )
    assert verdict == "ROBUST"
    assert m["dsr"] is not None and m["dsr"] >= wf.LOW_FREQ_DSR_BAR
    assert m["fold_return_positive_pct"] == 100.0


# ── Autocorrelation discount (Lo 2002) — the honesty guard ────────────────────


def test_iid_series_effective_n_near_raw_n():
    iid = _gauss_returns(400, mean=0.05, std=1.0)
    n_eff = wf.effective_sample_size(iid)
    # i.i.d. → lag-1 autocorr ≈ 0 → effective N within ~15% of raw count.
    assert 0.85 * 400 <= n_eff <= 400


def test_autocorrelated_series_shrinks_effective_n():
    # A near-random-walk equity (cumulative) has highly autocorrelated
    # *levels*; here build a strongly persistent series and confirm the
    # effective N is discounted well below the raw count.
    persistent = []
    prev = 0.0
    rng = random.Random(3)
    for _ in range(400):
        prev = 0.9 * prev + rng.gauss(0.05, 0.3)  # AR(1), ρ≈0.9
        persistent.append(prev)
    n_eff = wf.effective_sample_size(persistent)
    assert n_eff < 0.5 * 400  # serial dependence sharply cuts independent N


def test_negative_autocorr_not_credited_above_raw_n():
    alternating = [1.0 if i % 2 == 0 else -1.0 for i in range(100)]
    # ρ ≈ -1, but we clamp at 0 so n_eff never exceeds n.
    assert wf.effective_sample_size(alternating) <= 100.0
