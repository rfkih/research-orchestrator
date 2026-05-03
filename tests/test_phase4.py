"""Phase 4 unit tests — analyze + walk-forward pure functions.

Bootstrap CI, PSR, slippage haircut, regime stratification, statistical
verdict, decision verdict, and walk-forward fold builder + verdict logic
are all pure. The /walk-forward HTTP path needs DB+JVM and ships behind
the integration marker.
"""

from __future__ import annotations

import math
from datetime import date, datetime

from orchestrator.services.analyze import (
    MIN_TRADES_FOR_SIG,
    bootstrap_pf_ci,
    decision_verdict,
    excess_kurtosis,
    probabilistic_sharpe_ratio,
    profit_factor,
    regime_stratify,
    sample_size_adequate,
    sharpe_ratio,
    skewness,
    slippage_sensitivity,
    statistical_verdict,
)
from orchestrator.services.walk_forward import (
    aggregate_folds,
    build_folds,
    stability_verdict,
)


# ── PF ──────────────────────────────────────────────────────────────────


def test_profit_factor_positive_when_wins_exceed_losses() -> None:
    assert profit_factor([1.0, 1.0, -0.5]) == 4.0  # 2 / 0.5


def test_profit_factor_none_when_n_too_small() -> None:
    assert profit_factor([1.0]) is None


def test_profit_factor_sentinel_when_no_losses() -> None:
    pf = profit_factor([1.0, 2.0])
    assert pf == 9999.0


# ── bootstrap CI ────────────────────────────────────────────────────────


def test_bootstrap_pf_ci_skips_when_too_small() -> None:
    out = bootstrap_pf_ci([1.0, 1.0])
    assert out["low"] is None and out["high"] is None
    assert "n too small" in out["note"]


def test_bootstrap_pf_ci_seed_makes_it_deterministic() -> None:
    pnls = [1.0, 2.0, -1.0, 3.0, -0.5, 1.5, -0.2, 0.8, 1.1, -0.3]
    a = bootstrap_pf_ci(pnls, n_resamples=200, seed=42)
    b = bootstrap_pf_ci(pnls, n_resamples=200, seed=42)
    assert a == b


def test_bootstrap_pf_ci_lower_bound_above_one_for_clearly_profitable() -> None:
    # 80% wins, all winners 1.0, losers -0.5 → PF = 4.0; CI lower should exclude 1.0
    pnls = [1.0] * 80 + [-0.5] * 20
    out = bootstrap_pf_ci(pnls, n_resamples=500, seed=7)
    assert out["low"] is not None
    assert out["low"] > 1.0


# ── PSR ─────────────────────────────────────────────────────────────────


def test_psr_returns_none_when_n_too_small() -> None:
    assert probabilistic_sharpe_ratio(0.5, n_obs=10, skew=0.0, kurt=3.0) is None


def test_psr_in_unit_interval_for_normal_inputs() -> None:
    psr = probabilistic_sharpe_ratio(0.5, n_obs=200, skew=0.0, kurt=3.0)
    assert psr is not None
    assert 0.0 <= psr <= 1.0


def test_psr_is_higher_for_stronger_sharpe() -> None:
    weak = probabilistic_sharpe_ratio(0.1, n_obs=200, skew=0.0, kurt=3.0)
    strong = probabilistic_sharpe_ratio(1.5, n_obs=200, skew=0.0, kurt=3.0)
    assert weak is not None and strong is not None
    assert strong > weak


# ── moments ─────────────────────────────────────────────────────────────


def test_skewness_zero_for_symmetric() -> None:
    assert math.isclose(skewness([-2, -1, 0, 1, 2]), 0.0, abs_tol=1e-9)


def test_excess_kurtosis_zero_for_too_few() -> None:
    assert excess_kurtosis([1, 2, 3]) == 0.0


def test_sharpe_ratio_none_for_zero_variance() -> None:
    assert sharpe_ratio([1.0, 1.0, 1.0]) is None


# ── slippage ────────────────────────────────────────────────────────────


def test_slippage_sensitivity_haircuts_increase_with_bps() -> None:
    trades = [{"realized_pnl_amount": 1.0}] * 50
    out = slippage_sensitivity(trades, scenarios=(5, 10, 20, 50))
    assert out["+5bps"] > out["+10bps"] > out["+20bps"] > out["+50bps"]


def test_slippage_sensitivity_handles_empty() -> None:
    out = slippage_sensitivity([], scenarios=(20,))
    assert out == {"+20bps": 0.0}


# ── regimes ─────────────────────────────────────────────────────────────


def test_regime_stratify_groups_by_regime_and_quarter() -> None:
    trades = [
        {"realized_pnl_amount": 1.0, "entry_trend_regime": "UP",
         "entry_time": datetime(2025, 1, 15)},
        {"realized_pnl_amount": -0.5, "entry_trend_regime": "UP",
         "entry_time": datetime(2025, 4, 15)},
        {"realized_pnl_amount": 0.5, "entry_trend_regime": "DOWN",
         "entry_time": datetime(2025, 1, 20)},
    ]
    out = regime_stratify(trades)
    assert out["by_trend_regime"]["UP"]["n"] == 2
    assert out["by_trend_regime"]["UP"]["win_rate"] == 0.5  # 1 win / 2 trades
    assert out["by_trend_regime"]["DOWN"]["n"] == 1
    assert out["by_trend_regime"]["DOWN"]["win_rate"] == 1.0
    assert "2025Q1" in out["by_quarter"]
    assert "2025Q2" in out["by_quarter"]


def test_regime_stratify_buckets_missing_regime_as_unknown() -> None:
    out = regime_stratify([{"realized_pnl_amount": 1.0}])
    assert out["by_trend_regime"]["UNKNOWN"]["n"] == 1


# ── verdicts ────────────────────────────────────────────────────────────


def test_statistical_verdict_low_n_is_insufficient() -> None:
    out = statistical_verdict(50, {"low": 1.5, "high": 3.0})
    assert out["verdict"] == "INSUFFICIENT_EVIDENCE"


def test_statistical_verdict_no_edge_when_ci_below_one() -> None:
    out = statistical_verdict(200, {"low": 0.5, "high": 0.9})
    assert out["verdict"] == "NO_EDGE"


def test_statistical_verdict_significant_when_ci_above_one() -> None:
    out = statistical_verdict(200, {"low": 1.2, "high": 2.5})
    assert out["verdict"] == "SIGNIFICANT_EDGE"


def test_statistical_verdict_insufficient_when_ci_spans_one() -> None:
    out = statistical_verdict(200, {"low": 0.8, "high": 1.5})
    assert out["verdict"] == "INSUFFICIENT_EVIDENCE"


def test_decision_verdict_iterate_for_low_n() -> None:
    assert decision_verdict("SIGNIFICANT_EDGE", 50, {"+20bps": 100.0}) == "ITERATE"


def test_decision_verdict_discard_for_no_edge() -> None:
    assert decision_verdict("NO_EDGE", 200, {"+20bps": 100.0}) == "DISCARD"


def test_decision_verdict_pass_requires_positive_slippage_haircut() -> None:
    assert decision_verdict("SIGNIFICANT_EDGE", 200, {"+20bps": 100.0}) == "PASS"
    assert decision_verdict("SIGNIFICANT_EDGE", 200, {"+20bps": -5.0}) == "ITERATE"


def test_sample_size_adequate_at_threshold() -> None:
    assert sample_size_adequate(MIN_TRADES_FOR_SIG) is True
    assert sample_size_adequate(MIN_TRADES_FOR_SIG - 1) is False


# ── walk-forward fold builder ───────────────────────────────────────────


def test_build_folds_default_window_yields_expected_count() -> None:
    folds = build_folds(date(2024, 1, 1), date(2026, 4, 30), 12, 3, 3, 6)
    assert len(folds) == 5  # train12+test3 advancing by 3 months from 2024-01
    # First test is 2025-01 → 2025-04
    assert folds[0] == (date(2025, 1, 1), date(2025, 4, 1))


def test_build_folds_stops_when_test_end_overruns() -> None:
    folds = build_folds(date(2024, 1, 1), date(2024, 12, 31), 12, 3, 3, 6)
    # train_start 2024-01 → test_start 2025-01 already past full_end → 0 folds
    assert folds == []


def test_build_folds_n_folds_caps_count() -> None:
    folds = build_folds(date(2020, 1, 1), date(2030, 1, 1), 12, 3, 3, 2)
    assert len(folds) == 2


# ── aggregate + stability verdict ───────────────────────────────────────


def test_aggregate_folds_skips_errored_folds() -> None:
    fold_results = [
        {"metrics": {"pf": 1.5, "return_pct": 5.0, "sharpe": 1.0,
                     "trades": 100, "wr": 0.5}},
        {"error": "status=FAILED"},
        {"metrics": {"pf": 1.2, "return_pct": 3.0, "sharpe": 0.8,
                     "trades": 80, "wr": 0.4}},
    ]
    agg = aggregate_folds(fold_results)
    assert agg["n_folds_completed"] == 2
    assert agg["pf"]["mean"] == 1.35  # (1.5 + 1.2) / 2
    assert agg["total_trades_across_folds"] == 180
    assert agg["pf_positive_pct"] == 100.0


def test_stability_verdict_robust_for_consistent_profitable() -> None:
    agg = {
        "pf": {"mean": 1.4, "std": 0.2, "min": 1.2, "max": 1.6},
        "pf_positive_pct": 100.0,
        "total_trades_across_folds": 600,
    }
    assert stability_verdict(agg) == "ROBUST"


def test_stability_verdict_no_edge_when_pf_below_one() -> None:
    agg = {
        "pf": {"mean": 0.8, "std": 0.1, "min": 0.7, "max": 0.9},
        "pf_positive_pct": 0.0,
        "total_trades_across_folds": 600,
    }
    assert stability_verdict(agg) == "NO_EDGE"


def test_stability_verdict_overfit_when_few_positive_folds() -> None:
    agg = {
        "pf": {"mean": 1.3, "std": 0.4, "min": 0.5, "max": 2.5},
        "pf_positive_pct": 40.0,  # < 60
        "total_trades_across_folds": 600,
    }
    assert stability_verdict(agg) == "OVERFIT"


def test_stability_verdict_inconsistent_when_high_std() -> None:
    agg = {
        "pf": {"mean": 1.3, "std": 1.8, "min": 0.5, "max": 3.5},
        "pf_positive_pct": 70.0,
        "total_trades_across_folds": 600,
    }
    assert stability_verdict(agg) == "INCONSISTENT"


def test_stability_verdict_insufficient_evidence_for_low_trades() -> None:
    agg = {
        "pf": {"mean": 2.0, "std": 0.1, "min": 1.9, "max": 2.1},
        "pf_positive_pct": 100.0,
        "total_trades_across_folds": 50,
    }
    assert stability_verdict(agg) == "INSUFFICIENT_EVIDENCE"
