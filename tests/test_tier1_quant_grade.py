"""Tier 1 quant-grade fixes — pure-function tests.

Covers:
  * ``hypothesis_audit`` hash helpers (axis_set_hash + param_combo_hash)
  * ``deflated_sharpe_ratio`` selection-bias scaling
  * ``statistical_verdict`` DSR gate that demotes SIGNIFICANT_EDGE when
    selection-bias deflation rejects the candidate
  * ``analyze_run`` end-to-end exposing ``dsr`` + ``dsr_n_trials``

DB-touching paths (audit insert/update, queue gate 409) need
pytest-postgresql and live behind the integration marker — same pattern
as test_phase4.py.
"""

from __future__ import annotations

from datetime import datetime

import pytest

import inspect

from orchestrator.repo.hypothesis_audit import (
    axis_set_hash,
    count_data_universe_trials,
    insert_audit,
    param_combo_hash,
)
from orchestrator.services.analyze import (
    ANNUALIZED_RETURN_PASS_THRESHOLD_PCT,
    DAYS_PER_YEAR,
    DSR_SIGNIFICANCE_THRESHOLD,
    analyze_run,
    annualize_geometric_return,
    deflated_sharpe_ratio,
    statistical_verdict,
)


# ── Hash helpers ──────────────────────────────────────────────────────


def test_axis_set_hash_is_order_independent() -> None:
    # The whole point: {ATR_EXT, RSI_EXT} and {RSI_EXT, ATR_EXT} are the
    # same axis set. Without sort the hashes would diverge and the gate
    # would let duplicate sweeps slip through.
    assert axis_set_hash(["ATR_EXT", "RSI_EXT"]) == axis_set_hash(["RSI_EXT", "ATR_EXT"])


def test_axis_set_hash_distinguishes_distinct_axis_sets() -> None:
    assert axis_set_hash(["A", "B"]) != axis_set_hash(["A", "C"])


def test_axis_set_hash_empty_is_stable() -> None:
    # Empty sweep still produces a deterministic hash so an audit row
    # can record the trial.
    assert axis_set_hash([]) == axis_set_hash([])


def test_param_combo_hash_is_value_sensitive() -> None:
    a = param_combo_hash({"ATR_EXT": "1.5", "RSI_EXT": "25"})
    b = param_combo_hash({"ATR_EXT": "2.0", "RSI_EXT": "25"})
    assert a != b


def test_param_combo_hash_is_order_independent() -> None:
    a = param_combo_hash({"ATR_EXT": "1.5", "RSI_EXT": "25"})
    b = param_combo_hash({"RSI_EXT": "25", "ATR_EXT": "1.5"})
    assert a == b


def test_param_combo_hash_normalises_numeric_types() -> None:
    # The agent shouldn't be able to dodge the dedup gate by passing
    # 1 vs 1.0 vs "1" for the same parameter — all three must hash
    # identically. Same goes for fractional equivalents.
    int_form = param_combo_hash({"ATR_EXT": 1, "RSI_EXT": 25})
    float_form = param_combo_hash({"ATR_EXT": 1.0, "RSI_EXT": 25.0})
    str_form = param_combo_hash({"ATR_EXT": "1", "RSI_EXT": "25"})
    str_float_form = param_combo_hash({"ATR_EXT": "1.0", "RSI_EXT": "25"})
    assert int_form == float_form == str_form == str_float_form

    frac_int = param_combo_hash({"X": 1.5})
    frac_str = param_combo_hash({"X": "1.5"})
    assert frac_int == frac_str


# ── Deflated Sharpe ───────────────────────────────────────────────────


def test_deflated_sharpe_drops_as_trial_count_grows() -> None:
    # The whole point of HLZ scaling: the same sample SR looks weaker
    # once you account for more trials. Use a marginal SR so the CDF
    # doesn't saturate at 1.0 — strong inputs produce z-stats >> 6 where
    # the gradient is invisible at machine precision.
    sr, n, sk, kt = 0.15, 100, 0.0, 3.0
    dsr_5 = deflated_sharpe_ratio(sr, n, sk, kt, n_trials=5)
    dsr_50 = deflated_sharpe_ratio(sr, n, sk, kt, n_trials=50)
    dsr_500 = deflated_sharpe_ratio(sr, n, sk, kt, n_trials=500)
    assert dsr_5 is not None and dsr_50 is not None and dsr_500 is not None
    assert dsr_5 > dsr_50 > dsr_500


def test_deflated_sharpe_returns_none_for_too_few_obs() -> None:
    assert deflated_sharpe_ratio(0.5, n_obs=10, skew=0.0, kurt=3.0, n_trials=5) is None


def test_deflated_sharpe_clamps_zero_trials_to_one() -> None:
    # Don't blow up on n_trials=0 — clamp instead.
    out = deflated_sharpe_ratio(0.5, n_obs=200, skew=0.0, kurt=3.0, n_trials=0)
    assert out is not None
    assert 0.0 <= out <= 1.0


def test_deflated_sharpe_in_unit_interval() -> None:
    out = deflated_sharpe_ratio(1.5, n_obs=200, skew=0.0, kurt=3.0, n_trials=20)
    assert out is not None
    assert 0.0 <= out <= 1.0


# ── Statistical verdict — DSR gate ────────────────────────────────────


def test_statistical_verdict_demotes_when_dsr_below_threshold() -> None:
    # PF CI clearly excludes 1.0 favorably, but DSR rejects on
    # multiplicity — verdict must downgrade rather than promote.
    out = statistical_verdict(
        n=200,
        pf_ci={"low": 1.5, "high": 3.0},
        dsr=0.80,  # below the 0.95 quant-grade bar
    )
    assert out["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert "selection-bias" in out["reason"]


def test_statistical_verdict_promotes_when_dsr_clears_threshold() -> None:
    out = statistical_verdict(
        n=200,
        pf_ci={"low": 1.5, "high": 3.0},
        dsr=0.97,
    )
    assert out["verdict"] == "SIGNIFICANT_EDGE"


def test_statistical_verdict_falls_back_when_dsr_unavailable() -> None:
    # n<30 paths or test fixtures may pass dsr=None — must NOT block
    # otherwise-valid SIGNIFICANT_EDGE results.
    out = statistical_verdict(
        n=200,
        pf_ci={"low": 1.5, "high": 3.0},
        dsr=None,
    )
    assert out["verdict"] == "SIGNIFICANT_EDGE"


def test_statistical_verdict_no_edge_unaffected_by_dsr() -> None:
    # NO_EDGE is determined by PF CI alone — DSR doesn't change a clear
    # negative into ambiguity.
    out = statistical_verdict(
        n=200,
        pf_ci={"low": 0.5, "high": 0.9},
        dsr=0.99,
    )
    assert out["verdict"] == "NO_EDGE"


def test_dsr_threshold_constant_is_pinned() -> None:
    # Operator-controlled constant — moving it is a methodology change
    # (per quant-researcher.md's "out of bounds" list). Pin it to catch
    # accidental drift.
    assert DSR_SIGNIFICANCE_THRESHOLD == 0.95


# ── analyze_run integration ───────────────────────────────────────────


def _trades_with_clear_edge(n: int) -> list[dict]:
    # Weak positive expectancy with high noise so the annualised Sharpe
    # stays in the 0–0.5 band where DSR has a visible gradient at 4
    # decimal places. Stronger edges saturate the normal CDF to 1.0
    # and the test loses signal.
    import random
    rng = random.Random(7)
    out = []
    for i in range(n):
        pnl = rng.gauss(0.02, 2.0)
        out.append({
            "realized_pnl_amount": pnl,
            "entry_trend_regime": "UP",
            "entry_time": datetime(2025, 1, 15),
        })
    return out


def test_analyze_run_emits_dsr_field() -> None:
    run = {
        "start_time": datetime(2024, 1, 1),
        "end_time": datetime(2025, 1, 1),
        "return_pct": 30.0,
        "max_drawdown_pct": 5.0,
    }
    out = analyze_run(run, _trades_with_clear_edge(150))
    assert "dsr" in out
    assert "dsr_n_trials" in out
    # Default fallback (no cumulative_trials passed) uses
    # n_strategies_tested=5
    assert out["dsr_n_trials"] == 5


def test_analyze_run_uses_cumulative_trials_when_provided() -> None:
    run = {
        "start_time": datetime(2024, 1, 1),
        "end_time": datetime(2025, 1, 1),
        "return_pct": 30.0,
        "max_drawdown_pct": 5.0,
    }
    out_low = analyze_run(run, _trades_with_clear_edge(150), cumulative_trials=5)
    out_high = analyze_run(run, _trades_with_clear_edge(150), cumulative_trials=500)
    # Both annotate the trial count truthfully
    assert out_low["dsr_n_trials"] == 5
    assert out_high["dsr_n_trials"] == 500
    # And DSR shrinks with more trials (selection-bias deflation widens)
    assert out_low["dsr"] is not None and out_high["dsr"] is not None
    assert out_high["dsr"] < out_low["dsr"]


# ── annualize_geometric_return ────────────────────────────────────────


def test_annualize_geometric_return_one_year_passthrough() -> None:
    # 365 days = exactly 1 year — annualized value equals input.
    assert annualize_geometric_return(15.0, 365) == pytest.approx(15.0, abs=1e-6)


def test_annualize_geometric_return_half_year_doubles_compound() -> None:
    # 6-month +5% compounds to (1.05)^2 - 1 = 10.25% / yr.
    out = annualize_geometric_return(5.0, 182)
    assert out is not None
    assert out > 10.0 and out < 11.0


def test_annualize_geometric_return_multi_year_decays() -> None:
    # 2-year +25% annualized: (1.25)^0.5 - 1 ≈ 11.8%.
    out = annualize_geometric_return(25.0, 730)
    assert out is not None
    assert 11.0 < out < 13.0


def test_annualize_geometric_return_ruin_clamps_to_minus_hundred() -> None:
    # Multiplier <= 0 means the strategy compounded equity through zero
    # at least once — annualization is meaningless, return ruin sentinel.
    assert annualize_geometric_return(-100.0, 365) == -100.0
    assert annualize_geometric_return(-150.0, 365) == -100.0


def test_annualize_geometric_return_invalid_inputs_return_none() -> None:
    assert annualize_geometric_return(None, 365) is None
    assert annualize_geometric_return(15.0, None) is None
    assert annualize_geometric_return(15.0, 0) is None
    assert annualize_geometric_return(15.0, -10) is None


def test_annualize_geometric_return_uses_365_day_year() -> None:
    # Crypto markets are 24/7 — operator directive: 365 calendar days,
    # not 252 trading days. Encode the contract so it can't drift silently.
    assert DAYS_PER_YEAR == 365


# ── analyze_run surfaces annualized geom + window_days ────────────────


def test_analyze_run_includes_annualized_geom_when_run_has_geom() -> None:
    run = {
        "start_time": datetime(2024, 1, 1),
        "end_time": datetime(2025, 1, 1),
        "return_pct": 30.0,
        "max_drawdown_pct": 5.0,
        "geometric_return_pct_at_alloc_90": 25.0,
    }
    out = analyze_run(run, _trades_with_clear_edge(150))
    assert out["window_days"] == 366  # 2024 was a leap year
    assert out["geometric_return_pct_at_alloc_90"] == pytest.approx(25.0, abs=1e-6)
    # Slightly under 25 because the window is 366 days, not exactly 365.
    annualized = out["annualized_geometric_return_pct_at_alloc_90"]
    assert annualized is not None
    assert 24.0 < annualized < 25.0


def test_analyze_run_handles_missing_geom_column() -> None:
    # Legacy backtest_run rows pre-V60 don't carry the column.
    run = {
        "start_time": datetime(2024, 1, 1),
        "end_time": datetime(2025, 1, 1),
        "return_pct": 30.0,
        "max_drawdown_pct": 5.0,
    }
    out = analyze_run(run, _trades_with_clear_edge(150))
    assert out["geometric_return_pct_at_alloc_90"] is None
    assert out["annualized_geometric_return_pct_at_alloc_90"] is None


def test_pass_threshold_constant_is_ten_percent() -> None:
    # CLAUDE.md profitability bar — encode the contract so a stray edit
    # to the constant doesn't silently change which strategies promote.
    assert ANNUALIZED_RETURN_PASS_THRESHOLD_PCT == 10.0


# ── V93 data-universe scoping — signature pins ────────────────────────


def test_insert_audit_requires_symbol_and_interval_name() -> None:
    # V93: every new audit row must carry the data-universe identity so
    # count_data_universe_trials can scope DSR n_trials by (symbol,
    # interval). The V93 CHECK constraint allows NULL (it only rejects
    # non-NULL values outside the BTCUSDT/ETHUSDT allowlist), so NULL
    # symbol won't be caught at the DB layer - the only guards against
    # an accidental drop of these kwargs are (1) this pin test and (2)
    # the asyncpg arity mismatch the INSERT would raise. Catching it
    # here fails fastest and closest to the source.
    sig = inspect.signature(insert_audit)
    params = sig.parameters
    assert "symbol" in params, "insert_audit must take symbol"
    assert "interval_name" in params, "insert_audit must take interval_name"
    # Keyword-only so caller can't pass them positionally and accidentally
    # swap with strategy_code.
    assert params["symbol"].kind == inspect.Parameter.KEYWORD_ONLY
    assert params["interval_name"].kind == inspect.Parameter.KEYWORD_ONLY
    # Required (no default) — NULL data-universe defeats the whole point
    # of the V93 fix.
    assert params["symbol"].default is inspect.Parameter.empty
    assert params["interval_name"].default is inspect.Parameter.empty


def test_count_data_universe_trials_signature() -> None:
    # The replacement for count_cumulative_trials. Pin the args so a
    # rename in S3 (when tick.py swaps the caller) doesn't silently
    # break.
    sig = inspect.signature(count_data_universe_trials)
    params = sig.parameters
    assert list(params.keys()) == ["conn", "symbol", "interval_name"]
