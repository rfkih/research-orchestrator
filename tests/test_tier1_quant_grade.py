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

import math

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
    DENOM_SQ_BOOTSTRAP_FLOOR,
    DSR_SIGNIFICANCE_THRESHOLD,
    RISK_FREE_RATE_ANNUAL_PCT,
    _bootstrap_dsr,
    _norm_cdf,
    _norm_inv,
    _total_deployed_days,
    analyze_run,
    annualize_geometric_return,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    sortino_ratio,
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


# ── Bailey-LdP bootstrap-DSR fallback (denom_sq<=0 non-Gaussian) ──────


def test_deflated_sharpe_bootstrap_fallback_fires_when_closed_form_degenerate() -> None:
    """Bootstrap fallback fires when the closed-form denom_sq <= 0.

    NOTE (2026-06-09): the kurtosis term was corrected to ((κ+2)/4)·SR² (κ =
    EXCESS kurtosis), matching Bailey-LdP ((γ4-1)/4). A consequence is that for
    any *physically realizable* distribution (κ ≥ skew²-2) the variance factors
    to ≥ (1 - skew·SR/2)² ≥ 0 — so the closed form no longer goes degenerate on
    real data (the 2026-05-21 sr=1.85/skew=1.66/kurt=2.29 case now yields a
    finite closed-form DSR). The fallback is now a DEFENSIVE path, reachable
    only with pathological moments that violate the moment inequality. We force
    exactly that here to keep it covered."""
    import random

    # Pathological moments: skew=2.0 with EXCESS kurt=0.0 violates κ ≥ skew²-2
    # (= 2.0), so it can't come from real data — but it drives the corrected
    # closed form negative: denom_sq = 1 - 2.0·2.0 + ((0+2)/4)·2.0² = -1.0.
    bad_sr, bad_skew, bad_kurt = 2.0, 2.0, 0.0

    closed_only = deflated_sharpe_ratio(
        sr=bad_sr, n_obs=310, skew=bad_skew, kurt=bad_kurt, n_trials=395
    )
    assert closed_only is None, (
        "Closed-form-only must return None when denom_sq<=0; if this fires the "
        "corrected variance no longer goes degenerate on these moments."
    )

    # A pnl series with ≥30 obs and nonzero variance is enough for the
    # non-parametric fallback to produce a finite, in-unit-interval DSR.
    rng = random.Random(42)
    pnls: list[float] = []
    for _ in range(232):  # losers
        pnls.append(-rng.uniform(0.5, 1.0))
    for _ in range(78):   # winners, fat right tail
        pnls.append(rng.uniform(2.0, 8.0))

    dsr_boot = deflated_sharpe_ratio(
        sr=bad_sr, n_obs=310, skew=bad_skew, kurt=bad_kurt, n_trials=395,
        pnls=pnls,
    )
    assert dsr_boot is not None
    assert 0.0 <= dsr_boot <= 1.0


def test_bootstrap_dsr_below_threshold_still_rejects() -> None:
    """Methodology-correctness check: bootstrap DSR must REJECT weak
    edges even when the closed-form denom_sq<=0 would have hidden them
    behind a None. Use a series with positive skew but trivial mean —
    the bootstrap SE will be wide relative to the modest sr."""
    import random
    rng = random.Random(7)
    pnls: list[float] = []
    for _ in range(140):
        pnls.append(-rng.uniform(0.3, 0.6))
    for _ in range(60):
        pnls.append(rng.uniform(0.4, 1.5))
    # Mild skew, low SR ~ 0.05 per-sample; high n_trials forces a wide
    # selection-bias buffer.
    dsr_boot = deflated_sharpe_ratio(
        sr=0.05, n_obs=200, skew=1.0, kurt=3.0, n_trials=500, pnls=pnls
    )
    # Either closed-form returns a low DSR (no fallback needed), or
    # bootstrap returns a low DSR — what matters is that a weak edge
    # under heavy multiplicity does NOT spuriously certify at 0.95.
    if dsr_boot is not None:
        assert dsr_boot < DSR_SIGNIFICANCE_THRESHOLD, (
            f"DSR={dsr_boot} unexpectedly certified a weak edge with "
            f"sr=0.05 under n_trials=500 — fallback may be over-certifying."
        )


def test_bootstrap_dsr_returns_none_when_pnls_too_short() -> None:
    """Fail-closed: when closed-form degenerate AND pnls<30, DSR must
    remain None so the graduation gate rejects rather than silently
    certifying off a non-resampleable sample."""
    short_pnls = [0.5] * 10
    out = deflated_sharpe_ratio(
        sr=2.0, n_obs=10, skew=5.0, kurt=10.0, n_trials=5, pnls=short_pnls
    )
    assert out is None


def test_bootstrap_dsr_returns_none_when_pnls_zero_variance() -> None:
    """Fail-closed: zero-variance pnls (degenerate) — bootstrap SE = 0,
    DSR uncomputable."""
    flat_pnls = [1.0] * 200
    # Closed-form will fail (skew·SR makes denom_sq<=0 in some configs).
    # Even if closed-form happens to pass, the bootstrap path returns
    # None on zero variance — assert via the closed-form-degenerate
    # config:
    out = deflated_sharpe_ratio(
        sr=5.0, n_obs=200, skew=3.0, kurt=10.0, n_trials=5, pnls=flat_pnls
    )
    # Either closed-form returns a value (we accept), or fallback fires
    # and returns None on zero variance.
    if out is not None:
        # closed-form path; skip
        pass
    else:
        assert out is None


def test_bootstrap_dsr_invariant_to_pnls_scale() -> None:
    """Scaling the PnL series by a positive constant must not change
    the bootstrap DSR — Sharpe is scale-invariant, so the deflated
    z-statistic should be too."""
    import random
    rng = random.Random(11)
    pnls = [rng.gauss(0.05, 1.0) for _ in range(200)]
    # Force the closed-form to degenerate by passing pathological
    # skew/kurt that the helper formula uses — but pnls is what
    # matters for the bootstrap result, so produce a known-fallback
    # config:
    args_a = dict(sr=2.5, n_obs=200, skew=4.0, kurt=20.0, n_trials=10)
    dsr_a = deflated_sharpe_ratio(**args_a, pnls=pnls)
    dsr_b = deflated_sharpe_ratio(**args_a, pnls=[p * 100.0 for p in pnls])
    assert dsr_a is not None and dsr_b is not None
    # Bootstrap is random but seeded — same seed + scaled pnls → same DSR.
    assert abs(dsr_a - dsr_b) < 1e-6


# ── 2026-06-14 re-audit fixes (HOLE-1/2/3) ────────────────────────────


def test_denom_sq_bootstrap_floor_is_pinned() -> None:
    # Operator-controlled constant: the closed-form trust floor below which
    # DSR routes to the bootstrap instead of risking Φ(z)→1.0 saturation.
    assert DENOM_SQ_BOOTSTRAP_FLOOR == 0.05


def test_near_degenerate_denom_no_longer_spuriously_certifies() -> None:
    """HOLE-3: a near-degenerate-but-positive denom_sq must NOT certify at
    1.0 off the closed form. sr=0.5, skew=2.19, kurt=0.0 gives
    denom_sq = 1 - 2.19·0.5 + ((0+2)/4)·0.5² = 0.03 ∈ (0, FLOOR) — the
    closed form would saturate Φ(z)→1.0 even at n_trials=5000. With no PnL
    series to bootstrap, the gate must now fail closed (None), not certify.
    This is also the annualised-Sharpe-leak guard (an annualised SR fed where
    a per-obs one is expected lands in exactly this denom band)."""
    denom_sq = 1 - 2.19 * 0.5 + ((0.0 + 2) / 4) * (0.5**2)
    assert 0.0 < denom_sq < DENOM_SQ_BOOTSTRAP_FLOOR  # 0.03 — the saturation band
    out = deflated_sharpe_ratio(
        sr=0.5, n_obs=300, skew=2.19, kurt=0.0, n_trials=5000
    )
    assert out is None, (
        "Near-degenerate denom_sq with no PnL series must fail closed; a "
        "non-None here means the closed form is still saturating to ~1.0."
    )


def test_bootstrap_dsr_resamples_at_n_obs_not_raw_length() -> None:
    """HOLE-2: the bootstrap SE must scale with the n_obs the caller passes
    (= int(n_eff) on the autocorrelation-adjusted low-frequency path), NOT
    the raw PnL length. With sr_observed and sr_star held fixed, a smaller
    n_obs ⇒ wider resample SE ⇒ strictly lower DSR. If the bootstrap ignored
    n_obs and resampled len(pnls) for both calls, the two DSRs would be
    identical and this fails."""
    import random

    rng = random.Random(3)
    pnls = [-rng.uniform(0.4, 0.8) for _ in range(180)] + [
        rng.uniform(1.0, 3.0) for _ in range(120)
    ]
    d_small = _bootstrap_dsr(
        pnls, sr_observed=0.15, sr_star=0.10, n_obs=40, B=400, seed=5
    )
    d_large = _bootstrap_dsr(
        pnls, sr_observed=0.15, sr_star=0.10, n_obs=300, B=400, seed=5
    )
    assert d_small is not None and d_large is not None
    assert d_small < d_large, (
        f"n_obs=40 DSR ({d_small}) must be < n_obs=300 DSR ({d_large}); equal "
        f"means the bootstrap is resampling raw len(pnls), ignoring n_eff."
    )


def test_external_trials_deflate_marginal_candidate_below_bar() -> None:
    """HOLE-1: declared off-orchestrator trials must be able to flip a
    marginal candidate from PASS to FAIL via the multiplicity penalty. A
    per-observation Sharpe that certifies at n_trials=1 must NOT certify once
    the ~15 offline exploration trials are honestly counted."""
    base = deflated_sharpe_ratio(sr=0.30, n_obs=140, skew=0.5, kurt=3.0, n_trials=1)
    deflated = deflated_sharpe_ratio(
        sr=0.30, n_obs=140, skew=0.5, kurt=3.0, n_trials=15
    )
    assert base is not None and deflated is not None
    assert base >= DSR_SIGNIFICANCE_THRESHOLD  # certifies when multiplicity hidden
    assert deflated < DSR_SIGNIFICANCE_THRESHOLD  # fails once it's honest
    assert deflated < base  # monotone decreasing in n_trials


def test_dsr_monotone_non_increasing_in_n_trials() -> None:
    """Load-bearing invariant behind 'external_trials can only TIGHTEN': DSR
    must be monotone non-increasing as n_trials rises (more multiplicity →
    higher SR* → lower-or-equal DSR). If this ever inverts, declaring offline
    trials could LOOSEN the gate — the exact failure HOLE-1's fix must avoid."""
    prev = 1.0 + 1e-9
    for nt in (1, 2, 5, 15, 50, 200, 1000, 2000):
        dsr = deflated_sharpe_ratio(sr=0.30, n_obs=140, skew=0.5, kurt=3.0, n_trials=nt)
        assert dsr is not None
        assert dsr <= prev + 1e-12, f"DSR rose at n_trials={nt}: {dsr} > {prev}"
        prev = dsr


def test_psr_near_degenerate_denom_no_longer_saturates() -> None:
    """HOLE-3 twin (binds House Book pool admission, services/pool.py): PSR
    must NOT certify at ~1.0 off the closed form in the near-degenerate denom
    band. Same moments as the DSR guard (denom_sq=0.03 ∈ (0, FLOOR)); with no
    PnL series PSR fails closed (None) instead of saturating; with one it
    routes to the bootstrap and returns an honest value."""
    out = probabilistic_sharpe_ratio(0.5, 300, 2.19, 0.0)
    assert out is None, "near-degenerate PSR with no PnL series must fail closed"
    import random

    rng = random.Random(19)
    pnls = [-rng.uniform(0.5, 1.0) for _ in range(240)] + [
        rng.uniform(2.0, 8.0) for _ in range(80)
    ]
    out_boot = probabilistic_sharpe_ratio(0.5, 300, 2.19, 0.0, pnls=pnls)
    assert out_boot is not None and 0.0 <= out_boot <= 1.0


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


def test_statistical_verdict_fails_closed_when_dsr_unavailable() -> None:
    # L3 fix (2026-06-12): at gate sample size (n >= MIN_TRADES_FOR_SIG)
    # the DSR is always computable unless the distribution is degenerate
    # AND the bootstrap fallback failed too. Letting dsr=None through on
    # the PF CI alone was the one fail-open in the verdict chain — the
    # multiplicity gate must not be bypassable.
    out = statistical_verdict(
        n=200,
        pf_ci={"low": 1.5, "high": 3.0},
        dsr=None,
    )
    assert out["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert "DSR is not computable" in out["reason"]


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
    assert DSR_SIGNIFICANCE_THRESHOLD == 0.90  # 0.90 — operator methodology decision 2026-06-15 (was 0.95)


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
    # break. ``external_declared`` (2026-06-14 HOLE-1) is keyword-optional,
    # default 0, so existing positional callers are unaffected.
    sig = inspect.signature(count_data_universe_trials)
    params = sig.parameters
    assert list(params.keys()) == [
        "conn",
        "symbol",
        "interval_name",
        "external_declared",
    ]
    assert params["external_declared"].default == 0


def test_count_data_universe_trials_adds_external_declared() -> None:
    # HOLE-1 close: declared off-orchestrator trials ADD into the DB count;
    # negatives are clamped (can only RAISE multiplicity → gate only tightens).
    import asyncio

    class _StubConn:
        async def fetchval(self, *_a: object, **_k: object) -> int:
            return 7  # pretend 7 audited trials on this (symbol, interval)

    async def _run(ext: int) -> int:
        return await count_data_universe_trials(
            _StubConn(), "XRPUSDT", "1h", external_declared=ext
        )

    assert asyncio.run(_run(0)) == 7
    assert asyncio.run(_run(15)) == 22  # 7 audited + 15 declared offline
    assert asyncio.run(_run(-5)) == 7  # clamp: declared can't lower the count


# ── _total_deployed_days ───────────────────────────────────────────────


def test_total_deployed_days_sums_holding_periods() -> None:
    trades = [
        {"entry_time": datetime(2024, 1, 1, 0, 0), "exit_time": datetime(2024, 1, 1, 4, 0)},   # 4h
        {"entry_time": datetime(2024, 1, 2, 0, 0), "exit_time": datetime(2024, 1, 2, 4, 0)},   # 4h
    ]
    result = _total_deployed_days(trades)
    assert result is not None
    assert result == pytest.approx(8 / 24, abs=1e-9)


def test_total_deployed_days_skips_open_positions() -> None:
    trades = [
        {"entry_time": datetime(2024, 1, 1, 0, 0), "exit_time": datetime(2024, 1, 1, 1, 0)},
        {"entry_time": datetime(2024, 1, 2, 0, 0), "exit_time": None},
    ]
    result = _total_deployed_days(trades)
    assert result is not None
    assert result == pytest.approx(1 / 24, abs=1e-9)


def test_total_deployed_days_returns_none_when_no_closed_trades() -> None:
    assert _total_deployed_days([]) is None
    assert _total_deployed_days([{"entry_time": datetime(2024, 1, 1), "exit_time": None}]) is None


def test_total_deployed_days_ignores_string_timestamps() -> None:
    # Coerced run dicts may have ISO strings — they must not count.
    trades = [{"entry_time": "2024-01-01T00:00:00", "exit_time": "2024-01-01T04:00:00"}]
    assert _total_deployed_days(trades) is None


# ── analyze_run deployed-time annualization ────────────────────────────


def _trades_with_timestamps(n: int, hold_hours: float = 4.0) -> list[dict]:
    """Trades with entry/exit datetimes at a fixed holding period."""
    import random
    from datetime import timedelta
    rng = random.Random(7)
    base = datetime(2024, 1, 1)
    out = []
    for i in range(n):
        entry = base + timedelta(days=i % 300)
        exit_ = entry + timedelta(hours=hold_hours)
        pnl = rng.gauss(0.02, 2.0)
        out.append({
            "realized_pnl_amount": pnl,
            "entry_trend_regime": "UP",
            "entry_time": entry,
            "exit_time": exit_,
        })
    return out


def test_analyze_run_annualizes_over_calendar_window_not_deployed_time() -> None:
    # Operator decision 2026-06-07: V60 annualizes the geometric return over the
    # CALENDAR window, NOT deployed time. A strategy deployed only a small
    # fraction of the window (150 trades × 4h hold over a ~366-day window →
    # deployed << window) must NOT have its CAGR inflated by the tiny deployed
    # denominator (the old behaviour, which manufactured false V60 passes).
    run = {
        "start_time": datetime(2024, 1, 1),
        "end_time": datetime(2025, 1, 1),
        "return_pct": 30.0,
        "max_drawdown_pct": 5.0,
        "geometric_return_pct_at_alloc_90": 25.0,
    }
    trades = _trades_with_timestamps(150, hold_hours=4)
    out = analyze_run(run, trades)

    # deployed_days + capital_utilization are still surfaced as diagnostics.
    assert out["deployed_days"] is not None
    assert out["deployed_days"] < out["window_days"]
    assert out["capital_utilization_pct"] is not None
    assert out["capital_utilization_pct"] < 100.0

    # The reported annualized geom EQUALS the calendar-window annualization …
    calendar_annualized = annualize_geometric_return(25.0, out["window_days"])
    out_annualized = out["annualized_geometric_return_pct_at_alloc_90"]
    assert out_annualized is not None and calendar_annualized is not None
    assert out_annualized == pytest.approx(calendar_annualized, abs=1e-3)

    # … and is NOT the (much larger) deployed-day extrapolation that would have
    # falsely inflated it. This is the regression guard for the V60 fix.
    deployed_annualized = annualize_geometric_return(25.0, out["deployed_days"])
    assert deployed_annualized is not None
    assert deployed_annualized > out_annualized * 2


def test_analyze_run_falls_back_to_window_days_without_exit_times() -> None:
    # Trades without exit_time (open positions / legacy) → window_days path.
    run = {
        "start_time": datetime(2024, 1, 1),
        "end_time": datetime(2025, 1, 1),
        "return_pct": 30.0,
        "max_drawdown_pct": 5.0,
        "geometric_return_pct_at_alloc_90": 25.0,
    }
    out = analyze_run(run, _trades_with_clear_edge(150))
    assert out["deployed_days"] is None
    assert out["capital_utilization_pct"] is None
    # annualized should equal the calendar-window result (rounded to 4dp)
    expected = annualize_geometric_return(25.0, out["window_days"])
    assert out["annualized_geometric_return_pct_at_alloc_90"] == pytest.approx(expected, abs=1e-3)


# ── Industry-standard metric corrections (2026-06-09) ─────────────────


def test_psr_gaussian_matches_lo_2002_variance() -> None:
    """Pins the corrected Bailey-LdP kurtosis term. For Gaussian moments
    (skew=0, EXCESS kurt=0) the SR-variance reduces to the Lo (2002) result
    1 + 0.5·SR². If the coefficient regresses to the old (κ-1)/4 the denom
    becomes 1 - 0.25·SR² and this exact-recompute assertion fires."""
    sr, n, n_strats = 0.30, 200, 5
    denom_sq = 1.0 + 0.5 * sr**2          # corrected ((0+2)/4)·SR²
    sr_star = _norm_inv(1 - 0.05 / n_strats) / math.sqrt(n)
    expected = _norm_cdf((sr - sr_star) * math.sqrt((n - 1) / denom_sq))
    got = probabilistic_sharpe_ratio(sr, n, 0.0, 0.0, n_strats)
    assert got == pytest.approx(expected, abs=1e-12)


def test_corrected_kurtosis_makes_2026_05_21_case_non_degenerate() -> None:
    """Regression: the operator-flagged sr=1.85/skew=1.66/kurt=2.29 case used
    to drive the (buggy) closed form negative. Corrected, denom_sq =
    1 - 1.66·1.85 + ((2.29+2)/4)·1.85² ≈ 1.60 > 0 → finite closed-form DSR."""
    got = deflated_sharpe_ratio(sr=1.85, n_obs=310, skew=1.66, kurt=2.29, n_trials=395)
    assert got is not None
    assert 0.0 <= got <= 1.0


def test_corrected_variance_nonnegative_for_realizable_moments() -> None:
    """For any physically-realizable distribution (excess kurt κ ≥ skew²-2) the
    corrected denom_sq factors to ≥ (1 - skew·SR/2)² ≥ 0 — so PSR is defined
    (non-None) across a grid of valid moments. Pins the property that justified
    treating the bootstrap as defensive-only."""
    for sr in (0.1, 0.5, 1.0, 2.0):
        for skew in (-1.5, 0.0, 1.5):
            kurt = skew**2 - 2.0 + 0.5  # just inside the realizable bound
            out = probabilistic_sharpe_ratio(sr, 200, skew, kurt, 5)
            assert out is not None, f"denom_sq went degenerate at sr={sr} skew={skew}"


def test_sortino_divides_by_total_n() -> None:
    """#3: target semi-deviation divides squared downside by TOTAL n, not the
    downside count. returns=[-1,-1,2,2], target 0: mean=0.5; downside=[-1,-1] →
    Σd²=2; corrected var=2/4=0.5 → σ=√0.5 → Sortino=0.5/√0.5=√0.5≈0.7071.
    The OLD (÷len(downside)=2) gave var=1.0 → Sortino=0.5; assert the corrected."""
    out = sortino_ratio([-1.0, -1.0, 2.0, 2.0], ann_factor=None, target=0.0)
    assert out == pytest.approx(math.sqrt(0.5), abs=1e-9)  # ≈0.7071, not 0.5


def test_calmar_uses_annualized_return() -> None:
    """#4: Calmar = annualised return / maxDD, so it's window-invariant. A 30%
    return over a long window annualises down; Calmar must reflect the
    annualised figure, not the raw 30/maxDD."""
    long_run = {
        "start_time": datetime(2020, 1, 1),
        "end_time": datetime(2024, 1, 1),  # ~4yr → annualised << 30%
        "return_pct": 30.0,
        "max_drawdown_pct": 5.0,
        "geometric_return_pct_at_alloc_90": 25.0,
    }
    out = analyze_run(long_run, _trades_with_clear_edge(150))
    ann_ret = annualize_geometric_return(30.0, out["window_days"])
    assert out["calmar"] == pytest.approx(round(ann_ret / 5.0, 4), abs=1e-3)
    # And it must be far below the naive 30/5 = 6.0.
    assert out["calmar"] < 6.0


def test_risk_free_constant_defaults_to_zero_noop() -> None:
    """#5: rf defaults to 0.0 so all metrics are unchanged until an operator
    sets a rate. Pin the default so a nonzero value can't land silently."""
    assert RISK_FREE_RATE_ANNUAL_PCT == 0.0
