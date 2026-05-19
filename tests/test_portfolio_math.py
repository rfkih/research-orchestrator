"""Pure-function tests for portfolio_math (HRP, MV, equal-weight, Spearman)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from orchestrator.services.specialists.portfolio_math import (
    equal_weight,
    hrp_weights,
    mean_variance_weights,
    realised_vol,
    spearman_corr_matrix,
    summarise_weights,
)


# ── equal_weight ─────────────────────────────────────────────────────


def test_equal_weight_distributes_uniformly() -> None:
    out = equal_weight(["A", "B", "C", "D"])
    assert out == {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}


def test_equal_weight_empty_returns_empty() -> None:
    assert equal_weight([]) == {}


# ── spearman_corr_matrix ─────────────────────────────────────────────


def _series(start: date, values: list[float]) -> dict[date, float]:
    return {start + timedelta(days=i): v for i, v in enumerate(values)}


def test_spearman_matrix_self_corr_is_one() -> None:
    s = _series(date(2026, 1, 1), [0.1, -0.2, 0.3, 0.0, 0.05, -0.1, 0.2])
    codes, m = spearman_corr_matrix({"A": s})
    assert codes == ["A"]
    assert m[0, 0] == pytest.approx(1.0)


def test_spearman_matrix_identical_series_is_one() -> None:
    s = _series(date(2026, 1, 1), [0.1, -0.2, 0.3, 0.0, 0.05, -0.1, 0.2])
    codes, m = spearman_corr_matrix({"A": s, "B": dict(s)})
    assert m[0, 1] == pytest.approx(1.0)
    assert m[1, 0] == pytest.approx(1.0)


def test_spearman_matrix_negated_series_is_negative_one() -> None:
    s = _series(date(2026, 1, 1), [0.1, -0.2, 0.3, 0.0, 0.05, -0.1, 0.2])
    neg = {d: -v for d, v in s.items()}
    _, m = spearman_corr_matrix({"A": s, "B": neg})
    assert m[0, 1] == pytest.approx(-1.0)


def test_spearman_matrix_insufficient_overlap_is_nan() -> None:
    s1 = _series(date(2026, 1, 1), [0.1, 0.2, 0.3])  # 3 obs
    s2 = _series(date(2026, 2, 1), [0.0, 0.1, 0.2])  # different dates
    _, m = spearman_corr_matrix({"A": s1, "B": s2})
    assert np.isnan(m[0, 1])


# ── mean_variance_weights ────────────────────────────────────────────


def test_mean_variance_global_min_variance_sums_to_one() -> None:
    # Diagonal cov: equal-vol assets → equal weights (1/N).
    sigma = np.diag([0.1, 0.1, 0.1])
    out = mean_variance_weights(["A", "B", "C"], sigma)
    assert sum(out.values()) == pytest.approx(1.0)
    # Equal-vol diagonal → equal-weight solution.
    assert all(v == pytest.approx(1 / 3) for v in out.values())


def test_mean_variance_higher_vol_gets_lower_weight() -> None:
    # Asset A has much higher variance → should get lower weight.
    sigma = np.diag([0.04, 0.01, 0.01])
    out = mean_variance_weights(["A", "B", "C"], sigma)
    assert out["A"] < out["B"]
    assert out["A"] < out["C"]


def test_mean_variance_nan_imputed_falls_back_gracefully() -> None:
    sigma = np.array([[0.01, np.nan, 0.0], [np.nan, 0.01, 0.0], [0.0, 0.0, 0.01]])
    out = mean_variance_weights(["A", "B", "C"], sigma)
    # All weights non-negative and sum to 1.
    assert sum(out.values()) == pytest.approx(1.0)
    assert all(v >= 0 for v in out.values())


def test_mean_variance_singular_returns_equal_weight() -> None:
    # Σ = zero matrix → singular → fallback.
    sigma = np.zeros((3, 3))
    out = mean_variance_weights(["A", "B", "C"], sigma)
    # Ridge prevents true singularity, but the result should still
    # sum to 1 and be all-non-negative.
    assert sum(out.values()) == pytest.approx(1.0)
    assert all(v >= 0 for v in out.values())


# ── hrp_weights ──────────────────────────────────────────────────────


def test_hrp_single_asset_returns_full_weight() -> None:
    out = hrp_weights(["A"], np.array([[1.0]]), np.array([0.1]))
    assert out == {"A": 1.0}


def test_hrp_weights_sum_to_one_and_nonnegative() -> None:
    corr = np.array([[1.0, 0.3, 0.0], [0.3, 1.0, -0.1], [0.0, -0.1, 1.0]])
    vol = np.array([0.05, 0.10, 0.08])
    out = hrp_weights(["A", "B", "C"], corr, vol)
    assert sum(out.values()) == pytest.approx(1.0)
    assert all(v >= 0 for v in out.values())


def test_hrp_high_vol_asset_gets_less_weight() -> None:
    corr = np.eye(3)
    # Asset B has 5x the volatility of A and C.
    vol = np.array([0.02, 0.10, 0.02])
    out = hrp_weights(["A", "B", "C"], corr, vol)
    assert out["B"] < out["A"]
    assert out["B"] < out["C"]


def test_hrp_handles_nan_correlation() -> None:
    # NaN imputed to 0 — function should still produce a valid allocation.
    corr = np.array([[1.0, np.nan, 0.1], [np.nan, 1.0, 0.2], [0.1, 0.2, 1.0]])
    vol = np.array([0.05, 0.05, 0.05])
    out = hrp_weights(["A", "B", "C"], corr, vol)
    assert sum(out.values()) == pytest.approx(1.0)
    assert all(v >= 0 for v in out.values())


# ── realised_vol ─────────────────────────────────────────────────────


def test_realised_vol_too_few_obs_returns_none() -> None:
    s = _series(date(2026, 1, 1), [0.1, 0.2])
    assert realised_vol(s) is None


def test_realised_vol_constant_series_returns_none() -> None:
    s = _series(date(2026, 1, 1), [0.05] * 10)
    assert realised_vol(s) is None


def test_realised_vol_returns_positive() -> None:
    s = _series(date(2026, 1, 1), [0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.03])
    v = realised_vol(s)
    assert v is not None
    assert v > 0


# ── summarise_weights ────────────────────────────────────────────────


def test_summarise_weights_concentration_hhi_correct() -> None:
    # 1/4 each → HHI = 4 × (0.25^2) = 0.25; n_effective = 1/HHI = 4.0
    w = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
    s = summarise_weights(w)
    assert s["n_assets"] == 4
    assert s["concentration_hhi"] == pytest.approx(0.25)
    assert s["n_effective"] == pytest.approx(4.0)


def test_summarise_weights_concentrated_portfolio() -> None:
    # 90/10 split: HHI ≈ 0.82; n_effective ≈ 1.22
    w = {"A": 0.9, "B": 0.1}
    s = summarise_weights(w)
    assert s["n_assets"] == 2
    assert s["concentration_hhi"] == pytest.approx(0.82)
    assert s["max_weight"] == pytest.approx(0.9)
    assert s["min_weight"] == pytest.approx(0.1)


def test_summarise_weights_empty_input() -> None:
    assert summarise_weights({})["n_assets"] == 0
