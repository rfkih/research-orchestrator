"""Pure-function tests for the portfolio rebalance service.

Covers:
  * Pin tests for operator-controlled guardrails (MIN/MAX weight,
    MIN_OBS_FOR_REBALANCE).
  * ``_apply_guardrails`` — clamp + renormalise round-trip, edge cases.
  * ``_compute_raw_weights`` — single-strategy short-circuit, optimiser
    dispatch.

DB-touching ``rebalance_book`` end-to-end lives behind the integration
marker (uses pytest-postgresql + Flyway V109) and is not in this file.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any

import numpy as np
import pytest

from orchestrator.services.rebalance import (
    MAX_WEIGHT,
    MIN_OBS_FOR_REBALANCE,
    MIN_WEIGHT,
    _apply_guardrails,
    _compute_raw_weights,
)


# ── Pin tests for operator-controlled constants ───────────────────────


def test_rebalance_guardrails_pinned() -> None:
    """Changing MIN/MAX widens or narrows the rebalance batch's bounds.
    Update only with a documented ORCHESTRATOR_CHANGE journal entry.
    MIN_OBS_FOR_REBALANCE controls when HRP falls back to equal-weight."""
    assert MIN_WEIGHT == 0.05
    assert MAX_WEIGHT == 0.50
    assert MIN_OBS_FOR_REBALANCE == 30


# ── _apply_guardrails ────────────────────────────────────────────────


def test_apply_guardrails_empty_input() -> None:
    weights, diag = _apply_guardrails({})
    assert weights == {}
    assert diag["n_clamped_low"] == 0
    assert diag["n_clamped_high"] == 0


def test_apply_guardrails_passthrough_in_bounds() -> None:
    """Weights already in [MIN, MAX] survive the clamp and renormalise
    to sum=1 trivially since they already do."""
    raw = {"LSR": 0.3, "VCB": 0.3, "VBO": 0.4}
    weights, diag = _apply_guardrails(raw)
    assert diag["n_clamped_low"] == 0
    assert diag["n_clamped_high"] == 0
    assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9)
    for c in raw:
        assert math.isclose(weights[c], raw[c], abs_tol=1e-9)


def test_apply_guardrails_clamps_low_weight() -> None:
    """A weight below MIN_WEIGHT is floored to MIN_WEIGHT, then the
    whole vector is renormalised to sum=1."""
    raw = {"LSR": 0.01, "VCB": 0.495, "VBO": 0.495}
    weights, diag = _apply_guardrails(raw)
    assert diag["n_clamped_low"] == 1
    assert weights["LSR"] >= MIN_WEIGHT * 0.9   # post-renorm may shrink slightly
    assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9)


def test_apply_guardrails_clamps_high_weight() -> None:
    """A weight above MAX_WEIGHT is capped, then renormalised. The
    capped weight is reduced further by the renorm factor."""
    raw = {"LSR": 0.9, "VCB": 0.05, "VBO": 0.05}
    weights, diag = _apply_guardrails(raw)
    assert diag["n_clamped_high"] == 1
    assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9)
    # After clamp the raw sum is 0.5 + 0.05 + 0.05 = 0.6 → renorm by 1/0.6.
    # LSR: 0.5 * (1/0.6) ≈ 0.833 (post-renorm may exceed MAX, that's
    # accepted Phase A — see module docstring).
    assert weights["LSR"] >= MAX_WEIGHT


def test_apply_guardrails_zero_sum_is_safe() -> None:
    """A pathological all-zero raw input cannot renormalise to sum=1.
    The implementation returns the clamped-but-not-renorm version
    rather than dividing by zero."""
    raw = {"LSR": 0.0, "VCB": 0.0}
    weights, diag = _apply_guardrails(raw)
    # All clamped up to MIN_WEIGHT, then renorm. 2 * MIN_WEIGHT = 0.10 → factor 10.
    assert diag["n_clamped_low"] == 2
    assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9)


# ── _compute_raw_weights ─────────────────────────────────────────────


def _make_series(n_days: int, seed: int) -> dict[date, float]:
    """Deterministic synthetic daily-return series for testing."""
    rng = np.random.default_rng(seed)
    base = date(2026, 1, 1)
    return {base + timedelta(days=i): float(rng.normal(0.001, 0.02)) for i in range(n_days)}


def test_compute_raw_weights_single_strategy_shortcircuit() -> None:
    """One strategy → weight 1.0 regardless of optimiser. HRP/MV both
    require N>=2 so the dispatcher must short-circuit before reaching
    them."""
    series_by_code: dict[str, dict[date, float]] = {"LSR": _make_series(60, 1)}
    for opt in ("HRP", "EQUAL_WEIGHT", "MEAN_VARIANCE"):
        w = _compute_raw_weights(
            optimizer=opt,  # type: ignore[arg-type]
            codes=["LSR"],
            series_by_code=series_by_code,
            min_overlap_days=5,
            mu_by_code=None,
            risk_aversion=1.0,
        )
        assert w == {"LSR": 1.0}, opt


def test_compute_raw_weights_equal_weight_dispatch() -> None:
    """EQUAL_WEIGHT should give 1/N — doesn't even need the
    correlation matrix."""
    codes = ["LSR", "VCB", "VBO"]
    series_by_code = {c: _make_series(60, i) for i, c in enumerate(codes)}
    w = _compute_raw_weights(
        optimizer="EQUAL_WEIGHT",
        codes=codes,
        series_by_code=series_by_code,
        min_overlap_days=5,
        mu_by_code=None,
        risk_aversion=1.0,
    )
    assert all(math.isclose(w[c], 1.0 / 3.0, abs_tol=1e-9) for c in codes)


def test_compute_raw_weights_hrp_returns_simplex() -> None:
    """HRP output should be a non-negative, sum-to-1 weighting over
    the provided codes."""
    codes = ["LSR", "VCB", "VBO"]
    series_by_code = {c: _make_series(60, i) for i, c in enumerate(codes)}
    w = _compute_raw_weights(
        optimizer="HRP",
        codes=codes,
        series_by_code=series_by_code,
        min_overlap_days=5,
        mu_by_code=None,
        risk_aversion=1.0,
    )
    assert set(w.keys()) == set(codes)
    assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-6)
    assert all(v >= 0 for v in w.values())


def test_compute_raw_weights_mv_with_mu() -> None:
    """Mean-variance with mu provided should produce a long-only
    simplex weighting."""
    codes = ["LSR", "VCB", "VBO"]
    series_by_code = {c: _make_series(60, i) for i, c in enumerate(codes)}
    mu = {"LSR": 0.001, "VCB": 0.001, "VBO": 0.001}
    w = _compute_raw_weights(
        optimizer="MEAN_VARIANCE",
        codes=codes,
        series_by_code=series_by_code,
        min_overlap_days=5,
        mu_by_code=mu,
        risk_aversion=1.0,
    )
    assert set(w.keys()) == set(codes)
    assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-6)
    assert all(v >= 0 for v in w.values())
