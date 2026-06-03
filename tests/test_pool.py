"""Unit tests for the signal-pool marginal-Sharpe math (services/pool.py).

Pure functions, no DB. These pin the load-bearing decision: a diversifying
(uncorrelated/anti-correlated) add must show positive marginal Sharpe; a
redundant (duplicate) add must show ~zero. A bug here manufactures false alpha
into the House Book.
"""
from __future__ import annotations

from datetime import date, timedelta

from orchestrator.services import pool


def _series(values: list[float], start: date = date(2026, 1, 1)) -> dict[date, float]:
    return {start + timedelta(days=i): v for i, v in enumerate(values)}


def test_annualised_sharpe_zero_when_flat():
    assert pool.annualised_sharpe(_series([0.0] * 10)) == 0.0
    assert pool.annualised_sharpe(_series([0.01, 0.01, 0.01])) == 0.0  # < min obs


def test_annualised_sharpe_positive_for_positive_drift():
    s = pool.annualised_sharpe(_series([0.02, -0.01, 0.02, -0.01, 0.02, -0.01, 0.02, -0.01]))
    assert s > 0


def test_empty_pool_marginal_is_candidate_standalone_sharpe():
    cand = _series([0.02, -0.01, 0.02, -0.01, 0.02, -0.01, 0.02, -0.01])
    out = pool.marginal_sharpe_contribution(cand, {})
    assert out["sharpe_before"] == 0.0
    # With one series HRP gives it weight 1.0 → after-sharpe == standalone
    # (6-dp rounding in the result vs the unrounded helper → 1e-5 tolerance).
    assert abs(out["sharpe_after"] - pool.annualised_sharpe(cand)) < 1e-5
    assert out["marginal"] == out["sharpe_after"]


def test_diversifying_add_has_positive_marginal():
    # Two positive-drift series with DIFFERENT (low-correlation) patterns —
    # not a perfect hedge. Combining them keeps the mean, lowers variance →
    # higher combined Sharpe → positive marginal.
    member = _series([0.03, 0.00, 0.02, 0.01, 0.03, 0.00, 0.02, 0.01])
    cand = _series([0.01, 0.02, 0.00, 0.03, 0.01, 0.02, 0.00, 0.03])
    out = pool.marginal_sharpe_contribution(cand, {"M": member})
    assert out["marginal"] > 0, out


def test_redundant_add_has_near_zero_marginal():
    # Candidate is an exact duplicate of the only member → no diversification.
    member = _series([0.02, -0.01, 0.015, -0.005, 0.02, -0.01, 0.015, -0.005])
    cand = dict(member)
    out = pool.marginal_sharpe_contribution(cand, {"M": member})
    assert abs(out["marginal"]) < 1e-6, out
    assert out["max_abs_corr"] is not None and out["max_abs_corr"] > 0.99
