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


def test_non_overlapping_windows_give_zero_marginal():
    # #4 fix: member spans Jan, candidate spans a disjoint March. Combining over
    # the union + zero-fill would invent diversification; the overlap guard must
    # refuse instead.
    from datetime import date

    member = _series([0.02, -0.01, 0.02, -0.01, 0.02, -0.01], start=date(2026, 1, 1))
    cand = _series([0.02, -0.01, 0.02, -0.01, 0.02, -0.01], start=date(2026, 3, 1))
    out = pool.marginal_sharpe_contribution(cand, {"M": member})
    assert out["marginal"] == 0.0
    assert out["reason"] == "insufficient_overlap"


def test_sparse_series_is_annualised_on_calendar_grid():
    # #2 fix: a series with a long gap between trade-days must be diluted by the
    # zero-fill (calendar-daily), NOT treated as if every entry were consecutive.
    from datetime import date

    dense = {date(2026, 1, 1) + timedelta(days=i): (0.02 if i % 2 == 0 else -0.01) for i in range(8)}
    # Same 8 returns but spread far apart (weekly) → many zero days between.
    sparse = {date(2026, 1, 1) + timedelta(days=7 * i): (0.02 if i % 2 == 0 else -0.01) for i in range(8)}
    assert pool.annualised_sharpe(sparse) < pool.annualised_sharpe(dense)


# ── Phase 1-B: per-symbol cap (pure) ─────────────────────────────────

_SYM = {"BTC_a": "BTCUSDT", "BTC_b": "BTCUSDT", "ETH": "ETHUSDT"}


def test_per_symbol_cap_scales_down_concentrated_symbol():
    # BTC holds 0.8 across two members, over the 0.5 cap → scaled to 0.5, the
    # freed 0.3 water-filled to ETH. Weights still sum to 1.
    weights = {"BTC_a": 0.6, "BTC_b": 0.2, "ETH": 0.2}
    out, diag = pool.apply_per_symbol_cap(weights, _SYM, 0.50)
    assert diag["capped"] is True
    assert abs(sum(out.values()) - 1.0) < 1e-6
    assert out["BTC_a"] + out["BTC_b"] <= 0.5 + 1e-6
    assert diag["per_symbol"]["BTCUSDT"] <= 0.5 + 1e-6


def test_per_symbol_cap_noop_when_within_cap():
    weights = {"BTC_a": 0.3, "BTC_b": 0.2, "ETH": 0.5}  # BTC=0.5 exactly at cap
    out, diag = pool.apply_per_symbol_cap(weights, _SYM, 0.50)
    assert diag["capped"] is False
    assert abs(sum(out.values()) - 1.0) < 1e-6


def test_per_symbol_cap_infeasible_single_symbol_returns_unchanged():
    # One symbol can't be held below 0.5 while summing to 1 → cap skipped, flagged.
    weights = {"BTC_a": 0.7, "BTC_b": 0.3}
    out, diag = pool.apply_per_symbol_cap(weights, _SYM, 0.50)
    assert diag["capped"] is False and diag["reason"] == "infeasible_cap"
    assert out == weights
