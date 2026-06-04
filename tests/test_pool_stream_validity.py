"""Unit tests for the stream-validity pool-admission gate (market-neutral streams).

Pure-function: no DB. Validates that a real stationary premium passes and that a
regime-dependent stream (the FCARRY trap) fails even with a significant Sharpe.
"""
from __future__ import annotations

from datetime import date, timedelta

from orchestrator.services.pool import stream_sharpe_validity


def _series(start_year: int, n_days: int, daily_fn) -> dict[date, float]:
    out: dict[date, float] = {}
    d = date(start_year, 1, 1)
    for i in range(n_days):
        out[d] = daily_fn(i)
        d += timedelta(days=1)
    return out


def test_stationary_positive_premium_is_valid():
    # ~1.7yr of small positive drift + symmetric noise → significant Sharpe, positive
    # cumulative in every calendar year.
    s = _series(2022, 630, lambda i: 0.0015 + (0.004 if i % 2 else -0.004))
    v = stream_sharpe_validity(s)
    assert v["valid"] is True, v
    assert v["reason"] == "valid"
    assert v["psr"] >= 0.95
    assert v["years_positive"] is True


def test_regime_dependent_stream_fails_stationarity_even_with_high_psr():
    # Year 1 strongly positive, year 2 negative — a significant nonzero Sharpe, but
    # NOT positive every year (the FCARRY 2022-only pattern). Must be rejected.
    s = _series(2022, 730, lambda i: 0.002 if i < 365 else -0.001)
    v = stream_sharpe_validity(s)
    assert v["valid"] is False
    assert v["reason"] == "not_positive_every_year"
    assert v["years_positive"] is False


def test_insufficient_history_is_invalid():
    s = _series(2022, 100, lambda i: 0.0015 + (0.004 if i % 2 else -0.004))
    v = stream_sharpe_validity(s)
    assert v["valid"] is False
    assert v["reason"] == "insufficient_history"
    assert v["n_obs"] == 100


def test_zero_drift_noise_fails_psr():
    s = _series(2022, 504, lambda i: 0.003 * (1 if i % 2 else -1))  # mean 0
    v = stream_sharpe_validity(s)
    assert v["valid"] is False
    assert v["reason"] in ("psr_below_threshold", "not_positive_every_year")


def test_zero_variance_is_invalid():
    s = _series(2022, 504, lambda i: 0.001)  # constant
    v = stream_sharpe_validity(s)
    assert v["valid"] is False
    assert v["reason"] == "zero_variance"


def test_thin_boundary_year_is_not_gated():
    # A 5-day, slightly-negative boundary year (2021) must NOT fail the gate when every
    # FULL year is positive — the partial-year fix (Flaw 1).
    s: dict[date, float] = {}
    d = date(2021, 12, 27)
    for _ in range(5):
        s[d] = -0.0001
        d += timedelta(days=1)
    d = date(2022, 1, 1)
    for i in range(700):
        s[d] = 0.0015 + (0.004 if i % 2 else -0.004)
        d += timedelta(days=1)
    v = stream_sharpe_validity(s)
    assert v["valid"] is True, v
    assert 2021 not in v["judged_years"]   # thin year excluded
    assert 2022 in v["judged_years"] and 2023 in v["judged_years"]


def test_custom_min_obs_and_threshold():
    # A weak-but-positive drift that clears a relaxed PSR bar but not the default.
    s = _series(2022, 300, lambda i: 0.0005 + (0.004 if i % 2 else -0.004))
    strict = stream_sharpe_validity(s)              # default 0.95
    relaxed = stream_sharpe_validity(s, psr_threshold=0.50)
    assert relaxed["psr"] is not None
    # the relaxed run should be at least as permissive as strict
    assert (relaxed["valid"] or not strict["valid"])
