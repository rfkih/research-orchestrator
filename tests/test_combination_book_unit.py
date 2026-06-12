"""Unit tests for the residual-fed combination book (Phase 3, Task 4).

Pure functions only — no DB. Exercises the IC helper (Spearman rank
correlation + sign + significance) and the residual-fed ``admit`` decision
(idiosyncratic alpha + IC + marginal-Sharpe orthogonality + low redundancy).
"""
from __future__ import annotations

import random
from datetime import date, timedelta

from orchestrator.services import combination_book as cb
from orchestrator.services import factor_model as fm
from orchestrator.services import pool as pool_svc


# ── information_coefficient ──────────────────────────────────────────────────
# Fixtures use n >= _IC_MIN_OBS (raised 3 → 20 on 2026-06-12; n=8 series
# would now hit the not-judgeable sentinel by design).
def test_ic_monotonic_signal_is_plus_one_significant():
    signal = [float(i) for i in range(1, 25)]
    fwd = [10.0 * i for i in range(1, 25)]
    out = cb.information_coefficient(signal, fwd)
    assert round(out["ic"], 6) == 1.0
    assert out["significant"] is True
    assert out["sign"] == "+"


def test_ic_reversed_signal_is_minus_one_significant():
    signal = [float(i) for i in range(1, 25)]
    fwd = [10.0 * i for i in range(24, 0, -1)]
    out = cb.information_coefficient(signal, fwd)
    assert round(out["ic"], 6) == -1.0
    assert out["significant"] is True
    assert out["sign"] == "-"


def test_ic_noise_is_not_significant():
    # Deliberately uncorrelated-ish pairing → |ic| small → not significant.
    signal = [float(i) for i in range(1, 25)]
    fwd = [3.0, 1.0, 4.0, 2.0, 6.0, 5.0] * 4
    out = cb.information_coefficient(signal, fwd)
    assert out["significant"] is False


def test_ic_too_few_points_guarded():
    out = cb.information_coefficient([1.0, 2.0], [2.0, 1.0])
    assert out["significant"] is False
    assert out["sign"] == "0"


def test_ic_below_min_obs_is_not_judgeable():
    # n=8 perfectly monotonic — judgeable pre-2026-06-12, now below the
    # n>=20 floor (a 3-point |ic|=1 was auto-significant with P=1/3 under
    # the null; the floor closes that class).
    out = cb.information_coefficient(
        [float(i) for i in range(8)], [float(i) for i in range(8)]
    )
    assert out["significant"] is False
    assert out["sign"] == "0"


def test_ic_drops_non_finite_pairs():
    """A NaN/inf in either series must NOT silently corrupt the IC. The
    offending pair is dropped; if too few finite pairs remain → sentinel."""
    nan = float("nan")
    # Only the NaN pair is dropped; the remaining 24 are perfectly
    # monotonic → IC stays a clean +1, NOT a silently-wrong value.
    signal = [float(i) for i in range(1, 26)]
    fwd = [10.0 * i for i in range(1, 26)]
    signal[2] = nan
    out = cb.information_coefficient(signal, fwd)
    assert round(out["ic"], 6) == 1.0
    assert out["significant"] is True
    assert out["sign"] == "+"


def test_ic_inf_pairs_dropped_to_sentinel_when_too_thin():
    """If dropping non-finite pairs leaves fewer than the min obs, return the
    not-judgeable sentinel rather than a finite-but-wrong IC."""
    inf = float("inf")
    out = cb.information_coefficient([1.0, inf, inf, inf, 5.0],
                                    [1.0, 2.0, 3.0, 4.0, 5.0])
    # Only 2 finite pairs remain (< _IC_MIN_OBS) → sentinel.
    assert out["significant"] is False
    assert out["sign"] == "0"
    assert out["ic"] == 0.0


# ── admit ────────────────────────────────────────────────────────────────────
def _series(base: float, n: int = 40, *, start: date = date(2024, 1, 1)) -> dict:
    """A simple low-variance daily series with a tiny deterministic wiggle."""
    return {start + timedelta(days=i): base + 0.001 * ((-1) ** i) for i in range(n)}


def _good_inputs():
    """Build inputs that clear ALL four admission conditions.

    Candidate and member are independent positive-drift residual series (fixed
    seeds → deterministic) so the candidate adds genuinely uncorrelated return:
    positive marginal Sharpe, low |corr|.
    """
    n = 60
    start = date(2024, 1, 1)
    rng_m = random.Random(1)
    rng_c = random.Random(99)
    member = {start + timedelta(days=i): 0.005 + 0.01 * rng_m.gauss(0, 1) for i in range(n)}
    candidate = {start + timedelta(days=i): 0.005 + 0.01 * rng_c.gauss(0, 1) for i in range(n)}
    members = {"M1": member}
    ic = {"ic": 0.5, "significant": True, "sign": "+"}
    return candidate, members, ic


def test_admit_all_conditions_met_true():
    candidate, members, ic = _good_inputs()
    out = cb.admit(
        candidate_residuals=candidate,
        members_residuals=members,
        ic=ic,
        alpha_tstat=3.0,
        expected_sign="+",
    )
    assert out["admitted"] is True
    assert all(out["reasons"].values())
    assert "marginal" in out and "ic" in out


def test_admit_fails_on_low_alpha_tstat():
    candidate, members, ic = _good_inputs()
    out = cb.admit(
        candidate_residuals=candidate,
        members_residuals=members,
        ic=ic,
        alpha_tstat=1.0,  # below ALPHA_TSTAT_MIN
        expected_sign="+",
    )
    assert out["admitted"] is False
    assert out["reasons"]["alpha_significant"] is False


def test_admit_fails_on_none_alpha_tstat():
    candidate, members, ic = _good_inputs()
    out = cb.admit(
        candidate_residuals=candidate,
        members_residuals=members,
        ic=ic,
        alpha_tstat=None,
        expected_sign="+",
    )
    assert out["admitted"] is False
    assert out["reasons"]["alpha_significant"] is False


def test_admit_fails_on_ic_not_significant():
    candidate, members, _ = _good_inputs()
    ic = {"ic": 0.1, "significant": False, "sign": "+"}
    out = cb.admit(
        candidate_residuals=candidate,
        members_residuals=members,
        ic=ic,
        alpha_tstat=3.0,
        expected_sign="+",
    )
    assert out["admitted"] is False
    assert out["reasons"]["ic_significant_and_signed"] is False


def test_admit_fails_on_wrong_ic_sign():
    candidate, members, _ = _good_inputs()
    ic = {"ic": -0.5, "significant": True, "sign": "-"}
    out = cb.admit(
        candidate_residuals=candidate,
        members_residuals=members,
        ic=ic,
        alpha_tstat=3.0,
        expected_sign="+",  # candidate's IC sign disagrees with pre-registered
    )
    assert out["admitted"] is False
    assert out["reasons"]["ic_significant_and_signed"] is False


def test_admit_fails_on_low_marginal_sharpe():
    # A candidate identical to the sole member adds ~no marginal Sharpe.
    member = _series(0.01)
    candidate = dict(member)
    members = {"M1": member}
    ic = {"ic": 0.5, "significant": True, "sign": "+"}
    out = cb.admit(
        candidate_residuals=candidate,
        members_residuals=members,
        ic=ic,
        alpha_tstat=3.0,
        expected_sign="+",
    )
    assert out["admitted"] is False
    assert out["reasons"]["marginal_positive"] is False


def test_admit_fails_on_high_correlation():
    # Candidate perfectly correlated (a scaled copy) with the member → high
    # |corr| → redundancy condition fails.
    n = 40
    start = date(2024, 1, 1)
    member = {start + timedelta(days=i): 0.01 + 0.005 * (i % 7) for i in range(n)}
    candidate = {d: 1.5 * v for d, v in member.items()}
    members = {"M1": member}
    ic = {"ic": 0.5, "significant": True, "sign": "+"}
    out = cb.admit(
        candidate_residuals=candidate,
        members_residuals=members,
        ic=ic,
        alpha_tstat=3.0,
        expected_sign="+",
    )
    assert out["admitted"] is False
    assert out["reasons"]["low_corr"] is False


# ── Pinned constants ─────────────────────────────────────────────────────────
def test_ic_tstat_min_is_pinned():
    assert cb.IC_TSTAT_MIN == 2.0


def test_max_corr_is_pinned():
    assert cb.MAX_CORR == 0.80


def test_alpha_tstat_min_imported_not_redefined():
    # ALPHA_TSTAT_MIN must come from factor_model, not a local redefinition.
    assert cb.ALPHA_TSTAT_MIN is fm.ALPHA_TSTAT_MIN


def test_marginal_sharpe_min_imported_from_pool():
    # The marginal-Sharpe admission threshold is imported from pool, not
    # redefined locally.
    assert cb.MARGINAL_SHARPE_MIN == pool_svc.DEFAULT_THETA
