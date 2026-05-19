"""Tests for the review checklist (services/review.py) and storage
target_id helpers (repo/reviews.py).

Pure functions only — every check is exercised against fixture data
that mirrors the iteration_log/journal shapes. Pin tests guard the
operator-controlled thresholds. DB-touching review repo paths
(insert/fetch) need pytest-postgresql + the integration marker.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orchestrator.services.review import (
    COST_REALISM_BPS,
    DSR_THRESHOLD,
    MIN_TRADES_PASS,
    PF_CI_LOWER_BOUND_PASS,
    PORTFOLIO_CORR_THRESHOLD,
    ROBUSTNESS_NEIGHBOR_TOLERANCE,
    WARNING_FAIL_REJECT_THRESHOLD,
    aggregate_verdict,
    check_axis_not_recently_failed,
    check_cost_realism,
    check_dsr_threshold,
    check_mechanism_named,
    check_n_trades_ample,
    check_param_robustness,
    check_pf_ci_lower_bound,
    check_portfolio_fit,
    check_pre_registration,
    check_regime_concentration,
    graduation_review_checklist,
    plan_review_checklist,
)


# ── Pinned constants ──────────────────────────────────────────────────


def test_review_constants_are_pinned() -> None:
    # Operator-controlled thresholds. Update these only with a
    # documented ORCHESTRATOR_CHANGE journal entry.
    assert DSR_THRESHOLD == 0.95
    assert PF_CI_LOWER_BOUND_PASS == 1.0
    assert MIN_TRADES_PASS == 100
    assert COST_REALISM_BPS == 50
    assert PORTFOLIO_CORR_THRESHOLD == 0.5
    assert ROBUSTNESS_NEIGHBOR_TOLERANCE == 0.30
    assert WARNING_FAIL_REJECT_THRESHOLD == 2


# ── pre_registration ──────────────────────────────────────────────────


def test_pre_registration_no_hypothesis_fails() -> None:
    out = check_pre_registration(None, datetime.now(timezone.utc))
    assert out["severity"] == "blocker"
    assert out["passed"] is False


def test_pre_registration_hypothesis_after_plan_fails() -> None:
    plan_t = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    h = {"journal_id": "h1", "created_time": plan_t + timedelta(minutes=1)}
    out = check_pre_registration(h, plan_t)
    assert out["passed"] is False


def test_pre_registration_hypothesis_before_plan_passes() -> None:
    plan_t = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    h = {"journal_id": "h1", "created_time": plan_t - timedelta(hours=1)}
    out = check_pre_registration(h, plan_t)
    assert out["passed"] is True


def test_pre_registration_simultaneous_fails() -> None:
    # Strict inequality: hypothesis must STRICTLY predate the plan.
    t = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    h = {"journal_id": "h1", "created_time": t}
    out = check_pre_registration(h, t)
    assert out["passed"] is False


# ── mechanism_named ───────────────────────────────────────────────────


def test_mechanism_named_short_content_fails() -> None:
    h = {"content": "tighter ATR raises PF"}
    out = check_mechanism_named(h)
    assert out["passed"] is False


def test_mechanism_named_no_marker_fails() -> None:
    h = {"content": "Tighter ATR-extreme + RSI-extreme raises MFE/MAE without n collapse on MMR."}
    out = check_mechanism_named(h)
    assert out["passed"] is False


def test_mechanism_named_with_because_passes() -> None:
    h = {
        "content": (
            "Tighter ATR + RSI extremes raise MFE/MAE because overshoots in "
            "low-vol regimes have higher snap-back probability than at "
            "default thresholds."
        )
    }
    out = check_mechanism_named(h)
    assert out["passed"] is True


# ── axis_not_recently_failed ──────────────────────────────────────────


def test_axis_not_recently_failed_no_history_passes() -> None:
    out = check_axis_not_recently_failed([], ["ATR_EXT", "RSI_EXT"])
    assert out["passed"] is True


def test_axis_not_recently_failed_same_axis_warns() -> None:
    prior = [
        {
            "created_time": datetime(2026, 5, 1, tzinfo=timezone.utc),
            "structured_data": {
                "axis_names": ["ATR_EXT", "RSI_EXT"],
                "verdict": "INSUFFICIENT_EVIDENCE",
            },
            "title": "MMR INSUFFICIENT",
        }
    ]
    out = check_axis_not_recently_failed(prior, ["RSI_EXT", "ATR_EXT"])  # set-equal
    assert out["passed"] is False


def test_axis_not_recently_failed_different_axis_passes() -> None:
    prior = [
        {
            "created_time": datetime(2026, 5, 1, tzinfo=timezone.utc),
            "structured_data": {"axis_names": ["MA_FAST", "MA_SLOW"]},
            "title": "MMR DISCARD",
        }
    ]
    out = check_axis_not_recently_failed(prior, ["ATR_EXT", "RSI_EXT"])
    assert out["passed"] is True


# ── n_trades_ample ────────────────────────────────────────────────────


def test_n_trades_ample_below_threshold_blocks() -> None:
    out = check_n_trades_ample({"analysis": {"n_trades": 80}})
    assert out["severity"] == "blocker"
    assert out["passed"] is False


def test_n_trades_ample_at_threshold_passes() -> None:
    out = check_n_trades_ample({"analysis": {"n_trades": 100}})
    assert out["passed"] is True


def test_n_trades_ample_missing_field_blocks() -> None:
    out = check_n_trades_ample({})
    assert out["passed"] is False


# ── pf_ci_lower_bound ─────────────────────────────────────────────────


def test_pf_ci_lower_above_one_passes() -> None:
    out = check_pf_ci_lower_bound({"pf_95": {"low": 1.05, "high": 1.4}})
    assert out["passed"] is True


def test_pf_ci_lower_at_one_blocks() -> None:
    out = check_pf_ci_lower_bound({"pf_95": {"low": 1.0, "high": 1.4}})
    assert out["passed"] is False  # strict greater than


def test_pf_ci_lower_missing_blocks() -> None:
    out = check_pf_ci_lower_bound({"pf_95": {"low": None, "high": None}})
    assert out["severity"] == "blocker"
    assert out["passed"] is False


# ── dsr_threshold ─────────────────────────────────────────────────────


def test_dsr_threshold_above_passes() -> None:
    out = check_dsr_threshold({"analysis": {"dsr": 0.97, "dsr_n_trials": 20}})
    assert out["passed"] is True


def test_dsr_threshold_below_blocks() -> None:
    out = check_dsr_threshold({"analysis": {"dsr": 0.85, "dsr_n_trials": 100}})
    assert out["passed"] is False


def test_dsr_threshold_none_blocks() -> None:
    out = check_dsr_threshold({"analysis": {"dsr": None}})
    assert out["passed"] is False


# ── cost_realism ──────────────────────────────────────────────────────


def test_cost_realism_positive_passes() -> None:
    out = check_cost_realism(
        {"analysis": {"slippage_haircut_pnl": {"+50bps": 12.5}}}
    )
    assert out["passed"] is True


def test_cost_realism_negative_warns() -> None:
    out = check_cost_realism(
        {"analysis": {"slippage_haircut_pnl": {"+50bps": -3.0}}}
    )
    assert out["severity"] == "warning"
    assert out["passed"] is False


def test_cost_realism_missing_warns() -> None:
    out = check_cost_realism({"analysis": {"slippage_haircut_pnl": {}}})
    assert out["passed"] is False


# ── regime_concentration ──────────────────────────────────────────────


def test_regime_concentration_all_positive_passes() -> None:
    out = check_regime_concentration(
        {
            "analysis": {
                "regimes": {
                    "by_trend_regime": {
                        "TRENDING_UP": {"n": 50, "pnl": 10.0, "win_rate": 0.6},
                        "TRENDING_DN": {"n": 30, "pnl": 4.0, "win_rate": 0.55},
                        "CHOP": {"n": 20, "pnl": 1.0, "win_rate": 0.5},
                    }
                }
            }
        }
    )
    assert out["passed"] is True


def test_regime_concentration_one_losing_warns() -> None:
    out = check_regime_concentration(
        {
            "analysis": {
                "regimes": {
                    "by_trend_regime": {
                        "TRENDING_UP": {"n": 50, "pnl": 10.0},
                        "CHOP": {"n": 20, "pnl": -3.0},  # losing regime
                    }
                }
            }
        }
    )
    assert out["passed"] is False


def test_regime_concentration_thin_regime_ignored() -> None:
    # n<5 in CHOP — too few trades to call; ignore.
    out = check_regime_concentration(
        {
            "analysis": {
                "regimes": {
                    "by_trend_regime": {
                        "TRENDING_UP": {"n": 50, "pnl": 10.0},
                        "CHOP": {"n": 3, "pnl": -0.5},
                    }
                }
            }
        }
    )
    assert out["passed"] is True


# ── portfolio_fit ─────────────────────────────────────────────────────
#
# Methodology (updated 2026-05-19): the gate measures duplication risk
# via the *max-positive* correlation with the protected book, not
# |corr|. Negative correlations are diversification benefits and must
# pass — prior |corr| semantics produced false REJECTs on legitimate
# hedges (iter 88b4c6a3 was the first case to surface this drift).
# Operator-escalation journal c58a4b8a documents the diagnosis.


def test_portfolio_fit_low_positive_corr_passes() -> None:
    out = check_portfolio_fit(
        {
            "portfolio_corr": {
                "applied": True,
                "per_code_corr": {"VBO": 0.3, "VCB": 0.1, "LSR": -0.1},
            }
        }
    )
    assert out["passed"] is True
    assert out["details"]["max_positive_corr"] == 0.3
    assert out["details"]["most_positive_code"] == "VBO"


def test_portfolio_fit_high_positive_corr_warns() -> None:
    out = check_portfolio_fit(
        {
            "portfolio_corr": {
                "applied": True,
                "per_code_corr": {"VBO": 0.7, "VCB": 0.2, "LSR": 0.1},
            }
        }
    )
    assert out["passed"] is False
    assert out["details"]["max_positive_corr"] == 0.7
    assert out["details"]["most_positive_code"] == "VBO"


def test_portfolio_fit_high_negative_corr_passes_as_diversifier() -> None:
    """The bug fix: a candidate with strongly NEGATIVE correlation
    against a book member is a diversifier (improves portfolio Sharpe),
    not a duplicate. It must pass the portfolio_fit check.

    Before the 2026-05-19 fix, this case was rejected because the gate
    used |corr|, treating -0.55 the same as +0.55. Iter 88b4c6a3 was
    the first candidate to expose this drift in production.
    """
    out = check_portfolio_fit(
        {
            "portfolio_corr": {
                "applied": True,
                "per_code_corr": {"VBO": 0.2, "VCB": -0.55, "LSR": -0.1},
            }
        }
    )
    assert out["passed"] is True, (
        "Negative correlations are diversification benefits and must pass "
        "the portfolio_fit check. If this assertion fires, the |corr| bug "
        "has regressed (see operator-escalation journal c58a4b8a)."
    )
    assert out["details"]["max_positive_corr"] == 0.2
    assert out["details"]["most_negative_corr"] == -0.55
    assert out["details"]["most_negative_code"] == "VCB"


def test_portfolio_fit_all_negative_corrs_passes_as_pure_diversifier() -> None:
    """All correlations non-positive → pure diversifier. Pass with a
    'no duplication risk' finding explicitly naming the most-negative
    code so the journal entry reads clearly."""
    out = check_portfolio_fit(
        {
            "portfolio_corr": {
                "applied": True,
                "per_code_corr": {"VBO": -0.1, "VCB": -0.43, "LSR": -0.2},
            }
        }
    )
    assert out["passed"] is True
    assert out["details"]["max_positive_corr"] == 0.0
    assert out["details"]["most_positive_code"] is None
    assert out["details"]["most_negative_code"] == "VCB"
    assert "pure diversifier" in out["finding"]


def test_portfolio_fit_mixed_high_pos_and_high_neg_warns_on_positive() -> None:
    """When a candidate has BOTH a high-positive correlation (vs one
    book member) AND a high-negative correlation (vs another), the
    positive duplication risk dominates — the hedge against the other
    member does NOT compensate for being a duplicate of the first."""
    out = check_portfolio_fit(
        {
            "portfolio_corr": {
                "applied": True,
                "per_code_corr": {"VBO": 0.65, "VCB": -0.7, "LSR": 0.1},
            }
        }
    )
    assert out["passed"] is False
    assert out["details"]["max_positive_corr"] == 0.65
    assert out["details"]["most_negative_corr"] == -0.7


def test_portfolio_fit_gate_skipped_passes_with_note() -> None:
    out = check_portfolio_fit(
        {"portfolio_corr": {"applied": False, "reason": "no baselines"}}
    )
    assert out["passed"] is True
    assert "manually inspect" in out["finding"].lower() or "manually" in out["finding"].lower()


def test_portfolio_fit_legacy_iteration_passes_with_note() -> None:
    """Iterations from before 2026-05-19 do not have per_code_corr
    surfaced consistently. The check must fall back gracefully — pass
    with an explicit note — rather than retroactively REJECT historical
    candidates on missing data."""
    out = check_portfolio_fit(
        {
            "portfolio_corr": {
                "applied": True,
                "max_abs_corr": 0.7,  # legacy field; no per_code_corr
            }
        }
    )
    assert out["passed"] is True
    finding = out["finding"].lower()
    assert "legacy" in finding or "pre-2026-05-19" in finding


# ── param_robustness ──────────────────────────────────────────────────


def test_param_robustness_plateau_passes() -> None:
    # Optimum at (1.5, 25) with PF=1.5; neighbour (2.0, 25) at PF=1.3 (within 30%).
    history = [
        ({"ATR": "1.0", "RSI": "20"}, 1.0),
        ({"ATR": "1.5", "RSI": "20"}, 1.2),
        ({"ATR": "1.5", "RSI": "25"}, 1.5),  # optimum
        ({"ATR": "2.0", "RSI": "25"}, 1.3),  # Hamming-1 neighbour, within 30%
        ({"ATR": "2.5", "RSI": "30"}, 0.9),
    ]
    out = check_param_robustness(history[2][0], history)
    assert out["passed"] is True


def test_param_robustness_cliff_warns() -> None:
    # Optimum at (1.5, 25) with PF=2.0; all neighbours below 30% threshold.
    history = [
        ({"ATR": "1.0", "RSI": "20"}, 0.7),
        ({"ATR": "1.5", "RSI": "20"}, 0.8),  # neighbour, far below 1.4 threshold
        ({"ATR": "1.5", "RSI": "25"}, 2.0),  # optimum
        ({"ATR": "2.0", "RSI": "25"}, 0.9),  # neighbour, far below
    ]
    out = check_param_robustness(history[2][0], history)
    assert out["passed"] is False
    assert "cliff" in out["finding"].lower() or "overfit" in out["finding"].lower()


def test_param_robustness_no_neighbours_warns() -> None:
    # Optimum standalone — no Hamming-1 sweep neighbours.
    history = [({"ATR": "1.5", "RSI": "25"}, 2.0)]
    out = check_param_robustness(history[0][0], history)
    assert out["passed"] is False


def test_param_robustness_empty_history_passes_neutral() -> None:
    out = check_param_robustness({}, [])
    assert out["passed"] is True


# ── aggregate_verdict ─────────────────────────────────────────────────


def _check_record(
    name: str, severity: str, passed: bool
) -> dict:
    return {
        "check_name": name,
        "severity": severity,
        "passed": passed,
        "finding": "test fixture",
        "details": {},
    }


def test_aggregate_no_failures_approves() -> None:
    out = aggregate_verdict([
        _check_record("a", "blocker", True),
        _check_record("b", "warning", True),
    ])
    assert out["verdict"] == "APPROVED"


def test_aggregate_one_blocker_fail_rejects() -> None:
    out = aggregate_verdict([
        _check_record("a", "blocker", False),
        _check_record("b", "warning", True),
    ])
    assert out["verdict"] == "REJECTED"


def test_aggregate_one_warning_fail_conditional() -> None:
    out = aggregate_verdict([
        _check_record("a", "blocker", True),
        _check_record("b", "warning", False),
    ])
    assert out["verdict"] == "CONDITIONAL_APPROVAL"


def test_aggregate_two_warning_fails_rejects() -> None:
    out = aggregate_verdict([
        _check_record("a", "blocker", True),
        _check_record("b", "warning", False),
        _check_record("c", "warning", False),
    ])
    assert out["verdict"] == "REJECTED"


def test_aggregate_blocker_dominates_warnings() -> None:
    # A blocker fail rejects regardless of how many warnings pass.
    out = aggregate_verdict([
        _check_record("a", "blocker", False),
        _check_record("b", "warning", True),
        _check_record("c", "warning", True),
        _check_record("d", "info", True),
    ])
    assert out["verdict"] == "REJECTED"


# ── End-to-end checklist runs ─────────────────────────────────────────


def test_plan_review_checklist_full_pass() -> None:
    plan_t = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    h = {
        "journal_id": "h1",
        "created_time": plan_t - timedelta(hours=1),
        "content": (
            "Tighter ATR-extreme + RSI-extreme raise MFE/MAE because "
            "overshoot magnitude in low-vol regimes correlates with "
            "snap-back probability."
        ),
    }
    checks = plan_review_checklist(
        hypothesis_entry=h,
        plan_created_time=plan_t,
        proposed_axis_names=["ATR_EXT", "RSI_EXT"],
        prior_strategy_outcomes=[],
    )
    out = aggregate_verdict(checks)
    assert out["verdict"] == "APPROVED"


def test_graduation_review_checklist_full_pass() -> None:
    iteration_metrics = {
        "analysis": {
            "n_trades": 250,
            "dsr": 0.97,
            "dsr_n_trials": 20,
            "slippage_haircut_pnl": {"+50bps": 8.0},
            "regimes": {
                "by_trend_regime": {
                    "TRENDING_UP": {"n": 100, "pnl": 15.0},
                    "CHOP": {"n": 100, "pnl": 5.0},
                }
            },
        },
        "portfolio_corr": {"applied": True, "max_abs_corr": 0.2, "per_code_corr": {}},
    }
    iteration_ci = {"pf_95": {"low": 1.15, "high": 1.6}}
    iteration_params = {"ATR": "1.5", "RSI": "25"}
    history = [
        ({"ATR": "1.0", "RSI": "25"}, 1.2),
        ({"ATR": "1.5", "RSI": "25"}, 1.5),
        ({"ATR": "2.0", "RSI": "25"}, 1.25),
    ]
    checks = graduation_review_checklist(
        iteration_metrics=iteration_metrics,
        iteration_ci=iteration_ci,
        iteration_params=iteration_params,
        sweep_history=history,
    )
    out = aggregate_verdict(checks)
    assert out["verdict"] == "APPROVED"


# ── target_id helpers ────────────────────────────────────────────────


def test_plan_target_id_is_axis_order_invariant() -> None:
    from orchestrator.repo.reviews import plan_target_id

    a = plan_target_id("MMR", ["ATR_EXT", "RSI_EXT"], "h-1")
    b = plan_target_id("MMR", ["RSI_EXT", "ATR_EXT"], "h-1")
    assert a == b


def test_plan_target_id_distinguishes_strategy() -> None:
    from orchestrator.repo.reviews import plan_target_id

    a = plan_target_id("MMR", ["X"], "h-1")
    b = plan_target_id("VCB", ["X"], "h-1")
    assert a != b


def test_plan_target_id_distinguishes_hypothesis() -> None:
    from orchestrator.repo.reviews import plan_target_id

    a = plan_target_id("MMR", ["X"], "h-1")
    b = plan_target_id("MMR", ["X"], "h-2")
    assert a != b


def test_graduation_target_id_format() -> None:
    from orchestrator.repo.reviews import graduation_target_id

    out = graduation_target_id("550e8400-e29b-41d4-a716-446655440000")
    assert out.startswith("graduation:")
    assert "550e8400-e29b-41d4-a716-446655440000" in out


def test_review_kind_constants_pinned() -> None:
    # If these values change, queries against existing rows break silently.
    from orchestrator.repo.reviews import KIND_REQUEST, KIND_VERDICT

    assert KIND_REQUEST == "review_request"
    assert KIND_VERDICT == "review_verdict"


def test_graduation_review_one_warning_fail_is_conditional() -> None:
    # Set up: cost realism fails (negative +50bps), all else passes.
    iteration_metrics = {
        "analysis": {
            "n_trades": 250,
            "dsr": 0.97,
            "dsr_n_trials": 20,
            "slippage_haircut_pnl": {"+50bps": -2.0},  # fails
            "regimes": {
                "by_trend_regime": {
                    "TRENDING_UP": {"n": 100, "pnl": 15.0},
                    "CHOP": {"n": 100, "pnl": 5.0},
                }
            },
        },
        "portfolio_corr": {"applied": True, "max_abs_corr": 0.2, "per_code_corr": {}},
    }
    iteration_ci = {"pf_95": {"low": 1.15, "high": 1.6}}
    iteration_params = {"ATR": "1.5", "RSI": "25"}
    history = [
        ({"ATR": "1.0", "RSI": "25"}, 1.2),
        ({"ATR": "1.5", "RSI": "25"}, 1.5),
        ({"ATR": "2.0", "RSI": "25"}, 1.25),
    ]
    checks = graduation_review_checklist(
        iteration_metrics=iteration_metrics,
        iteration_ci=iteration_ci,
        iteration_params=iteration_params,
        sweep_history=history,
    )
    out = aggregate_verdict(checks)
    assert out["verdict"] == "CONDITIONAL_APPROVAL"
