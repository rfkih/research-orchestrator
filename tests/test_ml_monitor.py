"""Unit tests for the ``GET /ml/monitor`` health-derivation function.

Pure tests of :func:`orchestrator.api.ml_monitor.derive_health` and the
coverage-ratio helper. The endpoint itself + repo query are covered by
the integration suite (pytest-postgresql + real V66 migrations).
"""
from __future__ import annotations

from orchestrator.api.ml_monitor import (
    COVERAGE_AMBER_RATIO,
    COVERAGE_RED_RATIO,
    LAST_FIRE_AMBER_MULT,
    LAST_FIRE_RED_MULT,
    MODULATOR_AUC_AMBER,
    _coverage_ratio,
    _freshness_score,
    derive_health,
)


def _h(**overrides) -> tuple[str, str | None]:
    """Helper: build a healthy default kwarg set, then apply overrides."""
    base = dict(
        coverage_ratio=1.0,
        last_fire_age_seconds=60,        # 1 minute — well within 1h cadence
        expected_fire_seconds=3_600,
        walkforward_auc=0.55,
        purpose="modulator",
    )
    base.update(overrides)
    return derive_health(**base)


def test_green_when_everything_within_thresholds() -> None:
    health, reason = _h()
    assert health == "green"
    assert reason is None


def test_red_dominates_when_coverage_below_red_floor() -> None:
    """Coverage red beats every other condition — even if last-fire is
    also red, the coverage reason is reported first."""
    health, reason = _h(
        coverage_ratio=0.30,
        last_fire_age_seconds=20_000,   # also a red condition
    )
    assert health == "red"
    assert reason is not None
    assert "coverage_7d" in reason


def test_red_when_last_fire_exceeds_3x_expected() -> None:
    """1h cadence, 3.1× = 11_160s. Just over the line."""
    health, reason = _h(
        coverage_ratio=1.0,
        last_fire_age_seconds=11_160,
    )
    assert health == "red"
    assert reason is not None
    assert "last_fire" in reason


def test_amber_when_coverage_between_thresholds() -> None:
    health, reason = _h(coverage_ratio=0.75)
    assert health == "amber"
    assert reason is not None
    assert "coverage_7d" in reason


def test_amber_when_last_fire_between_1_5x_and_3x() -> None:
    health, reason = _h(last_fire_age_seconds=7_000)   # 1.94× 1h
    assert health == "amber"
    assert reason is not None
    assert "last_fire" in reason


def test_amber_when_modulator_auc_below_threshold() -> None:
    health, reason = _h(walkforward_auc=0.51)
    assert health == "amber"
    assert reason is not None
    assert "modulator_auc" in reason


def test_predictor_auc_below_modulator_floor_still_green() -> None:
    """The 0.52 AUC floor is modulator-specific. Predictor models are
    judged differently (eventually — Phase 2 lens), so for now they are
    NOT marked amber on AUC alone."""
    health, _ = _h(walkforward_auc=0.51, purpose="predictor")
    assert health == "green"


def test_unknown_interval_returns_amber_with_explicit_reason() -> None:
    health, reason = _h(expected_fire_seconds=None, last_fire_age_seconds=None)
    assert health == "amber"
    assert reason == "unknown_interval"


def test_no_signal_history_yet_does_not_dominate() -> None:
    """A freshly promoted signal with no firings yet should report
    amber-or-better — never red — purely on the basis of missing history.
    coverage_ratio=0 IS below COVERAGE_RED_RATIO, so it WILL fire red;
    this test pins that we surface a clear reason so the operator
    knows the fix is 'wait for the first fire' not 'something broke'."""
    health, reason = _h(coverage_ratio=0.0)
    assert health == "red"
    assert reason is not None and "coverage_7d" in reason


def test_coverage_ratio_known_interval() -> None:
    # 1h cadence over 7 days expects 168 fires
    assert _coverage_ratio(fires_7d=168, interval_seconds=3_600) == 1.0
    assert _coverage_ratio(fires_7d=84,  interval_seconds=3_600) == 0.5
    assert _coverage_ratio(fires_7d=0,   interval_seconds=3_600) == 0.0
    # over-fires (catchup_scan replay landed inside window) clamps at 1.0
    assert _coverage_ratio(fires_7d=500, interval_seconds=3_600) == 1.0


def test_coverage_ratio_unknown_interval_returns_none() -> None:
    assert _coverage_ratio(fires_7d=10, interval_seconds=None) is None
    assert _coverage_ratio(fires_7d=10, interval_seconds=0) is None


def test_freshness_score_full_when_feature_fresh() -> None:
    """Feature age < 1 interval = fully fresh (1.0)."""
    # 1h interval, 30m old = fresh
    assert _freshness_score(feature_age_hours=0.5, interval_seconds=3_600) == 1.0


def test_freshness_score_partial_when_1_to_2_intervals_old() -> None:
    """Feature age between 1-2 intervals = partially stale (0.5)."""
    # 1h interval, 1.5h old = partial
    assert _freshness_score(feature_age_hours=1.5, interval_seconds=3_600) == 0.5


def test_freshness_score_zero_when_more_than_2_intervals_old() -> None:
    """Feature age > 2 intervals = stale (0.0)."""
    # 1h interval, 2.5h old = stale
    assert _freshness_score(feature_age_hours=2.5, interval_seconds=3_600) == 0.0


def test_freshness_score_full_when_no_age_or_interval() -> None:
    """Unknown age or interval = assume fresh (1.0)."""
    assert _freshness_score(feature_age_hours=None, interval_seconds=3_600) == 1.0
    assert _freshness_score(feature_age_hours=1.0, interval_seconds=None) == 1.0
    assert _freshness_score(feature_age_hours=None, interval_seconds=None) == 1.0


def test_freshness_score_edge_case_exactly_1_interval() -> None:
    """Feature age = exactly 1 interval should be at boundary (0.5)."""
    assert _freshness_score(feature_age_hours=1.0, interval_seconds=3_600) == 0.5


def test_freshness_score_edge_case_exactly_2_intervals() -> None:
    """Feature age = exactly 2 intervals should be at boundary (0.0)."""
    assert _freshness_score(feature_age_hours=2.0, interval_seconds=3_600) == 0.0


def test_freshness_score_with_5m_interval() -> None:
    """Verify freshness calculation works with different intervals."""
    # 5m = 300s, so 1 interval = 5 min
    # 3 min old (0.05h) = fresh
    assert _freshness_score(feature_age_hours=0.05, interval_seconds=300) == 1.0
    # 7.5 min old (0.125h) = partial
    assert _freshness_score(feature_age_hours=0.125, interval_seconds=300) == 0.5
    # 12 min old (0.2h) = stale
    assert _freshness_score(feature_age_hours=0.2, interval_seconds=300) == 0.0


def test_constants_pinned() -> None:
    """Operator-controlled thresholds. Changing here without updating
    the playbook + dashboard would silently shift which signals get the
    red pill — make it loud."""
    assert COVERAGE_RED_RATIO   == 0.50
    assert COVERAGE_AMBER_RATIO == 0.90
    assert LAST_FIRE_RED_MULT   == 3.0
    assert LAST_FIRE_AMBER_MULT == 1.5
    assert MODULATOR_AUC_AMBER  == 0.52
