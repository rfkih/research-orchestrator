"""Cross-window verdict regime-coverage tests (H4 fix, 2026-06-12).

Pure-function tests of ``aggregate_windows`` + ``cross_window_verdict``:
ROBUST_CROSS_WINDOW must be unreachable when the BEAR regime is absent
from the *eligible* (non-thin, non-errored) evidence — the gate exists to
catch bull-riders, and thin/errored bear windows used to silently improve
the verdict by dropping out of both sides of the percentage.
"""

from __future__ import annotations

from orchestrator.services.cross_window import (
    MIN_TRADES_PER_WINDOW,
    aggregate_windows,
    cross_window_verdict,
)


def _window(label: str, tag: str, *, trades: int, slip_net: float,
            error: str | None = None) -> dict:
    w = {
        "label": label,
        "start": "2023-01-01",
        "end": "2024-01-01",
        "regime_tag": tag,
        "run_id": "r",
        "trades": trades,
        "pf": 1.4,
        "return_pct": 10.0,
        "slip_20bps_net": slip_net,
    }
    if error:
        w["error"] = error
        w.pop("trades")
    return w


def test_robust_requires_eligible_bear_window():
    # Bull-rider scenario: bear window thin (5 trades — excluded from both
    # sides of pct_pos), four non-bear windows positive → 100% positive.
    # Pre-fix this scored ROBUST_CROSS_WINDOW; now capped.
    results = [
        _window("2022-bear", "BEAR", trades=5, slip_net=100.0),
        _window("2023-chop", "CHOP", trades=60, slip_net=50.0),
        _window("2023-transition", "TRANSITION", trades=60, slip_net=50.0),
        _window("2024-bull", "BULL", trades=60, slip_net=50.0),
        _window("2024H2-bull", "BULL", trades=60, slip_net=50.0),
    ]
    agg = aggregate_windows(results)
    assert agg["pct_windows_net_positive"] == 100.0
    assert "BEAR" not in agg["eligible_regime_tags"]
    assert "BEAR" in agg["excluded_regime_tags"]
    assert cross_window_verdict(agg) == "INSUFFICIENT_DATA_IN_WINDOW"


def test_errored_bear_window_also_blocks_robust():
    # A JVM failure on the hard window must not improve the verdict.
    results = [
        _window("2022-bear", "BEAR", trades=0, slip_net=0.0,
                error="exception:TimeoutError"),
        _window("2023-chop", "CHOP", trades=60, slip_net=50.0),
        _window("2023-transition", "TRANSITION", trades=60, slip_net=50.0),
        _window("2024-bull", "BULL", trades=60, slip_net=50.0),
    ]
    agg = aggregate_windows(results)
    assert cross_window_verdict(agg) == "INSUFFICIENT_DATA_IN_WINDOW"


def test_robust_granted_with_eligible_bear_coverage():
    results = [
        _window("2022-bear", "BEAR", trades=60, slip_net=20.0),
        _window("2023-chop", "CHOP", trades=60, slip_net=50.0),
        _window("2023-transition", "TRANSITION", trades=60, slip_net=50.0),
        _window("2024-bull", "BULL", trades=60, slip_net=50.0),
    ]
    agg = aggregate_windows(results)
    assert "BEAR" in agg["eligible_regime_tags"]
    assert cross_window_verdict(agg) == "ROBUST_CROSS_WINDOW"


def test_negative_windows_still_grade_below_robust():
    # Coverage requirement must not rescue a weak result: bear eligible but
    # half the windows negative (pct_pos = 50 → below the 80% ROBUST bar).
    results = [
        _window("2022-bear", "BEAR", trades=60, slip_net=-20.0),
        _window("2023-chop", "CHOP", trades=60, slip_net=50.0),
        _window("2023-transition", "TRANSITION", trades=60, slip_net=-5.0),
        _window("2024-bull", "BULL", trades=60, slip_net=50.0),
    ]
    agg = aggregate_windows(results)
    assert cross_window_verdict(agg) == "INCONSISTENT_CROSS_WINDOW"


def test_thin_threshold_is_pinned():
    assert MIN_TRADES_PER_WINDOW == 30
