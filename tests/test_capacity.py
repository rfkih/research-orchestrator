"""Unit tests for the capacity-sweep verdict classifier.

Pure-function tests of ``_classify_verdict`` — no DB, no JVM. The
full endpoint (``POST /capacity-sweep``) is integration-tested
separately via the running stack.
"""
from __future__ import annotations

import pytest

from orchestrator.services.capacity import (
    DEGRADATION_PCT_GENTLE,
    DEGRADATION_PCT_GRADUAL,
    PF_NOISE_FLOOR,
    _classify_verdict,
)


def _scale(idx: int, scale_usd: float, pf: float | None) -> dict:
    """Build a per-scale result row."""
    if pf is None:
        return {
            "scale_index": idx,
            "scale_usd": scale_usd,
            "run_id": "stub",
            "error": "status=FAILED",
        }
    return {
        "scale_index": idx,
        "scale_usd": scale_usd,
        "run_id": "stub",
        "metrics": {"pf": pf, "return_pct": 0.0, "sharpe": 0.0},
    }


def test_classify_linear_scalable_when_decay_under_gentle_threshold() -> None:
    """Base PF 1.5, all higher scales within 20% (i.e., PF >= 1.2)."""
    per_scale = [
        _scale(1, 100, 1.50),
        _scale(2, 1_000, 1.45),
        _scale(3, 10_000, 1.42),
        _scale(4, 100_000, 1.40),
        _scale(5, 1_000_000, 1.35),
    ]
    verdict, max_viable = _classify_verdict(per_scale)
    assert verdict == "LINEAR_SCALABLE"
    assert max_viable == 1_000_000


def test_classify_gradual_decay_when_between_thresholds() -> None:
    """Base PF 1.5, biggest scale at PF 0.95 — 36.7% decay."""
    per_scale = [
        _scale(1, 100, 1.50),
        _scale(2, 1_000, 1.35),
        _scale(3, 10_000, 1.20),
        _scale(4, 100_000, 1.05),
        _scale(5, 1_000_000, 0.95),
    ]
    verdict, max_viable = _classify_verdict(per_scale)
    assert verdict == "GRADUAL_DECAY"
    # PF >= noise floor (1.2) at scales 100, 1k, 10k → max viable = 10k
    assert max_viable == 10_000


def test_classify_sharp_cliff_when_one_scale_collapses() -> None:
    """Base PF 1.6, but at 1M scale PF drops to 0.7 — 56% decay."""
    per_scale = [
        _scale(1, 100, 1.60),
        _scale(2, 1_000, 1.55),
        _scale(3, 10_000, 1.50),
        _scale(4, 100_000, 1.40),
        _scale(5, 1_000_000, 0.70),  # cliff
    ]
    verdict, max_viable = _classify_verdict(per_scale)
    assert verdict == "SHARP_CLIFF"
    assert max_viable == 100_000


def test_classify_inverted_when_base_pf_below_one() -> None:
    """If even the smallest scale produces PF < 1.0, the strategy has
    no edge to scale."""
    per_scale = [
        _scale(1, 100, 0.85),
        _scale(2, 1_000, 0.80),
        _scale(3, 10_000, 0.78),
    ]
    verdict, max_viable = _classify_verdict(per_scale)
    assert verdict == "INVERTED"
    assert max_viable is None


def test_classify_insufficient_data_when_all_scales_errored() -> None:
    per_scale = [
        _scale(1, 100, None),
        _scale(2, 1_000, None),
    ]
    verdict, max_viable = _classify_verdict(per_scale)
    assert verdict == "INSUFFICIENT_DATA"
    assert max_viable is None


def test_classify_max_viable_stops_at_first_below_floor() -> None:
    """Even when some larger scale recovers above the noise floor, the
    contiguous-decline max_viable still finds the last scale where PF
    is above the floor."""
    per_scale = [
        _scale(1, 100, 1.50),
        _scale(2, 1_000, 1.30),     # above floor (1.2)
        _scale(3, 10_000, 1.15),    # below floor
        _scale(4, 100_000, 1.05),
    ]
    verdict, max_viable = _classify_verdict(per_scale)
    # 1.50 → 1.05 = 30% decay → GRADUAL_DECAY
    assert verdict == "GRADUAL_DECAY"
    # The classifier walks scales in order, updating max_viable each time
    # PF >= noise_floor. Last passing scale was 1k.
    assert max_viable == 1_000


def test_constants_pinned() -> None:
    """Operator-controlled thresholds — changing them shifts what
    capacity verdicts the gate emits. Pin so accidental drift is loud."""
    assert PF_NOISE_FLOOR == 1.2
    assert DEGRADATION_PCT_GENTLE == 20.0
    assert DEGRADATION_PCT_GRADUAL == 40.0
