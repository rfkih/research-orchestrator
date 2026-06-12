"""Unit tests for the capacity-sweep verdict classifier.

Pure-function tests of ``_classify_verdict`` — no DB, no JVM. The
full endpoint (``POST /capacity-sweep``) is integration-tested
separately via the running stack.
"""
from __future__ import annotations

import pytest

from datetime import datetime, timezone

from orchestrator.services.capacity import (
    DEFAULT_SLIPPAGE_RATE,
    DEGRADATION_PCT_GENTLE,
    DEGRADATION_PCT_GRADUAL,
    PF_NOISE_FLOOR,
    _annualize_geom,
    _build_payload,
    _classify_verdict,
    _edge_at,
    _edge_at_floor,
    _provisional_judge_verdict,
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
    assert DEFAULT_SLIPPAGE_RATE == 0.0005


def _payload_kwargs(**extra) -> dict:
    base = dict(
        account_strategy_id="as-1",
        strategy_code="DCB",
        asset="ETHUSDT",
        interval_name="1h",
        start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2026, 5, 20, tzinfo=timezone.utc),
        initial_capital=10_000.0,
    )
    base.update(extra)
    return base


def test_build_payload_routes_ml_sentinels_to_v100_maps() -> None:
    """ML sentinels in ``overrides`` must land in the V100 JVM JSONB
    columns (``strategyMl*Overrides``), NOT inside
    ``strategyParamOverrides``. The JVM ignores them under the latter."""
    payload = _build_payload(
        **_payload_kwargs(
            overrides={
                "tpR": 5.0,
                "adxEntryMin": 21,
                "_ml_gate_enabled": True,
                "_ml_signal_name": "regime_eth_v2",
                "_ml_shadow_mode": False,
            },
        )
    )
    assert payload["strategyParamOverrides"] == {
        "DCB": {"tpR": 5.0, "adxEntryMin": 21}
    }
    assert payload["strategyMlGateOverrides"] == {"DCB": True}
    assert payload["strategyMlSignalNameOverrides"] == {"DCB": "regime_eth_v2"}
    assert payload["strategyMlShadowModeOverrides"] == {"DCB": False}


def test_build_payload_emits_slippage_rate_default_and_override() -> None:
    """``slippageRate`` must always appear so the JVM doesn't fall back
    to the null-guard zero-impact path. Both default and explicit
    override must reach the payload."""
    default_payload = _build_payload(**_payload_kwargs(overrides=None))
    assert default_payload["slippageRate"] == DEFAULT_SLIPPAGE_RATE

    custom_payload = _build_payload(
        **_payload_kwargs(overrides=None, slippage_rate=0.001)
    )
    assert custom_payload["slippageRate"] == 0.001


def test_build_payload_omits_ml_keys_when_no_sentinels() -> None:
    """If no ML sentinels are present, the V100 maps must NOT appear in
    the payload — empty maps would pollute the JVM payload with three
    null routes that the strategy resolver has to ignore."""
    payload = _build_payload(
        **_payload_kwargs(overrides={"tpR": 5.0, "adxEntryMin": 21})
    )
    assert "strategyMlGateOverrides" not in payload
    assert "strategyMlSignalNameOverrides" not in payload
    assert "strategyMlShadowModeOverrides" not in payload


def test_build_payload_enables_market_impact():
    # The capacity lever (V143): without marketImpactEnabled every tier returns
    # identical metrics (flat slippage is size-invariant) → meaningless sweep.
    payload = _build_payload(**_payload_kwargs(overrides=None))
    assert payload["marketImpactEnabled"] is True


def test_annualize_geom():
    assert _annualize_geom(100.0, 365.0) == pytest.approx(100.0, abs=1e-6)
    assert _annualize_geom(21.0, 182.5) == pytest.approx(46.41, abs=0.2)
    assert _annualize_geom(None, 365.0) is None
    assert _annualize_geom(10.0, 0.0) is None
    assert _annualize_geom(-100.0, 365.0) == -100.0  # ruin


def test_provisional_judge_verdict():
    floor = 10_000.0
    assert _provisional_judge_verdict("LINEAR_SCALABLE", None, floor) == "REJECT"
    assert _provisional_judge_verdict("LINEAR_SCALABLE", 5_000.0, floor) == "REJECT"
    assert _provisional_judge_verdict("GRADUAL_DECAY", 20_000.0, floor) == "CONCERN"
    assert _provisional_judge_verdict("SHARP_CLIFF", 1_000_000.0, floor) == "CONCERN"
    assert _provisional_judge_verdict("LINEAR_SCALABLE", 1_000_000.0, floor) == "PROCEED"


def test_edge_at():
    per_scale = [
        {"scale_usd": 10_000.0, "metrics": {"ann_geom_pct": 18.0}},
        {"scale_usd": 100_000.0, "metrics": {"ann_geom_pct": 12.0}},
    ]
    assert _edge_at(per_scale, 10_000.0) == 18.0
    assert _edge_at(per_scale, 100_000.0) == 12.0
    assert _edge_at(per_scale, 500_000.0) is None
    assert _edge_at(per_scale, None) is None


def test_edge_at_floor_handles_off_ladder_floor():
    per_scale = [
        {"scale_usd": 1_000.0, "metrics": {"ann_geom_pct": 25.0}},
        {"scale_usd": 10_000.0, "metrics": {"ann_geom_pct": 18.0}},
        {"scale_usd": 100_000.0, "metrics": {"ann_geom_pct": 9.0}},
    ]
    # Floor exactly on a scale → that scale.
    assert _edge_at_floor(per_scale, 10_000.0) == 18.0
    # Floor between scales → largest tested scale <= floor (deployable-at-floor).
    assert _edge_at_floor(per_scale, 50_000.0) == 18.0
    # Floor below all scales → smallest tested scale (fallback).
    assert _edge_at_floor(per_scale, 100.0) == 25.0
    # Floor above all scales → largest <= floor.
    assert _edge_at_floor(per_scale, 1_000_000.0) == 9.0
    assert _edge_at_floor(per_scale, None) is None
    # Errored scales (no metrics) are skipped.
    errored = [{"scale_usd": 10_000.0, "error": "status=FAILED"}]
    assert _edge_at_floor(errored, 10_000.0) is None


# ── 2026-06-12 hardening: fail-open + interior-cliff + ordering ───────────────


def test_classify_single_valid_scale_is_insufficient_data() -> None:
    # One surviving run (base tier errored) used to return the STRONGEST
    # verdict — LINEAR_SCALABLE with max_viable at the surviving tier.
    per_scale = [
        {"scale_usd": 100.0, "error": "status=FAILED"},
        {"scale_usd": 1_000_000.0, "metrics": {"pf": 1.8}},
    ]
    verdict, max_viable = _classify_verdict(per_scale)
    assert verdict == "INSUFFICIENT_DATA"
    assert max_viable is None


def test_classify_max_viable_does_not_jump_interior_cliff() -> None:
    # PF recovers above the floor after an interior collapse; the cap must
    # stay at the last scale BEFORE the collapse ("the prior scale is the
    # cap"), not jump to the recovered tier.
    per_scale = [
        {"scale_usd": 100.0, "metrics": {"pf": 1.5}},
        {"scale_usd": 10_000.0, "metrics": {"pf": 0.8}},
        {"scale_usd": 1_000_000.0, "metrics": {"pf": 1.3}},
    ]
    verdict, max_viable = _classify_verdict(per_scale)
    assert verdict == "SHARP_CLIFF"
    assert max_viable == 100.0


def test_classify_sorts_unordered_scales() -> None:
    # Descending input must not make the $1M tier the "base".
    per_scale = [
        {"scale_usd": 1_000_000.0, "metrics": {"pf": 1.0}},
        {"scale_usd": 100.0, "metrics": {"pf": 1.6}},
    ]
    verdict, max_viable = _classify_verdict(per_scale)
    # Decay from 1.6 → 1.0 is 37.5% → GRADUAL_DECAY, capped at the base.
    assert verdict == "GRADUAL_DECAY"
    assert max_viable == 100.0


def test_classify_base_below_noise_floor_has_no_viable_capital() -> None:
    # Base PF in [1.0, noise_floor): not INVERTED, but nothing is viable.
    per_scale = [
        {"scale_usd": 100.0, "metrics": {"pf": 1.1}},
        {"scale_usd": 10_000.0, "metrics": {"pf": 1.05}},
    ]
    _, max_viable = _classify_verdict(per_scale)
    assert max_viable is None
