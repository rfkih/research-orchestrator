"""Phase E paired-delta tests.

Pure-function tests on the pairing / delta-math / bootstrap / verdict
logic, plus auth + Pydantic-validation shape for the endpoint. DB-
touching paths (the queue_id fetch joining iteration_log) belong in
an integration suite.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from orchestrator.config import Settings
from orchestrator.main import create_app
from orchestrator.services.paired_delta import (
    DEFAULT_BOOTSTRAP_REPS,
    MIN_PAIRS_FOR_VERDICT,
    NEUTRAL_MAGNITUDE_THRESHOLD,
    _compute_delta_records,
    _compute_deltas,
    _decide_verdict,
    _pair_iterations,
    _paired_bootstrap_ci,
    _stable_key,
)


@pytest.fixture
def lenient_client(settings: Settings) -> TestClient:
    app = create_app(settings)
    return TestClient(app, raise_server_exceptions=False)


# ── Constants pinned ────────────────────────────────────────────────


def test_pinned_constants() -> None:
    # MIN_PAIRS raised 3 → 8 (2026-06-12): at n=3 the percentile bootstrap
    # had a 12.5%-per-direction false-positive rate (all-same-sign deltas
    # always exclude zero) on a verdict that can block graduation.
    assert MIN_PAIRS_FOR_VERDICT == 8
    assert DEFAULT_BOOTSTRAP_REPS == 1000
    assert NEUTRAL_MAGNITUDE_THRESHOLD == 0.05


# ── _stable_key (deterministic regardless of dict order) ────────────


def test_stable_key_is_order_insensitive() -> None:
    a = _stable_key({"rsi": 0.7, "lookback": 20})
    b = _stable_key({"lookback": 20, "rsi": 0.7})
    assert a == b


def test_stable_key_distinguishes_values() -> None:
    a = _stable_key({"rsi": 0.7})
    b = _stable_key({"rsi": 0.8})
    assert a != b


# ── _pair_iterations ────────────────────────────────────────────────


def _row(iteration_id: str, params: dict, sharpe: float) -> dict:
    return {
        "iteration_id": iteration_id,
        "params_snapshot": params,
        "metrics_snapshot": {"sharpe_ratio": sharpe},
    }


def test_pair_iterations_matches_clean_pair() -> None:
    rows = [
        _row("11", {"rsi": 0.7, "_ml_gate_enabled": True}, 1.5),
        _row("22", {"rsi": 0.7, "_ml_gate_enabled": False}, 1.0),
    ]
    pairs, unpaired = _pair_iterations(
        rows,
        treatment_key="_ml_gate_enabled",
        on_value=True,
        off_value=False,
    )
    assert len(pairs) == 1
    assert unpaired == []
    key, on_row, off_row = pairs[0]
    assert on_row["iteration_id"] == "11"
    assert off_row["iteration_id"] == "22"
    assert "rsi=0.7" in key


def test_pair_iterations_drops_lone_on_or_off() -> None:
    rows = [
        _row("11", {"rsi": 0.7, "_ml_gate_enabled": True}, 1.5),
        # no matching off-row → unpaired
        _row("22", {"rsi": 0.8, "_ml_gate_enabled": False}, 1.0),
    ]
    pairs, unpaired = _pair_iterations(
        rows,
        treatment_key="_ml_gate_enabled",
        on_value=True,
        off_value=False,
    )
    assert pairs == []
    assert len(unpaired) == 2  # both groups had a singleton


def test_pair_iterations_strips_all_ml_sentinels_from_key() -> None:
    """Pairing must hold ALL non-ML params constant — not just non-
    treatment ones. A pair that varies on _ml_signal_name while we're
    treating on _ml_gate_enabled would otherwise pair incorrectly."""
    rows = [
        _row(
            "11",
            {
                "rsi": 0.7,
                "_ml_gate_enabled": True,
                "_ml_signal_name": "regime_btc_v3",
            },
            1.5,
        ),
        _row(
            "22",
            {
                "rsi": 0.7,
                "_ml_gate_enabled": False,
                "_ml_signal_name": "regime_btc_v4",
            },
            1.0,
        ),
    ]
    pairs, _ = _pair_iterations(
        rows,
        treatment_key="_ml_gate_enabled",
        on_value=True,
        off_value=False,
    )
    # Both rows have rsi=0.7 (the only non-ML param), so they pair.
    # The differing _ml_signal_name is correctly stripped.
    assert len(pairs) == 1


def test_pair_iterations_distinguishes_non_ml_params() -> None:
    rows = [
        _row("11", {"rsi": 0.7, "_ml_gate_enabled": True}, 1.5),
        _row("22", {"rsi": 0.8, "_ml_gate_enabled": False}, 1.0),
        _row("33", {"rsi": 0.7, "_ml_gate_enabled": False}, 1.1),
        _row("44", {"rsi": 0.8, "_ml_gate_enabled": True}, 1.4),
    ]
    pairs, unpaired = _pair_iterations(
        rows,
        treatment_key="_ml_gate_enabled",
        on_value=True,
        off_value=False,
    )
    # 11+33 (rsi=0.7) and 44+22 (rsi=0.8) form pairs.
    assert len(pairs) == 2
    assert unpaired == []


def test_pair_iterations_ignores_rows_with_other_treatment_values() -> None:
    rows = [
        _row("11", {"rsi": 0.7, "_ml_gate_enabled": True}, 1.5),
        _row("22", {"rsi": 0.7, "_ml_gate_enabled": False}, 1.0),
        _row("33", {"rsi": 0.7, "_ml_gate_enabled": "weird"}, 1.2),
    ]
    pairs, unpaired = _pair_iterations(
        rows,
        treatment_key="_ml_gate_enabled",
        on_value=True,
        off_value=False,
    )
    assert len(pairs) == 1


# ── _compute_deltas ─────────────────────────────────────────────────


def test_compute_deltas_simple() -> None:
    pairs = [
        ("rsi=0.7", _row("11", {}, 1.5), _row("22", {}, 1.0)),
        ("rsi=0.8", _row("33", {}, 1.2), _row("44", {}, 1.4)),
    ]
    deltas = _compute_deltas(pairs, primary_metric="sharpe")
    assert deltas == pytest.approx([0.5, -0.2])


def test_compute_deltas_skips_missing_metric() -> None:
    on_row = {"iteration_id": "11", "metrics_snapshot": {"sharpe_ratio": 1.5}}
    off_row = {"iteration_id": "22", "metrics_snapshot": {}}  # missing
    pairs = [("k", on_row, off_row)]
    assert _compute_deltas(pairs, primary_metric="sharpe") == []


def test_compute_delta_records_bundles_metadata() -> None:
    """F3 regression — each delta carries its pair's metadata so the
    endpoint response can't drift out of sync with the pair list."""
    pairs = [
        ("rsi=0.7", _row("11", {}, 1.5), _row("22", {}, 1.0)),
        ("rsi=0.8", _row("33", {}, 1.2), _row("44", {}, 1.4)),
    ]
    records = _compute_delta_records(pairs, primary_metric="sharpe")
    assert len(records) == 2
    assert records[0]["pair_key"] == "rsi=0.7"
    assert records[0]["on_iteration_id"] == "11"
    assert records[0]["off_iteration_id"] == "22"
    assert records[0]["delta"] == pytest.approx(0.5)
    assert records[1]["delta"] == pytest.approx(-0.2)


def test_compute_delta_records_drops_pair_with_missing_metric() -> None:
    """F3 regression — a pair where one side lacks the metric is
    silently dropped (not in records). The endpoint's n_pairs ends up
    < len(pairs) but that's surfaced via n_dropped_missing_metric."""
    pairs = [
        ("rsi=0.7", _row("11", {}, 1.5), _row("22", {}, 1.0)),
        # missing metrics on the off side — dropped
        (
            "rsi=0.8",
            _row("33", {}, 1.2),
            {"iteration_id": "44", "metrics_snapshot": {}},
        ),
    ]
    records = _compute_delta_records(pairs, primary_metric="sharpe")
    assert len(records) == 1
    assert records[0]["pair_key"] == "rsi=0.7"


# ── _paired_bootstrap_ci ────────────────────────────────────────────


def test_bootstrap_ci_brackets_mean_for_consistent_signal() -> None:
    """A strongly positive set of deltas should bootstrap to a CI that
    sits entirely above zero (the test exercises the seed determinism
    + the quantile mechanics)."""
    deltas = [0.4, 0.5, 0.6, 0.45, 0.55, 0.5, 0.5, 0.5]
    lo, hi = _paired_bootstrap_ci(deltas, n_reps=1000, ci_level=0.95, seed=42)
    assert lo > 0
    assert hi > 0
    assert lo < sum(deltas) / len(deltas) < hi


def test_bootstrap_ci_straddles_zero_for_mixed() -> None:
    deltas = [-0.4, -0.3, 0.5, 0.6, -0.1, 0.0, 0.2, -0.5]
    lo, hi = _paired_bootstrap_ci(deltas, n_reps=1000, ci_level=0.95, seed=42)
    assert lo < 0
    assert hi > 0


def test_bootstrap_ci_deterministic_on_seed() -> None:
    deltas = [0.4, 0.5, 0.6, 0.45, 0.55]
    lo1, hi1 = _paired_bootstrap_ci(deltas, n_reps=500, ci_level=0.95, seed=42)
    lo2, hi2 = _paired_bootstrap_ci(deltas, n_reps=500, ci_level=0.95, seed=42)
    assert lo1 == lo2
    assert hi1 == hi2


# ── _decide_verdict ─────────────────────────────────────────────────


def test_verdict_positive_when_ci_above_zero() -> None:
    out = _decide_verdict(mean=0.4, ci_low=0.2, ci_high=0.6, primary_metric="sharpe")
    assert out == "POSITIVE_DELTA"


def test_verdict_negative_when_ci_below_zero() -> None:
    out = _decide_verdict(mean=-0.5, ci_low=-0.8, ci_high=-0.2, primary_metric="sharpe")
    assert out == "NEGATIVE_DELTA"


def test_verdict_neutral_when_ci_straddles_with_small_mean() -> None:
    out = _decide_verdict(mean=0.02, ci_low=-0.1, ci_high=0.15, primary_metric="sharpe")
    assert out == "NEUTRAL"


def test_verdict_indeterminate_when_wide_ci_with_large_mean() -> None:
    out = _decide_verdict(mean=0.4, ci_low=-0.2, ci_high=1.0, primary_metric="sharpe")
    assert out == "INDETERMINATE"


def test_verdict_trade_count_always_neutral() -> None:
    """trade_count has no good/bad direction — gate suppressing trades
    is descriptive, not a value verdict. Always NEUTRAL."""
    out = _decide_verdict(
        mean=-100, ci_low=-200, ci_high=-50, primary_metric="trade_count"
    )
    assert out == "NEUTRAL"


# ── API auth + validation ──────────────────────────────────────────


def test_paired_delta_requires_token(client: TestClient) -> None:
    r = client.post(
        "/paired-delta",
        json={"queue_id": "00000000-0000-0000-0000-000000000001"},
    )
    assert r.status_code == 401


def test_paired_delta_requires_exactly_one_target(lenient_client: TestClient) -> None:
    """@model_validator surfaces both 'both set' and 'neither set'. In
    production these become 422 validation_failed (F2 fix — was 500
    with model_post_init). In the no-DB test fixture, FastAPI resolves
    Depends(get_db_conn) before body validation, so we see 500 here."""
    r1 = lenient_client.post(
        "/paired-delta",
        headers={"X-Orch-Token": "test-token"},
        json={
            "queue_id": "00000000-0000-0000-0000-000000000001",
            "iteration_ids": [
                "00000000-0000-0000-0000-000000000010",
                "00000000-0000-0000-0000-000000000011",
            ],
        },
    )
    assert r1.status_code >= 400
    assert r1.json()["error_code"] in ("validation_failed", "internal_error")

    r2 = lenient_client.post(
        "/paired-delta",
        headers={"X-Orch-Token": "test-token"},
        json={},
    )
    assert r2.status_code >= 400
    assert r2.json()["error_code"] in ("validation_failed", "internal_error")


def test_paired_delta_rejects_bad_metric(lenient_client: TestClient) -> None:
    r = lenient_client.post(
        "/paired-delta",
        headers={"X-Orch-Token": "test-token"},
        json={
            "queue_id": "00000000-0000-0000-0000-000000000001",
            "primary_metric": "max_drawdown",  # not in the Literal
        },
    )
    assert r.status_code >= 400
    assert r.json()["error_code"] in ("validation_failed", "internal_error")


def test_paired_delta_rejects_bad_treatment_key(lenient_client: TestClient) -> None:
    r = lenient_client.post(
        "/paired-delta",
        headers={"X-Orch-Token": "test-token"},
        json={
            "queue_id": "00000000-0000-0000-0000-000000000001",
            "treatment_key": "rsi_threshold",  # not an ML sentinel
        },
    )
    assert r.status_code >= 400


def test_paired_delta_body_validator_rejects_both_targets() -> None:
    """Pure-Pydantic check that the @model_validator fires — bypasses
    FastAPI's Depends-before-body ordering and proves the validator
    actually catches the case (regardless of test-fixture quirks)."""
    from pydantic import ValidationError

    from orchestrator.api.specialists import PairedDeltaBody

    with pytest.raises(ValidationError):
        PairedDeltaBody(
            queue_id="00000000-0000-0000-0000-000000000001",
            iteration_ids=[
                "00000000-0000-0000-0000-000000000010",
                "00000000-0000-0000-0000-000000000011",
            ],
        )

    with pytest.raises(ValidationError):
        PairedDeltaBody()


def test_paired_delta_rejects_short_iteration_list(lenient_client: TestClient) -> None:
    """min_length=2 — a paired comparison requires at least 2 rows."""
    r = lenient_client.post(
        "/paired-delta",
        headers={"X-Orch-Token": "test-token"},
        json={
            "iteration_ids": ["00000000-0000-0000-0000-000000000010"],
        },
    )
    assert r.status_code >= 400
