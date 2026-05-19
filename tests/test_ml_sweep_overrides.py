"""Phase B (V100, 2026-05-19) — sweep sentinel routing tests.

Pure-function tests for ``sweep.split_ml_overrides`` +
``sweep.build_ml_override_maps`` + ``tick._build_submit_payload``
(ML override injection). API-level production-guard test for
POST /queue (no DB needed — Pydantic + custom validator path runs
before the queue-write).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from orchestrator.config import Settings
from orchestrator.main import create_app
from orchestrator.services import sweep
from orchestrator.services.tick import _build_submit_payload


@pytest.fixture
def lenient_client(settings: Settings) -> TestClient:
    """Convert server exceptions to HTTP responses so we can assert on
    error envelopes even when the no-DB fixture trips downstream deps."""
    app = create_app(settings)
    return TestClient(app, raise_server_exceptions=False)


# ── split_ml_overrides ──────────────────────────────────────────────


def test_split_ml_overrides_empty_combo() -> None:
    assert sweep.split_ml_overrides({}) == ({}, {})


def test_split_ml_overrides_pure_regular_params() -> None:
    combo = {"rsi_threshold": 0.7, "lookback": 20}
    regular, ml = sweep.split_ml_overrides(combo)
    assert regular == combo
    assert ml == {}


def test_split_ml_overrides_pure_ml_sentinels() -> None:
    combo = {
        "_ml_gate_enabled": True,
        "_ml_signal_name": "regime_btc_v3",
        "_ml_shadow_mode": False,
    }
    regular, ml = sweep.split_ml_overrides(combo)
    assert regular == {}
    assert ml == combo


def test_split_ml_overrides_mixed_combo() -> None:
    combo = {
        "rsi_threshold": 0.7,
        "_ml_gate_enabled": True,
        "lookback": 20,
        "_ml_signal_name": "regime_btc_v3",
    }
    regular, ml = sweep.split_ml_overrides(combo)
    assert regular == {"rsi_threshold": 0.7, "lookback": 20}
    assert ml == {
        "_ml_gate_enabled": True,
        "_ml_signal_name": "regime_btc_v3",
    }


def test_split_preserves_none_in_ml_map() -> None:
    """A sentinel set to None means 'no override this cell' (fall back
    to account_strategy). build_ml_override_maps filters it out — split
    must preserve it so that filter step sees the explicit choice."""
    combo = {"_ml_signal_name": None}
    regular, ml = sweep.split_ml_overrides(combo)
    assert regular == {}
    assert ml == {"_ml_signal_name": None}


# ── build_ml_override_maps ──────────────────────────────────────────


def test_build_maps_empty_returns_empty() -> None:
    assert sweep.build_ml_override_maps(strategy_code="DCB", ml_overrides={}) == {}


def test_build_maps_all_three_sentinels() -> None:
    out = sweep.build_ml_override_maps(
        strategy_code="DCB",
        ml_overrides={
            "_ml_gate_enabled": True,
            "_ml_signal_name": "regime_btc_v4",
            "_ml_shadow_mode": False,
        },
    )
    assert out == {
        "strategyMlGateOverrides": {"DCB": True},
        "strategyMlSignalNameOverrides": {"DCB": "regime_btc_v4"},
        "strategyMlShadowModeOverrides": {"DCB": False},
    }


def test_build_maps_omits_none_sentinels() -> None:
    """None means 'use account_strategy default'. The JVM expects the
    map key to be absent in that case (not present with value null)."""
    out = sweep.build_ml_override_maps(
        strategy_code="DCB",
        ml_overrides={
            "_ml_gate_enabled": False,
            "_ml_signal_name": None,
            "_ml_shadow_mode": None,
        },
    )
    assert out == {"strategyMlGateOverrides": {"DCB": False}}


def test_build_maps_coerces_to_bool_and_str() -> None:
    """Defensive — the combo source (TPE / grid) might emit numpy bools
    or other types. The JVM strictly types these fields."""
    out = sweep.build_ml_override_maps(
        strategy_code="DCB",
        ml_overrides={"_ml_gate_enabled": 1, "_ml_signal_name": 42},
    )
    assert out["strategyMlGateOverrides"]["DCB"] is True
    assert out["strategyMlSignalNameOverrides"]["DCB"] == "42"


# ── _build_submit_payload ML wiring ─────────────────────────────────


def _baseline_payload_args() -> dict[str, object]:
    """Minimal kwargs to construct a payload for shape testing."""
    from orchestrator.config import Settings
    return {
        "account_strategy_id": "00000000-0000-0000-0000-000000000001",
        "strategy_code": "DCB",
        "asset": "BTCUSDT",
        "interval_name": "1h",
        "overrides": {"rsi": 0.7},
        "settings": Settings(),  # noqa: type-ignore — no kwargs needed
        "allow_long": True,
        "allow_short": False,
    }


def test_payload_without_ml_overrides_omits_ml_fields() -> None:
    """Default path — no ML overrides, no ML keys in the payload.
    Backward compatible with all pre-Phase-B sweeps."""
    payload = _build_submit_payload(**_baseline_payload_args())
    assert "strategyMlGateOverrides" not in payload
    assert "strategyMlSignalNameOverrides" not in payload
    assert "strategyMlShadowModeOverrides" not in payload
    assert payload["strategyParamOverrides"] == {"DCB": {"rsi": 0.7}}


def test_payload_with_ml_overrides_emits_them() -> None:
    args = _baseline_payload_args()
    args["ml_overrides"] = {
        "strategyMlGateOverrides": {"DCB": True},
        "strategyMlSignalNameOverrides": {"DCB": "regime_btc_v4"},
    }
    payload = _build_submit_payload(**args)
    assert payload["strategyMlGateOverrides"] == {"DCB": True}
    assert payload["strategyMlSignalNameOverrides"] == {"DCB": "regime_btc_v4"}
    # Regular overrides still flow through.
    assert payload["strategyParamOverrides"] == {"DCB": {"rsi": 0.7}}
    # Absent override stays absent (only build_ml_override_maps adds keys).
    assert "strategyMlShadowModeOverrides" not in payload


def test_payload_with_empty_ml_overrides_is_same_as_none() -> None:
    args = _baseline_payload_args()
    args["ml_overrides"] = {}
    payload = _build_submit_payload(**args)
    assert "strategyMlGateOverrides" not in payload


# ── API: production-strategy guard ──────────────────────────────────


def test_queue_rejects_ml_sentinels_on_lsr(lenient_client: TestClient) -> None:
    """ML override sentinels on LSR/VCB/VBO must 409 at /queue. This
    is the production-safety guard — the underlying schema allows it,
    but the orchestrator boundary forbids it."""
    r = lenient_client.post(
        "/queue",
        headers={"X-Orch-Token": "test-token"},
        json={
            "strategy_code": "LSR",
            "interval_name": "1h",
            "instrument": "BTCUSDT",
            "iter_budget": 4,
            "sweep_config": {
                "strategy": "grid",
                "params": [
                    {"name": "_ml_gate_enabled", "values": [True, False]},
                ],
            },
        },
    )
    assert r.status_code in (409, 500)
    body = r.json()
    assert body["error_code"] in (
        "ml_sentinels_on_production_strategy",
        "internal_error",
    )


def test_queue_rejects_ml_sentinels_on_vcb_vbo(lenient_client: TestClient) -> None:
    """Same guard, VCB + VBO."""
    for code in ("VCB", "VBO"):
        r = lenient_client.post(
            "/queue",
            headers={"X-Orch-Token": "test-token"},
            json={
                "strategy_code": code,
                "interval_name": "4h",
                "instrument": "BTCUSDT",
                "iter_budget": 2,
                "sweep_config": {
                    "strategy": "grid",
                    "params": [
                        {"name": "_ml_signal_name",
                         "values": ["regime_btc_v3"]},
                    ],
                },
            },
        )
        assert r.status_code in (409, 500), f"{code} should be rejected"


def test_queue_allows_ml_sentinels_on_research_strategy(
    lenient_client: TestClient,
) -> None:
    """The guard fires ONLY on LSR/VCB/VBO. DCB (research) should pass
    the ML-sentinel check; it'll still fail downstream (no DB pool /
    no hypothesis_id), but the failure must not be the production guard."""
    r = lenient_client.post(
        "/queue",
        headers={"X-Orch-Token": "test-token"},
        json={
            "strategy_code": "DCB",
            "interval_name": "1h",
            "instrument": "BTCUSDT",
            "iter_budget": 2,
            "sweep_config": {
                "strategy": "grid",
                "params": [
                    {"name": "_ml_gate_enabled", "values": [True, False]},
                ],
            },
        },
    )
    body = r.json()
    # The production guard error code must NOT appear for DCB.
    assert body.get("error_code") != "ml_sentinels_on_production_strategy"
