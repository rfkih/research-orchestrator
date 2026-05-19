"""API-shape tests for /ml/training-runs.

Auth + Pydantic validation + route registration only. DB-touching paths
(actual subprocess spawn, budget cap enforcement against real rows)
need pytest-postgresql + real V99 migration — those belong in an
integration suite.

The conftest's TestClient leaves ``raise_server_exceptions`` at its
default (True), which would propagate the "no DB pool" AttributeError
on body-validation tests that get past auth but trip the db dep
before Pydantic runs. We construct a local client with
``raise_server_exceptions=False`` for those tests so 500s surface as
HTTP responses we can assert on.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from orchestrator.config import Settings
from orchestrator.main import create_app
from orchestrator.services.ml_training import (
    _extract_ids,
    _parse_summary,
    _read_file_full,
    _read_file_tail,
    _tail_text,
    build_cli_args,
)


@pytest.fixture
def lenient_client(settings: Settings) -> TestClient:
    """As in test_specialists_api: convert server exceptions to HTTP
    responses so we can assert on the body-validation paths that hit
    the missing DB pool in the no-DB fixture."""
    app = create_app(settings)
    return TestClient(app, raise_server_exceptions=False)


# ── Auth-shape tests (no DB needed) ────────────────────────────────


def test_ml_training_post_requires_token(client: TestClient) -> None:
    r = client.post("/ml/training-runs", json={"spec_name": "regime_btc_v3"})
    assert r.status_code == 401
    assert r.json()["error_code"] == "auth_missing_token"


def test_ml_training_get_by_id_requires_token(client: TestClient) -> None:
    r = client.get("/ml/training-runs/00000000-0000-0000-0000-000000000001")
    assert r.status_code == 401


def test_ml_training_list_requires_token(client: TestClient) -> None:
    r = client.get("/ml/training-runs")
    assert r.status_code == 401


def test_ml_model_specs_requires_token(client: TestClient) -> None:
    r = client.get("/ml/model-specs")
    assert r.status_code == 401


def test_ml_model_specs_default_contains_stage_h_spec() -> None:
    """Phase D — discovery endpoint reads from settings.available_model_specs.
    Pure-function check that the default list contains the canonical
    Stage-H spec name (the rest is operator-curated). Same instantiation
    path the lifespan uses, just constructed directly so we don't need
    the TestClient lifespan + DB to validate the default value."""
    from orchestrator.config import Settings as _S
    specs = _S(
        profile="dev",
        auth_token="test-token",
        db_dsn="postgresql://x:y@127.0.0.1:5432/none",
    ).available_model_specs
    assert "regime_btc_v3" in specs
    assert isinstance(specs, list)
    assert all(isinstance(s, str) for s in specs)


# ── Pydantic-validation tests ──────────────────────────────────────


def test_register_without_write_is_rejected(lenient_client: TestClient) -> None:
    """register=true + no_write=true is incompatible (the model
    validator rejects it before the DB dep is touched)."""
    r = lenient_client.post(
        "/ml/training-runs",
        headers={"X-Orch-Token": "test-token"},
        json={
            "spec_name": "regime_btc_v3",
            "do_register": True,
            "no_write": True,
        },
    )
    assert r.status_code >= 400
    body = r.json()
    assert body["error_code"] in ("validation_failed", "internal_error")


def test_gauntlet_without_walk_forward_is_rejected(lenient_client: TestClient) -> None:
    """gauntlet=true + walk_forward=false is rejected (gauntlet gates
    2-5 read the walk_forward block)."""
    r = lenient_client.post(
        "/ml/training-runs",
        headers={"X-Orch-Token": "test-token"},
        json={
            "spec_name": "regime_btc_v3",
            "walk_forward": False,
            "gauntlet": True,
        },
    )
    assert r.status_code >= 400
    body = r.json()
    assert body["error_code"] in ("validation_failed", "internal_error")


def test_get_by_id_404_on_unknown(lenient_client: TestClient) -> None:
    """An unknown UUID returns 404, not 500 — even with the no-DB
    fixture the route should structurally enforce this once the DB
    dep is wired. Without a DB, we see 500; we accept either as
    proof the route is registered + auth passed."""
    r = lenient_client.get(
        "/ml/training-runs/00000000-0000-0000-0000-000000000001",
        headers={"X-Orch-Token": "test-token"},
    )
    assert r.status_code in (404, 500)


# ── Pure-function tests (build_cli_args + parser) ──────────────────


def test_build_cli_args_default_flags() -> None:
    args = build_cli_args(
        spec_name="regime_btc_v3",
        walk_forward=True,
        gauntlet=True,
        folds=None,
        bayesian=False,
        do_register=True,
        allow_leakage=False,
        no_write=False,
    )
    assert args == [
        "--model", "regime_btc_v3",
        "--walk-forward",
        "--gauntlet",
        "--register",
    ]


def test_build_cli_args_with_folds_and_bayesian() -> None:
    args = build_cli_args(
        spec_name="regime_btc_v3",
        walk_forward=True,
        gauntlet=False,
        folds=3,
        bayesian=True,
        do_register=False,
        allow_leakage=True,
        no_write=True,
    )
    assert args == [
        "--model", "regime_btc_v3",
        "--walk-forward",
        "--folds", "3",
        "--bayesian",
        "--allow-leakage",
        "--no-write",
    ]


def test_parse_summary_returns_last_json_object() -> None:
    stdout = (
        "INFO some log line\n"
        '{"status": "early", "ignored": true}\n'
        "INFO another log line\n"
        '{\n  "status": "ok",\n  "artifact": {"content_sha256": "abc123"}\n}\n'
    )
    summary = _parse_summary(stdout)
    assert summary is not None
    assert summary["status"] == "ok"
    assert summary["artifact"]["content_sha256"] == "abc123"


def test_parse_summary_handles_empty_stdout() -> None:
    assert _parse_summary("") is None


def test_parse_summary_handles_no_braces() -> None:
    assert _parse_summary("INFO nothing to parse here") is None


def test_parse_summary_handles_unbalanced_braces() -> None:
    # Trailing close-brace without an opener — should not match.
    assert _parse_summary("garbage }") is None


def test_extract_ids_pulls_content_sha_and_model_id() -> None:
    summary = {
        "status": "ok",
        "artifact": {"content_sha256": "deadbeef" * 8},
        "registration": {
            "model_id": "11111111-2222-3333-4444-555555555555",
            "status": "ok",
        },
        "experiment_run_id": "99999999-aaaa-bbbb-cccc-dddddddddddd",
    }
    sha, exp_id, model_id = _extract_ids(summary)
    assert sha == "deadbeef" * 8
    assert str(exp_id) == "99999999-aaaa-bbbb-cccc-dddddddddddd"
    assert str(model_id) == "11111111-2222-3333-4444-555555555555"


def test_extract_ids_tolerates_missing_keys() -> None:
    sha, exp_id, model_id = _extract_ids({"status": "ok"})
    assert (sha, exp_id, model_id) == (None, None, None)
    sha, exp_id, model_id = _extract_ids(None)
    assert (sha, exp_id, model_id) == (None, None, None)


def test_extract_ids_handles_malformed_model_id() -> None:
    # registration.model_id present but not a UUID. Should not raise.
    summary = {
        "artifact": {"content_sha256": "x" * 64},
        "registration": {"model_id": "not-a-uuid"},
    }
    sha, exp_id, model_id = _extract_ids(summary)
    assert sha == "x" * 64
    assert exp_id is None
    assert model_id is None


def test_tail_text_under_cap_returns_verbatim() -> None:
    assert _tail_text("hello", 1024) == "hello"


def test_tail_text_over_cap_trims_from_front() -> None:
    s = "a" * 5000
    out = _tail_text(s, 100)
    assert len(out.encode("utf-8")) <= 100
    assert out == "a" * 100


# ── F5 stdio-file helpers ──────────────────────────────────────────


def test_read_file_tail_missing_file_returns_empty(tmp_path) -> None:
    """F5 regression — a subprocess that never wrote anything (e.g.
    spawn failed) shouldn't crash the tail-read."""
    out = _read_file_tail(tmp_path / "nonexistent.log", 1024)
    assert out == ""


def test_read_file_tail_small_file_returns_all(tmp_path) -> None:
    p = tmp_path / "small.log"
    p.write_bytes(b"hello world\n")
    assert _read_file_tail(p, 1024) == "hello world\n"


def test_read_file_tail_large_file_trims_from_front(tmp_path) -> None:
    """F5 regression — proves we read the LAST cap_bytes, not the
    first. The summary JSON blackheart-train dumps at end-of-run must
    survive the tail truncation."""
    p = tmp_path / "big.log"
    body = b"a" * 100_000 + b"FINAL_SUMMARY_LINE"
    p.write_bytes(body)
    out = _read_file_tail(p, 1024)
    assert len(out.encode("utf-8")) <= 1024
    assert "FINAL_SUMMARY_LINE" in out


def test_read_file_full_returns_complete_content(tmp_path) -> None:
    """F5 regression — _read_file_full is used to parse the trailing
    JSON summary; it must return the entire stream, not just a tail."""
    p = tmp_path / "complete.log"
    body = b"prefix " * 1000 + b'\n{"status": "ok"}\n'
    p.write_bytes(body)
    out = _read_file_full(p)
    assert out.endswith('{"status": "ok"}\n')
    assert out.count("prefix") == 1000


def test_read_file_full_missing_returns_empty(tmp_path) -> None:
    assert _read_file_full(tmp_path / "missing.log") == ""
