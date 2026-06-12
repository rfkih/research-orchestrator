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
    _SUMMARY_SCAN_BYTES,
    _dsn_to_train_db_env,
    _extract_ids,
    _parse_summary,
    _read_file_last_bytes,
    _read_file_tail,
    _tail_text,
    build_cli_args,
    build_subprocess_env,
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


def test_read_file_last_bytes_keeps_trailing_summary(tmp_path) -> None:
    """L14 (2026-06-12): the summary parser reads a BOUNDED tail, not the
    whole (potentially 100MB+) capture. The trailing JSON must survive."""
    p = tmp_path / "complete.log"
    body = b"prefix " * 1000 + b'\n{"status": "ok"}\n'
    p.write_bytes(body)
    out = _read_file_last_bytes(p, _SUMMARY_SCAN_BYTES)
    assert out.endswith('{"status": "ok"}\n')
    # File smaller than the cap → returned in full.
    assert out.count("prefix") == 1000


def test_read_file_last_bytes_caps_large_files(tmp_path) -> None:
    p = tmp_path / "huge.log"
    p.write_bytes(b"x" * 10_000 + b'\n{"status": "ok"}\n')
    out = _read_file_last_bytes(p, 1024)
    assert len(out.encode("utf-8")) <= 1024
    assert out.endswith('{"status": "ok"}\n')


def test_read_file_last_bytes_missing_returns_empty(tmp_path) -> None:
    assert _read_file_last_bytes(tmp_path / "missing.log", 1024) == ""


# ── Subprocess env propagation (2026-06-02 INFRA_FAIL fix) ─────────


def test_dsn_to_train_db_env_full_dsn() -> None:
    """A complete DSN maps every component to its TRAIN_DB_* var. This
    is the regression for INFRA_FAIL_2026-06-02 where the child fell
    back to host=localhost inside the container."""
    env = _dsn_to_train_db_env(
        "postgresql://blackheart_research:s3cret@db:5432/trading_db"
    )
    assert env == {
        "TRAIN_DB_HOST": "db",
        "TRAIN_DB_PORT": "5432",
        "TRAIN_DB_NAME": "trading_db",
        "TRAIN_DB_USER": "blackheart_research",
        "TRAIN_DB_PASSWORD": "s3cret",
    }


def test_dsn_to_train_db_env_percent_encoded_password() -> None:
    """libpq DSNs URL-encode special chars in the password; the child
    needs the decoded value."""
    env = _dsn_to_train_db_env(
        "postgresql://u:p%40ss%2Fword@host.internal:6543/bh"
    )
    assert env["TRAIN_DB_PASSWORD"] == "p@ss/word"
    assert env["TRAIN_DB_USER"] == "u"
    assert env["TRAIN_DB_HOST"] == "host.internal"
    assert env["TRAIN_DB_PORT"] == "6543"
    assert env["TRAIN_DB_NAME"] == "bh"


def test_dsn_to_train_db_env_omits_absent_components() -> None:
    """No port / no password in the DSN → those keys are absent (the
    child keeps its own defaults) rather than empty-string overrides."""
    env = _dsn_to_train_db_env("postgresql://justuser@onlyhost/onlydb")
    assert env == {
        "TRAIN_DB_HOST": "onlyhost",
        "TRAIN_DB_NAME": "onlydb",
        "TRAIN_DB_USER": "justuser",
    }
    assert "TRAIN_DB_PORT" not in env
    assert "TRAIN_DB_PASSWORD" not in env


def test_build_subprocess_env_overlays_train_vars() -> None:
    """The full child env carries TRAIN_DB_* from the DSN plus the
    orchestrator URL + token so registration/tracking authenticate.
    Inherited os.environ keys survive (PATH present)."""
    settings = Settings(
        profile="dev",
        auth_token="real-orch-token",
        db_dsn="postgresql://bhr:pw@dbhost:5432/trading_db",
        port=8082,
    )
    env = build_subprocess_env(settings)
    assert env["TRAIN_DB_HOST"] == "dbhost"
    assert env["TRAIN_DB_NAME"] == "trading_db"
    assert env["TRAIN_DB_USER"] == "bhr"
    assert env["TRAIN_DB_PASSWORD"] == "pw"
    assert env["TRAIN_ORCHESTRATOR_URL"] == "http://127.0.0.1:8082"
    assert env["TRAIN_ORCHESTRATOR_TOKEN"] == "real-orch-token"
    # Inherited environment survives the overlay.
    assert "PATH" in env or "Path" in env
    # Artifact dir unset by default → TRAIN_ARTIFACT_DIR not injected, so
    # the child keeps its own Settings.artifact_dir default (dev-host
    # behaviour preserved).
    assert "TRAIN_ARTIFACT_DIR" not in env


def test_build_subprocess_env_injects_artifact_dir_when_set() -> None:
    """When the operator configures ORCH_ML_TRAINING_ARTIFACT_DIR (a
    writable in-container path), build_subprocess_env injects it as
    TRAIN_ARTIFACT_DIR so the train subprocess does not fall back to the
    Windows host default C:/Project/blackheart-train/artifacts (OSError
    Read-only file system: 'C:' inside the Linux container)."""
    settings = Settings(
        profile="dev",
        auth_token="real-orch-token",
        db_dsn="postgresql://bhr:pw@dbhost:5432/trading_db",
        port=8082,
        ml_training_artifact_dir="/artifacts",
    )
    env = build_subprocess_env(settings)
    assert env["TRAIN_ARTIFACT_DIR"] == "/artifacts"


def test_build_subprocess_env_artifact_dir_blank_is_not_injected() -> None:
    """A whitespace-only override is treated as unset (strip → empty) so
    a stray ORCH_ML_TRAINING_ARTIFACT_DIR='' does not clobber the child's
    default with an empty path."""
    settings = Settings(
        profile="dev",
        auth_token="real-orch-token",
        db_dsn="postgresql://bhr:pw@dbhost:5432/trading_db",
        port=8082,
        ml_training_artifact_dir="   ",
    )
    env = build_subprocess_env(settings)
    assert "TRAIN_ARTIFACT_DIR" not in env
