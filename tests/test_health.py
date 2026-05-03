"""Smoke tests — no DB, no JVM. Phase 1 sanity check only."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_healthz_is_public_and_returns_ok(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_playbook_is_public_and_describes_auth(client: TestClient) -> None:
    r = client.get("/agent/playbook")
    assert r.status_code == 200
    body = r.json()
    assert body["auth"]["header"] == "X-Orch-Token"
    assert any(c["path"] == "/healthz" for c in body["capabilities"])


def test_protected_endpoint_rejects_missing_token(client: TestClient) -> None:
    r = client.get("/agent/state")
    assert r.status_code == 401
    body = r.json()
    assert body["error_code"] == "auth_missing_token"
    assert body["retryable"] is False


def test_protected_endpoint_rejects_bad_token(client: TestClient) -> None:
    r = client.get("/agent/state", headers={"X-Orch-Token": "wrong"})
    assert r.status_code == 401
    assert r.json()["error_code"] == "auth_bad_token"


def test_protected_endpoint_accepts_correct_token(client: TestClient) -> None:
    # Health probes will return False because the pool/client never opened
    # against a real DB/JVM, but the endpoint should answer 200 with that
    # state encoded in the body.
    r = client.get(
        "/agent/state",
        headers={"X-Orch-Token": "test-token", "X-Agent-Name": "quant-researcher"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["agent"] == "quant-researcher"
    assert body["db_ok"] is False
    assert body["jvm_ok"] is False


def test_queue_rejects_bad_status_filter(client: TestClient) -> None:
    r = client.get(
        "/queue?status=NOT_A_STATUS",
        headers={"X-Orch-Token": "test-token"},
    )
    assert r.status_code == 400
    assert r.json()["error_code"] == "bad_status_filter"


def test_iterations_rejects_bad_verdict_filter(client: TestClient) -> None:
    r = client.get(
        "/iterations?verdict=NOT_A_VERDICT",
        headers={"X-Orch-Token": "test-token"},
    )
    assert r.status_code == 400
    assert r.json()["error_code"] == "bad_verdict_filter"


def test_leaderboard_rejects_bad_sort(client: TestClient) -> None:
    r = client.get(
        "/leaderboard?sort=garbage",
        headers={"X-Orch-Token": "test-token"},
    )
    assert r.status_code == 400
    assert r.json()["error_code"] == "bad_leaderboard_sort"


def test_journal_rejects_bad_entry_type(client: TestClient) -> None:
    r = client.get(
        "/journal?entry_type=BANANA",
        headers={"X-Orch-Token": "test-token"},
    )
    assert r.status_code == 400
    assert r.json()["error_code"] == "bad_entry_type"


def test_bad_cursor_returns_envelope(client: TestClient) -> None:
    r = client.get(
        "/queue?cursor=not-base64-or-json",
        headers={"X-Orch-Token": "test-token"},
    )
    assert r.status_code == 400
    assert r.json()["error_code"] == "bad_cursor"
