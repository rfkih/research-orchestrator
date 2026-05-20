"""Tests for /specialist-review/* + repo/specialist_reviews.py.

Auth + Pydantic validation + route registration + pure-function helpers.
DB-touching paths need pytest-postgresql + real migrations; those belong
in an integration suite, not here. The same lenient_client pattern from
``test_specialists_api`` applies — body-validation tests that get past
auth trip the db dep before Pydantic so we accept 500 as "request did
not 2xx" in those cases.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from orchestrator.config import Settings
from orchestrator.main import create_app
from orchestrator.repo import specialist_reviews as repo


@pytest.fixture
def lenient_client(settings: Settings) -> TestClient:
    """Server exceptions surface as HTTP responses — required for body-
    validation tests behind the DB dep."""
    app = create_app(settings)
    return TestClient(app, raise_server_exceptions=False)


# ── pure-function helpers ───────────────────────────────────────────


def test_specialist_target_id_format_pinned() -> None:
    """The target_id format is part of the durable contract — the slash
    command and the resume-protocol parser both depend on this shape.
    Pin so renames surface as a test diff."""
    tid = repo.specialist_target_id(
        "quant-skeptic", "11111111-2222-3333-4444-555555555555"
    )
    assert tid == "specialist:quant-skeptic:11111111-2222-3333-4444-555555555555"


def test_specialist_target_id_rejects_unknown_specialist() -> None:
    with pytest.raises(ValueError, match="not in"):
        repo.specialist_target_id("not-a-real-agent", "abc")


def test_specialist_names_frozenset_pinned() -> None:
    """If a new specialist is added (e.g., capacity-judge once
    /capacity-sweep ships), this test catches the change so the slash
    command + researcher prompt stay in sync."""
    assert repo.SPECIALIST_NAMES == frozenset(
        {"quant-skeptic", "quant-portfolio-manager", "quant-ml-judge"}
    )


def test_verdict_literals_per_specialist_pinned() -> None:
    assert repo.VERDICT_LITERALS["quant-skeptic"] == frozenset(
        {"CONCUR", "CONCERN", "OVERRIDE_REJECT"}
    )
    assert repo.VERDICT_LITERALS["quant-portfolio-manager"] == frozenset(
        {"ADD", "CONCERN", "REJECT"}
    )
    assert repo.VERDICT_LITERALS["quant-ml-judge"] == frozenset(
        {"CONCUR", "CONCERN", "OVERRIDE_REJECT"}
    )


def test_veto_verdicts_pinned() -> None:
    """The veto set is what gates graduation. Drift here = silent
    weakening of the adversarial gate."""
    assert repo.VETO_VERDICTS["quant-skeptic"] == frozenset({"OVERRIDE_REJECT"})
    assert repo.VETO_VERDICTS["quant-portfolio-manager"] == frozenset({"REJECT"})
    assert repo.VETO_VERDICTS["quant-ml-judge"] == frozenset({"OVERRIDE_REJECT"})


def test_kind_discriminators_pinned() -> None:
    """The structured_data.kind values are how rows are found across
    the journal. Renaming them silently makes existing rows invisible.
    Pin both."""
    assert repo.KIND_REQUEST == "specialist_review_request"
    assert repo.KIND_VERDICT == "specialist_review_verdict"


# ── auth ────────────────────────────────────────────────────────────


def test_specialist_review_request_requires_token(client: TestClient) -> None:
    r = client.post(
        "/specialist-review/request",
        json={
            "specialist_name": "quant-skeptic",
            "iteration_id": "00000000-0000-0000-0000-000000000001",
            "strategy_code": "TEST",
        },
    )
    assert r.status_code == 401
    assert r.json()["error_code"] == "auth_missing_token"


def test_specialist_review_pending_requires_token(client: TestClient) -> None:
    r = client.get("/specialist-review/pending")
    assert r.status_code == 401


def test_specialist_review_complete_requires_token(client: TestClient) -> None:
    r = client.post(
        "/specialist-review/complete",
        json={
            "target_id": "specialist:quant-skeptic:abc",
            "specialist_name": "quant-skeptic",
            "iteration_id": "00000000-0000-0000-0000-000000000001",
            "verdict": "CONCUR",
        },
    )
    assert r.status_code == 401


def test_specialist_review_by_iteration_requires_token(client: TestClient) -> None:
    r = client.get(
        "/specialist-review/by-iteration",
        params={"iteration_id": "00000000-0000-0000-0000-000000000001"},
    )
    assert r.status_code == 401


# ── validation (post-auth) ──────────────────────────────────────────


def test_request_rejects_unknown_specialist_name(lenient_client: TestClient) -> None:
    """The Literal type on specialist_name rejects unknown values at
    the API layer before any DB call. 422 = clean Pydantic refusal.
    Same no-DB-fixture caveat as the prescreen tests — db dep may fire
    first in some routes; we accept any 4xx/5xx as "rejected"."""
    client = lenient_client
    r = client.post(
        "/specialist-review/request",
        headers={"X-Orch-Token": "test-token"},
        json={
            "specialist_name": "not-a-real-specialist",
            "iteration_id": "00000000-0000-0000-0000-000000000001",
            "strategy_code": "TEST",
        },
    )
    assert r.status_code >= 400
    body = r.json()
    assert body["error_code"] in ("validation_failed", "internal_error")


def test_request_rejects_missing_iteration_id(lenient_client: TestClient) -> None:
    client = lenient_client
    r = client.post(
        "/specialist-review/request",
        headers={"X-Orch-Token": "test-token"},
        json={
            "specialist_name": "quant-skeptic",
            "strategy_code": "TEST",
        },
    )
    assert r.status_code >= 400


def test_complete_rejects_unknown_specialist_name(lenient_client: TestClient) -> None:
    client = lenient_client
    r = client.post(
        "/specialist-review/complete",
        headers={"X-Orch-Token": "test-token"},
        json={
            "target_id": "specialist:bogus:abc",
            "specialist_name": "not-a-real-specialist",
            "iteration_id": "00000000-0000-0000-0000-000000000001",
            "verdict": "CONCUR",
        },
    )
    assert r.status_code >= 400


def test_complete_rejects_missing_verdict(lenient_client: TestClient) -> None:
    client = lenient_client
    r = client.post(
        "/specialist-review/complete",
        headers={"X-Orch-Token": "test-token"},
        json={
            "target_id": "specialist:quant-skeptic:abc",
            "specialist_name": "quant-skeptic",
            "iteration_id": "00000000-0000-0000-0000-000000000001",
        },
    )
    assert r.status_code >= 400


def test_by_iteration_rejects_bad_uuid(lenient_client: TestClient) -> None:
    client = lenient_client
    r = client.get(
        "/specialist-review/by-iteration",
        headers={"X-Orch-Token": "test-token"},
        params={"iteration_id": "not-a-uuid"},
    )
    assert r.status_code >= 400
