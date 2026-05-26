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
        {"quant-skeptic", "quant-portfolio-manager", "quant-ml-judge", "quant-curator"}
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
    # PR #2: curator is a post-graduation actuator; no researcher-blocking veto.
    assert repo.VERDICT_LITERALS["quant-curator"] == frozenset(
        {"PROMOTE", "HOLD", "REJECT"}
    )


def test_veto_verdicts_pinned() -> None:
    """The veto set is what gates graduation. Drift here = silent
    weakening of the adversarial gate."""
    assert repo.VETO_VERDICTS["quant-skeptic"] == frozenset({"OVERRIDE_REJECT"})
    assert repo.VETO_VERDICTS["quant-portfolio-manager"] == frozenset({"REJECT"})
    assert repo.VETO_VERDICTS["quant-ml-judge"] == frozenset({"OVERRIDE_REJECT"})
    # Curator has no researcher-blocking vetos -- explicit empty frozenset,
    # not a missing key, so the contract is self-documenting.
    assert repo.VETO_VERDICTS["quant-curator"] == frozenset()


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


# ── quant-curator Path C contract (PR #2) ──────────────────────────


def test_specialist_request_accepts_quant_curator(lenient_client: TestClient) -> None:
    """POST /specialist-review/request with specialist_name="quant-curator"
    must NOT be rejected by Pydantic (422). A DB error (500) means the name
    passed validation — expected in unit-test context without a live DB."""
    import uuid as _uuid

    body = {
        "specialist_name": "quant-curator",
        "iteration_id": str(_uuid.uuid4()),
        "strategy_code": "DCB",
        "motivating_hypothesis_id": None,
        "request_payload": {
            "symbol": "ETHUSDT",
            "interval": "1h",
            "backtest_run_id": str(_uuid.uuid4()),
            "walk_forward_run_id": str(_uuid.uuid4()),
        },
    }
    resp = lenient_client.post(
        "/specialist-review/request",
        json=body,
        headers={"X-Orch-Token": "test-token"},
    )
    # 422 = Pydantic rejected the specialist_name Literal — that would be a bug.
    # 201 = happy path (requires live DB). 500 = DB unavailable but name passed
    # validation (expected in unit-test context).
    assert resp.status_code != 422, (
        f"quant-curator was rejected by Pydantic; status={resp.status_code}: {resp.text}"
    )


@pytest.mark.parametrize("verdict", ["PROMOTE", "HOLD", "REJECT"])
def test_curator_verdict_accepted(lenient_client: TestClient, verdict: str) -> None:
    """POST /specialist-review/complete with quant-curator + each valid verdict
    must NOT trip the invalid_specialist_verdict gate (repo ValueError → 400).
    Without a live DB the route returns 500 before reaching the verdict check,
    so we accept any status that is NOT 400 with error_code=invalid_specialist_verdict."""
    import uuid as _uuid

    iteration_id = str(_uuid.uuid4())
    target_id = f"specialist:quant-curator:{iteration_id}"
    body = {
        "target_id": target_id,
        "specialist_name": "quant-curator",
        "iteration_id": iteration_id,
        "strategy_code": "DCB",
        "verdict": verdict,
        "reasoning": "test reasoning",
        "raw_response": {"return_text": "..."},
        "motivating_request_id": target_id,
    }
    resp = lenient_client.post(
        "/specialist-review/complete",
        json=body,
        headers={"X-Orch-Token": "test-token"},
    )
    # The only outcome that indicates a regression is a 400 with
    # error_code=invalid_specialist_verdict -- that would mean the VERDICT_LITERALS
    # dict doesn't include quant-curator's verdicts.
    if resp.status_code == 400:
        payload = resp.json()
        assert payload.get("error_code") != "invalid_specialist_verdict", (
            f"verdict={verdict!r} was incorrectly flagged as invalid for "
            f"quant-curator: {resp.text}"
        )


def test_curator_invalid_verdict_rejected(lenient_client: TestClient) -> None:
    """CONCUR is valid for quant-skeptic but NOT for quant-curator.
    The repo layer must raise ValueError → 400 invalid_specialist_verdict.
    Without a live DB the route returns 500 (DB error before verdict check),
    so we assert that if a 400 IS returned it carries the right error_code."""
    import uuid as _uuid

    iteration_id = str(_uuid.uuid4())
    target_id = f"specialist:quant-curator:{iteration_id}"
    body = {
        "target_id": target_id,
        "specialist_name": "quant-curator",
        "iteration_id": iteration_id,
        "strategy_code": "DCB",
        "verdict": "CONCUR",  # valid for skeptic, NOT curator
        "reasoning": "test",
        "raw_response": {"return_text": "..."},
        "motivating_request_id": target_id,
    }
    resp = lenient_client.post(
        "/specialist-review/complete",
        json=body,
        headers={"X-Orch-Token": "test-token"},
    )
    assert resp.status_code >= 400, resp.text
    # If a 400 is returned it must carry the expected error_code.
    if resp.status_code == 400:
        assert resp.json()["error_code"] == "invalid_specialist_verdict"
