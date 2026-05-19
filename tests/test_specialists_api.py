"""API-shape tests for /skeptic-prescreen, /portfolio/*, /agent-decisions/log.

Auth + Pydantic validation + route registration only. DB-touching paths
(actual prescreen execution, portfolio fetch) need pytest-postgresql +
real migrations — those belong in an integration suite, not here.

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


@pytest.fixture
def lenient_client(settings: Settings) -> TestClient:
    """Like the conftest's ``client`` but with server exceptions
    converted to HTTP responses — required for tests that hit
    body-validation paths protected by a DB dep that fails in the
    no-DB-pool fixture."""
    app = create_app(settings)
    return TestClient(app, raise_server_exceptions=False)


def test_skeptic_prescreen_requires_token(client: TestClient) -> None:
    r = client.post(
        "/skeptic-prescreen",
        json={
            "iteration_id": "00000000-0000-0000-0000-000000000001",
            "strategy_code": "TEST",
        },
    )
    assert r.status_code == 401
    assert r.json()["error_code"] == "auth_missing_token"


def test_portfolio_correlations_requires_token(client: TestClient) -> None:
    r = client.post(
        "/portfolio/correlations",
        json={"strategy_codes": ["LSR", "VCB"]},
    )
    assert r.status_code == 401


def test_portfolio_optimize_requires_token(client: TestClient) -> None:
    r = client.post(
        "/portfolio/optimize",
        json={"strategy_codes": ["LSR", "VCB"]},
    )
    assert r.status_code == 401


def test_agent_decisions_log_requires_token(client: TestClient) -> None:
    r = client.post(
        "/agent-decisions/log",
        json={
            "specialist": "quant-skeptic",
            "endpoint": "agent_spawn",
            "request_payload": {},
            "status": "ok",
        },
    )
    assert r.status_code == 401


def test_portfolio_correlations_rejects_too_few_codes(lenient_client: TestClient) -> None:
    client = lenient_client
    """Pydantic schema: strategy_codes min_length=1, max_length=20.
    Empty list MUST not 2xx. In the no-DB test fixture FastAPI opens
    the db dep before body validation so we see 500 instead of 422;
    both are non-success and prove the request was rejected."""
    r = client.post(
        "/portfolio/correlations",
        headers={"X-Orch-Token": "test-token"},
        json={"strategy_codes": []},
    )
    assert r.status_code >= 400
    body = r.json()
    assert body["error_code"] in ("validation_failed", "internal_error")


def test_agent_decisions_log_rejects_bad_status(lenient_client: TestClient) -> None:
    client = lenient_client
    """The Literal enum on status rejects unknown values at the API layer
    (also enforced at DB layer via V97 CHECK; both must agree). Same
    no-DB-fixture caveat as above."""
    r = client.post(
        "/agent-decisions/log",
        headers={"X-Orch-Token": "test-token"},
        json={
            "specialist": "quant-skeptic",
            "endpoint": "agent_spawn",
            "request_payload": {},
            "status": "NOT_A_REAL_STATUS",
        },
    )
    assert r.status_code >= 400
    assert r.json()["error_code"] in ("validation_failed", "internal_error")


def test_skeptic_prescreen_rejects_non_uuid_iteration_id(lenient_client: TestClient) -> None:
    client = lenient_client
    r = client.post(
        "/skeptic-prescreen",
        headers={"X-Orch-Token": "test-token"},
        json={"iteration_id": "not-a-uuid", "strategy_code": "TEST"},
    )
    assert r.status_code >= 400
    assert r.json()["error_code"] in ("validation_failed", "internal_error")
