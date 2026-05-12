"""Shared fixtures.

Tests build the app with an explicit ``Settings`` instance so they don't
depend on a ``.env`` file existing. The DB pool and JVM client are real
classes but never opened — health probes return False, which is what the
tests assert against.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from orchestrator.config import Settings
from orchestrator.main import create_app


@pytest.fixture
def settings() -> Settings:
    # research_account_id matches the pinned UUID seeded by Flyway V54 so
    # tests that exercise tick-account scoping see the agent's account, not
    # a random admin row.
    from uuid import UUID as _UUID
    return Settings(
        profile="dev",
        auth_token=SecretStr("test-token"),
        db_dsn=SecretStr("postgresql://x:y@127.0.0.1:5432/none"),
        jvm_base_url="http://127.0.0.1:8081",
        jvm_auth_mode="dev_bypass",
        research_account_id=_UUID("99999999-9999-9999-9999-000000000002"),
    )


@pytest.fixture
def client(settings: Settings) -> TestClient:
    app = create_app(settings)
    return TestClient(app)
