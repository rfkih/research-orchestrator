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

from orchestrator.api.deps import get_db_conn
from orchestrator.clients.inference import InferenceClient
from orchestrator.clients.jvm import JvmClient
from orchestrator.clients.trading_jvm import TradingJvmClient
from orchestrator.config import Settings
from orchestrator.infra.db import Database
from orchestrator.main import create_app
from orchestrator.services.idempotency import InMemoryIdempotencyStore


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
    # create_app wires routes but leaves app.state empty — population is the
    # lifespan's job, and TestClient is used WITHOUT a `with` block so the
    # lifespan never runs. Mirror the lifespan's cheap, no-I/O construction so
    # handlers that read app.state work. The pool + clients are real classes
    # but never .open()ed, so health_probe() returns False — which the smoke
    # tests assert against (see module docstring).
    app.state.settings = settings
    app.state.db = Database(settings)
    app.state.jvm = JvmClient(settings)
    app.state.trading_jvm = TradingJvmClient(settings)
    app.state.inference = InferenceClient(settings)
    app.state.idempotency = InMemoryIdempotencyStore()
    app.state.redis = None
    app.state.available_specs = list(settings.available_model_specs)
    app.state.excluded_ml_features = []

    # Validation-rejection endpoints (bad status/verdict/sort/cursor) resolve
    # get_db_conn before their body runs the input check; with no opened pool
    # acquire() would raise, so the 4xx envelope never gets returned. Override
    # it to yield a sentinel — the body then validates and returns the
    # expected error. Mirrors the per-test override in test_walk_forward.py.
    async def _no_db_conn():
        yield None

    app.dependency_overrides[get_db_conn] = _no_db_conn
    return TestClient(app)
