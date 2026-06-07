"""Live-Postgres fixtures for Phase 1 track-isolation integration tests.

Default: pytest-postgresql spawns an isolated server (postgresql_proc) — needs
initdb/pg_ctl on PATH. If unavailable on this host, switch the marked line to
postgresql_noproc against the dev server on 127.0.0.1:5432 (empty dev DB).
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import asyncpg
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from pydantic import SecretStr
from pytest_postgresql import factories

# NOTE: no pg binaries (initdb/pg_ctl/pg_config) on PATH on this host, so the
# postgresql_proc default cannot spawn an isolated server. Fall back to the
# existing dev server on 127.0.0.1:5432 (postgresql_noproc). Credentials are the
# verified dev superuser; pytest-postgresql creates/drops the throwaway test DB.
postgresql_proc = factories.postgresql_noproc(
    host="127.0.0.1", port=5432, user="postgres", password="admin",
)
postgresql_db = factories.postgresql("postgresql_proc", dbname="orch_phase1_test")

_SCHEMA = (Path(__file__).parent / "schema_phase1.sql").read_text(encoding="utf-8")


@pytest_asyncio.fixture
async def db_conn(postgresql_db):
    p = postgresql_db.info  # psycopg ConnectionInfo
    conn = await asyncpg.connect(
        host=p.host, port=p.port, user=p.user,
        password=p.password, database=p.dbname,
    )
    await conn.set_type_codec(
        "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog",
    )
    await conn.execute(_SCHEMA)
    try:
        yield conn
    finally:
        await conn.close()


@pytest.fixture
def integration_client(postgresql_db):
    """A ``TestClient`` driving the app with a REAL asyncpg pool against the
    same throwaway test DB the ``db_conn`` fixture uses.

    Unlike ``tests/conftest.py:client`` (which stubs ``get_db_conn`` to ``None``
    and never opens the pool), this enters the app's lifespan via the
    ``with TestClient(app)`` block so the pool is opened on the TestClient's
    event loop and ``get_db_conn`` acquires real connections from it. The
    handler's real digest path therefore runs end-to-end.
    """
    from orchestrator.config import Settings
    from orchestrator.main import create_app

    p = postgresql_db.info
    dsn = f"postgresql://{p.user}:{p.password}@{p.host}:{p.port}/{p.dbname}"

    settings = Settings(
        profile="dev",
        auth_token=SecretStr("test-token"),
        db_dsn=SecretStr(dsn),
        jvm_base_url="http://127.0.0.1:8081",
        jvm_auth_mode="dev_bypass",
    )

    # Ensure the schema exists in the target DB before the app boots. Idempotent
    # (db_conn applies the same DDL; both use CREATE TABLE IF NOT EXISTS). A
    # standalone connection on its own loop avoids cross-loop reuse.
    async def _apply_schema() -> None:
        conn = await asyncpg.connect(dsn)
        try:
            await conn.execute(_SCHEMA)
        finally:
            await conn.close()

    asyncio.new_event_loop().run_until_complete(_apply_schema())

    app = create_app(settings)
    # Entering the context manager runs lifespan_for, which opens app.state.db's
    # pool on the TestClient loop and populates the rest of app.state.
    with TestClient(app) as test_client:
        yield test_client
