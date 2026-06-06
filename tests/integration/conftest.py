"""Live-Postgres fixtures for Phase 1 track-isolation integration tests.

Default: pytest-postgresql spawns an isolated server (postgresql_proc) — needs
initdb/pg_ctl on PATH. If unavailable on this host, switch the marked line to
postgresql_noproc against the dev server on 127.0.0.1:5432 (empty dev DB).
"""
from __future__ import annotations

import json
from pathlib import Path

import asyncpg
import pytest_asyncio
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
