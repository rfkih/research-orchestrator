"""V54 — _resolve_account_strategy must scope by account_id when configured.

The orchestrator now runs as the dedicated research-agent user (its own
account, seeded by Flyway V54). When ``ORCH_RESEARCH_ACCOUNT_ID`` is set,
``_resolve_account_strategy`` must pin the lookup to that account so admin-
owned rows for the same strategy_code are never picked. When unset (local
dev convenience), it must fall back to the legacy "first matching row"
behaviour so a developer doesn't need to wire the env var to run tests.

These tests use a fake asyncpg connection that records the SQL + bound
params — no live DB required — so they run alongside the existing unit
suite (``PYTHONPATH=src pytest -q``).
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID, uuid4

from orchestrator.config import Settings
from orchestrator.services.tick import (
    _resolve_account_strategy,
    _resolve_account_strategy_id,
)
from pydantic import SecretStr


class _FakeConn:
    """Records the SQL + params each query was called with, returns the
    pre-seeded row(s). Enough to exercise the WHERE-clause shape; we don't
    need a real DB to assert the account_id filter is wired correctly."""

    def __init__(self, fetchval_result: Any = None, fetchrow_result: Any = None) -> None:
        self.fetchval_result = fetchval_result
        self.fetchrow_result = fetchrow_result
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetchval(self, sql: str, *params: Any) -> Any:
        self.calls.append((sql, params))
        return self.fetchval_result

    async def fetchrow(self, sql: str, *params: Any) -> Any:
        self.calls.append((sql, params))
        return self.fetchrow_result


def _run(coro: Any) -> Any:
    # asyncio.run gives each test a fresh event loop. Avoids the
    # "no current event loop" deprecation in 3.10+.
    return asyncio.run(coro)


# ── _resolve_account_strategy_id ──────────────────────────────────────────


def test_resolve_id_filters_by_account_id_when_set() -> None:
    agent_id = UUID("99999999-9999-9999-9999-000000000002")
    expected_strategy_id = uuid4()
    conn = _FakeConn(fetchval_result=expected_strategy_id)

    result = _run(_resolve_account_strategy_id(conn, "LSR", account_id=agent_id))

    assert result == expected_strategy_id
    sql, params = conn.calls[0]
    assert "account_id = $2" in sql
    assert params == ("LSR", agent_id)


def test_resolve_id_falls_back_when_account_id_unset() -> None:
    conn = _FakeConn(fetchval_result=None)

    _run(_resolve_account_strategy_id(conn, "LSR", account_id=None))

    sql, params = conn.calls[0]
    assert "account_id" not in sql
    assert params == ("LSR",)


# ── _resolve_account_strategy (full row) ──────────────────────────────────


def test_resolve_full_filters_by_account_id_when_set() -> None:
    agent_id = UUID("99999999-9999-9999-9999-000000000002")
    row = {
        "account_strategy_id": uuid4(),
        "allow_long": True,
        "allow_short": False,
    }
    conn = _FakeConn(fetchrow_result=row)

    result = _run(_resolve_account_strategy(conn, "LSR", account_id=agent_id))

    assert result is not None
    assert result["account_strategy_id"] == str(row["account_strategy_id"])
    assert result["allow_long"] is True
    assert result["allow_short"] is False
    sql, params = conn.calls[0]
    assert "account_id = $2" in sql
    assert params == ("LSR", agent_id)


def test_resolve_full_returns_none_when_no_match() -> None:
    agent_id = UUID("99999999-9999-9999-9999-000000000002")
    conn = _FakeConn(fetchrow_result=None)
    result = _run(_resolve_account_strategy(conn, "LSR", account_id=agent_id))
    assert result is None


# ── Settings.assert_prod_safe ─────────────────────────────────────────────


def test_assert_prod_safe_requires_research_account_id() -> None:
    s = Settings(
        profile="prod",
        auth_token=SecretStr("real-prod-secret-not-the-dev-one"),
        db_dsn=SecretStr("postgresql://x:y@127.0.0.1:5432/none"),
        jvm_auth_mode="service_account",
        jvm_service_user="research-agent@blackheart.local",
        jvm_service_password=SecretStr("real-secret"),
        research_account_id=None,
    )
    try:
        s.assert_prod_safe()
    except RuntimeError as e:
        assert "ORCH_RESEARCH_ACCOUNT_ID" in str(e)
    else:
        raise AssertionError("expected assert_prod_safe to refuse missing research_account_id")


def test_assert_prod_safe_passes_with_research_account_id() -> None:
    s = Settings(
        profile="prod",
        auth_token=SecretStr("real-prod-secret-not-the-dev-one"),
        db_dsn=SecretStr("postgresql://x:y@127.0.0.1:5432/none"),
        jvm_auth_mode="service_account",
        jvm_service_user="research-agent@blackheart.local",
        jvm_service_password=SecretStr("real-secret"),
        research_account_id=UUID("99999999-9999-9999-9999-000000000002"),
    )
    s.assert_prod_safe()  # no raise
