"""Unit tests for the nightly rebalance cron task (mocked DB; no real Postgres).

Verifies the pass orchestration: enumerate live accounts, skip zero-USDT,
count applied vs no-change, and ISOLATE per-account failures so one bad account
cannot abort the pass. asyncio_mode=auto runs the async defs directly.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from orchestrator.tasks import rebalance_cron


class _FakeAcquire:
    def __init__(self, conn: Any) -> None:
        self._conn = conn

    async def __aenter__(self) -> Any:
        return self._conn

    async def __aexit__(self, *exc: Any) -> bool:
        return False


class _FakeDB:
    """Stands in for infra.db.Database — only .acquire() is used by the task."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def acquire(self) -> _FakeAcquire:
        return _FakeAcquire(self._conn)


class _Settings:
    rebalance_optimizer = "HRP"
    rebalance_dry_run = False


def _conn(account_ids: list[str], usdt_by_acct: dict[str, float]) -> AsyncMock:
    conn = AsyncMock()
    conn.fetch.return_value = [{"account_id": a} for a in account_ids]

    def _fetchval(_sql: str, account_id: str) -> float | None:
        return usdt_by_acct.get(account_id)

    conn.fetchval.side_effect = _fetchval
    return conn


async def test_skips_accounts_with_no_usdt(monkeypatch: pytest.MonkeyPatch) -> None:
    a1 = str(uuid4())
    conn = _conn([a1], {a1: 0.0})
    rb = AsyncMock()
    monkeypatch.setattr(rebalance_cron, "rebalance_book", rb)
    summary = await rebalance_cron.run_rebalance_once(_FakeDB(conn), _Settings())
    assert summary["accounts"] == 1
    assert summary["skipped"] == 1
    rb.assert_not_called()


async def test_counts_applied_and_no_change(monkeypatch: pytest.MonkeyPatch) -> None:
    a1, a2 = str(uuid4()), str(uuid4())
    conn = _conn([a1, a2], {a1: 100.0, a2: 100.0})
    rb = AsyncMock(
        side_effect=[{"applied": True, "n_updated": 3}, {"applied": False, "n_updated": 0}]
    )
    monkeypatch.setattr(rebalance_cron, "rebalance_book", rb)
    summary = await rebalance_cron.run_rebalance_once(_FakeDB(conn), _Settings())
    assert summary["applied"] == 1
    assert summary["no_change"] == 1
    assert rb.await_count == 2


async def test_isolates_per_account_error(monkeypatch: pytest.MonkeyPatch) -> None:
    a1, a2 = str(uuid4()), str(uuid4())
    conn = _conn([a1, a2], {a1: 100.0, a2: 100.0})
    rb = AsyncMock(
        side_effect=[RuntimeError("422 below_min_notional_floor"), {"applied": True, "n_updated": 2}]
    )
    monkeypatch.setattr(rebalance_cron, "rebalance_book", rb)
    summary = await rebalance_cron.run_rebalance_once(_FakeDB(conn), _Settings())
    assert summary["errors"] == 1
    assert summary["applied"] == 1  # the second account is still processed
    assert rb.await_count == 2


async def test_no_live_accounts_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _conn([], {})
    rb = AsyncMock()
    monkeypatch.setattr(rebalance_cron, "rebalance_book", rb)
    summary = await rebalance_cron.run_rebalance_once(_FakeDB(conn), _Settings())
    assert summary["accounts"] == 0
    rb.assert_not_called()
