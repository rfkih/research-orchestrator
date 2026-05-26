"""Tests for ``GET /walk-forward/runs/{walk_forward_id}``.

Two layers:

1. **Repo-level** — exercise ``fetch_walk_forward_run`` against a fake
   asyncpg connection; confirms the SQL shape and dict-conversion.
2. **HTTP-level** — exercise the GET handler via TestClient with a
   dependency-override that injects a fake DB conn, confirming the 200
   happy-path and 404 on missing id.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from orchestrator.api.deps import get_db_conn
from orchestrator.repo import walk_forward as wf_repo


# ── Shared auth helper ────────────────────────────────────────────────────────


def _auth_headers() -> dict[str, str]:
    return {"X-Orch-Token": "test-token", "X-Agent-Name": "quant-curator"}


# ── _FakeConn ─────────────────────────────────────────────────────────────────


class _FakeConn:
    """Minimal asyncpg connection stub — returns pre-seeded fetchrow results."""

    def __init__(self, *, fetchrow_result: Any = None) -> None:
        self._fetchrow_result = fetchrow_result
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetchrow(self, sql: str, *params: Any) -> Any:
        self.calls.append(("fetchrow", sql, params))
        return self._fetchrow_result

    async def fetchval(self, sql: str, *params: Any) -> Any:  # noqa: ARG002
        return None


def _run(coro):
    return asyncio.run(coro)


# ── Repo-level tests ──────────────────────────────────────────────────────────


def _make_db_row(wf_id: UUID, verdict: str = "ROBUST") -> dict[str, Any]:
    """Minimal row dict that mirrors the asyncpg Record shape returned by
    ``fetch_walk_forward_run``."""
    return {
        "walk_forward_id": wf_id,
        "strategy_code": "DCB",
        "instrument": "BTCUSDT",
        "interval_name": "1h",
        "stability_verdict": verdict,
        "n_folds": 6,
        "fold_pf_mean": 1.25,
        "fold_pf_std": 0.10,
        "fold_pf_min": 1.05,
        "fold_pf_max": 1.50,
        "fold_pf_positive_pct": 100.0,
        "fold_sharpe_mean": 1.1,
        "fold_sharpe_std": 0.2,
        "fold_return_mean": 0.08,
        "fold_return_std": 0.03,
        "total_trades_across_folds": 420,
        "fold_results": [],
        "motivating_iteration_id": None,
        "created_time": datetime(2026, 5, 24, 10, 0, tzinfo=timezone.utc),
        "created_by": "orchestrator:test",
    }


def test_fetch_walk_forward_run_returns_dict_on_hit():
    wf_id = uuid4()
    conn = _FakeConn(fetchrow_result=_make_db_row(wf_id))
    result = _run(wf_repo.fetch_walk_forward_run(conn, wf_id))
    assert result is not None
    assert isinstance(result, dict)
    assert result["stability_verdict"] == "ROBUST"
    assert result["walk_forward_id"] == wf_id


def test_fetch_walk_forward_run_returns_none_on_miss():
    conn = _FakeConn(fetchrow_result=None)
    result = _run(wf_repo.fetch_walk_forward_run(conn, uuid4()))
    assert result is None


def test_fetch_walk_forward_run_queries_correct_table():
    wf_id = uuid4()
    conn = _FakeConn(fetchrow_result=_make_db_row(wf_id))
    _run(wf_repo.fetch_walk_forward_run(conn, wf_id))
    assert len(conn.calls) == 1
    sql = conn.calls[0][1]
    assert "walk_forward_run" in sql
    assert "stability_verdict" in sql
    assert "fold_results" in sql


def test_fetch_walk_forward_run_passes_id_as_first_param():
    wf_id = uuid4()
    conn = _FakeConn(fetchrow_result=_make_db_row(wf_id))
    _run(wf_repo.fetch_walk_forward_run(conn, wf_id))
    params = conn.calls[0][2]
    assert params[0] == wf_id


# ── HTTP-level tests ──────────────────────────────────────────────────────────


def _make_client_with_fake_db(
    client_fixture: TestClient, fetchrow_result: Any
) -> tuple[TestClient, _FakeConn]:
    """Return the same TestClient with get_db_conn overridden to use a
    _FakeConn seeded with ``fetchrow_result``."""
    fake_conn = _FakeConn(fetchrow_result=fetchrow_result)

    async def _override():
        yield fake_conn

    client_fixture.app.dependency_overrides[get_db_conn] = _override
    return client_fixture, fake_conn


def test_get_walk_forward_run_returns_stability_verdict(client: TestClient):
    """200 path: fake DB returns a row; endpoint surfaces stability_verdict."""
    wf_id = uuid4()
    row = _make_db_row(wf_id, verdict="ROBUST")
    client, _ = _make_client_with_fake_db(client, row)
    try:
        resp = client.get(f"/walk-forward/runs/{wf_id}", headers=_auth_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert "stability_verdict" in data
        assert data["stability_verdict"] in (
            "ROBUST", "DEGRADED", "INSUFFICIENT_EVIDENCE",
            "INCONSISTENT", "OVERFIT", "NO_EDGE",
        )
        assert data["stability_verdict"] == "ROBUST"
        assert data["walk_forward_id"] == str(wf_id)
        assert data["strategy_code"] == "DCB"
        assert data["n_folds"] == 6
    finally:
        client.app.dependency_overrides.pop(get_db_conn, None)


def test_get_walk_forward_run_404_when_missing(client: TestClient):
    """404 path: fake DB returns None; endpoint returns walk_forward_run_not_found."""
    fake_conn = _FakeConn(fetchrow_result=None)

    async def _override():
        yield fake_conn

    client.app.dependency_overrides[get_db_conn] = _override
    try:
        resp = client.get(f"/walk-forward/runs/{uuid4()}", headers=_auth_headers())
        assert resp.status_code == 404
        body = resp.json()
        assert body["error_code"] == "walk_forward_run_not_found"
        assert body["retryable"] is False
    finally:
        client.app.dependency_overrides.pop(get_db_conn, None)
