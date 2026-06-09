"""Tests for the walk-forward backlog query (Finding 2 fix, 2026-06-09).

``list_walk_forward_backlog`` surfaces PARKED SIGNIFICANT_EDGE candidates still
awaiting their out-of-sample walk-forward, so the bottleneck is drainable rather
than a silent pile-up. These are repo-level SQL-shape tests against a fake
asyncpg connection — they pin the STRUCTURAL discriminators (status, the
walk_forward_id NULL guard, the SIGNIFICANT_EDGE join) so the query can't drift
into surfacing exhaustion-parked rows.
"""

from __future__ import annotations

import asyncio
from typing import Any

from orchestrator.repo import queue as queue_repo


class _FetchConn:
    """Fake asyncpg conn that records the fetch() SQL + params and returns rows."""

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self._rows = rows or []
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        self.calls.append((sql, params))
        return self._rows


def _run(coro):
    return asyncio.run(coro)


def _sql_of(conn: _FetchConn) -> str:
    return conn.calls[0][0]


def test_backlog_filters_parked_unvalidated_significant_edge() -> None:
    conn = _FetchConn(rows=[])
    _run(queue_repo.list_walk_forward_backlog(conn))
    sql = _sql_of(conn)
    # Structural discriminators — all three must be present.
    assert "status = 'PARKED'" in sql
    assert "walk_forward_id IS NULL" in sql
    assert "statistical_verdict = 'SIGNIFICANT_EDGE'" in sql
    # Must JOIN the iteration log to read statistical_verdict (no note parsing).
    assert "research_iteration_log" in sql
    assert "JOIN" in sql


def test_backlog_no_track_filter_by_default() -> None:
    conn = _FetchConn(rows=[])
    _run(queue_repo.list_walk_forward_backlog(conn))
    # Only the limit is bound when track is None, and no track PREDICATE is added
    # (the SELECT still projects track as a column, but there's no WHERE filter).
    assert conn.calls[0][1] == (100,)
    assert "sweep_config->>'track' = $" not in _sql_of(conn)


def test_backlog_track_filter_is_parameterised() -> None:
    conn = _FetchConn(rows=[])
    _run(queue_repo.list_walk_forward_backlog(conn, track="hedging", limit=25))
    params = conn.calls[0][1]
    # track bound first, then limit.
    assert params == ("hedging", 25)
    assert "sweep_config->>'track' = $1" in _sql_of(conn)


def test_backlog_returns_dicts() -> None:
    seeded = [{"queue_id": "q1", "strategy_code": "EMA_BAND_RESP_BTC"}]
    conn = _FetchConn(rows=seeded)
    out = _run(queue_repo.list_walk_forward_backlog(conn))
    assert out == seeded
    assert isinstance(out[0], dict)


def test_backlog_orders_oldest_drainable_first() -> None:
    conn = _FetchConn(rows=[])
    _run(queue_repo.list_walk_forward_backlog(conn))
    assert "ORDER BY" in _sql_of(conn)
