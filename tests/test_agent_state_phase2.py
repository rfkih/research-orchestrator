"""Tests for ``repo/agent_state.get_state_digest``.

The digest composes 6 read queries. Pure functions, no DB needed —
``FakeConn`` returns canned responses keyed on the SQL fragment, and
each test asserts the digest's shape.

DB-touching paths (asyncpg round-trip behaviour, JSONB codec) are
already covered by the existing repo tests; this file covers the
composition logic only.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

import pytest

from orchestrator.repo import agent_state as repo


# ── FakeConn ─────────────────────────────────────────────────────────


class _Row(dict):
    """asyncpg.Record-ish — dict that also accepts attribute access."""

    def __getattr__(self, name: str) -> Any:
        return self[name]


class FakeConn:
    """Minimal asyncpg.Connection stub.

    Routes each query to a canned response by matching a substring of
    the SQL. The substrings track distinct query intents — when a query
    body changes, the test fails loudly instead of silently regressing.
    """

    def __init__(self) -> None:
        self.queue_rows: list[dict[str, Any]] = []
        self.iter_rows: list[dict[str, Any]] = []
        self.sig_edge_rows: list[dict[str, Any]] = []
        self.hypothesis_rows: list[dict[str, Any]] = []
        self.run_summary_row: dict[str, Any] | None = None
        self.null_screen_rows: list[dict[str, Any]] = []
        self.pending_specialist_rows: list[dict[str, Any]] = []
        self.recent_specialist_verdict_rows: list[dict[str, Any]] = []

    async def fetch(self, sql: str, *args: Any) -> list[_Row]:
        s = sql.lower()
        if "from research_queue" in s and "group by status" in s:
            return [_Row(r) for r in self.queue_rows]
        if "from research_iteration_log" in s and "statistical_verdict = 'significant_edge'" in s:
            return [_Row(r) for r in self.sig_edge_rows]
        if "from research_iteration_log" in s:
            return [_Row(r) for r in self.iter_rows]
        if "entry_type = 'hypothesis'" in s:
            return [_Row(r) for r in self.hypothesis_rows]
        if "entry_type = 'null_screen_result'" in s:
            return [_Row(r) for r in self.null_screen_rows]
        if "entry_type = 'idea_backlog'" in s and "specialist_review_request" in s:
            return [_Row(r) for r in self.pending_specialist_rows]
        if "entry_type = 'strategy_outcome'" in s and "specialist_review_verdict" in s:
            return [_Row(r) for r in self.recent_specialist_verdict_rows]
        raise AssertionError(f"Unexpected fetch SQL fragment:\n{sql}")

    async def fetchrow(self, sql: str, *args: Any) -> _Row | None:
        s = sql.lower()
        if "entry_type = 'run_summary'" in s:
            if self.run_summary_row is None:
                return None
            return _Row(self.run_summary_row)
        raise AssertionError(f"Unexpected fetchrow SQL fragment:\n{sql}")


# ── _queue_counts ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_queue_counts_fills_missing_statuses_with_zero() -> None:
    conn = FakeConn()
    conn.queue_rows = [
        {"status": "PENDING", "n": 3},
        {"status": "RUNNING", "n": 1},
    ]
    out = await repo._queue_counts(conn)  # type: ignore[arg-type]
    assert out == {
        "PENDING": 3,
        "RUNNING": 1,
        "PARKED": 0,
        "COMPLETED": 0,
        "FAILED": 0,
    }


@pytest.mark.asyncio
async def test_queue_counts_zero_rows() -> None:
    conn = FakeConn()
    out = await repo._queue_counts(conn)  # type: ignore[arg-type]
    assert out == {"PENDING": 0, "RUNNING": 0, "PARKED": 0, "COMPLETED": 0, "FAILED": 0}


# ── _last_iterations ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_last_iterations_shape() -> None:
    conn = FakeConn()
    iid = uuid4()
    conn.iter_rows = [
        {
            "iteration_id": iid,
            "strategy_code": "ATR_MOM",
            "iteration_number": 27,
            "statistical_verdict": "SIGNIFICANT_EDGE",
            "verdict": "ITERATE",
            "created_time": datetime.now(timezone.utc),
            "pf": 1.45,
            "n_trades": 132,
        }
    ]
    out = await repo._last_iterations(conn, 5)  # type: ignore[arg-type]
    assert len(out) == 1
    assert out[0]["iteration_id"] == str(iid)
    assert out[0]["strategy_code"] == "ATR_MOM"
    assert out[0]["pf"] == pytest.approx(1.45)
    assert out[0]["n_trades"] == 132


@pytest.mark.asyncio
async def test_last_iterations_handles_null_metrics() -> None:
    conn = FakeConn()
    conn.iter_rows = [
        {
            "iteration_id": uuid4(),
            "strategy_code": "FOO",
            "iteration_number": 1,
            "statistical_verdict": None,
            "verdict": None,
            "created_time": datetime.now(timezone.utc),
            "pf": None,
            "n_trades": None,
        }
    ]
    out = await repo._last_iterations(conn, 5)  # type: ignore[arg-type]
    assert out[0]["pf"] is None
    assert out[0]["n_trades"] is None


# ── _recent_sig_edge_ids ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_recent_sig_edge_ids() -> None:
    conn = FakeConn()
    ids = [uuid4(), uuid4()]
    conn.sig_edge_rows = [{"iteration_id": i} for i in ids]
    out = await repo._recent_sig_edge_ids(conn, 7)  # type: ignore[arg-type]
    assert out == [str(i) for i in ids]


# ── _active_hypotheses ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_active_hypotheses_shape() -> None:
    conn = FakeConn()
    jid = uuid4()
    conn.hypothesis_rows = [
        {
            "journal_id": jid,
            "strategy_code": "ATR_MOM",
            "title": "Tighter ATR-EXT raises MFE/MAE without n collapse",
            "created_time": datetime.now(timezone.utc),
        }
    ]
    out = await repo._active_hypotheses(conn)  # type: ignore[arg-type]
    assert len(out) == 1
    assert out[0]["journal_id"] == str(jid)
    assert out[0]["title"].startswith("Tighter")


# ── _last_run_summary ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_last_run_summary_returns_none_when_empty() -> None:
    conn = FakeConn()
    out = await repo._last_run_summary(conn)  # type: ignore[arg-type]
    assert out is None


@pytest.mark.asyncio
async def test_last_run_summary_populated() -> None:
    conn = FakeConn()
    jid = uuid4()
    conn.run_summary_row = {
        "journal_id": jid,
        "strategy_code": "ATR_MOM",
        "title": "Session 2026-05-18 — 31 iters",
        "content": "Pivoted BTC 1h → ETH 1h.",
        "structured_data": {"iters": 31, "verdicts": {"INSUF": 25, "SIG": 6}},
        "created_time": datetime.now(timezone.utc),
    }
    out = await repo._last_run_summary(conn)  # type: ignore[arg-type]
    assert out is not None
    assert out["journal_id"] == str(jid)
    assert out["structured_data"]["iters"] == 31


# ── _last_null_screen_per_surface ───────────────────────────────────


@pytest.mark.asyncio
async def test_null_screen_extracts_verdict_from_structured_data() -> None:
    conn = FakeConn()
    jid = uuid4()
    conn.null_screen_rows = [
        {
            "journal_id": jid,
            "strategy_code": "ATR_MOM",
            "instrument": "BTCUSDT",
            "interval_name": "4h",
            "title": "Null-screen ATR_MOM BTCUSDT 4h",
            "structured_data": {"verdict": "INCONCLUSIVE", "p75_pf": 0.95},
            "created_time": datetime.now(timezone.utc),
        }
    ]
    out = await repo._last_null_screen_per_surface(conn)  # type: ignore[arg-type]
    assert len(out) == 1
    assert out[0]["verdict"] == "INCONCLUSIVE"
    assert out[0]["instrument"] == "BTCUSDT"
    assert out[0]["interval_name"] == "4h"


@pytest.mark.asyncio
async def test_null_screen_missing_structured_data_yields_none_verdict() -> None:
    conn = FakeConn()
    conn.null_screen_rows = [
        {
            "journal_id": uuid4(),
            "strategy_code": "FOO",
            "instrument": "BTCUSDT",
            "interval_name": "1h",
            "title": "Old screen with no structured_data",
            "structured_data": None,
            "created_time": datetime.now(timezone.utc),
        }
    ]
    out = await repo._last_null_screen_per_surface(conn)  # type: ignore[arg-type]
    assert out[0]["verdict"] is None


# ── get_state_digest (composition) ──────────────────────────────────


@pytest.mark.asyncio
async def test_get_state_digest_composes_all_slices() -> None:
    conn = FakeConn()
    conn.queue_rows = [{"status": "PENDING", "n": 2}]
    conn.iter_rows = [
        {
            "iteration_id": uuid4(),
            "strategy_code": "X",
            "iteration_number": 1,
            "statistical_verdict": "NO_EDGE",
            "verdict": "ITERATE",
            "created_time": datetime.now(timezone.utc),
            "pf": 0.8,
            "n_trades": 110,
        }
    ]
    conn.sig_edge_rows = []
    conn.hypothesis_rows = []
    conn.run_summary_row = None
    conn.null_screen_rows = []

    out = await repo.get_state_digest(conn)  # type: ignore[arg-type]

    assert set(out.keys()) == {
        "queue_counts",
        "last_iterations",
        "recent_sig_edge_iteration_ids",
        "active_hypotheses",
        "last_run_summary",
        "last_null_screen_per_surface",
        "pending_specialist_reviews",
        "recent_specialist_verdicts",
        "ml_training_budget",
        "pending_ml_training_runs",
    }
    assert out["queue_counts"]["PENDING"] == 2
    assert len(out["last_iterations"]) == 1
    assert out["recent_sig_edge_iteration_ids"] == []
    assert out["active_hypotheses"] == []
    assert out["last_run_summary"] is None
    assert out["last_null_screen_per_surface"] == []
    # agent_name omitted in this call → ML budget slices are None.
    assert out["ml_training_budget"] is None
    assert out["pending_ml_training_runs"] is None
    assert out["pending_specialist_reviews"] == []
    assert out["recent_specialist_verdicts"] == []


# ── _pending_specialist_reviews / _recent_specialist_verdicts ───────


@pytest.mark.asyncio
async def test_pending_specialist_reviews_shape() -> None:
    """Path C resume-protocol slice: a candidate iteration with a still-
    open quant-skeptic review must surface in the digest so step 1a can
    detect it and exit again on SPECIALIST_REVIEW_PENDING."""
    conn = FakeConn()
    jid = uuid4()
    iid = uuid4()
    conn.pending_specialist_rows = [
        {
            "journal_id": jid,
            "strategy_code": "ATR_MOM",
            "specialist_name": "quant-skeptic",
            "iteration_id": str(iid),
            "target_id": f"specialist:quant-skeptic:{iid}",
            "created_time": datetime.now(timezone.utc),
        }
    ]
    out = await repo._pending_specialist_reviews(conn)  # type: ignore[arg-type]
    assert len(out) == 1
    assert out[0]["specialist_name"] == "quant-skeptic"
    assert out[0]["iteration_id"] == str(iid)
    assert out[0]["target_id"].startswith("specialist:quant-skeptic:")


@pytest.mark.asyncio
async def test_recent_specialist_verdicts_veto_flag() -> None:
    """is_veto=True on an OVERRIDE_REJECT is the load-bearing field the
    researcher reads to decide whether to journal STRATEGY_OUTCOME and
    pivot. Pin it explicitly."""
    conn = FakeConn()
    iid = uuid4()
    conn.recent_specialist_verdict_rows = [
        {
            "journal_id": uuid4(),
            "strategy_code": "ATR_MOM",
            "specialist_name": "quant-skeptic",
            "iteration_id": str(iid),
            "target_id": f"specialist:quant-skeptic:{iid}",
            "verdict": "OVERRIDE_REJECT",
            "is_veto": True,
            "created_time": datetime.now(timezone.utc),
        }
    ]
    out = await repo._recent_specialist_verdicts(conn)  # type: ignore[arg-type]
    assert len(out) == 1
    assert out[0]["verdict"] == "OVERRIDE_REJECT"
    assert out[0]["is_veto"] is True


@pytest.mark.asyncio
async def test_recent_specialist_verdicts_handles_null_is_veto() -> None:
    """A row written before the is_veto column was added (or by buggy
    upstream code) should default to is_veto=False — the researcher
    must not treat NULL as veto-by-accident."""
    conn = FakeConn()
    conn.recent_specialist_verdict_rows = [
        {
            "journal_id": uuid4(),
            "strategy_code": "ATR_MOM",
            "specialist_name": "quant-portfolio-manager",
            "iteration_id": str(uuid4()),
            "target_id": "specialist:quant-portfolio-manager:abc",
            "verdict": "ADD",
            "is_veto": None,
            "created_time": datetime.now(timezone.utc),
        }
    ]
    out = await repo._recent_specialist_verdicts(conn)  # type: ignore[arg-type]
    assert out[0]["is_veto"] is False
