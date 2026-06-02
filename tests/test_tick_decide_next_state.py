"""Unit tests for tick._decide_next_state — the queue next-state machine.

Regression coverage for the 2026-06-02 CRITICAL bug (rank 3): a
SIGNIFICANT_EDGE iteration whose decision_verdict misses the V60 economic
gate must KEEP ITERATING (reset to PENDING), not PARK. Parking it stranded a
valid stat-edge sweep because the tick still returned a CONTINUE-shaped
next_action while the queue went empty → drain terminated EMPTY_QUEUE.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest

from orchestrator.services import tick as tick_mod


class _FakeDb:
    @asynccontextmanager
    async def acquire(self):
        yield object()  # conn sentinel — repo calls are monkeypatched


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> dict:
    rec: dict[str, list] = {"park": [], "reset": [], "complete": []}

    async def _park(conn, queue_id, note):
        rec["park"].append(queue_id)

    async def _reset(conn, queue_id, note):
        rec["reset"].append(queue_id)

    async def _complete(conn, queue_id, *, final_verdict, walk_forward_id, note):
        rec["complete"].append((queue_id, final_verdict))

    async def _paper(db, queue_id, agent_name, state):
        return None

    monkeypatch.setattr(tick_mod.queue_write, "park_exhausted", _park)
    monkeypatch.setattr(tick_mod.queue_write, "reset_to_pending", _reset)
    monkeypatch.setattr(tick_mod.queue_write, "complete_queue", _complete)
    monkeypatch.setattr(tick_mod, "_try_generate_paper", _paper)
    return rec


@pytest.mark.asyncio
async def test_sig_edge_v60_pass_parks_for_walk_forward(calls: dict) -> None:
    actions = await tick_mod._decide_next_state(
        db=_FakeDb(), queue_id="q1", new_iter=3, iter_budget=10,
        early_stop=False, statistical_verdict="SIGNIFICANT_EDGE",
        decision_verdict="PASS", agent_name="quant-researcher",
    )
    assert calls["park"] == ["q1"]
    assert calls["reset"] == []
    assert actions[0]["path"] == "/walk-forward"


@pytest.mark.asyncio
async def test_sig_edge_v60_fail_keeps_iterating(calls: dict) -> None:
    # Rank-3 regression: stat edge real but economic return < 10%/yr and
    # budget remains → continue the sweep, do NOT park.
    actions = await tick_mod._decide_next_state(
        db=_FakeDb(), queue_id="q2", new_iter=3, iter_budget=10,
        early_stop=False, statistical_verdict="SIGNIFICANT_EDGE",
        decision_verdict="ITERATE", agent_name="quant-researcher",
    )
    assert calls["park"] == []
    assert calls["reset"] == ["q2"]
    assert actions[0]["path"] == "/tick"


@pytest.mark.asyncio
async def test_sig_edge_v60_fail_budget_exhausted_completes(calls: dict) -> None:
    actions = await tick_mod._decide_next_state(
        db=_FakeDb(), queue_id="q3", new_iter=10, iter_budget=10,
        early_stop=False, statistical_verdict="SIGNIFICANT_EDGE",
        decision_verdict="ITERATE", agent_name="quant-researcher",
    )
    assert calls["park"] == []
    assert calls["complete"] == [("q3", "INSUFFICIENT_EVIDENCE")]
    assert actions[0]["path"] == "/queue"
