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
    # No DSR passed (None) → no pool route; first action is a fresh hypothesis.
    assert actions[0]["path"] == "/queue"


@pytest.mark.asyncio
async def test_exhausted_near_miss_with_strong_dsr_routes_to_pool(calls: dict) -> None:
    # Signal Pool lane (Phase 0): an exhausted sub-10% SIGNIFICANT_EDGE whose
    # DSR already clears the pool's validity bar is routed to a pool-candidate
    # walk-forward (then /pool/evaluate) instead of being silently discarded.
    actions = await tick_mod._decide_next_state(
        db=_FakeDb(), queue_id="q4", new_iter=10, iter_budget=10,
        early_stop=False, statistical_verdict="SIGNIFICANT_EDGE",
        decision_verdict="ITERATE", agent_name="quant-researcher",
        iteration_id="11111111-1111-1111-1111-111111111111", dsr=0.97,
    )
    # Still honestly completed as a standalone non-graduation.
    assert calls["complete"] == [("q4", "INSUFFICIENT_EVIDENCE")]
    assert actions[0]["path"] == "/walk-forward"
    assert "pool_candidate" in actions[0]["hint"]
    assert "11111111-1111-1111-1111-111111111111" in actions[0]["hint"]
    # The fresh-hypothesis path is still offered as the fallback.
    assert actions[-1]["path"] == "/queue"


@pytest.mark.asyncio
async def test_exhausted_near_miss_with_weak_dsr_no_pool_route(calls: dict) -> None:
    # DSR below the pool validity bar → don't burn ~3h of walk-forward; just
    # enqueue a fresh hypothesis.
    actions = await tick_mod._decide_next_state(
        db=_FakeDb(), queue_id="q5", new_iter=10, iter_budget=10,
        early_stop=False, statistical_verdict="SIGNIFICANT_EDGE",
        decision_verdict="ITERATE", agent_name="quant-researcher",
        iteration_id="22222222-2222-2222-2222-222222222222", dsr=0.80,
    )
    assert [a["path"] for a in actions] == ["/queue"]
