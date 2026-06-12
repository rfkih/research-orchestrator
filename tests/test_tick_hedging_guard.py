"""Unit tests for the hedging-track defensive guard (Fix 2) and the
trade-level Signal-Pool near-miss track gate (Fix 3) in services/tick.py.

These are DB-free: the hedging gate's evaluate() is monkeypatched, and the
pool-routing branch is exercised through _decide_next_state with a stubbed
db/queue layer. The trading path is unaffected (it never reaches either change).
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import datetime

import pytest

from orchestrator.services import tick


class _DbShim:
    """Minimal Database stand-in: .acquire() yields a sentinel conn."""

    @asynccontextmanager
    async def acquire(self):
        yield object()


# ── M2 (2026-06-12): evaluate() exception is an INFRA failure, not a verdict ─


async def test_hedging_gate_exception_raises_retryable(monkeypatch):
    """When hedging_gate.evaluate raises, _select_track_gate must raise a
    RETRYABLE OrchestratorError — never fabricate a ("SIGNIFICANT_EDGE",
    "ITERATE") verdict. The old mapping wrote the fabricated verdict into the
    append-only audit tables, consumed an iteration of budget, and (on the
    last budgeted iteration) terminalized the queue off a DB blip. The outer
    tick handler rolls the row back to PENDING; the next tick resumes the
    same completed run via active_backtest_run_id and re-runs only the gate.
    """
    from orchestrator.errors import OrchestratorError

    async def _boom(*args, **kwargs):
        raise RuntimeError("simulated hedging gate failure")

    monkeypatch.setattr(tick.hedging_gate, "evaluate", _boom)

    with pytest.raises(OrchestratorError) as exc_info:
        await tick._select_track_gate(
            db=_DbShim(),
            track="hedging",
            stat_v="NO_EDGE",
            dec_v="DISCARD",
            backtest_run_id=uuid.uuid4(),
            instrument="BTCUSDT",
            window_start=datetime(2024, 1, 1),
            window_end=datetime(2024, 4, 1),
        )
    envelope = exc_info.value.envelope
    assert envelope.error_code == "hedging_gate_error"
    assert envelope.retryable is True
    # Not tagged terminal: the outer handler must run its PENDING rollback.
    assert not getattr(exc_info.value, "queue_terminalized", False)


async def test_hedging_gate_exception_never_marks_pass(monkeypatch):
    from orchestrator.errors import OrchestratorError

    async def _boom(*args, **kwargs):
        raise ValueError("kaboom")

    monkeypatch.setattr(tick.hedging_gate, "evaluate", _boom)
    # Even if the trading inputs say PASS, the error must raise — there is
    # no code path that converts a gate exception into any verdict at all.
    with pytest.raises(OrchestratorError):
        await tick._select_track_gate(
            db=_DbShim(),
            track="hedging",
            stat_v="SIGNIFICANT_EDGE",
            dec_v="PASS",
            backtest_run_id=uuid.uuid4(),
            instrument="ETHUSDT",
            window_start=datetime(2024, 1, 1),
            window_end=datetime(2024, 4, 1),
        )


# ── Fix 3: hedging FAIL must NOT enter the trade-level Signal-Pool lane ──────


def _has_pool_candidate(actions: list[dict]) -> bool:
    """A pool-candidate near-miss action is a POST /walk-forward whose hint
    mentions the trade-level pool ('Pool candidate' / 'pool_candidate')."""
    for a in actions:
        hint = a.get("hint", "")
        if a.get("path") == "/walk-forward" and (
            "Pool candidate" in hint or "pool_candidate" in hint
        ):
            return True
    return False


@pytest.fixture
def _stub_decide_io(monkeypatch):
    """Stub the DB-write + paper-gen side effects of _decide_next_state so the
    routing decision can be exercised without a live DB."""
    async def _noop_complete(conn, queue_id, **kwargs):
        return 1

    async def _noop_reset(conn, queue_id, note, *, expected_status=None):
        return 1

    async def _noop_paper(db, queue_id, agent_name, state):
        return None

    monkeypatch.setattr(tick.queue_write, "complete_queue", _noop_complete)
    monkeypatch.setattr(tick.queue_write, "reset_to_pending", _noop_reset)
    monkeypatch.setattr(tick, "_try_generate_paper", _noop_paper)


# A DSR comfortably above the pool's validity bar — so the ONLY thing that can
# keep a candidate out of the pool lane is the track gate.
_HIGH_DSR = tick.analyze.DSR_SIGNIFICANCE_THRESHOLD + 1.0


async def test_hedging_fail_does_not_route_to_pool_candidate(_stub_decide_io):
    """track=hedging row that FAILs the hedging gate (SIGNIFICANT_EDGE+ITERATE,
    budget exhausted, high trade-level DSR) must NOT enter the trade-level pool
    near-miss branch — it just continues/ITERATEs (enqueue a fresh hypothesis).
    """
    actions = await tick._decide_next_state(
        db=_DbShim(),
        queue_id=uuid.uuid4(),
        new_iter=5,
        iter_budget=5,  # exhausted → hits the ITERATE_EXHAUSTED block
        early_stop=False,
        statistical_verdict="SIGNIFICANT_EDGE",
        decision_verdict="ITERATE",
        agent_name="tester",
        iteration_id=uuid.uuid4(),
        dsr=_HIGH_DSR,
        track="hedging",
    )
    assert not _has_pool_candidate(actions), (
        "hedging FAIL must NOT be routed into the trade-level Signal Pool"
    )
    # It still continues the search (enqueue a fresh hypothesis).
    assert any(a.get("path") == "/queue" for a in actions)


@pytest.mark.parametrize("track", [None, "trading"])
async def test_trading_track_still_routes_to_pool_candidate(_stub_decide_io, track):
    """Regression guard: the trading path (track None/'trading') with a high DSR
    STILL routes the near-miss into the trade-level pool — Fix 3 only excludes
    hedging."""
    actions = await tick._decide_next_state(
        db=_DbShim(),
        queue_id=uuid.uuid4(),
        new_iter=5,
        iter_budget=5,
        early_stop=False,
        statistical_verdict="SIGNIFICANT_EDGE",
        decision_verdict="ITERATE",
        agent_name="tester",
        iteration_id=uuid.uuid4(),
        dsr=_HIGH_DSR,
        track=track,
    )
    assert _has_pool_candidate(actions), (
        "trading near-miss with high DSR must still reach the pool lane"
    )
