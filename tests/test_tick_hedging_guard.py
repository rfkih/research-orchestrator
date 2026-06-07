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


# ── Fix 2: evaluate() exception must NOT escape; maps to ITERATE ────────────


async def test_hedging_gate_exception_maps_to_iterate(monkeypatch):
    """When hedging_gate.evaluate raises, _select_track_gate must NOT propagate
    the exception. It maps to the same non-PASS ITERATE/continue outcome a
    hedging FAIL produces and stashes an auditable error marker on the payload.
    """
    async def _boom(*args, **kwargs):
        raise RuntimeError("simulated hedging gate failure")

    monkeypatch.setattr(tick.hedging_gate, "evaluate", _boom)

    stat_v, dec_v, payload = await tick._select_track_gate(
        db=_DbShim(),
        track="hedging",
        # Trading-path verdicts that, if leaked through, would NOT be PASS —
        # but the point is the hedging branch overrides + must not raise.
        stat_v="NO_EDGE",
        dec_v="DISCARD",
        backtest_run_id=uuid.uuid4(),
        instrument="BTCUSDT",
        window_start=datetime(2024, 1, 1),
        window_end=datetime(2024, 4, 1),
    )

    # Non-PASS ITERATE outcome (same as a hedging FAIL) — never PASS, no raise.
    assert dec_v == "ITERATE"
    assert stat_v == "SIGNIFICANT_EDGE"
    # Auditable error marker on the payload (surfaces on
    # metrics_snapshot.hedging_gate).
    assert payload is not None
    assert payload["passed"] is False
    assert "error" in payload
    assert "simulated hedging gate failure" in payload["error"]


async def test_hedging_gate_exception_never_marks_pass(monkeypatch):
    async def _boom(*args, **kwargs):
        raise ValueError("kaboom")

    monkeypatch.setattr(tick.hedging_gate, "evaluate", _boom)
    _, dec_v, payload = await tick._select_track_gate(
        db=_DbShim(),
        track="hedging",
        stat_v="SIGNIFICANT_EDGE",
        dec_v="PASS",  # even if the trading inputs say PASS, error must not pass
        backtest_run_id=uuid.uuid4(),
        instrument="ETHUSDT",
        window_start=datetime(2024, 1, 1),
        window_end=datetime(2024, 4, 1),
    )
    assert dec_v != "PASS"
    assert payload["passed"] is False


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
        return None

    async def _noop_reset(conn, queue_id, note):
        return None

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
