"""Queue-integrity fixes (2026-06-12): poll-cap resume semantics.

Unit tests for the H2/M1 repair in services/tick.py — a poll-cap expiry
must hand the row back for RESUME (PENDING, run id kept, retryable error),
never terminalize it to FAILED, and an already-logged iteration must be
ADOPTED rather than duplicated.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

import pytest

from orchestrator.errors import OrchestratorError
from orchestrator.services import tick as tick_mod


class _FakeDb:
    @asynccontextmanager
    async def acquire(self):
        yield object()


def _poll_kwargs(**over):
    base = dict(
        db=_FakeDb(),
        settings=_FakeSettings(),
        agent_name="tester",
        session_id=None,
        redis_client=None,
        queue_id=uuid.uuid4(),
        strategy_code="DCB",
        interval_name="1h",
        instrument="BTCUSDT",
        sweep_config={},
        new_iter=2,
        iter_budget=5,
        early_stop=True,
        started_at="2026-06-12T00:00:00",
        backtest_run_id=str(uuid.uuid4()),
        combo={"p": "1"},
        audit_id=uuid.uuid4(),
        cumulative_trials=3,
    )
    base.update(over)
    return base


class _FakeSettings:
    poll_timeout_s = 3600
    research_account_id = None


@pytest.mark.asyncio
async def test_poll_cap_expiry_resets_for_resume_not_failed(monkeypatch):
    """Run still RUNNING at poll-cap expiry → rollback to PENDING
    (run id survives for resume) + retryable error. The historical bug
    called fail_queue (a dead end — no requeue path) with retryable=False
    while the JVM run completed minutes later, orphaned."""
    calls = {"rollback": 0, "fail": 0}

    async def _poll(db, run_id, timeout_s=None):
        return "RUNNING"

    async def _rollback(conn, queue_id, note, *, expected_status=None):
        calls["rollback"] += 1
        assert expected_status == "RUNNING"
        return 1

    async def _fail(conn, queue_id, note, *, expected_status=None):
        calls["fail"] += 1
        return 1

    monkeypatch.setattr(tick_mod, "_poll_backtest_status", _poll)
    monkeypatch.setattr(tick_mod.queue_write, "rollback_iter", _rollback)
    monkeypatch.setattr(tick_mod.queue_write, "fail_queue", _fail)

    with pytest.raises(OrchestratorError) as exc_info:
        await tick_mod._poll_and_finish(**_poll_kwargs())

    envelope = exc_info.value.envelope
    assert envelope.error_code == "backtest_poll_cap_exceeded"
    assert envelope.retryable is True
    # The handler wrote the queue state itself — outer rollback must skip.
    assert getattr(exc_info.value, "queue_terminalized", False) is True
    assert calls == {"rollback": 1, "fail": 0}


@pytest.mark.asyncio
async def test_jvm_reported_failure_still_terminalizes(monkeypatch):
    calls = {"rollback": 0, "fail": 0}

    async def _poll(db, run_id, timeout_s=None):
        return "FAILED"

    async def _rollback(conn, queue_id, note, *, expected_status=None):
        calls["rollback"] += 1
        return 1

    async def _fail(conn, queue_id, note, *, expected_status=None):
        calls["fail"] += 1
        assert expected_status == "RUNNING"
        return 1

    monkeypatch.setattr(tick_mod, "_poll_backtest_status", _poll)
    monkeypatch.setattr(tick_mod.queue_write, "rollback_iter", _rollback)
    monkeypatch.setattr(tick_mod.queue_write, "fail_queue", _fail)

    with pytest.raises(OrchestratorError) as exc_info:
        await tick_mod._poll_and_finish(**_poll_kwargs())

    envelope = exc_info.value.envelope
    assert envelope.error_code == "backtest_did_not_complete"
    assert envelope.retryable is False
    assert calls == {"rollback": 0, "fail": 1}


@pytest.mark.asyncio
async def test_completed_run_with_existing_iteration_is_adopted(monkeypatch):
    """A prior attempt logged the iteration but died before the pointer
    update. The resuming tick must ADOPT that row (attach + next-state),
    not insert a duplicate — there is no UNIQUE(backtest_run_id) to save
    us from a double insert."""
    existing_iteration_id = uuid.uuid4()
    attach_calls: list[dict] = []

    async def _find(conn, run_id):
        return {
            "iteration_id": existing_iteration_id,
            "statistical_verdict": "NO_EDGE",
            "verdict": "DISCARD",
            "sample_size_adequate": True,
            "metrics_snapshot": {"analysis": {"dsr": 0.5, "n_trades": 42}},
        }

    async def _attach(conn, queue_id, *, iteration_id, backtest_run_id,
                      expected_started_at=None):
        attach_calls.append({"iteration_id": iteration_id})
        return 1

    async def _complete(conn, queue_id, *, final_verdict, walk_forward_id,
                        note, expected_status=None):
        return 1

    async def _paper(db, queue_id, agent_name, state):
        return None

    async def _log_activity(conn, **kwargs):
        return None

    monkeypatch.setattr(tick_mod.iterations_write, "find_iteration_for_run", _find)
    monkeypatch.setattr(tick_mod.queue_write, "attach_iteration", _attach)
    monkeypatch.setattr(tick_mod.queue_write, "complete_queue", _complete)
    monkeypatch.setattr(tick_mod, "_try_generate_paper", _paper)
    monkeypatch.setattr(tick_mod, "log_activity", _log_activity)

    result = await tick_mod._poll_and_finish(
        **_poll_kwargs(skip_poll=True)
    )

    assert attach_calls == [{"iteration_id": existing_iteration_id}]
    assert result.iteration_id == str(existing_iteration_id)
    assert result.statistical_verdict == "NO_EDGE"
    assert result.verdict == "DISCARD"


@pytest.mark.asyncio
async def test_fence_mismatch_keeps_iteration_and_raises_retryable(monkeypatch):
    """attach affecting 0 rows (reaper recovery / re-claim) must raise a
    RETRYABLE 409 whose message tells the truth: the iteration row IS
    committed (transaction A) and the next tick adopts it."""
    async def _attach(conn, queue_id, *, iteration_id, backtest_run_id,
                      expected_started_at=None):
        return 0

    monkeypatch.setattr(tick_mod.queue_write, "attach_iteration", _attach)

    with pytest.raises(OrchestratorError) as exc_info:
        await tick_mod._attach_fenced(
            _FakeDb(),
            queue_id=uuid.uuid4(),
            iteration_id=uuid.uuid4(),
            backtest_run_id=str(uuid.uuid4()),
            started_at="2026-06-12T00:00:00",
        )

    envelope = exc_info.value.envelope
    assert envelope.error_code == "queue_row_reclaimed"
    assert envelope.retryable is True
    assert "IS committed" in envelope.message
    assert getattr(exc_info.value, "queue_terminalized", False) is True
