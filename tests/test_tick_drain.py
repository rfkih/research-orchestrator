"""Tests for ``services/tick_drain.py`` — the server-side runner.

The drain wraps ``services.tick.run_tick`` in a loop; we monkeypatch
the run_tick reference inside ``services.tick_drain`` and assert the
shaped digest matches the runner playbook (terminal_action / iters /
verdicts / handoff sentence).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from orchestrator.errors import OrchestratorError
from orchestrator.services import tick_drain
from orchestrator.services.tick import TickResult


def _tick_iter(
    *,
    statistical_verdict: str,
    verdict: str | None = None,
    iteration_id: str = "iter-1",
) -> TickResult:
    return TickResult(
        outcome="iterated",
        queue_id="q-1",
        iteration_id=iteration_id,
        backtest_run_id="br-1",
        statistical_verdict=statistical_verdict,
        verdict=verdict,
        notes=[],
        next_actions=[{"kind": "call", "method": "POST", "path": "/tick"}],
        pf=1.3,
        n_trades=120,
    )


def _tick_graduate() -> TickResult:
    # statistical_verdict + verdict together pass through compute_tick_summary
    # to next_action=GRADUATE.
    return TickResult(
        outcome="iterated",
        queue_id="q-1",
        iteration_id="iter-graduate",
        backtest_run_id="br-graduate",
        statistical_verdict="SIGNIFICANT_EDGE",
        verdict="PASS",
        notes=[],
        next_actions=[
            {"kind": "call", "method": "POST", "path": "/walk-forward"},
        ],
        pf=2.1,
        n_trades=180,
    )


def _tick_empty_queue() -> TickResult:
    return TickResult(
        outcome="empty_queue",
        queue_id=None,
        iteration_id=None,
        backtest_run_id=None,
        statistical_verdict=None,
        verdict=None,
        notes=["queue empty"],
        next_actions=[{"kind": "call", "method": "POST", "path": "/queue"}],
    )


def _tick_pivot() -> TickResult:
    return TickResult(
        outcome="sweep_exhausted",
        queue_id="q-1",
        iteration_id=None,
        backtest_run_id=None,
        statistical_verdict=None,
        verdict=None,
        notes=["sweep exhausted"],
        next_actions=[{"kind": "read_doc", "doc_anchor": "queue.sweep_exhausted"}],
    )


class _FakeRunTick:
    """Replays a scripted sequence of TickResult / OrchestratorError objects."""

    def __init__(self, sequence: list[Any]) -> None:
        self._iter: Iterator[Any] = iter(sequence)
        self.calls = 0

    async def __call__(self, **kwargs: Any) -> TickResult:
        self.calls += 1
        try:
            item = next(self._iter)
        except StopIteration as exc:
            raise AssertionError(
                "drain_ticks called run_tick more times than the test scripted."
            ) from exc
        if isinstance(item, OrchestratorError):
            raise item
        return item


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace asyncio.sleep with an immediate no-op so WAIT branches don't
    actually delay the test suite. The drain service imports asyncio inline."""
    async def _sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(tick_drain.asyncio, "sleep", _sleep)


@pytest.mark.asyncio
async def test_drain_terminates_on_graduate(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeRunTick(
        [
            _tick_iter(statistical_verdict="INSUFFICIENT_EVIDENCE"),
            _tick_iter(statistical_verdict="NO_EDGE"),
            _tick_graduate(),
        ]
    )
    monkeypatch.setattr(tick_drain, "run_tick", fake)

    digest = await tick_drain.drain_ticks(
        db=None, jvm=None, settings=None,
        agent_name="quant-researcher", session_id=None, redis_client=None,
        max_iters=10,
    )

    assert digest["terminal_action"] == "GRADUATE"
    assert digest["iters_completed"] == 3
    assert digest["verdicts"]["insuf"] == 1
    assert digest["verdicts"]["no_edge"] == 1
    assert digest["verdicts"]["sig"] == 1
    assert digest["last_iteration_id"] == "iter-graduate"
    assert "graduation review" in digest["handoff_sentence"].lower()
    assert fake.calls == 3


@pytest.mark.asyncio
async def test_drain_terminates_on_empty_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeRunTick([_tick_empty_queue()])
    monkeypatch.setattr(tick_drain, "run_tick", fake)

    digest = await tick_drain.drain_ticks(
        db=None, jvm=None, settings=None,
        agent_name="quant-researcher", session_id=None, redis_client=None,
        max_iters=10,
    )

    assert digest["terminal_action"] == "EMPTY_QUEUE"
    assert digest["iters_completed"] == 0
    assert "POST /queue" in digest["handoff_sentence"]


@pytest.mark.asyncio
async def test_drain_terminates_on_sweep_exhausted_as_pivot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeRunTick(
        [
            _tick_iter(statistical_verdict="NO_EDGE", verdict="DISCARD"),
            _tick_pivot(),
        ]
    )
    monkeypatch.setattr(tick_drain, "run_tick", fake)

    digest = await tick_drain.drain_ticks(
        db=None, jvm=None, settings=None,
        agent_name="quant-researcher", session_id=None, redis_client=None,
        max_iters=10,
    )

    assert digest["terminal_action"] == "PIVOT"
    assert digest["verdicts"]["discard"] == 1
    assert digest["iters_completed"] == 1


@pytest.mark.asyncio
async def test_drain_catches_uncaught_exception_from_run_tick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_tick can raise non-OrchestratorError (e.g. asyncpg.PostgresError
    from _poll_backtest_status's fetchrow). The drain must return a partial
    digest with terminal_action=INFRA_FAIL, not propagate the exception and
    500 the whole endpoint."""

    async def crashy(**_kwargs: Any) -> TickResult:
        # Simulate an asyncpg.PostgresError surfacing through run_tick — not
        # wrapped in OrchestratorError because _poll_backtest_status doesn't
        # catch it.
        raise RuntimeError("asyncpg connection died mid-poll")

    monkeypatch.setattr(tick_drain, "run_tick", crashy)

    digest = await tick_drain.drain_ticks(
        db=None, jvm=None, settings=None,
        agent_name="quant-researcher", session_id=None, redis_client=None,
        max_iters=10,
    )

    assert digest["terminal_action"] == "INFRA_FAIL"
    err = digest["last_error"]
    assert err["error_code"] == "tick_uncaught_exception"
    assert err["exception_class"] == "RuntimeError"
    assert "asyncpg connection died" in err["message"]


@pytest.mark.asyncio
async def test_drain_treats_run_tick_error_as_infra_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    err = OrchestratorError(
        status_code=502,
        error_code="backtest_did_not_complete",
        message="JVM returned FAILED",
        retryable=False,
    )
    fake = _FakeRunTick([_tick_iter(statistical_verdict="NO_EDGE"), err])
    monkeypatch.setattr(tick_drain, "run_tick", fake)

    digest = await tick_drain.drain_ticks(
        db=None, jvm=None, settings=None,
        agent_name="quant-researcher", session_id=None, redis_client=None,
        max_iters=10,
    )

    assert digest["terminal_action"] == "INFRA_FAIL"
    assert digest["last_error"]["error_code"] == "backtest_did_not_complete"
    # The non-erroring tick before the failure still counts.
    assert digest["iters_completed"] == 1


@pytest.mark.asyncio
async def test_drain_caps_at_max_iters(monkeypatch: pytest.MonkeyPatch) -> None:
    # Five CONTINUE ticks, asked to cap at 3.
    fake = _FakeRunTick(
        [_tick_iter(statistical_verdict="INSUFFICIENT_EVIDENCE") for _ in range(5)]
    )
    monkeypatch.setattr(tick_drain, "run_tick", fake)

    digest = await tick_drain.drain_ticks(
        db=None, jvm=None, settings=None,
        agent_name="quant-researcher", session_id=None, redis_client=None,
        max_iters=3,
    )

    assert digest["terminal_action"] == "MAX_ITERS_REACHED"
    assert digest["iters_completed"] == 3
    assert fake.calls == 3
    next_actions = digest["next_actions"]
    assert any(a.get("path") == "/tick/drain" for a in next_actions)


@pytest.mark.asyncio
async def test_drain_escalates_three_consecutive_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Each "wait" comes back as a tick whose next_action is WAIT. Build a
    # TickResult that produces next_action=WAIT via the summary computer:
    # outcome != iterated and next_actions has a retry hint with wait_s.
    wait_result = TickResult(
        outcome="transient_db_error",
        queue_id=None,
        iteration_id=None,
        backtest_run_id=None,
        statistical_verdict=None,
        verdict=None,
        notes=["wait"],
        next_actions=[{"kind": "retry", "wait_s": 0.01}],
    )
    fake = _FakeRunTick([wait_result, wait_result, wait_result])
    monkeypatch.setattr(tick_drain, "run_tick", fake)

    digest = await tick_drain.drain_ticks(
        db=None, jvm=None, settings=None,
        agent_name="quant-researcher", session_id=None, redis_client=None,
        max_iters=10,
        max_consecutive_waits=3,
    )

    assert digest["terminal_action"] == "INFRA_FAIL"
    assert fake.calls == 3


def test_default_caps_are_sane() -> None:
    # Pinning so a refactor doesn't silently raise the wall-clock cap
    # past what an unattended drain should hold.
    assert tick_drain.DEFAULT_MAX_ITERS == 40
    assert tick_drain.DEFAULT_MAX_WALL_CLOCK_S == 10800
    assert tick_drain.DEFAULT_MAX_CONSECUTIVE_WAITS == 3
