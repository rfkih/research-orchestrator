"""``POST /tick/drain`` — server-side runner.

Replaces the quant-runner sub-agent's polling loop with an HTTP call.
The researcher no longer needs the ``Agent`` tool to keep ticking; one
POST drains an APPROVED sweep until a terminal ``summary.next_action``
is reached, then returns a structured digest matching the runner's
6-line shape.

Why this is safe to run in-process:
  * Every ``run_tick`` call honours the same V11 / V60 / portfolio /
    re-discovery gates as the standalone ``POST /tick`` — moving the
    loop into the orchestrator does NOT change what gets enqueued or
    journaled. The reviewer gate at ``POST /queue`` already ran before
    the queue rows existed.
  * Activity logging (``TICK_DISPATCHED`` / ``ITERATION_COMPLETED``)
    happens inside ``run_tick``; the drain wrapper does not duplicate
    those rows.
  * Idempotency: the drain endpoint itself caches its full response so
    replays are no-ops, BUT each interior ``run_tick`` call is NOT
    keyed with an Idempotency-Key (the queue's ``FOR UPDATE SKIP
    LOCKED`` claim is the durability primitive). A drain that crashes
    mid-loop leaves the queue in a clean state — ticks that completed
    are persisted, the in-flight one rolls back to PENDING via the
    same trap that protects the standalone endpoint.

Wall-clock budget: ``max_wall_clock_s`` is enforced as a soft cap
checked between ticks. A single ``run_tick`` can still take up to the
backtest poll cap (~30 min) so the actual wall time is bounded by
``max_iters × poll_cap``; callers wanting tighter control should set
``max_iters`` instead of relying on the wall-clock cap alone.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from uuid import UUID

from redis.asyncio import Redis as AsyncRedis

from ..clients.jvm import JvmClient
from ..config import Settings
from ..errors import OrchestratorError
from ..infra.db import Database
from ..logging import get_logger
from .tick import run_tick

log = get_logger(__name__)

# Defaults. ``max_iters`` is the runner playbook's working assumption
# (a sweep is 20-40 ticks); ``max_wall_clock_s`` mirrors a single
# poll-cap window so a single empty-queue drain returns fast and a
# stuck-row drain doesn't hold the connection for hours. The runner
# spec's three-consecutive-WAIT cap is preserved.
DEFAULT_MAX_ITERS = 40
DEFAULT_MAX_WALL_CLOCK_S = 10800
DEFAULT_MAX_CONSECUTIVE_WAITS = 3


# Error codes that represent a non-self-healing config gap (operator must
# act) rather than a transient infra outage (wait + retry). Drained ticks
# hitting one of these terminate as CONFIG_FAIL, not INFRA_FAIL.
_CONFIG_FAIL_ERROR_CODES: frozenset[str] = frozenset({
    "account_strategy_missing",
    "python_interpreter_missing",
    "unknown_spec_name",
})


_HANDOFF_BY_TERMINAL: dict[str, str] = {
    "GRADUATE": (
        "Request graduation review on iteration_id=<id> before /walk-forward."
    ),
    "PIVOT": (
        "Pick next archetype/surface; current axis is exhausted or DISCARDed."
    ),
    "EMPTY_QUEUE": "Enqueue next sweep via POST /queue.",
    "INFRA_FAIL": "Journal INFRA_FAILURE; resume after orchestrator/JVM/DB healthy.",
    "CONFIG_FAIL": (
        "Non-retryable config gap (see last_error.error_code) — operator "
        "action required; the failure will NOT self-heal on retry. Journal "
        "an INFRA_FAILURE with a fix_pointer and escalate."
    ),
    "MAX_ITERS_REACHED": (
        "Loop hit max_iters without a terminal verdict. Re-call /tick/drain to "
        "continue, or inspect the queue for stalled rows."
    ),
    "MAX_WALL_CLOCK_REACHED": (
        "Loop hit max_wall_clock_s without a terminal verdict. Re-call /tick/drain "
        "to continue, or inspect the queue for stalled rows."
    ),
}


async def drain_ticks(
    *,
    db: Database,
    jvm: JvmClient,
    settings: Settings,
    agent_name: str,
    session_id: UUID | None,
    redis_client: AsyncRedis | None,
    max_iters: int = DEFAULT_MAX_ITERS,
    max_wall_clock_s: int = DEFAULT_MAX_WALL_CLOCK_S,
    max_consecutive_waits: int = DEFAULT_MAX_CONSECUTIVE_WAITS,
) -> dict[str, Any]:
    """Drive ``run_tick`` until terminal or a cap fires.

    Returns the digest dict the API layer renders verbatim.
    """
    tally = {
        "iters": 0,
        "sig": 0,
        "insuf": 0,
        "no_edge": 0,
        "discard": 0,
        "last_iteration_id": None,
        "last_verdict_line": "",
        "last_decision_hint": "",
        "last_tick_outcome": None,
        "last_summary": None,
        "last_error": None,
    }
    terminal_action: str | None = None
    consecutive_waits = 0
    deadline = time.monotonic() + max_wall_clock_s

    for _ in range(max_iters):
        if time.monotonic() >= deadline:
            terminal_action = "MAX_WALL_CLOCK_REACHED"
            break

        try:
            result = await run_tick(
                db=db,
                jvm=jvm,
                settings=settings,
                agent_name=agent_name,
                session_id=session_id,
                redis_client=redis_client,
            )
        except OrchestratorError as err:
            envelope = err.envelope
            log.warn(
                "tick_drain.tick_error",
                error_code=envelope.error_code,
                retryable=envelope.retryable,
            )
            tally["last_error"] = envelope.model_dump(exclude_none=True)
            tally["last_verdict_line"] = (
                f"Tick failed: {envelope.error_code} — {envelope.message[:200]}"
            )
            # Specific config-gap error codes won't self-heal on retry —
            # route them to CONFIG_FAIL whose handoff tells the researcher to
            # escalate immediately instead of waiting for DB/JVM to recover.
            # (Kept as an explicit allowlist, not a blanket retryable=False
            # check, so genuine infra errors like backtest_did_not_complete
            # still read as INFRA_FAIL.)
            terminal_action = (
                "CONFIG_FAIL"
                if envelope.error_code in _CONFIG_FAIL_ERROR_CODES
                else "INFRA_FAIL"
            )
            break
        except Exception as exc:  # noqa: BLE001
            # Unwrapped exception from run_tick — typically asyncpg.PostgresError
            # from the backtest-status poll, or an asyncio cancellation that
            # crossed the boundary. run_tick has already rolled back the queue
            # row in its own ``except Exception`` arm (tick.py L389-392), so
            # the queue stays clean. We just need to return the partial digest
            # instead of letting the exception 500 the whole drain endpoint.
            log.error(
                "tick_drain.tick_uncaught_exception",
                exception_class=type(exc).__name__,
            )
            tally["last_error"] = {
                "error_code": "tick_uncaught_exception",
                "exception_class": type(exc).__name__,
                "message": str(exc)[:500],
                "retryable": False,
            }
            tally["last_verdict_line"] = (
                f"Tick crashed: {type(exc).__name__} — {str(exc)[:200]}"
            )
            terminal_action = "INFRA_FAIL"
            break

        payload = result.to_dict()
        summary = payload.get("summary") or {}
        next_action = summary.get("next_action")
        outcome = payload.get("outcome")
        sv = payload.get("statistical_verdict")
        verdict = payload.get("verdict")

        tally["last_tick_outcome"] = outcome
        tally["last_summary"] = summary
        tally["last_verdict_line"] = summary.get("verdict_line") or ""
        tally["last_decision_hint"] = summary.get("decision_hint") or ""

        if outcome == "iterated":
            tally["iters"] += 1
            if sv == "SIGNIFICANT_EDGE":
                tally["sig"] += 1
            elif sv == "INSUFFICIENT_EVIDENCE":
                tally["insuf"] += 1
            elif sv == "NO_EDGE":
                tally["no_edge"] += 1
            if verdict == "DISCARD":
                tally["discard"] += 1
            iteration_id = payload.get("iteration_id")
            if iteration_id:
                tally["last_iteration_id"] = str(iteration_id)

        if next_action == "GRADUATE":
            terminal_action = "GRADUATE"
            break
        if next_action == "PIVOT":
            terminal_action = "PIVOT"
            break
        if next_action == "EMPTY_QUEUE":
            terminal_action = "EMPTY_QUEUE"
            break
        if next_action == "INFRA_FAIL":
            terminal_action = "INFRA_FAIL"
            break
        if next_action == "WAIT":
            consecutive_waits += 1
            if consecutive_waits >= max_consecutive_waits:
                terminal_action = "INFRA_FAIL"
                tally["last_verdict_line"] = (
                    f"{max_consecutive_waits} consecutive WAIT responses; "
                    "escalating to INFRA_FAIL."
                )
                break
            wait_s = _resolve_wait_seconds(payload.get("next_actions"))
            if time.monotonic() + wait_s >= deadline:
                terminal_action = "MAX_WALL_CLOCK_REACHED"
                break
            await asyncio.sleep(wait_s)
            continue

        # CONTINUE (or any unknown next_action) — keep draining.
        consecutive_waits = 0

    if terminal_action is None:
        terminal_action = "MAX_ITERS_REACHED"

    return _shape_digest(tally, terminal_action)


def _resolve_wait_seconds(next_actions: list[dict[str, Any]] | None) -> float:
    """Read the WAIT hint's ``wait_s`` payload, falling back to 30.0.

    Mirrors the runner playbook: the orchestrator returns a ``retry``
    next-action with ``wait_s`` on transient infra errors; in their
    absence default to 30 seconds (the legacy ``research-tick.sh``
    poll interval).
    """
    for a in next_actions or []:
        if not isinstance(a, dict):
            continue
        if a.get("kind") == "retry":
            try:
                return float(a.get("wait_s") or 30.0)
            except (TypeError, ValueError):
                return 30.0
    return 30.0


def _shape_digest(tally: dict[str, Any], terminal_action: str) -> dict[str, Any]:
    """Render the runner's 6-line shape as JSON.

    The fields parallel the digest the haiku-tier quant-runner used to
    return so the researcher's branch table stays identical:

      Status / Iters / Verdicts / Last (iteration_id + verdict_line) /
      Next (handoff sentence).
    """
    handoff = _HANDOFF_BY_TERMINAL.get(
        terminal_action, "Inspect drain digest manually."
    )
    return {
        "terminal_action": terminal_action,
        "iters_completed": tally["iters"],
        "verdicts": {
            "sig": tally["sig"],
            "insuf": tally["insuf"],
            "no_edge": tally["no_edge"],
            "discard": tally["discard"],
        },
        "last_iteration_id": tally["last_iteration_id"],
        "last_verdict_line": tally["last_verdict_line"],
        "last_decision_hint": tally["last_decision_hint"],
        "last_tick_outcome": tally["last_tick_outcome"],
        "last_summary": tally["last_summary"],
        "last_error": tally["last_error"],
        "handoff_sentence": handoff,
        "next_actions": _next_actions_for_terminal(terminal_action),
    }


def _next_actions_for_terminal(terminal_action: str) -> list[dict[str, Any]]:
    if terminal_action == "GRADUATE":
        return [
            {
                "kind": "call",
                "method": "POST",
                "path": "/reviews/auto-run-checklist",
                "hint": (
                    "Submit target_kind='graduation' with the GRADUATE "
                    "iteration_id before /walk-forward."
                ),
            }
        ]
    if terminal_action == "EMPTY_QUEUE":
        return [
            {
                "kind": "call",
                "method": "POST",
                "path": "/queue",
                "hint": "Enqueue the next sweep.",
            }
        ]
    if terminal_action in ("MAX_ITERS_REACHED", "MAX_WALL_CLOCK_REACHED"):
        return [
            {
                "kind": "call",
                "method": "POST",
                "path": "/tick/drain",
                "hint": "Re-call to continue draining the same queue.",
            }
        ]
    if terminal_action == "PIVOT":
        return [
            {
                "kind": "read_doc",
                "doc_anchor": "researcher.pivot_archetype",
                "hint": "Pick the next archetype/axis combo.",
            }
        ]
    if terminal_action == "INFRA_FAIL":
        return [
            {
                "kind": "contact_human",
                "hint": "Inspect orchestrator/JVM/DB health; journal INFRA_FAILURE.",
            }
        ]
    return []
