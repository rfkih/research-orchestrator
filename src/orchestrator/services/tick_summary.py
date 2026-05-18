"""Distill a ``/tick`` response into a 3-field summary for the runner.

The runner (Stage B, haiku-tier agent) reads ONLY ``summary.next_action``
to decide whether to keep ticking or escalate to the researcher. That
means the summary computer below is load-bearing — a bug here causes the
runner to either spin forever (missing a terminal state) or stop too
early (missing edge that needed one more tick).

Pure function. No DB, no JVM, no Pydantic. Tests live in
``tests/test_tick_summary.py``.

The six ``next_action`` values exhaust the runner's branch table:

  * CONTINUE     — keep ticking; runner stays in loop.
  * GRADUATE     — SIG_EDGE + econ gate (verdict=PASS); researcher must
                   request a graduation review.
  * PIVOT        — sweep exhausted OR strategy DISCARDed; researcher must
                   pick a new archetype / surface.
  * EMPTY_QUEUE  — no PENDING row to claim; researcher must POST /queue.
  * WAIT         — transient infra hiccup; runner sleeps + retries.
  * INFRA_FAIL   — non-retryable infra failure; runner escalates to
                   researcher, researcher journals INFRA_FAILURE.

The pf+n_trades line in ``verdict_line`` is intentionally compact — at
runner-tier costs we don't want a paragraph per tick.
"""

from __future__ import annotations

from typing import Any, Literal

NextAction = Literal[
    "CONTINUE",
    "GRADUATE",
    "PIVOT",
    "EMPTY_QUEUE",
    "WAIT",
    "INFRA_FAIL",
]


def _retry_hint(next_actions: list[dict[str, Any]]) -> bool:
    """Inspect the orchestrator's next_actions[] for a retry hint.

    ``OrchestratorError`` returns ``next_action.kind == 'retry'`` with a
    ``wait_s`` payload on transient failures (DB unreachable, JVM
    degraded). Anything else (``contact_human``, ``read_doc``, ``call``)
    is non-retryable from the runner's POV.
    """
    for a in next_actions or []:
        kind = a.get("kind") if isinstance(a, dict) else None
        if kind == "retry":
            return True
    return False


def compute_tick_summary(
    *,
    outcome: str,
    statistical_verdict: str | None,
    verdict: str | None,
    next_actions: list[dict[str, Any]] | None,
    pf: float | None = None,
    n_trades: int | None = None,
) -> dict[str, str]:
    """Return ``{verdict_line, next_action, decision_hint}``.

    ``outcome`` is the orchestrator's coarse outcome enum from
    ``services/tick.py``: ``iterated``, ``empty_queue``, ``sweep_exhausted``,
    or an error sentinel surfaced through the error envelope.
    """
    nas = next_actions or []

    if outcome == "empty_queue":
        return {
            "verdict_line": "Queue empty — no PENDING row to claim.",
            "next_action": "EMPTY_QUEUE",
            "decision_hint": "POST /queue with a new sweep, or pivot archetype.",
        }

    if outcome == "sweep_exhausted":
        return {
            "verdict_line": "Sweep exhausted — iter_budget consumed.",
            "next_action": "PIVOT",
            "decision_hint": (
                "Review iteration_log + leaderboard; pick next axis or archetype."
            ),
        }

    # Any non-iterated, non-known outcome falls through to infra branching.
    if outcome != "iterated":
        if _retry_hint(nas):
            return {
                "verdict_line": f"Transient infra failure (outcome={outcome}).",
                "next_action": "WAIT",
                "decision_hint": "Sleep per next_actions[0].wait_s, then replay tick.",
            }
        return {
            "verdict_line": f"Non-retryable infra failure (outcome={outcome}).",
            "next_action": "INFRA_FAIL",
            "decision_hint": "Escalate to researcher; journal INFRA_FAILURE.",
        }

    # outcome == 'iterated' — branch on the verdict pair.
    metrics = []
    if pf is not None:
        metrics.append(f"PF={pf:.2f}")
    if n_trades is not None:
        metrics.append(f"n={n_trades}")
    metrics_str = " ".join(metrics)

    if verdict == "PASS":
        return {
            "verdict_line": (
                f"SIG_EDGE + econ gate cleared ({metrics_str})." if metrics_str
                else "SIG_EDGE + econ gate cleared."
            ),
            "next_action": "GRADUATE",
            "decision_hint": (
                "Request graduation review (POST /reviews/request, "
                "target_kind='graduation')."
            ),
        }

    if verdict == "DISCARD":
        return {
            "verdict_line": (
                f"DISCARD ({statistical_verdict or 'no edge'}, {metrics_str})."
                if metrics_str
                else f"DISCARD ({statistical_verdict or 'no edge'})."
            ),
            "next_action": "PIVOT",
            "decision_hint": "Strategy ruled out on this axis; pick next archetype.",
        }

    if statistical_verdict == "SIGNIFICANT_EDGE":
        # SIG_EDGE but verdict != PASS means the V60 economic gate missed
        # (annualized ann_geom@90% < 10%). Researcher can keep ticking but
        # this cell is not a graduation candidate.
        return {
            "verdict_line": (
                f"SIG_EDGE without econ gate ({metrics_str})."
                if metrics_str
                else "SIG_EDGE without econ gate."
            ),
            "next_action": "CONTINUE",
            "decision_hint": (
                "Stat edge real but ann_geom<10%; keep sweeping for a "
                "stronger cell."
            ),
        }

    if statistical_verdict == "NO_EDGE":
        return {
            "verdict_line": f"NO_EDGE ({metrics_str})." if metrics_str else "NO_EDGE.",
            "next_action": "CONTINUE",
            "decision_hint": "Continue sweep; pivot if cells run out.",
        }

    if statistical_verdict == "INSUFFICIENT_EVIDENCE":
        return {
            "verdict_line": (
                f"INSUF ({metrics_str})." if metrics_str else "INSUF (n likely<100)."
            ),
            "next_action": "CONTINUE",
            "decision_hint": (
                "Trade count likely below V11 threshold; continue or widen window."
            ),
        }

    # Catch-all: iterated but neither statistical_verdict nor verdict matched.
    # Treat as CONTINUE — the runner can keep going and the next tick will
    # produce a more conclusive verdict pair.
    return {
        "verdict_line": (
            f"Iterated; verdict={verdict}, stat={statistical_verdict}."
        ),
        "next_action": "CONTINUE",
        "decision_hint": "Keep ticking; verdict pair did not match known terminals.",
    }
