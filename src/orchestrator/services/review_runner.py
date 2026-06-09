"""Auto-reviewer orchestration — gather artifacts, run the checklist.

Replaces the quant-reviewer sub-agent spawn at the plan-review and
graduation-review gates with a pure HTTP-driven equivalent: the
orchestrator pulls the same journal/iteration rows the reviewer
playbook curls for, feeds them into the existing pure-function
checklists in ``services/review.py``, and aggregates the verdict.

The reviewer playbook documents two paths (Run them programmatically
via ``python -c …`` OR apply by hand). This module is the
programmatic path lifted server-side so the loop no longer needs a
nested-spawn-capable harness.

Loss of capability vs. a human reviewer sub-agent: the qualitative
APPROVED → CONDITIONAL_APPROVAL downgrade is gone — the verdict here
is exactly what ``aggregate_verdict`` returns. The cross-strategy
pattern-spotting (e.g. "8 archetypes in 24h, agent is fishing") is
also gone; if that capability matters, an operator runs a manual
reviewer pass with the old workflow. The mechanical contract is what
the orchestrator's ``/queue`` and ``/walk-forward`` gates have always
keyed on, so research-loop safety is unchanged.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

import asyncpg

from ..errors import NextAction, OrchestratorError
from ..repo import iterations as iterations_repo
from ..repo import journal as journal_repo
from ..repo import reviews as reviews_repo
from .review import (
    aggregate_verdict,
    graduation_review_checklist,
    plan_review_checklist,
)

# Identity stamped onto the verdict row when the orchestrator auto-runs
# the checklist. Distinct from the human ``quant-reviewer`` agent so the
# audit trail makes the source of the verdict obvious. The /queue gate
# does NOT branch on this — it keys on the verdict string only — so the
# stamp is for forensics, not gating.
AUTO_REVIEWER_AGENT = "research-orchestrator-auto-reviewer"

# Window for the "axis recently failed" check. Matches the default in
# ``check_axis_not_recently_failed`` so a 15-day-old STRATEGY_OUTCOME
# isn't counted as recent.
_RECENT_OUTCOME_LOOKBACK_DAYS = 14
_RECENT_OUTCOME_LIMIT = 50

# Sweep-history cap for the param-robustness check. The check only
# needs the points sharing the same param set as the candidate; an
# arbitrary cap keeps the query bounded for noisy strategies that have
# accumulated thousands of iterations.
_SWEEP_HISTORY_LIMIT = 200


async def _fetch_request_payload(
    conn: asyncpg.Connection, target_id: str, target_kind: str
) -> dict[str, Any]:
    """Latest review-request journal row for ``target_id``.

    The auto-reviewer needs the request payload (axis_names,
    hypothesis_id, iteration_id, …) because the checklist functions
    take those as arguments — they don't re-derive them from the
    target_id string. Falls back to a 404 envelope when no request
    has been submitted.
    """
    row = await conn.fetchrow(
        """
        SELECT journal_id, strategy_code, structured_data, created_time
          FROM research_journal
         WHERE entry_type = 'IDEA_BACKLOG'
           AND structured_data->>'kind' = $1
           AND structured_data->>'target_id' = $2
         ORDER BY created_time DESC
         LIMIT 1
        """,
        reviews_repo.KIND_REQUEST,
        target_id,
    )
    if row is None:
        raise OrchestratorError(
            status_code=404,
            error_code="review_request_not_found",
            message=(
                f"No /reviews/request row found for target_id={target_id}. "
                "POST /reviews/request first, then re-run the auto-checklist."
            ),
            retryable=False,
            hint=(
                "The auto-reviewer reads the request payload off the journal "
                "row written by /reviews/request. Without that row it cannot "
                "know which axis_names / iteration_id to audit."
            ),
            next_action=NextAction(
                kind="call",
                method="POST",
                path="/reviews/request",
            ),
            details={"target_id": target_id, "target_kind": target_kind},
        )
    sd = row["structured_data"] or {}
    if sd.get("target_kind") != target_kind:
        raise OrchestratorError(
            status_code=409,
            error_code="review_target_kind_mismatch",
            message=(
                f"Request row's target_kind={sd.get('target_kind')!r} does not "
                f"match the auto-checklist's target_kind={target_kind!r}."
            ),
            retryable=False,
        )
    payload = dict(sd.get("request_payload") or {})
    return {
        "journal_id": str(row["journal_id"]),
        "strategy_code": row["strategy_code"],
        "created_time": row["created_time"],
        "payload": payload,
        "structured_data": sd,
    }


async def _gather_plan_artifacts(
    conn: asyncpg.Connection,
    *,
    strategy_code: str,
    hypothesis_id: str,
    plan_created_time: datetime,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Returns (hypothesis_entry, prior_strategy_outcomes)."""
    hypothesis: dict[str, Any] | None = None
    # The reviewer playbook accepts hypothesis_id as a journal UUID. Some
    # callers may pass a non-UUID string (test fixtures); skip the lookup
    # rather than 500 on the cast — the pre_registration check then fires
    # the missing-hypothesis blocker, which is the right outcome.
    try:
        hyp_uuid = UUID(str(hypothesis_id))
    except (TypeError, ValueError):
        hyp_uuid = None
    if hyp_uuid is not None:
        hypothesis = await journal_repo.get_journal(conn, hyp_uuid)

    prior = await journal_repo.list_journal(
        conn,
        entry_type="STRATEGY_OUTCOME",
        status="ACTIVE",
        strategy_code=strategy_code,
        search=None,
        sort_by="created_time",
        sort_dir="desc",
        after_value=None,
        after_created_time=None,
        after_id=None,
        limit=_RECENT_OUTCOME_LIMIT,
    )
    cutoff = plan_created_time - timedelta(days=_RECENT_OUTCOME_LOOKBACK_DAYS)
    # ``created_time`` is timezone-aware on Postgres; harmonise so the
    # comparison never raises on a naive vs aware mismatch in fixtures.
    def _aware(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    cutoff_aware = _aware(cutoff)
    recent = [
        r for r in prior
        if r.get("created_time") and _aware(r["created_time"]) >= cutoff_aware
    ]
    return hypothesis, recent


async def _gather_graduation_artifacts(
    conn: asyncpg.Connection,
    *,
    strategy_code: str,
    iteration_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[tuple[dict[str, Any], float]]]:
    """Returns (metrics, ci, params, sweep_history)."""
    try:
        iter_uuid = UUID(str(iteration_id))
    except (TypeError, ValueError) as exc:
        raise OrchestratorError(
            status_code=400,
            error_code="iteration_id_invalid",
            message=f"iteration_id={iteration_id!r} is not a UUID.",
            retryable=False,
        ) from exc
    iteration = await iterations_repo.get_iteration(conn, iter_uuid)
    if iteration is None:
        raise OrchestratorError(
            status_code=404,
            error_code="iteration_not_found",
            message=(
                f"iteration_id={iteration_id} not in research_iteration_log; "
                "cannot run graduation checklist on a missing iteration."
            ),
            retryable=False,
        )

    metrics = iteration.get("metrics_snapshot") or {}
    ci = iteration.get("confidence_intervals") or {}
    params = iteration.get("params_snapshot") or {}

    # Sweep history for param-robustness: every iteration on this
    # strategy with a usable profit_factor. The robustness check
    # filters to Hamming-1 neighbours of the optimum cell, so passing
    # rows that don't share the candidate's param keys is harmless —
    # they're skipped by the diff count.
    history_rows = await iterations_repo.list_iterations(
        conn,
        strategy_code=strategy_code,
        verdict=None,
        after_created_time=None,
        after_id=None,
        limit=_SWEEP_HISTORY_LIMIT,
    )
    sweep_history: list[tuple[dict[str, Any], float]] = []
    for r in history_rows:
        p = dict(r.get("params_snapshot") or {})
        if not p:
            continue
        m = r.get("metrics_snapshot") or {}
        pf = m.get("profit_factor")
        if pf is None:
            continue
        try:
            sweep_history.append((p, float(pf)))
        except (TypeError, ValueError):
            continue
    return metrics, ci, params, sweep_history


async def auto_run_checklist(
    conn: asyncpg.Connection,
    *,
    target_id: str,
    target_kind: str,
) -> dict[str, Any]:
    """Pure server-side reviewer pass.

    Returns the aggregate-verdict shape augmented with the source
    request id so the caller knows which row was audited:

      {
        "verdict": "APPROVED" | "CONDITIONAL_APPROVAL" | "REJECTED",
        "reason": "...",
        "n_checks": int,
        "n_blocker_fails": int,
        "n_warning_fails": int,
        "checks": [...],
        "motivating_request_id": "<uuid>",
        "strategy_code": "MMR",
      }
    """
    if target_kind not in ("plan", "graduation"):
        raise OrchestratorError(
            status_code=400,
            error_code="bad_target_kind",
            message=f"target_kind must be 'plan' or 'graduation', got {target_kind!r}.",
            retryable=False,
        )

    request = await _fetch_request_payload(conn, target_id, target_kind)
    payload = request["payload"]
    strategy_code = request["strategy_code"] or payload.get("strategy_code")

    if target_kind == "plan":
        hypothesis_id = payload.get("hypothesis_id")
        axis_names = list(payload.get("axis_names") or [])
        if not hypothesis_id:
            raise OrchestratorError(
                status_code=400,
                error_code="plan_payload_missing_hypothesis_id",
                message=(
                    "Plan review request payload is missing hypothesis_id — "
                    "cannot run pre_registration check."
                ),
                retryable=False,
            )
        hypothesis, recent_outcomes = await _gather_plan_artifacts(
            conn,
            strategy_code=strategy_code or "",
            hypothesis_id=hypothesis_id,
            plan_created_time=request["created_time"],
        )
        checks = plan_review_checklist(
            hypothesis_entry=hypothesis,
            plan_created_time=request["created_time"],
            proposed_axis_names=axis_names,
            prior_strategy_outcomes=recent_outcomes,
        )
    else:  # graduation
        iteration_id = payload.get("iteration_id")
        if not iteration_id:
            raise OrchestratorError(
                status_code=400,
                error_code="graduation_payload_missing_iteration_id",
                message=(
                    "Graduation review request payload is missing iteration_id "
                    "— cannot run graduation checklist."
                ),
                retryable=False,
            )
        metrics, ci, params, sweep_history = await _gather_graduation_artifacts(
            conn,
            strategy_code=strategy_code or "",
            iteration_id=str(iteration_id),
        )
        # Track resolution (2026-06-09): AUTHORITATIVE source is the originating
        # queue's sweep_config track (repo.resolve_track), the same authority the
        # tick gate uses. Fall back to the ``hedging_gate`` metrics marker only
        # when the queue linkage is absent (legacy rows), so a hedge scored via
        # the trade-fallback path is still classified as hedging. Used to escalate
        # the overfit-signature checks to blockers.
        track = await iterations_repo.resolve_track(conn, UUID(str(iteration_id)))
        if track is None and isinstance(metrics, dict) and "hedging_gate" in metrics:
            track = "hedging"
        checks = graduation_review_checklist(
            iteration_metrics=metrics,
            iteration_ci=ci,
            iteration_params=params,
            sweep_history=sweep_history,
            track=track,
        )

    aggregate = aggregate_verdict(checks)
    aggregate["motivating_request_id"] = request["journal_id"]
    aggregate["strategy_code"] = strategy_code
    aggregate["target_id"] = target_id
    aggregate["target_kind"] = target_kind
    return aggregate
