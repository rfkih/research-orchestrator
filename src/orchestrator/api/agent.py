"""Agent-discovery endpoints.

The quant-researcher agent is the primary caller. It boots cold with no
prior session memory. These endpoints exist so the agent can:

  * ``GET /agent/playbook`` — read the contract (auth, headers, error codes,
    workflow recipes) without me having to bake them into the system prompt.
  * ``GET /agent/state`` — get a one-shot snapshot of "what's the state of
    research right now?" (queue depth, in-flight ops, last verdict). Phase 1
    returns only the lightweight bits; Phase 2 lights up the queries.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(prefix="/agent", tags=["agent"])


class PlaybookCapability(BaseModel):
    name: str
    method: str
    path: str
    purpose: str
    idempotent: bool


class PlaybookRecipe(BaseModel):
    name: str
    when: str
    steps: list[str]


class Playbook(BaseModel):
    version: str
    auth: dict[str, str]
    headers: dict[str, str]
    error_envelope: dict[str, str]
    capabilities: list[PlaybookCapability]
    recipes: list[PlaybookRecipe]


@router.get("/playbook", response_model=Playbook)
async def playbook() -> Playbook:
    """Discoverable contract. Public on purpose — no secrets here, only shape."""
    return Playbook(
        version="1",
        auth={
            "scheme": "shared-secret",
            "header": "X-Orch-Token",
            "note": "Send on every call except /healthz, /readyz, /agent/playbook.",
        },
        headers={
            "X-Agent-Name": "Optional. Stamped onto created_by. Default 'anonymous'.",
            "Idempotency-Key": "Optional. Required for POST /tick to make retries safe.",
        },
        error_envelope={
            "shape": "{error_code, message, retryable, hint, next_action, details}",
            "branch_on": "error_code (stable) and retryable (boolean).",
            "do_not_parse": "message — prose may change between releases.",
        },
        capabilities=[
            PlaybookCapability(
                name="liveness",
                method="GET",
                path="/healthz",
                purpose="Process is up. No I/O.",
                idempotent=True,
            ),
            PlaybookCapability(
                name="readiness",
                method="GET",
                path="/readyz",
                purpose="DB + JVM reachable. Probe before issuing /tick.",
                idempotent=True,
            ),
            PlaybookCapability(
                name="agent_state",
                method="GET",
                path="/agent/state",
                purpose="One-shot snapshot of research state.",
                idempotent=True,
            ),
            PlaybookCapability(
                name="list_queue",
                method="GET",
                path="/queue",
                purpose=(
                    "List research_queue rows, ordered by (priority, created_time) "
                    "ASC — same order the claim loop will pick. "
                    "Filters: status, strategy_code. Cursor-paginated."
                ),
                idempotent=True,
            ),
            PlaybookCapability(
                name="get_queue_row",
                method="GET",
                path="/queue/{queue_id}",
                purpose="Single research_queue row, including sweep_config.",
                idempotent=True,
            ),
            PlaybookCapability(
                name="list_iterations",
                method="GET",
                path="/iterations",
                purpose=(
                    "List research_iteration_log rows newest-first. "
                    "Filters: strategy_code, verdict. Cursor-paginated."
                ),
                idempotent=True,
            ),
            PlaybookCapability(
                name="get_iteration",
                method="GET",
                path="/iterations/{iteration_id}",
                purpose=(
                    "Full iteration row including params_snapshot, "
                    "metrics_snapshot, confidence_intervals, hypothesis match."
                ),
                idempotent=True,
            ),
            PlaybookCapability(
                name="leaderboard",
                method="GET",
                path="/leaderboard",
                purpose=(
                    "Iteration ranking. sort=pf|return_pct|sharpe|trade_count|created. "
                    "Pass significant_only=true to filter to SIGNIFICANT_EDGE."
                ),
                idempotent=True,
            ),
            PlaybookCapability(
                name="list_journal",
                method="GET",
                path="/journal",
                purpose=(
                    "Lessons / hypotheses / anti-patterns. Read on cold-boot to "
                    "avoid re-running ruled-out sweeps. Free-text search on ?search=."
                ),
                idempotent=True,
            ),
            PlaybookCapability(
                name="get_journal_entry",
                method="GET",
                path="/journal/{journal_id}",
                purpose="Single research_journal row.",
                idempotent=True,
            ),
            PlaybookCapability(
                name="enqueue_sweep",
                method="POST",
                path="/queue",
                purpose=(
                    "Insert a PENDING research_queue row. Body: strategy_code, "
                    "interval_name, instrument, sweep_config (params grid), "
                    "hypothesis, iter_budget. Honours Idempotency-Key. "
                    "Re-discovery gate: 409 axis_previously_discarded if this "
                    "strategy has already produced a DISCARD verdict on the "
                    "same axis-set; pass override_discard_gate=true with a "
                    "documented justification to bypass."
                ),
                idempotent=False,
            ),
            PlaybookCapability(
                name="cancel_queue_row",
                method="POST",
                path="/queue/{queue_id}/cancel",
                purpose=(
                    "Mark a PENDING/RUNNING row as PARKED with a reason note. "
                    "Already-COMPLETED rows return 409 queue_not_cancellable."
                ),
                idempotent=False,
            ),
            PlaybookCapability(
                name="run_tick",
                method="POST",
                path="/tick",
                purpose=(
                    "Execute one research iteration: claim queue row, record "
                    "hypothesis_audit row (cumulative trial count), submit "
                    "backtest, poll, write iteration_log with DSR using the "
                    "real cumulative trial count, decide next state. "
                    "Synchronous; up to 30 min for the JVM to finish. "
                    "Honours Idempotency-Key. Outcomes: iterated, empty_queue, "
                    "sweep_exhausted. metrics_snapshot.analysis includes "
                    "dsr (Deflated Sharpe per Bailey & Lopez de Prado 2014) "
                    "and dsr_n_trials (the real selection-bias multiplicity) "
                    "alongside the legacy psr field."
                ),
                idempotent=False,
            ),
            PlaybookCapability(
                name="run_walk_forward",
                method="POST",
                path="/walk-forward",
                purpose=(
                    "Validate a SIGNIFICANT_EDGE iteration with a 6-fold "
                    "rolling-window walk-forward (Bailey & López de Prado 2014). "
                    "Long-running: up to 3 hours (6 folds × 30min). Returns "
                    "stability_verdict ∈ {ROBUST, INCONSISTENT, OVERFIT, NO_EDGE, "
                    "INSUFFICIENT_EVIDENCE}. ROBUST is the gate for graduation."
                ),
                idempotent=False,
            ),
        ],
        recipes=[
            PlaybookRecipe(
                name="cold-boot",
                when="Agent has no prior session context.",
                steps=[
                    "GET /agent/playbook",
                    "GET /readyz — bail if status != 'ok'",
                    "GET /agent/state — read queue depth + last verdict",
                    "GET /journal?status=ACTIVE&entry_type=ANTI_PATTERN — load lessons",
                    "GET /leaderboard?significant_only=true&limit=10 — proven edge",
                ],
            ),
            PlaybookRecipe(
                name="reproduce-iteration",
                when="Inspecting why a past iteration earned its verdict.",
                steps=[
                    "GET /iterations/{iteration_id} — params + metrics + confidence",
                    "If statistical_verdict='SIGNIFICANT_EDGE' but verdict='ITERATE',",
                    "  re-read change_reasoning + diagnostics to find the gap.",
                ],
            ),
            PlaybookRecipe(
                name="run-one-iteration",
                when="The cron tick fires, or the agent wants to advance research.",
                steps=[
                    "GET /readyz — bail if degraded",
                    "POST /tick (with Idempotency-Key for safe retry)",
                    "Read response.outcome:",
                    "  'empty_queue'      → consider POST /queue with a new sweep",
                    "  'sweep_exhausted'  → review iterations, enqueue refined sweep",
                    "  'iterated'         → follow next_actions[] from the response",
                ],
            ),
            PlaybookRecipe(
                name="enqueue-then-run",
                when="Agent has a fresh hypothesis to test.",
                steps=[
                    "GET /journal?status=ACTIVE&entry_type=ANTI_PATTERN — confirm not previously discarded",
                    "POST /queue with sweep_config + hypothesis + Idempotency-Key",
                    "POST /tick — first iteration runs immediately",
                ],
            ),
            PlaybookRecipe(
                name="validate-significant-edge",
                when="A /tick returned statistical_verdict='SIGNIFICANT_EDGE' and parked the queue.",
                steps=[
                    "GET /iterations/{iteration_id} — confirm metrics + params_snapshot",
                    "POST /walk-forward with strategy_code, interval_name, motivating_iteration_id (and overrides matching params_snapshot if non-default)",
                    "Inspect response.stability_verdict:",
                    "  'ROBUST'        → strategy passes; promote per graduation rule",
                    "  'OVERFIT'/'INCONSISTENT' → re-design with regularisation, POST /queue",
                    "  'NO_EDGE'/'INSUFFICIENT_EVIDENCE' → abandon or extend window",
                ],
            ),
            PlaybookRecipe(
                name="retry-on-error",
                when="Any non-2xx response.",
                steps=[
                    "Parse JSON body as ErrorEnvelope.",
                    "If retryable=true and next_action.kind=='retry', wait next_action.wait_s and replay.",
                    "If retryable=false, do NOT replay. Follow next_action (read_doc / call / contact_human).",
                ],
            ),
        ],
    )


class AgentState(BaseModel):
    """Phase-1 placeholder. Phase 2 fills in queue + iteration counts."""

    agent: str
    profile: str
    db_ok: bool
    jvm_ok: bool
    notes: list[str]
    next_actions: list[dict[str, Any]]


@router.get("/state", response_model=AgentState)
async def agent_state(request: Request) -> AgentState:
    db_ok = await request.app.state.db.health_probe()
    jvm_ok = await request.app.state.jvm.health_probe()
    notes: list[str] = []
    next_actions: list[dict[str, Any]] = []
    if not db_ok:
        notes.append("DB pool is unreachable — read endpoints will 503.")
        next_actions.append({"kind": "contact_human"})
    if not jvm_ok:
        notes.append("Research JVM is unreachable — /tick will fail.")
        next_actions.append({"kind": "retry", "wait_s": 30.0})
    if not notes:
        notes.append("Service is healthy. Phase-2 endpoints (/queue, /iterations) ship next.")
    return AgentState(
        agent=request.state.agent_name,
        profile=request.app.state.settings.profile,
        db_ok=db_ok,
        jvm_ok=jvm_ok,
        notes=notes,
        next_actions=next_actions,
    )
