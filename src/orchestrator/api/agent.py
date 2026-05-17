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
    tooling: dict[str, str]
    capabilities: list[PlaybookCapability]
    recipes: list[PlaybookRecipe]


@router.get("/playbook", response_model=Playbook)
async def playbook() -> Playbook:
    """Discoverable contract. Public on purpose — no secrets here, only shape."""
    return Playbook(
        version="2",
        auth={
            "scheme": "shared-secret",
            "header": "X-Orch-Token",
            "note": "Send on every call except /healthz, /readyz, /agent/playbook.",
        },
        headers={
            "X-Agent-Name": "Optional. Stamped onto created_by. Default 'anonymous'.",
            "Idempotency-Key": "Optional. Required for POST /tick to make retries safe.",
            "X-Session-Id": (
                "Optional UUID. When provided, activity rows written by /tick, "
                "/queue, /walk-forward, /reviews/* are grouped under this session. "
                "When omitted, the orchestrator synthesizes a deterministic "
                "per-(agent, UTC date) UUID — all calls from one agent on one UTC "
                "day land in the same session automatically. Pass an explicit UUID "
                "only if you want finer grouping than per-day. "
                "Use POST /activity to log SESSION_START, SESSION_END, "
                "HYPOTHESIS_REGISTERED, PLAN_WRITTEN, GOAL_HIT events. "
                "GET /activity/sessions returns session-level summaries."
            ),
        },
        error_envelope={
            "shape": "{error_code, message, retryable, hint, next_action, details}",
            "branch_on": "error_code (stable) and retryable (boolean).",
            "do_not_parse": "message — prose may change between releases.",
        },
        tooling={
            "wrapper": (
                "research-orchestrator/scripts/orch.sh — generic HTTP wrapper for "
                "GET/HEAD/POST. Resolves ORCH_AUTH_TOKEN, sets X-Orch-Token + "
                "X-Agent-Name, defaults host to 127.0.0.1:8082. Invoke with "
                "relative or absolute path from ANY cwd — DO NOT `cd` first. "
                "Leading `cd && scripts/orch.sh ... | ...` trips a CC security "
                "guardrail no allow rule can override."
            ),
            "wrapper_examples_read": (
                "scripts/orch.sh GET /readyz | "
                "scripts/orch.sh GET /agent/state --pretty | "
                "scripts/orch.sh GET '/queue?limit=20' --pretty | "
                "scripts/orch.sh GET '/journal?status=ACTIVE&entry_type=ANTI_PATTERN' --pretty"
            ),
            "wrapper_examples_post": (
                "scripts/orch.sh POST /journal --body /tmp/body.json --ik my-key | "
                "scripts/orch.sh POST /queue --body /tmp/queue.json --ik queue-$(date +%s) | "
                "scripts/orch.sh POST /tick --body /tmp/tick.json --ik tick-$(uuidgen) --pretty | "
                "scripts/orch.sh POST /walk-forward --body /tmp/wf.json --ik wf-$(uuidgen)"
            ),
            "do_not_use_inline_curl_for_reads": (
                "Inline `cd ... && TOKEN=$(grep ... | cut ...) && curl -s -H ...` "
                "trips the harness permission prompt every time. Use scripts/orch.sh "
                "for reads. Inline curl is reserved for state-changing POSTs that "
                "need a JSON body (/queue, /tick, /walk-forward, /reviews/*)."
            ),
            "do_not_call_trading_jvm": (
                "Trading JVM (:8080) is OFF-LIMITS — it owns the live book. Never "
                "POST /api/v1/users/login or any other Trading JVM endpoint. "
                "Research JVM (:8081) IS allowed: use POST /api/v1/dev/login-as "
                "(dev profile only) when an endpoint isn't proxied through the "
                "orchestrator. Prefer the orchestrator (:8082) for the main loop — "
                "it owns JVM auth internally for /tick, /iterations/*, "
                "/walk-forward. Hitting :8081 directly is an escape hatch; if it "
                "becomes routine, journal a DATA_WISHLIST for the missing proxy."
            ),
            "json_body_pattern": (
                "For POST bodies, create the file via the agent's Write tool "
                "(not via bash heredoc), then call `scripts/orch.sh POST <path> "
                "--body <file> [--ik <key>] [--pretty]`. Three harness checks "
                "block the obvious approaches: (1) inline `-d '{...}'` triggers "
                "parser error `Unhandled node type: string`; (2) `cat > body.json "
                "<<'EOF' {...} EOF` triggers `Contains brace with quote character "
                "(expansion obfuscation)` because CC's static parser scans the "
                "heredoc body; (3) multi-line `RESP=$(curl -X POST \\ -H ... \\ "
                "-H ... ...)` with backslash-continued quoted headers also trips "
                "string-node parser errors. The wrapper's POST mode plus a "
                "Write-tool body file avoids all three — the bash command "
                "contains no JSON, no heredoc, no multi-line header soup."
            ),
            "post_pattern": (
                "ORCH_BASE=http://127.0.0.1:8082; "
                "TOKEN=$(grep ^ORCH_AUTH_TOKEN research-orchestrator/.env | cut -d= -f2-); "
                "RESP=$(curl -s -X POST -H \"X-Orch-Token: $TOKEN\" "
                "-H \"X-Agent-Name: quant-researcher\" "
                "-H \"Idempotency-Key: $(uuidgen)\" "
                "-H \"Content-Type: application/json\" "
                "-d @body.json \"$ORCH_BASE/<path>\")"
            ),
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
                    "interval_name, instrument, sweep_config, hypothesis, "
                    "iter_budget. sweep_config.strategy is 'grid' (default; "
                    "params[].values cross-product) or 'tpe' (Bayesian; "
                    "params[].type+low+high or type+values, n_trials budget, "
                    "seed). Honours Idempotency-Key. Re-discovery gate: 409 "
                    "axis_previously_discarded if this strategy has already "
                    "produced a DISCARD verdict on the same axis-set; pass "
                    "override_discard_gate=true with a documented "
                    "justification to bypass."
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
                    "alongside the legacy psr field. Portfolio gate (added "
                    "2026-05-05): SIGNIFICANT_EDGE candidates are demoted to "
                    "INSUFFICIENT_EVIDENCE when their daily-return correlation "
                    "with the protected book (LSR/VCB/VBO) yields "
                    "pf_lo × (1 - 0.5·|max_corr|) <= 1.0 — gate output stashed "
                    "on metrics_snapshot.portfolio_corr."
                ),
                idempotent=False,
            ),
            PlaybookCapability(
                name="submit_review_request",
                method="POST",
                path="/reviews/request",
                purpose=(
                    "Researcher submits a review request. Body: target_kind "
                    "('plan'|'graduation'), and either 'plan' "
                    "{strategy_code, axis_names, hypothesis_id, ...} or "
                    "'graduation' {iteration_id, strategy_code, ...}. "
                    "Returns target_id (stable identifier) for polling. "
                    "Honours Idempotency-Key. Creates an IDEA_BACKLOG "
                    "journal row with kind='review_request'."
                ),
                idempotent=False,
            ),
            PlaybookCapability(
                name="submit_review_verdict",
                method="POST",
                path="/reviews",
                purpose=(
                    "Reviewer posts a structured verdict on a target. Body: "
                    "{target_id, target_kind, verdict (APPROVED|"
                    "CONDITIONAL_APPROVAL|REJECTED), findings: [{check_name, "
                    "severity, passed, finding, details}], summary_reason, "
                    "summary_n_blocker_fails, summary_n_warning_fails, "
                    "motivating_request_id, strategy_code}. Closes any open "
                    "request rows for the target. Creates a STRATEGY_OUTCOME "
                    "journal row with kind='review_verdict'."
                ),
                idempotent=False,
            ),
            PlaybookCapability(
                name="list_pending_reviews",
                method="GET",
                path="/reviews/pending",
                purpose=(
                    "Reviewer's work queue. Returns open IDEA_BACKLOG rows "
                    "with kind='review_request', oldest first."
                ),
                idempotent=True,
            ),
            PlaybookCapability(
                name="get_review_by_target",
                method="GET",
                path="/reviews/by-target",
                purpose=(
                    "Researcher polls this to read the latest verdict for a "
                    "target_id. Pass history=true to get the full request "
                    "+ verdict audit trail. Returns next_action=retry when "
                    "no verdict has been posted yet."
                ),
                idempotent=True,
            ),
            PlaybookCapability(
                name="run_null_screen",
                method="POST",
                path="/null-screen",
                purpose=(
                    "Pre-sweep archetype edge sniff. Runs K random param "
                    "draws and inspects the PF distribution shape. Returns "
                    "EDGE_PRESENT / NO_EDGE_DETECTED / INCONCLUSIVE / "
                    "INSUFFICIENT_DATA — call BEFORE enqueuing a sweep so "
                    "the agent doesn't burn V11 ticks on a dead archetype. "
                    "Does NOT write hypothesis_audit (preserves DSR "
                    "multiplicity budget for the confirmatory sweep that "
                    "follows). Result journaled as NULL_SCREEN_RESULT. "
                    "Synchronous; ~30-60min for K=8."
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
            PlaybookCapability(
                name="log_activity",
                method="POST",
                path="/activity",
                purpose=(
                    "Log a named agent activity. Body: {session_id (optional UUID), "
                    "activity_type, title, strategy_code (optional), details (optional dict), "
                    "related_id (optional UUID), related_type (optional str), "
                    "status (default SUCCESS)}. "
                    "Use for: SESSION_START, SESSION_END, HYPOTHESIS_REGISTERED, "
                    "PLAN_WRITTEN, GOAL_HIT, and any other named milestones. "
                    "TICK_DISPATCHED, ITERATION_COMPLETED, SWEEP_QUEUED, "
                    "WALK_FORWARD_SUBMITTED, WALK_FORWARD_RESULT, REVIEW_REQUESTED, "
                    "REVIEW_RECEIVED are auto-logged by the orchestrator — do not "
                    "double-log them. X-Session-Id is optional: when omitted, the "
                    "orchestrator auto-groups by (agent, UTC date) so the activity "
                    "log still produces sensible per-day sessions."
                ),
                idempotent=False,
            ),
            PlaybookCapability(
                name="list_activities",
                method="GET",
                path="/activity",
                purpose=(
                    "List activity rows. Filters: session_id, agent_name, "
                    "activity_type, strategy_code. Offset-paginated (limit, offset). "
                    "Within a session, ordered ASC by created_at; without session "
                    "filter, ordered DESC."
                ),
                idempotent=True,
            ),
            PlaybookCapability(
                name="list_sessions",
                method="GET",
                path="/activity/sessions",
                purpose=(
                    "Session-level summary: per session_id, returns started_at, "
                    "last_activity_at, activity_count, strategy_codes[], "
                    "iterations_completed, significant_edge_count, no_edge_count, "
                    "discard_count, goal_hit. Ordered by most-recent activity DESC."
                ),
                idempotent=True,
            ),
            PlaybookCapability(
                name="list_features",
                method="GET",
                path="/features",
                purpose=(
                    "List rows in feature_registry. Filters: family "
                    "(macro/positioning/flow/market_structure/sentiment/label/...), "
                    "status, label_direction (forward/backward/null). "
                    "Offset-paginated (limit, offset). Read this on cold-boot "
                    "before designing a feature-driven strategy to see what's "
                    "already computed vs what would need a backfill."
                ),
                idempotent=True,
            ),
            PlaybookCapability(
                name="get_feature",
                method="GET",
                path="/features/{name}/v/{version}",
                purpose=(
                    "Registry spec + coverage stats for one (name, version). "
                    "Coverage includes row_count, earliest_ts, latest_ts, "
                    "distinct_symbols, distinct_intervals — read this to decide "
                    "if a backfill is needed before referencing the feature in a "
                    "sweep_config."
                ),
                idempotent=True,
            ),
            PlaybookCapability(
                name="feature_sample",
                method="GET",
                path="/features/{name}/v/{version}/sample",
                purpose=(
                    "Most-recent N rows from feature_values for one (name, "
                    "version), DESC by ts (default n=100, max 1000). Use "
                    "symbol/interval to scope per-bar features. Quick sanity "
                    "check on what the transformer actually emits."
                ),
                idempotent=True,
            ),
            PlaybookCapability(
                name="list_feature_runs",
                method="GET",
                path="/features/runs",
                purpose=(
                    "List recent feature_compute_run rows ordered by "
                    "started_at DESC. Filters: feature_name, version, status "
                    "(pending/running/done/failed/cancelled). Use this to see "
                    "if a backfill is still in flight or to find the run_id "
                    "of the latest compute for an audit."
                ),
                idempotent=True,
            ),
            PlaybookCapability(
                name="get_feature_run",
                method="GET",
                path="/features/runs/{run_id}",
                purpose=(
                    "Single feature_compute_run audit row. Returns "
                    "range_start, range_end, symbol, interval, status, "
                    "rows_written, started_at, finished_at, error_message. "
                    "Use this to inspect a failed backfill or confirm a "
                    "compute landed the expected row count."
                ),
                idempotent=True,
            ),
            PlaybookCapability(
                name="backfill_feature",
                method="POST",
                path="/features/{name}/v/{version}/backfill",
                purpose=(
                    "Trigger one feature compute on the blackheart-ingest "
                    "worker. Synchronous proxy — the orchestrator waits for "
                    "the worker to finish (default 600s timeout). Body: "
                    "{start, end, symbol, interval} — symbol+interval are "
                    "REQUIRED for per-bar features (any feature whose "
                    "raw_tables=['market_data']). Returns "
                    "{run_id, rows_written, status, duration_seconds, ...} "
                    "from the worker; immediately follow with "
                    "GET /features/runs/{run_id} for the persisted audit row. "
                    "Idempotency-Key (~24h TTL): success bodies AND "
                    "deterministic 4xx errors (unknown feature, missing "
                    "symbol/interval) are cached; transient failures "
                    "(503 unreachable, 502 compute crash) are NOT cached "
                    "so retries get a fresh attempt. "
                    "NOTE: POST /features/register is NOT exposed — new "
                    "features require a code change in "
                    "blackheart_ingest.features.definitions + a Flyway "
                    "migration; deferred to M5."
                ),
                idempotent=False,
            ),
            PlaybookCapability(
                name="list_raw_sources",
                method="GET",
                path="/raw/sources",
                purpose=(
                    "List ml_ingest_schedule rows LEFT JOINed with "
                    "ml_source_health. One row per (source, symbol) tuple "
                    "from the schedule; health rows are keyed by source so "
                    "every symbol-variant of a source shares the same "
                    "health_status. Per row: source, symbol, "
                    "cron_expression, lookback_hours, enabled, config, "
                    "last_run_at, last_success_at, last_error_message, "
                    "next_run_at, health_status (healthy/degraded/failed/"
                    "unknown), last_pull_at, consecutive_failures, "
                    "rows_inserted_total, rejected_pit_violations_total, "
                    "errors_total, health_message. Use this to confirm an "
                    "upstream source is wired + healthy before designing "
                    "a feature that depends on it."
                ),
                idempotent=True,
            ),
            PlaybookCapability(
                name="raw_table_coverage",
                method="GET",
                path="/raw/{table}/coverage",
                purpose=(
                    "Coverage stats for one whitelisted raw table: "
                    "macro_raw, onchain_raw, news_raw, social_raw. "
                    "(market_data is NOT in the whitelist — for per-bar "
                    "features whose raw_tables=['market_data'], rely on "
                    "GET /features/{name}/v/{version} coverage instead.) "
                    "Returns one row per (source, series_id): row_count, "
                    "first_event_time, last_event_time, "
                    "last_ingestion_time. Optional filters: source, "
                    "series_id, from, to (event_time window, inclusive). "
                    "Use this to verify a macro/on-chain/news/social "
                    "source has the historical depth a feature backfill "
                    "needs before issuing the compute."
                ),
                idempotent=True,
            ),
            PlaybookCapability(
                name="register_model",
                method="POST",
                path="/models/register",
                purpose=(
                    "Register a trained sub-model artifact in "
                    "model_registry (V66). Caller (typically the "
                    "blackheart-train CLI's --register flag) POSTs the "
                    "training payload's load-bearing fields — the "
                    "orchestrator does NOT read the pickle off disk. "
                    "Body: content_sha256 (64-char), artifact_uri, "
                    "artifact_size_bytes, spec{name,purpose in "
                    "(regime/positioning/flow/directional/meta_label/"
                    "stacker), symbol, interval, label_feature, "
                    "label_version, objective in (binary/regression), "
                    "train_start, train_end, hyperparams, "
                    "derived_features}, feature_names, metrics, "
                    "walk_forward, gauntlet{overall_verdict}, "
                    "deployment_readiness{deployment_ready, "
                    "unregistered_input_features, unregistered_label}, "
                    "horizon_bars (optional override; falls back to a "
                    "known-label lookup, NULL for unknown), "
                    "data_fingerprint. Status is DERIVED: gauntlet PASS + "
                    "deployment_ready=true → 'trained'; PASS + "
                    "deployment_ready=false → 'awaiting_operator_review'; "
                    "FAIL → 'rejected_by_operator'; no gauntlet block → "
                    "'training'. Idempotency-Key (~24h TTL) caches the "
                    "response; AND content_sha256 is the content-"
                    "addressed idempotency anchor (same sha always "
                    "returns the same model_id). Response: model_id, "
                    "content_sha256, status, version, registered_at, "
                    "idempotent_replay. 409 on (purpose, symbol, "
                    "interval, horizon_bars) UNIQUE race."
                ),
                idempotent=True,
            ),
            PlaybookCapability(
                name="list_models",
                method="GET",
                path="/models",
                purpose=(
                    "List model_registry rows with optional filters: "
                    "status (training/trained/staged/shadow/cooling_down/"
                    "live/retired/awaiting_operator_review/"
                    "rejected_by_operator), purpose, symbol, "
                    "synced (boolean — 'synced=false' answers 'what's "
                    "registered but not yet rsynced to VPS?', the "
                    "operator's pre-promotion checklist). Ordered by "
                    "created_time DESC. Offset-paginated (limit 1-500, "
                    "offset). Returns {models[], total, limit, offset}."
                ),
                idempotent=True,
            ),
            PlaybookCapability(
                name="get_model",
                method="GET",
                path="/models/{model_id}",
                purpose=(
                    "Read one model_registry row by UUID. Returns the "
                    "full row including feature_set JSONB, hyperparams, "
                    "metrics (saved_booster, walk_forward, gauntlet_"
                    "verdict, data_fingerprint, deployment_readiness), "
                    "artifact_sha256, artifact_synced_to_vps, status, "
                    "version, promoted_at, promoted_by, created_time, "
                    "updated_time. 404 if id unknown."
                ),
                idempotent=True,
            ),
            PlaybookCapability(
                name="mark_model_synced",
                method="POST",
                path="/models/{model_id}/mark-synced",
                purpose=(
                    "Flip artifact_synced_to_vps=true on a model_registry "
                    "row and set artifact_synced_at. Operator runs the "
                    "rsync transport manually, then POSTs here to record "
                    "completion (the orchestrator does NOT itself rsync). "
                    "Body (all optional, {} valid): remote_path (new "
                    "artifact_uri; if omitted, the existing URI is "
                    "preserved), synced_at (defaults to NOW; rejected if "
                    "more than 5 minutes in the future, MF6 guard). "
                    "Re-marking refreshes the timestamp (latest wins). "
                    "Honors Idempotency-Key. Logs a WARNING when applied "
                    "to a rejected_by_operator row (operationally "
                    "suspect — rejected artifacts shouldn't be promoted "
                    "to VPS in the first place)."
                ),
                idempotent=True,
            ),
            PlaybookCapability(
                name="promote_model",
                method="POST",
                path="/models/{model_id}/promote",
                purpose=(
                    "Advance model_registry.status along the V66 "
                    "lifecycle. State graph (forward-only — no rollback "
                    "edges; rolling back live models is via fresh "
                    "registration + kill switch, NOT status reversion):\n"
                    "  awaiting_operator_review → trained | rejected_by_operator\n"
                    "  trained                  → staged | rejected_by_operator\n"
                    "  staged                   → shadow | retired\n"
                    "  shadow                   → cooling_down | retired\n"
                    "  cooling_down             → live | retired\n"
                    "  live                     → retired\n"
                    "  rejected_by_operator     → retired (no rehab)\n"
                    "Body: target_status (Literal — 'training' and "
                    "'awaiting_operator_review' NOT allowed as targets), "
                    "reason (optional ≤500 char, logged not persisted), "
                    "reviewer_verdict (optional ≤30 char, COALESCE'd "
                    "onto row), reviewer_run_id (optional UUID, "
                    "COALESCE'd). Idempotent in two ways: (1) "
                    "Idempotency-Key header replays; (2) same target as "
                    "current status returns transition_applied=false "
                    "with current promoted_at/promoted_by. SELECT + "
                    "validate + UPDATE wrapped in a transaction with "
                    "optimistic locking (WHERE status=expected) — "
                    "concurrent transition surfaces as 409 "
                    "'concurrent transition: row status changed after "
                    "read'. 404 if id unknown. 409 if transition not "
                    "allowed (response details list valid targets from "
                    "the current status). Cooldown enforcement (the "
                    "7-day gate before cooling_down→live per blueprint "
                    "§ 12) is NOT yet implemented — operator workflow "
                    "enforces timing for now."
                ),
                idempotent=True,
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
                    "GET /journal?entry_type=NULL_SCREEN_RESULT&strategy_code=<code> — skip if recent NO_EDGE_DETECTED",
                    "POST /queue with sweep_config + hypothesis + Idempotency-Key",
                    "POST /tick — first iteration runs immediately",
                ],
            ),
            PlaybookRecipe(
                name="bayesian-sweep",
                when="Param ranges are continuous and a coarse grid would waste budget, or you want sample-efficient search.",
                steps=[
                    "POST /queue with sweep_config={'strategy':'tpe', "
                    "'params':[{'name':...,'type':'float'|'int'|'choice', "
                    "'low':...,'high':...,'values':...}], 'n_trials':20, "
                    "'seed':42} — n_trials sets the budget, not iter_budget "
                    "(which still bounds /tick retries on infra failures).",
                    "POST /tick repeatedly; each call asks Optuna for the "
                    "next point given the queue's history of (params, pf).",
                    "When iter_index >= n_trials the orchestrator parks the "
                    "queue with outcome='sweep_exhausted' (same as grid).",
                    "TPE history reads from hypothesis_audit JOIN "
                    "iteration_log; failed-backtest rows are skipped, "
                    "infinity-PF cells are capped at 10.0.",
                ],
            ),
            PlaybookRecipe(
                name="paired-research-loop",
                when=(
                    "User invokes quant-researcher; goal is a 4th profitable "
                    "strategy (>=10%/yr ROBUST). Researcher drives the loop, "
                    "reviewer audits at each gate."
                ),
                steps=[
                    "RESEARCHER: read state (journal, leaderboard, queue).",
                    "RESEARCHER: write HYPOTHESIS journal entry (status=ACTIVE) BEFORE designing the plan.",
                    "RESEARCHER: write RESEARCH_PLAN_<date>.md.",
                    "RESEARCHER: POST /reviews/request with target_kind='plan' "
                    "{strategy_code, axis_names, hypothesis_id}.",
                    "RESEARCHER: spawn quant-reviewer subagent (Agent tool) "
                    "with target_id from the response.",
                    "REVIEWER: GET /reviews/pending → fetch the request.",
                    "REVIEWER: GET /journal/{hypothesis_id} → read pre-registered "
                    "HYPOTHESIS.",
                    "REVIEWER: run plan_review_checklist (services/review.py) "
                    "against artifacts.",
                    "REVIEWER: POST /reviews with verdict + findings.",
                    "RESEARCHER: GET /reviews/by-target?target_id=... → read verdict.",
                    "If APPROVED: POST /queue (gated on APPROVED), POST /tick "
                    "loop until SIGNIFICANT_EDGE or sweep exhausted.",
                    "If REJECTED (round 1): address findings, re-submit; "
                    "(round 2): pivot to next archetype/axis.",
                    "On SIGNIFICANT_EDGE iteration: POST /reviews/request with "
                    "target_kind='graduation' {iteration_id}.",
                    "REVIEWER: graduation_review_checklist; POST /reviews.",
                    "If APPROVED: POST /walk-forward (gated). If "
                    "stability=ROBUST AND return >= 10%/yr: WIN, exit loop.",
                    "Else: journal lesson, pick next archetype, restart from "
                    "step 1.",
                ],
            ),
            PlaybookRecipe(
                name="reviewer-cold-boot",
                when="Reviewer agent is invoked (directly or via Agent subcall).",
                steps=[
                    "GET /agent/playbook — confirm contract.",
                    "GET /reviews/pending — fetch work queue (oldest first).",
                    "For the request: extract target_id, target_kind, "
                    "request_payload from structured_data.",
                    "GET /journal/{hypothesis_id} for plan reviews; "
                    "GET /iterations/{iteration_id} for graduation reviews.",
                    "Run the appropriate checklist (plan or graduation); "
                    "produce structured findings.",
                    "POST /reviews with verdict + findings list. Reviewer's "
                    "verdict is authoritative — researcher cannot override "
                    "without operator escape hatch.",
                ],
            ),
            PlaybookRecipe(
                name="archetype-edge-sniff",
                when="Considering a fresh archetype/axis combo and want to avoid burning V11 ticks on a dead landscape.",
                steps=[
                    "POST /null-screen with strategy_code, interval_name, "
                    "instrument, param_ranges (low/high/type per param), "
                    "n_draws (default 8), Idempotency-Key.",
                    "Read response.verdict:",
                    "  'EDGE_PRESENT'      → POST /queue with a focused sweep on this axis set",
                    "  'NO_EDGE_DETECTED'  → MOVE ON; do not sweep this archetype/axis combo",
                    "  'INCONCLUSIVE'      → POST /queue with a small (8-cell) confirmatory grid",
                    "  'INSUFFICIENT_DATA' → investigate JVM health; re-run later",
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
                name="feature-coverage-check",
                when=(
                    "Designing a strategy that references an ML feature "
                    "(macro_raw, market_data zscore, label, etc.) — verify the "
                    "feature exists, is computed for the relevant window, and "
                    "trigger a backfill if not before enqueuing a sweep."
                ),
                steps=[
                    "GET /features?family=<family> — discover what's registered.",
                    "GET /features/{name}/v/{version} — read spec + coverage "
                    "(row_count, first_ts, last_ts). Inspect spec.raw_tables "
                    "to know which upstream the feature depends on.",
                    "If coverage gap covers the backtest window:",
                    "  GET /raw/sources — confirm the upstream source is "
                    "healthy and has the depth required.",
                    "  Upstream-coverage check is split by raw_tables[0]:",
                    "    macro_raw/onchain_raw/news_raw/social_raw -> "
                    "GET /raw/{raw_table}/coverage?source=<src>&from=<lo>&to=<hi>",
                    "    market_data (per-bar features) -> no /raw/* endpoint; "
                    "rely on the GET /features/{name}/v/{version} coverage stats "
                    "already returned, since market_data is owned by the trading "
                    "JVM and partitioned by (symbol, interval, ts).",
                    "  POST /features/{name}/v/{version}/backfill with "
                    "{start, end, symbol?, interval?} + Idempotency-Key — "
                    "synchronous, may take minutes for per-bar features. "
                    "symbol+interval are REQUIRED when spec.symbols / "
                    "spec.intervals are non-empty.",
                    "  GET /features/runs/{run_id} — confirm status='done' "
                    "and rows_written matches expectation. If status='failed', "
                    "read error_message and adjust window/inputs.",
                    "GET /features/{name}/v/{version}/sample?n=20 (scope by "
                    "symbol+interval for per-bar features) — eyeball the "
                    "most-recent values for sanity (no all-NaN tail, no obvious "
                    "distribution shift).",
                    "Now safe to reference the feature in a sweep_config and "
                    "POST /queue.",
                ],
            ),
            PlaybookRecipe(
                name="model-lifecycle",
                when=(
                    "An ML sub-model has been trained (blackheart-train "
                    "CLI ran with --register) and the operator wants to "
                    "advance it through the V66 status lifecycle: "
                    "trained → staged → shadow → cooling_down → live. "
                    "The orchestrator owns writes; the trading JVM reads "
                    "via direct SELECT on model_registry (V66 grant)."
                ),
                steps=[
                    "GET /models?status=awaiting_operator_review — list "
                    "rows blocked on a deployment_readiness gap (gauntlet "
                    "passed but derived features or label aren't in "
                    "feature_registry yet).",
                    "GET /models/{model_id} — inspect the row. Check "
                    "metrics.gauntlet_verdict='PASS', "
                    "metrics.deployment_readiness.deployment_ready, and "
                    "metrics.deployment_readiness.unregistered_input_"
                    "features / unregistered_label.",
                    "If awaiting_operator_review: either resolve the "
                    "registry gap (register the missing features/label "
                    "and re-run --register; idempotent on content_sha256, "
                    "so a new row is only written if the sha changed), "
                    "OR force-clear via "
                    "POST /models/{model_id}/promote {target_status: "
                    "'trained', reason: '...'} once the gap is "
                    "addressed. Force-clear is a soft one-way door — "
                    "you cannot return to awaiting_operator_review.",
                    "On 'trained': operator rsyncs the artifact to VPS, "
                    "then POST /models/{model_id}/mark-synced {} (body "
                    "may be empty; synced_at defaults to NOW). "
                    "ONLY synced models are visible to the trading JVM's "
                    "serving path (the read repo filters on "
                    "artifact_synced_to_vps=true).",
                    "POST /models/{model_id}/promote "
                    "{target_status: 'staged', reason: '...', "
                    "reviewer_verdict?, reviewer_run_id?} — moves to "
                    "staged. Idempotency-Key recommended (~24h TTL "
                    "replay).",
                    "POST .../promote {target_status: 'shadow'} — "
                    "trading JVM's HYBRID strategies will now see the "
                    "model via findLatestForServing() when configured "
                    "with ml_mode_shadow=true (decisions logged, NOT "
                    "enforced).",
                    "After shadow validation (paired-delta significance "
                    "test per blueprint § M6): POST .../promote "
                    "{target_status: 'cooling_down'}. Cooldown is "
                    "currently operator-enforced timing — the future "
                    "model_promotion_gauntlet row will server-side gate "
                    "the cooling_down → live edge by 7 days.",
                    "POST .../promote {target_status: 'live'} — model "
                    "is now the serving model (HYBRID strategies with "
                    "ml_mode_shadow=false will use it for real "
                    "decisions).",
                    "Rollback / abandonment: POST .../promote "
                    "{target_status: 'retired'} from any state, or "
                    "'rejected_by_operator' from trained/"
                    "awaiting_operator_review only. Forward-only — to "
                    "'roll back' a live model, register a corrected "
                    "version (new content_sha256) and promote it; the "
                    "old one moves to retired.",
                    "On 409 'concurrent transition: row status changed "
                    "after read': GET /models/{model_id} to read the "
                    "current status, then retry with a target valid "
                    "from that source (or accept that another caller's "
                    "transition is the correct outcome).",
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
    current_session_id: str
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
        current_session_id=str(request.state.session_id) if request.state.session_id else "",
        notes=notes,
        next_actions=next_actions,
    )
