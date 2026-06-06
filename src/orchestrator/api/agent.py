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

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel

from ..repo import agent_state as agent_state_repo

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
                purpose=(
                    "One-shot research-state digest. Returns: db_ok, jvm_ok, "
                    "queue_counts (PENDING/RUNNING/PARKED/COMPLETED/FAILED), "
                    "last_iterations (5 most-recent with statistical_verdict, "
                    "verdict, pf, n_trades), recent_sig_edge_iteration_ids "
                    "(7d window), active_hypotheses (journal entry_type="
                    "HYPOTHESIS status=ACTIVE), last_run_summary (latest "
                    "RUN_SUMMARY journal row), last_null_screen_per_surface "
                    "(DISTINCT ON (strategy, instrument, interval) of the "
                    "latest NULL_SCREEN_RESULT). Replaces the legacy "
                    "5-query cold-boot chain — call once on session start "
                    "before designing the next iteration."
                    " Pass ?track=trading|hedging to scope lockout/run-summary/"
                    "hypotheses/null-screen/specialist rows to one research "
                    "track (omit for global)."
                ),
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
                    "justification to bypass. "
                    "Pass track ('trading'|'hedging') to scope this to one "
                    "research loop; omit for the legacy global queue."
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
                    "sweep_exhausted. Response carries a 'summary' field "
                    "{verdict_line, next_action, decision_hint} — next_action "
                    "is one of CONTINUE | GRADUATE | PIVOT | EMPTY_QUEUE | "
                    "WAIT | INFRA_FAIL. The quant-runner sub-agent (haiku) "
                    "branches on next_action alone to decide whether to "
                    "keep ticking or escalate. metrics_snapshot.analysis "
                    "includes dsr (Deflated Sharpe per Bailey & Lopez de "
                    "Prado 2014) and dsr_n_trials (the real selection-bias "
                    "multiplicity) alongside the legacy psr field. "
                    "Portfolio gate (added 2026-05-05): SIGNIFICANT_EDGE "
                    "candidates are demoted to INSUFFICIENT_EVIDENCE when "
                    "their daily-return correlation with the protected "
                    "book (LSR/VCB/VBO) yields pf_lo × (1 - 0.5·|max_corr|) "
                    "<= 1.0 — gate output stashed on "
                    "metrics_snapshot.portfolio_corr. "
                    "Pass track ('trading'|'hedging') to scope this to one "
                    "research loop; omit for the legacy global queue. "
                    "Gate selection by track (Phase 2, 2026-06-07): "
                    "track in (null,'trading') graduate on V11 + V60 "
                    "(>=10%/yr) as before; track='hedging' rows graduate on "
                    "the equity-level beats-buy-hold gate instead (risk-adj "
                    "improvement vs holding the underlying — bootstrap "
                    "significance replaces V11, beats_buy_hold_risk_adj "
                    "replaces the V60 economic check), with the full verdict "
                    "stashed on metrics_snapshot.hedging_gate. Walk-forward "
                    "ROBUST is still required for both tracks."
                ),
                idempotent=False,
            ),
            PlaybookCapability(
                name="skeptic_prescreen",
                method="POST",
                path="/skeptic-prescreen",
                purpose=(
                    "Mechanical pattern checks on a graduation candidate "
                    "BEFORE deciding whether to invoke the LLM "
                    "quant-skeptic sub-agent. Body: {iteration_id, "
                    "strategy_code, motivating_hypothesis_id?}. Returns "
                    "five check blocks (archetype_churn_24h, "
                    "repeat_axis_attempts, narrow_param_window, "
                    "post_hoc_hypothesis_edit, dsr_trial_undercount) plus "
                    "{any_flag_fired, n_flags, recommendation}. "
                    "Researcher uses recommendation ∈ "
                    "{invoke_skeptic_agent, skip_skeptic_agent} to decide "
                    "whether to spawn the sonnet skeptic sub-agent. "
                    "Saves Max-plan token budget on routine-clean grads."
                ),
                idempotent=False,
            ),
            PlaybookCapability(
                name="portfolio_correlations",
                method="POST",
                path="/portfolio/correlations",
                purpose=(
                    "Pairwise Spearman correlation matrix over recent "
                    "daily-return series of each strategy_code. Body: "
                    "{strategy_codes (1-20), min_overlap_days (2-60, "
                    "default 5)}. Returns {codes, matrix, "
                    "max_abs_per_code, missing_strategies}. NaN entries "
                    "indicate insufficient overlap — do NOT impute as "
                    "zero. Pure NumPy; no LLM cost. Same data the V11 "
                    "portfolio gate already consumes, exposed at the "
                    "agent surface."
                ),
                idempotent=True,
            ),
            PlaybookCapability(
                name="portfolio_optimize",
                method="POST",
                path="/portfolio/optimize",
                purpose=(
                    "Run three portfolio optimisers (equal_weight, HRP, "
                    "mean_variance) over the candidate + protected book. "
                    "Body: {strategy_codes, min_overlap_days, "
                    "mu_by_code (optional expected daily returns), "
                    "risk_aversion (default 1.0)}. Returns three weight "
                    "vectors side-by-side, each with a summary "
                    "{n_assets, max_weight, min_weight, "
                    "concentration_hhi, n_effective}. The "
                    "quant-portfolio-manager sub-agent reads these and "
                    "picks one; this endpoint does NOT decide which "
                    "optimiser to use. Pure NumPy; no LLM cost."
                ),
                idempotent=True,
            ),
            PlaybookCapability(
                name="log_agent_decision",
                method="POST",
                path="/agent-decisions/log",
                purpose=(
                    "Append-only audit-row writer for the agent_decisions "
                    "table (Flyway V97). Researcher posts here after "
                    "each spawned specialist sub-agent returns its "
                    "verdict (quant-skeptic, quant-portfolio-manager, "
                    "quant-capacity-judge, quant-data-scout). Body: "
                    "{specialist, endpoint, model_name?, "
                    "request_payload, response_payload?, verdict?, "
                    "latency_ms?, status ∈ {ok, parse_error, api_error, "
                    "timeout, refused}, error_envelope?, "
                    "motivating_iteration_id?, target_id?}. "
                    "Idempotency-Key honoured for safe retry. Returns "
                    "{decision_id}. The audit table is the durable "
                    "record of every specialist invocation regardless "
                    "of transport (sub-agent, HTTP, or future API)."
                ),
                idempotent=False,
            ),
            PlaybookCapability(
                name="paired_delta",
                method="POST",
                path="/paired-delta",
                purpose=(
                    "Phase E (2026-05-19) — paired-delta analytics on "
                    "an ML sweep. Partitions iteration_log rows into "
                    "matched (gate-on, gate-off) pairs holding ALL non-"
                    "ML params constant; computes per-pair delta on the "
                    "chosen metric; paired-bootstrap CI; emits a verdict "
                    "in {POSITIVE_DELTA, NEGATIVE_DELTA, NEUTRAL, "
                    "INDETERMINATE}. Body: {queue_id OR iteration_ids, "
                    "primary_metric ('sharpe' default), treatment_key "
                    "(default '_ml_gate_enabled'), treatment_on_value "
                    "(default true), treatment_off_value (default false), "
                    "bootstrap_reps (default 1000), ci_level (0.95), "
                    "rng_seed (42)}. NEGATIVE_DELTA is BINDING — the "
                    "researcher's loop must journal STRATEGY_OUTCOME and "
                    "back out to step 1 (Phase 4 Stage H signature: "
                    "Sharpe -1.23 → -3.13 = model destroyed value). "
                    "trade_count metric is always NEUTRAL (no clear "
                    "good/bad direction)."
                ),
                idempotent=True,
            ),
            PlaybookCapability(
                name="ml_prescreen",
                method="POST",
                path="/ml-prescreen",
                purpose=(
                    "Phase C (2026-05-19) — 5 mechanical overfit/leakage "
                    "checks on a trained model artifact. Body: "
                    "{model_id (UUID, optional), experiment_run_id (UUID, "
                    "optional)} — at least one required. Reads "
                    "model_registry + experiment_run rows. Returns "
                    "{checks: [{check_name, severity (HARD|SOFT|PASS), "
                    "flag, metric, threshold, finding}, ...], "
                    "any_flag_fired, n_hard_flags, n_soft_flags, "
                    "recommendation: 'invoke_ml_judge' | 'skip_ml_judge'}. "
                    "The 5 checks: label_leakage_signature (leaking "
                    "features in experiment_run.leakage_report), "
                    "train_oof_auc_gap (saved_booster vs walk_forward "
                    "mean), walk_forward_fold_cv (primary_std/mean), "
                    "gauntlet_verdict (FAIL/WARN/PASS), "
                    "deployment_ready_flag. Idempotency-Key honored. "
                    "404 ml_artifacts_not_found = upstream training "
                    "didn't register — do NOT proceed to wire the "
                    "model into a sweep."
                ),
                idempotent=True,
            ),
            PlaybookCapability(
                name="start_ml_training_run",
                method="POST",
                path="/ml/training-runs",
                purpose=(
                    "Spawn a blackheart-train subprocess to train one "
                    "ModelSpec (Phase A of researcher-drives-ML, V99 "
                    "ml_training_runs). Body: {spec_name, walk_forward?=true, "
                    "gauntlet?=true, folds?, bayesian?=false, "
                    "do_register?=true, allow_leakage?=false, no_write?=false, "
                    "notes?, motivating_hypothesis_id?}. Returns 202 with "
                    "{training_run_id, status:'PENDING', spec_name, cli_args, "
                    "poll_url}. Subprocess runs out-of-band; poll the GET "
                    "endpoint until status reaches a terminal value "
                    "({COMPLETED, FAILED, TIMED_OUT, ORPHANED}). Per-agent "
                    "daily cap (default 4/24h) enforced; FAILED rows don't "
                    "count toward the cap. Idempotency-Key honoured."
                ),
                idempotent=True,
            ),
            PlaybookCapability(
                name="get_ml_training_run",
                method="GET",
                path="/ml/training-runs/{training_run_id}",
                purpose=(
                    "Poll a single training run for status. Returns the "
                    "full V99 row: status, exit_code, duration_seconds, "
                    "stdout_tail (last 16KB), stderr_tail, content_sha256 "
                    "(set on COMPLETED), model_id (set on COMPLETED when "
                    "do_register was true), experiment_run_id (joinable "
                    "to V92 experiment_run for gauntlet detail), "
                    "error_envelope (set on FAILED/TIMED_OUT/ORPHANED)."
                ),
                idempotent=True,
            ),
            PlaybookCapability(
                name="list_ml_model_specs",
                method="GET",
                path="/ml/model-specs",
                purpose=(
                    "Phase D (2026-05-19) — researcher discovery. Returns "
                    "the operator-curated list of blackheart-train model "
                    "spec names that can be passed as spec_name to POST "
                    "/ml/training-runs. Mirrors the train CLI's "
                    "_spec_choices() but lives in orchestrator settings "
                    "(available_model_specs) so the read is cheap. The "
                    "subprocess validates the name at training time; the "
                    "orchestrator does not — invalid names fail with a "
                    "FAILED row carrying argparse's error in stderr_tail."
                ),
                idempotent=True,
            ),
            PlaybookCapability(
                name="list_ml_training_runs",
                method="GET",
                path="/ml/training-runs",
                purpose=(
                    "Recent training runs for the calling agent, newest "
                    "first. Resume protocol: read this on session start "
                    "to pick up RUNNING/PENDING work from a prior session "
                    "rather than starting a duplicate. ?limit= up to 200."
                ),
                idempotent=True,
            ),
            PlaybookCapability(
                name="auto_run_review_checklist",
                method="POST",
                path="/reviews/auto-run-checklist",
                purpose=(
                    "Run the reviewer checklist server-side and persist "
                    "the verdict — the orchestrator-owned alternative to "
                    "spawning a quant-reviewer sub-agent. Body: "
                    "{target_id, target_kind ('plan'|'graduation')}. The "
                    "request row written by POST /reviews/request must "
                    "exist for the target_id; the auto-reviewer reads its "
                    "payload to know which axis_names / hypothesis_id / "
                    "iteration_id to audit. Runs the same pure-function "
                    "checks as the standalone reviewer agent "
                    "(plan: pre_registration + mechanism_named + "
                    "axis_not_recently_failed; graduation: n_trades + "
                    "pf_ci + dsr + cost_realism + regime_concentration + "
                    "portfolio_fit + param_robustness) and the same "
                    "aggregate_verdict rule, then writes a "
                    "STRATEGY_OUTCOME journal row with reviewer="
                    "'research-orchestrator-auto-reviewer' and closes the "
                    "matching open request row. Idempotency-Key supported. "
                    "Returns the verdict shape POST /reviews returns "
                    "(verdict, findings, next_actions, etc.). The /queue "
                    "and /walk-forward gates branch on the verdict string "
                    "only — they do NOT discriminate auto-run verdicts "
                    "from human-reviewer verdicts, so behavior downstream "
                    "is unchanged."
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
                name="drain_tick_queue",
                method="POST",
                path="/tick/drain",
                purpose=(
                    "Drive POST /tick in a loop until summary.next_action "
                    "reaches a terminal value (GRADUATE | PIVOT | "
                    "EMPTY_QUEUE | INFRA_FAIL) or a cap fires. Replaces "
                    "the quant-runner sub-agent — the researcher posts "
                    "/tick/drain instead of spawning a haiku-tier "
                    "polling agent. Body (all optional): max_iters "
                    "(1-200, default 40), max_wall_clock_s (60-21600, "
                    "default 10800 = 3h), max_consecutive_waits (1-10, "
                    "default 3). Returns a structured digest: "
                    "{terminal_action, iters_completed, verdicts "
                    "{sig, insuf, no_edge, discard}, last_iteration_id, "
                    "last_verdict_line, last_decision_hint, "
                    "handoff_sentence, next_actions}. terminal_action "
                    "values include MAX_ITERS_REACHED and "
                    "MAX_WALL_CLOCK_REACHED — re-call to continue. "
                    "Each internal /tick honours its own queue-claim "
                    "semantics (FOR UPDATE SKIP LOCKED); concurrent "
                    "drains never collide on the same row. "
                    "Idempotency-Key replays the cached digest envelope "
                    "without re-driving the queue. "
                    "Pass track ('trading'|'hedging') to scope this to one "
                    "research loop; omit for the legacy global queue. "
                    "track='hedging' rows graduate on the equity-level "
                    "beats-buy-hold gate (not V11/V60) — see run_tick."
                ),
                idempotent=False,
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
                name="run_inference",
                method="POST",
                path="/inference/run",
                purpose=(
                    "Single-bar inference. Proxies to blackheart-inference "
                    "sidecar (:8000) which resolves signal_id → model → "
                    "artifact, fetches features from feature_values at ts, "
                    "runs Booster.predict(), and UPSERTs one signal_history "
                    "row. Body: {signal_id (UUID), symbol, ts (ISO-8601), "
                    "interval_name (5m|15m|1h|4h), source ('stream'|"
                    "'catchup_scan'|'historical_replay', default 'stream')}. "
                    "Idempotency-Key honoured. Returns {value, confidence, "
                    "model_id, artifact_sha256, spec_name, objective}. "
                    "Refuses retired signals/models, missing artifacts, "
                    "missing feature_registry entries (409 with specific "
                    "error_code). Use /features/{name}/v/{version}/backfill "
                    "to populate gaps before retrying."
                ),
                idempotent=False,
            ),
            PlaybookCapability(
                name="run_inference_backfill",
                method="POST",
                path="/inference/backfill",
                purpose=(
                    "Window inference. Same resolution path as /inference/run "
                    "but iterates over [start, end). Used to populate "
                    "signal_history over the historical range a paired-"
                    "backtest will replay. Body: {signal_id, symbol, "
                    "interval_name, start (ISO-8601), end (exclusive ISO-"
                    "8601), source ('catchup_scan'|'historical_replay', "
                    "default 'historical_replay')}. Capped at "
                    "INFERENCE_MAX_BACKFILL_ROWS (default 50000) to prevent "
                    "runaway windows; split + re-call on 413. Returns "
                    "{rows_in_window, rows_written, "
                    "rows_skipped_missing_features, model_id, "
                    "artifact_sha256}. Skipped rows are gaps in "
                    "feature_values — surface a /features backfill to fill "
                    "them, then re-run inference backfill (UPSERT semantics "
                    "make replay safe)."
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
                    "strategy (>=10%/yr ROBUST). Researcher drives the loop "
                    "via HTTP — the reviewer + runner are orchestrator "
                    "endpoints, not sub-agents."
                ),
                steps=[
                    "GET /agent/state — read queue depth + last verdict + "
                    "active hypotheses.",
                    "Write a HYPOTHESIS journal entry (status=ACTIVE) BEFORE "
                    "designing the plan. Use POST /journal entry_type=HYPOTHESIS.",
                    "Write RESEARCH_PLAN_<date>.md to disk.",
                    "POST /reviews/request {target_kind:'plan', plan:"
                    "{strategy_code, axis_names, hypothesis_id, plan_path}}.",
                    "POST /reviews/auto-run-checklist {target_id, "
                    "target_kind:'plan'} — orchestrator runs the checks, "
                    "writes the verdict, closes the request row in one call.",
                    "If verdict APPROVED or CONDITIONAL_APPROVAL: POST /queue "
                    "(gated; rejects 409 review_required otherwise).",
                    "If REJECTED (round 1): address findings, re-request, "
                    "re-run /reviews/auto-run-checklist; (round 2 reject) "
                    "pivot to next archetype/axis.",
                    "POST /tick/drain — orchestrator loops /tick until "
                    "terminal_action ∈ {GRADUATE, PIVOT, EMPTY_QUEUE, "
                    "INFRA_FAIL, MAX_ITERS_REACHED, MAX_WALL_CLOCK_REACHED}. "
                    "Returns the runner digest as JSON.",
                    "On GRADUATE: POST /reviews/request {target_kind:"
                    "'graduation', graduation:{iteration_id from digest}}, "
                    "then POST /reviews/auto-run-checklist {target_id, "
                    "target_kind:'graduation'}.",
                    "If graduation APPROVED: POST /walk-forward (gated). If "
                    "stability=ROBUST AND return >= 10%/yr: WIN, exit loop.",
                    "On PIVOT / non-ROBUST: journal STRATEGY_OUTCOME, pick "
                    "next archetype, restart from the HYPOTHESIS step.",
                    "On MAX_ITERS_REACHED / MAX_WALL_CLOCK_REACHED: re-call "
                    "/tick/drain to continue the same queue.",
                ],
            ),
            PlaybookRecipe(
                name="reviewer-cold-boot",
                when=(
                    "Operator invokes a quant-reviewer agent manually (the "
                    "qualitative-judgement path). The autonomous loop uses "
                    "POST /reviews/auto-run-checklist instead and does NOT "
                    "spawn a reviewer sub-agent."
                ),
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
                name="ml-inference-backfill",
                when=(
                    "Researcher wants to populate signal_history for an "
                    "ML model over a historical window — required for any "
                    "downstream paired-backtest evaluation. Requires a "
                    "registered model with status NOT IN (retired, "
                    "rejected_by_operator), a signal_definition referencing "
                    "it, and the features in feature_set already populated "
                    "in feature_values across the target window."
                ),
                steps=[
                    "GET /models?status=trained — find the registered model.",
                    "GET /models/{model_id} — confirm artifact_sha256 + "
                    "feature_set + objective.",
                    "For each feature in feature_set, POST "
                    "/features/{name}/v/{version}/backfill if coverage "
                    "is short of the target window. GET "
                    "/features/{name}/v/{version} first to see current "
                    "row_count + earliest_ts + latest_ts.",
                    "POST /inference/backfill {signal_id, symbol, "
                    "interval_name, start, end, source: 'historical_replay'} "
                    "— populate signal_history over the window. Estimated "
                    "row count is pre-checked from interval + window; 413 "
                    "backfill_window_too_large means split into chunks. "
                    "rows_skipped_missing_features in the response tells "
                    "you which timestamps had at least one feature absent "
                    "— surface a /features backfill for the gap and re-run "
                    "(UPSERT semantics make replay safe).",
                    "Verify with GET /raw/signal_history or psql: rows "
                    "should be present across the window with source="
                    "'historical_replay' and your X-Agent-Name in "
                    "created_by.",
                    "PAIRED-BACKTEST IS AUTOMATED (V100). Add ML sentinel "
                    "params to your sweep_config to test with-ML vs "
                    "without-ML in a single drain. Three sentinels: "
                    "_ml_gate_enabled (bool), _ml_signal_name (str), "
                    "_ml_shadow_mode (bool, true=log-only, false=live). "
                    "Example paired sweep: params=[{name: "
                    "'_ml_gate_enabled', values: [false, true]}, "
                    "{name: '_ml_signal_name', values: ['regime_eth_v2']}, "
                    "{name: '_ml_shadow_mode', values: [false]}, "
                    "{name: 'stopLossAtr', values: [1.5, 2.0, 2.5]}] — "
                    "the cross-product produces 6 cells: 3 without gate "
                    "and 3 with gate+signal+live-mode. Signal history "
                    "must be populated first (see steps above). "
                    "The sweep service routes these sentinels to "
                    "strategyMl*Overrides in the BacktestRunRequest; "
                    "account_strategy rows are never modified. "
                    "Do NOT touch production account_strategy rows.",
                    "Per the Phase 4 Stage H lesson (regime_btc_v3 "
                    "destroyed value on DCB-BTCUSDT-1h, Sharpe -1.23 -> "
                    "-3.13), a NEGATIVE paired-delta is a real signal — "
                    "journal STRATEGY_OUTCOME and abandon the gate, do "
                    "not loosen thresholds to fit it.",
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


class IterationDigest(BaseModel):
    iteration_id: str
    strategy_code: str | None = None
    iteration_number: int | None = None
    statistical_verdict: str | None = None
    verdict: str | None = None
    created_time: datetime
    pf: float | None = None
    n_trades: int | None = None


class HypothesisDigest(BaseModel):
    journal_id: str
    strategy_code: str | None = None
    title: str
    created_time: datetime


class RunSummaryDigest(BaseModel):
    journal_id: str
    strategy_code: str | None = None
    title: str
    content: str | None = None
    structured_data: dict[str, Any] | None = None
    created_time: datetime


class NullScreenDigest(BaseModel):
    journal_id: str
    strategy_code: str | None = None
    instrument: str | None = None
    interval_name: str | None = None
    verdict: str | None = None
    title: str
    created_time: datetime


class PendingSpecialistReviewDigest(BaseModel):
    """One open ``specialist_review_request`` row.

    Researcher's step-1a resume protocol reads these to know if the
    operator's ``/run-pending-specialists`` slash command still owes a
    verdict on a candidate that hit the SPECIALIST_REVIEW_PENDING
    terminal in a prior session.
    """

    journal_id: str
    strategy_code: str | None = None
    specialist_name: str | None = None
    iteration_id: str | None = None
    target_id: str | None = None
    created_time: datetime


class RecentSpecialistVerdictDigest(BaseModel):
    """One recent ``specialist_review_verdict`` row (≤72h window).

    ``is_veto=True`` means the specialist returned OVERRIDE_REJECT
    (skeptic / ml-judge) or REJECT (portfolio). The researcher's resume
    protocol journals STRATEGY_OUTCOME and pivots back to step 1 on any
    veto for the active candidate iteration.
    """

    journal_id: str
    strategy_code: str | None = None
    specialist_name: str | None = None
    iteration_id: str | None = None
    target_id: str | None = None
    verdict: str | None = None
    is_veto: bool
    created_time: datetime


class AgentState(BaseModel):
    """One-shot research-state digest (Phase 2).

    Composes queue counts + last 5 iterations + 7d SIGNIFICANT_EDGE ids +
    ACTIVE hypotheses + latest RUN_SUMMARY + latest NULL_SCREEN_RESULT
    per (strategy_code, instrument, interval_name). Designed to replace
    the legacy cold-boot SQL-script chain — one HTTP call, one round-trip
    bundle for the researcher and runner sub-agents.

    ``digest`` is the populated payload when DB is reachable; ``None``
    when ``db_ok=false``, in which case ``notes`` and ``next_actions``
    carry the failure signal (back-compat with the Phase-1 contract).
    """

    agent: str
    profile: str
    db_ok: bool
    jvm_ok: bool
    current_session_id: str
    notes: list[str]
    next_actions: list[dict[str, Any]]

    queue_counts: dict[str, int] | None = None
    last_iterations: list[IterationDigest] | None = None
    recent_sig_edge_iteration_ids: list[str] | None = None
    active_hypotheses: list[HypothesisDigest] | None = None
    last_run_summary: RunSummaryDigest | None = None
    last_null_screen_per_surface: list[NullScreenDigest] | None = None
    pending_specialist_reviews: list[PendingSpecialistReviewDigest] | None = None
    recent_specialist_verdicts: list[RecentSpecialistVerdictDigest] | None = None
    # 2026-06-02: per-agent ML-training budget + in-flight runs so the
    # researcher knows it can train before designing an ML hypothesis
    # (instead of discovering the 4/day cap via a mid-session 429).
    ml_training_budget: dict[str, Any] | None = None
    pending_ml_training_runs: list[dict[str, Any]] | None = None
    # 2026-06-02: server-computed resume lockout (rank 10) — collapses the
    # multi-call lockout scan and applies the created_time>terminal temporal
    # filter so a pre-existing hypothesis can't falsely clear a fresh lockout.
    lockout_state: dict[str, Any] | None = None


@router.get("/state", response_model=AgentState)
async def agent_state(request: Request, track: str | None = None) -> AgentState:
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

    digest: dict[str, Any] | None = None
    if db_ok:
        try:
            _settings = request.app.state.settings
            async with request.app.state.db.acquire() as conn:
                digest = await agent_state_repo.get_state_digest(
                    conn,
                    agent_name=request.state.agent_name,
                    track=track,
                    ml_training_cap=_settings.ml_training_daily_cap,
                    ml_training_window_hours=_settings.ml_training_daily_window_hours,
                )
        except Exception as exc:  # noqa: BLE001
            # Digest is best-effort. If a slice query fails we still return
            # health flags so the caller can fall back to the legacy queries.
            notes.append(f"State digest unavailable: {exc.__class__.__name__}.")
            digest = None

    if not notes:
        notes.append("Service is healthy.")

    return AgentState(
        agent=request.state.agent_name,
        profile=request.app.state.settings.profile,
        db_ok=db_ok,
        jvm_ok=jvm_ok,
        current_session_id=str(request.state.session_id) if request.state.session_id else "",
        notes=notes,
        next_actions=next_actions,
        queue_counts=(digest or {}).get("queue_counts") if digest else None,
        last_iterations=(digest or {}).get("last_iterations") if digest else None,
        recent_sig_edge_iteration_ids=(
            (digest or {}).get("recent_sig_edge_iteration_ids") if digest else None
        ),
        active_hypotheses=(digest or {}).get("active_hypotheses") if digest else None,
        last_run_summary=(digest or {}).get("last_run_summary") if digest else None,
        last_null_screen_per_surface=(
            (digest or {}).get("last_null_screen_per_surface") if digest else None
        ),
        pending_specialist_reviews=(
            (digest or {}).get("pending_specialist_reviews") if digest else None
        ),
        recent_specialist_verdicts=(
            (digest or {}).get("recent_specialist_verdicts") if digest else None
        ),
        ml_training_budget=(digest or {}).get("ml_training_budget") if digest else None,
        pending_ml_training_runs=(
            (digest or {}).get("pending_ml_training_runs") if digest else None
        ),
        lockout_state=(digest or {}).get("lockout_state") if digest else None,
    )
