"""Pydantic-validated configuration. No magic strings, no os.environ reads
elsewhere — every value the service depends on is declared here, typed, and
documented. The service refuses to start if any field is invalid.

Loaded once at startup and passed via FastAPI dependency injection. Tests
override by constructing Settings(**overrides) directly — never by mutating
env vars.
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEV_TOKEN_SENTINEL = "dev-sentinel-not-for-prod"


class Settings(BaseSettings):
    """Service-wide configuration. Read once at startup; never mutated."""

    model_config = SettingsConfigDict(
        env_prefix="ORCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",  # Misspelled env vars must fail loud, not silently.
    )

    profile: Literal["dev", "prod"] = "dev"

    host: str = "127.0.0.1"  # Loopback default — service is internal-only.
    port: int = Field(8082, ge=1024, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    auth_token: SecretStr = Field(
        ...,
        description=(
            "Shared secret callers must send in X-Orch-Token. "
            "Generate with `python -c 'import secrets; print(secrets.token_urlsafe(48))'`."
        ),
    )

    db_dsn: SecretStr = Field(
        ...,
        description="postgresql://blackheart_research:***@host:5432/blackheart",
    )
    db_pool_min: int = Field(2, ge=1, le=50)
    db_pool_max: int = Field(10, ge=1, le=100)

    jvm_base_url: str = "http://127.0.0.1:8081"
    jvm_request_timeout_s: float = Field(30.0, ge=1.0, le=300.0)

    # Backtest poll cap (tick.py ``_poll_backtest_status``). The orchestrator
    # submits a backtest to the research JVM then polls ``backtest_run.status``
    # until terminal; this is the max wall-clock it waits before treating the
    # run as failed. The legacy 1800s (30-min) default assumed a co-located DB.
    # When the orchestrator runs on the home box against the remote VPS prod DB,
    # a full-window backtest reads market_data per-5m-bar over Tailscale and
    # takes ~33 min, overrunning 30 min and falsely marking the queue FAILED
    # while the JVM run actually COMPLETES. 3600s (60 min) clears that headroom;
    # raise ORCH_POLL_TIMEOUT_S further for a slower link. The stuck-row reaper
    # threshold scales off this (poll cap + 5-min buffer).
    poll_timeout_s: int = Field(3600, ge=300, le=7200)

    # Trading-JVM base URL (Phase A.6, 2026-05-22). The orchestrator
    # calls /api/v1/internal/orch/** here for live equity + Kelly reads
    # that power the portfolio-rebalance min-notional guard. Auth is
    # the existing shared-secret X-Orch-Token, reused from the proxy
    # path (same value as ORCH_AUTH_TOKEN). Loopback default mirrors
    # jvm_base_url; an empty string disables the client (rebalance
    # falls back to caller-provided available_usdt + kelly=1.0).
    trading_jvm_base_url: str = "http://127.0.0.1:8080"
    trading_jvm_request_timeout_s: float = Field(20.0, ge=1.0, le=300.0)
    jvm_auth_mode: Literal["dev_bypass", "service_account", "static_jwt"] = "dev_bypass"
    jvm_service_user: str | None = None
    jvm_service_password: SecretStr | None = None
    # Optional: in dev_bypass mode, pin the login-as email so the JWT carries
    # the operator's userId. Without this, `/dev/login-as` returns the first
    # user, which may not own the account_strategy rows the tick targets
    # (404 on /api/v1/backtest because runBacktest scopes by JWT userId).
    jvm_dev_login_email: str | None = None
    # static_jwt mode: provide a pre-minted JWT directly. Used when the
    # research JVM runs without dev-tooling (so /api/v1/dev/login-as is
    # absent) and no service_account credentials are available. The token
    # must be valid for the JVM's HMAC secret and carry a real userId.
    # Refresh by re-setting this env var and restarting the orchestrator.
    jvm_static_jwt: SecretStr | None = None

    # Research agent identity (V54).
    # The account_id of the dedicated research-agent account seeded by
    # Flyway V54. When set, _resolve_account_strategy filters by this id
    # so concurrent admin-owned rows for the same strategy_code don't get
    # picked. Required in prod; in dev, leaving this unset falls back to
    # the legacy "first matching row" behaviour for local convenience.
    research_account_id: UUID | None = None

    # House Book account (Phase 1-A — signal pool live execution).
    # The account_id whose account_strategy rows are the live materialisation
    # of the signal pool. POST /pool/sync writes signal_pool.pool_weight onto
    # this account's matching rows (account_id + strategy_code + symbol +
    # interval_name). Unset → /pool/sync returns not_configured (no-op). The
    # orchestrator NEVER touches admin's live-trading account — the book is a
    # dedicated account and real-capital enablement stays operator-gated.
    house_book_account_id: UUID | None = None

    # House Book risk guard (Phase 1-B). Max aggregate book weight on any one
    # symbol — caps concentration when several pool members trade the same
    # symbol. Applied during /pool/rebalance. Infeasible compositions
    # (n_symbols × cap < 1) skip the cap and flag rather than under-deploy.
    house_book_max_per_symbol: float = Field(0.50, gt=0.0, le=1.0)

    telegram_bot_token: SecretStr | None = None
    telegram_chat_id: str | None = None

    # When set, ERROR-level log records and unhandled exception handlers
    # ship a row to the trading JVM's /api/v1/errors as
    # source=research-orch. Empty / None = disabled (no-op reporter).
    error_ingest_url: str | None = None
    error_ingest_timeout_s: float = Field(5.0, ge=0.5, le=30.0)

    redis_url: str | None = Field(
        None,
        description=(
            "redis://localhost:6379/0 — when set, activity events are published "
            "to 'research:activity' channel for real-time frontend streaming. "
            "Leave unset to disable."
        ),
    )

    # blackheart-ingest (compute backend).
    # Loopback by default — the orchestrator and ingest both live on the
    # home box. POST /features/{name}/v/{version}/backfill proxies to
    # ``{ingest_base_url}/compute/{name}/v/{version}``.
    # :8001 matches the deployed ingest server (Dockerfile EXPOSE 8001 /
    # healthcheck); the old :8089 default pointed at nothing.
    ingest_base_url: str = "http://127.0.0.1:8001"
    # 10-min ceiling covers a full-history per-bar feature compute over 5+
    # years on 1h bars (~38k rows). Most calls return in well under a
    # minute; the high default keeps the orchestrator from giving up while
    # the worker is still running.
    ingest_request_timeout_s: float = Field(600.0, ge=1.0, le=3600.0)

    # blackheart-inference (ML sidecar — Phase 4 stage A3, 2026-05-18).
    # Loopback by default. POST /inference/run + /inference/backfill on the
    # orchestrator proxy to ``{inference_base_url}/inference/*``. The token
    # is the inference service's X-Inference-Token; the orchestrator owns
    # its own ORCH_INFERENCE_AUTH_TOKEN env so caller-side and service-side
    # secrets stay separate (matches the JVM auth pattern).
    inference_base_url: str = "http://127.0.0.1:8000"
    inference_request_timeout_s: float = Field(120.0, ge=1.0, le=3600.0)
    # Sidecar's default INFERENCE_AUTH_TOKEN is the literal string "dev-token"
    # (see blackheart-inference/src/blackheart_inference/settings.py). The
    # orchestrator's own auth-token sentinel is a different value, so we
    # can't reuse DEV_TOKEN_SENTINEL here — defaulting to "dev-token" keeps
    # local dev frictionless. Production deployments MUST set both
    # ORCH_INFERENCE_AUTH_TOKEN (here) and INFERENCE_AUTH_TOKEN (sidecar)
    # to the same fresh secret; the sidecar's assert_prod_safe enforces
    # that its own value isn't the dev sentinel under PROFILE=prod.
    inference_auth_token: SecretStr = SecretStr("dev-token")

    # AI paper narrative (paper_narrator.py).
    # Set ORCH_AI_NARRATIVE_ENABLED=1 to enable Claude CLI-based IEEE prose
    # generation on POST /papers/{id}/generate. When disabled (default) the
    # deterministic template is always used — safe for installs without the
    # Max-plan claude CLI. Timeout is per-generation subprocess call.
    ai_narrative_enabled: bool = False
    ai_narrative_timeout_s: int = Field(240, ge=30, le=600)
    ai_narrative_claude_bin: str = "claude"

    # Trust the proxy-injected X-Viewer-Is-Admin header (api/deps.py).
    # Only safe when a header-stripping proxy fronts :8082 — a direct
    # caller holding the shared token can forge the header otherwise.
    # Set ORCH_TRUST_VIEWER_ADMIN_HEADER=false on proxyless deployments.
    trust_viewer_admin_header: bool = True

    # Specialist agent model pins (Path A, Max-plan-only — 2026-05-19).
    # No API client; specialists run as Claude Code Agent-tool sub-agents
    # (or, as a fallback, via /claude -p subprocess) so every token bills
    # against the operator's Max plan. The model names below get pinned
    # in each specialist's .claude/agents/<name>.md frontmatter — keeping
    # them here as well lets test_config catch any drift to opus that
    # would silently bloat per-decision cost.
    specialist_model_skeptic: str = "claude-sonnet-4-6"
    specialist_model_portfolio: str = "claude-sonnet-4-6"
    specialist_model_capacity: str = "claude-sonnet-4-6"
    specialist_model_data_scout: str = "claude-sonnet-4-6"

    # ML training orchestration (Phase A, 2026-05-19).
    # POST /ml/training-runs spawns blackheart-train as a subprocess.
    # The CLI lives at ``{ml_training_repo_path}/src/blackheart_train/cli.py``
    # and is invoked via the project's own venv interpreter so dependency
    # boundaries stay clean. Daily cap is per-agent to prevent any one
    # researcher session from racking up unbounded train-runs; FAILED rows
    # don't count toward the budget (infra failures shouldn't punish
    # exploration).
    ml_training_python_path: str = (
        r"C:\Project\blackheart-train\.venv\Scripts\python.exe"
    )
    ml_training_repo_path: str = r"C:\Project\blackheart-train"
    # Artifact output dir for the train subprocess (TRAIN_ARTIFACT_DIR).
    # blackheart-train's own Settings.artifact_dir defaults to the Windows
    # host path C:/Project/blackheart-train/artifacts — wrong inside the
    # Linux orchestrator container, where the train repo is bind-mounted
    # read-only at /train:ro and `C:` does not exist (OSError [Errno 30]
    # Read-only file system: 'C:' at write_artifact). When set, this value
    # is injected into the subprocess env as TRAIN_ARTIFACT_DIR so the
    # child writes to a WRITABLE in-container path (operator mounts a
    # writable volume there, e.g. ORCH_ML_TRAINING_ARTIFACT_DIR=/artifacts
    # with a `- artifacts:/artifacts` compose volume). Empty string = leave
    # the child's own default untouched (dev-host behaviour preserved).
    # Same wiring-only pattern as _dsn_to_train_db_env: only emit the env
    # var when we have a value to emit. Contract-safe — no Settings/auth/
    # gate change, just subprocess plumbing.
    ml_training_artifact_dir: str = ""
    ml_training_daily_cap: int = Field(4, ge=1, le=100)
    ml_training_daily_window_hours: int = Field(24, ge=1, le=168)
    # Wall-clock cap per training subprocess. blackheart-train walk-
    # forward + gauntlet on a 1h-bar 2-year dataset typically finishes
    # in 30-90 min; the 4h ceiling covers Bayesian + 6-fold without
    # leaving runaway processes alive forever.
    ml_training_max_seconds: int = Field(14400, ge=60, le=86400)
    # Per-row stdout/stderr capture cap. Tail-only — the artifact and
    # experiment_run rows carry the structured detail.
    ml_training_stdio_tail_bytes: int = Field(16384, ge=1024, le=1048576)
    # ──────────────────────────────────────────────────────────────────
    # Nightly portfolio-rebalance task (scorecard #10 portfolio construction).
    # The optimizer (services/rebalance.rebalance_book) is exposed at
    # POST /portfolio/rebalance, but nothing ran it on a schedule, so live
    # account_strategy.portfolio_weight stayed flat EQUAL_WEIGHT. This task
    # closes that loop. OFF by default — automated rebalancing writes weights
    # that multiply into live order sizing, so it is opt-in
    # (ORCH_REBALANCE_ENABLED=true). Safe on a thin/dormant book: rebalance_book
    # falls back to equal-weight on insufficient history and no-ops when there
    # are no eligible strategies; per-account errors are isolated.
    rebalance_enabled: bool = False
    rebalance_interval_seconds: int = Field(86400, ge=3600, le=604800)
    rebalance_optimizer: Literal["HRP", "EQUAL_WEIGHT", "MEAN_VARIANCE"] = "HRP"
    # When True, the task computes + journals proposed weights but does NOT
    # write them (observe-only). False = live writes once the task is enabled.
    rebalance_dry_run: bool = False
    # Phase D (2026-05-19) — researcher discovers what model specs it can
    # train via GET /ml/model-specs. This list mirrors
    # blackheart-train/src/blackheart_train/specs.py's _spec_choices().
    # Override via env: ORCH_AVAILABLE_MODEL_SPECS='["regime_btc_v3","regime_btc_v4"]'
    # The orchestrator does NOT validate spec_name on POST /ml/training-runs
    # — the subprocess argparse handles that. This list is a discovery
    # helper; it's the researcher's job not to ask for a spec not on it.
    available_model_specs: list[str] = Field(
        default_factory=lambda: [
            "regime_btc_v1",
            "regime_btc_v2",
            "regime_btc_v3",
            "positioning_btc_v1",
            "positioning_btc_v2",
            "flow_btc_v1",
            "flow_btc_v2",
            "directional_btc_1h_v1",
            # 2026-05-21: binary directional twin of v1 — resolves
            # ANTI_PATTERN 20fa437f (the orchestrator model_registry
            # validator only accepts objective ∈ {binary, regression}, so
            # v1's multiclass artifact cannot land in model_registry).
            # v2 reuses label_regime_risk_on_24h (V77/V110 binary
            # forward-Sharpe-sign label), single-base LightGBM, no
            # stacking, no meta-label — minimum-surface fix to unblock
            # HYBRID ML hypotheses on the researcher loop.
            "directional_btc_1h_v2",
            # 2026-05-21: bar-boundary triple-barrier binary spec.
            # Addresses the smoothed-aggregate label misalignment that
            # falsified v2's forward-Sharpe-sign gate on three host
            # strategies (DCB-BTC, MMR-BTC, DCB-ETH). Same feature
            # stack as v2; new in-train derived label
            # label_long_win_tb_1h_v1 (TP-first-within-24-bars binary).
            "directional_btc_1h_v3",
            # 2026-05-21 (Path B): same triple-barrier label as v3
            # (label_long_win_tb_1h_v1) but augments the registry-pulled
            # feature stack with 5 NEW bar-level entry-timing features
            # registered via Flyway V111: btc_rsi_14_1h, btc_atr_ratio_14_24,
            # btc_log_return_1h, btc_log_return_4h, btc_volume_zscore_4h.
            # Tests the operator hypothesis (RUN_SUMMARY a957d7cf) that
            # the gate harm in v2/v3 was structural to the 24h-aggregation
            # feature stack regardless of label choice.
            "directional_btc_1h_v4",
            # 2026-05-21 (Path C cont'd): short-horizon (6-bar) asymmetric
            # triple-barrier label, same feature stack as v4. Tests whether
            # label time-scope alignment to strategy holding period lifts
            # paired-delta past the -7 pp ceiling v4 hit.
            "directional_btc_1h_v5",
            # 2026-05-21 Path C late: loose-threshold (k_tp=1.0, k_sl=0.5)
            # short-horizon variant. Tests whether less restrictive model
            # → more gate-on trades through → V11 n-floor cleared.
            "directional_btc_1h_v6",
            # 2026-05-20: first per-symbol ETH ML spec. Authored after
            # bar-OHLCV parametric search hit lifecycle exhaustion across
            # BTC+ETH × 4 intervals. Backed by V110 migration which
            # expanded 5 derived features+label to symbols=
            # ('BTCUSDT','ETHUSDT'). See specs.py docstring on
            # regime_eth_v1 for design notes.
            "regime_eth_v1",
        ]
    )

    @field_validator("auth_token")
    @classmethod
    def _reject_dev_sentinel_on_prod(cls, v: SecretStr, info) -> SecretStr:
        # Pydantic-settings field validators don't get sibling values cleanly,
        # so we re-check at app startup too. This is the first line.
        return v

    @field_validator("db_pool_max")
    @classmethod
    def _pool_max_ge_min(cls, v: int, info) -> int:
        pool_min = info.data.get("db_pool_min", 2)
        if v < pool_min:
            raise ValueError(f"db_pool_max ({v}) must be >= db_pool_min ({pool_min})")
        return v

    def assert_prod_safe(self) -> None:
        """Called at startup. Refuses to boot a prod profile with dev secrets
        or dev-bypass JVM auth."""
        if self.profile != "prod":
            return
        if self.auth_token.get_secret_value() == DEV_TOKEN_SENTINEL:
            raise RuntimeError(
                "Refusing to start: ORCH_PROFILE=prod with dev-sentinel ORCH_AUTH_TOKEN. "
                "Generate a real secret."
            )
        if self.jvm_auth_mode == "dev_bypass":
            raise RuntimeError(
                "Refusing to start: ORCH_PROFILE=prod with ORCH_JVM_AUTH_MODE=dev_bypass. "
                "Switch to service_account."
            )
        if self.jvm_auth_mode == "service_account" and (
            not self.jvm_service_user or not self.jvm_service_password
        ):
            raise RuntimeError(
                "Refusing to start: ORCH_JVM_AUTH_MODE=service_account requires "
                "ORCH_JVM_SERVICE_USER and ORCH_JVM_SERVICE_PASSWORD."
            )
        if self.research_account_id is None:
            raise RuntimeError(
                "Refusing to start: ORCH_PROFILE=prod requires ORCH_RESEARCH_ACCOUNT_ID "
                "(the account_id of the research-agent account seeded by Flyway V54). "
                "Without it, tick.py would resolve account_strategy rows by "
                "strategy_code alone and could pick admin-owned rows."
            )

    def log_dev_mode_warnings(self, logger) -> None:
        """M7 (2026-05-16) — surface known-dangerous-if-promoted dev defaults.

        ``assert_prod_safe`` refuses to boot a *prod* profile with dev
        defaults — but a developer can still boot the *dev* profile with
        every safeguard disabled. If that JVM is then exposed beyond
        loopback (staging box, Tailscale-shared host, copy-pasted env
        file), the dev defaults silently become a privilege-escalation
        path: ``dev_bypass`` JVM auth means the orchestrator
        authenticates as "the first user" — typically the admin — and
        every backtest is attributed to that account.

        This method emits a single structured ``WARNING`` per known
        dev-only setting at startup so the operator sees the gap in
        their startup logs even when ``ORCH_PROFILE=dev``. No-op on
        ``ORCH_PROFILE=prod`` (assert_prod_safe has already refused the
        boot if any of these are unsafe).

        Callers: invoked once from the FastAPI lifespan hook after
        ``assert_prod_safe``.
        """
        if self.profile == "prod":
            return
        if self.jvm_auth_mode == "dev_bypass":
            logger.warning(
                "orchestrator.dev_warning",
                detail=(
                    "ORCH_JVM_AUTH_MODE=dev_bypass — authenticating to the "
                    "research JVM as the first user via /api/v1/dev/login-as. "
                    "DO NOT promote this configuration to a shared host. "
                    "Set ORCH_PROFILE=prod + ORCH_JVM_AUTH_MODE=service_account "
                    "before exposing this orchestrator beyond loopback."
                ),
                jvm_auth_mode=self.jvm_auth_mode,
                jvm_dev_login_email=self.jvm_dev_login_email,
            )
        if self.auth_token.get_secret_value() == DEV_TOKEN_SENTINEL:
            logger.warning(
                "orchestrator.dev_warning",
                detail=(
                    "ORCH_AUTH_TOKEN is the dev sentinel value. Any caller "
                    "that knows the sentinel can drive the orchestrator. "
                    "Generate a real token before exposing beyond loopback: "
                    "python -c \"import secrets; print(secrets.token_urlsafe(48))\""
                ),
            )
        if self.research_account_id is None:
            logger.warning(
                "orchestrator.dev_warning",
                detail=(
                    "ORCH_RESEARCH_ACCOUNT_ID is unset — account_strategy "
                    "lookup falls back to 'first matching row' which can "
                    "select an admin-owned row on a multi-account host."
                ),
            )
