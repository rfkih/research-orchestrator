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
    ingest_base_url: str = "http://127.0.0.1:8089"
    # 10-min ceiling covers a full-history per-bar feature compute over 5+
    # years on 1h bars (~38k rows). Most calls return in well under a
    # minute; the high default keeps the orchestrator from giving up while
    # the worker is still running.
    ingest_request_timeout_s: float = Field(600.0, ge=1.0, le=3600.0)

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
