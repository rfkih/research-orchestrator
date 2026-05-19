"""Tests for ``orchestrator.config.Settings``.

Covers M7 (2026-05-16) — the ``log_dev_mode_warnings`` startup hook
that surfaces dev defaults which become privilege-escalation paths if
the orchestrator is promoted to a shared host without re-configuring.
``assert_prod_safe`` already refuses to boot prod with those defaults;
this hook covers the *dev* path so the gap is visible in startup logs.
"""

from __future__ import annotations

from uuid import UUID

import pytest
from pydantic import SecretStr

from orchestrator.config import DEV_TOKEN_SENTINEL, Settings


# ── Fixtures ──────────────────────────────────────────────────────────────


class _RecordingLogger:
    """Minimal structlog-shaped fake — captures keyword args on warning()
    so tests can assert the structured fields without standing up structlog."""

    def __init__(self) -> None:
        self.warnings: list[tuple[str, dict]] = []

    def warning(self, event: str, **kwargs: object) -> None:
        self.warnings.append((event, dict(kwargs)))


def _settings(**overrides) -> Settings:
    base = dict(
        profile="dev",
        auth_token=SecretStr("real-secret-not-the-sentinel"),
        db_dsn=SecretStr("postgresql://x:y@127.0.0.1:5432/none"),
        jvm_base_url="http://127.0.0.1:8081",
        jvm_auth_mode="service_account",
        jvm_service_user="research-agent@blackheart.local",
        jvm_service_password=SecretStr("real-password"),
        research_account_id=UUID("99999999-9999-9999-9999-000000000002"),
    )
    base.update(overrides)
    return Settings(**base)


# ── M7 — log_dev_mode_warnings ─────────────────────────────────────────────


def test_clean_dev_config_no_warnings():
    """A fully-configured dev profile (service_account auth, real token,
    research_account_id pinned) emits no warnings — only mis-configured
    dev setups should appear in the startup log."""
    settings = _settings()
    log = _RecordingLogger()
    settings.log_dev_mode_warnings(log)
    assert log.warnings == [], (
        f"clean dev config should emit zero warnings; got {log.warnings}"
    )


def test_dev_bypass_emits_warning():
    """ORCH_JVM_AUTH_MODE=dev_bypass — the primary M7 footgun. Must emit
    one structured warning so the operator sees it on startup."""
    settings = _settings(jvm_auth_mode="dev_bypass")
    log = _RecordingLogger()
    settings.log_dev_mode_warnings(log)
    assert len(log.warnings) == 1
    event, fields = log.warnings[0]
    assert event == "orchestrator.dev_warning"
    assert "dev_bypass" in fields["detail"]
    assert fields["jvm_auth_mode"] == "dev_bypass"


def test_dev_sentinel_token_emits_warning():
    """ORCH_AUTH_TOKEN left at the dev sentinel must warn — sentinel is
    public knowledge and any caller who knows it can drive the service."""
    settings = _settings(auth_token=SecretStr(DEV_TOKEN_SENTINEL))
    log = _RecordingLogger()
    settings.log_dev_mode_warnings(log)
    detail_warnings = [w for w in log.warnings if "sentinel" in w[1].get("detail", "")]
    assert len(detail_warnings) == 1


def test_unset_research_account_id_emits_warning():
    """ORCH_RESEARCH_ACCOUNT_ID unset → tick falls back to first-matching
    row, which can be admin-owned on a multi-account host."""
    settings = _settings(research_account_id=None)
    log = _RecordingLogger()
    settings.log_dev_mode_warnings(log)
    detail_warnings = [w for w in log.warnings if "ORCH_RESEARCH_ACCOUNT_ID" in w[1].get("detail", "")]
    assert len(detail_warnings) == 1


def test_multiple_dev_defaults_emit_multiple_warnings():
    """A maximally-permissive dev config: every check must fire its own
    warning, not silently coalesce. Operator needs to see the full
    extent of the gap."""
    settings = _settings(
        jvm_auth_mode="dev_bypass",
        auth_token=SecretStr(DEV_TOKEN_SENTINEL),
        research_account_id=None,
    )
    log = _RecordingLogger()
    settings.log_dev_mode_warnings(log)
    assert len(log.warnings) == 3


def test_prod_profile_emits_no_warnings():
    """On ``ORCH_PROFILE=prod``, ``assert_prod_safe`` has already refused
    to boot if any of these settings are unsafe — the warning hook is
    no-op so the prod startup log isn't polluted with dev-only WARNs
    that don't apply."""
    settings = _settings(
        profile="prod",
        jvm_auth_mode="dev_bypass",  # would be refused by assert_prod_safe
    )
    log = _RecordingLogger()
    settings.log_dev_mode_warnings(log)
    assert log.warnings == [], (
        f"prod profile should never emit dev-mode warnings (assert_prod_safe "
        f"is the binding check); got {log.warnings}"
    )


# ── assert_prod_safe pins (existing behavior, not changed by M7) ──────────


def test_assert_prod_safe_refuses_dev_bypass_on_prod():
    """Pin the existing M7-adjacent invariant: prod + dev_bypass = refuse boot.
    Defense-in-depth check that the assert_prod_safe contract is unchanged
    by the M7 warning hook."""
    settings = _settings(profile="prod", jvm_auth_mode="dev_bypass")
    with pytest.raises(RuntimeError, match="ORCH_JVM_AUTH_MODE=dev_bypass"):
        settings.assert_prod_safe()


def test_assert_prod_safe_passes_clean_prod():
    settings = _settings(profile="prod")
    settings.assert_prod_safe()  # no raise


def test_assert_prod_safe_noop_on_dev():
    """Pin: assert_prod_safe doesn't raise on dev no matter how bad the
    config — the warning hook handles that signal."""
    settings = _settings(
        jvm_auth_mode="dev_bypass",
        auth_token=SecretStr(DEV_TOKEN_SENTINEL),
        research_account_id=None,
    )
    settings.assert_prod_safe()  # no raise — dev profile


# ── Specialist model pins (Path A, Max-plan-only — 2026-05-19) ────────────


def test_specialist_model_defaults_are_sonnet():
    """Pin: default specialist models are sonnet-4-6 across the board.
    A drift here (e.g. someone bumps skeptic to opus) shows up in the
    operator's Max-plan token budget before anywhere else; the pin
    catches it pre-deploy. The values get propagated into each
    .claude/agents/<specialist>.md frontmatter — single source of truth
    here means a model swap is one PR, not five."""
    settings = _settings()
    assert settings.specialist_model_skeptic == "claude-sonnet-4-6"
    assert settings.specialist_model_portfolio == "claude-sonnet-4-6"
    assert settings.specialist_model_capacity == "claude-sonnet-4-6"
    assert settings.specialist_model_data_scout == "claude-sonnet-4-6"
