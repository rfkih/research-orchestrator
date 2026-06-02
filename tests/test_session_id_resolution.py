"""Regression tests for X-Session-Id resolution + per-(agent, UTC date) synth.

Three layers:

1. Pure unit tests of ``_synth_session_id`` — determinism, agent isolation,
   day rollover, namespace stability.
2. TestClient tests for middleware-level rejection (paths that 400 before
   any handler runs).
3. Direct middleware ``dispatch`` invocation — verifies success-path state
   stamping (header parsed, synth fallback) without needing the full FastAPI
   dependency machinery, which would require a live DB pool.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from starlette.requests import Request

from orchestrator.auth import (
    AuthMiddleware,
    _SESSION_NS,
    _synth_session_id,
)
from orchestrator.config import Settings


# ── Synth helper ─────────────────────────────────────────────────────


def test_synth_is_deterministic_within_a_day() -> None:
    fixed = datetime(2026, 5, 9, 12, 0, 0, tzinfo=timezone.utc)
    a = _synth_session_id("quant-researcher", now=fixed)
    b = _synth_session_id("quant-researcher", now=fixed)
    assert a == b
    assert isinstance(a, uuid.UUID)


def test_synth_differs_per_agent_same_day() -> None:
    fixed = datetime(2026, 5, 9, tzinfo=timezone.utc)
    researcher = _synth_session_id("quant-researcher", now=fixed)
    reviewer = _synth_session_id("quant-reviewer", now=fixed)
    assert researcher != reviewer


def test_synth_differs_across_utc_day_boundary() -> None:
    # One second apart but on different UTC dates → different session.
    last_second = datetime(2026, 5, 9, 23, 59, 59, tzinfo=timezone.utc)
    first_second = datetime(2026, 5, 10, 0, 0, 0, tzinfo=timezone.utc)
    a = _synth_session_id("quant-researcher", now=last_second)
    b = _synth_session_id("quant-researcher", now=first_second)
    assert a != b


def test_synth_is_stable_within_a_day_regardless_of_hour() -> None:
    morning = datetime(2026, 5, 9, 0, 0, 0, tzinfo=timezone.utc)
    midnight_minus_one = datetime(2026, 5, 9, 23, 59, 59, tzinfo=timezone.utc)
    assert _synth_session_id("quant-researcher", now=morning) == _synth_session_id(
        "quant-researcher", now=midnight_minus_one
    )


def test_synth_namespace_is_pinned() -> None:
    # Pin the namespace constant so accidental drift surfaces — changing it
    # would re-bucket every historical session row.
    fixed = datetime(2026, 5, 9, tzinfo=timezone.utc)
    expected = uuid.uuid5(_SESSION_NS, "quant-researcher:2026-05-09")
    assert _synth_session_id("quant-researcher", now=fixed) == expected


# ── Middleware behavior ──────────────────────────────────────────────


def test_invalid_session_id_returns_400_envelope(client: TestClient) -> None:
    r = client.get(
        "/queue?status=PENDING",
        headers={
            "X-Orch-Token": "test-token",
            "X-Agent-Name": "quant-researcher",
            "X-Session-Id": "not-a-uuid",
        },
    )
    assert r.status_code == 400
    body = r.json()
    assert body["error_code"] == "invalid_session_id"
    assert body["retryable"] is False
    assert "X-Session-Id" in body["message"]


def test_public_path_skips_session_id_parsing(client: TestClient) -> None:
    # /agent/playbook is public — middleware short-circuits before reading
    # X-Session-Id. A garbage header on a public path must NOT 400.
    r = client.get(
        "/agent/playbook",
        headers={"X-Session-Id": "garbage"},
    )
    assert r.status_code == 200


# ── Middleware dispatch (success-path state stamping) ────────────────
#
# TestClient can't reach the handler success path without a live state.db
# (FastAPI deps run after middleware). Drive ``AuthMiddleware.dispatch``
# directly with a synthetic Request; capture state via a stub call_next.


def _make_request(path: str, headers: dict[str, str]) -> Request:
    """Build a minimal ASGI scope sufficient for AuthMiddleware.dispatch."""
    raw_headers = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": raw_headers,
        "scheme": "http",
        "server": ("testserver", 80),
        "client": ("test", 1234),
        "root_path": "",
    }
    return Request(scope)


@pytest.fixture
def middleware() -> AuthMiddleware:
    settings = Settings(
        profile="dev",
        auth_token=SecretStr("test-token"),
        db_dsn=SecretStr("postgresql://x:y@127.0.0.1:5432/none"),
        jvm_base_url="http://127.0.0.1:8081",
        jvm_auth_mode="dev_bypass",
    )
    # AuthMiddleware needs an ``app`` arg but never invokes it during dispatch
    # because we provide our own call_next.
    return AuthMiddleware(app=lambda *a, **kw: None, settings=settings)


@pytest.mark.asyncio
async def test_dispatch_stamps_parsed_session_id(middleware: AuthMiddleware) -> None:
    explicit = "12345678-1234-5678-1234-567812345678"
    captured: dict[str, object] = {}

    async def call_next(req: Request):
        captured["agent_name"] = req.state.agent_name
        captured["session_id"] = req.state.session_id
        return SimpleNamespace(status_code=200)

    req = _make_request(
        "/agent/state",
        {
            "X-Orch-Token": "test-token",
            "X-Agent-Name": "quant-researcher",
            "X-Session-Id": explicit,
        },
    )
    await middleware.dispatch(req, call_next)
    assert captured["agent_name"] == "quant-researcher"
    assert captured["session_id"] == uuid.UUID(explicit)


@pytest.mark.asyncio
async def test_dispatch_synthesizes_when_session_id_missing(
    middleware: AuthMiddleware,
) -> None:
    captured: dict[str, object] = {}

    async def call_next(req: Request):
        captured["session_id"] = req.state.session_id
        return SimpleNamespace(status_code=200)

    req = _make_request(
        "/agent/state",
        {"X-Orch-Token": "test-token", "X-Agent-Name": "quant-researcher"},
    )
    await middleware.dispatch(req, call_next)
    sid = captured["session_id"]
    assert isinstance(sid, uuid.UUID)
    # Same agent + same UTC day → must equal direct synth.
    assert sid == _synth_session_id("quant-researcher")


@pytest.mark.asyncio
async def test_dispatch_treats_empty_session_id_as_missing(
    middleware: AuthMiddleware,
) -> None:
    captured: dict[str, object] = {}

    async def call_next(req: Request):
        captured["session_id"] = req.state.session_id
        return SimpleNamespace(status_code=200)

    req = _make_request(
        "/agent/state",
        {
            "X-Orch-Token": "test-token",
            "X-Agent-Name": "quant-researcher",
            "X-Session-Id": "",
        },
    )
    await middleware.dispatch(req, call_next)
    # Empty header → fall through to synth, not a 400.
    assert isinstance(captured["session_id"], uuid.UUID)


@pytest.mark.asyncio
async def test_dispatch_public_path_sets_session_id_to_none(
    middleware: AuthMiddleware,
) -> None:
    captured: dict[str, object] = {}

    async def call_next(req: Request):
        captured["agent_name"] = req.state.agent_name
        captured["session_id"] = req.state.session_id
        return SimpleNamespace(status_code=200)

    req = _make_request("/agent/playbook", {"X-Session-Id": "garbage"})
    await middleware.dispatch(req, call_next)
    assert captured["agent_name"] == "anonymous"
    assert captured["session_id"] is None


@pytest.mark.asyncio
async def test_dispatch_non_ascii_token_returns_401_not_crash(
    middleware: AuthMiddleware,
) -> None:
    # A non-ASCII X-Orch-Token must produce a clean auth_bad_token 401, NOT a
    # 500 from secrets.compare_digest raising "comparing strings with non-ASCII
    # characters is not supported" (regression: it used to crash the request).
    call_next = AsyncMock()
    req = _make_request(
        "/agent/state",
        {"X-Orch-Token": "tökén-with-ünicode", "X-Agent-Name": "quant-researcher"},
    )
    response = await middleware.dispatch(req, call_next)
    call_next.assert_not_awaited()
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_dispatch_call_next_not_invoked_on_invalid_session_id(
    middleware: AuthMiddleware,
) -> None:
    call_next = AsyncMock()
    req = _make_request(
        "/agent/state",
        {
            "X-Orch-Token": "test-token",
            "X-Agent-Name": "quant-researcher",
            "X-Session-Id": "not-a-uuid",
        },
    )
    response = await middleware.dispatch(req, call_next)
    # Middleware short-circuits — handler never runs.
    call_next.assert_not_awaited()
    assert response.status_code == 400
