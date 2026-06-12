"""FastAPI dependencies.

Routers depend on these instead of poking ``request.app.state`` directly so
tests can override them without a real DB.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import asyncpg
from fastapi import Request


async def get_db_conn(request: Request) -> AsyncIterator[asyncpg.Connection]:
    """Yields a pooled connection. Released back to the pool on response."""
    async with request.app.state.db.acquire() as conn:
        yield conn


def get_agent_name(request: Request) -> str:
    """Identity stamped by ``AuthMiddleware``. Default ``'anonymous'``."""
    return getattr(request.state, "agent_name", "anonymous")


def get_viewer_is_admin(request: Request) -> bool:
    """Whether the end-user behind this request is an admin.

    Set by the trading-JVM proxy from the user's JWT role as the
    ``X-Viewer-Is-Admin`` header — the orchestrator has no user identity of its
    own, only the shared agent token. The header is only meaningful when a
    header-stripping proxy fronts the port; any caller that reaches :8082
    directly with the shared token can forge it. Deployments WITHOUT such a
    proxy must set ``ORCH_TRUST_VIEWER_ADMIN_HEADER=false``, which makes this
    dependency return False unconditionally (admin visibility then requires
    the proxy path).

    Fails CLOSED: anything other than a literal ``"true"`` (including an absent
    header on a direct loopback call) is treated as non-admin, so a proxy bug
    can only ever show *fewer* papers, never leak in-progress ones. Trusted
    direct callers (operator curl) opt in with ``-H 'X-Viewer-Is-Admin: true'``.
    """
    try:
        trusted = getattr(
            request.app.state.settings, "trust_viewer_admin_header", True
        )
    except (KeyError, AttributeError):
        # Bare Request (unit tests) — keep the header-only legacy behavior.
        trusted = True
    if not trusted:
        return False
    return request.headers.get("X-Viewer-Is-Admin", "").strip().lower() == "true"
