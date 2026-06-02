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
    own, only the shared agent token. The header is injected server-side by the
    proxy and the proxy never forwards a client-supplied copy, so an end user
    cannot spoof it.

    Fails CLOSED: anything other than a literal ``"true"`` (including an absent
    header on a direct loopback call) is treated as non-admin, so a proxy bug
    can only ever show *fewer* papers, never leak in-progress ones. Trusted
    direct callers (operator curl) opt in with ``-H 'X-Viewer-Is-Admin: true'``.
    """
    return request.headers.get("X-Viewer-Is-Admin", "").strip().lower() == "true"
