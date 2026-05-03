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
