"""FastAPI lifespan hook.

Owns the open/close lifecycle of the DB pool and JVM client. The app object
exposes them on ``app.state.db`` / ``app.state.jvm`` so handlers can reach
them via ``request.app.state``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ..clients.jvm import JvmClient
from ..config import Settings
from ..infra.db import Database
from ..logging import get_logger
from ..services.idempotency import InMemoryIdempotencyStore, PostgresIdempotencyStore

log = get_logger(__name__)


@asynccontextmanager
async def lifespan_for(settings: Settings, app: FastAPI) -> AsyncIterator[None]:
    settings.assert_prod_safe()

    db = Database(settings)
    jvm = JvmClient(settings)
    app.state.settings = settings
    app.state.db = db
    app.state.jvm = jvm
    await db.open()
    if settings.profile == "prod":
        app.state.idempotency = PostgresIdempotencyStore(db)
    else:
        app.state.idempotency = InMemoryIdempotencyStore()
    await jvm.open()

    log.info(
        "orchestrator.startup",
        profile=settings.profile,
        host=settings.host,
        port=settings.port,
        jvm_base_url=settings.jvm_base_url,
        jvm_auth_mode=settings.jvm_auth_mode,
    )
    try:
        yield
    finally:
        log.info("orchestrator.shutdown")
        await jvm.close()
        await db.close()
