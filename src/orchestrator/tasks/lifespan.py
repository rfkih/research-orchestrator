"""FastAPI lifespan hook.

Owns the open/close lifecycle of the DB pool and JVM client. The app object
exposes them on ``app.state.db`` / ``app.state.jvm`` so handlers can reach
them via ``request.app.state``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI

from ..clients.jvm import JvmClient
from ..config import Settings
from ..infra.db import Database
from ..logging import get_logger
from ..observability.error_reporter import get_reporter
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

    # Redis pub/sub — optional. None when ORCH_REDIS_URL is not configured.
    if settings.redis_url:
        app.state.redis = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
        log.info("orchestrator.redis_connected", redis_url=settings.redis_url)
    else:
        app.state.redis = None

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
        if app.state.redis is not None:
            await app.state.redis.aclose()
        await jvm.close()
        await db.close()
        reporter = get_reporter()
        if reporter is not None:
            await reporter.aclose()
