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

from ..clients.inference import InferenceClient
from ..clients.jvm import JvmClient
from ..clients.trading_jvm import TradingJvmClient
from ..config import Settings
from ..infra.db import Database
from ..logging import get_logger
from ..observability.error_reporter import get_reporter
from ..services.idempotency import InMemoryIdempotencyStore, PostgresIdempotencyStore
from ..services.ml_training import reap_orphaned_on_startup

log = get_logger(__name__)


@asynccontextmanager
async def lifespan_for(settings: Settings, app: FastAPI) -> AsyncIterator[None]:
    settings.assert_prod_safe()
    # M7 (2026-05-16) — print operator-visible WARN for dev defaults that
    # become privilege-escalation paths if the orchestrator is promoted
    # to a shared host without re-configuring. assert_prod_safe handles
    # the hard refuse on ORCH_PROFILE=prod; this surfaces the soft
    # gap on dev. No-op on prod.
    settings.log_dev_mode_warnings(log)

    db = Database(settings)
    jvm = JvmClient(settings)
    trading_jvm = TradingJvmClient(settings)
    inference = InferenceClient(settings)
    app.state.settings = settings
    app.state.db = db
    app.state.jvm = jvm
    app.state.trading_jvm = trading_jvm
    app.state.inference = inference
    await db.open()
    if settings.profile == "prod":
        app.state.idempotency = PostgresIdempotencyStore(db)
    else:
        app.state.idempotency = InMemoryIdempotencyStore()
    await jvm.open()
    await trading_jvm.open()
    await inference.open()

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

    # V99 reaper — flip any ml_training_runs row still RUNNING from a
    # prior process to ORPHANED. Subprocesses may still be alive
    # (detached at the OS level), but our handle is gone. Best-effort:
    # a DB-connection blip during startup must NOT prevent app boot.
    try:
        n_reaped = await reap_orphaned_on_startup(db.pool)
        if n_reaped:
            log.warning("orchestrator.ml_training_orphaned_reaped", n=n_reaped)
    except Exception:  # noqa: BLE001
        log.exception("orchestrator.ml_training_reaper_failed")

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
        await inference.close()
        await trading_jvm.close()
        await jvm.close()
        await db.close()
        reporter = get_reporter()
        if reporter is not None:
            await reporter.aclose()
