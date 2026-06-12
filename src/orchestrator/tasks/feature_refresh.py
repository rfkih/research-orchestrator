"""Hourly feature refresh cron task.

Polls ingest service every hour to refresh features from the last computed
timestamp to the present time. This keeps the ML signal pipeline fed with
fresh feature data without relying on streaming polling.

Task runs in background; failures are best-effort and do not fail the app.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from ..config import Settings
from ..logging import get_logger

log = get_logger(__name__)

# Refresh interval: 3600 seconds (1 hour)
FEATURE_REFRESH_INTERVAL_S = 3600
# After this many consecutive failures, escalate from warning to ERROR so
# the ErrorReporter ships it (error_log + Telegram). A permanent failure —
# wrong route, dead container — must not warn politely forever while the
# ML signal pipeline silently starves.
CONSECUTIVE_FAILURES_TO_ESCALATE = 3


async def hourly_feature_refresh(settings: Settings) -> None:
    """Coroutine that refreshes features on a 1-hour loop.

    Calls ``POST {ingest_base_url}/compute/incremental`` — the route served
    by the DEPLOYED ingest entrypoint (``blackheart_ingest.workers.server``,
    :8001). The pre-2026-06-12 version POSTed ``/compute/refresh``, which
    exists only on the never-served ``app.py`` — every hourly attempt 404'd
    and was demoted to a warning, forever (memory:
    project_ingest_two_fastapi_apps_dead_refresh_route).

    Body is empty: the ingest side then uses its own
    ``compute_lookback_hours`` setting for the incremental window.

    The first attempt fires shortly after startup (not a full interval
    later) so a deploy cadence faster than 1h still refreshes at least once.
    """
    consecutive_failures = 0
    first_run = True
    while True:
        try:
            await asyncio.sleep(30 if first_run else FEATURE_REFRESH_INTERVAL_S)
            first_run = False
            log.info("feature_refresh.start")

            async with httpx.AsyncClient(
                timeout=settings.ingest_request_timeout_s
            ) as client:
                response = await client.post(
                    f"{settings.ingest_base_url}/compute/incremental",
                )
                response.raise_for_status()
                result: dict[str, Any] = response.json()
                consecutive_failures = 0
                log.info(
                    "feature_refresh.complete",
                    features_computed=result.get("features_computed"),
                    total_rows=result.get("total_rows"),
                    failures=result.get("failures"),
                    lookback_hours=result.get("lookback_hours"),
                )
        except asyncio.CancelledError:
            log.info("feature_refresh.shutdown")
            raise
        except httpx.HTTPError as exc:
            consecutive_failures += 1
            if consecutive_failures >= CONSECUTIVE_FAILURES_TO_ESCALATE:
                log.error(
                    "feature_refresh.persistent_failure",
                    error=repr(exc),
                    consecutive_failures=consecutive_failures,
                    ingest_base_url=settings.ingest_base_url,
                )
            else:
                log.warning("feature_refresh.http_error", error=repr(exc))
        except Exception as exc:  # noqa: BLE001
            consecutive_failures += 1
            log.exception("feature_refresh.error", error=repr(exc))


async def start_feature_refresh_task(settings: Settings) -> asyncio.Task[None]:
    """Create and return the feature refresh background task.

    The task is NOT awaited by the caller; it runs until cancelled (on shutdown).
    Returns the task object so the caller can clean it up if needed.

    Args:
        settings: Orchestrator settings

    Returns:
        asyncio.Task that will run the refresh loop until cancelled
    """
    task = asyncio.create_task(hourly_feature_refresh(settings))
    log.info("feature_refresh.task_started")
    return task
