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


async def hourly_feature_refresh(settings: Settings) -> None:
    """Coroutine that refreshes features on a 1-hour loop.

    Calls POST {ingest_base_url}/compute/refresh to trigger feature compute
    from the last-persisted timestamp up to the present. This is a best-effort
    background task — exceptions are logged but do not crash the app.

    Args:
        settings: Orchestrator settings (ingest_base_url, ingest_request_timeout_s)
    """
    while True:
        try:
            await asyncio.sleep(FEATURE_REFRESH_INTERVAL_S)
            log.info("feature_refresh.start")

            async with httpx.AsyncClient(
                timeout=settings.ingest_request_timeout_s
            ) as client:
                response = await client.post(
                    f"{settings.ingest_base_url}/compute/refresh",
                    json={},
                )
                response.raise_for_status()
                result: dict[str, Any] = response.json()
                log.info(
                    "feature_refresh.complete",
                    features_computed=result.get("features_computed"),
                    rows_written=result.get("rows_written"),
                    compute_duration_s=result.get("compute_duration_s"),
                )
        except asyncio.CancelledError:
            log.info("feature_refresh.shutdown")
            raise
        except httpx.HTTPError as exc:
            log.warning("feature_refresh.http_error", error=repr(exc))
        except Exception as exc:  # noqa: BLE001
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
