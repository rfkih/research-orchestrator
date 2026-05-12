from __future__ import annotations
import json
import logging
from uuid import UUID
from typing import Any

from redis.asyncio import Redis as AsyncRedis

from orchestrator.repo import activity as activity_repo

logger = logging.getLogger(__name__)


async def log_activity(
    conn,
    *,
    session_id: UUID | None,
    agent_name: str,
    activity_type: str,
    title: str,
    strategy_code: str | None = None,
    details: dict[str, Any] | None = None,
    related_id: UUID | None = None,
    related_type: str | None = None,
    status: str = "SUCCESS",
    redis_client: AsyncRedis | None = None,  # NEW — pass app.state.redis
) -> None:
    """Fire-and-forget activity log. Never raises — logs error on failure.

    When ``redis_client`` is provided (non-None), publishes a JSON payload to
    the ``research:activity`` Redis channel after the DB insert so the frontend
    can stream events in real time. Silently skipped when ``redis_client`` is None.
    """
    try:
        row = await activity_repo.insert_activity(
            conn,
            session_id=session_id,
            agent_name=agent_name,
            activity_type=activity_type,
            title=title,
            strategy_code=strategy_code,
            details=details,
            related_id=related_id,
            related_type=related_type,
            status=status,
        )
        # Publish to Redis for real-time frontend streaming
        if redis_client is not None:
            payload = json.dumps(
                {
                    "activityId": str(row["activity_id"]),
                    "sessionId": str(session_id) if session_id else None,
                    "agentName": agent_name,
                    "activityType": activity_type,
                    "strategyCode": strategy_code,
                    "title": title,
                    "details": details,
                    "relatedId": str(related_id) if related_id else None,
                    "relatedType": related_type,
                    "status": status,
                    "createdAt": row["created_at"].isoformat(),
                },
                default=str,
            )
            await redis_client.publish("research:activity", payload)
    except Exception as exc:
        logger.warning("Activity log failed (non-fatal): %s", exc)
