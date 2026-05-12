from __future__ import annotations
from uuid import UUID
from fastapi import APIRouter, Request, Query
from pydantic import BaseModel
from typing import Any
import logging
from orchestrator.repo import activity as activity_repo
from orchestrator.services.activity_logger import log_activity

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/activity", tags=["activity"])


class ActivityCreate(BaseModel):
    session_id: UUID | None = None
    activity_type: str
    title: str
    strategy_code: str | None = None
    details: dict[str, Any] | None = None
    related_id: UUID | None = None
    related_type: str | None = None
    status: str = "SUCCESS"


@router.post("", status_code=201)
async def create_activity(body: ActivityCreate, request: Request):
    # Prefer agent identity from auth middleware (already validated) over
    # the raw header — middleware default is "anonymous" so this matches
    # the pre-existing fallback semantics.
    agent_name = getattr(request.state, "agent_name", None) or request.headers.get("X-Agent-Name", "unknown")
    # Body wins for explicit grouping. Fall back to the middleware-resolved
    # session_id (header value or per-(agent, day) synth) so SESSION_START
    # and similar markers never land with NULL session_id.
    session_id = body.session_id or getattr(request.state, "session_id", None)
    redis_client = request.app.state.redis
    async with request.app.state.db.acquire() as conn:
        row = await activity_repo.insert_activity(
            conn,
            session_id=session_id,
            agent_name=agent_name,
            activity_type=body.activity_type,
            title=body.title,
            strategy_code=body.strategy_code,
            details=body.details,
            related_id=body.related_id,
            related_type=body.related_type,
            status=body.status,
        )
        # Publish to Redis for real-time frontend streaming
        if redis_client is not None:
            import json
            payload = json.dumps(
                {
                    "activityId": str(row["activity_id"]),
                    "sessionId": str(session_id) if session_id else None,
                    "agentName": agent_name,
                    "activityType": body.activity_type,
                    "strategyCode": body.strategy_code,
                    "title": body.title,
                    "details": body.details,
                    "relatedId": str(body.related_id) if body.related_id else None,
                    "relatedType": body.related_type,
                    "status": body.status,
                    "createdAt": row["created_at"].isoformat(),
                },
                default=str,
            )
            try:
                await redis_client.publish("research:activity", payload)
            except Exception as _pub_exc:  # noqa: BLE001
                logger.warning("Redis publish failed (non-fatal): %s", _pub_exc)
    return {"activity_id": str(row["activity_id"])}


@router.get("")
async def list_activities(
    request: Request,
    session_id: UUID | None = None,
    agent_name: str | None = None,
    activity_type: str | None = None,
    strategy_code: str | None = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    async with request.app.state.db.acquire() as conn:
        rows = await activity_repo.list_activities(
            conn,
            session_id=session_id,
            agent_name=agent_name,
            activity_type=activity_type,
            strategy_code=strategy_code,
            limit=limit,
            offset=offset,
        )
        total = await activity_repo.count_activities(
            conn,
            session_id=session_id,
            agent_name=agent_name,
            activity_type=activity_type,
            strategy_code=strategy_code,
        )
    # Serialize UUIDs and datetimes
    serialized = [
        {k: str(v) if isinstance(v, UUID) else v.isoformat() if hasattr(v, 'isoformat') else v
         for k, v in row.items()}
        for row in rows
    ]
    return {"activities": serialized, "total": total, "limit": limit, "offset": offset}


@router.get("/sessions")
async def list_sessions(
    request: Request,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    async with request.app.state.db.acquire() as conn:
        sessions = await activity_repo.list_sessions(conn, limit=limit, offset=offset)
        total = await activity_repo.count_sessions(conn)
    # Serialize
    result = []
    for s in sessions:
        row = {
            k: (str(v) if isinstance(v, UUID) else
                v.isoformat() if hasattr(v, 'isoformat') else
                list(v) if v is not None and hasattr(v, '__iter__') and not isinstance(v, (str, dict)) else v)
            for k, v in s.items()
        }
        if row.get('strategy_codes') is None:
            row['strategy_codes'] = []
        result.append(row)
    return {"sessions": result, "total": total, "limit": limit, "offset": offset}
