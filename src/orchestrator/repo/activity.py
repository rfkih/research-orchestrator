from __future__ import annotations
from uuid import UUID, uuid4
from typing import Any


async def insert_activity(
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
) -> dict:
    """Insert one activity row. Returns dict with activity_id and created_at."""
    row = await conn.fetchrow(
        """
        INSERT INTO research_agent_activity
            (activity_id, session_id, agent_name, activity_type, strategy_code,
             title, details, related_id, related_type, status)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
        RETURNING activity_id, created_time
        """,
        uuid4(), session_id, agent_name, activity_type, strategy_code,
        title, details, related_id, related_type, status,
    )
    return {"activity_id": row["activity_id"], "created_at": row["created_time"]}


async def list_activities(
    conn,
    *,
    session_id: UUID | None = None,
    agent_name: str | None = None,
    activity_type: str | None = None,
    strategy_code: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    wheres, params = ["TRUE"], []
    n = 1
    for val, col in [
        (session_id, "session_id"),
        (agent_name, "agent_name"),
        (activity_type, "activity_type"),
        (strategy_code, "strategy_code"),
    ]:
        if val is not None:
            wheres.append(f"{col} = ${n}"); params.append(val); n += 1
    params += [limit, offset]
    # activity_id tiebreaker: same-timestamp rows are common (a tick logs
    # several activities in one transaction) and OFFSET pagination over an
    # unstable order skips/duplicates rows at page boundaries.
    direction = "ASC" if session_id else "DESC"
    rows = await conn.fetch(
        f"""
        SELECT activity_id, session_id, agent_name, activity_type, strategy_code,
               title, details, related_id, related_type, status, created_time
        FROM research_agent_activity
        WHERE {" AND ".join(wheres)}
        ORDER BY created_time {direction}, activity_id {direction}
        LIMIT ${n} OFFSET ${n+1}
        """,
        *params,
    )
    return [dict(r) for r in rows]


async def count_activities(
    conn,
    *,
    session_id: UUID | None = None,
    agent_name: str | None = None,
    activity_type: str | None = None,
    strategy_code: str | None = None,
) -> int:
    wheres, params = ["TRUE"], []
    n = 1
    for val, col in [
        (session_id, "session_id"),
        (agent_name, "agent_name"),
        (activity_type, "activity_type"),
        (strategy_code, "strategy_code"),
    ]:
        if val is not None:
            wheres.append(f"{col} = ${n}"); params.append(val); n += 1
    row = await conn.fetchrow(
        f"SELECT COUNT(*) FROM research_agent_activity WHERE {' AND '.join(wheres)}",
        *params,
    )
    return row[0]


async def list_sessions(conn, *, limit: int = 20, offset: int = 0) -> list[dict]:
    rows = await conn.fetch(
        """
        SELECT
            session_id,
            agent_name,
            MIN(created_time)   AS started_at,
            MAX(created_time)   AS last_activity_at,
            COUNT(*)          AS activity_count,
            array_agg(DISTINCT strategy_code) FILTER (WHERE strategy_code IS NOT NULL)
                              AS strategy_codes,
            COUNT(*) FILTER (WHERE activity_type = 'ITERATION_COMPLETED')
                              AS iterations_completed,
            COUNT(*) FILTER (WHERE activity_type = 'ITERATION_COMPLETED'
                AND details->>'statistical_verdict' = 'SIGNIFICANT_EDGE')
                              AS significant_edge_count,
            COUNT(*) FILTER (WHERE activity_type = 'ITERATION_COMPLETED'
                AND details->>'statistical_verdict' = 'NO_EDGE')
                              AS no_edge_count,
            COUNT(*) FILTER (WHERE activity_type = 'ITERATION_COMPLETED'
                AND details->>'statistical_verdict' = 'DISCARD')
                              AS discard_count,
            BOOL_OR(activity_type = 'GOAL_HIT') AS goal_hit
        FROM research_agent_activity
        GROUP BY session_id, agent_name
        ORDER BY MAX(created_time) DESC
        LIMIT $1 OFFSET $2
        """,
        limit, offset,
    )
    return [dict(r) for r in rows]


async def count_sessions(conn) -> int:
    # Must match list_sessions' GROUP BY (session_id, agent_name) exactly:
    # COUNT(DISTINCT session_id) drops NULL session_ids (which form real
    # groups in the list) and collapses a session shared by two agent
    # names — making the paginated total smaller than the rows served.
    row = await conn.fetchrow(
        """
        SELECT COUNT(*) FROM (
            SELECT 1
            FROM research_agent_activity
            GROUP BY session_id, agent_name
        ) g
        """
    )
    return row[0]
