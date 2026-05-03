"""``research_journal`` reads.

Schema reference: ``V10__create_research_journal.sql``. Free-text search
uses the ``idx_research_journal_fulltext`` GIN tsvector index — the agent
sends a query string, we wrap it in ``plainto_tsquery`` so it stays safe.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

import asyncpg


async def list_journal(
    conn: asyncpg.Connection,
    *,
    entry_type: str | None,
    status: str | None,
    strategy_code: str | None,
    search: str | None,
    after_created_time: datetime | None,
    after_id: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    where = ["1=1"]
    args: list[Any] = []
    if entry_type is not None:
        args.append(entry_type)
        where.append(f"entry_type = ${len(args)}")
    if status is not None:
        args.append(status)
        where.append(f"status = ${len(args)}")
    if strategy_code is not None:
        args.append(strategy_code)
        where.append(f"strategy_code = ${len(args)}")
    if search:
        # plainto_tsquery escapes user input — no injection vector.
        args.append(search)
        where.append(
            f"to_tsvector('english', title || ' ' || content) "
            f"@@ plainto_tsquery('english', ${len(args)})"
        )
    if after_created_time is not None and after_id is not None:
        args.append(after_created_time)
        args.append(after_id)
        where.append(
            f"(created_time, journal_id::text) < (${len(args) - 1}, ${len(args)})"
        )
    args.append(limit)
    sql = f"""
        SELECT journal_id, entry_type, strategy_code, interval_name, instrument,
               title, content, structured_data, status,
               iteration_id_refs, related_journal_ids,
               created_time, created_by, updated_time, updated_by
        FROM research_journal
        WHERE {' AND '.join(where)}
        ORDER BY created_time DESC, journal_id::text DESC
        LIMIT ${len(args)}
    """
    rows = await conn.fetch(sql, *args)
    return [dict(r) for r in rows]


async def insert_journal(
    conn: asyncpg.Connection,
    *,
    entry_type: str,
    title: str,
    content: str,
    strategy_code: str | None,
    interval_name: str | None,
    instrument: str | None,
    structured_data: dict[str, Any],
    status: str,
    iteration_id_refs: list[UUID] | None,
    related_journal_ids: list[UUID] | None,
    created_by: str,
) -> dict[str, Any]:
    # research_journal.journal_id is UUID NOT NULL with no DEFAULT (V10);
    # gen_random_uuid() in-SQL keeps parity with insert_queue.
    row = await conn.fetchrow(
        """
        INSERT INTO research_journal (
            journal_id,
            entry_type, strategy_code, interval_name, instrument,
            title, content, structured_data, status,
            iteration_id_refs, related_journal_ids,
            created_time, created_by, updated_time, updated_by
        ) VALUES (
            gen_random_uuid(),
            $1, $2, $3, $4,
            $5, $6, $7, $8,
            $9, $10,
            NOW(), $11, NOW(), $11
        )
        RETURNING journal_id, entry_type, strategy_code, interval_name, instrument,
                  title, content, structured_data, status,
                  iteration_id_refs, related_journal_ids,
                  created_time, created_by, updated_time, updated_by
        """,
        entry_type,
        strategy_code,
        interval_name,
        instrument,
        title,
        content,
        structured_data,
        status,
        iteration_id_refs,
        related_journal_ids,
        created_by,
    )
    return dict(row)


async def get_journal(conn: asyncpg.Connection, journal_id: UUID) -> dict[str, Any] | None:
    row = await conn.fetchrow(
        """
        SELECT journal_id, entry_type, strategy_code, interval_name, instrument,
               title, content, structured_data, status,
               iteration_id_refs, related_journal_ids,
               created_time, created_by, updated_time, updated_by
        FROM research_journal
        WHERE journal_id = $1
        """,
        journal_id,
    )
    return dict(row) if row else None
