"""Idempotency registry — pluggable store interface.

Two implementations:

  * ``InMemoryIdempotencyStore`` — process-local LRU, fine for dev where the
    orchestrator runs as a single uvicorn worker and retries land within
    seconds.
  * ``PostgresIdempotencyStore`` — durable across restarts, backed by the
    ``idempotency_record`` table (Flyway V28). Used in prod so that an
    agent retry after an orchestrator redeploy doesn't slip past the cache
    and double-submit a queue insert / walk-forward.

Both implement the same async ``get(agent, key)`` / ``put(agent, key, value)``
contract. Lifespan picks one based on ``Settings.profile``: ``prod`` ⇒
Postgres; everything else ⇒ in-memory.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from ..infra.db import Database

# Tuned for one agent firing daily ticks plus ad-hoc operator curl. 2048
# entries is generous; entries TTL out so memory stays bounded.
_MAX_ENTRIES = 2048
_TTL_SECONDS = 24 * 60 * 60


class IdempotencyStore(Protocol):
    async def get(self, agent: str, key: str) -> Any | None: ...
    async def put(self, agent: str, key: str, value: Any) -> None: ...


class InMemoryIdempotencyStore:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._cache: OrderedDict[tuple[str, str], tuple[float, Any]] = OrderedDict()

    async def get(self, agent: str, key: str) -> Any | None:
        async with self._lock:
            entry = self._cache.get((agent, key))
            if entry is None:
                return None
            inserted_at, value = entry
            if time.monotonic() - inserted_at > _TTL_SECONDS:
                self._cache.pop((agent, key), None)
                return None
            self._cache.move_to_end((agent, key))
            return value

    async def put(self, agent: str, key: str, value: Any) -> None:
        async with self._lock:
            self._cache[(agent, key)] = (time.monotonic(), value)
            self._cache.move_to_end((agent, key))
            while len(self._cache) > _MAX_ENTRIES:
                self._cache.popitem(last=False)


class PostgresIdempotencyStore:
    """Reads & writes ``idempotency_record``. Each call holds a single
    short-lived connection — the volume is tiny (one row per non-GET call)."""

    def __init__(self, db: Database, ttl_seconds: int = _TTL_SECONDS) -> None:
        self._db = db
        self._ttl = ttl_seconds

    async def get(self, agent: str, key: str) -> Any | None:
        async with self._db.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT response_jsonb, expires_at
                FROM idempotency_record
                WHERE agent_name = $1 AND key = $2
                """,
                agent,
                key,
            )
            if row is None:
                return None
            if row["expires_at"] <= datetime.now(timezone.utc):
                # Lazily evict so a stale replay doesn't poison the next call.
                await conn.execute(
                    "DELETE FROM idempotency_record WHERE agent_name = $1 AND key = $2",
                    agent,
                    key,
                )
                return None
            payload = row["response_jsonb"]
            # asyncpg returns JSONB as str when the codec isn't registered;
            # callers expect a dict, so normalise.
            if isinstance(payload, str):
                return json.loads(payload)
            return payload

    async def put(self, agent: str, key: str, value: Any) -> None:
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=self._ttl)
        # asyncpg's JSONB codec accepts dicts directly when registered; the
        # codebase registers it in infra/db.py, so passing the dict is fine.
        # We serialise to text as a defensive fallback for fixtures that
        # bypass the codec.
        payload = json.dumps(value, default=str)
        async with self._db.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO idempotency_record (agent_name, key, response_jsonb, expires_at)
                VALUES ($1, $2, $3::jsonb, $4)
                ON CONFLICT (agent_name, key) DO UPDATE
                SET response_jsonb = EXCLUDED.response_jsonb,
                    expires_at     = EXCLUDED.expires_at,
                    created_time   = NOW()
                """,
                agent,
                key,
                payload,
                expires_at,
            )
