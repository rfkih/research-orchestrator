"""asyncpg pool wrapper.

Raw SQL on purpose — quant-grade typing means we read columns we declare,
not whatever an ORM happens to hydrate. The pool is opened once at startup
via the lifespan hook and closed on shutdown.

``health_probe`` runs ``SELECT 1`` against a connection from the pool. Used
by ``/readyz`` and by the startup self-check that refuses to mark the
service ready until the DB answers.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from uuid import UUID

import asyncpg

from ..config import Settings

if TYPE_CHECKING:
    from asyncpg.pool import Pool


def _json_default(obj: Any) -> Any:
    # asyncpg's JSON codec hands dicts straight to json.dumps; UUID,
    # Decimal, datetime, and date routinely show up nested inside metrics
    # / params snapshots and would otherwise raise TypeError mid-INSERT.
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, default=_json_default)


async def _init_connection(conn: asyncpg.Connection) -> None:
    # Decode jsonb/json into Python dicts/lists at the asyncpg layer so
    # repos return native types, not raw strings. Encoder is symmetric.
    await conn.set_type_codec(
        "jsonb",
        encoder=_json_dumps,
        decoder=json.loads,
        schema="pg_catalog",
        format="text",
    )
    await conn.set_type_codec(
        "json",
        encoder=_json_dumps,
        decoder=json.loads,
        schema="pg_catalog",
        format="text",
    )


class Database:
    """Thin pool holder. The ``acquire`` context manager is the API."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pool: Pool | None = None

    async def open(self) -> None:
        if self._pool is not None:
            return
        self._pool = await asyncpg.create_pool(
            dsn=self._settings.db_dsn.get_secret_value(),
            min_size=self._settings.db_pool_min,
            max_size=self._settings.db_pool_max,
            init=_init_connection,
            # Statement cache off: we may run as ``blackheart_research`` against
            # a schema migrated under ``blackheart_trading``. asyncpg's prepared-
            # statement cache breaks across role/search-path edges.
            statement_cache_size=0,
        )

    async def close(self) -> None:
        if self._pool is None:
            return
        await self._pool.close()
        self._pool = None

    @property
    def pool(self) -> Pool:
        if self._pool is None:
            raise RuntimeError("Database pool is not open. Call open() first.")
        return self._pool

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[asyncpg.Connection]:
        async with self.pool.acquire() as conn:
            yield conn

    async def health_probe(self) -> bool:
        """Returns True iff a fresh connection answers ``SELECT 1``."""
        try:
            async with self.acquire() as conn:
                value = await conn.fetchval("SELECT 1")
                return value == 1
        except Exception:
            return False
