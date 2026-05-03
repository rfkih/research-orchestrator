"""JSON response that handles asyncpg's native Python types.

Starlette's default ``JSONResponse`` chokes on ``Decimal``. Repo rows come
straight out of asyncpg with ``Decimal`` (numeric), ``UUID``, ``datetime``,
and ``date`` mixed in — we serialise them all here so handlers can return
``dict[str, Any]`` without pre-massaging.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any
from uuid import UUID

from starlette.responses import JSONResponse


class _Encoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, Decimal):
            # Stringify rather than float() — Decimals from asyncpg can carry
            # more precision than IEEE-754 holds, and floats lose pf=1.42357...
            return str(obj)
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, datetime | date | time):
            return obj.isoformat()
        return super().default(obj)


class TypedJSONResponse(JSONResponse):
    """Drop-in replacement for ``JSONResponse`` that knows about DB types."""

    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            cls=_Encoder,
        ).encode("utf-8")
