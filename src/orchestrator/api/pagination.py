"""Cursor pagination helpers.

Offset pagination is a footgun for an agent — page numbers shift under
concurrent inserts, and the agent has no way to detect it. Cursor
pagination keeps "give me the next batch" referentially stable: the cursor
encodes the last seen ``(created_time, id)``, and the server only returns
rows strictly older than that.

The cursor is opaque to callers: base64(JSON). Don't introspect it client-
side; the shape may change.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from ..errors import OrchestratorError


class Page(BaseModel):
    """Generic page shape. Items typed by the calling router."""

    items: list[Any]
    next_cursor: str | None = Field(
        None, description="Opaque. Pass back as ?cursor= to fetch the next page."
    )
    next_actions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Agent-readable suggestions: paginate, refine filters, etc.",
    )


def encode_cursor(created_time: datetime, row_id: UUID | str) -> str:
    payload = {"t": created_time.isoformat(), "id": str(row_id)}
    return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")


def decode_cursor(cursor: str) -> tuple[datetime, str]:
    """Returns (created_time, id). Raises ``OrchestratorError`` on garbage."""
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode()).decode())
        return datetime.fromisoformat(payload["t"]), str(payload["id"])
    except Exception as e:
        raise OrchestratorError(
            status_code=400,
            error_code="bad_cursor",
            message="Cursor is not a value previously returned by this API.",
            retryable=False,
            hint="Drop the cursor parameter to start over from the newest row.",
            details={"reason": type(e).__name__},
        ) from e
