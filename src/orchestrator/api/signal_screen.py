"""``POST /signal-screen`` -- feature-level IC screen (advisory triage).

The cheapest rung of the research cost ladder: IC screen (minutes) ->
null-screen (~1h) -> sweep (hours) -> walk-forward. Run it BEFORE
pre-registering a HYPOTHESIS on a not-yet-traded signal family (playbook
methodology entry ``ic_screen_before_hypothesis``). Read-only on
market_data / feature_store / feature_values; the only write is one
research_journal row (structured_data.kind='signal_screen').

ADVISORY ONLY -- never touches the frozen V11/V60 gate logic.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

import asyncpg
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field, field_validator

from ..errors import NextAction, OrchestratorError
from ..services import signal_screen as svc
from ..services.idempotency import cache_response, replay_cached_response
from .constants import VALID_INTERVAL_NAMES
from .deps import get_agent_name, get_db_conn

router = APIRouter(prefix="/signal-screen", tags=["signal-screen"])

# Identifier shape for the signal name -- it resolves against feature_store
# columns / feature_values names, both snake_case identifiers.
_SIGNAL_PATTERN = r"^[A-Za-z][A-Za-z0-9_]{0,119}$"


class SignalScreenRequest(BaseModel):
    signal: str = Field(
        ...,
        pattern=_SIGNAL_PATTERN,
        description=(
            "Feature name. Resolution order: feature_store column (the wide "
            "per-bar table the JVM engines read), else feature_values series "
            "(point-in-time aligned: latest value with ts <= bar start)."
        ),
    )
    instruments: list[str] = Field(..., min_length=1, max_length=20)
    interval: str = Field(..., description="5m|15m|1h|4h|1d")
    horizons_bars: list[int] = Field(
        default_factory=lambda: [1, 6, 24], min_length=1, max_length=10
    )
    transform: Literal["zscore", "rank", "raw"] = "zscore"
    start: datetime | None = None
    end: datetime | None = None

    @field_validator("instruments")
    @classmethod
    def _instruments_shape(cls, v: list[str]) -> list[str]:
        cleaned = []
        for s in v:
            s = s.strip().upper()
            if not s or len(s) > 30 or not s.isalnum():
                raise ValueError(f"instrument {s!r} is not symbol-shaped")
            cleaned.append(s)
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("instruments contains duplicates")
        return cleaned

    @field_validator("horizons_bars")
    @classmethod
    def _horizons_shape(cls, v: list[int]) -> list[int]:
        for h in v:
            if not (1 <= h <= 2000):
                raise ValueError(f"horizon {h} out of range [1, 2000]")
        if len(set(v)) != len(v):
            raise ValueError("horizons_bars contains duplicates")
        return v


@router.post("", status_code=200)
async def run_signal_screen(
    body: SignalScreenRequest,
    request: Request,
    conn: asyncpg.Connection = Depends(get_db_conn),
    agent: str = Depends(get_agent_name),
) -> dict[str, Any]:
    if body.interval not in VALID_INTERVAL_NAMES:
        raise OrchestratorError(
            status_code=400,
            error_code="bad_interval",
            message=f"interval must be one of {sorted(VALID_INTERVAL_NAMES)}.",
            retryable=False,
        )

    cached, idempotency_key = await replay_cached_response(
        request, agent, "signal-screen"
    )
    if cached is not None:
        return cached

    try:
        result = await svc.run_signal_screen(
            conn,
            agent_name=agent,
            signal=body.signal,
            instruments=body.instruments,
            interval=body.interval,
            horizons_bars=body.horizons_bars,
            transform=body.transform,
            start=body.start,
            end=body.end,
        )
    except OrchestratorError:
        raise
    except Exception as e:  # noqa: BLE001
        raise OrchestratorError(
            status_code=502,
            error_code="signal_screen_failed",
            message=f"Signal screen crashed: {type(e).__name__}: {e}",
            retryable=False,
            next_action=NextAction(kind="contact_human"),
        ) from e

    await cache_response(request, agent, "signal-screen", idempotency_key, result)
    return result
