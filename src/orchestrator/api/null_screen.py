"""``POST /null-screen`` — archetype edge sniff before sweeping.

See ``services/null_screen.py`` for the decision rule and rationale. The
endpoint is synchronous (mirrors ``/tick`` and ``/walk-forward``); a
default-K=8 screen takes ~30-60 minutes on the JVM. Returns ``EDGE_PRESENT``
/ ``NO_EDGE_DETECTED`` / ``INCONCLUSIVE`` / ``INSUFFICIENT_DATA`` along with
the full PF distribution.
"""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID  # noqa: F401 — for type-doc consistency

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field, model_validator

from ..errors import NextAction, OrchestratorError
from ..services import null_screen as svc
from .deps import get_agent_name

router = APIRouter(prefix="/null-screen", tags=["null-screen"])

_VALID_INTERVALS = {"5m", "15m", "1h", "4h"}


class NullScreenParam(BaseModel):
    """One parameter range. Either ``values`` (choice) or ``low``+``high``
    (numeric) must be set, mirroring the Optuna ``suggest_*`` distinction."""

    name: str = Field(..., min_length=1, max_length=80)
    type: Literal["float", "int", "choice"] = "float"
    low: str | None = Field(None, max_length=40)
    high: str | None = Field(None, max_length=40)
    values: list[Any] | None = Field(None, max_length=50)

    @model_validator(mode="after")
    def _check_shape(self) -> "NullScreenParam":
        if self.type == "choice":
            if not self.values:
                raise ValueError(f"choice param {self.name!r} requires values")
        else:
            if self.low is None or self.high is None:
                raise ValueError(
                    f"numeric param {self.name!r} requires low and high"
                )
        return self


class NullScreenRequest(BaseModel):
    strategy_code: str = Field(..., min_length=1, max_length=60)
    interval_name: str = Field(..., description="5m|15m|1h|4h")
    instrument: str = Field("BTCUSDT", min_length=1, max_length=30)
    param_ranges: list[NullScreenParam] = Field(..., min_length=1, max_length=10)
    n_draws: int = Field(svc.DEFAULT_N_DRAWS, ge=4, le=30)
    seed: int = Field(42, ge=0)


@router.post("", status_code=200)
async def run_null_screen(
    body: NullScreenRequest,
    request: Request,
    agent: str = Depends(get_agent_name),
) -> dict[str, Any]:
    if body.interval_name not in _VALID_INTERVALS:
        raise OrchestratorError(
            status_code=400,
            error_code="bad_interval",
            message=f"interval_name must be one of {sorted(_VALID_INTERVALS)}.",
            retryable=False,
        )

    idempotency_key = getattr(request.state, "idempotency_key", None)
    store = request.app.state.idempotency
    if idempotency_key:
        cached = await store.get(agent, f"null-screen:{idempotency_key}")
        if cached is not None:
            return cached

    db = request.app.state.db
    jvm = request.app.state.jvm
    settings = request.app.state.settings

    param_ranges = [p.model_dump(exclude_none=True) for p in body.param_ranges]

    try:
        result = await svc.run_null_screen(
            db=db,
            jvm=jvm,
            settings=settings,
            agent_name=agent,
            strategy_code=body.strategy_code,
            interval_name=body.interval_name,
            instrument=body.instrument,
            param_ranges=param_ranges,
            n_draws=body.n_draws,
            seed=body.seed,
        )
    except OrchestratorError:
        raise
    except Exception as e:  # noqa: BLE001
        raise OrchestratorError(
            status_code=502,
            error_code="null_screen_failed",
            message=f"Null screen crashed: {type(e).__name__}: {e}",
            retryable=False,
            next_action=NextAction(kind="contact_human"),
        ) from e

    if idempotency_key:
        await store.put(agent, f"null-screen:{idempotency_key}", result)
    return result
