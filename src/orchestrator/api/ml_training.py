"""``/ml/training-runs`` — researcher-driven ML training.

Phase A of the researcher-drives-ML lift (2026-05-19).

  * ``POST /ml/training-runs`` — start a blackheart-train subprocess
    with the requested spec + flags. Inserts a PENDING row in
    ``ml_training_runs`` (V99), spawns the subprocess via
    :mod:`services.ml_training`, returns the row id + status immediately.
    Honors ``Idempotency-Key`` so a retried POST returns the same
    training_run_id and does NOT spawn a duplicate subprocess.
    Enforces ``ml_training_daily_cap`` (default 4/agent/24h); FAILED
    rows don't count toward the cap.

  * ``GET /ml/training-runs/{id}`` — poll for status. Returns the row
    verbatim. Researcher polls this until status reaches a terminal
    value ({COMPLETED, FAILED, TIMED_OUT, ORPHANED}).

  * ``GET /ml/training-runs`` — recent runs for the calling agent.
    Resume protocol: on session start, the researcher reads this to
    pick up any RUNNING/PENDING work and resume polling rather than
    starting a duplicate.
"""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

import asyncpg
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field, model_validator

from ..errors import OrchestratorError
from ..repo import ml_training_runs as repo
from ..services.idempotency import cache_response, replay_cached_response
from ..services.ml_training import (
    build_cli_args,
    enforce_daily_cap,
    start_training_run,
)
from .deps import get_agent_name, get_db_conn

router = APIRouter(tags=["ml-training"])


# ── Request models ──────────────────────────────────────────────────


class TrainingRunBody(BaseModel):
    """blackheart-train CLI flags, plus orchestration metadata.

    ``spec_name`` MUST be a valid ModelSpec name (validated by the
    subprocess; the orchestrator does not maintain its own spec
    registry). Invalid names fail the run with a FAILED row showing
    the CLI's argparse error in stderr_tail."""

    spec_name: str = Field(..., min_length=1, max_length=120)
    walk_forward: bool = True
    gauntlet: bool = True
    folds: int | None = Field(None, ge=1, le=20)
    bayesian: bool = False
    do_register: bool = True
    allow_leakage: bool = False
    no_write: bool = False
    notes: str | None = Field(None, max_length=500)
    motivating_hypothesis_id: UUID | None = None

    @model_validator(mode="after")
    def _no_register_without_write(self) -> "TrainingRunBody":
        if self.do_register and self.no_write:
            raise ValueError(
                "do_register=true and no_write=true are incompatible: "
                "registration requires an artifact to point at"
            )
        if self.gauntlet and not self.walk_forward:
            raise ValueError(
                "gauntlet=true requires walk_forward=true (gauntlet "
                "gates 2-5 read the walk_forward block)"
            )
        return self


# ── Endpoints ───────────────────────────────────────────────────────


@router.post("/ml/training-runs", status_code=202)
async def post_training_run(
    body: TrainingRunBody,
    request: Request,
    conn: asyncpg.Connection = Depends(get_db_conn),
    agent: str = Depends(get_agent_name),
) -> dict[str, Any]:
    """Start a training run. Returns 202 with the training_run_id and
    initial status. The subprocess runs out-of-band; poll
    GET /ml/training-runs/{id} for completion."""
    cached, idempotency_key = await replay_cached_response(
        request, agent, "ml-training-runs"
    )
    if cached is not None:
        return cached

    await enforce_daily_cap(
        conn,
        settings=request.app.state.settings,
        agent_name=agent,
    )

    cli_args = build_cli_args(
        spec_name=body.spec_name,
        walk_forward=body.walk_forward,
        gauntlet=body.gauntlet,
        folds=body.folds,
        bayesian=body.bayesian,
        do_register=body.do_register,
        allow_leakage=body.allow_leakage,
        no_write=body.no_write,
    )
    training_run_id = await repo.insert_pending(
        conn,
        agent_name=agent,
        spec_name=body.spec_name,
        hyperparams={
            "walk_forward": body.walk_forward,
            "gauntlet": body.gauntlet,
            "folds": body.folds,
            "bayesian": body.bayesian,
            "do_register": body.do_register,
            "allow_leakage": body.allow_leakage,
            "no_write": body.no_write,
        },
        cli_args=cli_args,
        notes=body.notes,
        motivating_hypothesis_id=body.motivating_hypothesis_id,
        created_by=agent,
    )

    start_training_run(
        settings=request.app.state.settings,
        db_pool=request.app.state.db,
        training_run_id=training_run_id,
        cli_args=cli_args,
        agent_name=agent,
    )

    response = {
        "training_run_id": str(training_run_id),
        "status": "PENDING",
        "spec_name": body.spec_name,
        "cli_args": cli_args,
        "poll_url": f"/ml/training-runs/{training_run_id}",
    }
    await cache_response(
        request, agent, "ml-training-runs", idempotency_key, response
    )
    return response


@router.get("/ml/training-runs/{training_run_id}")
async def get_training_run(
    training_run_id: UUID,
    conn: asyncpg.Connection = Depends(get_db_conn),
    _: str = Depends(get_agent_name),
) -> dict[str, Any]:
    """Fetch a single training run. 404 if not found."""
    row = await repo.get_by_id(conn, training_run_id=training_run_id)
    if row is None:
        raise OrchestratorError(
            status_code=404,
            error_code="training_run_not_found",
            message=f"ml_training_runs.training_run_id={training_run_id} not found",
            retryable=False,
        )
    return _serialize(row)


@router.get("/ml/model-specs")
async def list_model_specs(
    request: Request,
    _: str = Depends(get_agent_name),
) -> dict[str, Any]:
    """Phase D — researcher discovery. Returns the operator-curated
    list of blackheart-train model specs that can be invoked via
    POST /ml/training-runs.

    The list mirrors :mod:`blackheart_train.specs` but is stored
    in orchestrator settings (``available_model_specs``) to avoid an
    expensive subprocess import at request time. Operator keeps it in
    sync when adding/removing specs.

    Read-only, no DB, no auth except the standard ``X-Orch-Token``."""
    settings = request.app.state.settings
    return {
        "model_specs": list(settings.available_model_specs),
        "note": (
            "spec names mirror blackheart-train/src/blackheart_train/specs.py "
            "_spec_choices(). Pass any of these as POST /ml/training-runs body "
            "spec_name. The subprocess validates the name; the orchestrator "
            "does not — invalid names produce a FAILED row with the CLI's "
            "argparse error in stderr_tail."
        ),
    }


@router.get("/ml/training-runs")
async def list_training_runs(
    conn: asyncpg.Connection = Depends(get_db_conn),
    agent: str = Depends(get_agent_name),
    limit: int = 50,
) -> dict[str, Any]:
    """Recent training runs for the calling agent. Used by the
    researcher's resume protocol on session start."""
    if not 1 <= limit <= 200:
        raise OrchestratorError(
            status_code=422,
            error_code="validation_failed",
            message="limit must be in [1, 200]",
            retryable=False,
        )
    rows = await repo.list_by_agent(conn, agent_name=agent, limit=limit)
    return {
        "agent_name": agent,
        "training_runs": [_serialize_summary(r) for r in rows],
    }


# ── Serialization ───────────────────────────────────────────────────


def _serialize(row: dict[str, Any]) -> dict[str, Any]:
    """Full-row response. UUIDs / datetimes go to strings; JSONB cols
    stay as dicts (TypedJSONResponse handles datetime, but UUIDs need
    explicit str() to keep clients on the simple-types path)."""
    return {
        "training_run_id": str(row["training_run_id"]),
        "agent_name": row["agent_name"],
        "spec_name": row["spec_name"],
        "hyperparams": row["hyperparams"],
        "cli_args": row["cli_args"],
        "status": row["status"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "exit_code": row["exit_code"],
        "duration_seconds": row["duration_seconds"],
        "pid": row["pid"],
        "stdout_tail": row["stdout_tail"],
        "stderr_tail": row["stderr_tail"],
        "experiment_run_id": (
            str(row["experiment_run_id"]) if row["experiment_run_id"] else None
        ),
        "content_sha256": row["content_sha256"],
        "model_id": str(row["model_id"]) if row["model_id"] else None,
        "error_envelope": row["error_envelope"],
        "notes": row["notes"],
        "motivating_hypothesis_id": (
            str(row["motivating_hypothesis_id"])
            if row["motivating_hypothesis_id"]
            else None
        ),
        "created_time": row["created_time"],
        "updated_time": row["updated_time"],
    }


def _serialize_summary(row: dict[str, Any]) -> dict[str, Any]:
    """Slimmer summary for list responses."""
    return {
        "training_run_id": str(row["training_run_id"]),
        "spec_name": row["spec_name"],
        "status": row["status"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "exit_code": row["exit_code"],
        "model_id": str(row["model_id"]) if row["model_id"] else None,
        "content_sha256": row["content_sha256"],
        "notes": row["notes"],
        "created_time": row["created_time"],
    }
