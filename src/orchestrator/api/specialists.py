"""``/agent-decisions/*``, ``/portfolio/*``, ``/skeptic-prescreen``.

Researcher-facing surface for the dream-team's mechanical specialists
+ the audit-row writer. Three deterministic endpoints (no LLM, free
to invoke) plus the post-hoc audit-log endpoint researcher uses after
collecting verdicts from sub-agent specialists.

  * ``POST /skeptic-prescreen`` — mechanical pattern checks. Returns
    ``recommendation`` ∈ {invoke_skeptic_agent, skip_skeptic_agent}.
  * ``POST /portfolio/correlations`` — Spearman matrix over per-
    strategy daily returns from recent backtest_run trades.
  * ``POST /portfolio/optimize`` — three weight vectors (HRP, MV,
    equal-weight) for the candidate + protected book. The
    quant-portfolio-manager sub-agent reads these to pick one.
  * ``POST /agent-decisions/log`` — append-only audit-row writer.
    Researcher posts here after each spawned specialist returns its
    verdict.

None of these require API access. All run on the orchestrator's own
DB pool + NumPy.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

import asyncpg
import numpy as np
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field, model_validator

from ..errors import OrchestratorError
from ..services.activity_logger import log_activity
from ..services.idempotency import cache_response, replay_cached_response
from ..services.portfolio import (
    _fetch_one_baseline,
    daily_returns_from_trades,
)
from ..services.paired_delta import compute_paired_delta
from ..services.specialists import (
    equal_weight,
    hrp_weights,
    log_agent_decision,
    mean_variance_weights,
    ml_prescreen,
    skeptic_prescreen,
    spearman_corr_matrix,
)
from ..services.specialists.portfolio_math import realised_vol, summarise_weights
from ..repo import trades as trades_repo
from .deps import get_agent_name, get_db_conn

router = APIRouter(tags=["specialists"])


# ── /skeptic-prescreen ──────────────────────────────────────────────


class PrescreenBody(BaseModel):
    iteration_id: UUID
    strategy_code: str = Field(..., min_length=1, max_length=60)
    motivating_hypothesis_id: UUID | None = None


@router.post("/skeptic-prescreen")
async def post_skeptic_prescreen(
    body: PrescreenBody,
    request: Request,
    conn: asyncpg.Connection = Depends(get_db_conn),
    agent: str = Depends(get_agent_name),
) -> dict[str, Any]:
    """Mechanical pattern checks. Idempotent at the DB level — running
    twice returns the same metrics. Idempotency-Key caches the response
    envelope across replays."""
    cached, idempotency_key = await replay_cached_response(
        request, agent, "skeptic-prescreen"
    )
    if cached is not None:
        return cached

    result = await skeptic_prescreen(
        conn,
        iteration_id=str(body.iteration_id),
        strategy_code=body.strategy_code,
        motivating_hypothesis_id=(
            str(body.motivating_hypothesis_id)
            if body.motivating_hypothesis_id
            else None
        ),
    )
    await cache_response(request, agent, "skeptic-prescreen", idempotency_key, result)
    return result


# ── /ml-prescreen (Phase C, 2026-05-19) ─────────────────────────────


class MlPrescreenBody(BaseModel):
    """Body for ``POST /ml-prescreen``. Either model_id OR
    experiment_run_id must be set; both is OK (model_id wins on resolve).

    The researcher typically passes model_id (it's the canonical handle
    from ml_training_runs.model_id after Phase A completion). When that
    column is NULL (e.g. --register failed), pass experiment_run_id
    directly from ml_training_runs.experiment_run_id."""

    model_id: UUID | None = None
    experiment_run_id: UUID | None = None

    @model_validator(mode="after")
    def _require_at_least_one_id(self) -> "MlPrescreenBody":
        # Use @model_validator so the failure surfaces as 422
        # validation_failed; model_post_init exceptions in Pydantic v2
        # are runtime errors and would leak as 500 internal_error.
        if self.model_id is None and self.experiment_run_id is None:
            raise ValueError(
                "ml-prescreen requires at least one of model_id, "
                "experiment_run_id"
            )
        return self


@router.post("/ml-prescreen")
async def post_ml_prescreen(
    body: MlPrescreenBody,
    request: Request,
    conn: asyncpg.Connection = Depends(get_db_conn),
    agent: str = Depends(get_agent_name),
) -> dict[str, Any]:
    """5 mechanical checks against a trained model artifact +
    experiment_run leakage report. Idempotent at the DB level (pure
    SELECT + compute); Idempotency-Key caches the response envelope."""
    cached, idempotency_key = await replay_cached_response(
        request, agent, "ml-prescreen"
    )
    if cached is not None:
        return cached

    result = await ml_prescreen(
        conn,
        model_id=body.model_id,
        experiment_run_id=body.experiment_run_id,
    )
    if result.get("model_id") is None and result.get("experiment_run_id") is None:
        # Neither resolver hit a row — surface 404 instead of returning a
        # silently empty checks list (which the researcher might misread
        # as PASS).
        raise OrchestratorError(
            status_code=404,
            error_code="ml_artifacts_not_found",
            message=(
                f"Neither model_registry nor experiment_run resolved for "
                f"model_id={body.model_id} experiment_run_id={body.experiment_run_id}"
            ),
            retryable=False,
        )
    await cache_response(request, agent, "ml-prescreen", idempotency_key, result)
    return result


# ── /paired-delta (Phase E, 2026-05-19) ─────────────────────────────


class PairedDeltaBody(BaseModel):
    """Body for ``POST /paired-delta``. Exactly one of queue_id /
    iteration_ids must be set. ``primary_metric`` defaults to sharpe
    because Phase 4 Stage H showed Sharpe collapse was the canonical
    "model destroyed value" signature."""

    queue_id: UUID | None = None
    iteration_ids: list[UUID] | None = Field(None, min_length=2, max_length=200)
    primary_metric: Literal[
        "sharpe", "return", "profit_factor", "win_rate", "trade_count"
    ] = "sharpe"
    treatment_key: Literal[
        "_ml_gate_enabled", "_ml_signal_name", "_ml_shadow_mode"
    ] = "_ml_gate_enabled"
    treatment_on_value: Any = True
    treatment_off_value: Any = False
    bootstrap_reps: int = Field(1000, ge=200, le=10000)
    ci_level: float = Field(0.95, gt=0.5, lt=1.0)
    rng_seed: int = Field(42, ge=0)

    @model_validator(mode="after")
    def _require_exactly_one_target(self) -> "PairedDeltaBody":
        # @model_validator surfaces as 422, not 500 — see MlPrescreenBody
        # comment for the Pydantic v2 rationale.
        if (self.queue_id is None) == (self.iteration_ids is None):
            raise ValueError(
                "paired-delta requires exactly one of queue_id, iteration_ids"
            )
        return self


@router.post("/paired-delta")
async def post_paired_delta(
    body: PairedDeltaBody,
    request: Request,
    conn: asyncpg.Connection = Depends(get_db_conn),
    agent: str = Depends(get_agent_name),
) -> dict[str, Any]:
    """Phase E — compute paired delta on an ML sweep. Returns
    {n_pairs, mean_delta, ci_low, ci_high, verdict}. The verdict
    is BINDING — the researcher's loop must reject graduation on
    NEGATIVE_DELTA."""
    cached, idempotency_key = await replay_cached_response(
        request, agent, "paired-delta"
    )
    if cached is not None:
        return cached

    result = await compute_paired_delta(
        conn,
        queue_id=body.queue_id,
        iteration_ids=body.iteration_ids,
        primary_metric=body.primary_metric,
        treatment_key=body.treatment_key,
        treatment_on_value=body.treatment_on_value,
        treatment_off_value=body.treatment_off_value,
        bootstrap_reps=body.bootstrap_reps,
        ci_level=body.ci_level,
        rng_seed=body.rng_seed,
    )
    await cache_response(
        request, agent, "paired-delta", idempotency_key, result
    )
    return result


# ── /portfolio/correlations ─────────────────────────────────────────


class CorrelationsBody(BaseModel):
    strategy_codes: list[str] = Field(..., min_length=1, max_length=20)
    min_overlap_days: int = Field(5, ge=2, le=60)


@router.post("/portfolio/correlations")
async def post_portfolio_correlations(
    body: CorrelationsBody,
    request: Request,
    conn: asyncpg.Connection = Depends(get_db_conn),
    agent: str = Depends(get_agent_name),
) -> dict[str, Any]:
    """Pairwise Spearman correlation matrix over the recent daily-
    return series of each strategy_code. Reuses the existing
    ``_fetch_one_baseline`` helper that the V11 portfolio gate already
    consumes — same data, exposed at the agent surface."""
    series_by_code: dict[str, dict] = {}
    missing: list[str] = []
    for code in body.strategy_codes:
        series = await _fetch_one_baseline(conn, code)
        if series is None or len(series) < body.min_overlap_days:
            missing.append(code)
            continue
        series_by_code[code] = series
    if len(series_by_code) < 2:
        raise OrchestratorError(
            status_code=409,
            error_code="insufficient_series",
            message=(
                f"At least 2 strategies need recent daily-return series "
                f"with >= {body.min_overlap_days} days; "
                f"got {len(series_by_code)} ({sorted(series_by_code)})."
            ),
            retryable=False,
            details={"missing": missing, "available": sorted(series_by_code)},
        )
    codes, matrix = spearman_corr_matrix(
        series_by_code, min_overlap=body.min_overlap_days
    )
    # Convert NaN → None for JSON serialisation.
    matrix_jsonable = [
        [None if np.isnan(v) else round(float(v), 4) for v in row]
        for row in matrix
    ]
    # max_abs_corr per code excluding self.
    max_abs_per_code: dict[str, float | None] = {}
    for i, code in enumerate(codes):
        others = [
            abs(matrix[i, j]) for j in range(len(codes))
            if j != i and not np.isnan(matrix[i, j])
        ]
        max_abs_per_code[code] = round(max(others), 4) if others else None

    return {
        "codes": codes,
        "matrix": matrix_jsonable,
        "max_abs_per_code": max_abs_per_code,
        "min_overlap_days": body.min_overlap_days,
        "missing_strategies": missing,
        "computed_at": datetime.utcnow().isoformat() + "Z",
    }


# ── /portfolio/optimize ─────────────────────────────────────────────


class OptimizeBody(BaseModel):
    """Inputs to all three optimisers.

    ``mu_by_code`` is optional. When None, mean-variance returns the
    global minimum-variance solution (no return forecasts assumed).
    When provided, it's the expected daily return per strategy — the
    agent typically sources these from recent backtest geometric_return
    / 252 or similar.
    """

    strategy_codes: list[str] = Field(..., min_length=1, max_length=20)
    min_overlap_days: int = Field(5, ge=2, le=60)
    mu_by_code: dict[str, float] | None = None
    risk_aversion: float = Field(1.0, gt=0.0, le=100.0)


@router.post("/portfolio/optimize")
async def post_portfolio_optimize(
    body: OptimizeBody,
    request: Request,
    conn: asyncpg.Connection = Depends(get_db_conn),
    agent: str = Depends(get_agent_name),
) -> dict[str, Any]:
    """Compute equal-weight, HRP, and mean-variance allocations. Return
    all three side-by-side; the agent picks based on context (book
    state, drawdown tolerance, candidate's marginal Sharpe estimate)."""
    series_by_code: dict[str, dict] = {}
    missing: list[str] = []
    for code in body.strategy_codes:
        series = await _fetch_one_baseline(conn, code)
        if series is None or len(series) < body.min_overlap_days:
            missing.append(code)
            continue
        series_by_code[code] = series
    if not series_by_code:
        raise OrchestratorError(
            status_code=409,
            error_code="no_series_available",
            message=(
                f"None of {body.strategy_codes} have a recent daily-return "
                f"series with >= {body.min_overlap_days} days."
            ),
            retryable=False,
            details={"missing": missing},
        )
    codes, corr = spearman_corr_matrix(
        series_by_code, min_overlap=body.min_overlap_days
    )
    vol_arr = np.asarray(
        [realised_vol(series_by_code[c]) or 1e-8 for c in codes],
        dtype=np.float64,
    )
    # Build covariance from correlation + vol for mean-variance.
    # NaN correlation → 0 for the optimiser (same imputation HRP uses).
    corr_filled = np.where(np.isnan(corr), 0.0, corr)
    np.fill_diagonal(corr_filled, 1.0)
    sigma = np.outer(vol_arr, vol_arr) * corr_filled

    mu_arr: np.ndarray | None = None
    if body.mu_by_code:
        mu_arr = np.asarray(
            [body.mu_by_code.get(c, 0.0) for c in codes], dtype=np.float64
        )

    weights_eq = equal_weight(codes)
    weights_hrp = hrp_weights(codes, corr_filled, vol_arr)
    weights_mv = mean_variance_weights(
        codes, sigma, mu=mu_arr, risk_aversion=body.risk_aversion
    )

    return {
        "codes": codes,
        "min_overlap_days": body.min_overlap_days,
        "missing_strategies": missing,
        "equal_weight": {
            "weights": {c: round(w, 4) for c, w in weights_eq.items()},
            "summary": summarise_weights(weights_eq),
        },
        "hrp": {
            "weights": {c: round(w, 4) for c, w in weights_hrp.items()},
            "summary": summarise_weights(weights_hrp),
        },
        "mean_variance": {
            "weights": {c: round(w, 4) for c, w in weights_mv.items()},
            "summary": summarise_weights(weights_mv),
            "mu_provided": body.mu_by_code is not None,
            "risk_aversion": body.risk_aversion,
        },
        "computed_at": datetime.utcnow().isoformat() + "Z",
    }


# ── /agent-decisions/log ────────────────────────────────────────────


class AgentDecisionBody(BaseModel):
    """Audit-row body. Researcher posts here after each spawned
    specialist returns its verdict. The orchestrator does NOT validate
    the verdict content — that's the agent's responsibility. We pin
    only the structural shape + the status enum."""

    specialist: str = Field(..., min_length=1, max_length=60)
    endpoint: str = Field(..., min_length=1, max_length=120)
    model_name: str | None = Field(None, max_length=120)
    request_payload: dict[str, Any]
    response_payload: dict[str, Any] | None = None
    verdict: str | None = Field(None, max_length=40)
    latency_ms: int | None = Field(None, ge=0)
    status: Literal["ok", "parse_error", "api_error", "timeout", "refused"] = "ok"
    error_envelope: dict[str, Any] | None = None
    motivating_iteration_id: UUID | None = None
    target_id: str | None = Field(None, max_length=200)


@router.post("/agent-decisions/log", status_code=201)
async def post_agent_decision(
    body: AgentDecisionBody,
    request: Request,
    conn: asyncpg.Connection = Depends(get_db_conn),
    agent: str = Depends(get_agent_name),
) -> dict[str, Any]:
    """Write one audit row. Honors Idempotency-Key so the researcher
    can retry safely after a network blip without writing two rows
    for the same spawned-specialist verdict."""
    cached, idempotency_key = await replay_cached_response(
        request, agent, "agent-decisions-log"
    )
    if cached is not None:
        return cached

    decision_id = await log_agent_decision(
        conn,
        specialist=body.specialist,
        endpoint=body.endpoint,
        model_name=body.model_name,
        request_payload=body.request_payload,
        response_payload=body.response_payload,
        verdict=body.verdict,
        latency_ms=body.latency_ms,
        status=body.status,
        error_envelope=body.error_envelope,
        motivating_iteration_id=body.motivating_iteration_id,
        target_id=body.target_id,
        agent_name=agent,
    )

    try:
        await log_activity(
            conn,
            session_id=request.state.session_id,
            agent_name=agent,
            activity_type="AGENT_DECISION_LOGGED",
            title=f"{body.specialist} verdict={body.verdict or '<no-verdict>'}",
            strategy_code=None,
            details={
                "specialist": body.specialist,
                "endpoint": body.endpoint,
                "verdict": body.verdict,
                "status": body.status,
                "decision_id": str(decision_id),
                "motivating_iteration_id": (
                    str(body.motivating_iteration_id)
                    if body.motivating_iteration_id
                    else None
                ),
                "target_id": body.target_id,
            },
            redis_client=request.app.state.redis,
        )
    except Exception:  # noqa: BLE001 — activity log is best-effort
        pass

    response = {
        "decision_id": str(decision_id),
        "specialist": body.specialist,
        "endpoint": body.endpoint,
        "status": body.status,
        "verdict": body.verdict,
    }
    await cache_response(
        request, agent, "agent-decisions-log", idempotency_key, response
    )
    return response
