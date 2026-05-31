"""``/signals`` read + status-transition surface backing the frontend
``/ml/signals`` and ``/ml/signals/[id]`` pages (Phase 1).

Endpoints:
* ``GET  /signals``                       — paginated list with filters
* ``GET  /signals/{id}``                  — single signal + derived health
* ``GET  /signals/{id}/firings``          — paginated signal_history rows
* ``GET  /signals/{id}/daily-counts``     — 30-day daily aggregate (for the timeline chart)
* ``POST /signals/{id}/promote``          — shadow ↔ active ↔ retired

The health field on the detail endpoint is computed via the same
:func:`orchestrator.api.ml_monitor.derive_health` function the monitor
rollup uses — single source of truth so the dashboard pill and the
detail page never disagree.

Status transitions are validated server-side via the
:data:`_SIGNAL_STATUS_TRANSITIONS` map. The transition rules are pinned
in :func:`tests.test_signals_transitions` so accidental drift is loud.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field

from ..errors import OrchestratorError
from ..repo import signals as signals_repo
from ..services.idempotency import cache_response, replay_cached_response
from .deps import get_agent_name, get_db_conn
from .ml_monitor import _INTERVAL_SECONDS, _coverage_ratio, _extract_auc, derive_health


router = APIRouter(prefix="/signals", tags=["signals"])


# ── allowed status transitions ────────────────────────────────────────────
# Mirror of the V66 ``chk_signal_definition_status`` enum + the operator's
# promotion lifecycle. ``retired`` is terminal — once a signal is retired,
# any rebirth happens by registering a new signal definition.
_SIGNAL_STATUS_TRANSITIONS: dict[str, frozenset[str]] = {
    "shadow":  frozenset({"active", "retired"}),
    "active":  frozenset({"shadow", "retired"}),
    "retired": frozenset(),
}


# ── pydantic shapes ───────────────────────────────────────────────────────
class SignalRow(BaseModel):
    """List-row shape. Joins model_registry for the symbol/interval the
    frontend filters and displays in the cell."""
    signalId: str
    signalName: str
    modelId: str
    modelSpecName: str          # ``{family}/{purpose}``
    status: str                 # active | shadow | retired
    symbol: str | None
    intervalName: str | None
    boundStrategyCodes: list[str]
    createdAt: str
    gauntletVerdict: str | None  # PASS | FAIL | None (pre-gauntlet models)


class SignalListResponse(BaseModel):
    signals: list[SignalRow]
    total: int
    limit: int
    offset: int


class SignalHealth(BaseModel):
    health: str                 # green | amber | red
    healthReason: str | None
    lastFireTs: str | None
    lastProducedAt: str | None  # wall-clock write time — pipeline liveness
    lastFireAgeSeconds: int | None
    expectedFireSeconds: int | None
    featureAgeHours: float | None   # age of latest feature_values row
    coverage7dRatio: float | None
    fires24h: int
    fires7d: int
    walkforwardAuc: float | None


class WalkForwardSummary(BaseModel):
    nFolds: int
    primaryMetric: str
    primaryMean: float
    primaryMedian: float
    primaryStd: float


class ModelInfo(BaseModel):
    """Full model metadata exposed on the signal detail page."""
    algorithm: str                          # e.g. "LightGBM"
    objective: str                          # "binary" | "multiclass" | "regression"
    predicts: str                           # human-readable prediction description
    labelFeature: str                       # e.g. "label_regime_risk_on_24h"
    trainStart: str | None
    trainEnd: str | None
    nTrainRows: int | None
    nValRows: int | None
    trainedAt: str | None
    features: list[str]
    featureImportance: dict[str, float]     # name → gain score (sorted desc)
    auc: float | None                       # final hold-out AUC
    accuracy: float | None                  # final hold-out accuracy
    logLoss: float | None
    adversarialAuc: float | None            # train/val covariate shift score (1.0 = bad)
    walkForward: WalkForwardSummary | None
    gauntletVerdict: str | None             # PASS | FAIL
    gauntletGates: list[str]               # gate names that were checked
    leakageVerdict: str | None             # PASS | FAIL
    leakageMaxPearson: float | None
    hyperparams: dict[str, Any]


class SignalDetail(SignalRow):
    description: str | None
    updatedAt: str
    health: SignalHealth
    model: ModelInfo | None


class FiringRow(BaseModel):
    signalId: str
    symbol: str
    ts: str
    value: float
    confidence: float | None
    producedAt: str
    source: str                 # stream | catchup_scan | historical_replay
    meta: dict[str, Any] | None


class FiringsResponse(BaseModel):
    firings: list[FiringRow]
    total: int
    limit: int
    offset: int


class DailyCount(BaseModel):
    date: str                   # YYYY-MM-DD
    fireCount: int


class DailyCountsResponse(BaseModel):
    days: list[DailyCount]


class PromoteRequest(BaseModel):
    to: str = Field(..., description="Target status — one of active|shadow|retired.")
    reason: str = Field(
        ..., min_length=1, max_length=500,
        description="Operator-supplied reason; lands in updated_by + audit log.",
    )


# ── helpers ───────────────────────────────────────────────────────────────
def _row_to_signal_row(r: dict[str, Any]) -> SignalRow:
    family = r.get("model_family")
    purpose = r.get("model_purpose")
    if family and purpose:
        spec = f"{family}/{purpose}"
    else:
        spec = family or purpose or "unknown"
    return SignalRow(
        signalId=str(r["signal_id"]),
        signalName=r["name"],
        modelId=str(r["model_id"]),
        modelSpecName=spec,
        status=r["status"],
        symbol=r.get("model_symbol"),
        intervalName=r.get("model_interval"),
        boundStrategyCodes=list(r.get("bound_strategy_codes") or []),
        createdAt=r["created_time"].isoformat() if r.get("created_time") else "",
        gauntletVerdict=r.get("gauntlet_verdict"),
    )


def _last_fire_age_seconds(last_fire_ts: datetime | None) -> int | None:
    if last_fire_ts is None:
        return None
    return int((datetime.now(timezone.utc) - last_fire_ts).total_seconds())


_LABEL_DESCRIPTION: dict[str, str] = {
    "label_regime_risk_on_24h":   "Probability that the next 24 hours will have a positive Sharpe ratio (risk-on regime). Value close to 1.0 = model expects bullish conditions; close to 0.0 = bearish.",
    "label_regime_risk_on_48h":   "Probability that the next 48 hours will have a positive Sharpe ratio.",
    "label_return_7d":            "Probability of positive return over the next 7 days.",
    "label_meanrev_24h":          "Probability of mean-reversion within 24 hours.",
    "label_triple_barrier":       "Triple-barrier labelling: 1 = price hits take-profit before stop-loss within the horizon.",
}

_ALGO_NAME: dict[str, str] = {
    "lightgbm_modulator": "LightGBM (gradient boosted trees)",
    "xgboost":            "XGBoost",
    "random_forest":      "Random Forest",
    "logistic":           "Logistic Regression",
}


def _build_model_info(row: dict[str, Any], metrics: dict[str, Any]) -> ModelInfo | None:
    """Build a ModelInfo from model_registry row + parsed metrics JSON."""
    feature_set = row.get("model_feature_set") or {}
    hyperparams = row.get("model_hyperparams") or {}
    if not feature_set and not metrics:
        return None

    label_feature: str = feature_set.get("label_feature", "")
    features: list[str] = feature_set.get("names", [])
    family: str = row.get("model_family") or ""

    wf_raw = metrics.get("walk_forward") or {}
    wf = WalkForwardSummary(
        nFolds=wf_raw.get("n_folds", 0),
        primaryMetric=wf_raw.get("primary_metric", "auc"),
        primaryMean=wf_raw.get("primary_mean", 0.0),
        primaryMedian=wf_raw.get("primary_median", 0.0),
        primaryStd=wf_raw.get("primary_std", 0.0),
    ) if wf_raw else None

    fi_raw: dict[str, float] = metrics.get("feature_importance_gain") or {}
    fi_sorted = dict(sorted(fi_raw.items(), key=lambda kv: -kv[1]))

    leakage = metrics.get("leakage") or {}
    gauntlet = metrics.get("gauntlet") or {}

    return ModelInfo(
        algorithm=_ALGO_NAME.get(family, family or "Unknown"),
        objective=metrics.get("eval_kind", "binary").replace("walk_forward_last_fold", "binary"),
        predicts=_LABEL_DESCRIPTION.get(label_feature, label_feature),
        labelFeature=label_feature,
        trainStart=str(feature_set.get("train_start", "")) or None,
        trainEnd=str(feature_set.get("train_end", "")) or None,
        nTrainRows=metrics.get("n_train_rows"),
        nValRows=metrics.get("n_val_rows"),
        trainedAt=metrics.get("trained_at"),
        features=features,
        featureImportance=fi_sorted,
        auc=metrics.get("auc"),
        accuracy=metrics.get("accuracy"),
        logLoss=metrics.get("log_loss"),
        adversarialAuc=metrics.get("adversarial_auc"),
        walkForward=wf,
        gauntletVerdict=gauntlet.get("verdict") or row.get("gauntlet_verdict"),
        gauntletGates=gauntlet.get("gates", []),
        leakageVerdict=leakage.get("verdict"),
        leakageMaxPearson=leakage.get("max_pearson_abs"),
        hyperparams=hyperparams,
    )


# ── endpoints ─────────────────────────────────────────────────────────────
@router.get("", response_model=SignalListResponse)
async def list_signals(
    status: str | None = Query(None, description="active|shadow|retired"),
    symbol: str | None = Query(None, description="Filter by model.symbol"),
    interval_name: str | None = Query(
        None, alias="intervalName",
        description="Filter by model.interval (5m|15m|1h|4h|1d|…).",
    ),
    q: str | None = Query(None, description="Substring match on signal_definition.name."),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    conn=Depends(get_db_conn),
    agent: str = Depends(get_agent_name),  # noqa: ARG001
) -> SignalListResponse:
    if status and status not in {"active", "shadow", "retired"}:
        raise OrchestratorError(
            status_code=400,
            error_code="invalid_signal_status",
            message=f"status must be one of active|shadow|retired (got '{status}').",
            retryable=False,
        )
    rows = await signals_repo.list_signals(
        conn,
        status=status, symbol=symbol, interval_name=interval_name, q=q,
        limit=limit, offset=offset,
    )
    total = await signals_repo.count_signals(
        conn, status=status, symbol=symbol, interval_name=interval_name, q=q,
    )
    return SignalListResponse(
        signals=[_row_to_signal_row(r) for r in rows],
        total=total, limit=limit, offset=offset,
    )


@router.get("/{signal_id}", response_model=SignalDetail)
async def get_signal(
    signal_id: UUID,
    conn=Depends(get_db_conn),
    agent: str = Depends(get_agent_name),  # noqa: ARG001
) -> SignalDetail:
    row = await signals_repo.get_signal(conn, signal_id)
    if row is None:
        raise OrchestratorError(
            status_code=404,
            error_code="signal_not_found",
            message=f"No signal_definition with signal_id={signal_id}.",
            retryable=False,
        )
    symbol: str | None = row.get("model_symbol")
    counters = await signals_repo.get_signal_health_counters(conn, signal_id, symbol=symbol)
    interval_name: str | None = row.get("model_interval")
    expected = _INTERVAL_SECONDS.get(interval_name) if interval_name else None
    coverage = _coverage_ratio(
        fires_7d=counters["fires_7d"], interval_seconds=expected,
    )
    # Liveness age = time since the signal was WRITTEN (produced_at), not the bar ts
    # (open-keyed → reads ~1 interval stale every bar on a healthy pipeline). Mirrors
    # the same fix in ml_monitor.get_ml_monitor; both feed the interval-aware derive_health.
    last_age = _last_fire_age_seconds(counters["last_produced_at"])
    metrics: dict[str, Any] = row.get("model_metrics") or {}
    wf_auc = _extract_auc(metrics)
    health, reason = derive_health(
        coverage_ratio=coverage,
        last_fire_age_seconds=last_age,
        expected_fire_seconds=expected,
        walkforward_auc=wf_auc,
        purpose=row.get("model_purpose"),
    )
    base = _row_to_signal_row(row)
    return SignalDetail(
        **base.model_dump(),
        description=row.get("description"),
        updatedAt=row["updated_time"].isoformat() if row.get("updated_time") else "",
        health=SignalHealth(
            health=health,
            healthReason=reason,
            lastFireTs=counters["last_fire_ts"].isoformat() if counters["last_fire_ts"] else None,
            lastProducedAt=counters["last_produced_at"].isoformat() if counters["last_produced_at"] else None,
            lastFireAgeSeconds=last_age,
            expectedFireSeconds=expected,
            featureAgeHours=counters.get("feature_age_hours"),
            coverage7dRatio=coverage,
            fires24h=counters["fires_24h"],
            fires7d=counters["fires_7d"],
            walkforwardAuc=wf_auc,
        ),
        model=_build_model_info(row, metrics),
    )


@router.get("/{signal_id}/firings", response_model=FiringsResponse)
async def list_firings(
    signal_id: UUID,
    since: datetime | None = Query(None),
    until: datetime | None = Query(None),
    source: str | None = Query(None, description="stream|catchup_scan|historical_replay"),
    limit: int = Query(200, ge=1, le=1_000),
    offset: int = Query(0, ge=0),
    conn=Depends(get_db_conn),
    agent: str = Depends(get_agent_name),  # noqa: ARG001
) -> FiringsResponse:
    if source and source not in {"stream", "catchup_scan", "historical_replay"}:
        raise OrchestratorError(
            status_code=400,
            error_code="invalid_signal_source",
            message=(
                "source must be one of stream|catchup_scan|historical_replay "
                f"(got '{source}')."
            ),
            retryable=False,
        )
    rows = await signals_repo.list_firings(
        conn, signal_id=signal_id, since=since, until=until, source=source,
        limit=limit, offset=offset,
    )
    total = await signals_repo.count_firings(
        conn, signal_id=signal_id, since=since, until=until, source=source,
    )
    firings = [
        FiringRow(
            signalId=str(r["signal_id"]),
            symbol=r["symbol"],
            ts=r["ts"].isoformat(),
            value=float(r["value"]),
            confidence=float(r["confidence"]) if r.get("confidence") is not None else None,
            producedAt=r["produced_at"].isoformat(),
            source=r["source"],
            meta=r.get("meta"),
        )
        for r in rows
    ]
    return FiringsResponse(
        firings=firings, total=total, limit=limit, offset=offset,
    )


@router.get("/{signal_id}/daily-counts", response_model=DailyCountsResponse)
async def get_daily_counts(
    signal_id: UUID,
    days: int = Query(30, ge=1, le=180),
    conn=Depends(get_db_conn),
    agent: str = Depends(get_agent_name),  # noqa: ARG001
) -> DailyCountsResponse:
    rows = await signals_repo.daily_counts(conn, signal_id, days=days)
    return DailyCountsResponse(
        days=[
            DailyCount(date=r["date"].isoformat(), fireCount=int(r["fire_count"]))
            for r in rows
        ]
    )


@router.post("/{signal_id}/promote")
async def promote_signal(
    request: Request,
    signal_id: UUID,
    body: PromoteRequest,
    agent: str = Depends(get_agent_name),
    conn=Depends(get_db_conn),  # noqa: ARG001 — ensures DB reachable
) -> dict[str, Any]:
    cached, idempotency_key = await replay_cached_response(request, agent, "promote-signal")
    if cached is not None:
        return cached

    if body.to not in {"active", "shadow", "retired"}:
        raise OrchestratorError(
            status_code=400,
            error_code="invalid_signal_status",
            message=f"to must be one of active|shadow|retired (got '{body.to}').",
            retryable=False,
        )

    async with request.app.state.db.acquire() as c:
        current = await signals_repo.get_signal(c, signal_id)
        if current is None:
            raise OrchestratorError(
                status_code=404,
                error_code="signal_not_found",
                message=f"No signal_definition with signal_id={signal_id}.",
                retryable=False,
            )
        current_status = current["status"]
        allowed = _SIGNAL_STATUS_TRANSITIONS.get(current_status, frozenset())
        if body.to == current_status:
            # No-op promotion — return the row unchanged. Matches the
            # idempotent-by-construction shape of the rest of the API.
            updated = current
        elif body.to not in allowed:
            raise OrchestratorError(
                status_code=409,
                error_code="signal_transition_forbidden",
                message=(
                    f"Cannot transition signal from '{current_status}' to '{body.to}'. "
                    f"Allowed: {sorted(allowed) or 'none — terminal state'}."
                ),
                retryable=False,
                details={"current": current_status, "to": body.to, "allowed": sorted(allowed)},
            )
        else:
            updated = await signals_repo.update_status(
                c, signal_id=signal_id, new_status=body.to,
                updated_by=f"{agent}:{body.reason}"[:150],
            )
            if updated is None:
                raise OrchestratorError(
                    status_code=404,
                    error_code="signal_not_found",
                    message=(
                        "signal_definition row vanished between read + update — "
                        "concurrent retire? retry the read."
                    ),
                    retryable=True,
                )

    payload = {
        "signalId": str(updated["signal_id"]),
        "signalName": updated["name"],
        "status": updated["status"],
        "updatedAt": updated["updated_time"].isoformat() if updated.get("updated_time") else None,
    }
    if idempotency_key is not None:
        await cache_response(request, agent, "promote-signal", idempotency_key, payload)
    return payload
