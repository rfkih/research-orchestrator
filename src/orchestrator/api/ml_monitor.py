"""``GET /ml/monitor`` — cross-signal health rollup for the frontend.

One row per non-retired ``signal_definition`` bound to ≥1 strategy with
``ml_regime_gate_enabled = TRUE``. Health colour is derived server-side
so the dashboard pill and the monitor table agree without each client
re-deriving from raw counts.

The endpoint is read-only, idempotent, and cheap — a single SQL round
trip aggregated in :func:`orchestrator.repo.ml_monitor.fetch_monitor_rows`.
The handler only adds derived fields (expected fire cadence, health
verdict + reason) and renders the response.

Thresholds are pinned at module scope so accidental drift is loud;
:func:`derive_health` is unit-tested in ``tests/test_ml_monitor.py``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ..repo.ml_monitor import fetch_monitor_rows
from .deps import get_agent_name, get_db_conn


router = APIRouter(tags=["ml-monitor"])


# ── Health-verdict thresholds ──────────────────────────────────────────────
# Pinned constants. The auto-checklist + dashboard both read these via the
# /agent/playbook constants block — changing here without updating the
# pinned tests is a deliberate red-flag.
COVERAGE_RED_RATIO: float    = 0.50
COVERAGE_AMBER_RATIO: float  = 0.90
LAST_FIRE_RED_MULT: float    = 3.0   # > 3× expected interval = red
LAST_FIRE_AMBER_MULT: float  = 1.5
MODULATOR_AUC_AMBER: float   = 0.52  # AUC below this on a modulator is amber

# Map interval names → expected fire cadence in seconds. Mirrored from
# the trading-JVM `IntervalName` enum. Unknown interval = no cadence-based
# health gate (defaults to amber via the explicit `unknown_interval` reason).
_INTERVAL_SECONDS: dict[str, int] = {
    "1m":  60,
    "5m":  300,
    "15m": 900,
    "30m": 1_800,
    "1h":  3_600,
    "4h":  14_400,
    "1d":  86_400,
}


class _MonitorBoundStrategy(BaseModel):
    strategyCode: str


class MlMonitorRow(BaseModel):
    signalId: str
    signalName: str
    modelSpecName: str       # `{family}/{purpose}` — the human-readable model handle
    status: str              # active | shadow
    symbol: str | None
    intervalName: str | None
    boundStrategyCodes: list[str]
    health: str              # green | amber | red
    healthReason: str | None
    walkforwardAuc: float | None
    coverage7dRatio: float | None
    fires24h: int
    lastFireTs: str | None          # bar (open-label) timestamp — how recent the signal's market data is
    lastProducedAt: str | None      # wall-clock write time — pipeline liveness ("last written")
    freshnessScore: float           # 1.0=fresh (<1 interval), 0.5=(2 intervals), 0.0=(>3 intervals)
    inferenceLatencyMs: int | None  # wall-clock latency: produced_at - last_fire_ts
    featureAgeHours: float | None   # hours since feature_values max(ts)
    signalWriteLatencyMs: int | None  # signal write latency: produced_at - MAX(feature_values.ts)


class MlMonitorResponse(BaseModel):
    rows: list[MlMonitorRow]
    generatedAt: str


def _coverage_ratio(*, fires_7d: int, interval_seconds: int | None) -> float | None:
    """Ratio of observed fires to expected fires in the trailing 7d window.

    Returns ``None`` when the interval is unknown — the dashboard renders
    a muted coverage bar in that case rather than guessing."""
    if interval_seconds is None or interval_seconds <= 0:
        return None
    expected = (7 * 86_400) // interval_seconds
    if expected == 0:
        return None
    return min(1.0, fires_7d / expected)


def _last_fire_age_seconds(last_fire_ts: datetime | None) -> int | None:
    if last_fire_ts is None:
        return None
    now = datetime.now(timezone.utc)
    # asyncpg returns aware UTC timestamps for TIMESTAMPTZ.
    return int((now - last_fire_ts).total_seconds())


def derive_health(
    *,
    coverage_ratio: float | None,
    last_fire_age_seconds: int | None,
    expected_fire_seconds: int | None,
    walkforward_auc: float | None,
    purpose: str | None,
) -> tuple[str, str | None]:
    """Return ``(health, reason)``. Order of evaluation matters — red
    short-circuits amber, amber short-circuits green."""
    if coverage_ratio is not None and coverage_ratio < COVERAGE_RED_RATIO:
        return "red", f"coverage_7d {coverage_ratio:.0%} < {COVERAGE_RED_RATIO:.0%}"
    if (
        last_fire_age_seconds is not None
        and expected_fire_seconds is not None
        and last_fire_age_seconds > LAST_FIRE_RED_MULT * expected_fire_seconds
    ):
        return "red", (
            f"last_fire {last_fire_age_seconds}s > "
            f"{LAST_FIRE_RED_MULT:g}× expected ({expected_fire_seconds}s)"
        )
    if coverage_ratio is not None and coverage_ratio < COVERAGE_AMBER_RATIO:
        return "amber", f"coverage_7d {coverage_ratio:.0%} < {COVERAGE_AMBER_RATIO:.0%}"
    if (
        last_fire_age_seconds is not None
        and expected_fire_seconds is not None
        and last_fire_age_seconds > LAST_FIRE_AMBER_MULT * expected_fire_seconds
    ):
        return "amber", (
            f"last_fire {last_fire_age_seconds}s > "
            f"{LAST_FIRE_AMBER_MULT:g}× expected ({expected_fire_seconds}s)"
        )
    if (
        purpose == "modulator"
        and walkforward_auc is not None
        and walkforward_auc < MODULATOR_AUC_AMBER
    ):
        return "amber", f"modulator_auc {walkforward_auc:.3f} < {MODULATOR_AUC_AMBER}"
    if expected_fire_seconds is None:
        return "amber", "unknown_interval"
    return "green", None


def _freshness_score(
    feature_age_hours: float | None,
    interval_seconds: int | None,
) -> float:
    """Calculate freshness score based on feature age relative to interval.

    Returns:
        1.0 if feature age < 1 interval old
        0.5 if feature age between 1-2 intervals old
        0.0 if feature age > 2 intervals old

    Returns 1.0 (fully fresh) if interval or age is unknown."""
    if interval_seconds is None or interval_seconds <= 0:
        return 1.0
    if feature_age_hours is None:
        return 1.0

    # Convert interval to hours
    interval_hours = interval_seconds / 3600.0

    # Score decay: 1.0 → 0.5 → 0.0 as age increases
    if feature_age_hours < interval_hours:
        return 1.0
    elif feature_age_hours < 2 * interval_hours:
        return 0.5
    else:
        return 0.0


def _extract_auc(metrics: dict[str, Any] | None) -> float | None:
    """Pull a walkforward AUC from the model_registry.metrics JSONB.

    The blackheart-train writer stores under either ``walkforward.auc_mean``
    or top-level ``walkforward_auc_mean`` depending on spec version. Try
    both; return ``None`` when neither is present."""
    if not metrics:
        return None
    wf = metrics.get("walkforward")
    if isinstance(wf, dict):
        v = wf.get("auc_mean")
        if isinstance(v, (int, float)):
            return float(v)
    v = metrics.get("walkforward_auc_mean")
    if isinstance(v, (int, float)):
        return float(v)
    return None


@router.get("/ml/monitor", response_model=MlMonitorResponse)
async def get_ml_monitor(
    conn=Depends(get_db_conn),
    agent: str = Depends(get_agent_name),  # noqa: ARG001 — required for auth dep chain
) -> MlMonitorResponse:
    raw = await fetch_monitor_rows(conn)
    rows: list[MlMonitorRow] = []
    for r in raw:
        interval_name: str | None = r["model_interval"]
        expected_fire_seconds = _INTERVAL_SECONDS.get(interval_name) if interval_name else None
        coverage = _coverage_ratio(
            fires_7d=int(r["fires_7d"]),
            interval_seconds=expected_fire_seconds,
        )
        # Pipeline liveness = time since the signal was WRITTEN (produced_at), NOT the bar
        # timestamp (last_fire_ts). signal_history.ts is the open-keyed bar the prediction is
        # FOR, so a freshly-written 1h signal is already ~1 interval "old" by ts — which trips
        # derive_health's 1.5×/3× cadence gates every bar on a perfectly healthy pipeline.
        last_age = _last_fire_age_seconds(r.get("last_produced_at"))
        wf_auc = _extract_auc(r["model_metrics"])
        purpose: str | None = r["model_purpose"]
        health, reason = derive_health(
            coverage_ratio=coverage,
            last_fire_age_seconds=last_age,
            expected_fire_seconds=expected_fire_seconds,
            walkforward_auc=wf_auc,
            purpose=purpose,
        )

        # Calculate freshness and latency metrics
        feature_age_hours: float | None = None
        inference_latency_ms: int | None = None
        signal_write_latency_ms: int | None = None

        # Feature age: hours since feature_values max(ts)
        latest_feature_ts = r.get("latest_feature_ts")
        if latest_feature_ts is not None:
            now = datetime.now(timezone.utc)
            age_seconds = (now - latest_feature_ts).total_seconds()
            feature_age_hours = age_seconds / 3600.0

        # Inference latency: produced_at - last_fire_ts (wall-clock latency)
        if r.get("last_produced_at") is not None and r.get("last_fire_ts") is not None:
            latency_seconds = (r["last_produced_at"] - r["last_fire_ts"]).total_seconds()
            inference_latency_ms = max(0, int(latency_seconds * 1000))

        # Signal write latency: produced_at - latest_feature_ts
        if r.get("last_produced_at") is not None and latest_feature_ts is not None:
            write_latency_seconds = (r["last_produced_at"] - latest_feature_ts).total_seconds()
            signal_write_latency_ms = max(0, int(write_latency_seconds * 1000))

        # Calculate freshness score
        freshness = _freshness_score(feature_age_hours, expected_fire_seconds)

        rows.append(MlMonitorRow(
            signalId=str(r["signal_id"]),
            signalName=r["signal_name"],
            modelSpecName=f"{r['model_family']}/{r['model_purpose']}"
                if r.get("model_family") and r.get("model_purpose")
                else (r.get("model_family") or r.get("model_purpose") or "unknown"),
            status=r["status"],
            symbol=r["model_symbol"],
            intervalName=interval_name,
            boundStrategyCodes=list(r["bound_strategy_codes"] or []),
            health=health,
            healthReason=reason,
            walkforwardAuc=wf_auc,
            coverage7dRatio=coverage,
            fires24h=int(r["fires_24h"]),
            lastFireTs=r["last_fire_ts"].isoformat() if r["last_fire_ts"] else None,
            lastProducedAt=r["last_produced_at"].isoformat() if r.get("last_produced_at") else None,
            freshnessScore=freshness,
            inferenceLatencyMs=inference_latency_ms,
            featureAgeHours=feature_age_hours,
            signalWriteLatencyMs=signal_write_latency_ms,
        ))
    # Sort red → amber → green, then by fires_24h desc — same rule the
    # dashboard strip uses client-side, applied server-side so all callers
    # see the same ordering.
    health_rank = {"red": 0, "amber": 1, "green": 2}
    rows.sort(key=lambda row: (health_rank.get(row.health, 3), -row.fires24h))
    return MlMonitorResponse(
        rows=rows,
        generatedAt=datetime.now(timezone.utc).isoformat(),
    )
