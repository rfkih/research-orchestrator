"""Capacity sweep — scale a strategy across initial-capital values to
measure edge degradation.

Built 2026-05-20 under autonomous-mode operator delegation. Addresses
Scorecard #3 (capacity modeling) per memory ``project_quant_grade_scorecard``.

Mechanism: given a graduation-candidate iteration's params + a list of
notional scales (e.g. [100, 1_000, 10_000, 100_000, 1_000_000]), run a
fresh backtest at each scale and compare per-scale metrics (PF, Sharpe,
ann_geom) against the base scale. The edge-decay verdict is one of:

  LINEAR_SCALABLE      — PF degrades < 20% across all scales relative
                         to the base. Strategy scales cleanly; the
                         max viable capital is the largest tested.
  GRADUAL_DECAY        — PF degrades 20-40% across the range; max
                         viable capital is the largest scale with
                         PF >= 1.2 (above the noise floor for V11).
  SHARP_CLIFF          — PF degrades > 40% at some specific scale;
                         the prior scale is the cap.
  INVERTED             — even at the smallest scale, PF < 1.0; the
                         strategy doesn't have an edge to scale.

The verdict is intentionally coarse — operator-quant-capacity-judge can
read the per-scale series and apply a finer threshold. This service's
job is to produce comparable backtests across capital tiers, not to
make the graduation call.

Per-scale backtest reuses ``tick._build_submit_payload``-style shape
with only ``initialCapital`` varied. All other inputs (account_strategy
id, instrument, interval, window, overrides) stay constant — so the
verdict isolates the capital lever.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Literal
from uuid import UUID, uuid4

from ..clients.jvm import JvmClient
from ..config import Settings
from ..errors import OrchestratorError
from ..infra.db import Database
from ..logging import get_logger
from ..repo.capacity_write import insert_capacity_sweep_result
from .sweep import build_ml_override_maps, split_ml_overrides
from .tick import _fetch_run_metrics, _poll_backtest_status, _resolve_account_strategy

# Operator's live-trading floor (USD). Capacity verdict is REJECT below this.
DEFAULT_FLOOR_CAPITAL_USD: float = 10_000.0

log = get_logger(__name__)


EdgeDecayVerdict = Literal[
    "LINEAR_SCALABLE", "GRADUAL_DECAY", "SHARP_CLIFF", "INVERTED",
    "INSUFFICIENT_DATA",
]


# Verdict thresholds — operator can override per-call via the request
# but the defaults below are the binding ones for the auto-classifier.
PF_NOISE_FLOOR: float = 1.2          # PF below this = no useful edge
DEGRADATION_PCT_GENTLE: float = 20.0  # < 20% decay = LINEAR_SCALABLE
DEGRADATION_PCT_GRADUAL: float = 40.0  # 20-40% = GRADUAL_DECAY

# Default per-fill slippage rate (entry-side, fractional units). 0.0005 =
# 5 bps — conservative for crypto spot at the smallest tested notional;
# the JVM applies this haircut to every entry fill, so $1M-tier capacity
# can actually decay against $100-tier. Operator override via the API
# request body (``slippage_rate``). NULL on the JVM side disables impact
# entirely, which is why the prior default produced bit-identical metrics
# across all five tiers.
DEFAULT_SLIPPAGE_RATE: float = 0.0005


def _classify_verdict(
    per_scale: list[dict[str, Any]],
    *,
    noise_floor: float = PF_NOISE_FLOOR,
    gentle_pct: float = DEGRADATION_PCT_GENTLE,
    gradual_pct: float = DEGRADATION_PCT_GRADUAL,
) -> tuple[EdgeDecayVerdict, float | None]:
    """Reduce a list of per-scale results to a single verdict + the
    largest capital where edge holds (None if INVERTED)."""
    valid = [
        s for s in per_scale
        if "metrics" in s and s["metrics"].get("pf") is not None
    ]
    if not valid:
        return "INSUFFICIENT_DATA", None
    base = valid[0]
    base_pf = base["metrics"]["pf"]
    if base_pf is None or base_pf < 1.0:
        return "INVERTED", None
    max_viable: float | None = base["scale_usd"]
    max_decay_pct = 0.0
    for s in valid[1:]:
        pf = s["metrics"]["pf"]
        if pf is None:
            continue
        decay_pct = ((base_pf - pf) / base_pf) * 100.0
        max_decay_pct = max(max_decay_pct, decay_pct)
        if pf >= noise_floor:
            max_viable = s["scale_usd"]
    if max_decay_pct < gentle_pct:
        return "LINEAR_SCALABLE", max_viable
    if max_decay_pct < gradual_pct:
        return "GRADUAL_DECAY", max_viable
    return "SHARP_CLIFF", max_viable


def _annualize_geom(geom_pct: float | None, days: float) -> float | None:
    """Annualize a cumulative geometric return % over ``days`` (365-day year).

    NB: uses CALENDAR days, whereas the V60 gate's
    annualized_geometric_return_pct_at_alloc_90 uses deployed-time (sum of
    holding periods). The edge_at_floor/edge_at_max absolute values therefore
    differ from the gate metric for idle-heavy strategies — but the whole sweep
    shares ONE factor (same window), so the leaderboard's edge_at_max/edge_at_floor
    RATIO is unaffected. Returns None on bad inputs."""
    if geom_pct is None or days is None or days <= 0:
        return None
    multiplier = 1.0 + geom_pct / 100.0
    if multiplier <= 0:  # ruin — annualized return is -100%
        return -100.0
    annual_mult = multiplier ** (365.0 / days)
    return (annual_mult - 1.0) * 100.0


def _provisional_judge_verdict(
    verdict: EdgeDecayVerdict, max_viable: float | None, floor: float | None
) -> str:
    """Derive a PROCEED/CONCERN/REJECT verdict for the capacity_sweep_result row
    from the edge-decay classification. Mirrors the quant-capacity-judge criteria;
    the full agent (Phase 3) may overwrite. judge_verdict is NOT NULL on the table.

      REJECT  — no viable capital, or max-viable below the operator's live floor.
      CONCERN — viable only near the floor (< 5×) or a sharp cliff in the curve.
      PROCEED — max-viable ≥ 5× floor with non-cliff decay (room to scale safely).
    """
    if max_viable is None or (floor is not None and max_viable < floor):
        return "REJECT"
    if verdict in ("SHARP_CLIFF", "INVERTED") or (floor is not None and max_viable < floor * 5):
        return "CONCERN"
    return "PROCEED"


def _edge_at(per_scale: list[dict[str, Any]], target_scale: float | None) -> float | None:
    """Annualized edge (ann_geom_pct) at the scale whose capital == target.
    Exact match — used for max_viable, which is always one of the tested scales."""
    if target_scale is None:
        return None
    for s in per_scale:
        if s.get("scale_usd") == target_scale and "metrics" in s:
            return s["metrics"].get("ann_geom_pct")
    return None


def _edge_at_floor(per_scale: list[dict[str, Any]], floor: float | None) -> float | None:
    """Annualized edge at the operator's floor capital. The floor need NOT be a
    tested scale, so pick the largest tested scale ≤ floor (the most capital you'd
    actually deploy at the floor); fall back to the smallest tested scale if every
    tier is above the floor. Exact-match would silently return None for an
    off-ladder floor, breaking the leaderboard's edge_at_max/edge_at_floor ratio."""
    if floor is None:
        return None
    valid = [
        s for s in per_scale
        if "metrics" in s and s["metrics"].get("ann_geom_pct") is not None
        and s.get("scale_usd") is not None
    ]
    if not valid:
        return None
    at_or_below = [s for s in valid if s["scale_usd"] <= floor]
    chosen = (
        max(at_or_below, key=lambda s: s["scale_usd"])
        if at_or_below
        else min(valid, key=lambda s: s["scale_usd"])
    )
    return chosen["metrics"].get("ann_geom_pct")


def _build_payload(
    *,
    account_strategy_id: str,
    strategy_code: str,
    asset: str,
    interval_name: str,
    start_time: datetime,
    end_time: datetime,
    initial_capital: float,
    overrides: dict[str, Any] | None,
    allow_long: bool = True,
    allow_short: bool = True,
    slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
) -> dict[str, Any]:
    """Same shape as walk_forward._build_payload, parameterized on
    ``initialCapital`` — the lever the capacity sweep varies. All other
    fields match the production submit payload.

    ML sentinel keys (``_ml_gate_enabled``, ``_ml_signal_name``,
    ``_ml_shadow_mode``) inside ``overrides`` are routed to the V100
    JVM columns (``strategyMl*Overrides``) via the shared sweep helper,
    NOT into ``strategyParamOverrides`` — the JVM ignores them there.
    """
    regular_overrides, ml_overrides = split_ml_overrides(overrides or {})
    ml_payload = build_ml_override_maps(
        strategy_code=strategy_code, ml_overrides=ml_overrides
    )
    payload: dict[str, Any] = {
        "accountStrategyId": account_strategy_id,
        "strategyCode": strategy_code,
        "asset": asset,
        "interval": interval_name,
        "startTime": start_time.isoformat(),
        "endTime": end_time.isoformat(),
        "initialCapital": float(initial_capital),
        "riskPerTradePct": 2.0,
        "feeRate": 0.00075,
        "slippageRate": float(slippage_rate),
        # THE capacity lever (V143): size-dependent market impact. Flat
        # slippageRate is identical across all tiers (a fixed %), so WITHOUT this
        # every notional scale returns bit-identical metrics and the sweep is
        # meaningless. With it on, the JVM charges
        # coeff*sqrt(order_notional/bar_volume) (round-trip) using the
        # tick-calibrated per-symbol coeff (omitting marketImpactCoeff ⇒ JVM
        # resolves it), so larger capital → larger orders → more impact → edge
        # decays. This is what produces the AUM-scaling curve.
        "marketImpactEnabled": True,
        "minNotional": 5,
        "minQty": 0.000001,
        "qtyStep": 0.000001,
        "maxOpenPositions": 1,
        "allowLong": allow_long,
        "allowShort": allow_short,
        "strategyParamOverrides": {strategy_code: regular_overrides},
        # JVM regex is ^(USER|RESEARCHER)$ (see
        # BacktestRunRequest.triggeredBy). Using RESEARCHER matches the
        # walk_forward convention — keeps capacity-sweep runs visible to
        # the same multi-tenant read queries that filter on RESEARCHER.
        # A future ``CAPACITY_SWEEP`` enum value would require a JVM
        # regex update + DB CHECK extension; not worth the migration.
        "triggeredBy": "RESEARCHER",
        **ml_payload,
    }
    return payload


class CapacitySweepResult:
    def __init__(
        self,
        *,
        capacity_sweep_id: UUID,
        edge_decay_verdict: EdgeDecayVerdict,
        max_viable_capital_usd: float | None,
        per_scale: list[dict[str, Any]],
        motivating_iteration_id: UUID | None,
        judge_verdict: str,
        floor_capital_usd: float | None,
        edge_at_floor_pct: float | None,
        edge_at_max_pct: float | None,
        result_row_id: str | None,
    ) -> None:
        self.capacity_sweep_id = capacity_sweep_id
        self.edge_decay_verdict = edge_decay_verdict
        self.max_viable_capital_usd = max_viable_capital_usd
        self.per_scale = per_scale
        self.motivating_iteration_id = motivating_iteration_id
        self.judge_verdict = judge_verdict
        self.floor_capital_usd = floor_capital_usd
        self.edge_at_floor_pct = edge_at_floor_pct
        self.edge_at_max_pct = edge_at_max_pct
        self.result_row_id = result_row_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "capacity_sweep_id": str(self.capacity_sweep_id),
            "edge_decay_verdict": self.edge_decay_verdict,
            "judge_verdict": self.judge_verdict,
            "max_viable_capital_usd": self.max_viable_capital_usd,
            "floor_capital_usd": self.floor_capital_usd,
            "edge_at_floor_pct": self.edge_at_floor_pct,
            "edge_at_max_pct": self.edge_at_max_pct,
            "result_row_id": self.result_row_id,
            "motivating_iteration_id": (
                str(self.motivating_iteration_id)
                if self.motivating_iteration_id else None
            ),
            "per_scale": self.per_scale,
        }


async def run_capacity_sweep(
    *,
    db: Database,
    jvm: JvmClient,
    settings: Settings,
    agent_name: str,  # noqa: ARG001
    strategy_code: str,
    interval_name: str,
    instrument: str,
    start_time: datetime,
    end_time: datetime,
    notional_scales: list[float],
    overrides: dict[str, Any] | None = None,
    motivating_iteration_id: UUID | None = None,
    slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
    floor_capital_usd: float = DEFAULT_FLOOR_CAPITAL_USD,
) -> CapacitySweepResult:
    """Run one backtest per notional scale; aggregate verdict; persist a
    capacity_sweep_result row (provisional judge verdict + AUM-curve edges)."""
    if not notional_scales:
        raise OrchestratorError(
            status_code=400,
            error_code="capacity_no_scales",
            message="notional_scales must contain at least one value.",
            retryable=False,
        )
    if any(s <= 0 for s in notional_scales):
        raise OrchestratorError(
            status_code=400,
            error_code="capacity_invalid_scale",
            message="All notional_scales must be positive.",
            retryable=False,
        )

    async with db.acquire() as conn:
        as_row = await _resolve_account_strategy(
            conn, strategy_code, settings.research_account_id
        )
    if not as_row:
        raise OrchestratorError(
            status_code=412,
            error_code="account_strategy_missing",
            message=f"No account_strategy row for strategy_code={strategy_code}.",
            retryable=False,
            hint="Seed an account_strategy row before requesting capacity sweep.",
        )
    as_id = as_row["account_strategy_id"]

    # Same window for every scale → one annualization factor for the whole sweep.
    window_days = max(1.0, (end_time - start_time).total_seconds() / 86400.0)

    per_scale: list[dict[str, Any]] = []
    sweep_id = uuid4()

    for idx, scale in enumerate(notional_scales, start=1):
        log.info(
            "capacity.scale_start",
            strategy_code=strategy_code,
            sweep_id=str(sweep_id),
            scale=idx, n_scales=len(notional_scales),
            initial_capital=scale,
        )
        payload = _build_payload(
            account_strategy_id=as_id,
            strategy_code=strategy_code,
            asset=instrument,
            interval_name=interval_name,
            start_time=start_time,
            end_time=end_time,
            initial_capital=scale,
            overrides=overrides,
            allow_long=as_row["allow_long"],
            allow_short=as_row["allow_short"],
            slippage_rate=slippage_rate,
        )
        try:
            run_id = await jvm.submit_backtest(payload)
        except OrchestratorError as e:
            per_scale.append({
                "scale_index": idx,
                "scale_usd": scale,
                "run_id": None,
                "error": e.envelope.error_code,
            })
            continue
        final_status = await _poll_backtest_status(db, run_id)
        if final_status != "COMPLETED":
            per_scale.append({
                "scale_index": idx,
                "scale_usd": scale,
                "run_id": run_id,
                "error": f"status={final_status}",
            })
            continue
        async with db.acquire() as conn:
            metrics = await _fetch_run_metrics(conn, run_id)
        per_scale.append({
            "scale_index": idx,
            "scale_usd": scale,
            "run_id": run_id,
            "metrics": {
                "pf": _safe_float(metrics.get("profit_factor")),
                "return_pct": _safe_float(metrics.get("return_pct")),
                "sharpe": _safe_float(metrics.get("sharpe_ratio")),
                "wr": _safe_float(metrics.get("win_rate")),
                "trades": metrics.get("total_trades"),
                "max_dd_pct": _safe_float(metrics.get("max_drawdown_pct")),
                "net_profit": _safe_float(metrics.get("net_profit")),
                # _fetch_run_metrics returns the RAW geometric_return_pct_at_alloc_90;
                # annualize it here (the prior 'annualized_…' key never existed → always None).
                "ann_geom_pct": _annualize_geom(
                    _safe_float(metrics.get("geometric_return_pct_at_alloc_90")),
                    window_days,
                ),
            },
        })

    verdict, max_viable = _classify_verdict(per_scale)
    judge_verdict = _provisional_judge_verdict(verdict, max_viable, floor_capital_usd)
    edge_at_floor = _edge_at_floor(per_scale, floor_capital_usd)
    edge_at_max = _edge_at(per_scale, max_viable)

    # Persist the summary row for the leaderboard CapacityFactor (V135).
    result_row_id: str | None = None
    try:
        async with db.acquire() as conn:
            result_row_id = await insert_capacity_sweep_result(
                conn,
                symbol=instrument,
                strategy_code=strategy_code,
                interval_name=interval_name,
                judge_verdict=judge_verdict,
                floor_capital_usd=floor_capital_usd,
                max_viable_capital_usd=max_viable,
                edge_at_floor_pct=edge_at_floor,
                edge_at_max_pct=edge_at_max,
            )
    except Exception:  # noqa: BLE001 — the sweep result is still returned even if persist fails
        log.exception("capacity.persist_failed", sweep_id=str(sweep_id), strategy_code=strategy_code)

    return CapacitySweepResult(
        capacity_sweep_id=sweep_id,
        edge_decay_verdict=verdict,
        max_viable_capital_usd=max_viable,
        per_scale=per_scale,
        motivating_iteration_id=motivating_iteration_id,
        judge_verdict=judge_verdict,
        floor_capital_usd=floor_capital_usd,
        edge_at_floor_pct=edge_at_floor,
        edge_at_max_pct=edge_at_max,
        result_row_id=result_row_id,
    )


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
