"""Pooled-book certification — orchestrator-side pooling of coordinated
single-symbol runs (operator methodology approval 2026-07-10).

Motivation: the JVM's ``PooledIndependentBacktestCoordinatorService`` cannot
honestly host stop-based archetypes (no intra-bar listener step, allowShort
hardcoded false, single interval per run, no per-symbol ML-gate wiring — see
research_journal ``eaeb1220``). A pooled DCB run through it would certify a
DIFFERENT strategy (long-only, stop-less, ungated) than the measured sleeves.

The approved alternative certifies a multi-sleeve book with ZERO JVM changes:

  1. ``run_pooled_analyze`` — each sleeve is ONE standard single-symbol
     backtest through the fully-verified coordinator (intra-bar SL/TP,
     shorts, per-strategy ML gates all honest). The per-trade
     ``realized_pnl_amount`` series are merged orchestrator-side and scored
     with the UNCHANGED ``services/analyze.py`` primitives (bootstrap PF CI,
     per-observation Sharpe, Bailey-LdP DSR with the real cumulative trial
     count, ``statistical_verdict`` / ``decision_verdict``). The result is
     persisted as a ``research_iteration_log`` row on the BOOK strategy_code
     so the standard graduation-review machinery targets it unchanged.
  2. ``run_pooled_walk_forward`` — the standard rolling-fold plan
     (``build_folds``), one single-symbol run per fold x sleeve, per-fold
     trade series pooled, and the verdict computed by the byte-identical
     ``aggregate_folds`` + ``stability_verdict`` cutoffs (total n>=100,
     pf_mean>=1.0, pf_positive>=60%, pf_std<=1.5 -> ROBUST). Persisted as a
     ``walk_forward_run`` row.

STATISTICAL CONTRACT — nothing loosened, nothing redefined:

  * every threshold is IMPORTED from ``analyze`` / ``walk_forward``; this
    module defines no verdict constant of its own;
  * DSR ``n_trials`` = sum over the DISTINCT sleeve (instrument, interval)
    data universes of ``hypothesis_audit.count_data_universe_trials``
    + caller-declared ``external_trials`` + 1 — strictly MORE punitive than
    any single-surface count (monotone: callers can only raise it);
  * certification re-runs of the frozen pre-registered sleeve configs write
    NO ``hypothesis_audit`` rows — they are confirmatory, not selection —
    mirroring the existing ``/walk-forward`` fold-run precedent.

SIZING MODEL (shared-equity-contention approximation, documented for the
reviewer): pooling merges trades of INDEPENDENT full-capital runs. Per-trade
PF / Sharpe / DSR are sizing-scale-invariant, so those are first-order exact
for an equal-slice book. The economic (return) metrics use an explicit
equal-slice no-rebalance model: each of the K sleeves owns 1/K of book
capital and compounds internally at the engine's own alloc-90 semantics;
book multiplier = arithmetic mean of the sleeve multipliers over the COMMON
calendar window (a late-start sleeve idles at multiplier 1.0 for the
uncovered prefix — honest drag). The model never over-deploys (sum of
slices <= 90% of book even with every sleeve concurrently open, so no margin
contention by construction) and claims no cross-sleeve cash reuse (idle
slices earn nothing, so return is understated vs an adaptive shared-cash
book). Residual unmodeled effect — same-symbol sleeves concurrently open on
one live account — is surfaced in the ``contention`` diagnostics.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from ..clients.jvm import JvmClient
from ..config import Settings
from ..errors import OrchestratorError
from ..infra.db import Database
from ..logging import get_logger
from ..repo import hypothesis_audit as audit_repo
from ..repo import iterations as iterations_repo
from ..repo import iterations_write
from ..repo import trades as trades_repo
from ..repo import walk_forward as wf_repo
from . import analyze
from .tick import _fetch_run_metrics, _poll_backtest_status, _resolve_account_strategy
from .walk_forward import (
    MIN_BEAR_OVERLAP_DAYS,
    _build_payload,
    aggregate_folds,
    build_folds,
    stability_verdict,
    window_covers_bear,
)

log = get_logger(__name__)

MIN_SLEEVES = 2
MAX_SLEEVES = 5
POOLED_INTERVAL_LABEL = "pooled"
# walk_forward_run.instrument is VARCHAR(30); the joined distinct-instrument
# label is truncated to fit rather than failing the insert.
_INSTRUMENT_COL_MAX = 30

SIZING_MODEL_DOC = (
    "equal-slice no-rebalance book: each of K sleeves owns 1/K of book "
    "capital, compounds internally at the engine's alloc-90 semantics; "
    "book multiplier = mean(sleeve multipliers) over the common calendar "
    "window; late-start sleeves idle at 1.0. Per-trade PF/Sharpe/DSR are "
    "sizing-scale-invariant (exact); the return metric never over-deploys "
    "and claims no cross-sleeve cash reuse (conservative)."
)


@dataclass
class SleeveSpec:
    """One sleeve of the book — a frozen single-symbol strategy config."""

    strategy_code: str
    instrument: str
    interval_name: str
    window_start: date
    overrides: dict[str, Any] = field(default_factory=dict)

    def to_snapshot(self) -> dict[str, Any]:
        return {
            "strategy_code": self.strategy_code,
            "instrument": self.instrument,
            "interval_name": self.interval_name,
            "window_start": self.window_start.isoformat(),
            "overrides": self.overrides,
        }

    @classmethod
    def from_snapshot(cls, snap: dict[str, Any]) -> "SleeveSpec":
        return cls(
            strategy_code=str(snap["strategy_code"]),
            instrument=str(snap["instrument"]),
            interval_name=str(snap["interval_name"]),
            window_start=date.fromisoformat(str(snap["window_start"])),
            overrides=dict(snap.get("overrides") or {}),
        )


def _f(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def flatten_sleeve_params(sleeves: list[SleeveSpec]) -> dict[str, Any]:
    """Flat scalar-only params_snapshot so the graduation reviewer's
    Hamming-1 ``check_param_robustness`` works across book variants: two
    books differing in exactly one sleeve param are Hamming-distance-1.
    Nested structures would stringify as one mega-key and break the
    distance metric, so the full sleeve detail lives in metrics_snapshot
    instead."""
    flat: dict[str, Any] = {"n_sleeves": str(len(sleeves))}
    for i, s in enumerate(sleeves):
        prefix = f"s{i}"
        flat[f"{prefix}.strategy_code"] = s.strategy_code
        flat[f"{prefix}.instrument"] = s.instrument
        flat[f"{prefix}.interval"] = s.interval_name
        flat[f"{prefix}.window_start"] = s.window_start.isoformat()
        for k in sorted(s.overrides):
            flat[f"{prefix}.{k}"] = str(s.overrides[k])
    return flat


def slice_book_return_pct(sleeve_return_pcts: list[float | None]) -> float | None:
    """Equal-slice no-rebalance book return (percent) from per-sleeve
    cumulative returns over the SAME calendar window. ``None`` entries
    (sleeve metric missing) make the book return non-computable — we never
    fabricate a slice."""
    if not sleeve_return_pcts:
        return None
    mults: list[float] = []
    for r in sleeve_return_pcts:
        if r is None:
            return None
        mults.append(1.0 + r / 100.0)
    return (statistics.fmean(mults) - 1.0) * 100.0


def combine_slice_equity(
    sleeve_daily: list[list[tuple[date, float]]],
) -> tuple[list[tuple[date, float]], float | None]:
    """Combine per-sleeve dated daily-return series into a book equity curve
    under the equal-slice model.

    Returns ``(book_curve, book_max_drawdown_pct)`` where ``book_curve`` is
    ``[(date, book_multiplier), ...]`` and the drawdown is peak-to-trough in
    percent (positive number). Sleeves missing a date hold their multiplier
    flat (idle slice). Returns ``([], None)`` when no sleeve has any data.
    """
    if not sleeve_daily or all(not s for s in sleeve_daily):
        return [], None
    all_dates = sorted({d for series in sleeve_daily for d, _ in series})
    by_sleeve: list[dict[date, float]] = [dict(series) for series in sleeve_daily]
    mults = [1.0] * len(sleeve_daily)
    curve: list[tuple[date, float]] = []
    peak = 1.0
    max_dd = 0.0
    for d in all_dates:
        for i, series_map in enumerate(by_sleeve):
            r = series_map.get(d)
            if r is not None:
                mults[i] *= 1.0 + r / 100.0
        book = statistics.fmean(mults)
        curve.append((d, book))
        peak = max(peak, book)
        if peak > 0:
            max_dd = max(max_dd, (peak - book) / peak * 100.0)
    return curve, round(max_dd, 4)


def contention_diagnostics(
    trades: list[dict[str, Any]],
) -> dict[str, Any]:
    """Concurrency profile of the pooled book — the honesty diagnostic for
    the equal-slice approximation. Requires each trade dict to carry the
    ``_sleeve_instrument`` annotation added at pooling time.

    Reports the share of open-position time with >=2 / >=3 concurrent
    positions, the max concurrency, and the share of open time where two
    sleeves on the SAME instrument are simultaneously open (a live-account
    deployment constraint — the platform's multi-strategy coordinator is
    first-to-open-wins per account)."""
    events: list[tuple[datetime, int, str | None]] = []
    for t in trades:
        entry, exit_ = t.get("entry_time"), t.get("exit_time")
        if isinstance(entry, datetime) and isinstance(exit_, datetime) and exit_ > entry:
            inst = t.get("_sleeve_instrument")
            events.append((entry, 1, inst))
            events.append((exit_, -1, inst))
    if not events:
        return {"computable": False}
    events.sort(key=lambda e: (e[0], -e[1]))
    open_count = 0
    open_by_inst: dict[str, int] = {}
    last_ts: datetime | None = None
    total_open = 0.0
    open_ge2 = 0.0
    open_ge3 = 0.0
    same_symbol = 0.0
    max_concurrent = 0
    for ts, delta, inst in events:
        if last_ts is not None and open_count > 0:
            span = (ts - last_ts).total_seconds()
            total_open += span
            if open_count >= 2:
                open_ge2 += span
            if open_count >= 3:
                open_ge3 += span
            if any(c >= 2 for c in open_by_inst.values()):
                same_symbol += span
        open_count += delta
        if inst is not None:
            open_by_inst[inst] = open_by_inst.get(inst, 0) + delta
        max_concurrent = max(max_concurrent, open_count)
        last_ts = ts

    def _pct(x: float) -> float | None:
        return round(100.0 * x / total_open, 2) if total_open > 0 else None

    return {
        "computable": True,
        "max_concurrent_positions": max_concurrent,
        "open_time_pct_ge2_concurrent": _pct(open_ge2),
        "open_time_pct_ge3_concurrent": _pct(open_ge3),
        "same_symbol_overlap_pct": _pct(same_symbol),
        "note": (
            "Under the equal-slice model concurrent sleeves never contend "
            "for cash (each deploys only its own slice); same-symbol "
            "overlap is a live-deployment constraint (first-to-open-wins "
            "coordinator), not a certification-math error."
        ),
    }


async def _pooled_n_trials(
    conn: Any,
    sleeves: list[SleeveSpec],
    external_trials: int,
) -> dict[str, Any]:
    """Honest multiplicity for the pooled series: the sum of the trial
    counts of every DISTINCT sleeve data universe, plus declared external
    trials, plus 1 for this certification itself. Strictly more punitive
    than any single-surface count."""
    surfaces = sorted({(s.instrument, s.interval_name) for s in sleeves})
    per_surface: dict[str, int] = {}
    total = 0
    for instrument, interval_name in surfaces:
        c = await audit_repo.count_data_universe_trials(conn, instrument, interval_name)
        per_surface[f"{instrument}:{interval_name}"] = c
        total += c
    extra = max(0, int(external_trials or 0))
    return {
        "n_trials": total + extra + 1,
        "per_surface": per_surface,
        "external_declared": extra,
    }


def _pool_trades(
    per_sleeve_trades: list[list[dict[str, Any]]],
    sleeves: list[SleeveSpec],
) -> list[dict[str, Any]]:
    pooled: list[dict[str, Any]] = []
    for sleeve, trades in zip(sleeves, per_sleeve_trades, strict=True):
        for t in trades:
            t = dict(t)
            t["_sleeve_instrument"] = sleeve.instrument
            t["_sleeve_interval"] = sleeve.interval_name
            pooled.append(t)
    pooled.sort(
        key=lambda t: t.get("entry_time") or datetime.min.replace(tzinfo=None)
    )
    return pooled


async def _run_sleeve_backtest(
    *,
    db: Database,
    jvm: JvmClient,
    settings: Settings,
    sleeve: SleeveSpec,
    test_start: date,
    test_end: date,
    as_cache: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    """One standard single-symbol run for a sleeve over [test_start, test_end).
    Returns (run_id, run_metrics, trades). Raises OrchestratorError on
    submit failure / non-COMPLETED terminal / poll-cap expiry."""
    as_row = as_cache.get(sleeve.strategy_code)
    if as_row is None:
        async with db.acquire() as conn:
            as_row = await _resolve_account_strategy(
                conn, sleeve.strategy_code, settings.research_account_id
            )
        if not as_row:
            raise OrchestratorError(
                status_code=412,
                error_code="account_strategy_missing",
                message=(
                    f"No account_strategy row for sleeve strategy_code="
                    f"{sleeve.strategy_code}."
                ),
                retryable=False,
                hint="Seed the research account_strategy row before certifying.",
            )
        as_cache[sleeve.strategy_code] = as_row
    payload = _build_payload(
        account_strategy_id=as_row["account_strategy_id"],
        strategy_code=sleeve.strategy_code,
        asset=sleeve.instrument,
        interval_name=sleeve.interval_name,
        test_start=test_start,
        test_end=test_end,
        overrides=sleeve.overrides,
        allow_long=as_row["allow_long"],
        allow_short=as_row["allow_short"],
    )
    run_id = await jvm.submit_backtest(payload)
    final_status = await _poll_backtest_status(
        db, run_id, timeout_s=settings.poll_timeout_s
    )
    if final_status != "COMPLETED":
        raise OrchestratorError(
            status_code=502,
            error_code="pooled_sleeve_run_failed",
            message=(
                f"Sleeve {sleeve.strategy_code}/{sleeve.instrument}/"
                f"{sleeve.interval_name} run {run_id} terminal status "
                f"{final_status} (expected COMPLETED)."
            ),
            retryable=final_status not in ("FAILED", "CANCELLED", "ERROR"),
            details={"run_id": run_id, "status": final_status},
        )
    async with db.acquire() as conn:
        run_metrics = await _fetch_run_metrics(conn, run_id)
        trades = await trades_repo.fetch_trades(conn, UUID(str(run_id)))
    return str(run_id), run_metrics, trades


async def run_pooled_analyze(
    *,
    db: Database,
    jvm: JvmClient,
    settings: Settings,
    agent_name: str,
    book_strategy_code: str,
    sleeves: list[SleeveSpec],
    full_end: date | None = None,
    external_trials: int = 0,
    motivating_hypothesis_id: UUID | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Full-window pooled analysis. Persists ONE research_iteration_log row
    on the book strategy_code and returns the pooled scoring payload."""
    if not (MIN_SLEEVES <= len(sleeves) <= MAX_SLEEVES):
        raise OrchestratorError(
            status_code=400,
            error_code="pooled_sleeve_count_invalid",
            message=(
                f"Pooled certification requires {MIN_SLEEVES}..{MAX_SLEEVES} "
                f"sleeves, got {len(sleeves)}."
            ),
            retryable=False,
        )
    if full_end is None:
        full_end = (datetime.now(timezone.utc) - timedelta(days=1)).date()
    for s in sleeves:
        if s.window_start >= full_end:
            raise OrchestratorError(
                status_code=400,
                error_code="pooled_sleeve_window_invalid",
                message=(
                    f"Sleeve {s.instrument}/{s.interval_name} window_start "
                    f"{s.window_start} is not before full_end {full_end}."
                ),
                retryable=False,
            )

    as_cache: dict[str, dict[str, Any]] = {}
    sleeve_results: list[dict[str, Any]] = []
    per_sleeve_trades: list[list[dict[str, Any]]] = []
    per_sleeve_daily: list[list[tuple[date, float]]] = []
    for idx, sleeve in enumerate(sleeves):
        log.info(
            "pooled_cert.sleeve_start",
            sleeve=idx,
            strategy_code=sleeve.strategy_code,
            instrument=sleeve.instrument,
            interval=sleeve.interval_name,
        )
        run_id, run_metrics, trades = await _run_sleeve_backtest(
            db=db,
            jvm=jvm,
            settings=settings,
            sleeve=sleeve,
            test_start=sleeve.window_start,
            test_end=full_end,
            as_cache=as_cache,
        )
        async with db.acquire() as conn:
            daily = await wf_repo.fetch_daily_returns_dated(conn, UUID(run_id))
        per_sleeve_daily.append(daily)
        per_sleeve_trades.append(trades)
        sleeve_results.append(
            {
                **sleeve.to_snapshot(),
                "run_id": run_id,
                "n_trades": int(run_metrics.get("total_trades") or 0),
                "profit_factor": _f(run_metrics.get("profit_factor")),
                "return_pct": _f(run_metrics.get("return_pct")),
                "geometric_return_pct_at_alloc_90": _f(
                    run_metrics.get("geometric_return_pct_at_alloc_90")
                ),
                "max_drawdown_pct": _f(run_metrics.get("max_drawdown_pct")),
            }
        )

    pooled_trades = _pool_trades(per_sleeve_trades, sleeves)
    full_start = min(s.window_start for s in sleeves)

    book_return_pct = slice_book_return_pct(
        [r["return_pct"] for r in sleeve_results]
    )
    book_geom90_pct = slice_book_return_pct(
        [r["geometric_return_pct_at_alloc_90"] for r in sleeve_results]
    )
    _, book_max_dd = combine_slice_equity(per_sleeve_daily)

    async with db.acquire() as conn:
        trials = await _pooled_n_trials(conn, sleeves, external_trials)

    pseudo_run = {
        "start_time": datetime.combine(full_start, datetime.min.time()),
        "end_time": datetime.combine(full_end, datetime.min.time()),
        "return_pct": book_return_pct,
        "max_drawdown_pct": book_max_dd,
        "geometric_return_pct_at_alloc_90": book_geom90_pct,
    }
    analysis = analyze.analyze_run(
        pseudo_run, pooled_trades, cumulative_trials=trials["n_trials"]
    )
    stat = analysis["statistical_verdict"]
    stat_v = stat["verdict"]
    n = analysis["n_trades"]
    dec_v = analyze.decision_verdict(
        stat_v, n, analysis.get("annualized_geometric_return_pct_at_alloc_90")
    )
    sample_ok = analyze.sample_size_adequate(n)
    contention = contention_diagnostics(pooled_trades)

    distinct_instruments = sorted({s.instrument for s in sleeves})
    asset_label = ",".join(distinct_instruments)[:_INSTRUMENT_COL_MAX]
    pnls = [_f(t.get("realized_pnl_amount")) or 0.0 for t in pooled_trades]

    pooled_certification = {
        "sleeves": sleeve_results,
        "full_start": full_start.isoformat(),
        "full_end": full_end.isoformat(),
        "sizing_model": SIZING_MODEL_DOC,
        "contention": contention,
        "n_trials_breakdown": trials,
        "book_slice_return_pct": book_return_pct,
        "book_slice_geom90_pct": book_geom90_pct,
        "book_slice_max_drawdown_pct": book_max_dd,
        "motivating_hypothesis_id": (
            str(motivating_hypothesis_id) if motivating_hypothesis_id else None
        ),
        "notes": notes,
    }
    metrics_snapshot = {
        "backtest_run_id": None,
        "strategy_code": book_strategy_code,
        "status": "POOLED",
        "total_trades": n,
        "profit_factor": analysis.get("pf_point_estimate"),
        "net_profit": round(sum(pnls), 4),
        "return_pct": book_return_pct,
        "max_drawdown_pct": book_max_dd,
        "sharpe_ratio": analysis.get("sharpe_annualized"),
        "asset": asset_label,
        "interval_name": POOLED_INTERVAL_LABEL,
        "start_time": pseudo_run["start_time"].isoformat(),
        "end_time": pseudo_run["end_time"].isoformat(),
        "geometric_return_pct_at_alloc_90": book_geom90_pct,
        "analysis": analysis,
        "pooled_certification": pooled_certification,
    }

    async with db.acquire() as conn:
        async with conn.transaction():
            iter_n = await iterations_write.next_iteration_number(
                conn, book_strategy_code
            )
            iteration_id = await iterations_write.insert_iteration(
                conn,
                strategy_code=book_strategy_code,
                iteration_number=iter_n,
                backtest_run_id=None,
                params_snapshot=flatten_sleeve_params(sleeves),
                metrics_snapshot=metrics_snapshot,
                confidence_intervals={"pf_95": analysis["pf_95_ci"]},
                verdict=dec_v,
                statistical_verdict=stat_v,
                sample_size_adequate=sample_ok,
                git_commit_hash=None,
                code_change_summary=(
                    f"Pooled certification analyze ({len(sleeves)} sleeves) via "
                    f"POST /pooled-certification/analyze."
                ),
                change_reasoning=(
                    "Orchestrator-side pooled-book certification (operator "
                    "methodology approval 2026-07-10): coordinated standard "
                    "single-symbol runs pooled at trade level; V11 gates "
                    "unchanged."
                ),
                quant_audit_notes=(
                    f"Pooled series over sleeves="
                    f"{[f'{s.instrument}/{s.interval_name}' for s in sleeves]}; "
                    f"DSR n_trials={trials['n_trials']} "
                    f"(per-surface={trials['per_surface']}, "
                    f"external={trials['external_declared']}, +1 self). "
                    f"Economic metrics use the equal-slice no-rebalance model "
                    f"(see metrics_snapshot.pooled_certification.sizing_model). "
                    f"Sleeve runs: {[r['run_id'] for r in sleeve_results]}."
                ),
                created_by=agent_name,
            )

    log.info(
        "pooled_cert.analyze_complete",
        iteration_id=str(iteration_id),
        book=book_strategy_code,
        n_trades=n,
        statistical_verdict=stat_v,
        verdict=dec_v,
    )
    return {
        "iteration_id": str(iteration_id),
        "strategy_code": book_strategy_code,
        "statistical_verdict": stat_v,
        "verdict": dec_v,
        "n_trades": n,
        "analysis": analysis,
        "sleeves": sleeve_results,
        "contention": contention,
        "n_trials_breakdown": trials,
        "book_slice_return_pct": book_return_pct,
        "book_slice_geom90_pct": book_geom90_pct,
        "book_slice_max_drawdown_pct": book_max_dd,
        "sizing_model": SIZING_MODEL_DOC,
    }


async def run_pooled_walk_forward(
    *,
    db: Database,
    jvm: JvmClient,
    settings: Settings,
    agent_name: str,
    iteration_id: UUID,
    train_months: int = 12,
    test_months: int = 3,
    step_months: int = 3,
    n_folds: int = 18,
    require_bear_coverage: bool = True,
) -> dict[str, Any]:
    """Pooled-series walk-forward for a book iteration produced by
    ``run_pooled_analyze``. Fold plan + verdict cutoffs are byte-identical
    to the standard path; only the per-fold trade series is the pooled one."""
    if step_months < test_months:
        raise OrchestratorError(
            status_code=400,
            error_code="walk_forward_overlapping_folds",
            message=(
                f"step_months={step_months} < test_months={test_months} produces "
                f"overlapping OOS test windows."
            ),
            retryable=False,
            hint="Use step_months >= test_months (default 3/3).",
        )

    async with db.acquire() as conn:
        iteration = await iterations_repo.get_iteration(conn, iteration_id)
    if iteration is None:
        raise OrchestratorError(
            status_code=404,
            error_code="iteration_not_found",
            message=f"iteration_id={iteration_id} not in research_iteration_log.",
            retryable=False,
        )
    pooled_meta = (iteration.get("metrics_snapshot") or {}).get(
        "pooled_certification"
    )
    if not pooled_meta or not pooled_meta.get("sleeves"):
        raise OrchestratorError(
            status_code=400,
            error_code="pooled_iteration_invalid",
            message=(
                f"iteration_id={iteration_id} is not a pooled-certification "
                "iteration (metrics_snapshot.pooled_certification missing)."
            ),
            retryable=False,
            hint="Run POST /pooled-certification/analyze first.",
        )
    book_strategy_code = str(iteration["strategy_code"])
    sleeves = [SleeveSpec.from_snapshot(s) for s in pooled_meta["sleeves"]]
    full_start = date.fromisoformat(str(pooled_meta["full_start"]))
    full_end = date.fromisoformat(str(pooled_meta["full_end"]))

    if require_bear_coverage and not window_covers_bear(full_start, full_end):
        raise OrchestratorError(
            status_code=422,
            error_code="validation_window_excludes_bear",
            message=(
                f"Pooled walk-forward window [{full_start}, {full_end}] contains "
                f"no known bear regime (>= {MIN_BEAR_OVERLAP_DAYS}d overlap)."
            ),
            retryable=False,
        )

    folds = build_folds(
        full_start, full_end, train_months, test_months, step_months, n_folds
    )
    if not folds:
        raise OrchestratorError(
            status_code=400,
            error_code="walk_forward_no_folds",
            message="No folds fit in the pooled certification window.",
            retryable=False,
        )

    as_cache: dict[str, dict[str, Any]] = {}
    fold_results: list[dict[str, Any]] = []
    for idx, (test_start, test_end) in enumerate(folds, start=1):
        log.info(
            "pooled_cert.fold_start",
            fold=idx,
            n_folds=len(folds),
            test_start=test_start.isoformat(),
            test_end=test_end.isoformat(),
        )
        sleeve_rows: list[dict[str, Any]] = []
        fold_trades: list[dict[str, Any]] = []
        fold_return_slices: list[float] = []
        fold_error: str | None = None
        for sleeve in sleeves:
            eff_start = max(test_start, sleeve.window_start)
            if eff_start >= test_end:
                # Sleeve's data starts after this fold — idle slice
                # (contributes 0% return and no trades; matches the
                # documented E3b convention for late-start sleeves).
                sleeve_rows.append(
                    {
                        "instrument": sleeve.instrument,
                        "interval_name": sleeve.interval_name,
                        "run_id": None,
                        "skipped": "window_starts_after_fold",
                    }
                )
                fold_return_slices.append(0.0)
                continue
            try:
                run_id, run_metrics, trades = await _run_sleeve_backtest(
                    db=db,
                    jvm=jvm,
                    settings=settings,
                    sleeve=sleeve,
                    test_start=eff_start,
                    test_end=test_end,
                    as_cache=as_cache,
                )
            except OrchestratorError as e:
                sleeve_rows.append(
                    {
                        "instrument": sleeve.instrument,
                        "interval_name": sleeve.interval_name,
                        "run_id": None,
                        "error": e.envelope.error_code,
                    }
                )
                fold_error = e.envelope.error_code
                continue
            for t in trades:
                t = dict(t)
                t["_sleeve_instrument"] = sleeve.instrument
                t["_sleeve_interval"] = sleeve.interval_name
                fold_trades.append(t)
            ret = _f(run_metrics.get("return_pct"))
            fold_return_slices.append(ret if ret is not None else 0.0)
            sleeve_rows.append(
                {
                    "instrument": sleeve.instrument,
                    "interval_name": sleeve.interval_name,
                    "run_id": run_id,
                    "trades": int(run_metrics.get("total_trades") or 0),
                    "pf": _f(run_metrics.get("profit_factor")),
                    "return_pct": ret,
                }
            )
        if fold_error is not None:
            # A failed sleeve makes the fold's pooled series incomplete —
            # scoring a partial book would silently change the strategy
            # under test. Mark the fold errored (aggregate_folds skips it).
            fold_results.append(
                {
                    "fold": idx,
                    "test_start": test_start.isoformat(),
                    "test_end": test_end.isoformat(),
                    "sleeves": sleeve_rows,
                    "error": fold_error,
                }
            )
            continue
        pnls = [_f(t.get("realized_pnl_amount")) or 0.0 for t in fold_trades]
        n_fold = len(pnls)
        fold_days = (test_end - test_start).days
        ann_factor = (
            (analyze.DAYS_PER_YEAR / fold_days) * n_fold
            if fold_days > 0 and n_fold > 0
            else None
        )
        wins = len([p for p in pnls if p > 0])
        # Equal-slice fold return: every sleeve contributes its slice
        # (skipped/idle sleeves contribute 0.0), divided by the FULL
        # sleeve count — an idle slice dilutes, it doesn't vanish.
        fold_return = (
            round(sum(fold_return_slices) / len(sleeves), 6) if sleeves else None
        )
        fold_results.append(
            {
                "fold": idx,
                "test_start": test_start.isoformat(),
                "test_end": test_end.isoformat(),
                "sleeves": sleeve_rows,
                "metrics": {
                    "pf": analyze.profit_factor(pnls),
                    "return_pct": fold_return,
                    "sharpe": analyze.sharpe_ratio(pnls, ann_factor),
                    "wr": round(wins / n_fold, 4) if n_fold else None,
                    "trades": n_fold,
                    "net_profit": round(sum(pnls), 4),
                },
            }
        )

    agg = aggregate_folds(fold_results)
    agg["validation_track"] = "POOLED_STANDARD"
    verdict = stability_verdict(agg)

    distinct_instruments = sorted({s.instrument for s in sleeves})
    fold_results.append(
        {
            "validation_summary": {
                "validation_track": "POOLED_STANDARD",
                "n_sleeves": len(sleeves),
                "sizing_model": SIZING_MODEL_DOC,
                "pooled_iteration_id": str(iteration_id),
            }
        }
    )

    pf = agg.get("pf") or {}
    ret = agg.get("return_pct") or {}
    sharpe = agg.get("sharpe") or {}
    async with db.acquire() as conn:
        wf_id = await wf_repo.insert_walk_forward(
            conn,
            strategy_code=book_strategy_code,
            interval_name=POOLED_INTERVAL_LABEL,
            instrument=",".join(distinct_instruments)[:_INSTRUMENT_COL_MAX],
            full_start=datetime.combine(full_start, datetime.min.time()),
            full_end=datetime.combine(full_end, datetime.min.time()),
            train_months=train_months,
            test_months=test_months,
            n_folds=len(folds),
            fold_pf_mean=pf.get("mean"),
            fold_pf_std=pf.get("std"),
            fold_pf_min=pf.get("min"),
            fold_pf_max=pf.get("max"),
            fold_pf_positive_pct=agg.get("pf_positive_pct"),
            fold_return_mean=ret.get("mean"),
            fold_return_std=ret.get("std"),
            fold_sharpe_mean=sharpe.get("mean"),
            fold_sharpe_std=sharpe.get("std"),
            total_trades_across_folds=agg.get("total_trades_across_folds") or 0,
            stability_verdict=verdict,
            motivating_iteration_id=iteration_id,
            fold_results=fold_results,
            git_commit_hash=None,
            notes=(
                f"Pooled-book walk-forward ({len(sleeves)} sleeves) via "
                f"POST /pooled-certification/walk-forward. Fold cutoffs "
                f"byte-identical to services/walk_forward.stability_verdict."
            ),
            created_by=agent_name,
        )

    log.info(
        "pooled_cert.walk_forward_complete",
        walk_forward_id=str(wf_id),
        verdict=verdict,
        n_folds=len(folds),
    )
    return {
        "walk_forward_id": str(wf_id),
        "stability_verdict": verdict,
        "aggregate": agg,
        "fold_results": fold_results,
    }
