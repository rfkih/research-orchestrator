"""Walk-forward orchestration — port of ``research/scripts/walk-forward.sh``.

Splits the standard window into N rolling folds, runs a backtest per
fold's TEST period, aggregates fold-level metrics, and persists a single
``walk_forward_run`` row. The stability verdict mirrors the bash logic:

  total trades < 100             → INSUFFICIENT_EVIDENCE
  fold_pf_mean < 1.0             → NO_EDGE
  pf_positive_pct < 60           → OVERFIT (mean>1) | NO_EDGE (mean<=1)
  pf_std > 1.5                   → INCONSISTENT
  otherwise                      → ROBUST

Each fold's backtest is submitted to the JVM and polled to completion,
mirroring tick.py. With 6 folds × up to 30min, a full walk-forward can
take 3 hours — too long for an HTTP request from the cron tick. We
expose this as ``POST /walk-forward`` so the agent calls it deliberately
(or a separate cron does), and gate the PASS verdict on ROBUST.
"""

from __future__ import annotations

import statistics
from datetime import date, datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from ..clients.jvm import JvmClient
from ..config import Settings
from ..errors import OrchestratorError
from ..infra.db import Database
from ..logging import get_logger
from ..repo import walk_forward as wf_repo
from .tick import _fetch_run_metrics, _poll_backtest_status, _resolve_account_strategy

log = get_logger(__name__)


def _add_months(d: date, months: int) -> date:
    """Calendar-month addition without dateutil. Clamps day-of-month to
    the new month's max (matches ``relativedelta`` behaviour)."""
    total = d.year * 12 + (d.month - 1) + months
    year, month = divmod(total, 12)
    month += 1
    # Days in target month
    if month == 12:
        next_first = date(year + 1, 1, 1)
    else:
        next_first = date(year, month + 1, 1)
    last_day = (next_first - timedelta(days=1)).day
    return date(year, month, min(d.day, last_day))


def build_folds(
    full_start: date,
    full_end: date,
    train_months: int,
    test_months: int,
    step_months: int,
    n_folds: int,
) -> list[tuple[date, date]]:
    """Return ``[(test_start, test_end), ...]``. Stops when test_end
    overruns full_end. Bash counts folds where every test_end fits."""
    folds: list[tuple[date, date]] = []
    for i in range(n_folds):
        train_start = _add_months(full_start, i * step_months)
        test_start = _add_months(train_start, train_months)
        test_end = _add_months(test_start, test_months)
        if test_end > full_end:
            break
        folds.append((test_start, test_end))
    return folds


def aggregate_folds(fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    metric_data: dict[str, list[float]] = {
        "pf": [], "return_pct": [], "sharpe": [], "trades": [], "wr": [],
    }
    for f in fold_results:
        if "metrics" not in f or f.get("error"):
            continue
        for k in metric_data:
            v = f["metrics"].get(k)
            if v is not None:
                metric_data[k].append(float(v))

    def stats(key: str) -> dict[str, float] | None:
        vals = metric_data[key]
        if not vals:
            return None
        return {
            "mean": round(statistics.fmean(vals), 4),
            "std": round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0,
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
        }

    n_completed = len([f for f in fold_results if "metrics" in f and not f.get("error")])
    pf_pos = (
        round(100 * len([v for v in metric_data["pf"] if v > 1.0])
              / max(len(metric_data["pf"]), 1), 2)
    )
    return {
        "n_folds_completed": n_completed,
        "pf": stats("pf"),
        "return_pct": stats("return_pct"),
        "sharpe": stats("sharpe"),
        "pf_positive_pct": pf_pos,
        "total_trades_across_folds": int(sum(metric_data["trades"])),
    }


def stability_verdict(agg: dict[str, Any]) -> str:
    pf = agg.get("pf") or {}
    pf_mean = pf.get("mean") or 0.0
    pf_pos = agg.get("pf_positive_pct") or 0.0
    pf_std = pf.get("std") or 0.0
    n = agg.get("total_trades_across_folds") or 0

    if n < 100:
        return "INSUFFICIENT_EVIDENCE"
    if pf_mean < 1.0:
        return "NO_EDGE"
    if pf_pos < 60:
        return "OVERFIT" if pf_mean > 1.0 else "NO_EDGE"
    if pf_std > 1.5:
        return "INCONSISTENT"
    return "ROBUST"


def _build_payload(
    *,
    account_strategy_id: str,
    strategy_code: str,
    asset: str,
    interval_name: str,
    test_start: date,
    test_end: date,
    overrides: dict[str, Any] | None,
    allow_long: bool = True,
    allow_short: bool = True,
) -> dict[str, Any]:
    """Same shape as tick._build_submit_payload, with the per-fold window."""
    return {
        "accountStrategyId": account_strategy_id,
        "strategyCode": strategy_code,
        "asset": asset,
        "interval": interval_name,
        "startTime": f"{test_start.isoformat()}T00:00:00",
        "endTime": f"{test_end.isoformat()}T00:00:00",
        "initialCapital": 100,
        "riskPerTradePct": 2.0,
        "feeRate": 0.00075,
        "minNotional": 5,
        "minQty": 0.000001,
        "qtyStep": 0.000001,
        "maxOpenPositions": 1,
        # Mirror the production strategy's direction policy — see tick.py.
        "allowLong": allow_long,
        "allowShort": allow_short,
        "strategyParamOverrides": {strategy_code: overrides or {}},
        "triggeredBy": "RESEARCHER",
    }


class WalkForwardResult:
    def __init__(
        self,
        *,
        walk_forward_id: UUID,
        stability_verdict: str,
        aggregate: dict[str, Any],
        fold_results: list[dict[str, Any]],
    ) -> None:
        self.walk_forward_id = walk_forward_id
        self.stability_verdict = stability_verdict
        self.aggregate = aggregate
        self.fold_results = fold_results

    def to_dict(self) -> dict[str, Any]:
        return {
            "walk_forward_id": str(self.walk_forward_id),
            "stability_verdict": self.stability_verdict,
            "aggregate": self.aggregate,
            "fold_results": self.fold_results,
        }


async def run_walk_forward(
    *,
    db: Database,
    jvm: JvmClient,
    settings: Settings,  # noqa: ARG001 — reserved for prod-window/capital config
    agent_name: str,
    strategy_code: str,
    interval_name: str,
    instrument: str = "BTCUSDT",
    full_start: date = date(2024, 1, 1),
    full_end: date | None = None,
    train_months: int = 12,
    test_months: int = 3,
    step_months: int = 3,
    n_folds: int = 6,
    overrides: dict[str, Any] | None = None,
    motivating_iteration_id: UUID | None = None,
) -> WalkForwardResult:
    if full_end is None:
        full_end = (datetime.now(timezone.utc) - timedelta(days=1)).date()

    async with db.acquire() as conn:
        as_row = await _resolve_account_strategy(conn, strategy_code)
    if not as_row:
        raise OrchestratorError(
            status_code=412,
            error_code="account_strategy_missing",
            message=f"No account_strategy row for strategy_code={strategy_code}.",
            retryable=False,
            hint="Seed an account_strategy row before requesting walk-forward.",
        )
    as_id = as_row["account_strategy_id"]

    folds = build_folds(full_start, full_end, train_months, test_months, step_months, n_folds)
    if not folds:
        raise OrchestratorError(
            status_code=400,
            error_code="walk_forward_no_folds",
            message="No folds fit in the requested window.",
            retryable=False,
            details={
                "full_start": full_start.isoformat(),
                "full_end": full_end.isoformat(),
                "train_months": train_months,
                "test_months": test_months,
                "step_months": step_months,
                "n_folds": n_folds,
            },
        )

    fold_results: list[dict[str, Any]] = []
    for idx, (test_start, test_end) in enumerate(folds, start=1):
        log.info(
            "walk_forward.fold_start",
            strategy_code=strategy_code,
            fold=idx,
            n_folds=len(folds),
            test_start=test_start.isoformat(),
            test_end=test_end.isoformat(),
        )
        payload = _build_payload(
            account_strategy_id=as_id,
            strategy_code=strategy_code,
            asset=instrument,
            interval_name=interval_name,
            test_start=test_start,
            test_end=test_end,
            overrides=overrides,
            allow_long=as_row["allow_long"],
            allow_short=as_row["allow_short"],
        )
        try:
            run_id = await jvm.submit_backtest(payload)
        except OrchestratorError as e:
            fold_results.append({
                "fold": idx,
                "test_start": test_start.isoformat(),
                "test_end": test_end.isoformat(),
                "run_id": None,
                "error": e.error_code,
            })
            continue

        final_status = await _poll_backtest_status(db, run_id)
        if final_status != "COMPLETED":
            fold_results.append({
                "fold": idx,
                "test_start": test_start.isoformat(),
                "test_end": test_end.isoformat(),
                "run_id": run_id,
                "error": f"status={final_status}",
            })
            continue

        async with db.acquire() as conn:
            metrics = await _fetch_run_metrics(conn, run_id)
        fold_results.append({
            "fold": idx,
            "test_start": test_start.isoformat(),
            "test_end": test_end.isoformat(),
            "run_id": run_id,
            "metrics": {
                "pf": metrics.get("profit_factor"),
                "return_pct": metrics.get("return_pct"),
                "sharpe": metrics.get("sharpe_ratio"),
                "wr": metrics.get("win_rate"),
                "trades": metrics.get("total_trades"),
                "max_dd_pct": metrics.get("max_drawdown_pct"),
                "net_profit": metrics.get("net_profit"),
            },
        })

    agg = aggregate_folds(fold_results)
    verdict = stability_verdict(agg)
    pf = agg.get("pf") or {}
    ret = agg.get("return_pct") or {}
    sharpe = agg.get("sharpe") or {}

    async with db.acquire() as conn:
        wf_id = await wf_repo.insert_walk_forward(
            conn,
            strategy_code=strategy_code,
            interval_name=interval_name,
            instrument=instrument,
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
            motivating_iteration_id=motivating_iteration_id,
            fold_results=fold_results,
            git_commit_hash=None,
            notes=None,
            created_by=agent_name,
        )

    log.info(
        "walk_forward.complete",
        strategy_code=strategy_code,
        walk_forward_id=str(wf_id),
        verdict=verdict,
        n_folds=len(folds),
    )
    return WalkForwardResult(
        walk_forward_id=wf_id,
        stability_verdict=verdict,
        aggregate=agg,
        fold_results=fold_results,
    )
