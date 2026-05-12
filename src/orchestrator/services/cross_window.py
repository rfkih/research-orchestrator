"""Cross-window validation — Phase 5.

Sibling to walk_forward.py. Where walk-forward asks "does the edge hold
across rolling chronological folds?", cross-window asks "does the edge
hold across *named regime epochs*?". The two are complementary: a
strategy can pass walk-forward (60% folds positive) by riding a single
multi-year regime, but fail cross-window (need 80% regime-labeled
windows net-positive after +20bps slippage).

The verdict ``ROBUST_CROSS_WINDOW`` is the harder bar of the two.

Window catalog lives in ``regime_windows.py`` (Python module — type-safe,
no YAML dep). Each backtest is submitted to the JVM via the same
submit/poll path tick.py uses.

End-date convention: ``window.end`` is **exclusive**. A window covering
"all of 2024" is ``start=2024-01-01, end=2025-01-01`` — the JVM payload
emits ``endTime=2025-01-01T00:00:00`` and the trade with the largest
entry_time <= that bound is the last included.

Reliability notes:
  * Per-window failures are contained — a transient exception inside any
    single window is captured into ``window_results[i].error`` and the
    loop continues. Five hours of validation work is never lost to one
    flaky JVM poll.
  * Windows whose date range falls entirely before the instrument's
    data-availability minimum are skipped up front (status=``no_data``)
    instead of being submitted to the JVM only to come back empty.
"""

from __future__ import annotations

import statistics
from datetime import date
from typing import Any
from uuid import UUID

import asyncpg

from ..clients.jvm import JvmClient
from ..config import Settings
from ..errors import OrchestratorError
from ..infra.db import Database
from ..logging import get_logger
from ..repo import cross_window as cw_repo
from ..repo import trades as trades_repo
from .analyze import PF_INFINITY_SENTINEL, slippage_sensitivity
from .regime_windows import CATALOG_VERSION, RegimeWindow, windows_for
from .tick import _fetch_run_metrics, _poll_backtest_status, _resolve_account_strategy
from .walk_forward import _build_payload

log = get_logger(__name__)

# Stricter than walk-forward's 60% fold-positive threshold. The whole
# point of cross-window is to demand consistency *across regimes*.
ROBUST_PCT_THRESHOLD = 80.0

# Below this many trades total across all eligible windows the edge claim
# is statistically empty regardless of % positive.
MIN_TOTAL_TRADES = 100

# Per-window minimum trades. A window with 3 trades that happen to be
# winners shouldn't count as evidence; mark such windows as "thin" and
# exclude from the % positive numerator AND denominator.
MIN_TRADES_PER_WINDOW = 30

# Slippage haircut bps used for the "net positive" gate. Matches the
# +20bps decision_verdict gate in analyze.py — a window is "net positive"
# only if it survives realistic execution costs.
SLIPPAGE_GATE_BPS = 20

# Disclosure note attached to every aggregate. The +20bps haircut is
# computed via analyze.slippage_sensitivity, which proxies notional with
# `max(|pnl|, $10) × 50`. For high-frequency strategies with many small-
# PnL trades this is conservative (over-charges); for low-frequency
# strategies with large PnL trades it can under-charge. Order-of-magnitude
# correct, not exact. Documented in the response so reviewers don't read
# the % positive number as a perfect post-cost figure.
SLIPPAGE_PROXY_NOTE = (
    "Slippage haircut uses analyze.slippage_sensitivity proxy "
    "(notional ≈ max(|pnl|, $10) × 50). Tends to over-charge HFT, "
    "under-charge low-frequency."
)


class CrossWindowResult:
    def __init__(
        self,
        *,
        cross_window_id: UUID,
        cross_window_verdict: str,
        windows_catalog_version: str,
        aggregate: dict[str, Any],
        window_results: list[dict[str, Any]],
        motivating_walk_forward_id: UUID | None = None,
        motivating_iteration_id: UUID | None = None,
    ) -> None:
        self.cross_window_id = cross_window_id
        self.cross_window_verdict = cross_window_verdict
        self.windows_catalog_version = windows_catalog_version
        self.aggregate = aggregate
        self.window_results = window_results
        self.motivating_walk_forward_id = motivating_walk_forward_id
        self.motivating_iteration_id = motivating_iteration_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "cross_window_id": str(self.cross_window_id),
            "cross_window_verdict": self.cross_window_verdict,
            "windows_catalog_version": self.windows_catalog_version,
            "aggregate": self.aggregate,
            "window_results": self.window_results,
            "motivating_walk_forward_id": (
                str(self.motivating_walk_forward_id)
                if self.motivating_walk_forward_id else None
            ),
            "motivating_iteration_id": (
                str(self.motivating_iteration_id)
                if self.motivating_iteration_id else None
            ),
        }


async def _data_min_start(
    conn: asyncpg.Connection, instrument: str, interval_name: str
) -> date | None:
    """Return the earliest ``market_data.start_time::date`` for the
    instrument+interval, or None if the instrument isn't plumbed.

    Used to skip catalog windows that fall entirely before our data — a
    cross-window run on ETH today would otherwise submit 6 backtests for
    2022-2024 that all return zero trades, wasting ~3 hours.
    """
    row = await conn.fetchval(
        """
        SELECT MIN(start_time)::date
        FROM market_data
        WHERE symbol = $1 AND interval = $2
        """,
        instrument,
        interval_name,
    )
    return row  # asyncpg returns date or None


def aggregate_windows(window_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-window metrics into the summary used by the verdict.

    ``thin`` windows (trades < MIN_TRADES_PER_WINDOW) are excluded from
    pf/return stats AND from the positive numerator/denominator, so a
    handful of lucky winners in a 5-trade window can't lift the verdict.

    PF values >= PF_INFINITY_SENTINEL (no-loss sentinel) are dropped from
    pf stats — otherwise one outlier window of 9999 turns the mean into
    garbage and renders the persisted column meaningless.
    """
    pf_vals: list[float] = []
    ret_vals: list[float] = []
    total_trades = 0
    eligible_trades = 0  # trades only from non-thin, non-error windows
    eligible_n = 0
    net_positive_n = 0
    thin_windows: list[str] = []
    no_data_windows: list[str] = []
    pf_sentinel_dropped = 0

    for w in window_results:
        if w.get("error") == "no_data":
            no_data_windows.append(w["label"])
            continue
        if w.get("error"):
            continue
        trades = int(w.get("trades") or 0)
        total_trades += trades
        if trades < MIN_TRADES_PER_WINDOW:
            thin_windows.append(w["label"])
            continue
        eligible_n += 1
        eligible_trades += trades
        pf = w.get("pf")
        if pf is not None:
            pf_f = float(pf)
            if pf_f >= PF_INFINITY_SENTINEL:
                pf_sentinel_dropped += 1
            else:
                pf_vals.append(pf_f)
        ret = w.get("return_pct")
        if ret is not None:
            ret_vals.append(float(ret))
        slip_net = w.get("slip_20bps_net")
        if slip_net is not None and float(slip_net) > 0:
            net_positive_n += 1

    def stats(vals: list[float]) -> dict[str, float] | None:
        if not vals:
            return None
        return {
            "mean": round(statistics.fmean(vals), 4),
            "std": round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0,
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
        }

    pct_positive = (
        round(100.0 * net_positive_n / eligible_n, 2) if eligible_n > 0 else 0.0
    )

    notes: list[str] = [SLIPPAGE_PROXY_NOTE]
    if pf_sentinel_dropped:
        notes.append(
            f"{pf_sentinel_dropped} window(s) had no losses (PF sentinel) — "
            "excluded from pf stats."
        )

    return {
        "n_windows_completed": len(
            [w for w in window_results if not w.get("error")]
        ),
        "n_windows_eligible": eligible_n,
        "n_windows_net_positive": net_positive_n,
        "pct_windows_net_positive": pct_positive,
        "pf": stats(pf_vals),
        "return_pct": stats(ret_vals),
        # All-windows total (includes thin); kept for parity with walk_forward.
        "total_trades_across_windows": total_trades,
        # Eligible-only total — the number that actually backs the verdict.
        "total_trades_eligible": eligible_trades,
        "thin_windows": thin_windows,
        "no_data_windows": no_data_windows,
        "notes": notes,
    }


def cross_window_verdict(agg: dict[str, Any]) -> str:
    eligible = agg.get("n_windows_eligible") or 0
    # Use eligible-window trades for the gate, not the all-windows total.
    # A run with 5 thin windows of 20 trades each isn't 100 evidentiary
    # trades, it's 0.
    eligible_trades = agg.get("total_trades_eligible") or 0
    pct_pos = agg.get("pct_windows_net_positive") or 0.0

    # Need enough eligible windows to even talk about a percentage.
    # With <3 eligible windows the cross-regime claim is unfounded.
    if eligible < 3 or eligible_trades < MIN_TOTAL_TRADES:
        return "INSUFFICIENT_DATA_IN_WINDOW"
    if pct_pos < 50.0:
        return "NO_EDGE_CROSS_WINDOW"
    if pct_pos < ROBUST_PCT_THRESHOLD:
        return "INCONSISTENT_CROSS_WINDOW"
    return "ROBUST_CROSS_WINDOW"


async def _run_one_window(
    *,
    db: Database,
    jvm: JvmClient,
    window: RegimeWindow,
    account_strategy_id: str,
    strategy_code: str,
    instrument: str,
    interval_name: str,
    overrides: dict[str, Any] | None,
    allow_long: bool,
    allow_short: bool,
) -> dict[str, Any]:
    """Submit one window to the JVM, poll, fetch metrics + trades.

    Bug H1 fix: the entire body is wrapped so that *any* exception —
    asyncpg disconnect, KeyError on a malformed metrics row, JVM 5xx —
    becomes a structured error entry rather than aborting the whole
    cross-window run. Walk-forward had the same data-loss footgun;
    cross-window declines to inherit it.
    """
    base = {
        "label": window.label,
        "start": window.start.isoformat(),
        "end": window.end.isoformat(),
        "regime_tag": window.regime_tag,
    }
    try:
        payload = _build_payload(
            account_strategy_id=account_strategy_id,
            strategy_code=strategy_code,
            asset=instrument,
            interval_name=interval_name,
            test_start=window.start,
            test_end=window.end,
            overrides=overrides,
            allow_long=allow_long,
            allow_short=allow_short,
        )
        try:
            run_id = await jvm.submit_backtest(payload)
        except OrchestratorError as e:
            return {**base, "run_id": None, "error": e.error_code}

        final_status = await _poll_backtest_status(db, run_id)
        if final_status != "COMPLETED":
            return {**base, "run_id": run_id, "error": f"status={final_status}"}

        async with db.acquire() as conn:
            metrics = await _fetch_run_metrics(conn, run_id)
            trades = await trades_repo.fetch_trades(conn, run_id)

        slip = slippage_sensitivity(trades, scenarios=(SLIPPAGE_GATE_BPS,))
        slip_net = slip.get(f"+{SLIPPAGE_GATE_BPS}bps")

        def _f(key: str) -> float | None:
            v = metrics.get(key)
            if v is None or v == "":
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        return {
            **base,
            "run_id": run_id,
            "trades": int(metrics.get("total_trades") or 0),
            "pf": _f("profit_factor"),
            "return_pct": _f("return_pct"),
            "sharpe": _f("sharpe_ratio"),
            "win_rate": _f("win_rate"),
            "net_profit": _f("net_profit"),
            "slip_20bps_net": slip_net,
        }
    except Exception as exc:  # noqa: BLE001 — last-line containment
        log.exception(
            "cross_window.window_failed",
            window=window.label,
            strategy_code=strategy_code,
        )
        return {
            **base,
            "run_id": None,
            "error": f"exception:{type(exc).__name__}",
        }


async def run_cross_window(
    *,
    db: Database,
    jvm: JvmClient,
    settings: Settings,  # noqa: ARG001 — reserved for prod-window/capital config
    agent_name: str,
    strategy_code: str,
    interval_name: str,
    instrument: str = "BTCUSDT",
    overrides: dict[str, Any] | None = None,
    motivating_iteration_id: UUID | None = None,
    motivating_walk_forward_id: UUID | None = None,
) -> CrossWindowResult:
    async with db.acquire() as conn:
        as_row = await _resolve_account_strategy(
            conn, strategy_code, settings.research_account_id
        )
        data_min = await _data_min_start(conn, instrument, interval_name)
    if not as_row:
        raise OrchestratorError(
            status_code=412,
            error_code="account_strategy_missing",
            message=f"No account_strategy row for strategy_code={strategy_code}.",
            retryable=False,
            hint="Seed an account_strategy row before requesting cross-window.",
        )
    as_id = as_row["account_strategy_id"]

    catalog = windows_for(instrument)
    if not catalog:
        raise OrchestratorError(
            status_code=500,
            error_code="cross_window_no_catalog",
            message=f"regime_windows catalog is empty for instrument={instrument}.",
            retryable=False,
        )

    if data_min is None:
        raise OrchestratorError(
            status_code=412,
            error_code="market_data_not_plumbed",
            message=(
                f"No market_data rows for instrument={instrument}, "
                f"interval={interval_name}. Backfill before running cross-window."
            ),
            retryable=False,
        )

    # Bug H2 fix: skip windows that fall entirely before data-availability.
    # We mark them as no_data in window_results so the response shows
    # explicitly which windows were unevaluable for this instrument.
    window_results: list[dict[str, Any]] = []
    for idx, window in enumerate(catalog, start=1):
        # Catalog ends are exclusive: data must reach at least one day
        # into the window for the JVM to produce any trades.
        if window.end <= data_min:
            log.info(
                "cross_window.window_skipped_no_data",
                strategy_code=strategy_code,
                window=window.label,
                window_end=window.end.isoformat(),
                data_min=data_min.isoformat(),
            )
            window_results.append({
                "label": window.label,
                "start": window.start.isoformat(),
                "end": window.end.isoformat(),
                "regime_tag": window.regime_tag,
                "run_id": None,
                "error": "no_data",
            })
            continue

        log.info(
            "cross_window.window_start",
            strategy_code=strategy_code,
            window=window.label,
            regime=window.regime_tag,
            idx=idx,
            n=len(catalog),
        )
        result = await _run_one_window(
            db=db,
            jvm=jvm,
            window=window,
            account_strategy_id=as_id,
            strategy_code=strategy_code,
            instrument=instrument,
            interval_name=interval_name,
            overrides=overrides,
            allow_long=as_row["allow_long"],
            allow_short=as_row["allow_short"],
        )
        window_results.append(result)

    agg = aggregate_windows(window_results)
    verdict = cross_window_verdict(agg)

    pf = agg.get("pf") or {}
    ret = agg.get("return_pct") or {}

    async with db.acquire() as conn:
        cw_id = await cw_repo.insert_cross_window(
            conn,
            strategy_code=strategy_code,
            interval_name=interval_name,
            instrument=instrument,
            windows_catalog_version=CATALOG_VERSION,
            n_windows=len(catalog),
            n_windows_completed=agg.get("n_windows_completed") or 0,
            window_pf_mean=pf.get("mean"),
            window_pf_std=pf.get("std"),
            window_pf_min=pf.get("min"),
            window_pf_max=pf.get("max"),
            pct_windows_net_positive=agg.get("pct_windows_net_positive"),
            window_return_mean=ret.get("mean"),
            window_return_std=ret.get("std"),
            total_trades_across_windows=agg.get("total_trades_across_windows") or 0,
            cross_window_verdict=verdict,
            motivating_iteration_id=motivating_iteration_id,
            motivating_walk_forward_id=motivating_walk_forward_id,
            window_results=window_results,
            git_commit_hash=None,
            notes=None,
            created_by=agent_name,
        )

    log.info(
        "cross_window.complete",
        strategy_code=strategy_code,
        cross_window_id=str(cw_id),
        verdict=verdict,
        n_windows=len(catalog),
        pct_positive=agg.get("pct_windows_net_positive"),
    )
    return CrossWindowResult(
        cross_window_id=cw_id,
        cross_window_verdict=verdict,
        windows_catalog_version=CATALOG_VERSION,
        aggregate=agg,
        window_results=window_results,
        motivating_walk_forward_id=motivating_walk_forward_id,
        motivating_iteration_id=motivating_iteration_id,
    )


__all__ = [
    "ROBUST_PCT_THRESHOLD",
    "MIN_TOTAL_TRADES",
    "MIN_TRADES_PER_WINDOW",
    "SLIPPAGE_GATE_BPS",
    "SLIPPAGE_PROXY_NOTE",
    "CrossWindowResult",
    "aggregate_windows",
    "cross_window_verdict",
    "run_cross_window",
]
