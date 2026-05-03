"""Phase 7.5 — PROMOTED verdict drift scan.

For every PROMOTED account_strategy (enabled AND not simulated AND not
soft-deleted), pull the two most-recent ``cross_window_run`` rows by
(strategy_code, interval_name, symbol) and report cases where the
*current* verdict ranks below the prior baseline.

Verdict rank ordering — higher is healthier::

    NO_EDGE_CROSS_WINDOW          0
    INSUFFICIENT_DATA_IN_WINDOW   1
    INCONSISTENT_CROSS_WINDOW     2
    ROBUST_CROSS_WINDOW           3

The trading JVM calls this nightly via the watchdog scheduler, then
raises ``VERDICT_DRIFT`` alerts via its local ``AlertService`` for each
strategy that dropped a rank or worse. We intentionally do **not** kick
off fresh ``cross_window`` runs here — those take ~3 hours each and
belong in the agent's normal research loop. This scan is a comparator
over whatever the agent has most recently produced.

A strategy with fewer than two cross_window rows is silently skipped:
no baseline = no comparison.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from ..infra.db import Database
from ..logging import get_logger
from ..repo import cross_window as cw_repo

log = get_logger(__name__)


VERDICT_RANK = {
    "NO_EDGE_CROSS_WINDOW": 0,
    "INSUFFICIENT_DATA_IN_WINDOW": 1,
    "INCONSISTENT_CROSS_WINDOW": 2,
    "ROBUST_CROSS_WINDOW": 3,
}


async def _fetch_promoted(conn: asyncpg.Connection) -> list[dict[str, Any]]:
    rows = await conn.fetch(
        """
        SELECT account_strategy_id, strategy_code, interval_name, symbol
        FROM account_strategy
        WHERE enabled = true
          AND simulated = false
          AND is_deleted = false
        """
    )
    return [dict(r) for r in rows]


async def scan_promoted(db: Database) -> list[dict[str, Any]]:
    """Returns one entry per PROMOTED strategy whose current
    cross-window verdict has dropped below its prior baseline.

    Result entries have the shape consumed by the JVM
    ``VerdictDriftScanService``:

        {
          "account_strategy_id": str,
          "strategy_code": str,
          "interval_name": str,
          "instrument": str,
          "baseline_verdict": str,
          "current_verdict": str,
          "rank_drop": int,        # baseline_rank - current_rank, ≥1
          "baseline_created_time": str (ISO),
          "current_created_time": str (ISO),
        }

    Strategies with <2 cross_window rows are silently skipped.
    Unknown verdict strings are treated as rank 0 (worst) — better to
    over-alert on a misspelt verdict than to silently ignore drift.
    """
    drifts: list[dict[str, Any]] = []
    async with db.acquire() as conn:
        promoted = await _fetch_promoted(conn)
        log.info(
            "verdict_drift.scan_started",
            extra={"promoted_count": len(promoted)},
        )
        for s in promoted:
            strategy_code = s["strategy_code"]
            interval_name = s["interval_name"]
            instrument = s["symbol"]
            # symbol is nullable in account_strategy; passing NULL through
            # the asyncpg `=` comparison silently returns zero rows. Skip
            # with a warn so the operator sees the data-shape problem
            # instead of "no drift detected" masking it.
            if not strategy_code or not interval_name or not instrument:
                log.warn(
                    "verdict_drift.skipping_incomplete_row",
                    extra={
                        "account_strategy_id": str(s["account_strategy_id"]),
                        "strategy_code": strategy_code,
                        "interval_name": interval_name,
                        "instrument": instrument,
                    },
                )
                continue
            try:
                rows = await cw_repo.fetch_recent_two(
                    conn,
                    strategy_code=strategy_code,
                    interval_name=interval_name,
                    instrument=instrument,
                )
            except Exception:
                log.exception(
                    "verdict_drift.fetch_failed",
                    extra={
                        "strategy_code": strategy_code,
                        "interval_name": interval_name,
                        "instrument": instrument,
                    },
                )
                continue

            if len(rows) < 2:
                continue

            current = rows[0]
            baseline = rows[1]
            current_v = current["cross_window_verdict"]
            baseline_v = baseline["cross_window_verdict"]
            current_r = VERDICT_RANK.get(current_v, 0)
            baseline_r = VERDICT_RANK.get(baseline_v, 0)
            if current_r >= baseline_r:
                continue

            drifts.append({
                "account_strategy_id": str(s["account_strategy_id"]),
                "strategy_code": strategy_code,
                "interval_name": interval_name,
                "instrument": instrument,
                "baseline_verdict": baseline_v,
                "current_verdict": current_v,
                "rank_drop": baseline_r - current_r,
                "baseline_created_time": baseline["created_time"].isoformat(),
                "current_created_time": current["created_time"].isoformat(),
            })

    log.info(
        "verdict_drift.scan_complete",
        extra={"drift_count": len(drifts)},
    )
    return drifts
