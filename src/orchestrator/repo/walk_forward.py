"""``walk_forward_run`` reads and writes.

Schema reference: V12__create_walk_forward_run.sql (definition inlined in
V1__baseline.sql). The CHECK constraint requires ``stability_verdict`` ∈
{ROBUST, INCONSISTENT, INSUFFICIENT_EVIDENCE, OVERFIT, NO_EDGE}; the caller
is responsible for computing it.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import asyncpg


async def insert_walk_forward(
    conn: asyncpg.Connection,
    *,
    strategy_code: str,
    interval_name: str,
    instrument: str,
    full_start: Any,
    full_end: Any,
    train_months: int,
    test_months: int,
    n_folds: int,
    fold_pf_mean: float | None,
    fold_pf_std: float | None,
    fold_pf_min: float | None,
    fold_pf_max: float | None,
    fold_pf_positive_pct: float | None,
    fold_return_mean: float | None,
    fold_return_std: float | None,
    fold_sharpe_mean: float | None,
    fold_sharpe_std: float | None,
    total_trades_across_folds: int,
    stability_verdict: str,
    motivating_iteration_id: UUID | None,
    fold_results: list[dict[str, Any]],
    git_commit_hash: str | None,
    notes: str | None,
    created_by: str,
) -> UUID:
    walk_forward_id = await conn.fetchval(
        """
        INSERT INTO walk_forward_run (
            walk_forward_id, strategy_code, interval_name, instrument,
            full_window_start, full_window_end, train_months, test_months, n_folds,
            fold_pf_mean, fold_pf_std, fold_pf_min, fold_pf_max, fold_pf_positive_pct,
            fold_return_mean, fold_return_std,
            fold_sharpe_mean, fold_sharpe_std,
            total_trades_across_folds, stability_verdict,
            motivating_iteration_id, fold_results, git_commit_hash,
            notes, created_time, created_by
        ) VALUES (
            gen_random_uuid(), $1, $2, $3,
            $4, $5, $6, $7, $8,
            $9, $10, $11, $12, $13,
            $14, $15, $16, $17,
            $18, $19,
            $20, $21, $22,
            $23, NOW(), $24
        )
        RETURNING walk_forward_id
        """,
        strategy_code,
        interval_name,
        instrument,
        full_start,
        full_end,
        train_months,
        test_months,
        n_folds,
        fold_pf_mean,
        fold_pf_std,
        fold_pf_min,
        fold_pf_max,
        fold_pf_positive_pct,
        fold_return_mean,
        fold_return_std,
        fold_sharpe_mean,
        fold_sharpe_std,
        total_trades_across_folds,
        stability_verdict,
        motivating_iteration_id,
        fold_results,
        git_commit_hash,
        notes,
        created_by,
    )
    return walk_forward_id


async def fetch_fold_daily_returns(
    conn: asyncpg.Connection, backtest_run_id: UUID
) -> list[float]:
    """Return the ordered daily-return series for one backtest run.

    Reads ``backtest_equity_point.daily_return_pct`` (already computed by
    the JVM) ordered by ``equity_date``. Used by the frequency-aware
    (low-turnover) walk-forward track, which validates on the out-of-sample
    daily-equity return series rather than trade-count fold statistics — a
    1d strategy yields ~1 return/bar (hundreds of obs) even when it trades
    only a handful of times. NULL returns are skipped (first bar carries a
    NULL daily_return_pct). Values are kept in percent units, matching how
    the JVM stores them.
    """
    rows = await conn.fetch(
        """
        SELECT daily_return_pct
          FROM backtest_equity_point
         WHERE backtest_run_id = $1
           AND daily_return_pct IS NOT NULL
         ORDER BY equity_date ASC
        """,
        backtest_run_id,
    )
    return [float(r["daily_return_pct"]) for r in rows]


async def fetch_walk_forward_run(
    conn: asyncpg.Connection, walk_forward_id: UUID
) -> dict[str, Any] | None:
    """Return a curator-relevant subset of ``walk_forward_run`` by PK.

    Returns ``None`` when no matching row exists.
    """
    row = await conn.fetchrow(
        """
        SELECT walk_forward_id,
               strategy_code,
               instrument,
               interval_name,
               stability_verdict,
               n_folds,
               fold_pf_mean,
               fold_pf_std,
               fold_pf_min,
               fold_pf_max,
               fold_pf_positive_pct,
               fold_sharpe_mean,
               fold_sharpe_std,
               fold_return_mean,
               fold_return_std,
               total_trades_across_folds,
               fold_results,
               motivating_iteration_id,
               created_time,
               created_by
          FROM walk_forward_run
         WHERE walk_forward_id = $1
        """,
        walk_forward_id,
    )
    return dict(row) if row else None


async def fetch_daily_returns_dated(
    conn: asyncpg.Connection, backtest_run_id: UUID
) -> list[tuple[Any, float]]:
    """Dated variant of ``fetch_fold_daily_returns`` — returns
    ``[(equity_date, daily_return_pct), ...]`` ordered by date.

    Used by the pooled-certification slice-equity combiner
    (``services/pooled_certification.combine_slice_equity``), which must
    ALIGN sleeve equity curves by calendar date before averaging them into
    a book curve; the value-only variant cannot express idle-slice days.
    NULL returns are skipped (first bar carries NULL). Percent units,
    matching the JVM storage.
    """
    rows = await conn.fetch(
        """
        SELECT equity_date, daily_return_pct
          FROM backtest_equity_point
         WHERE backtest_run_id = $1
           AND daily_return_pct IS NOT NULL
         ORDER BY equity_date ASC
        """,
        backtest_run_id,
    )
    return [(r["equity_date"], float(r["daily_return_pct"])) for r in rows]
