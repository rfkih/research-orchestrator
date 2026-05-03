"""``cross_window_run`` writes — single INSERT after all windows complete.

Schema reference: V38__create_cross_window_run.sql. The CHECK constraint
requires ``cross_window_verdict`` ∈ {ROBUST_CROSS_WINDOW,
INCONSISTENT_CROSS_WINDOW, NO_EDGE_CROSS_WINDOW,
INSUFFICIENT_DATA_IN_WINDOW}; the caller computes it.

Reads added for Phase 7.5 verdict-drift scan: pull the two most-recent
runs per (strategy_code, interval_name, instrument) so the alert service
can compare a fresh run against the previous baseline.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import asyncpg


async def insert_cross_window(
    conn: asyncpg.Connection,
    *,
    strategy_code: str,
    interval_name: str,
    instrument: str,
    windows_catalog_version: str,
    n_windows: int,
    n_windows_completed: int,
    window_pf_mean: float | None,
    window_pf_std: float | None,
    window_pf_min: float | None,
    window_pf_max: float | None,
    pct_windows_net_positive: float | None,
    window_return_mean: float | None,
    window_return_std: float | None,
    total_trades_across_windows: int,
    cross_window_verdict: str,
    motivating_iteration_id: UUID | None,
    motivating_walk_forward_id: UUID | None,
    window_results: list[dict[str, Any]],
    git_commit_hash: str | None,
    notes: str | None,
    created_by: str,
) -> UUID:
    cross_window_id = await conn.fetchval(
        """
        INSERT INTO cross_window_run (
            cross_window_id, strategy_code, interval_name, instrument,
            windows_catalog_version, n_windows, n_windows_completed,
            window_pf_mean, window_pf_std, window_pf_min, window_pf_max,
            pct_windows_net_positive,
            window_return_mean, window_return_std,
            total_trades_across_windows, cross_window_verdict,
            motivating_iteration_id, motivating_walk_forward_id,
            window_results, git_commit_hash,
            notes, created_time, created_by
        ) VALUES (
            gen_random_uuid(), $1, $2, $3,
            $4, $5, $6,
            $7, $8, $9, $10,
            $11,
            $12, $13,
            $14, $15,
            $16, $17,
            $18, $19,
            $20, NOW(), $21
        )
        RETURNING cross_window_id
        """,
        strategy_code,
        interval_name,
        instrument,
        windows_catalog_version,
        n_windows,
        n_windows_completed,
        window_pf_mean,
        window_pf_std,
        window_pf_min,
        window_pf_max,
        pct_windows_net_positive,
        window_return_mean,
        window_return_std,
        total_trades_across_windows,
        cross_window_verdict,
        motivating_iteration_id,
        motivating_walk_forward_id,
        window_results,
        git_commit_hash,
        notes,
        created_by,
    )
    return cross_window_id


async def fetch_recent_two(
    conn: asyncpg.Connection,
    *,
    strategy_code: str,
    interval_name: str,
    instrument: str,
) -> list[dict[str, Any]]:
    """Returns the two most recent ``cross_window_run`` rows for the tuple,
    ordered by ``created_time DESC`` (so element [0] is the current run and
    element [1] is the prior baseline).

    May return 0, 1, or 2 rows. Drift detection treats <2 rows as
    "no comparison available" — never alerts on a missing baseline.
    """
    rows = await conn.fetch(
        """
        SELECT cross_window_id, strategy_code, interval_name, instrument,
               cross_window_verdict, n_windows, n_windows_completed,
               pct_windows_net_positive, total_trades_across_windows,
               created_time
        FROM cross_window_run
        WHERE strategy_code = $1
          AND interval_name = $2
          AND instrument = $3
        ORDER BY created_time DESC
        LIMIT 2
        """,
        strategy_code,
        interval_name,
        instrument,
    )
    return [dict(r) for r in rows]
