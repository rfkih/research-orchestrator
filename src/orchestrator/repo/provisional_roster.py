"""Candidate pool for the provisional trading-roster floor (2026-06-09).

Returns TRADING-track SIGNIFICANT_EDGE near-misses with the metrics the
``services/provisional_roster`` ranker needs (annualised return, DSR, latest
walk-forward verdict + fold-positive%). The service applies eligibility +
balanced-composite ranking; this query just assembles the evidence.

Track resolution: an iteration's track lives on the originating
``research_queue`` row (``sweep_config->>'track'``), reachable via
``hypothesis_audit.queue_id``. When that's absent (legacy/untagged), a
``hedging_gate`` block in metrics_snapshot marks a hedge; everything else is
trading. We filter to trading here so hedges never reach the trading roster.
"""

from __future__ import annotations

from typing import Any

import asyncpg

_CANDIDATE_SQL = """
WITH cand AS (
    SELECT DISTINCT ON (ril.iteration_id)
        ril.iteration_id::TEXT                                          AS iteration_id,
        br.backtest_run_id::TEXT                                        AS backtest_run_id,
        ril.strategy_code,
        br.asset                                                        AS symbol,
        br.interval_name,
        ril.statistical_verdict,
        (ril.metrics_snapshot->'analysis'->>'annualized_geometric_return_pct_at_alloc_90')::NUMERIC
                                                                        AS ann_return_pct,
        (ril.metrics_snapshot->'analysis'->>'dsr')::NUMERIC             AS dsr,
        (ril.metrics_snapshot->'analysis'->>'psr')::NUMERIC            AS psr,
        (ril.metrics_snapshot->'analysis'->>'max_drawdown_pct')::NUMERIC AS max_drawdown_pct,
        (ril.metrics_snapshot->'analysis'->>'n_trades')::INT            AS n_trades,
        COALESCE(
            rq.sweep_config->>'track',
            CASE WHEN ril.metrics_snapshot ? 'hedging_gate' THEN 'hedging' ELSE 'trading' END
        )                                                               AS track,
        wf.stability_verdict                                            AS walk_forward_verdict,
        (wf.fold_pf_positive_pct)::NUMERIC                              AS fold_positive_pct,
        ril.created_time
    FROM research_iteration_log ril
    JOIN backtest_run br ON br.backtest_run_id = ril.backtest_run_id
    LEFT JOIN hypothesis_audit ha ON ha.iteration_id = ril.iteration_id
    LEFT JOIN research_queue rq ON rq.queue_id = ha.queue_id
    LEFT JOIN LATERAL (
        SELECT w.stability_verdict, w.fold_pf_positive_pct
        FROM walk_forward_run w
        WHERE w.motivating_iteration_id = ril.iteration_id
        ORDER BY w.created_time DESC
        LIMIT 1
    ) wf ON TRUE
    WHERE br.status = 'COMPLETED'
      AND ril.statistical_verdict = 'SIGNIFICANT_EDGE'
      AND ril.verdict NOT IN ('FAILED', 'DISCARD')
      AND (ril.metrics_snapshot->'analysis'->>'annualized_geometric_return_pct_at_alloc_90') IS NOT NULL
    -- DISTINCT ON dedupes the hypothesis_audit LEFT-JOIN fan-out (an iteration
    -- can have multiple audit rows). The ORDER BY must lead with iteration_id;
    -- the tie-break picks the most recent audit row deterministically so the
    -- resolved track / queue linkage is stable, not arbitrary.
    ORDER BY ril.iteration_id, ha.created_time DESC NULLS LAST
)
SELECT *
FROM cand
WHERE track = 'trading'
  AND ann_return_pct > 0
ORDER BY ann_return_pct DESC NULLS LAST
LIMIT $1
"""


async def list_provisional_candidates(
    conn: asyncpg.Connection, *, limit: int = 200
) -> list[dict[str, Any]]:
    """Trading-track SIGNIFICANT_EDGE, positive-return near-misses, best-return
    first. The service re-ranks by the balanced composite and de-dupes by
    strategy_code, so we return ALL qualifying cells (not one-per-strategy) up
    to ``limit`` and let the ranker pick the best cell per strategy by COMPOSITE
    rather than by raw return."""
    rows = await conn.fetch(_CANDIDATE_SQL, limit)
    return [dict(r) for r in rows]
