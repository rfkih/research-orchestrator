"""Strategy profitability ranking query.

Joins ``research_iteration_log`` (analyzed metrics + params) against
``backtest_run`` (window, symbol, interval).  For each
``(strategy_code, asset, interval_name)`` tuple we pick the single best
run (highest annualized geometric return at 90% sizing) then rank all
tuples globally by that metric.

``min_days`` filters out short backtests that would inflate the
annualised number — default 365 (one full year).  The caller can lower
it when the platform doesn't yet have long history for a symbol.
"""
from __future__ import annotations

from typing import Any

_RANKING_SQL = """
WITH best_per_combo AS (
    SELECT DISTINCT ON (ril.strategy_code, br.asset, br.interval_name)
        ril.strategy_code,
        br.asset                                                      AS symbol,
        br.interval_name,
        ril.verdict,
        ril.statistical_verdict,
        ril.params_snapshot,
        ril.metrics_snapshot,
        ril.confidence_intervals,
        br.backtest_run_id,
        br.start_time,
        br.end_time,
        br.initial_capital,
        br.effective_params_snapshot,
        EXTRACT(DAY FROM (br.end_time - br.start_time))::INT         AS window_days,
        -- Analyzer-computed metrics live nested under metrics_snapshot->'analysis'
        -- (tick.py builds metrics_snapshot = {**run_metrics, "analysis": analysis}).
        -- Reading them at the top level returned NULL for every row, so the
        -- WHERE/ORDER below excluded every row and this endpoint returned [].
        -- win_rate is the exception: it is a run-level column, only top-level.
        (ril.metrics_snapshot->'analysis'->>'annualized_geometric_return_pct_at_alloc_90')::NUMERIC
                                                                      AS ann_return_pct,
        (ril.metrics_snapshot->'analysis'->>'sharpe_annualized')::NUMERIC AS sharpe_ann,
        (ril.metrics_snapshot->'analysis'->>'max_drawdown_pct')::NUMERIC  AS max_drawdown_pct,
        (ril.metrics_snapshot->'analysis'->>'n_trades')::INT             AS n_trades,
        (ril.metrics_snapshot->'analysis'->>'pf_point_estimate')::NUMERIC AS profit_factor,
        (ril.metrics_snapshot->'analysis'->>'dsr')::NUMERIC             AS dsr,
        (ril.metrics_snapshot->'analysis'->>'psr')::NUMERIC             AS psr,
        (ril.metrics_snapshot->>'win_rate')::NUMERIC                  AS win_rate
    FROM research_iteration_log ril
    JOIN backtest_run br ON br.backtest_run_id = ril.backtest_run_id
    WHERE br.status = 'COMPLETED'
      AND ril.verdict NOT IN ('FAILED', 'DISCARD')
      AND EXTRACT(DAY FROM (br.end_time - br.start_time)) >= $1
      AND (ril.metrics_snapshot->'analysis'->>'annualized_geometric_return_pct_at_alloc_90') IS NOT NULL
    ORDER BY
        ril.strategy_code,
        br.asset,
        br.interval_name,
        (ril.metrics_snapshot->'analysis'->>'annualized_geometric_return_pct_at_alloc_90')::NUMERIC DESC NULLS LAST
)
SELECT
    ROW_NUMBER() OVER (ORDER BY ann_return_pct DESC NULLS LAST)::INT AS rank,
    strategy_code,
    symbol,
    interval_name,
    verdict,
    statistical_verdict,
    params_snapshot,
    metrics_snapshot,
    confidence_intervals,
    backtest_run_id::TEXT,
    start_time,
    end_time,
    initial_capital,
    effective_params_snapshot,
    window_days,
    ann_return_pct,
    sharpe_ann,
    max_drawdown_pct,
    n_trades,
    profit_factor,
    dsr,
    psr,
    win_rate
FROM best_per_combo
ORDER BY ann_return_pct DESC NULLS LAST
LIMIT $2
"""


async def fetch_rankings(
    conn,
    min_days: int = 365,
    limit: int = 50,
) -> list[dict[str, Any]]:
    rows = await conn.fetch(_RANKING_SQL, min_days, limit)
    return [dict(r) for r in rows]
