"""asyncpg query backing ``GET /ml/monitor``.

One row per ``signal_definition`` whose ``status != 'retired'`` AND which
is bound to at least one ``account_strategy`` row with
``ml_regime_gate_enabled = TRUE``. The aggregation pulls everything the
dashboard strip + monitor table need in a single round-trip so the
handler stays a thin transformer.

Schema (V66, V79):
* ``signal_definition.name``   — used as the signal's display name
* ``signal_definition.status`` — active | shadow | retired
* ``model_registry.symbol`` / ``interval`` — the model's training symbol
* ``model_registry.metrics`` (JSONB) — may carry ``walkforward_auc_mean``
* ``signal_history.ts``        — bar (open-label) timestamp; drives coverage + last fire
* ``signal_history.produced_at`` — wall-clock write time; drives "last written" (pipeline liveness)
* ``account_strategy.ml_regime_signal_name`` / ``ml_regime_gate_enabled``
  — the binding edge between strategy and signal

The handler ``api/ml_monitor.py`` then derives ``health`` (green/amber/red)
plus the strategy's expected fire cadence (in seconds) from
``model_registry.interval`` — the SQL stays declarative.
"""
from __future__ import annotations

from typing import Any


_MONITOR_SQL = """
WITH bound_signals AS (
    SELECT DISTINCT
        sd.signal_id,
        sd.name        AS signal_name,
        sd.model_id,
        sd.status,
        sd.created_time
    FROM signal_definition sd
    WHERE sd.status <> 'retired'
      AND EXISTS (
          SELECT 1 FROM account_strategy ac
          WHERE ac.ml_regime_signal_name = sd.name
            AND ac.ml_regime_gate_enabled = TRUE
      )
),
bound_strategies AS (
    SELECT
        sd.signal_id,
        array_agg(DISTINCT ac.strategy_code ORDER BY ac.strategy_code) AS strategy_codes
    FROM signal_definition sd
    JOIN account_strategy ac
      ON ac.ml_regime_signal_name = sd.name
    WHERE sd.signal_id IN (SELECT signal_id FROM bound_signals)
    GROUP BY sd.signal_id
),
fires_30d AS (
    SELECT
        signal_id,
        MAX(ts)                                                              AS last_fire_ts,
        MAX(produced_at)                                                     AS last_produced_at,
        COUNT(*) FILTER (WHERE ts > NOW() - INTERVAL '24 hours')              AS fires_24h,
        COUNT(*) FILTER (WHERE ts > NOW() - INTERVAL '7 days')                AS fires_7d
    FROM signal_history
    WHERE signal_id IN (SELECT signal_id FROM bound_signals)
      AND ts        >  NOW() - INTERVAL '30 days'
    GROUP BY signal_id
),
feature_max_ts AS (
    SELECT
        bs.signal_id,
        MAX(fv.ts) AS latest_feature_ts
    FROM bound_signals bs
    JOIN model_registry mr ON mr.id = bs.model_id
    JOIN feature_values fv
        ON fv.symbol   = mr.symbol
       AND fv.interval = COALESCE(mr.serving_interval, mr.interval)
    GROUP BY bs.signal_id
)
SELECT
    bs.signal_id,
    bs.signal_name,
    bs.status,
    bs.created_time,
    mr.family                                  AS model_family,
    mr.purpose                                 AS model_purpose,
    mr.symbol                                  AS model_symbol,
    mr.interval                                AS model_interval,
    mr.metrics                                 AS model_metrics,
    COALESCE(bst.strategy_codes, ARRAY[]::text[]) AS bound_strategy_codes,
    fr.last_fire_ts,
    fr.last_produced_at,
    COALESCE(fr.fires_24h, 0)                  AS fires_24h,
    COALESCE(fr.fires_7d,  0)                  AS fires_7d,
    fmt.latest_feature_ts
FROM bound_signals bs
LEFT JOIN model_registry  mr  ON mr.id        = bs.model_id
LEFT JOIN bound_strategies bst ON bst.signal_id = bs.signal_id
LEFT JOIN fires_30d        fr  ON fr.signal_id  = bs.signal_id
LEFT JOIN feature_max_ts   fmt ON fmt.signal_id = bs.signal_id
ORDER BY bs.signal_name;
"""


async def fetch_monitor_rows(conn) -> list[dict[str, Any]]:
    """Run the rollup. Returns raw dicts; the API layer adds derived
    fields (``expected_fire_seconds``, ``health``, ``health_reason``)."""
    records = await conn.fetch(_MONITOR_SQL)
    return [dict(r) for r in records]
