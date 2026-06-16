"""Strategy Research Registry — curated roster + live-metrics join (DML).

The curated editorial layer lives in ``strategy_research_registry`` (Flyway
V182). At read time we LEFT JOIN LATERAL onto each curated row:

  * the best completed run's metrics (research_iteration_log -> backtest_run),
    keyed by (strategy_code, symbol, interval_name);
  * the latest walk-forward stability verdict (walk_forward_run);
  * a live-status EXISTS check (account_strategy enabled & not simulated).

Explicit evidence pointers, when set on the curated row, win over the by-key
lookup (ordered first in each lateral). Rows with no ``strategy_code`` (offline
leads) get no live metrics — every lateral is guarded by
``r.strategy_code IS NOT NULL``.

This module is DML-only; the schema is owned by the trading-JVM Flyway repo.
"""
from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# SELECT core: curated columns + lateral live-metric joins. No bind params —
# the laterals correlate on r.* columns only, so this fragment is reused by
# both the list query (filters appended) and the single-row fetch.
# ---------------------------------------------------------------------------
_SELECT = """
SELECT
    r.registry_id::text                 AS registry_id,
    r.slug,
    r.rank,
    r.promise_tier,
    r.display_name,
    r.signal_family,
    r.strategy_code,
    r.symbol,
    r.interval_name,
    r.verdict_tag,
    r.lifecycle_status,
    r.thesis,
    r.detail,
    r.evidence_iteration_id::text       AS evidence_iteration_id,
    r.evidence_walk_forward_id::text    AS evidence_walk_forward_id,
    r.evidence_backtest_run_id::text    AS evidence_backtest_run_id,
    r.journal_id::text                  AS journal_id,
    r.memory_ref,
    r.is_offline_lead,
    r.archived,
    r.auto_managed,
    r.created_time,
    r.updated_time,
    r.updated_by,
    m.dsr,
    m.psr,
    m.ann_return_pct,
    m.sharpe_ann,
    m.n_trades,
    m.profit_factor,
    m.statistical_verdict,
    m.iteration_id::text                AS live_iteration_id,
    m.backtest_run_id::text             AS live_backtest_run_id,
    CASE
        WHEN m.iteration_id IS NULL                       THEN NULL
        WHEN m.iteration_id = r.evidence_iteration_id     THEN 'pointer'
        ELSE 'lookup'
    END                                 AS resolved_from,
    w.stability_verdict                 AS walk_forward_verdict,
    COALESCE(al.is_live, FALSE)         AS is_live
FROM strategy_research_registry r
LEFT JOIN LATERAL (
    SELECT
        ril.iteration_id,
        br.backtest_run_id,
        ril.statistical_verdict,
        (ril.metrics_snapshot->'analysis'->>'dsr')::NUMERIC                                          AS dsr,
        (ril.metrics_snapshot->'analysis'->>'psr')::NUMERIC                                          AS psr,
        (ril.metrics_snapshot->'analysis'->>'annualized_geometric_return_pct_at_alloc_90')::NUMERIC  AS ann_return_pct,
        (ril.metrics_snapshot->'analysis'->>'sharpe_annualized')::NUMERIC                            AS sharpe_ann,
        (ril.metrics_snapshot->'analysis'->>'n_trades')::INT                                         AS n_trades,
        (ril.metrics_snapshot->'analysis'->>'pf_point_estimate')::NUMERIC                            AS profit_factor
    FROM research_iteration_log ril
    JOIN backtest_run br ON br.backtest_run_id = ril.backtest_run_id
    WHERE r.strategy_code IS NOT NULL
      AND ril.strategy_code = r.strategy_code
      AND (r.symbol IS NULL OR br.asset = r.symbol)
      AND (r.interval_name IS NULL OR br.interval_name = r.interval_name)
      AND br.status = 'COMPLETED'
      AND ril.verdict <> 'FAILED'
      AND (ril.metrics_snapshot->'analysis'->>'annualized_geometric_return_pct_at_alloc_90') IS NOT NULL
    ORDER BY
        -- An explicit evidence pointer wins; otherwise the LATEST completed run
        -- (the strategy's CURRENT condition), NOT the best-ever cell — a "best by
        -- return" pick would surface a stale over-fit after a live param swap.
        (ril.iteration_id = r.evidence_iteration_id) DESC NULLS LAST,
        ril.created_time DESC NULLS LAST,
        br.backtest_run_id DESC
    LIMIT 1
) m ON TRUE
LEFT JOIN LATERAL (
    SELECT wfr.stability_verdict
    FROM walk_forward_run wfr
    WHERE r.strategy_code IS NOT NULL
      AND wfr.strategy_code = r.strategy_code
      AND (r.symbol IS NULL OR wfr.instrument = r.symbol)
      AND (r.interval_name IS NULL OR wfr.interval_name = r.interval_name)
    ORDER BY
        (wfr.walk_forward_id = r.evidence_walk_forward_id) DESC NULLS LAST,
        wfr.created_time DESC NULLS LAST
    LIMIT 1
) w ON TRUE
LEFT JOIN LATERAL (
    -- "Live" = an enabled, non-simulated, non-deleted account_strategy at the
    -- SAME interval. Interval matters: DCB is live at 1h but falsified at 4h —
    -- without the interval match the 4h row would falsely report is_live (and
    -- trip a spurious "falsified but live" divergence) off the 1h live row.
    SELECT TRUE AS is_live
    FROM account_strategy a
    WHERE r.strategy_code IS NOT NULL
      AND a.strategy_code = r.strategy_code
      AND (r.symbol IS NULL OR a.symbol = r.symbol)
      AND (r.interval_name IS NULL OR a.interval_name = r.interval_name)
      AND a.enabled = TRUE
      AND a.simulated = FALSE
      AND COALESCE(a.is_deleted, FALSE) = FALSE
    LIMIT 1
) al ON TRUE
"""

_LIST_WHERE_ORDER = """
WHERE (r.archived = FALSE OR $1::boolean = TRUE)
  AND ($2::text IS NULL OR r.promise_tier = $2)
  AND ($3::text IS NULL OR r.lifecycle_status = $3)
  AND ($4::text IS NULL OR r.signal_family = $4)
  AND ($5::text IS NULL OR r.display_name ILIKE '%' || $5 || '%'
                        OR r.strategy_code ILIKE '%' || $5 || '%'
                        OR r.thesis ILIKE '%' || $5 || '%')
ORDER BY
    CASE r.promise_tier WHEN 'TIER_A' THEN 0 WHEN 'TIER_B' THEN 1 ELSE 2 END,
    r.rank NULLS LAST,
    r.display_name
"""

# Editorial columns an admin may set. Column names are whitelisted here (never
# taken from request keys) so the dynamic UPDATE below is injection-safe.
_UPDATABLE: frozenset[str] = frozenset({
    "slug", "rank", "promise_tier", "display_name", "signal_family",
    "strategy_code", "symbol", "interval_name", "verdict_tag",
    "lifecycle_status", "thesis", "detail", "evidence_iteration_id",
    "evidence_walk_forward_id", "evidence_backtest_run_id", "journal_id",
    "memory_ref", "is_offline_lead", "archived", "auto_managed",
})
_UUID_FIELDS: frozenset[str] = frozenset({
    "evidence_iteration_id", "evidence_walk_forward_id",
    "evidence_backtest_run_id", "journal_id",
})

_INSERT = """
INSERT INTO strategy_research_registry
    (slug, rank, promise_tier, display_name, signal_family, strategy_code,
     symbol, interval_name, verdict_tag, lifecycle_status, thesis, detail,
     evidence_iteration_id, evidence_walk_forward_id, evidence_backtest_run_id,
     journal_id, memory_ref, is_offline_lead, auto_managed, updated_by)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12,
        $13::uuid, $14::uuid, $15::uuid, $16::uuid, $17, $18, $19, $20)
RETURNING registry_id::text
"""


async def list_registry(
    conn,
    *,
    tier: str | None = None,
    status: str | None = None,
    family: str | None = None,
    search: str | None = None,
    include_archived: bool = False,
) -> list[dict[str, Any]]:
    rows = await conn.fetch(
        _SELECT + _LIST_WHERE_ORDER, include_archived, tier, status, family, search
    )
    return [dict(r) for r in rows]


async def get_registry(conn, registry_id: str) -> dict[str, Any] | None:
    row = await conn.fetchrow(_SELECT + " WHERE r.registry_id = $1::uuid", registry_id)
    return dict(row) if row is not None else None


async def get_registry_by_slug(conn, slug: str) -> dict[str, Any] | None:
    """Fetch the merged (curated + live) row by its stable slug — the key the
    agent addresses entries by (it never sees registry_id UUIDs)."""
    row = await conn.fetchrow(_SELECT + " WHERE r.slug = $1", slug)
    return dict(row) if row is not None else None


async def insert_registry(
    conn, fields: dict[str, Any], updated_by: str, *, auto_managed: bool = False
) -> dict[str, Any] | None:
    registry_id = await conn.fetchval(
        _INSERT,
        fields["slug"],
        fields.get("rank"),
        fields["promise_tier"],
        fields["display_name"],
        fields.get("signal_family"),
        fields.get("strategy_code"),
        fields.get("symbol"),
        fields.get("interval_name"),
        fields["verdict_tag"],
        fields["lifecycle_status"],
        fields["thesis"],
        fields.get("detail"),
        fields.get("evidence_iteration_id"),
        fields.get("evidence_walk_forward_id"),
        fields.get("evidence_backtest_run_id"),
        fields.get("journal_id"),
        fields.get("memory_ref"),
        bool(fields.get("is_offline_lead", False)),
        auto_managed,
        updated_by,
    )
    return await get_registry(conn, registry_id)


async def find_registry_by_research_key(
    conn, strategy_code: str, symbol: str | None, interval: str | None
) -> dict[str, Any] | None:
    """Find the registry row that COVERS a (strategy_code, symbol, interval)
    research result — NULL-tolerant: a curated row with NULL symbol/interval
    (a universe strategy) covers any symbol/interval for that code. Exact matches
    are preferred. Used by the auto-sync to decide create-vs-update-vs-skip
    without spawning per-symbol duplicates of a universe entry.
    """
    row = await conn.fetchrow(
        """
        SELECT registry_id::text AS registry_id, auto_managed, archived, thesis
        FROM strategy_research_registry
        WHERE strategy_code = $1
          AND (symbol IS NULL OR symbol = $2)
          AND (interval_name IS NULL OR interval_name = $3)
        ORDER BY (symbol = $2) DESC NULLS LAST,
                 (interval_name = $3) DESC NULLS LAST
        LIMIT 1
        """,
        strategy_code, symbol, interval,
    )
    return dict(row) if row is not None else None


async def update_registry(
    conn, registry_id: str, patch: dict[str, Any], updated_by: str
) -> dict[str, Any] | None:
    set_clauses: list[str] = []
    values: list[Any] = []
    idx = 1
    for key, value in patch.items():
        if key not in _UPDATABLE:
            continue
        cast = "::uuid" if key in _UUID_FIELDS else ""
        set_clauses.append(f"{key} = ${idx}{cast}")
        values.append(value)
        idx += 1

    if not set_clauses:
        # Nothing updatable supplied — treat as a no-op read so the caller still
        # gets the (possibly-unchanged) merged row, or None if the id is unknown.
        return await get_registry(conn, registry_id)

    set_clauses.append(f"updated_by = ${idx}")
    values.append(updated_by)
    idx += 1
    set_clauses.append("updated_time = now()")
    values.append(registry_id)

    sql = (
        "UPDATE strategy_research_registry SET "
        + ", ".join(set_clauses)
        + f" WHERE registry_id = ${idx}::uuid RETURNING registry_id::text"
    )
    updated_id = await conn.fetchval(sql, *values)
    if updated_id is None:
        return None
    return await get_registry(conn, registry_id)


async def archive_registry(conn, registry_id: str, updated_by: str) -> bool:
    updated_id = await conn.fetchval(
        "UPDATE strategy_research_registry "
        "SET archived = TRUE, updated_by = $1, updated_time = now() "
        "WHERE registry_id = $2::uuid RETURNING registry_id::text",
        updated_by,
        registry_id,
    )
    return updated_id is not None
