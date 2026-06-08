"""``research_paper`` + ``research_paper_trade_data`` reads and writes.

Also exposes the supporting queries (queue, iterations, backtest_run,
journal) that ``services/paper_generator.py`` needs so the generator
never touches another repo directly.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

import asyncpg


# ---------------------------------------------------------------------------
# research_paper
# ---------------------------------------------------------------------------

async def get_paper(
    conn: asyncpg.Connection,
    paper_id: str,
) -> dict[str, Any] | None:
    row = await conn.fetchrow(
        """
        SELECT paper_id, queue_id, title, abstract, paper_status, version,
               win_rate, annualized_return_pct, profit_factor, n_trades,
               max_drawdown_pct, sharpe_ratio,
               sections_cache, created_time, created_by, updated_time, updated_by
        FROM research_paper
        WHERE paper_id = $1
        """,
        paper_id,
    )
    return dict(row) if row else None


_SORT_COLUMN_MAP = {
    "annualized_return_pct": "rp.annualized_return_pct",
    "profit_factor": "rp.profit_factor",
    "win_rate": "rp.win_rate",
    "n_trades": "rp.n_trades",
    "sharpe_ratio": "rp.sharpe_ratio",
    "created_time": "rp.created_time",
    "updated_time": "rp.updated_time",
}

# Timestamp sorts use a keyset on (sort_col, paper_id); metric sorts need
# the numeric value in the cursor to build a proper NULLS-aware keyset.
_TIMESTAMP_SORT_FIELDS = {"created_time", "updated_time"}


async def list_papers(
    conn: asyncpg.Connection,
    *,
    paper_status: str | None,
    strategy_code: str | None,
    strategy_kind: str | None,
    instrument: str | None,
    interval_name: str | None,
    # Default (created_time) cursor
    after_created_time: datetime | None,
    after_id: str | None,
    # Metric sort cursor — only set when sort_by != "created_time" and cursor present
    after_metric_val: Decimal | None = None,  # the sort column value from the last row (None = was NULL)
    has_metric_cursor: bool = False,          # True iff a metric-sort cursor was provided
    limit: int = 20,
    sort_by: str = "created_time",
    sort_dir: str = "desc",
) -> list[dict[str, Any]]:
    where = ["1=1"]
    args: list[Any] = []

    if paper_status is not None:
        args.append(paper_status)
        where.append(f"rp.paper_status = ${len(args)}")
    if strategy_code is not None:
        args.append(strategy_code)
        where.append(f"rq.strategy_code = ${len(args)}")
    if strategy_kind is not None:
        # Kind is sourced from strategy_definition.strategy_kind (V153), the same
        # single source the trading JVM's StrategyKindResolver reads. Codes with no
        # definition row (or a null kind) default to TRADING, so a HEDGING filter
        # never accidentally captures an unclassified paper.
        args.append(strategy_kind.upper())
        where.append(
            f"COALESCE(UPPER(sd.strategy_kind), 'TRADING') = ${len(args)}"
        )
    if instrument is not None:
        args.append(instrument)
        where.append(f"rq.instrument = ${len(args)}")
    if interval_name is not None:
        args.append(interval_name)
        where.append(f"rq.interval_name = ${len(args)}")

    sort_col = _SORT_COLUMN_MAP.get(sort_by, "rp.created_time")
    direction = "DESC" if sort_dir.lower() != "asc" else "ASC"
    nulls = "NULLS LAST" if direction == "DESC" else "NULLS FIRST"

    if sort_by in _TIMESTAMP_SORT_FIELDS and after_created_time is not None and after_id is not None:
        # Timestamp sort keyset: cursor "t" carries the sort column value
        # (created_time or updated_time). paper_id breaks ties.
        ts_col = _SORT_COLUMN_MAP.get(sort_by, "rp.created_time")
        args.append(after_created_time)
        args.append(after_id)
        op = "<" if direction == "DESC" else ">"
        where.append(
            f"({ts_col}, rp.paper_id) {op} (${len(args) - 1}, ${len(args)})"
        )
    elif has_metric_cursor and after_created_time is not None and after_id is not None:
        # Keyset pagination for metric-sorted results.
        # For DESC NULLS LAST: after a row with sort_val=V, t=T, id=ID we want rows where:
        #   - sort_val < V  (strictly less, non-null), OR
        #   - sort_val = V AND (created_time, paper_id) < (T, ID) (tie-break), OR
        #   - sort_val IS NULL  (nulls are last, so they all follow any non-null value)
        # When V itself was NULL (cursor in the null section):
        #   - sort_val IS NULL AND (created_time, paper_id) < (T, ID)
        # For ASC NULLS FIRST: symmetric — nulls come first so they are already consumed by then.
        args.append(after_created_time)
        args.append(after_id)
        t_idx = len(args) - 1
        id_idx = len(args)

        if after_metric_val is None:
            # Cursor was in the NULL section of the sort column.
            if direction == "DESC":
                # NULLs LAST: remaining rows are NULLs with earlier (t, id)
                where.append(
                    f"{sort_col} IS NULL "
                    f"AND (rp.created_time, rp.paper_id) < (${t_idx}, ${id_idx})"
                )
            else:
                # ASC NULLS FIRST: after a NULL row we want NULLs with later (t,id) OR any non-null
                where.append(
                    f"({sort_col} IS NOT NULL "
                    f"OR ({sort_col} IS NULL AND (rp.created_time, rp.paper_id) > (${t_idx}, ${id_idx})))"
                )
        else:
            # Cursor had a non-null sort value.
            args.append(after_metric_val)
            sv_idx = len(args)
            if direction == "DESC":
                where.append(
                    f"({sort_col} < ${sv_idx} "
                    f"OR ({sort_col} IS NOT DISTINCT FROM ${sv_idx} "
                    f"    AND (rp.created_time, rp.paper_id) < (${t_idx}, ${id_idx})) "
                    f"OR {sort_col} IS NULL)"
                )
            else:
                where.append(
                    f"({sort_col} > ${sv_idx} "
                    f"OR ({sort_col} IS NOT DISTINCT FROM ${sv_idx} "
                    f"    AND (rp.created_time, rp.paper_id) > (${t_idx}, ${id_idx})))"
                )
    args.append(limit)
    sql = f"""
        SELECT rp.paper_id,
               rp.queue_id,
               rp.title,
               rp.abstract,
               rp.paper_status,
               rp.version,
               rp.win_rate,
               rp.annualized_return_pct,
               rp.profit_factor,
               rp.n_trades,
               rp.max_drawdown_pct,
               rp.sharpe_ratio,
               rq.strategy_code,
               COALESCE(UPPER(sd.strategy_kind), 'TRADING') AS strategy_kind,
               rq.instrument,
               rq.interval_name,
               rq.final_verdict,
               rq.iteration_number  AS total_iterations,
               rp.created_time,
               rp.updated_time
        FROM research_paper rp
        JOIN research_queue  rq ON rq.queue_id = rp.queue_id
        LEFT JOIN strategy_definition sd ON sd.strategy_code = rq.strategy_code
        WHERE {' AND '.join(where)}
        ORDER BY {sort_col} {direction} {nulls}, rp.created_time DESC, rp.paper_id DESC
        LIMIT ${len(args)}
    """
    rows = await conn.fetch(sql, *args)
    return [dict(r) for r in rows]


async def upsert_paper(
    conn: asyncpg.Connection,
    *,
    paper_id: str,
    queue_id: UUID,
    title: str,
    abstract: str,
    paper_status: str,
    version: int,
    created_by: str,
    sections_cache: list | None = None,
    win_rate: float | None = None,
    annualized_return_pct: float | None = None,
    profit_factor: float | None = None,
    n_trades: int | None = None,
    max_drawdown_pct: float | None = None,
    sharpe_ratio: float | None = None,
) -> dict[str, Any]:
    row = await conn.fetchrow(
        """
        INSERT INTO research_paper
            (paper_id, queue_id, title, abstract, paper_status, version,
             sections_cache,
             win_rate, annualized_return_pct, profit_factor, n_trades,
             max_drawdown_pct, sharpe_ratio,
             created_time, created_by, updated_time, updated_by)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, NOW(), $14, NOW(), $14)
        ON CONFLICT (paper_id) DO UPDATE
            SET queue_id              = EXCLUDED.queue_id,
                title                 = EXCLUDED.title,
                abstract              = EXCLUDED.abstract,
                paper_status          = EXCLUDED.paper_status,
                version               = EXCLUDED.version,
                sections_cache        = COALESCE(EXCLUDED.sections_cache, research_paper.sections_cache),
                win_rate              = COALESCE(EXCLUDED.win_rate, research_paper.win_rate),
                annualized_return_pct = COALESCE(EXCLUDED.annualized_return_pct, research_paper.annualized_return_pct),
                profit_factor         = COALESCE(EXCLUDED.profit_factor, research_paper.profit_factor),
                n_trades              = COALESCE(EXCLUDED.n_trades, research_paper.n_trades),
                max_drawdown_pct      = COALESCE(EXCLUDED.max_drawdown_pct, research_paper.max_drawdown_pct),
                sharpe_ratio          = COALESCE(EXCLUDED.sharpe_ratio, research_paper.sharpe_ratio),
                updated_time          = NOW(),
                updated_by            = EXCLUDED.updated_by
        RETURNING paper_id, queue_id, title, abstract, paper_status, version,
                  win_rate, annualized_return_pct, profit_factor, n_trades,
                  max_drawdown_pct, sharpe_ratio,
                  sections_cache, created_time, created_by, updated_time, updated_by
        """,
        paper_id,
        queue_id,
        title,
        abstract,
        paper_status,
        version,
        sections_cache,
        win_rate,
        annualized_return_pct,
        profit_factor,
        n_trades,
        max_drawdown_pct,
        sharpe_ratio,
        created_by,
    )
    return dict(row)


# ---------------------------------------------------------------------------
# research_paper_trade_data
# ---------------------------------------------------------------------------

async def get_chart_data(
    conn: asyncpg.Connection,
    paper_id: str,
) -> dict[str, Any] | None:
    # Return the most recently stored trade dataset for this paper.
    row = await conn.fetchrow(
        """
        SELECT equity_curve, trades, wf_equity_curve,
               iteration_id, created_time
        FROM research_paper_trade_data
        WHERE paper_id = $1
        ORDER BY created_time DESC
        LIMIT 1
        """,
        paper_id,
    )
    return dict(row) if row else None


async def upsert_trade_data(
    conn: asyncpg.Connection,
    *,
    paper_id: str,
    iteration_id: UUID,
    equity_curve: list[dict[str, Any]],
    trades: list[dict[str, Any]],
    wf_equity_curve: list[dict[str, Any]] | None,
    created_by: str,
) -> dict[str, Any]:
    row = await conn.fetchrow(
        """
        INSERT INTO research_paper_trade_data
            (paper_id, iteration_id, equity_curve, trades, wf_equity_curve,
             created_time, created_by, updated_time, updated_by)
        VALUES ($1, $2, $3, $4, $5, NOW(), $6, NOW(), $6)
        ON CONFLICT (paper_id, iteration_id) DO UPDATE
            SET equity_curve    = EXCLUDED.equity_curve,
                trades          = EXCLUDED.trades,
                wf_equity_curve = EXCLUDED.wf_equity_curve,
                updated_time    = NOW(),
                updated_by      = EXCLUDED.updated_by
        RETURNING id, paper_id, iteration_id, created_time, updated_time
        """,
        paper_id,
        iteration_id,
        equity_curve,
        trades,
        wf_equity_curve,
        created_by,
    )
    return dict(row)


# ---------------------------------------------------------------------------
# Supporting reads for paper_generator
# ---------------------------------------------------------------------------

async def get_queue_for_paper(
    conn: asyncpg.Connection,
    queue_id: UUID,
) -> dict[str, Any] | None:
    row = await conn.fetchrow(
        """
        SELECT queue_id, strategy_code, interval_name, instrument,
               sweep_config, hypothesis, status, iteration_number,
               iter_budget, final_verdict, walk_forward_id,
               last_iteration_id, last_run_id,
               created_time, completed_at, notes
        FROM research_queue
        WHERE queue_id = $1
        """,
        queue_id,
    )
    return dict(row) if row else None


async def get_iterations_for_queue(
    conn: asyncpg.Connection,
    queue_id: UUID,
) -> list[dict[str, Any]]:
    """All completed iterations for a queue, ordered oldest-first.

    Joins through ``hypothesis_audit`` (which carries ``queue_id``) to
    ``research_iteration_log``. Rows where ``iteration_id IS NULL`` mean
    the tick crashed before logging — excluded here.
    """
    rows = await conn.fetch(
        """
        SELECT il.iteration_id,
               il.iteration_number,
               il.strategy_code,
               il.backtest_run_id,
               il.params_snapshot,
               il.metrics_snapshot,
               il.verdict,
               il.statistical_verdict,
               il.confidence_intervals,
               il.quant_audit_notes,
               il.created_time
        FROM hypothesis_audit ha
        JOIN research_iteration_log il ON il.iteration_id = ha.iteration_id
        WHERE ha.queue_id = $1
          AND ha.iteration_id IS NOT NULL
        ORDER BY il.created_time ASC
        """,
        queue_id,
    )
    return [dict(r) for r in rows]


async def get_backtest_run_meta(
    conn: asyncpg.Connection,
    backtest_run_id: UUID,
) -> dict[str, Any] | None:
    row = await conn.fetchrow(
        """
        SELECT backtest_run_id, initial_capital, start_time, end_time,
               return_pct, status
        FROM backtest_run
        WHERE backtest_run_id = $1
        """,
        backtest_run_id,
    )
    return dict(row) if row else None


async def get_journal_for_strategy(
    conn: asyncpg.Connection,
    strategy_code: str,
    instrument: str,
) -> list[dict[str, Any]]:
    rows = await conn.fetch(
        """
        SELECT journal_id, entry_type, title, content, status,
               structured_data, created_time, created_by
        FROM research_journal
        WHERE strategy_code = $1
          AND (instrument IS NULL OR instrument = $2)
        ORDER BY created_time ASC
        """,
        strategy_code,
        instrument,
    )
    return [dict(r) for r in rows]
