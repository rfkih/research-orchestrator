"""``research_paper`` + ``research_paper_trade_data`` reads and writes.

Also exposes the supporting queries (queue, iterations, backtest_run,
journal) that ``services/paper_generator.py`` needs so the generator
never touches another repo directly.
"""

from __future__ import annotations

from datetime import datetime
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
               created_time, created_by, updated_time, updated_by
        FROM research_paper
        WHERE paper_id = $1
        """,
        paper_id,
    )
    return dict(row) if row else None


async def list_papers(
    conn: asyncpg.Connection,
    *,
    paper_status: str | None,
    strategy_code: str | None,
    instrument: str | None,
    interval_name: str | None,
    after_created_time: datetime | None,
    after_id: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    where = ["1=1"]
    args: list[Any] = []

    if paper_status is not None:
        args.append(paper_status)
        where.append(f"rp.paper_status = ${len(args)}")
    if strategy_code is not None:
        args.append(strategy_code)
        where.append(f"rq.strategy_code = ${len(args)}")
    if instrument is not None:
        args.append(instrument)
        where.append(f"rq.instrument = ${len(args)}")
    if interval_name is not None:
        args.append(interval_name)
        where.append(f"rq.interval_name = ${len(args)}")
    if after_created_time is not None and after_id is not None:
        args.append(after_created_time)
        args.append(after_id)
        where.append(
            f"(rp.created_time, rp.paper_id) < (${len(args) - 1}, ${len(args)})"
        )

    args.append(limit)
    sql = f"""
        SELECT rp.paper_id,
               rp.queue_id,
               rp.title,
               rp.abstract,
               rp.paper_status,
               rp.version,
               rq.strategy_code,
               rq.instrument,
               rq.interval_name,
               rq.final_verdict,
               rq.iteration_number  AS total_iterations,
               rp.created_time,
               rp.updated_time
        FROM research_paper rp
        JOIN research_queue  rq ON rq.queue_id = rp.queue_id
        WHERE {' AND '.join(where)}
        ORDER BY rp.created_time DESC, rp.paper_id DESC
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
) -> dict[str, Any]:
    row = await conn.fetchrow(
        """
        INSERT INTO research_paper
            (paper_id, queue_id, title, abstract, paper_status, version,
             created_time, created_by, updated_time, updated_by)
        VALUES ($1, $2, $3, $4, $5, $6, NOW(), $7, NOW(), $7)
        ON CONFLICT (paper_id) DO UPDATE
            SET title        = EXCLUDED.title,
                abstract     = EXCLUDED.abstract,
                paper_status = EXCLUDED.paper_status,
                version      = EXCLUDED.version,
                updated_time = NOW(),
                updated_by   = EXCLUDED.updated_by
        RETURNING paper_id, queue_id, title, abstract, paper_status, version,
                  created_time, created_by, updated_time, updated_by
        """,
        paper_id,
        queue_id,
        title,
        abstract,
        paper_status,
        version,
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
