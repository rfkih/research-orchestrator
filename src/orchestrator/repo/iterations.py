"""``research_iteration_log`` reads + leaderboard.

Iteration ranking matches ``leaderboard.sh``: pulls metrics out of the
JSONB ``metrics_snapshot`` and ``confidence_intervals`` columns, sorts by
the requested key. The agent uses this to decide whether to enqueue a
follow-up sweep or graduate a strategy.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any
from uuid import UUID

import asyncpg

from ..errors import OrchestratorError

# PF sentinel used in services/analyze.py when a backtest had no losses.
# We cap at this value when feeding TPE so the sampler doesn't extrapolate
# wildly off a single near-perfect cell.
_TPE_PF_CAP = 10.0

# PF infinity sentinel stamped by services/analyze.py for no-loss runs
# (gross loss == 0). Mirrors analyze.PF_INFINITY_SENTINEL. A degenerate
# 1-trade winner gets PF=9999 and, under a naive DESC sort, floats to the
# top of the board ahead of every real strategy. The ``pf`` sort below
# treats the sentinel as NULL so an unreliable "infinite" PF ranks LAST,
# never above a genuine finite PF. (Display-only — does NOT touch the V11
# PF 95% CI gate, which already ignores the sentinel via its own path.)
_PF_INFINITY_SENTINEL = "9999"

_LEADERBOARD_SORTS = {
    # NULLIF drops the no-loss sentinel out of the ranking so a lucky
    # 1-trade PF=9999 cannot outrank a real strategy's finite PF.
    "pf": (
        f"NULLIF((metrics_snapshot->>'profit_factor')::numeric, "
        f"{_PF_INFINITY_SENTINEL})"
    ),
    "return_pct": "(metrics_snapshot->>'return_pct')::numeric",
    "sharpe": "(metrics_snapshot->>'sharpe_ratio')::numeric",
    # metrics_snapshot = {**run_metrics, "analysis": analysis} (tick.py). The
    # run-level trade count is the top-level ``total_trades`` key (from
    # backtest_run); there is no ``trade_count`` key, so the prior read was
    # always NULL → this sort silently no-op'd. (The analyzer's nested count
    # is metrics_snapshot->'analysis'->>'n_trades'.)
    "trade_count": "(metrics_snapshot->>'total_trades')::int",
    "created": "created_time",
}


async def list_iterations(
    conn: asyncpg.Connection,
    *,
    strategy_code: str | None,
    verdict: str | None,
    after_created_time: datetime | None,
    after_id: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    where = ["1=1"]
    args: list[Any] = []
    if strategy_code is not None:
        args.append(strategy_code)
        where.append(f"strategy_code = ${len(args)}")
    if verdict is not None:
        args.append(verdict)
        where.append(f"verdict = ${len(args)}")
    if after_created_time is not None and after_id is not None:
        # Iteration list orders newest-first, so cursor pagination uses
        # strict-less on the same composite.
        args.append(after_created_time)
        args.append(after_id)
        where.append(
            f"(created_time, iteration_id::text) < (${len(args) - 1}, ${len(args)})"
        )
    args.append(limit)
    sql = f"""
        SELECT iteration_id, strategy_code, iteration_number, backtest_run_id,
               params_snapshot, metrics_snapshot, verdict, statistical_verdict,
               confidence_intervals, sample_size_adequate,
               git_commit_hash, hypothesis_predicted, hypothesis_outcome,
               hypothesis_match, code_change_summary, change_reasoning,
               next_action, diagnostics, quant_audit_notes,
               created_time, created_by
        FROM research_iteration_log
        WHERE {' AND '.join(where)}
        ORDER BY created_time DESC, iteration_id::text DESC
        LIMIT ${len(args)}
    """
    rows = await conn.fetch(sql, *args)
    return [dict(r) for r in rows]


async def fetch_tpe_history_for_queue(
    conn: asyncpg.Connection, queue_id: UUID
) -> list[tuple[dict[str, Any], float | None]]:
    """Past trials for a queue, formatted for ``next_combo_tpe``.

    Returns ``(params, pf | None)`` tuples in creation order:
      * ``pf`` set → completed trial; TPE adds it as a normal trial.
      * ``pf`` None → failed trial (JVM crash, backtest FAILED, or n<5
        trades so PF is uncomputable). TPE adds these as PRUNED so the
        sampler learns to avoid the region rather than re-suggesting it.

    Joins ``hypothesis_audit`` (queue_id linkage) to
    ``research_iteration_log`` via LEFT JOIN so failed audits (where
    iteration_id IS NULL) still appear in history. The ``params_snapshot``
    is read from whichever side is non-null.

    Sentinel-PF (no losses, ``analyze.PF_INFINITY_SENTINEL``) is capped
    at ``_TPE_PF_CAP`` so a single anomaly doesn't dominate the sampler.
    """
    rows = await conn.fetch(
        """
        SELECT COALESCE(il.params_snapshot, ha.params_snapshot) AS params_snapshot,
               il.metrics_snapshot
          FROM hypothesis_audit ha
          LEFT JOIN research_iteration_log il ON il.iteration_id = ha.iteration_id
         WHERE ha.queue_id = $1
         ORDER BY ha.created_time ASC
        """,
        queue_id,
    )
    out: list[tuple[dict[str, Any], float | None]] = []
    for r in rows:
        params = dict(r["params_snapshot"] or {})
        if not params:
            continue
        metrics = r["metrics_snapshot"]
        analysis = metrics.get("analysis") if isinstance(metrics, dict) else None
        if not isinstance(analysis, dict):
            # Either iteration_id NULL (failed audit) or no analysis block.
            # Either way, treat as failed trial.
            out.append((params, None))
            continue
        pf = analysis.get("pf_point_estimate")
        if pf is None:
            out.append((params, None))
            continue
        try:
            pf_f = float(pf)
        except (TypeError, ValueError):
            out.append((params, None))
            continue
        if not math.isfinite(pf_f):
            out.append((params, None))
            continue
        if pf_f >= _TPE_PF_CAP:
            pf_f = _TPE_PF_CAP
        out.append((params, pf_f))
    return out


async def get_iteration(
    conn: asyncpg.Connection, iteration_id: UUID
) -> dict[str, Any] | None:
    row = await conn.fetchrow(
        """
        SELECT iteration_id, strategy_code, iteration_number, backtest_run_id,
               params_snapshot, metrics_snapshot, feature_buckets, notable_trades,
               verdict, statistical_verdict, confidence_intervals,
               effective_pf_after_slippage, sample_size_adequate,
               git_commit_hash, hypothesis_predicted, hypothesis_outcome,
               hypothesis_match, code_change_summary, change_reasoning,
               next_action, diagnostics, quant_audit_notes,
               created_time, created_by, updated_time, updated_by
        FROM research_iteration_log
        WHERE iteration_id = $1
        """,
        iteration_id,
    )
    return dict(row) if row else None


async def resolve_track(
    conn: asyncpg.Connection, iteration_id: UUID
) -> str | None:
    """Authoritative research track for an iteration (2026-06-09).

    Sourced from the ORIGINATING queue row's ``sweep_config->>'track'`` via the
    ``hypothesis_audit.queue_id`` linkage — the same authority the tick gate uses
    — rather than inferring from metrics shape. Returns ``None`` for an untagged
    iteration (legacy/global enqueue ⇒ caller treats as the default trading
    track). Most-recent audit row wins when an iteration spans more than one.
    """
    return await conn.fetchval(
        """
        SELECT rq.sweep_config->>'track'
        FROM research_iteration_log ril
        LEFT JOIN hypothesis_audit ha ON ha.iteration_id = ril.iteration_id
        LEFT JOIN research_queue rq ON rq.queue_id = ha.queue_id
        WHERE ril.iteration_id = $1
        ORDER BY ha.created_time DESC NULLS LAST
        LIMIT 1
        """,
        iteration_id,
    )


async def leaderboard(
    conn: asyncpg.Connection,
    *,
    strategy_code: str | None,
    verdict: str | None,
    significant_only: bool,
    min_trades: int,
    sort: str,
    limit: int,
) -> list[dict[str, Any]]:
    if sort not in _LEADERBOARD_SORTS:
        raise OrchestratorError(
            status_code=400,
            error_code="bad_leaderboard_sort",
            message=f"sort must be one of {sorted(_LEADERBOARD_SORTS)}.",
            retryable=False,
            hint="Pass sort=pf for default profit-factor ranking.",
        )
    where = ["1=1"]
    args: list[Any] = []
    if strategy_code is not None:
        args.append(strategy_code)
        where.append(f"strategy_code = ${len(args)}")
    if verdict is not None:
        args.append(verdict)
        where.append(f"verdict = ${len(args)}")
    if significant_only:
        where.append("statistical_verdict = 'SIGNIFICANT_EDGE'")
    if min_trades > 0:
        # Minimum-sample floor so degenerate n=1-2 runs (whose PF is a
        # no-loss 9999 sentinel or a single-lucky-trade artifact) do not
        # surface on the board. Display sanity floor only — NOT the V11
        # n>=100 significance gate, which is enforced separately at verdict
        # time. NULL total_trades (very old rows) is excluded by the
        # numeric comparison; that is the safe default for a quality board.
        args.append(min_trades)
        where.append(f"(metrics_snapshot->>'total_trades')::int >= ${len(args)}")
    args.append(limit)
    sql = f"""
        SELECT strategy_code,
               iteration_id,
               iteration_number,
               verdict,
               statistical_verdict,
               (metrics_snapshot->>'profit_factor')::numeric           AS pf,
               (confidence_intervals->'pf_95'->>'low')::numeric        AS pf_lo,
               (confidence_intervals->'pf_95'->>'high')::numeric       AS pf_hi,
               (metrics_snapshot->>'return_pct')::numeric              AS return_pct,
               (metrics_snapshot->>'sharpe_ratio')::numeric            AS sharpe,
               (metrics_snapshot->>'total_trades')::int               AS trade_count,
               sample_size_adequate,
               git_commit_hash,
               created_time
        FROM research_iteration_log
        WHERE {' AND '.join(where)}
        ORDER BY {_LEADERBOARD_SORTS[sort]} DESC NULLS LAST,
                 created_time DESC
        LIMIT ${len(args)}
    """
    rows = await conn.fetch(sql, *args)
    return [dict(r) for r in rows]
