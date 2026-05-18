"""Composite state digest for ``GET /agent/state`` (Phase 2).

A new agent session — researcher *or* runner — needs a compact snapshot of
"where is research right now?" Before this helper, the researcher rolled
its own with 5+ separate queries (queue rows, iteration_log, leaderboard,
journal HYPOTHESIS, journal RUN_SUMMARY, journal NULL_SCREEN_RESULT). At
opus token rates that's the bulk of cold-boot spend per loop iteration.

This module composes one DB round-trip per logical slice and returns a
flat dict the API layer renders through Pydantic. All slices are read-only
and idempotent — safe to call on every session start.
"""

from __future__ import annotations

from typing import Any

import asyncpg


async def _queue_counts(conn: asyncpg.Connection) -> dict[str, int]:
    rows = await conn.fetch(
        """
        SELECT status, COUNT(*)::int AS n
        FROM research_queue
        GROUP BY status
        """
    )
    counts = {r["status"]: int(r["n"]) for r in rows}
    # Always include the four states the researcher branches on, even when
    # zero — saves a None-check at the call site.
    for k in ("PENDING", "RUNNING", "PARKED", "COMPLETED", "FAILED"):
        counts.setdefault(k, 0)
    return counts


async def _last_iterations(conn: asyncpg.Connection, n: int) -> list[dict[str, Any]]:
    rows = await conn.fetch(
        """
        SELECT iteration_id, strategy_code, iteration_number,
               statistical_verdict, verdict, created_time,
               (metrics_snapshot->>'profit_factor')::numeric AS pf,
               (metrics_snapshot->>'trade_count')::int       AS n_trades
        FROM research_iteration_log
        ORDER BY created_time DESC, iteration_id::text DESC
        LIMIT $1
        """,
        n,
    )
    return [
        {
            "iteration_id": str(r["iteration_id"]),
            "strategy_code": r["strategy_code"],
            "iteration_number": r["iteration_number"],
            "statistical_verdict": r["statistical_verdict"],
            "verdict": r["verdict"],
            "created_time": r["created_time"],
            "pf": float(r["pf"]) if r["pf"] is not None else None,
            "n_trades": r["n_trades"],
        }
        for r in rows
    ]


async def _recent_sig_edge_ids(
    conn: asyncpg.Connection, window_days: int, *, hard_cap: int = 50
) -> list[str]:
    """Iteration ids with statistical_verdict=SIGNIFICANT_EDGE in the window.

    Capped at ``hard_cap`` so a long-running researcher with hundreds of
    SIG_EDGE iterations doesn't inflate the digest payload. The cap is
    informational only — if it's hit, the researcher should query
    ``GET /iterations?verdict=...`` directly for the full set.
    """
    rows = await conn.fetch(
        """
        SELECT iteration_id
        FROM research_iteration_log
        WHERE statistical_verdict = 'SIGNIFICANT_EDGE'
          AND created_time > NOW() - ($1::int * INTERVAL '1 day')
        ORDER BY created_time DESC
        LIMIT $2
        """,
        window_days,
        hard_cap,
    )
    return [str(r["iteration_id"]) for r in rows]


async def _active_hypotheses(conn: asyncpg.Connection) -> list[dict[str, Any]]:
    rows = await conn.fetch(
        """
        SELECT journal_id, strategy_code, title, created_time
        FROM research_journal
        WHERE entry_type = 'HYPOTHESIS' AND status = 'ACTIVE'
        ORDER BY created_time DESC
        """
    )
    return [
        {
            "journal_id": str(r["journal_id"]),
            "strategy_code": r["strategy_code"],
            "title": r["title"],
            "created_time": r["created_time"],
        }
        for r in rows
    ]


async def _last_run_summary(conn: asyncpg.Connection) -> dict[str, Any] | None:
    row = await conn.fetchrow(
        """
        SELECT journal_id, strategy_code, title, content,
               structured_data, created_time
        FROM research_journal
        WHERE entry_type = 'RUN_SUMMARY'
        ORDER BY created_time DESC
        LIMIT 1
        """
    )
    if row is None:
        return None
    return {
        "journal_id": str(row["journal_id"]),
        "strategy_code": row["strategy_code"],
        "title": row["title"],
        "content": row["content"],
        "structured_data": row["structured_data"],
        "created_time": row["created_time"],
    }


async def _last_null_screen_per_surface(
    conn: asyncpg.Connection,
) -> list[dict[str, Any]]:
    """Latest NULL_SCREEN_RESULT per (strategy_code, instrument, interval_name).

    The researcher uses this to skip a sweep when a recent NO_EDGE_DETECTED
    or INCONCLUSIVE screen rules out the surface. DISTINCT ON is the right
    primitive — one row per (code, instrument, interval) tuple, sorted to
    the most recent.
    """
    rows = await conn.fetch(
        """
        SELECT DISTINCT ON (strategy_code, instrument, interval_name)
               journal_id, strategy_code, instrument, interval_name,
               title, structured_data, created_time
        FROM research_journal
        WHERE entry_type = 'NULL_SCREEN_RESULT'
        ORDER BY strategy_code, instrument, interval_name, created_time DESC
        """
    )
    out: list[dict[str, Any]] = []
    for r in rows:
        sd = r["structured_data"] or {}
        verdict = sd.get("verdict") if isinstance(sd, dict) else None
        out.append(
            {
                "journal_id": str(r["journal_id"]),
                "strategy_code": r["strategy_code"],
                "instrument": r["instrument"],
                "interval_name": r["interval_name"],
                "verdict": verdict,
                "title": r["title"],
                "created_time": r["created_time"],
            }
        )
    return out


async def get_state_digest(
    conn: asyncpg.Connection,
    *,
    sig_edge_window_days: int = 7,
    last_n_iterations: int = 5,
) -> dict[str, Any]:
    """Compose the Phase-2 agent-state digest. Six small read queries.

    Designed to replace the cold-boot SQL-script chain. Total latency on a
    local PG with warm caches is < 50ms; well under the cost of spawning a
    research sub-agent.
    """
    queue_counts = await _queue_counts(conn)
    last_iters = await _last_iterations(conn, last_n_iterations)
    sig_edge_ids = await _recent_sig_edge_ids(conn, sig_edge_window_days)
    hypotheses = await _active_hypotheses(conn)
    run_summary = await _last_run_summary(conn)
    null_screens = await _last_null_screen_per_surface(conn)

    return {
        "queue_counts": queue_counts,
        "last_iterations": last_iters,
        "recent_sig_edge_iteration_ids": sig_edge_ids,
        "active_hypotheses": hypotheses,
        "last_run_summary": run_summary,
        "last_null_screen_per_surface": null_screens,
    }
