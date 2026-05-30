"""Backfill win_rate + performance metrics for all existing research_paper rows.

Run once after V129 migration.  Uses direct DB access — no orchestrator API call
needed (avoids re-running the AI narrator).

Usage:
    python scripts/backfill_paper_metrics.py
"""

from __future__ import annotations

import asyncio
import json
import math
from typing import Any

import asyncpg

DSN = "postgresql://blackheart_research:CLjj505OEU5emM5aaLCmHJlRrwcx@127.0.0.1:5432/trading_db"


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


async def compute_metrics(
    conn: asyncpg.Connection,
    paper_id: str,
    queue_id: str,
) -> dict[str, Any] | None:
    # Find best iteration (highest annualized_geometric_return_pct_at_alloc_90)
    iterations = await conn.fetch(
        """
        SELECT il.iteration_id,
               il.backtest_run_id,
               il.metrics_snapshot
        FROM hypothesis_audit ha
        JOIN research_iteration_log il ON il.iteration_id = ha.iteration_id
        WHERE ha.queue_id = $1
          AND ha.iteration_id IS NOT NULL
        ORDER BY il.created_time ASC
        """,
        queue_id,
    )
    if not iterations:
        return None

    best = None
    best_score = float("-inf")
    for row in iterations:
        raw = row["metrics_snapshot"]
        if isinstance(raw, str):
            raw = json.loads(raw)
        metrics = raw or {}
        analysis = metrics.get("analysis") or {}
        geom = analysis.get("annualized_geometric_return_pct_at_alloc_90")
        try:
            score = float(geom) if geom is not None else float("-inf")
        except (TypeError, ValueError):
            score = float("-inf")
        if math.isfinite(score) and score > best_score:
            best_score = score
            best = row

    if best is None:
        best = iterations[-1]

    raw_snap = best["metrics_snapshot"]
    if isinstance(raw_snap, str):
        raw_snap = json.loads(raw_snap)
    metrics_snap = raw_snap or {}
    analysis = metrics_snap.get("analysis") or {}
    backtest_run_id = best["backtest_run_id"]

    # Win rate from trades
    win_rate = None
    if backtest_run_id:
        trades = await conn.fetch(
            """
            SELECT realized_pnl_amount, exit_time
            FROM backtest_trade
            WHERE backtest_run_id = $1 AND exit_time IS NOT NULL
            """,
            backtest_run_id,
        )
        n_closed = len(trades)
        if n_closed > 0:
            wins = sum(1 for t in trades if (t["realized_pnl_amount"] or 0) > 0)
            win_rate = round((wins / n_closed) * 100.0, 2)

    pf = _safe_float(metrics_snap.get("profit_factor"))
    if pf is not None and pf > 9999.0:
        pf = 9999.0

    return {
        "win_rate": win_rate,
        "annualized_return_pct": _safe_float(
            analysis.get("annualized_geometric_return_pct_at_alloc_90")
        ),
        "profit_factor": pf,
        "n_trades": int(metrics_snap.get("total_trades") or 0) or None,
        "max_drawdown_pct": _safe_float(metrics_snap.get("max_drawdown_pct")),
        "sharpe_ratio": _safe_float(metrics_snap.get("sharpe_ratio")),
    }


async def main() -> None:
    conn = await asyncpg.connect(DSN)

    papers = await conn.fetch(
        "SELECT paper_id, queue_id FROM research_paper ORDER BY created_time"
    )
    print(f"Backfilling {len(papers)} papers...")

    ok = skip = err = 0
    for p in papers:
        paper_id = p["paper_id"]
        queue_id = str(p["queue_id"])
        try:
            m = await compute_metrics(conn, paper_id, queue_id)
            if m is None:
                print(f"  SKIP {paper_id} — no iterations")
                skip += 1
                continue

            await conn.execute(
                """
                UPDATE research_paper
                SET win_rate              = $2,
                    annualized_return_pct = $3,
                    profit_factor         = $4,
                    n_trades              = $5,
                    max_drawdown_pct      = $6,
                    sharpe_ratio          = $7,
                    updated_time          = NOW()
                WHERE paper_id = $1
                """,
                paper_id,
                m["win_rate"],
                m["annualized_return_pct"],
                m["profit_factor"],
                m["n_trades"],
                m["max_drawdown_pct"],
                m["sharpe_ratio"],
            )
            ann = m["annualized_return_pct"]
            wr = m["win_rate"]
            pf = m["profit_factor"]
            print(
                f"  OK   {paper_id:<50} "
                f"ann={ann:>8.1f}%  wr={wr:>5.1f}%  pf={pf:>5.2f}"
                if ann is not None and wr is not None and pf is not None
                else f"  OK   {paper_id} (partial metrics)"
            )
            ok += 1
        except Exception as e:
            print(f"  ERR  {paper_id} — {e}")
            err += 1

    await conn.close()
    print(f"\nDone: {ok} updated, {skip} skipped, {err} errors")


if __name__ == "__main__":
    asyncio.run(main())
