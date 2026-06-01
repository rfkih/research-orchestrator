"""One-time backfill of ``backtest_run.deflated_sharpe`` / ``dsr_n_trials``.

V134 added these columns for the backtest leaderboard to rank on, but the
writer was never wired — DSR lived only in
``research_iteration_log.metrics_snapshot``, so every ``backtest_run`` row had a
NULL ``deflated_sharpe`` and the leaderboard's ``deflated_sharpe IS NOT NULL``
feed returned nothing. The live path is now fixed
(``iterations_write.update_backtest_run_dsr`` called from ``tick.py`` step 5);
this script populates the pre-existing rows.

It does NOT copy the snapshot's stored ``dsr`` verbatim: those were computed
with the buggy strategy_code-scoped ``dsr_n_trials`` (B1), which under-counts
multiplicity and inflates DSR. Instead it RECOMPUTES DSR with the corrected
data-universe trial count (``count_data_universe_trials(symbol, interval)``) via
the same ``analyze.deflated_sharpe_ratio`` the live path uses — so backfilled
rows are methodology-consistent with new ones.

Inputs per run come from the iteration's snapshot
(``sharpe_annualized``, ``skewness``, ``excess_kurtosis``, ``n_trades``); the
per-trade PnL series is fetched so the Bailey-LdP bootstrap fallback fires on
degenerate closed-form cases exactly as live. (symbol, interval) come from the
``backtest_run`` row.

Idempotent: re-running recomputes and overwrites with the same values. Safe to
run repeatedly (e.g. after more research extends the universe count).

Usage:
    PYTHONPATH=src python scripts/backfill_backtest_dsr.py [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from typing import Any

import asyncpg

from orchestrator.repo import hypothesis_audit
from orchestrator.services import analyze

DSN = os.environ.get(
    "ORCH_DB_DSN",
    "postgresql://blackheart_research:CLjj505OEU5emM5aaLCmHJlRrwcx@127.0.0.1:5432/trading_db",
)


def _f(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


async def _universe_count(
    conn: asyncpg.Connection, cache: dict[tuple[str, str], int], symbol: str, interval: str
) -> int:
    key = (symbol, interval)
    if key not in cache:
        # No +1 here: unlike the live path (which counts BEFORE logging the
        # in-flight trial), every row we backfill is already a COMPLETED,
        # iteration-logged trial, so the universe count already includes it.
        cache[key] = await hypothesis_audit.count_data_universe_trials(conn, symbol, interval)
    return cache[key]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="compute but do not UPDATE")
    args = parser.parse_args()

    conn = await asyncpg.connect(DSN)
    universe_cache: dict[tuple[str, str], int] = {}

    # Every COMPLETED run that has a linked iteration snapshot with the DSR
    # inputs. LEFT-of-NULL deflated_sharpe is not required — recompute is
    # idempotent, so this also corrects any rows the live wire wrote with the
    # stale count before this backfill ran.
    rows = await conn.fetch(
        """
        SELECT br.backtest_run_id, br.asset, br.interval_name,
               il.metrics_snapshot
        FROM backtest_run br
        JOIN research_iteration_log il ON il.backtest_run_id = br.backtest_run_id
        WHERE br.status = 'COMPLETED'
        ORDER BY br.created_time
        """
    )
    print(f"Examining {len(rows)} completed run/iteration pairs...")

    updated = skipped = errors = 0
    for r in rows:
        run_id = r["backtest_run_id"]
        try:
            snap = r["metrics_snapshot"]
            if isinstance(snap, str):
                snap = json.loads(snap)
            analysis = (snap or {}).get("analysis") or {}

            sr = _f(analysis.get("sharpe_annualized"))
            n = analysis.get("n_trades")
            skew = _f(analysis.get("skewness"))
            kurt = _f(analysis.get("excess_kurtosis"))
            if sr is None or not n or skew is None or kurt is None:
                skipped += 1
                continue

            # Per-trade PnL series for the bootstrap fallback (mirrors live).
            trades = await conn.fetch(
                "SELECT realized_pnl_amount FROM backtest_trade "
                "WHERE backtest_run_id = $1 AND exit_time IS NOT NULL",
                run_id,
            )
            pnls = [_f(t["realized_pnl_amount"]) or 0.0 for t in trades]

            n_trials = await _universe_count(conn, universe_cache, r["asset"], r["interval_name"])
            dsr = analyze.deflated_sharpe_ratio(
                sr, int(n), skew, kurt, n_trials, pnls=pnls or None
            )
            if dsr is None:
                # n<30 or degenerate-without-pnls — leave NULL (fails closed,
                # same as live). The leaderboard filters these out by design.
                skipped += 1
                continue

            if args.dry_run:
                print(f"  DRY  {run_id} {r['asset']}/{r['interval_name']} "
                      f"dsr={dsr:.4f} n_trials={n_trials}")
            else:
                await conn.execute(
                    "UPDATE backtest_run SET deflated_sharpe = $2, dsr_n_trials = $3 "
                    "WHERE backtest_run_id = $1",
                    run_id, dsr, n_trials,
                )
            updated += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ERR  {run_id} — {e}")
            errors += 1

    await conn.close()
    verb = "would update" if args.dry_run else "updated"
    print(f"\nDone: {verb} {updated}, skipped {skipped}, errors {errors}")


if __name__ == "__main__":
    asyncio.run(main())
