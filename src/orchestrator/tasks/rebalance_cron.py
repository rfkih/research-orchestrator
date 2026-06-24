"""Nightly portfolio-rebalance cron task (scorecard #10 portfolio construction).

Runs the existing ``rebalance_book`` optimizer (``services/rebalance``) on every
live account on a schedule, so ``account_strategy.portfolio_weight`` stops being
flat ``EQUAL_WEIGHT``. The optimizer was already exposed at
``POST /portfolio/rebalance`` but nothing invoked it periodically — this closes
the loop.

OFF by default (``ORCH_REBALANCE_ENABLED``) — automated rebalancing writes
weights that multiply into live order sizing. Safe on a thin/dormant book:
``rebalance_book`` falls back to equal-weight on insufficient history and no-ops
when no eligible strategies exist. Per-account failures (e.g. a 422
``below_min_notional_floor`` on a tiny account) are isolated so one account can
never stop the loop. Best-effort: the task never crashes the app.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import asyncpg

from ..config import Settings
from ..logging import get_logger
from ..services.rebalance import rebalance_book

if TYPE_CHECKING:
    from ..infra.db import Database

log = get_logger(__name__)

# Escalate from warning to ERROR after this many consecutive whole-pass
# failures (mirrors feature_refresh) so a permanently broken loop is shipped
# to the error pipeline rather than warning politely forever.
CONSECUTIVE_FAILURES_TO_ESCALATE = 3

# Short delay before the first pass after startup so a fast deploy cadence
# still rebalances at least once without waiting a full interval.
FIRST_RUN_DELAY_S = 60


async def _live_account_ids(conn: asyncpg.Connection) -> list[str]:
    """Distinct account_ids that own at least one live (enabled, real,
    non-deleted) strategy — the rebalance is a per-account operation."""
    rows = await conn.fetch(
        """
        SELECT DISTINCT account_id
          FROM account_strategy
         WHERE enabled = TRUE
           AND is_deleted = FALSE
           AND simulated = FALSE
         ORDER BY account_id
        """
    )
    return [str(r["account_id"]) for r in rows]


async def _free_usdt(conn: asyncpg.Connection, account_id: str) -> float:
    """Free USDT balance for the account (Portfolio.balance, asset='USDT') —
    the same number the JVM executor sizes LONG entries against, which is what
    rebalance_book's min-notional guard expects. 0.0 when the row is absent."""
    val = await conn.fetchval(
        "SELECT balance FROM portfolio WHERE account_id = $1::uuid AND asset = 'USDT'",
        account_id,
    )
    return float(val) if val is not None else 0.0


async def run_rebalance_once(db: "Database", settings: Settings) -> dict[str, Any]:
    """One rebalance pass over every live account. Returns a summary dict.

    Per-account exceptions are caught and recorded (never re-raised) so a single
    account's failure cannot abort the pass.
    """
    summary: dict[str, Any] = {
        "accounts": 0,
        "applied": 0,
        "no_change": 0,
        "skipped": 0,
        "errors": 0,
        "results": [],
    }
    async with db.acquire() as conn:
        account_ids = await _live_account_ids(conn)
        summary["accounts"] = len(account_ids)
        for account_id in account_ids:
            usdt = await _free_usdt(conn, account_id)
            if usdt <= 0:
                summary["skipped"] += 1
                summary["results"].append(
                    {"account_id": account_id, "status": "skipped_no_usdt"}
                )
                continue
            try:
                result = await rebalance_book(
                    conn,
                    account_id=account_id,
                    available_usdt=usdt,
                    optimizer=settings.rebalance_optimizer,
                    dry_run=settings.rebalance_dry_run,
                    agent_name="orchestrator-rebalance-cron",
                )
                if bool(result.get("applied")):
                    summary["applied"] += 1
                    status = "applied"
                else:
                    summary["no_change"] += 1
                    status = "no_change"
                summary["results"].append(
                    {
                        "account_id": account_id,
                        "status": status,
                        "n_updated": result.get("n_updated"),
                    }
                )
            except Exception as exc:  # noqa: BLE001 — isolate per-account failure
                summary["errors"] += 1
                summary["results"].append(
                    {"account_id": account_id, "status": "error", "error": repr(exc)}
                )
                log.warning(
                    "rebalance_cron.account_error",
                    account_id=account_id,
                    error=repr(exc),
                )
    return summary


async def rebalance_loop(db: "Database", settings: Settings) -> None:
    """Infinite loop: sleep the configured interval, then run one pass."""
    consecutive_failures = 0
    first_run = True
    while True:
        try:
            await asyncio.sleep(
                FIRST_RUN_DELAY_S if first_run else settings.rebalance_interval_seconds
            )
            first_run = False
            log.info(
                "rebalance_cron.start",
                optimizer=settings.rebalance_optimizer,
                dry_run=settings.rebalance_dry_run,
            )
            summary = await run_rebalance_once(db, settings)
            consecutive_failures = 0
            log.info(
                "rebalance_cron.complete",
                accounts=summary["accounts"],
                applied=summary["applied"],
                no_change=summary["no_change"],
                skipped=summary["skipped"],
                errors=summary["errors"],
            )
        except asyncio.CancelledError:
            log.info("rebalance_cron.shutdown")
            raise
        except Exception as exc:  # noqa: BLE001
            consecutive_failures += 1
            if consecutive_failures >= CONSECUTIVE_FAILURES_TO_ESCALATE:
                log.error(
                    "rebalance_cron.persistent_failure",
                    error=repr(exc),
                    consecutive_failures=consecutive_failures,
                )
            else:
                log.warning("rebalance_cron.error", error=repr(exc))


async def start_rebalance_task(db: "Database", settings: Settings) -> asyncio.Task[None]:
    """Create and return the rebalance background task (not awaited by caller)."""
    task = asyncio.create_task(rebalance_loop(db, settings))
    log.info(
        "rebalance_cron.task_started",
        interval_s=settings.rebalance_interval_seconds,
        optimizer=settings.rebalance_optimizer,
        dry_run=settings.rebalance_dry_run,
    )
    return task
