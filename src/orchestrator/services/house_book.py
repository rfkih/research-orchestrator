"""House Book live execution (Phase 1-A) — sync signal-pool weights onto the
book account's ``account_strategy`` rows.

The signal pool (Phase 0) decides membership + HRP weights. This module is the
overlay that makes the book *tradable*: it propagates ``signal_pool.pool_weight``
onto the matching ``account_strategy.portfolio_weight`` (the V109 column the
live sizer already multiplies into every entry via ``applyPortfolioWeight``).

Boundary (deliberate, mirrors ``rebalance.py``): this only **UPDATEs weights on
rows that already exist** on the dedicated House Book account. It never creates
live-trading rows and never touches admin's account. A pool member with no live
row is reported as ``unmaterialized`` — the operator provisions it through the
trading JVM's normal strategy-creation flow, then a re-sync picks it up. A row
that is no longer an active member is soft-disabled by zeroing its weight
(reversible; ``enabled`` is left untouched so the operator stays in control of
real-capital on/off).
"""
from __future__ import annotations

from typing import Any
from uuid import UUID

import asyncpg

_WEIGHT_SOURCE = "HOUSE_BOOK"
_WEIGHT_SOURCE_EVICTED = "HOUSE_BOOK_EVICTED"
# Rows the book is allowed to soft-disable on eviction — only ones it wrote.
_BOOK_WEIGHT_SOURCES = frozenset({_WEIGHT_SOURCE, _WEIGHT_SOURCE_EVICTED})


def _key(strategy_code: str, symbol: str, interval_name: str) -> str:
    return f"{strategy_code}:{symbol}:{interval_name}"


def classify_book(
    members: list[dict[str, Any]],
    live_rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Pure reconciliation of active pool members against the book account's
    live ``account_strategy`` rows. Buckets:

    * ``to_sync``       — member has a live row AND a non-null pool_weight →
                          set portfolio_weight = pool_weight.
    * ``needs_weight``  — member exists but pool_weight is NULL (run
                          /pool/rebalance first) → skipped, reported.
    * ``unmaterialized``— member has no live row → operator must provision it.
    * ``to_zero``       — a row THIS BOOK set (weight_source HOUSE_BOOK*) that
                          is NOT an active member and still carries weight>0 →
                          soft-disable (evicted/removed member).

    Membership key is (strategy_code, symbol, interval_name).

    SAFETY: ``to_zero`` is restricted to rows the book itself wrote
    (``weight_source`` in the HOUSE_BOOK family). A row the book never touched
    is NEVER zeroed — so a misconfigured ``ORCH_HOUSE_BOOK_ACCOUNT_ID`` pointed
    at an account holding other strategies cannot wipe their weights.
    """
    live_by_key = {r["key"]: r for r in live_rows}
    member_keys: set[str] = set()
    to_sync: list[dict[str, Any]] = []
    needs_weight: list[dict[str, Any]] = []
    unmaterialized: list[dict[str, Any]] = []

    for m in members:
        member_keys.add(m["key"])
        if m["pool_weight"] is None:
            needs_weight.append(m)
            continue
        row = live_by_key.get(m["key"])
        if row is None:
            unmaterialized.append(m)
        else:
            to_sync.append({
                **m,
                "account_strategy_id": row["account_strategy_id"],
                "current_weight": row["portfolio_weight"],
            })

    to_zero = [
        r for r in live_rows
        if r["key"] not in member_keys
        and float(r["portfolio_weight"] or 0) > 0
        and r.get("weight_source") in _BOOK_WEIGHT_SOURCES
    ]
    return {
        "to_sync": to_sync,
        "needs_weight": needs_weight,
        "unmaterialized": unmaterialized,
        "to_zero": to_zero,
    }


async def _active_members(conn: asyncpg.Connection) -> list[dict[str, Any]]:
    rows = await conn.fetch(
        """
        SELECT strategy_code, symbol, interval_name, pool_weight
          FROM signal_pool
         WHERE status = 'active' AND kind = 'signal_pool'
        """
    )
    return [
        {
            "key": _key(r["strategy_code"], r["symbol"], r["interval_name"]),
            "strategy_code": r["strategy_code"],
            "symbol": r["symbol"],
            "interval_name": r["interval_name"],
            "pool_weight": float(r["pool_weight"]) if r["pool_weight"] is not None else None,
        }
        for r in rows
    ]


async def _book_rows(conn: asyncpg.Connection, account_id: UUID) -> list[dict[str, Any]]:
    rows = await conn.fetch(
        """
        SELECT account_strategy_id, strategy_code, symbol, interval_name,
               portfolio_weight, weight_source, enabled, simulated
          FROM account_strategy
         WHERE account_id = $1 AND is_deleted = false
        """,
        account_id,
    )
    return [
        {
            "key": _key(r["strategy_code"], r["symbol"], r["interval_name"]),
            "account_strategy_id": r["account_strategy_id"],
            "strategy_code": r["strategy_code"],
            "symbol": r["symbol"],
            "interval_name": r["interval_name"],
            "portfolio_weight": float(r["portfolio_weight"]) if r["portfolio_weight"] is not None else None,
            "weight_source": r["weight_source"],
            "enabled": r["enabled"],
            "simulated": r["simulated"],
        }
        for r in rows
    ]


async def reconcile_plan(
    conn: asyncpg.Connection, account_id: UUID
) -> dict[str, Any]:
    """Read the pool + the book account's rows and return the classification
    (no writes)."""
    members = await _active_members(conn)
    live = await _book_rows(conn, account_id)
    plan = classify_book(members, live)
    plan["n_members"] = len(members)
    plan["n_live_rows"] = len(live)
    return plan


async def apply_sync(
    conn: asyncpg.Connection,
    account_id: UUID,
    *,
    agent: str,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Propagate pool weights onto the book account. Idempotent. When
    ``dry_run`` (the default), classifies and reports but writes nothing."""
    plan = await reconcile_plan(conn, account_id)

    applied = 0
    zeroed = 0
    if not dry_run:
        for s in plan["to_sync"]:
            await conn.execute(
                """
                UPDATE account_strategy
                   SET portfolio_weight  = $1,
                       weight_source     = $2,
                       weight_updated_at = NOW(),
                       updated_time      = NOW(),
                       updated_by        = $3
                 WHERE account_strategy_id = $4
                """,
                round(s["pool_weight"], 5), _WEIGHT_SOURCE, agent,
                s["account_strategy_id"],
            )
            applied += 1
        for z in plan["to_zero"]:
            await conn.execute(
                """
                UPDATE account_strategy
                   SET portfolio_weight  = 0,
                       weight_source     = $1,
                       weight_updated_at = NOW(),
                       updated_time      = NOW(),
                       updated_by        = $2
                 WHERE account_strategy_id = $3
                """,
                _WEIGHT_SOURCE_EVICTED, agent, z["account_strategy_id"],
            )
            zeroed += 1
        await _journal(conn, account_id, plan, applied, zeroed, agent)

    # Capital actually deployed = Σ synced pool_weights. <1.0 means some members
    # are still unmaterialized (their weight isn't on the book yet) — surfaced so
    # an under-invested book isn't mistaken for a fully-deployed one.
    deployed = sum(s["pool_weight"] for s in plan["to_sync"])
    return {
        "applied": not dry_run,
        "dry_run": dry_run,
        "account_id": str(account_id),
        "n_members": plan["n_members"],
        "n_live_rows": plan["n_live_rows"],
        "n_synced": applied if not dry_run else len(plan["to_sync"]),
        "n_zeroed": zeroed if not dry_run else len(plan["to_zero"]),
        "deployed_fraction": round(deployed, 5),
        "undeployed_fraction": round(max(0.0, 1.0 - deployed), 5),
        "synced": [
            {"surface": s["key"], "weight": round(s["pool_weight"], 5),
             "from": s["current_weight"]}
            for s in plan["to_sync"]
        ],
        "zeroed": [{"surface": z["key"], "from": z["portfolio_weight"]} for z in plan["to_zero"]],
        "unmaterialized": [
            {"surface": m["key"], "pool_weight": round(m["pool_weight"], 5)}
            for m in plan["unmaterialized"]
        ],
        "needs_rebalance": [{"surface": m["key"]} for m in plan["needs_weight"]],
    }


async def _journal(
    conn: asyncpg.Connection,
    account_id: UUID,
    plan: dict[str, Any],
    applied: int,
    zeroed: int,
    agent: str,
) -> None:
    """Audit the sync into research_journal (reuses the CROSS_STRATEGY_FINDING
    entry_type, as rebalance.py does — House Book sync is cross-strategy)."""
    title = f"House Book sync — {applied} synced, {zeroed} zeroed"
    content = (
        f"Account: {account_id}\n"
        f"Members: {plan['n_members']}  Live rows: {plan['n_live_rows']}\n"
        f"Synced: {applied}  Zeroed (evicted): {zeroed}  "
        f"Unmaterialized: {len(plan['unmaterialized'])}  "
        f"Needs-rebalance: {len(plan['needs_weight'])}"
    )
    structured = {
        "account_id": str(account_id),
        "n_synced": applied,
        "n_zeroed": zeroed,
        "synced": [{"surface": s["key"], "weight": s["pool_weight"]} for s in plan["to_sync"]],
        "unmaterialized": [m["key"] for m in plan["unmaterialized"]],
        "needs_rebalance": [m["key"] for m in plan["needs_weight"]],
    }
    await conn.execute(
        """
        INSERT INTO research_journal
          (journal_id, entry_type, title, content, structured_data,
           status, created_by, created_time)
        VALUES (gen_random_uuid(), 'CROSS_STRATEGY_FINDING', $1, $2, $3,
                'ACTIVE', $4, NOW())
        """,
        title, content, structured, agent,
    )
