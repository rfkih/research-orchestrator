"""Signal Pool / House Book API (Phase 0).

Three endpoints over the ``signal_pool`` table (Flyway V147):

* ``POST /pool/evaluate {iteration_id}`` — run the admission rule (validity +
  marginal-Sharpe contribution) on a validated candidate; insert a member row
  on admit. Idempotent per active surface.
* ``GET /pool`` — list active members + weights + admission metrics.
* ``POST /pool/rebalance`` — HRP-weight the active members, write pool_weight.

All deterministic; no specialist agent. The standalone V11/V60 gate is
untouched — this is the additive pool lane.
"""
from __future__ import annotations

from typing import Any
from uuid import UUID

import asyncpg
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from ..errors import OrchestratorError
from ..services import house_book
from ..services import pool as pool_svc
from ..services.idempotency import cache_response, replay_cached_response
from ..services.pool import DEFAULT_THETA, _active_members_returns, _hrp_for
from .deps import get_agent_name, get_db_conn

router = APIRouter(prefix="/pool", tags=["pool"])

# Weight clamp — same band as the account_strategy rebalance guardrails so a
# single pool member can't dominate or be dust.
_MIN_W = 0.05
_MAX_W = 0.50


class EvaluateBody(BaseModel):
    iteration_id: UUID
    theta: float = Field(DEFAULT_THETA, ge=0.0, le=5.0)


class SyncBody(BaseModel):
    # Safe by default: a bare POST /pool/sync only previews. Live weights move
    # only when the caller explicitly sets dry_run=false.
    dry_run: bool = True


class ReconcileBody(BaseModel):
    # Preview by default. Evicts active members whose marginal Sharpe vs the
    # rest of the pool has fallen to/below evict_theta (gone redundant).
    dry_run: bool = True
    evict_theta: float = Field(0.0, ge=-5.0, le=5.0)


async def _existing_active(conn: asyncpg.Connection, ctx: dict[str, Any]) -> Any:
    return await conn.fetchval(
        """
        SELECT pool_id FROM signal_pool
         WHERE status = 'active'
           AND strategy_code = $1 AND symbol = $2 AND interval_name = $3
        """,
        ctx["strategy_code"], ctx["symbol"], ctx["interval_name"],
    )


@router.post("/evaluate")
async def evaluate(
    body: EvaluateBody,
    request: Request,
    conn: asyncpg.Connection = Depends(get_db_conn),
    agent: str = Depends(get_agent_name),
) -> dict[str, Any]:
    cached, idem_key = await replay_cached_response(request, agent, "pool_evaluate")
    if cached is not None:
        return cached

    verdict = await pool_svc.evaluate_admission(conn, body.iteration_id, theta=body.theta)
    ctx = verdict.get("ctx")
    if not verdict["admit"]:
        return {
            "admitted": False,
            "reason": verdict["reason"],
            "iteration_id": str(body.iteration_id),
            "contribution": verdict.get("contribution"),
        }

    metrics = {
        "dsr": ctx.get("dsr"),
        "statistical_verdict": ctx.get("statistical_verdict"),
        **{k: verdict["contribution"][k] for k in
           ("marginal", "sharpe_before", "sharpe_after", "max_abs_corr", "n_overlap")},
        "theta": verdict["theta"],
    }
    # Atomic admission: ON CONFLICT against the active-surface partial-unique
    # index makes this race-safe (no check-then-insert gap). A conflict means a
    # concurrent admit already pooled this surface → report already_pooled.
    pool_id = await conn.fetchval(
        """
        INSERT INTO signal_pool
          (iteration_id, strategy_code, symbol, interval_name,
           admission_metrics, status, created_by, updated_by)
        VALUES ($1, $2, $3, $4, $5, 'active', $6, $6)
        ON CONFLICT (strategy_code, symbol, interval_name)
            WHERE status = 'active'
        DO NOTHING
        RETURNING pool_id
        """,
        body.iteration_id, ctx["strategy_code"], ctx["symbol"],
        ctx["interval_name"], metrics, agent,
    )
    if pool_id is None:
        existing = await _existing_active(conn, ctx)
        response = {
            "admitted": True,
            "reason": "already_pooled",
            "pool_id": str(existing) if existing is not None else None,
            "iteration_id": str(body.iteration_id),
        }
    else:
        response = {
            "admitted": True,
            "reason": "admitted",
            "pool_id": str(pool_id),
            "iteration_id": str(body.iteration_id),
            "surface": f"{ctx['strategy_code']}:{ctx['symbol']}:{ctx['interval_name']}",
            "contribution": verdict["contribution"],
        }
    await cache_response(request, agent, "pool_evaluate", idem_key, response)
    return response


@router.get("")
async def list_pool(
    conn: asyncpg.Connection = Depends(get_db_conn),
    _: str = Depends(get_agent_name),
) -> dict[str, Any]:
    rows = await conn.fetch(
        """
        SELECT pool_id, iteration_id, strategy_code, symbol, interval_name,
               admitted_at, admission_metrics, pool_weight, weight_source,
               weight_updated_at
          FROM signal_pool
         WHERE status = 'active'
         ORDER BY strategy_code, symbol, interval_name
        """
    )
    members = [
        {
            "pool_id": str(r["pool_id"]),
            "iteration_id": str(r["iteration_id"]),
            "surface": f"{r['strategy_code']}:{r['symbol']}:{r['interval_name']}",
            "strategy_code": r["strategy_code"],
            "symbol": r["symbol"],
            "interval_name": r["interval_name"],
            "admitted_at": r["admitted_at"].isoformat() if r["admitted_at"] else None,
            "pool_weight": float(r["pool_weight"]) if r["pool_weight"] is not None else None,
            "weight_source": r["weight_source"],
            "admission_metrics": r["admission_metrics"],
        }
        for r in rows
    ]
    return {"count": len(members), "members": members}


@router.post("/rebalance")
async def rebalance(
    request: Request,
    conn: asyncpg.Connection = Depends(get_db_conn),
    agent: str = Depends(get_agent_name),
) -> dict[str, Any]:
    """HRP-weight the active pool members from their admitted backtest return
    series, clamp to [0.05, 0.50], renormalise, and write ``pool_weight``."""
    cached, idem_key = await replay_cached_response(request, agent, "pool_rebalance")
    if cached is not None:
        return cached

    members = await conn.fetch(
        "SELECT pool_id, strategy_code, symbol, interval_name "
        "FROM signal_pool WHERE status = 'active'"
    )
    if not members:
        return {"applied": False, "reason": "empty_pool", "n_updated": 0}

    series = await _active_members_returns(conn)  # keyed code:symbol:interval
    if not series:
        return {"applied": False, "reason": "no_return_series", "n_updated": 0}

    raw = _hrp_for(series)
    clamped = {k: min(_MAX_W, max(_MIN_W, w)) for k, w in raw.items()}
    total = sum(clamped.values()) or 1.0
    weights = {k: w / total for k, w in clamped.items()}

    # Phase 1-B book risk guard: cap aggregate weight per symbol (diversify
    # across symbols, not just members) before persisting.
    symbol_by_key = {
        f"{m['strategy_code']}:{m['symbol']}:{m['interval_name']}": m["symbol"]
        for m in members
    }
    cap = request.app.state.settings.house_book_max_per_symbol
    weights, cap_diag = pool_svc.apply_per_symbol_cap(weights, symbol_by_key, cap)

    n = 0
    for m in members:
        key = f"{m['strategy_code']}:{m['symbol']}:{m['interval_name']}"
        w = weights.get(key)
        if w is None:
            continue
        await conn.execute(
            """
            UPDATE signal_pool
               SET pool_weight = $1, weight_source = 'HRP',
                   weight_updated_at = NOW(), updated_time = NOW(), updated_by = $2
             WHERE pool_id = $3
            """,
            round(w, 5), agent, m["pool_id"],
        )
        n += 1

    response = {
        "applied": True,
        "optimizer": "HRP",
        "n_updated": n,
        "weights": {k: round(v, 5) for k, v in weights.items()},
        "guardrails": {"min_weight": _MIN_W, "max_weight": _MAX_W,
                       "max_per_symbol": cap},
        "per_symbol_cap": cap_diag,
    }
    await cache_response(request, agent, "pool_rebalance", idem_key, response)
    return response


# ── House Book live execution (Phase 1-A) ────────────────────────────


@router.post("/sync")
async def sync_book(
    body: SyncBody,
    request: Request,
    conn: asyncpg.Connection = Depends(get_db_conn),
    agent: str = Depends(get_agent_name),
) -> dict[str, Any]:
    """Propagate signal_pool weights onto the House Book account's
    account_strategy rows. Preview-only unless body.dry_run=false. Members
    without a live row are reported as ``unmaterialized`` (operator provisions
    them via the trading JVM); evicted rows are soft-disabled (weight→0)."""
    account_id = request.app.state.settings.house_book_account_id
    if account_id is None:
        return {
            "applied": False,
            "reason": "not_configured",
            "hint": "Set ORCH_HOUSE_BOOK_ACCOUNT_ID to the dedicated book account.",
        }
    # Idempotency only matters for a real write; a dry-run preview is read-only.
    if not body.dry_run:
        cached, idem_key = await replay_cached_response(request, agent, "pool_sync")
        if cached is not None:
            return cached
        response = await house_book.apply_sync(
            conn, account_id, agent=agent, dry_run=False
        )
        await cache_response(request, agent, "pool_sync", idem_key, response)
        return response
    return await house_book.apply_sync(conn, account_id, agent=agent, dry_run=True)


@router.get("/book")
async def get_book(
    request: Request,
    conn: asyncpg.Connection = Depends(get_db_conn),
    _: str = Depends(get_agent_name),
) -> dict[str, Any]:
    """Live composition of the House Book: pool members reconciled against the
    book account's account_strategy rows (synced / unmaterialized / evictable)."""
    account_id = request.app.state.settings.house_book_account_id
    if account_id is None:
        return {"configured": False, "members": [], "count": 0}
    plan = await house_book.reconcile_plan(conn, account_id)
    return {
        "configured": True,
        "account_id": str(account_id),
        "n_members": plan["n_members"],
        "n_live_rows": plan["n_live_rows"],
        "synced": [
            {"surface": s["key"], "pool_weight": round(s["pool_weight"], 5),
             "live_weight": s["current_weight"],
             "account_strategy_id": str(s["account_strategy_id"])}
            for s in plan["to_sync"]
        ],
        "unmaterialized": [
            {"surface": m["key"], "pool_weight": round(m["pool_weight"], 5)}
            for m in plan["unmaterialized"]
        ],
        "needs_rebalance": [{"surface": m["key"]} for m in plan["needs_weight"]],
        "evictable": [
            {"surface": z["key"], "live_weight": z["portfolio_weight"],
             "account_strategy_id": str(z["account_strategy_id"])}
            for z in plan["to_zero"]
        ],
    }


# ── Phase 1-B: eviction lifecycle ────────────────────────────────────


@router.post("/reconcile")
async def reconcile(
    body: ReconcileBody,
    request: Request,
    conn: asyncpg.Connection = Depends(get_db_conn),
    agent: str = Depends(get_agent_name),
) -> dict[str, Any]:
    """Re-score each active member's marginal contribution vs the rest of the
    pool; evict members that have gone redundant (marginal ≤ evict_theta).
    Preview-only unless body.dry_run=false."""
    verdicts = await pool_svc.evaluate_members_for_eviction(conn, evict_theta=body.evict_theta)
    to_evict = [v for v in verdicts if v["evict"]]

    evicted = 0
    if not body.dry_run and to_evict:
        cached, idem_key = await replay_cached_response(request, agent, "pool_reconcile")
        if cached is not None:
            return cached
        for v in to_evict:
            await conn.execute(
                """
                UPDATE signal_pool
                   SET status = 'evicted', evicted_at = NOW(),
                       evicted_reason = $1, updated_time = NOW(), updated_by = $2
                 WHERE pool_id = $3 AND status = 'active'
                """,
                f"marginal {v['marginal']} ≤ evict_theta {body.evict_theta}",
                agent, v["pool_id"],
            )
            evicted += 1
        response = {
            "applied": True, "dry_run": False, "n_evicted": evicted,
            "evict_theta": body.evict_theta,
            "evicted": [{"surface": v["key"], "marginal": v["marginal"]} for v in to_evict],
            "verdicts": _redact_verdicts(verdicts),
        }
        await cache_response(request, agent, "pool_reconcile", idem_key, response)
        return response

    return {
        "applied": False, "dry_run": True,
        "evict_theta": body.evict_theta,
        "would_evict": [{"surface": v["key"], "marginal": v["marginal"]} for v in to_evict],
        "verdicts": _redact_verdicts(verdicts),
    }


def _redact_verdicts(verdicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"surface": v["key"], "marginal": v["marginal"],
         "evict": v["evict"], "reason": v["reason"]}
        for v in verdicts
    ]


# ── Phase 1-C: book performance + attribution ────────────────────────


@router.get("/performance")
async def performance(
    conn: asyncpg.Connection = Depends(get_db_conn),
    _: str = Depends(get_agent_name),
) -> dict[str, Any]:
    """Combined HRP-weighted book performance vs each member standalone, with
    per-member attribution and the combined equity curve. Headline:
    ``combined_beats_best_member`` — the diversification payoff the pool exists
    to capture."""
    return await pool_svc.book_performance(conn)
