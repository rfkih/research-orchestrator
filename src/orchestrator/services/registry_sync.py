"""Auto-sync the Strategy Research Registry from the research loop.

Called best-effort from the tick (every iteration) and walk-forward (on a
stability verdict) so the registry always reflects the latest research — the
agent never has to inject SQL.

Two invariants:
  * BEST-EFFORT — callers wrap this; a failure here must never affect a tick.
  * CURATION-SAFE — the loop only creates/updates rows it OWNS
    (`auto_managed=TRUE`); curated rows (the hand-ranked seed + any dashboard-
    admin edit, `auto_managed=FALSE`) are never modified.
"""
from __future__ import annotations

import asyncpg

from ..repo import strategy_registry as registry_repo

# (promise_tier, verdict_tag, lifecycle_status) derived from a research verdict.
# Coarse by design — auto rows are a starting point the operator can curate
# (which flips them to auto_managed=FALSE and freezes them from further sync).
_ROBUST = ("TIER_A", "REAL_LEAD", "LEAD")
_DEAD = ("TIER_C", "FALSIFIED", "FALSIFIED")
_THIN = ("TIER_B", "REAL_UNCERTIFIABLE", "PARKED")
_EDGE = ("TIER_B", "REAL_UNCERTIFIABLE", "LEAD")


def derive_status(
    statistical_verdict: str | None, walk_forward_verdict: str | None = None
) -> tuple[str, str, str] | None:
    """Map a research verdict to (promise_tier, verdict_tag, lifecycle_status).

    Walk-forward (out-of-sample) wins when present. Returns None for an
    inconclusive result (NOT_TESTED / unknown) → the caller writes nothing.
    """
    wf = (walk_forward_verdict or "").upper()
    sv = (statistical_verdict or "").upper()
    if wf == "ROBUST":
        return _ROBUST
    if wf in ("OVERFIT", "NO_EDGE"):
        return _DEAD
    if wf == "INCONSISTENT":
        return _THIN
    # No / inconclusive walk-forward → fall back to the iteration verdict.
    if sv == "SIGNIFICANT_EDGE":
        return _EDGE
    if sv == "INSUFFICIENT_EVIDENCE":
        return _THIN
    if sv == "NO_EDGE":
        return _DEAD
    return None  # NOT_TESTED / unknown → no conclusion


def _slug(strategy_code: str, symbol: str | None, interval: str | None) -> str:
    parts = [strategy_code]
    if symbol:
        parts.append(symbol)
    if interval:
        parts.append(interval)
    return "-".join(parts).lower().replace("_", "-")


def _display_name(strategy_code: str, symbol: str | None, interval: str | None) -> str:
    bits = [strategy_code]
    if symbol:
        bits.append(symbol)
    if interval:
        bits.append(interval)
    return " ".join(bits)


async def sync_from_research(
    conn,
    *,
    strategy_code: str,
    symbol: str | None,
    interval: str | None,
    statistical_verdict: str | None,
    walk_forward_verdict: str | None = None,
    hypothesis: str | None = None,
    signal_family: str | None = None,
    agent: str = "quant-researcher",
) -> str:
    """Upsert the agent-owned registry row for one research result.

    Returns: 'created' | 'updated' | 'skipped' (curated row) | 'no-op' (no
    strategy_code or inconclusive verdict).
    """
    if not strategy_code:
        return "no-op"
    derived = derive_status(statistical_verdict, walk_forward_verdict)
    if derived is None:
        return "no-op"
    tier, verdict_tag, lifecycle = derived

    existing = await registry_repo.find_registry_by_research_key(
        conn, strategy_code, symbol, interval
    )
    if existing is not None:
        if not existing["auto_managed"]:
            return "skipped"  # curated / human-owned — never clobber
        patch: dict[str, object] = {
            "promise_tier": tier,
            "verdict_tag": verdict_tag,
            "lifecycle_status": lifecycle,
        }
        if signal_family:
            patch["signal_family"] = signal_family
        await registry_repo.update_registry(conn, existing["registry_id"], patch, agent)
        return "updated"

    fields = {
        "slug": _slug(strategy_code, symbol, interval),
        "promise_tier": tier,
        "verdict_tag": verdict_tag,
        "lifecycle_status": lifecycle,
        "display_name": _display_name(strategy_code, symbol, interval),
        "strategy_code": strategy_code,
        "symbol": symbol,
        "interval_name": interval,
        "signal_family": signal_family,
        "thesis": (hypothesis or f"Auto-tracked from research ({strategy_code}).").strip()[:480],
        "is_offline_lead": False,
    }
    try:
        await registry_repo.insert_registry(conn, fields, agent, auto_managed=True)
        return "created"
    except asyncpg.UniqueViolationError:
        # The derived slug collides with an existing (likely curated) row whose
        # research key didn't match — leave that row alone.
        return "skipped"
