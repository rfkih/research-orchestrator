"""``/strategy-registry`` — curated researched-strategy roster + live metrics.

A hand-curated, ranked roster of every strategy the platform has researched
(most -> least promising) with a qualitative verdict the raw metrics tables
cannot express, LEFT-JOINed at read time to LIVE metrics (DSR / walk-forward
verdict / annualized return / live status). Powers the admin "Strategy Research
Registry" page; reached from the dashboard via the trading-JVM proxy at
``/api/v1/research-orch/strategy-registry``.

GET is read-only; POST/PATCH/DELETE are admin-gated on the proxy-injected
``X-Viewer-Is-Admin`` header (same trust model as in-progress papers).
"""
from __future__ import annotations

from typing import Any, Literal, Optional

import asyncpg
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from ..errors import OrchestratorError
from ..repo import strategy_registry as registry_repo
from .deps import get_agent_name, get_db_conn, get_viewer_is_admin

router = APIRouter(tags=["strategy-registry"])

# Read-only context for the divergence flag below — NOT a gate. The V11 DSR gate
# constant (0.90 since 2026-06-15) is mirrored here only to flag curated rows
# whose live number now disagrees with their narrative. Keep in sync if the gate
# ever moves; it does not import the gate to avoid coupling this read path to it.
_DSR_GATE = 0.90

PromiseTier = Literal["TIER_A", "TIER_B", "TIER_C"]
VerdictTag = Literal[
    "REAL_LEAD", "REAL_UNCERTIFIABLE", "BETA_NOT_ALPHA", "DATA_GATED",
    "PARKED", "FALSIFIED", "FALSIFIED_OOS", "EXHAUSTED",
]
LifecycleStatus = Literal["LIVE", "LEAD", "PARKED", "DATA_GATED", "FALSIFIED"]

_TIER_KEYS = ("TIER_A", "TIER_B", "TIER_C")
_STATUS_KEYS = ("LIVE", "LEAD", "PARKED", "DATA_GATED", "FALSIFIED")
_REAL_VERDICTS = frozenset({"REAL_LEAD", "REAL_UNCERTIFIABLE", "BETA_NOT_ALPHA"})
_DEAD_VERDICTS = frozenset({"FALSIFIED", "FALSIFIED_OOS", "PARKED", "EXHAUSTED"})


# ── pure helpers (DB-free; unit-tested directly) ───────────────────────────

def _f(value: Any) -> float | None:
    """Decimal/None -> float/None (asyncpg returns NUMERIC as Decimal)."""
    return float(value) if value is not None else None


def compute_divergence(
    lifecycle_status: str, verdict_tag: str, dsr: float | None, is_live: bool
) -> tuple[bool, str | None]:
    """Flag when the curated narrative and a fresh DB number disagree.

    Two cases worth an operator's attention:
      * curated LIVE but nothing is actually trading it; and
      * curated dead/parked but a live run now clears the DSR gate (resurrection).
    """
    if lifecycle_status == "LIVE" and not is_live:
        return True, "Marked LIVE but no enabled account_strategy is trading it."
    if verdict_tag in _DEAD_VERDICTS and dsr is not None and dsr >= _DSR_GATE:
        return (
            True,
            f"Marked {verdict_tag} but a live run now clears the {_DSR_GATE:.2f} "
            f"DSR gate (DSR {dsr:.3f}) — re-examine.",
        )
    return False, None


def to_item(row: dict[str, Any]) -> dict[str, Any]:
    """Map a merged repo row (curated + live) to the camelCase API item."""
    dsr = _f(row.get("dsr"))
    is_live = bool(row.get("is_live"))
    flag, reason = compute_divergence(
        row["lifecycle_status"], row["verdict_tag"], dsr, is_live
    )
    return {
        "registryId": row["registry_id"],
        "slug": row["slug"],
        "rank": row.get("rank"),
        "promiseTier": row["promise_tier"],
        "displayName": row["display_name"],
        "signalFamily": row.get("signal_family"),
        "strategyCode": row.get("strategy_code"),
        "symbol": row.get("symbol"),
        "intervalName": row.get("interval_name"),
        "verdictTag": row["verdict_tag"],
        "lifecycleStatus": row["lifecycle_status"],
        "thesis": row["thesis"],
        "detail": row.get("detail"),
        "memoryRef": row.get("memory_ref"),
        "isOfflineLead": bool(row.get("is_offline_lead")),
        "archived": bool(row.get("archived")),
        "autoManaged": bool(row.get("auto_managed")),
        "evidence": {
            "iterationId": row.get("evidence_iteration_id"),
            "walkForwardId": row.get("evidence_walk_forward_id"),
            "backtestRunId": row.get("evidence_backtest_run_id"),
            "journalId": row.get("journal_id"),
        },
        "live": {
            "dsr": dsr,
            "psr": _f(row.get("psr")),
            "annualizedReturnPct": _f(row.get("ann_return_pct")),
            "sharpeAnnualized": _f(row.get("sharpe_ann")),
            "nTrades": row.get("n_trades"),
            "profitFactor": _f(row.get("profit_factor")),
            "statisticalVerdict": row.get("statistical_verdict"),
            "walkForwardVerdict": row.get("walk_forward_verdict"),
            "isLive": is_live,
            "resolvedFrom": row.get("resolved_from"),
            "backtestRunId": row.get("live_backtest_run_id"),
        },
        "divergence": {"flag": flag, "reason": reason},
        "updatedTime": row.get("updated_time"),
        "updatedBy": row.get("updated_by"),
    }


def tally(items: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, int]]:
    tier_counts = {k: 0 for k in _TIER_KEYS}
    status_counts = {k: 0 for k in _STATUS_KEYS}
    for it in items:
        tier_counts[it["promiseTier"]] = tier_counts.get(it["promiseTier"], 0) + 1
        status_counts[it["lifecycleStatus"]] = (
            status_counts.get(it["lifecycleStatus"], 0) + 1
        )
    return tier_counts, status_counts


# Who may MUTATE the registry. Every caller already holds the X-Orch-Token (the
# real boundary; the orchestrator is loopback-only). The X-Viewer-Is-Admin gate
# exists only to stop non-admin DASHBOARD end-users (their requests arrive via the
# proxy stamped X-Agent-Name=dashboard). A trusted server-side AGENT presents its
# own X-Agent-Name (e.g. quant-researcher) and IS allowed to write — that is how
# the agent maintains the roster through the API instead of injecting SQL.
_NON_WRITER_AGENTS = frozenset({"dashboard", "anonymous", ""})


def _require_writer(is_admin: bool, agent: str) -> None:
    if is_admin or (agent and agent not in _NON_WRITER_AGENTS):
        return
    raise OrchestratorError(
        status_code=403,
        error_code="writer_required",
        message="Modifying the strategy registry requires a dashboard admin or a named agent.",
        retryable=False,
        hint=(
            "Dashboard users need an admin JWT (the proxy sets X-Viewer-Is-Admin); "
            "server-side callers must send a non-dashboard X-Agent-Name with the orch token."
        ),
    )


# ── request bodies ─────────────────────────────────────────────────────────

class RegistryCreate(BaseModel):
    model_config = {"extra": "forbid"}
    slug: str
    promise_tier: PromiseTier
    display_name: str
    verdict_tag: VerdictTag
    lifecycle_status: LifecycleStatus
    thesis: str
    rank: int | None = None
    signal_family: str | None = None
    strategy_code: str | None = None
    symbol: str | None = None
    interval_name: str | None = None
    detail: str | None = None
    evidence_iteration_id: str | None = None
    evidence_walk_forward_id: str | None = None
    evidence_backtest_run_id: str | None = None
    journal_id: str | None = None
    memory_ref: str | None = None
    is_offline_lead: bool = False


class RegistryUpdate(BaseModel):
    """All optional — only supplied fields are applied (``exclude_unset``)."""
    model_config = {"extra": "forbid"}
    slug: str | None = None
    rank: int | None = None
    promise_tier: PromiseTier | None = None
    display_name: str | None = None
    signal_family: str | None = None
    strategy_code: str | None = None
    symbol: str | None = None
    interval_name: str | None = None
    verdict_tag: VerdictTag | None = None
    lifecycle_status: LifecycleStatus | None = None
    thesis: str | None = None
    detail: str | None = None
    evidence_iteration_id: str | None = None
    evidence_walk_forward_id: str | None = None
    evidence_backtest_run_id: str | None = None
    journal_id: str | None = None
    memory_ref: str | None = None
    is_offline_lead: bool | None = None
    archived: bool | None = None


# ── routes ───────────────────────────────────────────────────────────────

@router.get("/strategy-registry")
async def list_strategy_registry(
    tier: Optional[PromiseTier] = Query(None, description="Filter by promise tier."),
    status: Optional[LifecycleStatus] = Query(None, description="Filter by lifecycle status."),
    family: Optional[str] = Query(None, description="Filter by signal_family."),
    search: Optional[str] = Query(None, description="Substring over name / code / thesis."),
    include_archived: bool = Query(False, description="Include soft-deleted rows."),
    conn: asyncpg.Connection = Depends(get_db_conn),
) -> dict[str, Any]:
    rows = await registry_repo.list_registry(
        conn,
        tier=tier,
        status=status,
        family=family,
        search=search,
        include_archived=include_archived,
    )
    items = [to_item(r) for r in rows]
    tier_counts, status_counts = tally(items)
    return {
        "items": items,
        "total": len(items),
        "tierCounts": tier_counts,
        "statusCounts": status_counts,
    }


@router.get("/strategy-registry/{registry_id}")
async def get_strategy_registry(
    registry_id: str,
    conn: asyncpg.Connection = Depends(get_db_conn),
) -> dict[str, Any]:
    row = await registry_repo.get_registry(conn, registry_id)
    if row is None:
        raise OrchestratorError(
            status_code=404,
            error_code="registry_not_found",
            message=f"No strategy-registry entry with id {registry_id}.",
            retryable=False,
        )
    return to_item(row)


@router.post("/strategy-registry", status_code=201)
async def create_strategy_registry(
    body: RegistryCreate,
    is_admin: bool = Depends(get_viewer_is_admin),
    agent: str = Depends(get_agent_name),
    conn: asyncpg.Connection = Depends(get_db_conn),
) -> dict[str, Any]:
    _require_writer(is_admin, agent)
    try:
        # A dashboard-admin create is curated (auto_managed=FALSE); a server-side
        # agent create is loop-owned (TRUE) so the auto-sync may keep it current.
        row = await registry_repo.insert_registry(
            conn, body.model_dump(), agent, auto_managed=not is_admin
        )
    except asyncpg.UniqueViolationError as exc:
        raise OrchestratorError(
            status_code=409,
            error_code="slug_exists",
            message=f"A registry entry with slug '{body.slug}' already exists.",
            retryable=False,
        ) from exc
    return to_item(row)  # type: ignore[arg-type]  # insert returns the merged row


@router.patch("/strategy-registry/{registry_id}")
async def update_strategy_registry(
    registry_id: str,
    body: RegistryUpdate,
    is_admin: bool = Depends(get_viewer_is_admin),
    agent: str = Depends(get_agent_name),
    conn: asyncpg.Connection = Depends(get_db_conn),
) -> dict[str, Any]:
    _require_writer(is_admin, agent)
    patch = body.model_dump(exclude_unset=True)
    if is_admin:
        patch["auto_managed"] = False  # a human edit claims the row from the auto-sync
    try:
        row = await registry_repo.update_registry(conn, registry_id, patch, agent)
    except asyncpg.UniqueViolationError as exc:
        raise OrchestratorError(
            status_code=409,
            error_code="slug_exists",
            message="That slug is already in use by another registry entry.",
            retryable=False,
        ) from exc
    if row is None:
        raise OrchestratorError(
            status_code=404,
            error_code="registry_not_found",
            message=f"No strategy-registry entry with id {registry_id}.",
            retryable=False,
        )
    return to_item(row)


@router.delete("/strategy-registry/{registry_id}")
async def archive_strategy_registry(
    registry_id: str,
    is_admin: bool = Depends(get_viewer_is_admin),
    agent: str = Depends(get_agent_name),
    conn: asyncpg.Connection = Depends(get_db_conn),
) -> dict[str, Any]:
    _require_writer(is_admin, agent)
    archived = await registry_repo.archive_registry(conn, registry_id, agent)
    if not archived:
        raise OrchestratorError(
            status_code=404,
            error_code="registry_not_found",
            message=f"No strategy-registry entry with id {registry_id}.",
            retryable=False,
        )
    return {"registryId": registry_id, "archived": True}


# ── agent-facing by-slug surface ─────────────────────────────────────────────
# The dashboard edits by registry_id (it has the UUIDs); agents address entries
# by their stable `slug`. These give a seamless, idempotent maintenance API so an
# agent never has to inject SQL: PUT upserts (insert-or-merge), DELETE archives.

# Fields a brand-new entry must carry (an upsert that CREATES requires them; an
# upsert that UPDATES an existing slug may send any subset).
_CREATE_REQUIRED = ("promise_tier", "display_name", "verdict_tag", "lifecycle_status", "thesis")


@router.get("/strategy-registry/by-slug/{slug}")
async def get_strategy_registry_by_slug(
    slug: str,
    conn: asyncpg.Connection = Depends(get_db_conn),
) -> dict[str, Any]:
    row = await registry_repo.get_registry_by_slug(conn, slug)
    if row is None:
        raise OrchestratorError(
            status_code=404,
            error_code="registry_not_found",
            message=f"No strategy-registry entry with slug '{slug}'.",
            retryable=False,
        )
    return to_item(row)


@router.put("/strategy-registry/by-slug/{slug}")
async def upsert_strategy_registry_by_slug(
    slug: str,
    body: RegistryUpdate,
    is_admin: bool = Depends(get_viewer_is_admin),
    agent: str = Depends(get_agent_name),
    conn: asyncpg.Connection = Depends(get_db_conn),
) -> dict[str, Any]:
    """Idempotent upsert by slug. Existing slug → merge the supplied fields;
    new slug → insert (requires the create fields). The slug in the path is
    authoritative (any `slug` in the body is ignored). Returns the merged entry.
    """
    _require_writer(is_admin, agent)
    fields = body.model_dump(exclude_unset=True)
    fields.pop("slug", None)  # path wins
    if is_admin:
        fields["auto_managed"] = False  # admin edits (even via by-slug) claim the row

    existing = await registry_repo.get_registry_by_slug(conn, slug)
    if existing is not None:
        row = await registry_repo.update_registry(conn, existing["registry_id"], fields, agent)
        return to_item(row)  # type: ignore[arg-type]

    missing = [f for f in _CREATE_REQUIRED if not fields.get(f)]
    if missing:
        raise OrchestratorError(
            status_code=422,
            error_code="registry_missing_fields",
            message=f"Creating '{slug}' requires: {', '.join(missing)}.",
            retryable=False,
            hint="A PUT on a NEW slug inserts — supply the required fields. PUT on an existing slug may send any subset.",
        )
    try:
        row = await registry_repo.insert_registry(
            conn, {**fields, "slug": slug}, agent, auto_managed=not is_admin
        )
    except asyncpg.UniqueViolationError:
        # Lost a create race — the row now exists; fall back to update.
        existing = await registry_repo.get_registry_by_slug(conn, slug)
        row = await registry_repo.update_registry(conn, existing["registry_id"], fields, agent)  # type: ignore[index]
    return to_item(row)  # type: ignore[arg-type]


@router.delete("/strategy-registry/by-slug/{slug}")
async def archive_strategy_registry_by_slug(
    slug: str,
    is_admin: bool = Depends(get_viewer_is_admin),
    agent: str = Depends(get_agent_name),
    conn: asyncpg.Connection = Depends(get_db_conn),
) -> dict[str, Any]:
    _require_writer(is_admin, agent)
    existing = await registry_repo.get_registry_by_slug(conn, slug)
    if existing is None:
        raise OrchestratorError(
            status_code=404,
            error_code="registry_not_found",
            message=f"No strategy-registry entry with slug '{slug}'.",
            retryable=False,
        )
    await registry_repo.archive_registry(conn, existing["registry_id"], agent)
    return {"slug": slug, "archived": True}
