"""Unit tests for the Strategy Research Registry endpoint (2026-06-16).

DB-free: pure helpers (divergence / mapping / counts) are tested directly, and
the HTTP routes are exercised with a fake connection via dependency_overrides
(mirrors tests/test_provisional_roster.py). The live-metrics SQL join is
covered separately by tests/integration/test_strategy_registry_db.py.
"""
from __future__ import annotations

from typing import Any

import asyncpg
import pytest
from fastapi.testclient import TestClient

from orchestrator.api.deps import get_db_conn
from orchestrator.api.strategy_registry import (
    _require_writer,
    compute_divergence,
    tally,
    to_item,
)
from orchestrator.errors import OrchestratorError


def _agent_auth() -> dict[str, str]:
    """A server-side agent caller (its own X-Agent-Name, no admin header)."""
    return {"X-Orch-Token": "test-token", "X-Agent-Name": "quant-researcher"}


def _auth(admin: bool = False) -> dict[str, str]:
    h = {"X-Orch-Token": "test-token", "X-Agent-Name": "dashboard"}
    if admin:
        h["X-Viewer-Is-Admin"] = "true"
    return h


def _merged(**over: Any) -> dict[str, Any]:
    """A full merged repo row (curated + live join), with overridable fields."""
    row = {
        "registry_id": "11111111-1111-1111-1111-111111111111",
        "slug": "vrp-btc",
        "rank": 4,
        "promise_tier": "TIER_A",
        "display_name": "VRP vol-timing (BTC)",
        "signal_family": "vrp",
        "strategy_code": "VRP_BTC",
        "symbol": "BTCUSDT",
        "interval_name": "1d",
        "verdict_tag": "REAL_UNCERTIFIABLE",
        "lifecycle_status": "LIVE",
        "thesis": "Long BTC when implied vol is rich vs realized.",
        "detail": "in-sample Sharpe 1.21...",
        "evidence_iteration_id": None,
        "evidence_walk_forward_id": None,
        "evidence_backtest_run_id": None,
        "journal_id": None,
        "memory_ref": "project_vrp_btc_beta_alpha_decomp_2026-06-15",
        "is_offline_lead": False,
        "archived": False,
        "created_time": "2026-06-16T00:00:00+00:00",
        "updated_time": "2026-06-16T00:00:00+00:00",
        "updated_by": None,
        "dsr": 0.053,
        "psr": 0.997,
        "ann_return_pct": 46.0,
        "sharpe_ann": 1.21,
        "n_trades": 84,
        "profit_factor": None,
        "statistical_verdict": "INSUFFICIENT_EVIDENCE",
        "live_iteration_id": "aaaaaaaa-0000-0000-0000-000000000000",
        "live_backtest_run_id": "bbbbbbbb-0000-0000-0000-000000000000",
        "resolved_from": "lookup",
        "walk_forward_verdict": "INSUFFICIENT_EVIDENCE",
        "is_live": True,
    }
    row.update(over)
    return row


class _FakeConn:
    """Captures calls; returns canned results for fetch / fetchrow / fetchval."""

    def __init__(
        self,
        *,
        fetch_rows: list[dict[str, Any]] | None = None,
        fetchrow_row: dict[str, Any] | None = None,
        fetchrow_seq: list[dict[str, Any] | None] | None = None,
        fetchval_result: Any = None,
        fetchval_exc: Exception | None = None,
    ) -> None:
        self._fetch_rows = fetch_rows or []
        self._fetchrow_row = fetchrow_row
        # When set, each fetchrow() returns the next item (then None) — lets a
        # test script the get-then-read sequence an upsert performs.
        self._fetchrow_seq = list(fetchrow_seq) if fetchrow_seq is not None else None
        self._fetchval_result = fetchval_result
        self._fetchval_exc = fetchval_exc
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        self.calls.append(("fetch", params))
        return self._fetch_rows

    async def fetchrow(self, sql: str, *params: Any) -> dict[str, Any] | None:
        self.calls.append(("fetchrow", params))
        if self._fetchrow_seq is not None:
            return self._fetchrow_seq.pop(0) if self._fetchrow_seq else None
        return self._fetchrow_row

    async def fetchval(self, sql: str, *params: Any) -> Any:
        self.calls.append(("fetchval", params))
        if self._fetchval_exc is not None:
            raise self._fetchval_exc
        return self._fetchval_result


def _override(client: TestClient, conn: _FakeConn) -> None:
    async def _yield():
        yield conn

    client.app.dependency_overrides[get_db_conn] = _yield


# ── compute_divergence ─────────────────────────────────────────────────────

def test_divergence_live_but_not_trading() -> None:
    flag, reason = compute_divergence("LIVE", "REAL_UNCERTIFIABLE", 0.05, is_live=False)
    assert flag is True
    assert "no enabled account_strategy" in reason


def test_divergence_live_and_trading_is_clean() -> None:
    flag, reason = compute_divergence("LIVE", "REAL_UNCERTIFIABLE", 0.05, is_live=True)
    assert flag is False
    assert reason is None


def test_divergence_dead_but_clears_gate() -> None:
    flag, reason = compute_divergence("FALSIFIED", "FALSIFIED", 0.94, is_live=False)
    assert flag is True
    assert "0.90 DSR gate" in reason


def test_divergence_dead_below_gate_is_clean() -> None:
    flag, reason = compute_divergence("FALSIFIED", "FALSIFIED", 0.40, is_live=False)
    assert flag is False
    assert reason is None


def test_divergence_dead_with_no_live_dsr_is_clean() -> None:
    # An offline lead / never-run strategy must never flag on a missing number.
    flag, reason = compute_divergence("PARKED", "PARKED", None, is_live=False)
    assert flag is False


# ── to_item ────────────────────────────────────────────────────────────────

def test_to_item_maps_live_row() -> None:
    item = to_item(_merged())
    assert item["registryId"] == "11111111-1111-1111-1111-111111111111"
    assert item["promiseTier"] == "TIER_A"
    assert item["live"]["dsr"] == 0.053
    assert item["live"]["isLive"] is True
    assert item["live"]["resolvedFrom"] == "lookup"
    assert item["divergence"]["flag"] is False
    # Decimal-style numbers coerced to float, missing ones stay None.
    assert item["live"]["profitFactor"] is None


def test_to_item_offline_lead_has_no_live_metrics() -> None:
    row = _merged(
        slug="top-trader-lsr-fade",
        strategy_code="TOPTRADER_LSR_FADE",
        symbol=None,
        verdict_tag="REAL_LEAD",
        lifecycle_status="LEAD",
        is_offline_lead=True,
        dsr=None,
        psr=None,
        ann_return_pct=None,
        sharpe_ann=None,
        n_trades=None,
        statistical_verdict=None,
        walk_forward_verdict=None,
        resolved_from=None,
        is_live=False,
    )
    item = to_item(row)
    assert item["isOfflineLead"] is True
    assert item["live"]["dsr"] is None
    assert item["live"]["resolvedFrom"] is None
    assert item["live"]["isLive"] is False
    assert item["divergence"]["flag"] is False


def test_to_item_flags_live_without_account_strategy() -> None:
    item = to_item(_merged(is_live=False))  # lifecycle LIVE but nothing trading
    assert item["divergence"]["flag"] is True


# ── tally ──────────────────────────────────────────────────────────────────

def test_tally_counts_tiers_and_statuses() -> None:
    items = [
        to_item(_merged(promise_tier="TIER_A", lifecycle_status="LIVE")),
        to_item(_merged(promise_tier="TIER_A", lifecycle_status="LEAD")),
        to_item(_merged(promise_tier="TIER_C", lifecycle_status="FALSIFIED")),
    ]
    tier_counts, status_counts = tally(items)
    assert tier_counts == {"TIER_A": 2, "TIER_B": 0, "TIER_C": 1}
    assert status_counts["LIVE"] == 1
    assert status_counts["LEAD"] == 1
    assert status_counts["FALSIFIED"] == 1


# ── GET /strategy-registry ───────────────────────────────────────────────────

def test_list_returns_items_and_counts(client: TestClient) -> None:
    conn = _FakeConn(fetch_rows=[
        _merged(promise_tier="TIER_A", lifecycle_status="LIVE"),
        _merged(slug="vbo", promise_tier="TIER_C", lifecycle_status="FALSIFIED",
                verdict_tag="FALSIFIED", strategy_code="VBO", dsr=None, is_live=False),
    ])
    _override(client, conn)
    try:
        resp = client.get("/strategy-registry", headers=_auth())
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert data["tierCounts"]["TIER_A"] == 1
        assert data["tierCounts"]["TIER_C"] == 1
        assert {it["slug"] for it in data["items"]} == {"vrp-btc", "vbo"}
    finally:
        client.app.dependency_overrides.pop(get_db_conn, None)


def test_list_passes_filters_to_repo(client: TestClient) -> None:
    conn = _FakeConn(fetch_rows=[])
    _override(client, conn)
    try:
        resp = client.get(
            "/strategy-registry",
            params={"tier": "TIER_A", "family": "vrp", "include_archived": "true"},
            headers=_auth(),
        )
        assert resp.status_code == 200
        # repo.list_registry -> conn.fetch(sql, include_archived, tier, status, family, search)
        kind, params = conn.calls[0]
        assert kind == "fetch"
        assert params == (True, "TIER_A", None, "vrp", None)
    finally:
        client.app.dependency_overrides.pop(get_db_conn, None)


def test_list_rejects_bad_tier(client: TestClient) -> None:
    resp = client.get("/strategy-registry", params={"tier": "TIER_Z"}, headers=_auth())
    assert resp.status_code == 422


# ── admin gating on mutations ────────────────────────────────────────────────

_CREATE_BODY = {
    "slug": "new-lead",
    "promise_tier": "TIER_A",
    "display_name": "New lead",
    "verdict_tag": "REAL_LEAD",
    "lifecycle_status": "LEAD",
    "thesis": "A fresh idea.",
}


def test_create_blocked_for_dashboard_nonadmin(client: TestClient) -> None:
    # Dashboard non-admin (agent="dashboard", no admin header) cannot write.
    resp = client.post("/strategy-registry", json=_CREATE_BODY, headers=_auth(admin=False))
    assert resp.status_code == 403
    assert resp.json()["error_code"] == "writer_required"


def test_patch_requires_admin(client: TestClient) -> None:
    resp = client.patch(
        "/strategy-registry/11111111-1111-1111-1111-111111111111",
        json={"rank": 1},
        headers=_auth(admin=False),
    )
    assert resp.status_code == 403


def test_delete_requires_admin(client: TestClient) -> None:
    resp = client.delete(
        "/strategy-registry/11111111-1111-1111-1111-111111111111",
        headers=_auth(admin=False),
    )
    assert resp.status_code == 403


def test_create_as_admin_returns_item(client: TestClient) -> None:
    conn = _FakeConn(
        fetchval_result="22222222-2222-2222-2222-222222222222",
        fetchrow_row=_merged(
            registry_id="22222222-2222-2222-2222-222222222222",
            slug="new-lead",
            promise_tier="TIER_A",
            display_name="New lead",
            verdict_tag="REAL_LEAD",
            lifecycle_status="LEAD",
        ),
    )
    _override(client, conn)
    try:
        resp = client.post("/strategy-registry", json=_CREATE_BODY, headers=_auth(admin=True))
        assert resp.status_code == 201
        assert resp.json()["slug"] == "new-lead"
    finally:
        client.app.dependency_overrides.pop(get_db_conn, None)


def test_create_duplicate_slug_conflicts(client: TestClient) -> None:
    conn = _FakeConn(
        fetchval_exc=asyncpg.UniqueViolationError("duplicate key value")
    )
    _override(client, conn)
    try:
        resp = client.post("/strategy-registry", json=_CREATE_BODY, headers=_auth(admin=True))
        assert resp.status_code == 409
        assert resp.json()["error_code"] == "slug_exists"
    finally:
        client.app.dependency_overrides.pop(get_db_conn, None)


def test_create_rejects_unknown_field(client: TestClient) -> None:
    body = dict(_CREATE_BODY, bogus="x")
    resp = client.post("/strategy-registry", json=body, headers=_auth(admin=True))
    assert resp.status_code == 422


# ── writer gate (admin OR named agent) ───────────────────────────────────────

def test_require_writer_allows_admin() -> None:
    _require_writer(True, "dashboard")  # admin → ok (no raise)


def test_require_writer_allows_named_agent() -> None:
    _require_writer(False, "quant-researcher")  # named agent → ok


def test_require_writer_blocks_dashboard_nonadmin() -> None:
    with pytest.raises(OrchestratorError):
        _require_writer(False, "dashboard")


def test_require_writer_blocks_anonymous() -> None:
    with pytest.raises(OrchestratorError):
        _require_writer(False, "anonymous")


# ── agent by-slug surface ────────────────────────────────────────────────────

def test_get_by_slug_found(client: TestClient) -> None:
    _override(client, _FakeConn(fetchrow_row=_merged(slug="vrp-btc")))
    try:
        resp = client.get("/strategy-registry/by-slug/vrp-btc", headers=_auth())
        assert resp.status_code == 200
        assert resp.json()["slug"] == "vrp-btc"
    finally:
        client.app.dependency_overrides.pop(get_db_conn, None)


def test_get_by_slug_404(client: TestClient) -> None:
    _override(client, _FakeConn(fetchrow_row=None))
    try:
        resp = client.get("/strategy-registry/by-slug/nope", headers=_auth())
        assert resp.status_code == 404
    finally:
        client.app.dependency_overrides.pop(get_db_conn, None)


def test_upsert_creates_new_slug_as_agent(client: TestClient) -> None:
    # get_by_slug -> None (absent); insert -> rid; get_registry -> created row.
    conn = _FakeConn(
        fetchrow_seq=[None, _merged(slug="new-lead", display_name="New lead",
                                    promise_tier="TIER_A", verdict_tag="REAL_LEAD",
                                    lifecycle_status="LEAD")],
        fetchval_result="22222222-2222-2222-2222-222222222222",
    )
    _override(client, conn)
    try:
        resp = client.put(
            "/strategy-registry/by-slug/new-lead",
            json={"promise_tier": "TIER_A", "display_name": "New lead",
                  "verdict_tag": "REAL_LEAD", "lifecycle_status": "LEAD", "thesis": "t"},
            headers=_agent_auth(),
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["slug"] == "new-lead"
    finally:
        client.app.dependency_overrides.pop(get_db_conn, None)


def test_upsert_updates_existing_slug_as_agent(client: TestClient) -> None:
    # get_by_slug -> existing; update -> rid; get_registry -> merged row.
    conn = _FakeConn(
        fetchrow_seq=[_merged(slug="vrp-btc"),
                      _merged(slug="vrp-btc", lifecycle_status="PARKED")],
        fetchval_result="11111111-1111-1111-1111-111111111111",
    )
    _override(client, conn)
    try:
        resp = client.put(
            "/strategy-registry/by-slug/vrp-btc",
            json={"lifecycle_status": "PARKED"},  # partial update — no required fields
            headers=_agent_auth(),
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["lifecycleStatus"] == "PARKED"
    finally:
        client.app.dependency_overrides.pop(get_db_conn, None)


def test_upsert_create_missing_required_is_422(client: TestClient) -> None:
    _override(client, _FakeConn(fetchrow_seq=[None]))  # slug absent
    try:
        resp = client.put(
            "/strategy-registry/by-slug/bare",
            json={"signal_family": "x"},  # no required create fields
            headers=_agent_auth(),
        )
        assert resp.status_code == 422
        assert resp.json()["error_code"] == "registry_missing_fields"
    finally:
        client.app.dependency_overrides.pop(get_db_conn, None)


def test_upsert_blocked_for_dashboard_nonadmin(client: TestClient) -> None:
    # _require_writer runs before any DB access, so no conn override is needed.
    resp = client.put(
        "/strategy-registry/by-slug/x", json={"thesis": "t"}, headers=_auth(admin=False)
    )
    assert resp.status_code == 403
    assert resp.json()["error_code"] == "writer_required"


def test_delete_by_slug_as_agent(client: TestClient) -> None:
    conn = _FakeConn(
        fetchrow_row=_merged(slug="zombie"),
        fetchval_result="11111111-1111-1111-1111-111111111111",
    )
    _override(client, conn)
    try:
        resp = client.delete("/strategy-registry/by-slug/zombie", headers=_agent_auth())
        assert resp.status_code == 200
        assert resp.json() == {"slug": "zombie", "archived": True}
    finally:
        client.app.dependency_overrides.pop(get_db_conn, None)


def test_delete_by_slug_404(client: TestClient) -> None:
    _override(client, _FakeConn(fetchrow_row=None))
    try:
        resp = client.delete("/strategy-registry/by-slug/nope", headers=_agent_auth())
        assert resp.status_code == 404
    finally:
        client.app.dependency_overrides.pop(get_db_conn, None)
