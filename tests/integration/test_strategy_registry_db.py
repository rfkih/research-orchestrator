"""Integration tests for the Strategy Research Registry live-metrics join.

Exercises the real ``repo.strategy_registry`` SQL against a live Postgres
(``schema_phase1.sql``): the LEFT JOIN LATERAL onto research_iteration_log ->
backtest_run (metrics), walk_forward_run (verdict), and account_strategy
(live status), keyed by (strategy_code, symbol, interval_name). The pure
mapping/divergence logic is unit-tested in tests/test_strategy_registry.py.
"""
from __future__ import annotations

import uuid

import pytest

from orchestrator.repo import strategy_registry as registry_repo
from orchestrator.services import registry_sync

pytestmark = pytest.mark.integration

_AUTH = {"X-Orch-Token": "test-token", "X-Agent-Name": "dashboard"}
_AGENT = {"X-Orch-Token": "test-token", "X-Agent-Name": "quant-researcher"}


async def _seed_live_strategy(conn, *, code, symbol, interval, dsr, ann_ret, wf_verdict,
                              enabled, simulated, is_deleted=False):
    """A completed run + iteration + walk-forward + account_strategy for `code`."""
    run_id = uuid.uuid4()
    await conn.execute(
        "INSERT INTO backtest_run (backtest_run_id, strategy_code, asset, interval_name,"
        " status, start_time, end_time) VALUES ($1,$2,$3,$4,'COMPLETED',"
        " NOW() - INTERVAL '400 days', NOW())",
        run_id, code, symbol, interval,
    )
    await conn.execute(
        "INSERT INTO research_iteration_log (strategy_code, iteration_number,"
        " backtest_run_id, metrics_snapshot, verdict, statistical_verdict)"
        " VALUES ($1, 1, $2, $3, 'ITERATE', 'INSUFFICIENT_EVIDENCE')",
        code, run_id,
        {
            "win_rate": 0.6,
            "analysis": {
                "dsr": dsr,
                "psr": 0.99,
                "annualized_geometric_return_pct_at_alloc_90": ann_ret,
                "sharpe_annualized": 1.21,
                "n_trades": 84,
                "pf_point_estimate": 1.3,
            },
        },
    )
    await conn.execute(
        "INSERT INTO walk_forward_run (strategy_code, instrument, interval_name,"
        " stability_verdict) VALUES ($1,$2,$3,$4)",
        code, symbol, interval, wf_verdict,
    )
    await conn.execute(
        "INSERT INTO account_strategy (strategy_code, symbol, interval_name, enabled,"
        " simulated, is_deleted) VALUES ($1,$2,$3,$4,$5,$6)",
        code, symbol, interval, enabled, simulated, is_deleted,
    )


async def _seed_registry(conn, **kw):
    await conn.execute(
        "INSERT INTO strategy_research_registry (slug, rank, promise_tier, display_name,"
        " signal_family, strategy_code, symbol, interval_name, verdict_tag,"
        " lifecycle_status, thesis, is_offline_lead) VALUES"
        " ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)",
        kw["slug"], kw.get("rank"), kw["promise_tier"], kw["display_name"],
        kw.get("signal_family"), kw.get("strategy_code"), kw.get("symbol"),
        kw.get("interval_name"), kw["verdict_tag"], kw["lifecycle_status"],
        kw.get("thesis", "t"), kw.get("is_offline_lead", False),
    )


@pytest.mark.asyncio
async def test_live_row_resolves_metrics_by_key(db_conn):
    await _seed_live_strategy(
        db_conn, code="VRP_BTC", symbol="BTCUSDT", interval="1d",
        dsr=0.053, ann_ret=46.0, wf_verdict="INSUFFICIENT_EVIDENCE",
        enabled=True, simulated=False,
    )
    await _seed_registry(
        db_conn, slug="t-vrp", rank=1, promise_tier="TIER_A", display_name="VRP BTC",
        signal_family="vrp", strategy_code="VRP_BTC", symbol="BTCUSDT",
        interval_name="1d", verdict_tag="REAL_UNCERTIFIABLE", lifecycle_status="LIVE",
    )

    rows = await registry_repo.list_registry(db_conn)
    row = next(r for r in rows if r["slug"] == "t-vrp")
    assert float(row["dsr"]) == pytest.approx(0.053)
    assert float(row["ann_return_pct"]) == pytest.approx(46.0)
    assert row["n_trades"] == 84
    assert row["walk_forward_verdict"] == "INSUFFICIENT_EVIDENCE"
    assert row["statistical_verdict"] == "INSUFFICIENT_EVIDENCE"
    assert row["is_live"] is True
    assert row["resolved_from"] == "lookup"


@pytest.mark.asyncio
async def test_offline_lead_has_no_live_metrics(db_conn):
    await _seed_registry(
        db_conn, slug="t-offline", rank=1, promise_tier="TIER_A",
        display_name="Top-trader fade", signal_family="positioning",
        strategy_code=None, symbol=None, interval_name="1d",
        verdict_tag="REAL_LEAD", lifecycle_status="LEAD", is_offline_lead=True,
    )
    rows = await registry_repo.list_registry(db_conn)
    row = next(r for r in rows if r["slug"] == "t-offline")
    assert row["dsr"] is None
    assert row["is_live"] is False
    assert row["resolved_from"] is None
    assert row["walk_forward_verdict"] is None


@pytest.mark.asyncio
async def test_simulated_account_is_not_live(db_conn):
    # An enabled-but-simulated account_strategy must NOT count as live.
    await _seed_live_strategy(
        db_conn, code="VRP_ETH", symbol="ETHUSDT", interval="1d",
        dsr=0.17, ann_ret=20.0, wf_verdict="INSUFFICIENT_EVIDENCE",
        enabled=True, simulated=True,
    )
    await _seed_registry(
        db_conn, slug="t-sim", rank=1, promise_tier="TIER_B", display_name="VRP ETH sim",
        strategy_code="VRP_ETH", symbol="ETHUSDT", interval_name="1d",
        verdict_tag="REAL_UNCERTIFIABLE", lifecycle_status="LEAD",
    )
    rows = await registry_repo.list_registry(db_conn)
    row = next(r for r in rows if r["slug"] == "t-sim")
    assert float(row["dsr"]) == pytest.approx(0.17)  # metrics still resolve
    assert row["is_live"] is False                    # but it is not "live"


@pytest.mark.asyncio
async def test_interval_disambiguates_same_code(db_conn):
    # DCB exists at 1h (the live set) and 4h (falsified) — the join must not mix.
    await _seed_live_strategy(
        db_conn, code="DCB", symbol="ETHUSDT", interval="1h",
        dsr=0.02, ann_ret=195.0, wf_verdict="ROBUST", enabled=True, simulated=False,
    )
    await _seed_live_strategy(
        db_conn, code="DCB", symbol="ETHUSDT", interval="4h",
        dsr=0.13, ann_ret=8.1, wf_verdict="INSUFFICIENT_EVIDENCE",
        enabled=False, simulated=True,
    )
    await _seed_registry(
        db_conn, slug="t-dcb-4h", rank=1, promise_tier="TIER_C", display_name="DCB 4h",
        strategy_code="DCB", symbol="ETHUSDT", interval_name="4h",
        verdict_tag="FALSIFIED", lifecycle_status="FALSIFIED",
    )
    rows = await registry_repo.list_registry(db_conn)
    row = next(r for r in rows if r["slug"] == "t-dcb-4h")
    assert float(row["ann_return_pct"]) == pytest.approx(8.1)  # the 4h cell, not 1h
    assert row["walk_forward_verdict"] == "INSUFFICIENT_EVIDENCE"
    # ...and the 4h row must NOT be "live" off the 1h live account_strategy.
    assert row["is_live"] is False


@pytest.mark.asyncio
async def test_latest_run_wins_not_best(db_conn):
    # Two completed runs for one key: the OLDER has the BETTER return but the
    # LATER one is the current condition — the join must surface the latest,
    # never the best-ever cell (the "always latest condition" guarantee).
    import uuid

    old_run, new_run = uuid.uuid4(), uuid.uuid4()
    for rid in (old_run, new_run):
        await db_conn.execute(
            "INSERT INTO backtest_run (backtest_run_id,strategy_code,asset,interval_name,"
            "status,start_time,end_time) VALUES ($1,'LR','BTCUSDT','1d','COMPLETED',"
            "NOW()-INTERVAL '400 days',NOW())", rid)
    await db_conn.execute(
        "INSERT INTO research_iteration_log (strategy_code,backtest_run_id,metrics_snapshot,"
        "verdict,statistical_verdict,created_time) VALUES ('LR',$1,$2,'ITERATE','NO_EDGE',"
        "NOW()-INTERVAL '2 days')",
        old_run, {"analysis": {"dsr": 0.01, "annualized_geometric_return_pct_at_alloc_90": 99.0, "n_trades": 5}})
    await db_conn.execute(
        "INSERT INTO research_iteration_log (strategy_code,backtest_run_id,metrics_snapshot,"
        "verdict,statistical_verdict,created_time) VALUES ('LR',$1,$2,'ITERATE',"
        "'INSUFFICIENT_EVIDENCE',NOW())",
        new_run, {"analysis": {"dsr": 0.053, "annualized_geometric_return_pct_at_alloc_90": 46.0, "n_trades": 84}})
    await _seed_registry(
        db_conn, slug="t-lr", rank=1, promise_tier="TIER_A", display_name="LR",
        strategy_code="LR", symbol="BTCUSDT", interval_name="1d",
        verdict_tag="REAL_UNCERTIFIABLE", lifecycle_status="LEAD")

    rows = await registry_repo.list_registry(db_conn)
    row = next(r for r in rows if r["slug"] == "t-lr")
    assert float(row["dsr"]) == pytest.approx(0.053)  # latest, NOT the better-return 0.01
    assert row["n_trades"] == 84


@pytest.mark.asyncio
async def test_deleted_account_is_not_live(db_conn):
    # Enabled + non-simulated but is_deleted=TRUE must not count as live.
    await _seed_live_strategy(
        db_conn, code="DEL", symbol="BTCUSDT", interval="1d", dsr=0.5, ann_ret=20.0,
        wf_verdict="ROBUST", enabled=True, simulated=False, is_deleted=True)
    await _seed_registry(
        db_conn, slug="t-del", rank=1, promise_tier="TIER_C", display_name="Del",
        strategy_code="DEL", symbol="BTCUSDT", interval_name="1d",
        verdict_tag="FALSIFIED", lifecycle_status="FALSIFIED")
    rows = await registry_repo.list_registry(db_conn)
    row = next(r for r in rows if r["slug"] == "t-del")
    assert row["is_live"] is False


def test_http_list_and_divergence(db_conn, integration_client):
    """End-to-end through the proxy-equivalent stack: a curated FALSIFIED row
    whose strategy_code now has a live run clearing the DSR gate must surface a
    divergence flag in the HTTP response."""
    import asyncio

    async def _seed():
        await _seed_live_strategy(
            db_conn, code="ZOMBIE", symbol="BTCUSDT", interval="1d",
            dsr=0.95, ann_ret=30.0, wf_verdict="ROBUST", enabled=False, simulated=True,
        )
        await _seed_registry(
            db_conn, slug="t-zombie", rank=1, promise_tier="TIER_C",
            display_name="Zombie", strategy_code="ZOMBIE", symbol="BTCUSDT",
            interval_name="1d", verdict_tag="FALSIFIED", lifecycle_status="FALSIFIED",
        )

    asyncio.get_event_loop().run_until_complete(_seed())

    resp = integration_client.get("/strategy-registry", headers=_AUTH)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["total"] >= 1
    zombie = next(it for it in data["items"] if it["slug"] == "t-zombie")
    assert zombie["live"]["dsr"] == pytest.approx(0.95)
    assert zombie["divergence"]["flag"] is True
    assert "DSR gate" in zombie["divergence"]["reason"]


def test_agent_upsert_by_slug_create_then_update(integration_client):
    """The agent maintains the roster via PUT-by-slug (no SQL): create, then a
    partial update; the second PUT merges, not replaces."""
    r1 = integration_client.put(
        "/strategy-registry/by-slug/it-upsert",
        json={"promise_tier": "TIER_A", "display_name": "Upsert lead",
              "verdict_tag": "REAL_LEAD", "lifecycle_status": "LEAD",
              "thesis": "created by agent", "signal_family": "positioning"},
        headers=_AGENT,
    )
    assert r1.status_code == 200, r1.text
    assert r1.json()["lifecycleStatus"] == "LEAD"

    r2 = integration_client.put(
        "/strategy-registry/by-slug/it-upsert",
        json={"lifecycle_status": "PARKED"},  # partial — only the status changes
        headers=_AGENT,
    )
    assert r2.status_code == 200, r2.text
    body = r2.json()
    assert body["lifecycleStatus"] == "PARKED"
    assert body["displayName"] == "Upsert lead"      # merged, not wiped
    assert body["signalFamily"] == "positioning"
    assert body["updatedBy"] == "quant-researcher"

    got = integration_client.get("/strategy-registry/by-slug/it-upsert", headers=_AGENT)
    assert got.json()["lifecycleStatus"] == "PARKED"


def test_agent_upsert_create_missing_required_is_422(integration_client):
    r = integration_client.put(
        "/strategy-registry/by-slug/it-bare", json={"signal_family": "x"}, headers=_AGENT,
    )
    assert r.status_code == 422
    assert r.json()["error_code"] == "registry_missing_fields"


def test_agent_delete_by_slug_archives(integration_client):
    integration_client.put(
        "/strategy-registry/by-slug/it-del",
        json={"promise_tier": "TIER_C", "display_name": "Doomed",
              "verdict_tag": "FALSIFIED", "lifecycle_status": "FALSIFIED", "thesis": "x"},
        headers=_AGENT,
    )
    d = integration_client.delete("/strategy-registry/by-slug/it-del", headers=_AGENT)
    assert d.status_code == 200
    assert d.json() == {"slug": "it-del", "archived": True}

    default = integration_client.get("/strategy-registry", headers=_AGENT).json()
    assert all(i["slug"] != "it-del" for i in default["items"])
    arch = integration_client.get(
        "/strategy-registry?include_archived=true", headers=_AGENT
    ).json()
    assert any(i["slug"] == "it-del" for i in arch["items"])


def test_dashboard_nonadmin_cannot_write_by_slug(integration_client):
    r = integration_client.put(
        "/strategy-registry/by-slug/it-x", json={"thesis": "t"}, headers=_AUTH,
    )
    assert r.status_code == 403
    assert r.json()["error_code"] == "writer_required"


# ── auto-sync from the research loop ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_auto_sync_create_update_and_graduate(db_conn):
    # CREATE: a never-seen strategy with SIGNIFICANT_EDGE → agent-owned LEAD row.
    assert await registry_sync.sync_from_research(
        db_conn, strategy_code="AUTO_X", symbol="BTCUSDT", interval="1h",
        statistical_verdict="SIGNIFICANT_EDGE") == "created"
    row = next(r for r in await registry_repo.list_registry(db_conn)
               if r["strategy_code"] == "AUTO_X")
    assert row["auto_managed"] is True
    assert row["lifecycle_status"] == "LEAD"
    assert row["verdict_tag"] == "REAL_UNCERTIFIABLE"

    # UPDATE: same key, later NO_EDGE → falsified (latest condition).
    assert await registry_sync.sync_from_research(
        db_conn, strategy_code="AUTO_X", symbol="BTCUSDT", interval="1h",
        statistical_verdict="NO_EDGE") == "updated"
    row = next(r for r in await registry_repo.list_registry(db_conn)
               if r["strategy_code"] == "AUTO_X")
    assert row["lifecycle_status"] == "FALSIFIED"

    # GRADUATE: a ROBUST walk-forward promotes it to TIER_A / REAL_LEAD.
    assert await registry_sync.sync_from_research(
        db_conn, strategy_code="AUTO_X", symbol="BTCUSDT", interval="1h",
        statistical_verdict=None, walk_forward_verdict="ROBUST") == "updated"
    row = next(r for r in await registry_repo.list_registry(db_conn)
               if r["strategy_code"] == "AUTO_X")
    assert row["promise_tier"] == "TIER_A"
    assert row["verdict_tag"] == "REAL_LEAD"


@pytest.mark.asyncio
async def test_auto_sync_never_clobbers_curated(db_conn):
    # A curated row (auto_managed=FALSE by default) must survive the sync intact.
    await _seed_registry(
        db_conn, slug="cur-x", rank=1, promise_tier="TIER_A", display_name="Curated",
        strategy_code="CUR_X", symbol="BTCUSDT", interval_name="1d",
        verdict_tag="REAL_UNCERTIFIABLE", lifecycle_status="LIVE")
    assert await registry_sync.sync_from_research(
        db_conn, strategy_code="CUR_X", symbol="BTCUSDT", interval="1d",
        statistical_verdict="NO_EDGE") == "skipped"
    row = next(r for r in await registry_repo.list_registry(db_conn) if r["slug"] == "cur-x")
    assert row["lifecycle_status"] == "LIVE"            # unchanged
    assert row["verdict_tag"] == "REAL_UNCERTIFIABLE"   # not clobbered


@pytest.mark.asyncio
async def test_auto_sync_universe_seed_suppresses_per_symbol_dup(db_conn):
    # A curated universe row (symbol NULL) covers any symbol for that code →
    # the per-symbol auto-sync must SKIP, not create a duplicate.
    await _seed_registry(
        db_conn, slug="uni-x", rank=1, promise_tier="TIER_A", display_name="Universe",
        strategy_code="UNI_X", symbol=None, interval_name="1d",
        verdict_tag="REAL_LEAD", lifecycle_status="LEAD")
    assert await registry_sync.sync_from_research(
        db_conn, strategy_code="UNI_X", symbol="BTCUSDT", interval="1d",
        statistical_verdict="SIGNIFICANT_EDGE") == "skipped"
    rows = await registry_repo.list_registry(db_conn, include_archived=True)
    assert sum(1 for r in rows if r["strategy_code"] == "UNI_X") == 1
