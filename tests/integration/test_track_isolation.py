import pytest

from orchestrator.api import agent as agent_api
from orchestrator.repo import agent_state


pytestmark = pytest.mark.integration


async def _insert_run_summary(conn, *, title, track, content="x"):
    await conn.execute(
        """
        INSERT INTO research_journal
            (entry_type, strategy_code, title, content, structured_data, status, created_time)
        VALUES ('RUN_SUMMARY', 'TRK_TEST', $1, $2, $3, 'ACTIVE', NOW())
        """,
        title, content, {"track": track} if track else {},
    )


@pytest.mark.asyncio
async def test_lockout_is_scoped_to_track(db_conn):
    # A hedging-track exhaustion must NOT lock out the trading track.
    await _insert_run_summary(
        db_conn, title="ARCHETYPE_EXHAUSTION_HEDGE", track="hedging",
    )

    trading = await agent_state._lockout_state(db_conn, track="trading")
    hedging = await agent_state._lockout_state(db_conn, track="hedging")

    assert trading["in_lockout"] is False, "trading must not see hedging's lockout"
    assert hedging["in_lockout"] is True, "hedging must see its own lockout"


@pytest.mark.asyncio
async def test_last_run_summary_is_scoped_to_track(db_conn):
    await _insert_run_summary(db_conn, title="SESSION_CHECKPOINT_TRADING", track="trading", content="t")
    await _insert_run_summary(db_conn, title="SESSION_CHECKPOINT_HEDGING", track="hedging", content="h")

    trading = await agent_state._last_run_summary(db_conn, track="trading")
    hedging = await agent_state._last_run_summary(db_conn, track="hedging")

    assert trading["title"] == "SESSION_CHECKPOINT_TRADING"
    assert hedging["title"] == "SESSION_CHECKPOINT_HEDGING"


@pytest.mark.asyncio
async def test_get_state_digest_threads_track(db_conn):
    # Active hypotheses are per-track.
    await db_conn.execute(
        """
        INSERT INTO research_journal
            (entry_type, strategy_code, title, content, structured_data, status, created_time)
        VALUES ('HYPOTHESIS', 'TRK_TEST', 'h-trading', 'x', '{"track":"trading"}', 'ACTIVE', NOW()),
               ('HYPOTHESIS', 'TRK_TEST', 'h-hedging', 'x', '{"track":"hedging"}', 'ACTIVE', NOW())
        """
    )
    digest = await agent_state.get_state_digest(db_conn, track="trading")
    titles = {h["title"] for h in digest["active_hypotheses"]}
    assert "h-trading" in titles
    assert "h-hedging" not in titles


@pytest.mark.asyncio
async def test_digest_no_track_is_global(db_conn):
    await db_conn.execute(
        """
        INSERT INTO research_journal
            (entry_type, strategy_code, title, content, structured_data, status, created_time)
        VALUES ('HYPOTHESIS', 'TRK_TEST', 'h-untagged', 'x', '{}', 'ACTIVE', NOW()),
               ('HYPOTHESIS', 'TRK_TEST', 'h-tagged',   'x', '{"track":"trading"}', 'ACTIVE', NOW())
        """
    )
    digest = await agent_state.get_state_digest(db_conn)  # no track → global
    titles = {h["title"] for h in digest["active_hypotheses"]}
    assert {"h-untagged", "h-tagged"} <= titles


def test_agent_state_endpoint_forwards_track(integration_client, monkeypatch):
    """GET /agent/state?track=trading returns 200 AND the track value reaches
    ``get_state_digest`` as the ``track`` kwarg.

    Approach: the request is driven through the FULL real stack — auth
    middleware, the live DB health probe (real asyncpg pool → db_ok=True), and
    the real handler. We monkeypatch the repo's ``get_state_digest`` to record
    the kwarg it receives and return a minimal digest, so the assertion proves
    the wiring (the point of Task 3) without depending on ML/iteration tables
    that are outside the minimal Phase-1 test schema.
    """
    recorded: dict[str, object] = {}

    async def _fake_get_state_digest(conn, **kwargs):
        recorded["track"] = kwargs.get("track")
        recorded["agent_name"] = kwargs.get("agent_name")
        return {
            "queue_counts": {"PENDING": 0, "RUNNING": 0, "PARKED": 0,
                             "COMPLETED": 0, "FAILED": 0},
            "last_iterations": [],
            "recent_sig_edge_iteration_ids": [],
            "active_hypotheses": [],
            "last_run_summary": None,
            "last_null_screen_per_surface": [],
            "pending_specialist_reviews": [],
            "recent_specialist_verdicts": [],
            "lockout_state": {"in_lockout": False, "terminal_title": None,
                              "lockout_expires_at": None, "bypass_available": False},
            "ml_training_budget": None,
            "pending_ml_training_runs": None,
        }

    monkeypatch.setattr(
        agent_api.agent_state_repo, "get_state_digest", _fake_get_state_digest,
    )

    resp = integration_client.get(
        "/agent/state?track=trading",
        headers={"X-Orch-Token": "test-token", "X-Agent-Name": "quant-researcher"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["db_ok"] is True, "real pool should answer SELECT 1"
    assert "queue_counts" in body
    assert recorded["track"] == "trading", "?track= must reach get_state_digest"


def test_agent_state_track_param_is_validated(integration_client, monkeypatch):
    """GET /agent/state validates ?track= against the same Literal the POST
    bodies use: a valid value → 200, a bogus value → 422 (not a silently
    global-shaped digest)."""
    async def _fake_get_state_digest(conn, **kwargs):
        return {
            "queue_counts": {"PENDING": 0, "RUNNING": 0, "PARKED": 0,
                             "COMPLETED": 0, "FAILED": 0},
            "last_iterations": [],
            "recent_sig_edge_iteration_ids": [],
            "active_hypotheses": [],
            "last_run_summary": None,
            "last_null_screen_per_surface": [],
            "pending_specialist_reviews": [],
            "recent_specialist_verdicts": [],
            "lockout_state": {"in_lockout": False, "terminal_title": None,
                              "lockout_expires_at": None, "bypass_available": False},
            "ml_training_budget": None,
            "pending_ml_training_runs": None,
        }

    monkeypatch.setattr(
        agent_api.agent_state_repo, "get_state_digest", _fake_get_state_digest,
    )

    ok = integration_client.get("/agent/state?track=trading", headers=_AUTH_HEADERS)
    assert ok.status_code == 200, ok.text

    bad = integration_client.get("/agent/state?track=bogus", headers=_AUTH_HEADERS)
    assert bad.status_code == 422, bad.text


async def _insert_queue(conn, *, strategy_code, sweep_config, track=None):
    """Enqueue via the real repo writer (exercises track stamping)."""
    from orchestrator.repo import queue_write

    return await queue_write.insert_queue(
        conn,
        strategy_code=strategy_code,
        interval_name="1h",
        instrument="BTCUSDT",
        sweep_config=sweep_config,
        hypothesis="h",
        priority=100,
        iter_budget=1,
        early_stop_on_no_edge=True,
        require_walk_forward=True,
        created_by="trk-test",
        track=track,
    )


@pytest.mark.asyncio
async def test_insert_queue_stamps_track_without_mutating_caller(db_conn):
    from orchestrator.repo import queue_write

    caller_cfg = {"strategy": "grid", "params": []}
    row = await queue_write.insert_queue(
        db_conn,
        strategy_code="TRK_STAMP",
        interval_name="1h",
        instrument="BTCUSDT",
        sweep_config=caller_cfg,
        hypothesis="h",
        priority=100,
        iter_budget=1,
        early_stop_on_no_edge=True,
        require_walk_forward=True,
        created_by="trk-test",
        track="hedging",
    )
    # The persisted sweep_config carries the track tag...
    assert row["sweep_config"]["track"] == "hedging"
    stored = await db_conn.fetchval(
        "SELECT sweep_config->>'track' FROM research_queue WHERE queue_id = $1",
        row["queue_id"],
    )
    assert stored == "hedging"
    # ...but the caller's dict was NOT mutated.
    assert "track" not in caller_cfg


@pytest.mark.asyncio
async def test_claim_only_picks_matching_track(db_conn):
    from orchestrator.repo import queue_write

    await _insert_queue(
        db_conn,
        strategy_code="TRK_A",
        sweep_config={"strategy": "grid", "params": [], "track": "hedging"},
        track="hedging",
    )
    claimed = await queue_write.claim_next(db_conn, track="trading")
    assert claimed is None, "trading claim must skip a hedging-tagged row"

    claimed_h = await queue_write.claim_next(db_conn, track="hedging")
    assert claimed_h is not None and claimed_h["strategy_code"] == "TRK_A"


@pytest.mark.asyncio
async def test_claim_no_track_is_global(db_conn):
    from orchestrator.repo import queue_write

    await _insert_queue(
        db_conn,
        strategy_code="TRK_B",
        sweep_config={"strategy": "grid", "params": []},
        track="hedging",
    )
    # No track on the claim → backward-compatible, picks any PENDING row.
    claimed = await queue_write.claim_next(db_conn)
    assert claimed is not None and claimed["strategy_code"] == "TRK_B"


@pytest.mark.asyncio
async def test_claim_trading_picks_up_untagged_legacy_row(db_conn):
    """A legacy PENDING row with NO 'track' key in sweep_config must be claimable
    by a track='trading' claim (treated as trading) — and NOT by a hedging claim.

    Without this, legacy untagged rows strand: a trading drain would skip them
    forever once dual-track claims are scoped."""
    from orchestrator.repo import queue_write

    # Untagged row: track=None means insert_queue does NOT stamp a 'track' key.
    await _insert_queue(
        db_conn,
        strategy_code="TRK_LEGACY",
        sweep_config={"strategy": "grid", "params": []},  # no 'track'
        track=None,
    )

    # A hedging claim must NOT pick it up.
    h = await queue_write.claim_next(db_conn, track="hedging")
    assert h is None, "untagged row must not be claimed as hedging"

    # A trading claim treats the untagged row as trading and claims it.
    t = await queue_write.claim_next(db_conn, track="trading")
    assert t is not None and t["strategy_code"] == "TRK_LEGACY"


@pytest.mark.asyncio
async def test_queue_counts_scoped_to_track(db_conn):
    await _insert_queue(
        db_conn,
        strategy_code="TRK_C",
        sweep_config={"strategy": "grid", "params": []},
        track="hedging",
    )
    counts = await agent_state._queue_counts(db_conn, track="trading")
    assert counts["PENDING"] == 0
    counts_h = await agent_state._queue_counts(db_conn, track="hedging")
    assert counts_h["PENDING"] >= 1


@pytest.mark.asyncio
async def test_queue_counts_no_track_is_global(db_conn):
    await _insert_queue(
        db_conn,
        strategy_code="TRK_D",
        sweep_config={"strategy": "grid", "params": []},
        track="hedging",
    )
    counts = await agent_state._queue_counts(db_conn)
    assert counts["PENDING"] >= 1


# ── Task 5: POST /queue track param + per-track re-discovery gate ──────────

_AUTH_HEADERS = {"X-Orch-Token": "test-token", "X-Agent-Name": "quant-researcher"}


async def _seed_discard(conn, *, strategy_code, param_names, track):
    """Plant a DISCARD audit row whose originating queue row carries ``track``.

    Mirrors the tick lifecycle: insert the research_queue row (track-stamped),
    insert the hypothesis_audit row pointing at it, then backfill the DISCARD
    verdict (the gate keys on decision_verdict='DISCARD')."""
    from orchestrator.repo import hypothesis_audit as audit_repo
    from orchestrator.repo import queue_write

    params_snapshot = {n: 14 for n in param_names}
    qrow = await queue_write.insert_queue(
        conn,
        strategy_code=strategy_code,
        interval_name="1h",
        instrument="BTCUSDT",
        sweep_config={"strategy": "grid",
                      "params": [{"name": n, "values": [14]} for n in param_names]},
        hypothesis="seed",
        priority=100,
        iter_budget=1,
        early_stop_on_no_edge=True,
        require_walk_forward=True,
        created_by="trk-test",
        track=track,
    )
    audit_id = await audit_repo.insert_audit(
        conn,
        strategy_code=strategy_code,
        symbol="BTCUSDT",
        interval_name="1h",
        params_snapshot=params_snapshot,
        queue_id=qrow["queue_id"],
        created_by="trk-test",
    )
    # iteration_id must be non-NULL-shaped; the gate doesn't require an
    # iteration_log row to exist (it SELECTs the audit row directly).
    from uuid import uuid4

    await audit_repo.update_audit_verdict(
        conn,
        audit_id=audit_id,
        iteration_id=uuid4(),
        statistical_verdict="NO_EDGE",
        decision_verdict="DISCARD",
    )


def _enqueue_body(strategy_code, param_names, track):
    return {
        "strategy_code": strategy_code,
        "interval_name": "1h",
        "instrument": "BTCUSDT",
        "sweep_config": {
            "strategy": "grid",
            "params": [{"name": n, "values": [14]} for n in param_names],
        },
        "hypothesis": "post-gate test",
        # Bypass the paired-review gate (orthogonal to this task); the
        # re-discovery gate is NOT bypassed.
        "override_review_gate": True,
        "track": track,
    }


@pytest.mark.asyncio
async def test_rediscovery_gate_is_per_track(db_conn, integration_client):
    strategy_code = "TRK_GATE"
    param_names = ["rsi_period"]

    # account_strategy row so POST /queue's pre-insert existence check passes
    # (else 412 account_strategy_missing fires before the gate).
    await db_conn.execute(
        "INSERT INTO account_strategy (strategy_code, is_deleted) VALUES ($1, false)",
        strategy_code,
    )
    # Seed a DISCARD on the HEDGING track for this axis-set.
    await _seed_discard(
        db_conn, strategy_code=strategy_code, param_names=param_names,
        track="hedging",
    )

    # SAME axis-set on the TRADING track must NOT be blocked.
    r_trading = integration_client.post(
        "/queue", json=_enqueue_body(strategy_code, param_names, "trading"),
        headers=_AUTH_HEADERS,
    )
    assert r_trading.status_code == 201, r_trading.text

    # A repeat on the HEDGING track SHOULD 409 (same track's prior discard).
    r_hedging = integration_client.post(
        "/queue", json=_enqueue_body(strategy_code, param_names, "hedging"),
        headers=_AUTH_HEADERS,
    )
    assert r_hedging.status_code == 409, r_hedging.text
    assert r_hedging.json()["error_code"] == "axis_previously_discarded"


@pytest.mark.asyncio
async def test_rediscovery_gate_global_discard_blocks_any_track(db_conn,
                                                                integration_client):
    """A legacy un-tracked (NULL-track) discard must still block a tracked
    enqueue — never weaken the existing gate."""
    strategy_code = "TRK_GATE_GLOBAL"
    param_names = ["ema_fast"]

    await db_conn.execute(
        "INSERT INTO account_strategy (strategy_code, is_deleted) VALUES ($1, false)",
        strategy_code,
    )
    await _seed_discard(
        db_conn, strategy_code=strategy_code, param_names=param_names,
        track=None,  # global / legacy discard
    )

    r = integration_client.post(
        "/queue", json=_enqueue_body(strategy_code, param_names, "trading"),
        headers=_AUTH_HEADERS,
    )
    assert r.status_code == 409, r.text
    assert r.json()["error_code"] == "axis_previously_discarded"


@pytest.mark.asyncio
async def test_drain_only_drains_its_track(db_conn, integration_client):
    """POST /tick/drain {track:'trading'} must leave hedging-tagged PENDING
    rows untouched."""
    # One hedging PENDING row, seeded directly.
    hedge = await _insert_queue(
        db_conn,
        strategy_code="TRK_DRAIN_H",
        sweep_config={"strategy": "grid", "params": []},
        track="hedging",
    )

    resp = integration_client.post(
        "/tick/drain", json={"track": "trading", "max_iters": 3},
        headers=_AUTH_HEADERS,
    )
    assert resp.status_code == 200, resp.text
    # No trading row exists → drain hits EMPTY_QUEUE without claiming anything.
    assert resp.json()["terminal_action"] == "EMPTY_QUEUE"

    # The hedging row is still PENDING (never claimed by the trading drain).
    status = await db_conn.fetchval(
        "SELECT status FROM research_queue WHERE queue_id = $1", hedge["queue_id"],
    )
    assert status == "PENDING"


# ── Task 6: end-to-end per-track isolation guarantee ──────────────────────


@pytest.mark.asyncio
async def test_hedging_exhaustion_invisible_to_trading_digest(db_conn):
    # The headline guarantee: a hedging ARCHETYPE_EXHAUSTION never appears
    # in the trading digest, and vice-versa.
    await _insert_run_summary(db_conn, title="ARCHETYPE_EXHAUSTION_X", track="hedging")
    trading = await agent_state.get_state_digest(db_conn, track="trading")
    hedging = await agent_state.get_state_digest(db_conn, track="hedging")
    assert trading["lockout_state"]["in_lockout"] is False
    assert hedging["lockout_state"]["in_lockout"] is True
