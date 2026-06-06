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
