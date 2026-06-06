import pytest

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
