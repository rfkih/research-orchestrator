"""Integration tests for combination-book persistence (Phase 3, Task 5).

The signal-level combination book persists admitted *idiosyncratic residual*
signals in the SAME ``signal_pool`` table the strategy pool uses, discriminated
by ``admission_metrics->>'kind' = 'signal_combination'`` (+ a ``track`` tag, both
carried INSIDE the existing ``admission_metrics`` JSONB column — NOT new physical
columns) so the two books coexist without a separate table / Flyway migration.
These tests prove the round-trip AND the coexistence isolation: a combination
member never leaks into the strategy-pool reads, a strategy-pool member never
leaks into the combination-book reads, and an UNTAGGED legacy strategy-pool row
(``admission_metrics`` with no ``kind``) is still returned by the strategy-pool
reads (backward-compat for existing prod rows).
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


async def _seed_iteration(db_conn, *, strategy_code, n=1):
    """A minimal research_iteration_log row to satisfy the iteration_id link."""
    return await db_conn.fetchval(
        """
        INSERT INTO research_iteration_log
            (strategy_code, iteration_number, verdict, statistical_verdict)
        VALUES ($1, $2, 'ITERATE', 'SIGNIFICANT_EDGE')
        RETURNING iteration_id
        """,
        strategy_code, n,
    )


@pytest.mark.asyncio
async def test_persist_and_read_back_combination_member(db_conn):
    from orchestrator.services import combination_book as cb

    iteration_id = await _seed_iteration(db_conn, strategy_code="CMB_A")
    residual_metrics = {
        "alpha_tstat": 3.1, "ic": 0.42, "marginal": 0.07, "max_abs_corr": 0.12,
    }

    pool_id = await cb.add_member(
        db_conn,
        iteration_id=iteration_id,
        strategy_code="CMB_A",
        symbol="BTCUSDT",
        interval_name="1h",
        track="trading",
        residual_metrics=residual_metrics,
        created_by="quant-researcher",
    )
    assert pool_id is not None

    members = await cb.list_members(db_conn)
    assert len(members) == 1
    m = members[0]
    assert m["strategy_code"] == "CMB_A"
    assert m["kind"] == "signal_combination"
    assert m["track"] == "trading"
    assert m["admission_metrics"]["alpha_tstat"] == 3.1
    assert m["surface"] == "CMB_A:BTCUSDT:1h"


@pytest.mark.asyncio
async def test_list_members_filters_by_track(db_conn):
    from orchestrator.services import combination_book as cb

    it_t = await _seed_iteration(db_conn, strategy_code="CMB_T")
    it_h = await _seed_iteration(db_conn, strategy_code="CMB_H")
    await cb.add_member(db_conn, iteration_id=it_t, strategy_code="CMB_T",
                        symbol="BTCUSDT", interval_name="1h", track="trading",
                        residual_metrics={}, created_by="x")
    await cb.add_member(db_conn, iteration_id=it_h, strategy_code="CMB_H",
                        symbol="ETHUSDT", interval_name="4h", track="hedging",
                        residual_metrics={}, created_by="x")

    trading = await cb.list_members(db_conn, track="trading")
    hedging = await cb.list_members(db_conn, track="hedging")
    assert {m["strategy_code"] for m in trading} == {"CMB_T"}
    assert {m["strategy_code"] for m in hedging} == {"CMB_H"}

    # No track filter → both.
    everyone = await cb.list_members(db_conn)
    assert {m["strategy_code"] for m in everyone} == {"CMB_T", "CMB_H"}


@pytest.mark.asyncio
async def test_combination_and_strategy_pool_are_isolated(db_conn):
    """Coexistence isolation: a combination member must NOT appear in the
    strategy-pool reads, and a strategy-pool member must NOT appear in the
    combination-book reads — even though both live in ``signal_pool``."""
    from orchestrator.services import combination_book as cb
    from orchestrator.services import house_book

    # A strategy-pool member self-tagged kind='signal_pool' in admission_metrics
    # (exactly how api/pool.py:evaluate writes it).
    it_pool = await _seed_iteration(db_conn, strategy_code="POOL_X")
    await db_conn.execute(
        """
        INSERT INTO signal_pool
            (iteration_id, strategy_code, symbol, interval_name,
             admission_metrics, status, created_by)
        VALUES ($1, 'POOL_X', 'BTCUSDT', '1h', $2, 'active', 'x')
        """,
        it_pool, {"kind": "signal_pool"},
    )

    # A combination member.
    it_cmb = await _seed_iteration(db_conn, strategy_code="CMB_X")
    await cb.add_member(db_conn, iteration_id=it_cmb, strategy_code="CMB_X",
                        symbol="ETHUSDT", interval_name="4h", track="trading",
                        residual_metrics={"marginal": 0.05}, created_by="x")

    # Combination read sees ONLY the combination member.
    cmb_members = await cb.list_members(db_conn)
    assert {m["strategy_code"] for m in cmb_members} == {"CMB_X"}

    # The live strategy-pool service read (house_book._active_members, which
    # carries the JSONB kind filter) sees ONLY the strategy member — it must
    # never weight a residual combination signal as a strategy.
    svc_members = await house_book._active_members(db_conn)
    assert {m["strategy_code"] for m in svc_members} == {"POOL_X"}


@pytest.mark.asyncio
async def test_untagged_strategy_pool_row_is_still_returned(db_conn):
    """Backward-compat: an EXISTING prod strategy-pool row whose
    ``admission_metrics`` has NO ``kind`` key (e.g. ``{}``, the V147 default)
    must still be returned by the strategy-pool reads, and must NOT leak into
    the combination-book read."""
    from orchestrator.services import combination_book as cb
    from orchestrator.services import house_book

    it = await _seed_iteration(db_conn, strategy_code="LEGACY_X")
    # Default admission_metrics '{}' — no kind tag, like a pre-Phase-3 row.
    await db_conn.execute(
        """
        INSERT INTO signal_pool
            (iteration_id, strategy_code, symbol, interval_name, status, created_by)
        VALUES ($1, 'LEGACY_X', 'BTCUSDT', '1h', 'active', 'x')
        """,
        it,
    )

    # Strategy-pool service read INCLUDES the untagged legacy row.
    svc_members = await house_book._active_members(db_conn)
    assert {m["strategy_code"] for m in svc_members} == {"LEGACY_X"}

    # Combination-book read does NOT see it (kind filter is strict equality).
    cmb_members = await cb.list_members(db_conn)
    assert cmb_members == []


@pytest.mark.asyncio
async def test_aggregate_book_hrp_weights_on_residuals(db_conn):
    """The combination book's aggregate HRP weights are computed on members'
    residual series via ``pool._hrp_for`` — weights sum to ~1 and cover all
    members."""
    from orchestrator.services import combination_book as cb

    from datetime import date

    residuals = {
        "CMB_1:BTCUSDT:1h": {date(2024, 1, i): 0.01 * ((-1) ** i) for i in range(1, 30)},
        "CMB_2:ETHUSDT:1h": {date(2024, 1, i): 0.008 * ((-1) ** (i + 1)) for i in range(1, 30)},
    }
    weights = cb.aggregate_book_weights(residuals)
    assert set(weights) == set(residuals)
    assert abs(sum(weights.values()) - 1.0) < 1e-6
