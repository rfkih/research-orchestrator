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

from datetime import datetime, timedelta
from uuid import uuid4

import pytest

pytestmark = pytest.mark.integration

_AUTH_HEADERS = {"X-Orch-Token": "test-token", "X-Agent-Name": "quant-researcher"}

# A window long enough to clear MIN_FACTOR_OBS (=60 paired obs after the
# leading MOM_LOOKBACK days the momentum factor consumes). Naive datetimes:
# market_data.start_time / backtest_trade.entry_time are TIMESTAMP (naive); the
# orchestrator passes the same naive window to macro_raw (TIMESTAMPTZ, asyncpg
# binds naive as the session TZ). Matches the prod tick path (window comes from
# backtest_run.start_time, a naive TIMESTAMP column).
_WIN_START = datetime(2024, 1, 1)
_WIN_DAYS = 140
_WIN_END = _WIN_START + timedelta(days=_WIN_DAYS)


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


# ── Task 6: POST /combination/evaluate endpoint ──────────────────────────────
#
# These exercise the full handler path through ``integration_client``: seed the
# factor inputs (market_data for 5 syms + macro_raw funding/dvol), seed the
# candidate's backtest_trade rows, then POST and assert the composite shape, the
# disguised-beta short-circuit, and that the admit() path is invoked.

import math


def _btc_close(i: int) -> float:
    """Deterministic BTC close path with day-over-day variation (so MARKET has
    real variance for the OLS to regress on)."""
    return 30000.0 * (1.0 + 0.02 * math.sin(i / 3.0))


async def _seed_market_data(db_conn):
    """Daily closes for all 5 factor symbols over the window. Each symbol gets a
    distinct deterministic path so MOMENTUM/CARRY legs are non-degenerate."""
    syms = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
    base = {"BTCUSDT": 30000.0, "ETHUSDT": 2000.0, "SOLUSDT": 100.0,
            "BNBUSDT": 300.0, "XRPUSDT": 0.5}
    rows = []
    for i in range(_WIN_DAYS):
        st = _WIN_START + timedelta(days=i)
        et = st + timedelta(days=1)
        for j, s in enumerate(syms):
            if s == "BTCUSDT":
                px = _btc_close(i)
            else:
                px = base[s] * (1.0 + 0.015 * math.sin((i + j * 5) / 4.0))
            rows.append((s, "1d", st, et, px, px, px, px, 1.0, 1, 1.0, 1.0, 1.0))
    await db_conn.executemany(
        "INSERT INTO market_data (symbol, interval, start_time, end_time, "
        "open_price, close_price, high_price, low_price, volume, trade_count, "
        "quote_asset_volume, taker_buy_base_volume, taker_buy_quote_volume) "
        "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)",
        rows,
    )


async def _seed_macro(db_conn):
    """Per-symbol funding (CARRY sort key) + BTC DVOL (VOL factor)."""
    syms = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
    rows = []
    for i in range(_WIN_DAYS):
        et = _WIN_START + timedelta(days=i)
        for j, s in enumerate(syms):
            rows.append(("binance_macro", "funding_rate", s, et,
                         0.0001 * (j + 1) + 0.00001 * math.sin(i / 5.0)))
        rows.append(("deribit", "deribit_btc_dvol", None, et,
                     60.0 + 5.0 * math.sin(i / 6.0)))
    await db_conn.executemany(
        "INSERT INTO macro_raw (source, series_id, symbol, event_time, value) "
        "VALUES ($1,$2,$3,$4,$5)",
        rows,
    )


async def _seed_candidate_trades(db_conn, *, run_id, daily_return_fn):
    """One backtest_trade per window day; ``realized_pnl_amount`` is chosen so
    that daily_returns_from_trades (÷ default 100 capital) yields exactly
    ``daily_return_fn(i)`` for day i. Trade exits on the same day it opens."""
    rows = []
    for i in range(1, _WIN_DAYS):  # day 0 has no prior close → no return
        st = _WIN_START + timedelta(days=i, hours=1)
        et = _WIN_START + timedelta(days=i, hours=2)
        pnl = daily_return_fn(i) * 100.0
        rows.append((uuid4(), run_id, "LONG", "CLOSED", "TP", pnl, st, et))
    await db_conn.executemany(
        "INSERT INTO backtest_trade (backtest_trade_id, backtest_run_id, side, "
        "status, exit_reason, realized_pnl_amount, entry_time, exit_time) "
        "VALUES ($1,$2,$3,$4,$5,$6,$7,$8)",
        rows,
    )


def _btc_ret(i: int) -> float:
    return _btc_close(i) / _btc_close(i - 1) - 1.0


@pytest.mark.asyncio
async def test_evaluate_disguised_beta_short_circuits(db_conn, integration_client):
    """A candidate whose daily returns are exactly 2x BTC (pure market beta) →
    neutralization flags disguised_beta and the handler returns early with
    admitted=False, disguised_beta=True (no IC / no admit needed)."""
    await _seed_market_data(db_conn)
    await _seed_macro(db_conn)
    run_id = uuid4()
    # Pure 2x-market candidate → wholly factor-explained → disguised beta.
    await _seed_candidate_trades(
        db_conn, run_id=run_id, daily_return_fn=lambda i: 2.0 * _btc_ret(i)
    )

    body = {
        "backtest_run_id": str(run_id),
        "strategy_code": "CMB_BETA",
        "symbol": "BTCUSDT",
        "interval": "1d",
        "start": _WIN_START.isoformat(),
        "end": _WIN_END.isoformat(),
        "expected_sign": "+",
        "track": "trading",
    }
    r = integration_client.post("/combination/evaluate", json=body,
                                headers=_AUTH_HEADERS)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["neutralization"]["disguised_beta"] is True
    assert out["admission"]["admitted"] is False
    assert out["disguised_beta"] is True


@pytest.mark.asyncio
async def test_evaluate_real_alpha_runs_full_admit_path(db_conn, integration_client):
    """A candidate with idiosyncratic drift NOT explained by the factors →
    not disguised; the handler runs IC + admit and returns the full composite
    (neutralization + ic + admission). Asserts the shape + that admit ran."""
    await _seed_market_data(db_conn)
    await _seed_macro(db_conn)
    run_id = uuid4()
    # 0.3x market (small beta) + a steady idiosyncratic positive drift +
    # alternating idiosyncratic noise → residual carries real, significant alpha.
    def cand(i: int) -> float:
        return 0.3 * _btc_ret(i) + 0.004 + 0.001 * ((-1) ** i)

    await _seed_candidate_trades(db_conn, run_id=run_id, daily_return_fn=cand)

    body = {
        "backtest_run_id": str(run_id),
        "strategy_code": "CMB_ALPHA",
        "symbol": "BTCUSDT",
        "interval": "1d",
        "start": _WIN_START.isoformat(),
        "end": _WIN_END.isoformat(),
        "expected_sign": "+",
        "track": "trading",
    }
    r = integration_client.post("/combination/evaluate", json=body,
                                headers=_AUTH_HEADERS)
    assert r.status_code == 200, r.text
    out = r.json()
    # Not a disguised-beta short-circuit → full path ran.
    assert out["neutralization"]["disguised_beta"] is False
    assert "ic" in out and "ic" in out["ic"]
    assert "admission" in out and "reasons" in out["admission"]
    # Idiosyncratic alpha should be significant for this construction.
    assert out["admission"]["reasons"]["alpha_significant"] is True


@pytest.mark.asyncio
async def test_evaluate_resolves_iteration_id(db_conn, integration_client):
    """When the body passes iteration_id (not backtest_run_id), the handler
    resolves it to the run's backtest_run_id + strategy_code via the iteration
    log and still produces the composite."""
    await _seed_market_data(db_conn)
    await _seed_macro(db_conn)
    run_id = uuid4()
    await _seed_candidate_trades(
        db_conn, run_id=run_id, daily_return_fn=lambda i: 2.0 * _btc_ret(i)
    )
    iteration_id = await db_conn.fetchval(
        """
        INSERT INTO research_iteration_log
            (strategy_code, iteration_number, backtest_run_id, verdict,
             statistical_verdict)
        VALUES ('CMB_IT', 1, $1, 'ITERATE', 'SIGNIFICANT_EDGE')
        RETURNING iteration_id
        """,
        run_id,
    )
    body = {
        "iteration_id": str(iteration_id),
        "symbol": "BTCUSDT",
        "interval": "1d",
        "start": _WIN_START.isoformat(),
        "end": _WIN_END.isoformat(),
        "expected_sign": "+",
        "track": "trading",
    }
    r = integration_client.post("/combination/evaluate", json=body,
                                headers=_AUTH_HEADERS)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["strategy_code"] == "CMB_IT"
    assert out["neutralization"]["disguised_beta"] is True
