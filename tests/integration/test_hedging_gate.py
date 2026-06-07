import math
import uuid
import pytest
from datetime import datetime, timedelta

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_fetch_daily_closes_returns_ordered_series(db_conn):
    from orchestrator.repo import market_data
    rows = [("BTCUSDT", "1d", datetime(2024, 1, d), 100.0 + d) for d in range(1, 6)]
    await db_conn.executemany(
        "INSERT INTO market_data (symbol, interval, start_time, end_time, open_price, close_price, high_price, low_price, volume, trade_count, quote_asset_volume, taker_buy_base_volume, taker_buy_quote_volume) "
        "VALUES ($1,$2,$3,$3,0,$4,0,0,0,0,0,0,0)",
        rows,
    )
    series = await market_data.fetch_daily_closes(
        db_conn, symbol="BTCUSDT", interval="1d",
        start=datetime(2024, 1, 1), end=datetime(2024, 1, 6),
    )
    assert [c for _, c in series] == [101.0, 102.0, 103.0, 104.0, 105.0]
    assert series == sorted(series)  # ascending by date


def _volatile_closes(n_days):
    """Deterministic volatile-but-rising underlying: modest daily up-drift
    punctuated by periodic sharp drops → real drawdowns, moderate Sharpe,
    a benchmark a smart hedge can beat risk-adjusted."""
    px = 100.0
    out = [px]
    for d in range(n_days - 1):
        r = -0.05 if d % 7 == 3 else 0.012
        px *= 1.0 + r
        out.append(px)
    return out


async def _seed_volatile_buyhold(db_conn, *, symbol, start, n_days):
    closes = _volatile_closes(n_days)
    rows = [
        (symbol, "1d", start + timedelta(days=d), start + timedelta(days=d),
         0, closes[d], 0, 0, 0, 0, 0, 0, 0)
        for d in range(n_days)
    ]
    await db_conn.executemany(
        "INSERT INTO market_data (symbol, interval, start_time, end_time, open_price, close_price, high_price, low_price, volume, trade_count, quote_asset_volume, taker_buy_base_volume, taker_buy_quote_volume) "
        "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)",
        rows,
    )


async def _seed_trades(db_conn, run_id, *, start, daily_pnls, initial_capital=100.0):
    """One trade per day with the given realized pnl (quote amount), exiting
    on that calendar day."""
    rows = []
    for d, pnl in enumerate(daily_pnls):
        t = start + timedelta(days=d)
        rows.append((
            uuid.uuid4(), run_id, "LONG", "CLOSED", "TP",
            float(pnl), 0.0, initial_capital, 0.0, 0.0, 1,
            t, t, 0.0, 0.0, 0.0, 0.0, "BULL",
        ))
    await db_conn.executemany(
        "INSERT INTO backtest_trade (backtest_trade_id, backtest_run_id, side, status, exit_reason, "
        "realized_pnl_amount, realized_r_multiple, notional_size, max_favorable_excursion_r, "
        "max_adverse_excursion_r, bars_held, entry_time, exit_time, entry_adx, entry_rsi, "
        "entry_close_location_value, entry_relative_volume20, entry_trend_regime) "
        "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18)",
        rows,
    )


async def _seed_equity_points(db_conn, run_id, *, start, equities):
    """One backtest_equity_point per calendar day with the given total_equity."""
    rows = []
    prev = None
    for d, eq in enumerate(equities):
        dt = (start + timedelta(days=d)).date()
        dr = None if prev in (None, 0) else (eq / prev - 1.0) * 100.0
        rows.append((
            run_id, uuid.uuid4(), dt, 0.0, float(eq), float(eq), 0.0,
            None if dr is None else float(dr), 0,
        ))
        prev = eq
    await db_conn.executemany(
        "INSERT INTO backtest_equity_point (backtest_run_id, account_id, equity_date, "
        "cash_balance, asset_value, total_equity, drawdown_percent, daily_return_pct, "
        "open_positions) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)",
        rows,
    )


def _rising_shallow_dd_equity(n_days, start=1000.0):
    """Rising equity with only shallow drawdowns: steady up-drift with small
    periodic dips. Clearly beats a choppy buy-hold risk-adjusted (high Sharpe,
    low maxDD) so the hedging gate PASSES — and the metrics come straight from
    the seeded curve."""
    eq = start
    out = [eq]
    for d in range(n_days - 1):
        r = -0.004 if d % 11 == 5 else 0.006
        eq *= 1.0 + r
        out.append(eq)
    return out


@pytest.mark.asyncio
async def test_evaluate_uses_equity_points_when_present(db_conn):
    """When backtest_equity_point rows exist, the strat metrics are derived from
    that REAL total_equity series (not trade reconstruction), and the source is
    flagged. A rising shallow-DD curve clearly beats the choppy buy-hold."""
    from orchestrator.services import hedging_gate
    symbol = "BTCUSDT"
    start = datetime(2024, 4, 1)
    n = 120
    await _seed_volatile_buyhold(db_conn, symbol=symbol, start=start, n_days=n)
    run_id = uuid.uuid4()
    equities = _rising_shallow_dd_equity(n)
    await _seed_equity_points(db_conn, run_id, start=start, equities=equities)
    # Seed DELIBERATELY-MISLEADING trades that would reconstruct a LOSING curve,
    # to prove the equity-point path is used (not trades).
    await _seed_trades(db_conn, run_id, start=start, daily_pnls=[-5.0] * n)
    end = start + timedelta(days=n)
    out = await hedging_gate.evaluate(
        db_conn, backtest_run_id=run_id, symbol=symbol, interval="1d",
        start=start, end=end,
    )
    assert out["strat_equity_source"] == "equity_points"
    # CAGR is positive (rising equity), NOT the negative the bad trades imply.
    assert out["strategy"]["cagr_pct"] > 0
    # maxDD matches the seeded curve's true peak-to-trough (shallow), computed
    # independently here from total_equity.
    expected_mdd = hedging_gate._max_drawdown_pct(equities)
    assert out["strategy"]["max_drawdown_pct"] == pytest.approx(expected_mdd, abs=1e-9)
    assert expected_mdd < 5.0  # shallow
    # Genuinely beats buy-hold risk-adjusted → PASS.
    assert out["decision"]["passed"] is True
    assert out["significance"]["sharpe_improvement_significant"] is True
    assert out["passed"] is True


@pytest.mark.asyncio
async def test_evaluate_falls_back_to_trades_without_equity_points(db_conn):
    """A run with NO backtest_equity_point rows still works via the legacy
    trade-reconstruction path, flagged as such — nothing regresses."""
    from orchestrator.services import hedging_gate
    symbol = "ETHUSDT"
    start = datetime(2024, 5, 1)
    n = 120
    await _seed_volatile_buyhold(db_conn, symbol=symbol, start=start, n_days=n)
    run_id = uuid.uuid4()
    # No equity points seeded — only trades.
    pnls = [0.6 + (0.05 if d % 2 == 0 else -0.05) for d in range(n)]
    await _seed_trades(db_conn, run_id, start=start, daily_pnls=pnls)
    end = start + timedelta(days=n)
    out = await hedging_gate.evaluate(
        db_conn, backtest_run_id=run_id, symbol=symbol, interval="1d",
        start=start, end=end,
    )
    assert out["strat_equity_source"] == "trades"
    assert out["strategy"]["cagr_pct"] is not None
    # Same superior-strategy shape as the legacy path → still PASSes.
    assert out["passed"] is True


@pytest.mark.asyncio
async def test_fetch_equity_points_ascending(db_conn):
    from orchestrator.repo import backtest_equity
    run_id = uuid.uuid4()
    start = datetime(2024, 7, 1)
    equities = [1000.0, 1010.0, 1005.0, 1030.0]
    await _seed_equity_points(db_conn, run_id, start=start, equities=equities)
    pts = await backtest_equity.fetch_equity_points(db_conn, run_id)
    assert [eq for _, eq in pts] == equities
    assert pts == sorted(pts)  # ascending by date


@pytest.mark.asyncio
async def test_evaluate_returns_composite_dict(db_conn):
    from orchestrator.services import hedging_gate
    symbol = "BTCUSDT"
    start = datetime(2024, 1, 1)
    n = 90
    await _seed_volatile_buyhold(db_conn, symbol=symbol, start=start, n_days=n)
    run_id = uuid.uuid4()
    # a benign trade so fetch_trades returns rows
    await _seed_trades(db_conn, run_id, start=start, daily_pnls=[0.5] * n)
    end = start + timedelta(days=n)
    out = await hedging_gate.evaluate(
        db_conn, backtest_run_id=run_id, symbol=symbol, interval="1d",
        start=start, end=end,
    )
    assert set(out.keys()) == {
        "passed", "decision", "significance", "benchmark", "strategy",
        "strat_equity_source",
    }
    assert isinstance(out["passed"], bool)
    assert "cagr_pct" in out["benchmark"]
    assert "cagr_pct" in out["strategy"]
    assert "passed" in out["decision"]
    assert "sharpe_improvement_significant" in out["significance"]


@pytest.mark.asyncio
async def test_evaluate_forces_daily_benchmark_interval(db_conn):
    """The buy-hold benchmark must ALWAYS be fetched at the daily interval,
    regardless of the strategy ``interval`` the caller passes — otherwise the
    sqrt(252)/365-day annualisations in buy_hold_metrics are silently wrong.

    Only DAILY bars are seeded; calling evaluate with a NON-daily interval must
    still produce real benchmark metrics (proving it fetched "1d", not "4h")."""
    from orchestrator.services import hedging_gate
    symbol = "BTCUSDT"
    start = datetime(2024, 2, 1)
    n = 90
    await _seed_volatile_buyhold(db_conn, symbol=symbol, start=start, n_days=n)
    run_id = uuid.uuid4()
    await _seed_trades(db_conn, run_id, start=start, daily_pnls=[0.5] * n)
    end = start + timedelta(days=n)
    out = await hedging_gate.evaluate(
        db_conn, backtest_run_id=run_id, symbol=symbol,
        interval="4h",  # NOT daily — must be ignored for the benchmark fetch
        start=start, end=end,
    )
    # Daily bars exist, so the forced "1d" fetch finds them → real metrics.
    assert out["benchmark"]["cagr_pct"] is not None
    assert out["benchmark"]["sharpe"] is not None
    assert out["benchmark"]["max_drawdown_pct"] is not None


@pytest.mark.asyncio
async def test_evaluate_superior_strategy_passes(db_conn):
    from orchestrator.services import hedging_gate
    symbol = "ETHUSDT"
    start = datetime(2024, 3, 1)
    n = 120
    # choppy buy-hold (high vol, drawdowns, modest net up-drift)
    await _seed_volatile_buyhold(db_conn, symbol=symbol, start=start, n_days=n)
    run_id = uuid.uuid4()
    # strategy: steady small daily gains (slight alternation → nonzero variance,
    # very high Sharpe) clearing the buy-hold CAGR floor with near-zero drawdown.
    pnls = [0.6 + (0.05 if d % 2 == 0 else -0.05) for d in range(n)]
    await _seed_trades(db_conn, run_id, start=start, daily_pnls=pnls)
    end = start + timedelta(days=n)
    out = await hedging_gate.evaluate(
        db_conn, backtest_run_id=run_id, symbol=symbol, interval="1d",
        start=start, end=end,
    )
    assert out["decision"]["passed"] is True
    assert out["significance"]["sharpe_improvement_significant"] is True
    assert out["passed"] is True


@pytest.mark.asyncio
async def test_evaluate_buyhold_equivalent_fails(db_conn):
    from orchestrator.services import hedging_gate
    symbol = "SOLUSDT"
    start = datetime(2024, 6, 1)
    n = 120
    await _seed_volatile_buyhold(db_conn, symbol=symbol, start=start, n_days=n)
    # strategy that replicates the underlying: its daily pnl tracks the same
    # close path the buy-hold uses (on a ~$100 position) → strat ≈ bench, so
    # no material risk-adjusted edge.
    closes = _volatile_closes(n)
    pnls = [closes[d + 1] - closes[d] for d in range(n - 1)] + [0.0]
    run_id = uuid.uuid4()
    await _seed_trades(db_conn, run_id, start=start, daily_pnls=pnls)
    end = start + timedelta(days=n)
    out = await hedging_gate.evaluate(
        db_conn, backtest_run_id=run_id, symbol=symbol, interval="1d",
        start=start, end=end,
    )
    assert out["passed"] is False


# ── Task 6: tick.py gate selection by track ─────────────────────────────────
#
# Driving a full run_tick requires a JVM mock + account_strategy/hypothesis_audit
# seeding, which is far heavier than the decision under test. Per the plan, the
# track-selection decision is extracted into tick._select_track_gate and tested
# directly. The trading path's UNCHANGED behavior is additionally proven by the
# default (non-integration) tick suite staying green.

from contextlib import asynccontextmanager


class _DbShim:
    """Minimal Database stand-in: .acquire() yields the test connection so
    tick._select_track_gate's `async with db.acquire() as conn` works against
    the live integration DB."""

    def __init__(self, conn):
        self._conn = conn

    @asynccontextmanager
    async def acquire(self):
        yield self._conn


@pytest.mark.asyncio
async def test_select_track_gate_hedging_passes_via_equity_gate(db_conn):
    """track=hedging + a backtest that beats buy-hold risk-adjusted → PASS via
    the hedging path, NOT the V11/V60 path. Mirrors test_evaluate_superior."""
    from orchestrator.services import tick

    symbol = "ETHUSDT"
    start = datetime(2024, 3, 1)
    n = 120
    await _seed_volatile_buyhold(db_conn, symbol=symbol, start=start, n_days=n)
    run_id = uuid.uuid4()
    pnls = [0.6 + (0.05 if d % 2 == 0 else -0.05) for d in range(n)]
    await _seed_trades(db_conn, run_id, start=start, daily_pnls=pnls)
    end = start + timedelta(days=n)

    # Trading-path verdicts that would FAIL V60 (ITERATE) — proving the hedging
    # branch overrides them rather than deferring to V11/V60.
    stat_v, dec_v, payload = await tick._select_track_gate(
        db=_DbShim(db_conn),
        track="hedging",
        stat_v="INSUFFICIENT_EVIDENCE",
        dec_v="ITERATE",
        backtest_run_id=run_id,
        instrument=symbol,
        window_start=start,
        window_end=end,
    )
    # Reaches PASS through the hedging equity-level gate.
    assert stat_v == "SIGNIFICANT_EDGE"
    assert dec_v == "PASS"
    # Full hedging verdict is stashed for audit (the caller writes it to
    # metrics_snapshot.hedging_gate).
    assert payload is not None
    assert payload["passed"] is True
    assert set(payload.keys()) == {
        "passed", "decision", "significance", "benchmark", "strategy",
        "strat_equity_source",
    }


@pytest.mark.asyncio
async def test_select_track_gate_hedging_fails_routes_iterate(db_conn):
    """track=hedging + a buy-hold-equivalent backtest → non-PASS (ITERATE), the
    same control-flow signal the V60-miss case uses."""
    from orchestrator.services import tick

    symbol = "SOLUSDT"
    start = datetime(2024, 6, 1)
    n = 120
    await _seed_volatile_buyhold(db_conn, symbol=symbol, start=start, n_days=n)
    closes = _volatile_closes(n)
    pnls = [closes[d + 1] - closes[d] for d in range(n - 1)] + [0.0]
    run_id = uuid.uuid4()
    await _seed_trades(db_conn, run_id, start=start, daily_pnls=pnls)
    end = start + timedelta(days=n)

    stat_v, dec_v, payload = await tick._select_track_gate(
        db=_DbShim(db_conn),
        track="hedging",
        stat_v="NO_EDGE",
        dec_v="DISCARD",
        backtest_run_id=run_id,
        instrument=symbol,
        window_start=start,
        window_end=end,
    )
    assert dec_v == "ITERATE"
    assert payload is not None and payload["passed"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("track", [None, "trading"])
async def test_select_track_gate_trading_path_unchanged(db_conn, track):
    """track in (None, 'trading') is a pure pass-through: V11/V60 verdicts are
    returned verbatim and NO hedging payload is produced. (No DB work either —
    proven by passing a closed-shim that would error if .acquire() were used.)"""
    from orchestrator.services import tick

    class _ExplodingDb:
        @asynccontextmanager
        async def acquire(self):
            raise AssertionError("trading path must not touch the DB / hedging gate")
            yield  # pragma: no cover

    stat_v, dec_v, payload = await tick._select_track_gate(
        db=_ExplodingDb(),
        track=track,
        stat_v="SIGNIFICANT_EDGE",
        dec_v="PASS",
        backtest_run_id=uuid.uuid4(),
        instrument="BTCUSDT",
        window_start=datetime(2024, 1, 1),
        window_end=datetime(2024, 4, 1),
    )
    assert (stat_v, dec_v, payload) == ("SIGNIFICANT_EDGE", "PASS", None)
