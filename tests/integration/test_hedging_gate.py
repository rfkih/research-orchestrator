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
    assert set(out.keys()) == {"passed", "decision", "significance", "benchmark", "strategy"}
    assert isinstance(out["passed"], bool)
    assert "cagr_pct" in out["benchmark"]
    assert "cagr_pct" in out["strategy"]
    assert "passed" in out["decision"]
    assert "sharpe_improvement_significant" in out["significance"]


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
