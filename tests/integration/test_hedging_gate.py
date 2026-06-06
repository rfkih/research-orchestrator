import pytest
from datetime import datetime

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
