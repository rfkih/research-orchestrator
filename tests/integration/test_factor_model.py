"""Integration tests for the dual-track Phase 3 factor-combination layer."""
import pytest
from datetime import datetime, timezone

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_fetch_series_ascending(db_conn):
    from orchestrator.repo import macro_raw
    rows = [("binance_macro", "funding_rate", "BTCUSDT",
             datetime(2024, 1, d, tzinfo=timezone.utc), 0.0001 * d) for d in range(1, 5)]
    await db_conn.executemany(
        "INSERT INTO macro_raw (source, series_id, symbol, event_time, value, source_uri, content_hash, ingestion_time) "
        "VALUES ($1,$2,$3,$4,$5,'u','h',$4)", rows)
    # Half-open [start, end): end=Jan 5 (exclusive) includes the Jan 1-4 rows.
    out = await macro_raw.fetch_series(db_conn, series_id="funding_rate", symbol="BTCUSDT",
                                       start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                                       end=datetime(2024, 1, 5, tzinfo=timezone.utc))
    assert [round(v, 4) for _, v in out] == [0.0001, 0.0002, 0.0003, 0.0004]
    assert out == sorted(out)
    # end=Jan 4 (exclusive) drops the Jan 4 row, proving the upper bound is open.
    out_open = await macro_raw.fetch_series(db_conn, series_id="funding_rate", symbol="BTCUSDT",
                                            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                                            end=datetime(2024, 1, 4, tzinfo=timezone.utc))
    assert [round(v, 4) for _, v in out_open] == [0.0001, 0.0002, 0.0003]
    # symbol=None ignores the symbol filter (single-series feeds like DVOL).
    out_all = await macro_raw.fetch_series(db_conn, series_id="funding_rate", symbol=None,
                                           start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                                           end=datetime(2024, 1, 5, tzinfo=timezone.utc))
    assert len(out_all) == 4
