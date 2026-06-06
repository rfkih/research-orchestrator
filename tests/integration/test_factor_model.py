"""Integration tests for the dual-track Phase 3 factor-combination layer."""
import pytest
from datetime import datetime, timedelta, timezone

pytestmark = pytest.mark.integration


async def _seed_closes(db_conn, *, symbol, start, prices):
    """One daily market_data bar per element of ``prices`` (close only)."""
    rows = [
        (symbol, "1d", start + timedelta(days=d), start + timedelta(days=d),
         0, float(px), 0, 0, 0, 0, 0, 0, 0)
        for d, px in enumerate(prices)
    ]
    await db_conn.executemany(
        "INSERT INTO market_data (symbol, interval, start_time, end_time, open_price, "
        "close_price, high_price, low_price, volume, trade_count, quote_asset_volume, "
        "taker_buy_base_volume, taker_buy_quote_volume) "
        "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)",
        rows,
    )


async def _seed_macro(db_conn, *, series_id, symbol, start, values):
    rows = [
        (series_id, symbol, start + timedelta(days=d), float(v))
        for d, v in enumerate(values)
    ]
    await db_conn.executemany(
        "INSERT INTO macro_raw (source, series_id, symbol, event_time, value, "
        "source_uri, content_hash, ingestion_time) "
        "VALUES ('s',$1,$2,$3,$4,'u','h',$3)",
        rows,
    )


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


# ── Task 2: build_factors assembler ──────────────────────────────────────────
@pytest.mark.asyncio
async def test_build_factors_returns_four_series(db_conn):
    from orchestrator.services import factor_model as fm

    start = datetime(2024, 1, 1)  # naive: market_data.start_time is TIMESTAMP
    n = 40  # > MOM_LOOKBACK(20) so momentum has realised days
    # Distinct trending paths per symbol so the cross-sectional ranks are stable.
    drifts = {"BTCUSDT": 1.03, "ETHUSDT": 1.02, "SOLUSDT": 1.01,
              "BNBUSDT": 0.99, "XRPUSDT": 0.98}
    for sym, g in drifts.items():
        await _seed_closes(db_conn, symbol=sym, start=start,
                           prices=[100.0 * (g ** d) for d in range(n)])
    # Per-symbol funding (CARRY sort key) — A-high through E-low.
    fundings = {"BTCUSDT": 0.03, "ETHUSDT": 0.02, "SOLUSDT": 0.01,
                "BNBUSDT": -0.01, "XRPUSDT": -0.02}
    for sym, f in fundings.items():
        await _seed_macro(db_conn, series_id="funding_rate", symbol=sym,
                          start=start, values=[f] * n)
    # BTC DVOL (VOL) — single series, varying so daily change is nonzero.
    await _seed_macro(db_conn, series_id="deribit_btc_dvol", symbol="BTCUSDT",
                      start=start, values=[50.0 + (d % 5) for d in range(n)])

    end = start + timedelta(days=n)
    out = await fm.build_factors(db_conn, start, end)

    assert set(out.keys()) == {"MARKET", "MOMENTUM", "CARRY", "VOL"}
    for key, series in out.items():
        assert isinstance(series, dict) and series, f"{key} empty"
        for d, r in series.items():
            from datetime import date as _date
            assert isinstance(d, _date)
            assert isinstance(r, float)
    # MARKET == BTC daily simple return (3% drift each step).
    some_day = sorted(out["MARKET"])[0]
    assert round(out["MARKET"][some_day], 4) == 0.03


@pytest.mark.asyncio
async def test_build_factors_omits_vol_when_dvol_absent(db_conn):
    """Degrade gracefully: no DVOL rows → VOL omitted, other factors survive,
    no crash."""
    from orchestrator.services import factor_model as fm

    start = datetime(2024, 5, 1)  # naive: market_data.start_time is TIMESTAMP
    n = 40
    drifts = {"BTCUSDT": 1.03, "ETHUSDT": 1.02, "SOLUSDT": 1.01,
              "BNBUSDT": 0.99, "XRPUSDT": 0.98}
    for sym, g in drifts.items():
        await _seed_closes(db_conn, symbol=sym, start=start,
                           prices=[100.0 * (g ** d) for d in range(n)])
    fundings = {"BTCUSDT": 0.03, "ETHUSDT": 0.02, "SOLUSDT": 0.01,
                "BNBUSDT": -0.01, "XRPUSDT": -0.02}
    for sym, f in fundings.items():
        await _seed_macro(db_conn, series_id="funding_rate", symbol=sym,
                          start=start, values=[f] * n)
    # NB: no deribit_btc_dvol rows seeded.

    end = start + timedelta(days=n)
    out = await fm.build_factors(db_conn, start, end)

    assert "VOL" not in out
    assert {"MARKET", "MOMENTUM", "CARRY"}.issubset(out.keys())


@pytest.mark.asyncio
async def test_build_factors_empty_db_returns_empty_no_crash(db_conn):
    """No source data at all → empty result, never crashes."""
    from orchestrator.services import factor_model as fm

    start = datetime(2023, 1, 1)  # naive: market_data.start_time is TIMESTAMP
    out = await fm.build_factors(db_conn, start, start + timedelta(days=30))
    assert out == {}
