"""Real-Postgres integration test for the /signal-screen service path.

Exercises what the mocked-repo unit tests cannot: the information_schema
feature_store resolution, the feature_values scope-fallback SQL, the
point-in-time alignment against real rows, the research_journal insert
(CHECK-constrained entry_type + JSONB codec), and the multiplicity count
incrementing across calls.

Skips cleanly when the dev Postgres on 127.0.0.1:5432 is unreachable
(the noproc harness needs it -- see tests/integration/conftest.py).
"""
from __future__ import annotations

import math
import random
import socket
from datetime import datetime, timedelta

import pytest

from orchestrator.services import signal_screen as svc

pytestmark = pytest.mark.integration

T0 = datetime(2024, 1, 1)


def _dev_pg_reachable() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 5432), timeout=0.5):
            return True
    except OSError:
        return False


@pytest.fixture
def screen_conn(request):
    """db_conn, guarded: skip (not error) when the dev PG is down."""
    if not _dev_pg_reachable():
        pytest.skip("dev Postgres 127.0.0.1:5432 unreachable -- noproc harness needs it")
    return request.getfixturevalue("db_conn")


async def _seed(conn, *, n_bars: int = 120) -> None:
    rng = random.Random(7)
    px = 100.0
    for k in range(n_bars):
        ts = T0 + timedelta(hours=k)
        px *= math.exp(rng.gauss(0, 0.01))
        await conn.execute(
            """
            INSERT INTO market_data
              (symbol, interval, start_time, end_time, open_price, close_price,
               high_price, low_price, volume, trade_count, quote_asset_volume,
               taker_buy_base_volume, taker_buy_quote_volume)
            VALUES ('BTCUSDT', '1h', $1, $2, $3, $3, $3, $3, 1, 1, 1, 1, 1)
            """,
            ts, ts + timedelta(hours=1), px,
        )
        await conn.execute(
            """
            INSERT INTO feature_store
              (symbol, interval, start_time, end_time, price, rsi)
            VALUES ('BTCUSDT', '1h', $1, $2, $3, $4)
            """,
            ts, ts + timedelta(hours=1), px, rng.uniform(20, 80),
        )
        # Macro series: symbol-less/interval-less, stamped 30 min BEFORE the
        # bar -- visible at that bar via the PIT join.
        await conn.execute(
            """
            INSERT INTO feature_values (feature_name, version, symbol, interval, ts, value)
            VALUES ('macro_x', 1, '', '', $1, $2)
            """,
            ts - timedelta(minutes=30), rng.gauss(0, 1),
        )


@pytest.mark.asyncio
async def test_signal_screen_feature_store_path_journals_and_counts(screen_conn):
    conn = screen_conn
    await _seed(conn)

    out = await svc.run_signal_screen(
        conn,
        agent_name="integration-test",
        signal="rsi",
        instruments=["BTCUSDT"],
        interval="1h",
        horizons_bars=[1, 6],
        transform="zscore",
        start=T0,
        end=T0 + timedelta(days=30),
    )
    assert out["resolved_source"] == "feature_store"
    assert len(out["rows"]) == 2
    assert out["multiplicity"]["prior_screens_same_family"] == 0

    row = await conn.fetchrow(
        "SELECT entry_type, structured_data, created_by FROM research_journal "
        "WHERE structured_data->>'kind' = 'signal_screen'"
    )
    assert row is not None
    assert row["structured_data"]["signal"] == "rsi"
    assert row["created_by"] == "integration-test"
    assert row["entry_type"] in ("ANTI_PATTERN", "CROSS_STRATEGY_FINDING")

    # Second screen of the same family sees the first in its multiplicity.
    out2 = await svc.run_signal_screen(
        conn,
        agent_name="integration-test",
        signal="rsi",
        instruments=["BTCUSDT"],
        interval="1h",
        horizons_bars=[1],
        transform="rank",
        start=T0,
        end=T0 + timedelta(days=30),
    )
    assert out2["multiplicity"]["prior_screens_same_family"] == 1


@pytest.mark.asyncio
async def test_signal_screen_feature_values_macro_path(screen_conn):
    conn = screen_conn
    await _seed(conn)

    out = await svc.run_signal_screen(
        conn,
        agent_name="integration-test",
        signal="macro_x",
        instruments=["BTCUSDT"],
        interval="1h",
        horizons_bars=[6],
        transform="zscore",
        start=T0,
        end=T0 + timedelta(days=30),
    )
    assert out["resolved_source"] == "feature_values"
    r = out["rows"][0]
    assert r["n_obs"] >= svc.MIN_OBS  # PIT join found values for the bars
