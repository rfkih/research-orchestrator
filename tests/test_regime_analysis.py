"""Regime analysis market_data fallback — the SELECT must use the REAL schema.

Production bug: ``_classify_from_market_data`` queried ``market_data`` with
columns that do not exist (``open_time``, ``close``, ``high``, ``low``). The
real V1 baseline columns are ``start_time``, ``close_price``, ``high_price``,
``low_price``. The wrong query threw ``UndefinedColumnError`` which the broad
``except Exception`` swallowed → "market_data unavailable" → returned None →
regime classification SILENTLY no-op'd in prod.

The fix aliases the real columns back to the names the Python reads
(``open_time``/``close``/``hl_range``) so only the SQL string changes.

These tests use a fake asyncpg connection that records the executed SQL and
returns canned rows — no live DB required — matching the repo-native style
(see ``tests/test_account_scoping.py``).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any

from orchestrator.services.regime_analysis import _classify_from_market_data


class _FakeConn:
    """Records the SQL + params of each query, returns pre-seeded rows."""

    def __init__(self, fetch_rows: Any = None) -> None:
        self.fetch_rows = fetch_rows if fetch_rows is not None else []
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, sql: str, *params: Any) -> Any:
        self.calls.append((sql, params))
        return self.fetch_rows


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _make_rows(n: int) -> list[dict]:
    """Build n canned bars using the ALIAS keys the code reads
    (open_time/close/hl_range), as the aliased SQL would return them.

    A rising close ensures bars land above their SMA-200 (BULL), giving a
    deterministic non-None classification."""
    base = datetime(2025, 1, 1)
    rows = []
    for i in range(n):
        rows.append(
            {
                "open_time": base + timedelta(hours=i),
                "close": 100.0 + i,        # strictly rising
                "hl_range": 1.0 + (i % 5),  # some vol variation
            }
        )
    return rows


def _make_trades(entry_times: list[datetime]) -> list[dict]:
    return [
        {
            "entry_time": et,
            "realized_pnl_amount": 5.0,
            "notional_size": 100.0,
        }
        for et in entry_times
    ]


# ── SQL shape: must use REAL columns, not the bad ones ─────────────────────


def test_query_uses_real_market_data_columns() -> None:
    conn = _FakeConn(fetch_rows=_make_rows(210))
    trades = _make_trades([datetime(2025, 1, 1) + timedelta(hours=205)])

    _run(_classify_from_market_data(conn, trades, "BTCUSDT", "1h"))

    assert conn.calls, "expected market_data to be queried"
    sql, params = conn.calls[0]

    # Real V1 baseline columns must be present.
    assert "start_time" in sql
    assert "close_price" in sql
    assert "high_price" in sql
    assert "low_price" in sql

    # The non-existent columns from the bug must NOT be selected.
    # (bare aliases stay in the SELECT-list as "AS open_time"/"AS close", so
    # check for the *source* column forms that would have hit the DB.)
    assert "high - low" not in sql, "must not subtract bare high/low"
    assert "ORDER BY start_time" in sql, "must order by the real time column"
    assert "FROM market_data" in sql
    assert params == ("BTCUSDT", "1h")


# ── End-to-end read path with aliased rows must classify (non-None) ────────


def test_classifies_with_aliased_rows_returns_non_none() -> None:
    rows = _make_rows(210)
    conn = _FakeConn(fetch_rows=rows)
    # Match a trade onto a real bar time well past the SMA-200 warmup.
    entry = rows[205]["open_time"]
    trades = _make_trades([entry])

    result = _run(_classify_from_market_data(conn, trades, "BTCUSDT", "1h"))

    assert result is not None, "read path must produce a classification"
    assert len(result) == 1
    assert result[0]["regime"].startswith("BULL")  # rising closes ⇒ above SMA200
    assert result[0]["pnl"] == 5.0
    assert result[0]["notional"] == 100.0


def test_too_few_rows_returns_none() -> None:
    conn = _FakeConn(fetch_rows=_make_rows(209))
    trades = _make_trades([datetime(2025, 1, 1)])

    result = _run(_classify_from_market_data(conn, trades, "BTCUSDT", "1h"))

    assert result is None
