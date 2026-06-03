"""Unit tests for house_book.classify_book — the pool↔live reconciliation.

Pure function, no DB. Pins the load-bearing routing: a member with a live row
syncs, a member without one is flagged for the operator (never silently
dropped), a stale live row gets soft-disabled, and a member still missing its
weight waits for a rebalance.
"""
from __future__ import annotations

from orchestrator.services import house_book
from orchestrator.services.house_book import _key


def _member(code, sym, iv, w):
    return {"key": _key(code, sym, iv), "strategy_code": code, "symbol": sym,
            "interval_name": iv, "pool_weight": w}


def _row(code, sym, iv, w, asid="as-1", weight_source="HOUSE_BOOK"):
    return {"key": _key(code, sym, iv), "account_strategy_id": asid,
            "strategy_code": code, "symbol": sym, "interval_name": iv,
            "portfolio_weight": w, "weight_source": weight_source,
            "enabled": True, "simulated": True}


def test_member_with_live_row_and_weight_syncs():
    out = house_book.classify_book(
        [_member("XS_MOM", "SOLUSDT", "1h", 0.3)],
        [_row("XS_MOM", "SOLUSDT", "1h", 1.0)],
    )
    assert len(out["to_sync"]) == 1
    assert out["to_sync"][0]["pool_weight"] == 0.3
    assert out["to_sync"][0]["account_strategy_id"] == "as-1"
    assert out["unmaterialized"] == [] and out["to_zero"] == []


def test_member_without_live_row_is_unmaterialized():
    out = house_book.classify_book(
        [_member("XS_MOM", "SOLUSDT", "1h", 0.3)],
        [],
    )
    assert [m["key"] for m in out["unmaterialized"]] == ["XS_MOM:SOLUSDT:1h"]
    assert out["to_sync"] == []


def test_member_with_null_weight_needs_rebalance():
    out = house_book.classify_book(
        [_member("XS_MOM", "SOLUSDT", "1h", None)],
        [_row("XS_MOM", "SOLUSDT", "1h", 1.0)],
    )
    assert [m["key"] for m in out["needs_weight"]] == ["XS_MOM:SOLUSDT:1h"]
    assert out["to_sync"] == []


def test_stale_live_row_is_zeroed_only_if_weighted():
    members = [_member("XS_MOM", "SOLUSDT", "1h", 0.5)]
    live = [
        _row("XS_MOM", "SOLUSDT", "1h", 1.0, "keep"),
        _row("OLD", "BTCUSDT", "4h", 0.4, "stale-weighted"),
        _row("OLD2", "ETHUSDT", "4h", 0.0, "stale-already-zero"),
    ]
    out = house_book.classify_book(members, live)
    # Only the non-member row that still carries weight is evicted.
    assert [z["account_strategy_id"] for z in out["to_zero"]] == ["stale-weighted"]


def test_distinct_intervals_are_distinct_surfaces():
    # Same code+symbol on 1h vs 4h must NOT collide — one can be a member while
    # the other is evicted.
    out = house_book.classify_book(
        [_member("XS_MOM", "SOLUSDT", "1h", 0.6)],
        [_row("XS_MOM", "SOLUSDT", "1h", 1.0, "h1"),
         _row("XS_MOM", "SOLUSDT", "4h", 1.0, "h4")],
    )
    assert [s["account_strategy_id"] for s in out["to_sync"]] == ["h1"]
    assert [z["account_strategy_id"] for z in out["to_zero"]] == ["h4"]


def test_foreign_weighted_row_is_never_zeroed():
    # SAFETY (#1): a weighted, non-member row that the book did NOT write
    # (weight_source outside the HOUSE_BOOK family) must never be touched — so a
    # misconfigured book account pointed at a populated trading account can't
    # have its weights wiped.
    out = house_book.classify_book(
        [_member("XS_MOM", "SOLUSDT", "1h", 0.5)],
        [_row("LSR", "BTCUSDT", "1h", 0.30, "foreign", weight_source="EQUAL_WEIGHT"),
         _row("VBO", "ETHUSDT", "4h", 0.20, "foreign-hrp", weight_source="HRP")],
    )
    assert out["to_zero"] == []
