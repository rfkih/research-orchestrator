"""Unit tests for the research→registry auto-sync (2026-06-16).

DB-free: `derive_status` is pure; `sync_from_research` is driven by a fake
connection that scripts the find-then-mutate sequence. The real SQL path is
covered by tests/integration/test_strategy_registry_db.py.
"""
from __future__ import annotations

from typing import Any

import pytest

from orchestrator.services.registry_sync import derive_status, sync_from_research


# ── derive_status (pure) ─────────────────────────────────────────────────────

def test_derive_robust_walk_forward_is_tier_a_real_lead() -> None:
    assert derive_status("INSUFFICIENT_EVIDENCE", "ROBUST") == ("TIER_A", "REAL_LEAD", "LEAD")


def test_derive_overfit_and_no_edge_wf_are_falsified() -> None:
    assert derive_status("SIGNIFICANT_EDGE", "OVERFIT") == ("TIER_C", "FALSIFIED", "FALSIFIED")
    assert derive_status(None, "NO_EDGE") == ("TIER_C", "FALSIFIED", "FALSIFIED")


def test_derive_inconsistent_wf_is_parked() -> None:
    assert derive_status("SIGNIFICANT_EDGE", "INCONSISTENT") == ("TIER_B", "REAL_UNCERTIFIABLE", "PARKED")


def test_derive_significant_edge_no_wf_is_lead() -> None:
    assert derive_status("SIGNIFICANT_EDGE") == ("TIER_B", "REAL_UNCERTIFIABLE", "LEAD")


def test_derive_insufficient_is_parked() -> None:
    assert derive_status("INSUFFICIENT_EVIDENCE") == ("TIER_B", "REAL_UNCERTIFIABLE", "PARKED")


def test_derive_no_edge_is_falsified() -> None:
    assert derive_status("NO_EDGE") == ("TIER_C", "FALSIFIED", "FALSIFIED")


def test_derive_not_tested_and_unknown_are_none() -> None:
    assert derive_status("NOT_TESTED") is None
    assert derive_status(None) is None
    assert derive_status("") is None


def test_derive_inconclusive_wf_falls_back_to_iteration_verdict() -> None:
    # WF INSUFFICIENT_EVIDENCE is not decisive → use the iteration verdict.
    assert derive_status("SIGNIFICANT_EDGE", "INSUFFICIENT_EVIDENCE") == ("TIER_B", "REAL_UNCERTIFIABLE", "LEAD")


# ── sync_from_research (fake conn) ───────────────────────────────────────────

class _Conn:
    """Scripts the calls registry_sync makes: find_by_research_key (fetchrow #1),
    then insert/update (fetchval) + get_registry (fetchrow #2)."""

    def __init__(self, *, find_row: dict[str, Any] | None, after_row: dict[str, Any] | None = None,
                 rid: Any = "rid") -> None:
        self._find_row = find_row
        self._after_row = after_row or {"registry_id": "rid", "slug": "s"}
        self._rid = rid
        self._fetchrow_n = 0
        self.fetchval_params: tuple[Any, ...] | None = None

    async def fetchrow(self, sql: str, *params: Any) -> dict[str, Any] | None:
        self._fetchrow_n += 1
        return self._find_row if self._fetchrow_n == 1 else self._after_row

    async def fetchval(self, sql: str, *params: Any) -> Any:
        self.fetchval_params = params
        return self._rid


@pytest.mark.asyncio
async def test_sync_creates_when_absent() -> None:
    conn = _Conn(find_row=None)
    out = await sync_from_research(
        conn, strategy_code="NEW_X", symbol="BTCUSDT", interval="1h",
        statistical_verdict="SIGNIFICANT_EDGE",
    )
    assert out == "created"
    # insert_registry sets auto_managed (19th positional param) = True for the loop.
    assert conn.fetchval_params is not None
    assert conn.fetchval_params[18] is True


@pytest.mark.asyncio
async def test_sync_updates_agent_owned_row() -> None:
    conn = _Conn(find_row={"registry_id": "r1", "auto_managed": True, "archived": False, "thesis": "t"})
    out = await sync_from_research(
        conn, strategy_code="X", symbol="BTCUSDT", interval="1h",
        statistical_verdict="NO_EDGE",
    )
    assert out == "updated"


@pytest.mark.asyncio
async def test_sync_skips_curated_row() -> None:
    # auto_managed=False → human/seed owns it → never clobber.
    conn = _Conn(find_row={"registry_id": "r1", "auto_managed": False, "archived": False, "thesis": "t"})
    out = await sync_from_research(
        conn, strategy_code="VRP_BTC", symbol="BTCUSDT", interval="1d",
        statistical_verdict="NO_EDGE",
    )
    assert out == "skipped"
    assert conn.fetchval_params is None  # no mutation attempted


@pytest.mark.asyncio
async def test_sync_noop_on_inconclusive_verdict() -> None:
    conn = _Conn(find_row=None)
    out = await sync_from_research(
        conn, strategy_code="X", symbol="BTCUSDT", interval="1h",
        statistical_verdict="NOT_TESTED",
    )
    assert out == "no-op"
    assert conn.fetchval_params is None


@pytest.mark.asyncio
async def test_sync_noop_on_missing_strategy_code() -> None:
    conn = _Conn(find_row=None)
    out = await sync_from_research(
        conn, strategy_code="", symbol="BTCUSDT", interval="1h",
        statistical_verdict="SIGNIFICANT_EDGE",
    )
    assert out == "no-op"
