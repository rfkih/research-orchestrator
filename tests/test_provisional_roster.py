"""Pin tests for the provisional trading-roster floor (2026-06-09).

Locks the min-roster constant, the eligibility rules (real edge + positive
return + trading-track), and the balanced composite ranking, so the floor can't
silently start promoting noise, money-losers, or hedges into the live book.
"""

from __future__ import annotations

import math
from typing import Any

from fastapi.testclient import TestClient

from orchestrator.api.deps import get_db_conn
from orchestrator.services.provisional_roster import (
    ELIGIBLE_STAT_VERDICT,
    MIN_ACTIVE_TRADING_STRATEGIES,
    composite_score,
    is_eligible,
    overfit_quality,
    rank_provisional_candidates,
    roster_gap,
    select_roster_fill,
)


def _auth_headers() -> dict[str, str]:
    return {"X-Orch-Token": "test-token", "X-Agent-Name": "quant-curator"}


class _FetchConn:
    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self._rows = rows or []

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        return self._rows


def _cand(**kw):
    base = {
        "strategy_code": "S",
        "track": "trading",
        "statistical_verdict": "SIGNIFICANT_EDGE",
        "ann_return_pct": 20.0,
        "dsr": 0.9,
    }
    base.update(kw)
    return base


# ── constants ─────────────────────────────────────────────────────────


def test_constants_are_pinned() -> None:
    assert MIN_ACTIVE_TRADING_STRATEGIES == 5
    assert ELIGIBLE_STAT_VERDICT == "SIGNIFICANT_EDGE"


# ── overfit_quality ───────────────────────────────────────────────────


def test_quality_dsr_only_returns_dsr() -> None:
    assert overfit_quality(0.8) == 0.8


def test_quality_geomean_of_evidence() -> None:
    # dsr 0.81 and fold 49% → sqrt(0.81 * 0.49) = 0.63.
    q = overfit_quality(0.81, fold_positive_pct=49.0)
    assert abs(q - math.sqrt(0.81 * 0.49)) < 1e-9
    assert abs(q - 0.63) < 1e-9


def test_quality_cliff_penalised() -> None:
    plateau = overfit_quality(0.9, param_robust=True)
    cliff = overfit_quality(0.9, param_robust=False)
    assert cliff < plateau


def test_quality_no_evidence_is_zero() -> None:
    assert overfit_quality(None) == 0.0


# ── composite_score ───────────────────────────────────────────────────


def test_composite_money_loser_scores_zero() -> None:
    assert composite_score(-5.0, 0.9) == 0.0
    assert composite_score(0.0, 0.9) == 0.0


def test_composite_is_return_times_quality() -> None:
    assert composite_score(20.0, 0.5) == 10.0


# ── eligibility ───────────────────────────────────────────────────────


def test_eligible_trading_sig_edge_positive() -> None:
    assert is_eligible(_cand()) is True


def test_legacy_untagged_track_treated_as_trading() -> None:
    assert is_eligible(_cand(track=None)) is True


def test_hedging_track_ineligible() -> None:
    assert is_eligible(_cand(track="hedging")) is False


def test_non_significant_edge_ineligible() -> None:
    assert is_eligible(_cand(statistical_verdict="INSUFFICIENT_EVIDENCE")) is False


def test_money_loser_ineligible() -> None:
    assert is_eligible(_cand(ann_return_pct=-3.0)) is False


def test_overfit_walk_forward_ineligible() -> None:
    assert is_eligible(_cand(walk_forward_verdict="OVERFIT")) is False


# ── ranking ───────────────────────────────────────────────────────────


def test_rank_drops_ineligible_and_sorts_by_composite() -> None:
    cands = [
        _cand(strategy_code="LOW", ann_return_pct=12.0, dsr=0.96),   # 11.52
        _cand(strategy_code="HIGH", ann_return_pct=30.0, dsr=0.97),  # 29.1
        _cand(strategy_code="NOISE", statistical_verdict="NO_EDGE"),  # dropped
        _cand(strategy_code="LOSER", ann_return_pct=-1.0),           # dropped
    ]
    ranked = rank_provisional_candidates(cands)
    assert [r["strategy_code"] for r in ranked] == ["HIGH", "LOW"]
    assert ranked[0]["provisional_score"] > ranked[1]["provisional_score"]


# ── floor selection ───────────────────────────────────────────────────


def test_roster_gap() -> None:
    assert roster_gap(2) == 3
    assert roster_gap(5) == 0
    assert roster_gap(7) == 0


def test_select_fills_gap_only() -> None:
    cands = [_cand(strategy_code=f"S{i}", ann_return_pct=float(30 - i)) for i in range(8)]
    picks = select_roster_fill(2, cands)  # gap = 3
    assert len(picks) == 3
    assert all(p["provisional"] is True for p in picks)
    # highest return first
    assert picks[0]["strategy_code"] == "S0"


def test_select_dedupes_by_strategy_code() -> None:
    # Same strategy, three cells — only the best one takes a slot.
    cands = [
        _cand(strategy_code="DUP", ann_return_pct=30.0),
        _cand(strategy_code="DUP", ann_return_pct=25.0),
        _cand(strategy_code="DUP", ann_return_pct=20.0),
        _cand(strategy_code="OTHER", ann_return_pct=10.0),
    ]
    picks = select_roster_fill(3, cands)  # gap = 2
    codes = [p["strategy_code"] for p in picks]
    assert codes == ["DUP", "OTHER"]


def test_select_excludes_already_active() -> None:
    cands = [_cand(strategy_code="ACTIVE", ann_return_pct=99.0), _cand(strategy_code="NEW")]
    picks = select_roster_fill(4, cands, exclude_strategy_codes=frozenset({"ACTIVE"}))
    assert [p["strategy_code"] for p in picks] == ["NEW"]


def test_select_noop_when_roster_full() -> None:
    assert select_roster_fill(5, [_cand()]) == []
    assert select_roster_fill(6, [_cand()]) == []


# ── HTTP endpoint ─────────────────────────────────────────────────────


def _override_db(client: TestClient, rows: list[dict[str, Any]]) -> None:
    async def _override():
        yield _FetchConn(rows)

    client.app.dependency_overrides[get_db_conn] = _override


def test_endpoint_returns_fills_when_below_floor(client: TestClient) -> None:
    rows = [
        _cand(strategy_code="A", ann_return_pct=40.0, dsr=0.97),
        _cand(strategy_code="B", ann_return_pct=25.0, dsr=0.96),
    ]
    _override_db(client, rows)
    try:
        resp = client.get(
            "/provisional-roster", params={"active_count": 3}, headers=_auth_headers()
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["floor"] == 5
        assert data["gap"] == 2
        codes = [f["strategy_code"] for f in data["fills"]]
        assert codes == ["A", "B"]  # highest composite first
        assert all(f["provisional"] is True for f in data["fills"])
    finally:
        client.app.dependency_overrides.pop(get_db_conn, None)


def test_endpoint_noop_when_roster_full(client: TestClient) -> None:
    _override_db(client, [_cand()])
    try:
        resp = client.get(
            "/provisional-roster", params={"active_count": 5}, headers=_auth_headers()
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["gap"] == 0
        assert data["fills"] == []
    finally:
        client.app.dependency_overrides.pop(get_db_conn, None)


def test_endpoint_excludes_active_codes(client: TestClient) -> None:
    rows = [
        _cand(strategy_code="LIVE", ann_return_pct=99.0),
        _cand(strategy_code="FRESH", ann_return_pct=10.0),
    ]
    _override_db(client, rows)
    try:
        resp = client.get(
            "/provisional-roster",
            params={"active_count": 4, "exclude": ["LIVE"]},
            headers=_auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert [f["strategy_code"] for f in data["fills"]] == ["FRESH"]
    finally:
        client.app.dependency_overrides.pop(get_db_conn, None)
