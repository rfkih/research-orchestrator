"""Unit tests for the combinatorial-block CPCV diagnostic (services/cpcv.py).

Pure-function coverage on synthetic ledgers — no DB, no JVM. Runs anywhere:

    PYTHONPATH=src pytest tests/test_cpcv.py -q
"""

from __future__ import annotations

from datetime import datetime, timedelta
from math import comb

import pytest

from orchestrator.services import cpcv


# --------------------------------------------------------------------------- #
# Synthetic ledger helpers
# --------------------------------------------------------------------------- #
def _trade(dt: datetime, pnl: float | None, regime: str = "BULL") -> dict:
    return {
        "realized_pnl_amount": pnl,
        "entry_time": dt,
        "entry_trend_regime": regime,
        "side": "LONG",
        "status": "CLOSED",
    }


def _ledger(n: int, win_every_lt: int, win: float, loss: float,
            days: int = 730, regimes=("BULL", "BEAR")) -> list[dict]:
    """``n`` trades evenly spread over ``days``. A trade is a winner when
    ``i % 5 < win_every_lt`` — so win_every_lt=3 ⇒ 60% win rate."""
    start = datetime(2024, 1, 1)
    out = []
    for i in range(n):
        dt = start + timedelta(days=i * days / n)
        pnl = win if (i % 5) < win_every_lt else loss
        out.append(_trade(dt, pnl, regimes[i % len(regimes)]))
    return out


STRONG = lambda n=120: _ledger(n, 3, 2.0, -1.0)   # 60% win, PF≈3.0
WEAK = lambda n=120: _ledger(n, 2, 1.0, -1.0)      # 40% win, PF≈0.667


# --------------------------------------------------------------------------- #
# closed_trades / assign_blocks
# --------------------------------------------------------------------------- #
def test_closed_trades_filters_and_sorts():
    t0 = datetime(2024, 6, 1)
    rows = [
        _trade(t0 + timedelta(days=2), 5.0),
        _trade(t0, 3.0),
        {"realized_pnl_amount": None, "entry_time": t0},        # open → dropped
        {"realized_pnl_amount": 1.0, "entry_time": "nope"},     # bad ts → dropped
        _trade(t0 + timedelta(days=1), -2.0),
    ]
    out = cpcv.closed_trades(rows)
    assert len(out) == 3
    assert [t["entry_time"] for t in out] == sorted(t["entry_time"] for t in out)


def test_assign_blocks_partitions_by_calendar():
    blocks = cpcv.assign_blocks(STRONG(120), n_blocks=8)
    assert len(blocks) == 8
    assert sum(len(b) for b in blocks) == 120
    # Evenly spread ⇒ each block holds roughly 120/8 = 15 trades.
    assert all(10 <= len(b) <= 20 for b in blocks)


def test_assign_blocks_rejects_under_two():
    with pytest.raises(ValueError):
        cpcv.assign_blocks(STRONG(50), n_blocks=1)


def test_assign_blocks_single_instant_falls_into_block_zero():
    t = datetime(2024, 1, 1)
    rows = [_trade(t, 1.0), _trade(t, -1.0)]
    blocks = cpcv.assign_blocks(rows, n_blocks=4)
    assert len(blocks[0]) == 2
    assert all(len(b) == 0 for b in blocks[1:])


# --------------------------------------------------------------------------- #
# combinatorial_paths / path_metrics
# --------------------------------------------------------------------------- #
def test_combinatorial_path_count_is_n_choose_k():
    blocks = cpcv.assign_blocks(STRONG(120), n_blocks=8)
    paths = cpcv.combinatorial_paths(blocks, k_test=4)
    assert len(paths) == comb(8, 4) == 70


def test_combinatorial_paths_rejects_bad_k():
    blocks = cpcv.assign_blocks(STRONG(120), n_blocks=8)
    with pytest.raises(ValueError):
        cpcv.combinatorial_paths(blocks, k_test=8)   # k must be < n_blocks
    with pytest.raises(ValueError):
        cpcv.combinatorial_paths(blocks, k_test=0)


def test_path_metrics_none_when_no_losers():
    assert cpcv.path_metrics([1.0, 2.0, 3.0]) is None       # PF infinite → None
    m = cpcv.path_metrics([2.0, 2.0, -1.0, -1.0])
    assert m is not None and m["pf"] == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# summarize_paths
# --------------------------------------------------------------------------- #
def test_summarize_paths_positivity():
    blocks = cpcv.assign_blocks(STRONG(120), n_blocks=8)
    summary = cpcv.summarize_paths(cpcv.combinatorial_paths(blocks, 4))
    assert summary["n_paths"] == 70
    assert summary["path_positive_pct"] == 100.0          # every path PF≈3
    assert summary["pf"]["median"] > 1.5


# --------------------------------------------------------------------------- #
# stratification
# --------------------------------------------------------------------------- #
def test_pf_by_year_and_regime():
    by_year = cpcv._pf_by_key(cpcv.closed_trades(STRONG(120)), cpcv._year_key)
    assert set(by_year) == {"2024", "2025"}
    assert all(st["pf"] is not None and st["pf"] > 1.0 for st in by_year.values())
    by_regime = cpcv._pf_by_key(cpcv.closed_trades(STRONG(120)), cpcv._regime_key)
    assert set(by_regime) == {"BULL", "BEAR"}


# --------------------------------------------------------------------------- #
# cpcv_advisory_verdict — the gate-adjacent logic, tested directly
# --------------------------------------------------------------------------- #
def _robust_year(): return {"2024": {"n": 60, "pf": 1.3}, "2025": {"n": 60, "pf": 1.4}}


def test_verdict_fragile_low_positivity():
    out = cpcv.cpcv_advisory_verdict(
        {"path_positive_pct": 50.0, "pf": {"median": 1.5}},
        {"low": 1.2}, _robust_year(),
    )
    assert out["verdict"] == "FRAGILE"


def test_verdict_fragile_median_below_one():
    out = cpcv.cpcv_advisory_verdict(
        {"path_positive_pct": 90.0, "pf": {"median": 0.9}},
        {"low": 1.1}, _robust_year(),
    )
    assert out["verdict"] == "FRAGILE"


def test_verdict_robust_all_legs():
    out = cpcv.cpcv_advisory_verdict(
        {"path_positive_pct": 90.0, "pf": {"median": 1.3}},
        {"low": 1.05}, _robust_year(),
    )
    assert out["verdict"] == "ROBUST_ADVISORY"


def test_verdict_mixed_when_ci_low_not_above_one():
    out = cpcv.cpcv_advisory_verdict(
        {"path_positive_pct": 90.0, "pf": {"median": 1.3}},
        {"low": 0.95}, _robust_year(),
    )
    assert out["verdict"] == "MIXED"


def test_verdict_mixed_on_nonstationary_year():
    by_year = {"2023": {"n": 40, "pf": 0.5}, "2024": {"n": 60, "pf": 1.4}}
    out = cpcv.cpcv_advisory_verdict(
        {"path_positive_pct": 90.0, "pf": {"median": 1.3}},
        {"low": 1.05}, by_year,
    )
    assert out["verdict"] == "MIXED"
    assert any("non-stationary" in r for r in out["reasons"])


def test_verdict_ignores_thin_year_stratum():
    by_year = {"2023": {"n": 5, "pf": 0.1}, "2024": {"n": 60, "pf": 1.4}}  # 2023 too thin
    out = cpcv.cpcv_advisory_verdict(
        {"path_positive_pct": 90.0, "pf": {"median": 1.3}},
        {"low": 1.05}, by_year,
    )
    assert out["verdict"] == "ROBUST_ADVISORY"


# --------------------------------------------------------------------------- #
# analyze_ledger — end to end on synthetic ledgers
# --------------------------------------------------------------------------- #
def test_analyze_ledger_insufficient_data():
    res = cpcv.analyze_ledger(STRONG(20))   # < CPCV_MIN_TRADES (40)
    assert res["cpcv_verdict"] == "INSUFFICIENT_DATA"
    assert "too" in res["note"]


def test_analyze_ledger_strong_is_robust():
    res = cpcv.analyze_ledger(STRONG(120))
    assert res["n_trades"] == 120
    assert res["paths"]["n_paths"] == comb(8, 4)
    assert len(res["block_trade_counts"]) == 8
    assert res["pooled"]["pf"] > 1.5
    assert set(res["stratify"]["by_year_pf"]) == {"2024", "2025"}
    assert res["cpcv_verdict"] == "ROBUST_ADVISORY"


def test_analyze_ledger_weak_is_fragile():
    res = cpcv.analyze_ledger(WEAK(120))
    assert res["pooled"]["pf"] < 1.0
    assert res["cpcv_verdict"] == "FRAGILE"


def test_analyze_ledger_clamps_k_test():
    # k_test >= n_blocks should be clamped, not raise.
    res = cpcv.analyze_ledger(STRONG(120), n_blocks=6, k_test=99)
    assert res["config"]["k_test"] == 5
    assert res["paths"]["n_paths"] == comb(6, 5)
