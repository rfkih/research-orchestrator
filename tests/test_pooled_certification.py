"""Unit tests for the pooled-certification pathway (2026-07-10).

Focus: the pure book-level math (slice returns, combined equity, contention),
the Hamming-1 params_snapshot contract the graduation reviewer depends on,
and the V11 pin — the pooled module must IMPORT every verdict primitive from
analyze/walk_forward, never redefine one.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from typing import Any
from uuid import uuid4

import pytest

from orchestrator.errors import OrchestratorError
from orchestrator.services import analyze
from orchestrator.services import pooled_certification as pc
from orchestrator.services import walk_forward as wf


def _run(coro):
    return asyncio.run(coro)


def _sleeve(instrument="SOLUSDT", interval="4h", start="2021-01-01", **ov):
    return pc.SleeveSpec(
        strategy_code="DCB",
        instrument=instrument,
        interval_name=interval,
        window_start=date.fromisoformat(start),
        overrides={"tpR": "2.0", "stopAtrMult": "3.25", **ov},
    )


# -- V11 pin: no verdict primitive is redefined in the pooled module ---------

def test_pooled_module_reuses_walk_forward_stability_verdict_object():
    assert pc.stability_verdict is wf.stability_verdict
    assert pc.aggregate_folds is wf.aggregate_folds
    assert pc.build_folds is wf.build_folds
    assert pc.window_covers_bear is wf.window_covers_bear


def test_pooled_module_defines_no_own_thresholds():
    forbidden = (
        "DSR_SIGNIFICANCE_THRESHOLD",
        "MIN_TRADES_FOR_SIG",
        "PF_CI_LOWER_BOUND_PASS",
        "ANNUALIZED_RETURN_PASS_THRESHOLD_PCT",
    )
    import inspect

    src = inspect.getsource(pc)
    for name in forbidden:
        assert f"{name} =" not in src, f"pooled module redefines {name}"


# -- params_snapshot flattening: Hamming-1 across book variants --------------

def test_flatten_sleeve_params_hamming_one_between_neighbor_books():
    book_a = [_sleeve(), _sleeve(instrument="ETHUSDT")]
    book_b = [_sleeve(), _sleeve(instrument="ETHUSDT", tpR="3.0")]
    fa = pc.flatten_sleeve_params(book_a)
    fb = pc.flatten_sleeve_params(book_b)
    assert fa.keys() == fb.keys()
    diffs = [k for k in fa if str(fa[k]) != str(fb[k])]
    assert diffs == ["s1.tpR"]
    # All values scalar (str) — nested dicts would break the Hamming metric.
    assert all(not isinstance(v, (dict, list)) for v in fa.values())


# -- slice-model book return --------------------------------------------------

def test_slice_book_return_is_mean_of_multipliers():
    # Sleeves at +100%, +50%, 0% -> multipliers 2.0, 1.5, 1.0 -> book 1.5.
    assert pc.slice_book_return_pct([100.0, 50.0, 0.0]) == pytest.approx(50.0)


def test_slice_book_return_none_propagates():
    assert pc.slice_book_return_pct([100.0, None]) is None
    assert pc.slice_book_return_pct([]) is None


def test_slice_book_return_late_start_sleeve_is_idle_multiplier_one():
    # A late-start sleeve reports its own run-window return; the idle
    # prefix is 0% by construction. Idle slice == multiplier 1.0.
    with_idle = pc.slice_book_return_pct([80.0, 0.0])
    assert with_idle == pytest.approx(40.0)


# -- combined slice equity + book drawdown ------------------------------------

def test_combine_slice_equity_idle_slice_holds_flat():
    d1 = date(2021, 1, 1)
    d2 = date(2021, 1, 2)
    d3 = date(2021, 1, 3)
    # Sleeve A: +10% then -10%; sleeve B idle until d3 then +20%.
    a = [(d1, 10.0), (d2, -10.0)]
    b = [(d3, 20.0)]
    curve, max_dd = pc.combine_slice_equity([a, b])
    assert [d for d, _ in curve] == [d1, d2, d3]
    # d1: mean(1.1, 1.0) = 1.05
    assert curve[0][1] == pytest.approx(1.05)
    # d2: mean(0.99, 1.0) = 0.995
    assert curve[1][1] == pytest.approx(0.995)
    # d3: mean(0.99, 1.2) = 1.095
    assert curve[2][1] == pytest.approx(1.095)
    # dd from 1.05 peak to 0.995 = 5.238%
    assert max_dd == pytest.approx(5.2381, abs=1e-3)


def test_combine_slice_equity_empty_returns_none():
    assert pc.combine_slice_equity([]) == ([], None)
    assert pc.combine_slice_equity([[], []]) == ([], None)


# -- contention diagnostics ----------------------------------------------------

def _trade(entry, exit_, inst, pnl=1.0):
    return {
        "entry_time": datetime.fromisoformat(entry),
        "exit_time": datetime.fromisoformat(exit_),
        "_sleeve_instrument": inst,
        "realized_pnl_amount": pnl,
    }


def test_contention_counts_overlap_and_same_symbol():
    trades = [
        _trade("2021-01-01T00:00", "2021-01-01T10:00", "SOLUSDT"),
        _trade("2021-01-01T05:00", "2021-01-01T15:00", "ETHUSDT"),
        _trade("2021-01-01T06:00", "2021-01-01T08:00", "ETHUSDT"),
    ]
    diag = pc.contention_diagnostics(trades)
    assert diag["computable"] is True
    assert diag["max_concurrent_positions"] == 3
    # Open span 00:00-15:00 = 15h. >=2 concurrent: 05:00-10:00 = 5h.
    assert diag["open_time_pct_ge2_concurrent"] == pytest.approx(33.33, abs=0.1)
    # >=3 concurrent: 06:00-08:00 = 2h.
    assert diag["open_time_pct_ge3_concurrent"] == pytest.approx(13.33, abs=0.1)
    # Same-symbol (ETH x2): 06:00-08:00 = 2h.
    assert diag["same_symbol_overlap_pct"] == pytest.approx(13.33, abs=0.1)


def test_contention_no_datetimes_not_computable():
    assert pc.contention_diagnostics([{"entry_time": None}]) == {
        "computable": False
    }


# -- pooled scoring parity ------------------------------------------------------

def test_pooled_scoring_uses_analyze_run_unchanged():
    # Two sleeves' pnl series merged must produce exactly the PF that
    # analyze.profit_factor gives on the concatenation — the pooled path
    # scores with analyze_run, so this parity is the replication guard.
    a = [10.0, -5.0, 8.0]
    b = [-2.0, 6.0]
    merged = a + b
    pf = analyze.profit_factor(merged)
    assert pf == pytest.approx((10 + 8 + 6) / (5 + 2))


def test_pool_trades_annotates_and_sorts():
    s1 = _sleeve()
    s2 = _sleeve(instrument="ETHUSDT")
    t1 = {"entry_time": datetime(2021, 1, 2), "realized_pnl_amount": 1}
    t2 = {"entry_time": datetime(2021, 1, 1), "realized_pnl_amount": 2}
    pooled = pc._pool_trades([[t1], [t2]], [s1, s2])
    assert [t["entry_time"].day for t in pooled] == [1, 2]
    assert pooled[0]["_sleeve_instrument"] == "ETHUSDT"
    assert pooled[1]["_sleeve_instrument"] == "SOLUSDT"


# -- async gates (light fakes — no DB) ------------------------------------------

class _FakeConn:
    def __init__(self, row: Any = None) -> None:
        self._row = row

    async def fetchrow(self, sql: str, *params: Any) -> Any:  # noqa: ARG002
        return self._row

    async def fetch(self, sql: str, *params: Any) -> Any:  # noqa: ARG002
        return []

    async def fetchval(self, sql: str, *params: Any) -> Any:  # noqa: ARG002
        return 0


class _FakeAcquire:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    async def __aenter__(self) -> _FakeConn:
        return self._conn

    async def __aexit__(self, *exc: Any) -> None:
        return None


class _FakeDb:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    def acquire(self) -> _FakeAcquire:
        return _FakeAcquire(self._conn)


def test_pooled_wf_rejects_non_pooled_iteration():
    row = {
        "iteration_id": uuid4(),
        "strategy_code": "DCB_POOL",
        "metrics_snapshot": {"analysis": {}},  # no pooled_certification key
    }
    db = _FakeDb(_FakeConn(row=row))
    with pytest.raises(OrchestratorError) as ei:
        _run(
            pc.run_pooled_walk_forward(
                db=db,  # type: ignore[arg-type]
                jvm=None,  # type: ignore[arg-type]
                settings=None,  # type: ignore[arg-type]
                agent_name="test",
                iteration_id=row["iteration_id"],
            )
        )
    assert ei.value.envelope.error_code == "pooled_iteration_invalid"


def test_pooled_wf_rejects_overlapping_folds_before_db():
    with pytest.raises(OrchestratorError) as ei:
        _run(
            pc.run_pooled_walk_forward(
                db=None,  # type: ignore[arg-type]
                jvm=None,  # type: ignore[arg-type]
                settings=None,  # type: ignore[arg-type]
                agent_name="test",
                iteration_id=uuid4(),
                test_months=6,
                step_months=3,
            )
        )
    assert ei.value.envelope.error_code == "walk_forward_overlapping_folds"


def test_pooled_wf_rejects_bull_only_window():
    row = {
        "iteration_id": uuid4(),
        "strategy_code": "DCB_POOL",
        "metrics_snapshot": {
            "pooled_certification": {
                "sleeves": [_sleeve(start="2023-01-01").to_snapshot()],
                "full_start": "2023-01-01",
                "full_end": "2024-01-01",
            }
        },
    }
    db = _FakeDb(_FakeConn(row=row))
    with pytest.raises(OrchestratorError) as ei:
        _run(
            pc.run_pooled_walk_forward(
                db=db,  # type: ignore[arg-type]
                jvm=None,  # type: ignore[arg-type]
                settings=None,  # type: ignore[arg-type]
                agent_name="test",
                iteration_id=row["iteration_id"],
            )
        )
    assert ei.value.envelope.error_code == "validation_window_excludes_bear"


def test_pooled_analyze_rejects_sleeve_counts():
    with pytest.raises(OrchestratorError) as ei:
        _run(
            pc.run_pooled_analyze(
                db=None,  # type: ignore[arg-type]
                jvm=None,  # type: ignore[arg-type]
                settings=None,  # type: ignore[arg-type]
                agent_name="test",
                book_strategy_code="DCB_POOL",
                sleeves=[_sleeve()],
            )
        )
    assert ei.value.envelope.error_code == "pooled_sleeve_count_invalid"
