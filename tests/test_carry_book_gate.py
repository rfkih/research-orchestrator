"""Unit tests for the levered carry-book approval gate (PR #2).

Pure-function: no DB. Covers the threshold pin-tests (incl. the imported
economic floor), a synthetic PASS book, each FAIL boundary in isolation, the
require_basis_coverage hard-fail, and the ISOLATION contract (analyze.py's
frozen constants are untouched and the gate never calls analyze's gate fn).
"""
from __future__ import annotations

import ast
import inspect
import random
from datetime import date, timedelta
from pathlib import Path

import pytest

from orchestrator.services import analyze
from orchestrator.services import carry_book_gate as g

# ── helpers ──────────────────────────────────────────────────────────────────


def _book_curve(
    *,
    annual_return: float,
    daily_sd: float,
    n_days: int,
    seed: int,
    start_year: int = 2021,
) -> list[tuple[date, float]]:
    """Deterministic per-day equity curve: a positive drift sized to compound to
    ``annual_return`` over a 365-day year, plus seeded Gaussian noise. Seeded RNG
    → identical curve every run (no flakiness)."""
    mu = (1.0 + annual_return) ** (1.0 / 365.0) - 1.0
    rng = random.Random(seed)
    pts: list[tuple[date, float]] = []
    d = date(start_year, 1, 1)
    eq = 1.0
    for _ in range(n_days):
        eq *= 1.0 + mu + rng.gauss(0.0, daily_sd)
        pts.append((d, eq))
        d += timedelta(days=1)
    return pts


def _legs(symbols: list[str]) -> list[dict]:
    return [
        {"symbol": s, "return": 0.02, "basis_unmodeled": s not in g.BASIS_COVERED_SYMBOLS}
        for s in symbols
    ]


_PASS_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT"]


def _pass_book() -> dict:
    """A book that clears every clause: Sharpe ~3.6, maxDD ~-1.6%, +13%/yr, 5
    legs, leverage 2.0 (cap 3.0)."""
    pts = _book_curve(annual_return=0.12, daily_sd=0.0015, n_days=365 * 3, seed=7)
    return g.evaluate_book(
        equity_points=pts,
        n_legs=5,
        leverage=2.0,
        per_leg=_legs(_PASS_SYMBOLS),
        leverage_cap=3.0,
    )


# ── pin-tests on the threshold constants ─────────────────────────────────────


def test_threshold_constants_pinned():
    assert g.MIN_BOOK_SHARPE == 1.0
    assert g.MAX_BOOK_MAXDD_PCT == -15.0
    assert g.MIN_N_LEGS == 4
    assert g.DEFAULT_LEVERAGE_CAP == 3.0
    assert set(g.BASIS_COVERED_SYMBOLS) == {"BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT"}


def test_economic_floor_is_imported_not_redefined():
    # The book gate's economic floor MUST equal analyze's shared constant — so it
    # can never silently drift from the platform-wide bar (0.0 since the 2026-06-19
    # operator removal of the 10%/yr economic floor).
    assert g.ANNUALIZED_RETURN_PASS_THRESHOLD_PCT == analyze.ANNUALIZED_RETURN_PASS_THRESHOLD_PCT
    assert g.ANNUALIZED_RETURN_PASS_THRESHOLD_PCT == 0.0
    # And the verdict echoes that exact value (not a local copy).
    floor = analyze.ANNUALIZED_RETURN_PASS_THRESHOLD_PCT
    assert _pass_book()["thresholds"]["min_net_of_cost_return_pct"] == floor


# ── PASS ─────────────────────────────────────────────────────────────────────


def test_synthetic_book_passes():
    v = _pass_book()
    assert v["passed"] is True, v
    assert v["reason"] == "passed"
    assert v["book_sharpe"] >= g.MIN_BOOK_SHARPE
    assert v["book_maxdd_pct"] >= g.MAX_BOOK_MAXDD_PCT  # both negative
    assert v["n_legs"] == 5
    assert v["net_of_cost_return_pct"] >= 10.0
    assert v["leverage_used"] <= v["leverage_cap"]
    assert v["positive_every_year"] is True
    # Sanity: the calibrated synthetic lands near the carry book's real profile.
    assert 3.0 <= v["book_sharpe"] <= 4.5
    assert -3.0 <= v["book_maxdd_pct"] <= -0.5
    # per_leg carries the basis flag (SOLUSDT is outside coverage).
    by_sym = {leg["symbol"]: leg for leg in v["per_leg"]}
    assert by_sym["BTCUSDT"]["basis_unmodeled"] is False
    assert by_sym["SOLUSDT"]["basis_unmodeled"] is True


# ── FAIL boundaries — each threshold breached in isolation ───────────────────


def test_fail_low_sharpe():
    # High daily vol crushes Sharpe below 1.0 while keeping return/legs fine.
    pts = _book_curve(annual_return=0.12, daily_sd=0.05, n_days=365 * 3, seed=3)
    v = g.evaluate_book(
        equity_points=pts, n_legs=5, leverage=2.0, per_leg=_legs(_PASS_SYMBOLS),
        leverage_cap=3.0,
    )
    assert v["passed"] is False
    assert v["reason"] == "book_sharpe_below_min"
    assert v["book_sharpe"] < g.MIN_BOOK_SHARPE


def test_fail_deep_maxdd():
    # A curve that draws down >15% mid-window then recovers to a >10%/yr finish
    # with a positive yearly profile — so ONLY the maxDD clause fails.
    d = date(2021, 1, 1)
    pts: list[tuple[date, float]] = []
    eq = 1.0
    # year 1: small positive
    for _ in range(120):
        eq *= 1.0 + 0.0008
        pts.append((d, eq))
        d += timedelta(days=1)
    # sharp -20% drawdown over a few days
    for _ in range(10):
        eq *= 1.0 - 0.022
        pts.append((d, eq))
        d += timedelta(days=1)
    # recover + grind up so each calendar year still ends net-positive and the
    # overall CAGR clears 10%/yr.
    for _ in range(365 * 3):
        eq *= 1.0 + 0.0013
        pts.append((d, eq))
        d += timedelta(days=1)
    v = g.evaluate_book(
        equity_points=pts, n_legs=5, leverage=2.0, per_leg=_legs(_PASS_SYMBOLS),
        leverage_cap=3.0,
    )
    assert v["reason"] == "book_maxdd_too_deep", v
    assert v["passed"] is False
    assert v["book_maxdd_pct"] < g.MAX_BOOK_MAXDD_PCT


def test_fail_too_few_legs():
    pts = _book_curve(annual_return=0.12, daily_sd=0.0015, n_days=365 * 3, seed=7)
    v = g.evaluate_book(
        equity_points=pts, n_legs=3, leverage=2.0,
        per_leg=_legs(["BTCUSDT", "ETHUSDT", "BNBUSDT"]), leverage_cap=3.0,
    )
    assert v["passed"] is False
    assert v["reason"] == "too_few_legs"
    assert v["n_legs"] < g.MIN_N_LEGS


def test_modest_positive_return_passes_after_v60_floor_removed():
    # ~3%/yr — real Sharpe, shallow DD, positive every year. Before the 2026-06-19
    # operator decision this FAILED the 10%/yr economic floor; with the floor
    # removed (ANNUALIZED_RETURN_PASS_THRESHOLD_PCT=0.0) a positive, modest-return
    # book PASSES on its risk-adjusted merits. The return clause now only rejects
    # net-NEGATIVE books — which the Sharpe clause already catches first.
    pts = _book_curve(annual_return=0.03, daily_sd=0.0010, n_days=365 * 3, seed=11)
    v = g.evaluate_book(
        equity_points=pts, n_legs=5, leverage=2.0, per_leg=_legs(_PASS_SYMBOLS),
        leverage_cap=3.0,
    )
    assert v["passed"] is True, v
    assert v["reason"] == "passed"
    assert 0.0 <= v["net_of_cost_return_pct"] < 10.0


def test_fail_leverage_over_cap():
    pts = _book_curve(annual_return=0.12, daily_sd=0.0015, n_days=365 * 3, seed=7)
    v = g.evaluate_book(
        equity_points=pts, n_legs=5, leverage=5.0, per_leg=_legs(_PASS_SYMBOLS),
        leverage_cap=3.0,
    )
    assert v["passed"] is False
    assert v["reason"] == "leverage_over_cap"
    assert v["leverage_used"] > v["leverage_cap"]


def test_fail_negative_judged_year():
    # Year 1 strongly positive, year 2 net-negative (but enough days to be
    # judged) — a real Sharpe/return overall, yet NOT positive every year.
    d = date(2021, 1, 1)
    pts: list[tuple[date, float]] = []
    eq = 1.0
    for _ in range(365):           # 2021: strong positive
        eq *= 1.0 + 0.0018
        pts.append((d, eq))
        d += timedelta(days=1)
    # 2022: net-negative over 120 judged days, but a SHALLOW decline so the
    # book maxDD stays inside the -15% floor — only the year-guard clause fails.
    for _ in range(120):
        eq *= 1.0 - 0.0006
        pts.append((d, eq))
        d += timedelta(days=1)
    v = g.evaluate_book(
        equity_points=pts, n_legs=5, leverage=2.0, per_leg=_legs(_PASS_SYMBOLS),
        leverage_cap=3.0,
    )
    assert v["passed"] is False
    assert v["reason"] == "not_positive_every_year", v
    assert v["positive_every_year"] is False
    assert "2022" in v["year_guard"]["by_year_return"]
    assert v["year_guard"]["by_year_return"]["2022"] < 0


# ── require_basis_coverage ───────────────────────────────────────────────────


def test_require_basis_coverage_fails_on_unmodeled_leg():
    # The default PASS book includes SOLUSDT (basis_unmodeled). With the switch
    # ON it must FAIL on coverage even though every other clause passes.
    pts = _book_curve(annual_return=0.12, daily_sd=0.0015, n_days=365 * 3, seed=7)
    v = g.evaluate_book(
        equity_points=pts, n_legs=5, leverage=2.0, per_leg=_legs(_PASS_SYMBOLS),
        leverage_cap=3.0, require_basis_coverage=True,
    )
    assert v["passed"] is False
    assert v["reason"] == "basis_coverage_required"
    assert v["n_basis_unmodeled_legs"] == 1


def test_require_basis_coverage_passes_when_all_covered():
    # Same metrics but a 4-leg fully-covered book → coverage switch is satisfied.
    covered = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT"]
    pts = _book_curve(annual_return=0.12, daily_sd=0.0015, n_days=365 * 3, seed=7)
    v = g.evaluate_book(
        equity_points=pts, n_legs=4, leverage=2.0, per_leg=_legs(covered),
        leverage_cap=3.0, require_basis_coverage=True,
    )
    assert v["passed"] is True, v
    assert v["n_basis_unmodeled_legs"] == 0


# ── thin / degenerate input ──────────────────────────────────────────────────


def test_insufficient_equity_points():
    v = g.evaluate_book(
        equity_points=[(date(2021, 1, 1), 1.0)], n_legs=5, leverage=2.0,
        per_leg=_legs(_PASS_SYMBOLS), leverage_cap=3.0,
    )
    assert v["passed"] is False
    assert v["reason"] == "insufficient_equity_points"


# ── stateless preview lane ───────────────────────────────────────────────────


def test_preview_pass_from_leg_returns():
    # Five legs each with a small positive daily drift; equal-notional 2x book
    # over ~3 years clears the gate.
    rng = random.Random(5)
    n = 365 * 3
    legs = {
        s: [0.0002 + rng.gauss(0.0, 0.0007) for _ in range(n)]
        for s in _PASS_SYMBOLS
    }
    v = g.preview_from_leg_returns(leg_returns=legs, leverage=2.0)
    assert v["source"] == "preview"
    assert v["passed"] is True, v
    assert v["n_legs"] == 5
    assert v["leverage_used"] == 2.0


def test_preview_borrow_drag_can_flip_verdict():
    rng = random.Random(5)
    n = 365 * 3
    legs = {
        s: [0.0002 + rng.gauss(0.0, 0.0007) for _ in range(n)]
        for s in _PASS_SYMBOLS
    }
    # A large flat borrow drag eats the return below the 10%/yr floor.
    v = g.preview_from_leg_returns(
        leg_returns=legs, leverage=2.0, borrow_cost_bps_per_day=20.0
    )
    assert v["net_of_cost_return_pct"] < 10.0
    assert v["passed"] is False


def test_preview_require_basis_coverage():
    rng = random.Random(5)
    n = 365 * 3
    legs = {
        s: [0.0003 + rng.gauss(0.0, 0.0006) for _ in range(n)]
        for s in ["BTCUSDT", "ETHUSDT", "BNBUSDT", "DOGEUSDT"]
    }
    v = g.preview_from_leg_returns(
        leg_returns=legs, leverage=2.0, require_basis_coverage=True
    )
    assert v["passed"] is False
    assert v["reason"] == "basis_coverage_required"
    by_sym = {leg["symbol"]: leg for leg in v["per_leg"]}
    assert by_sym["DOGEUSDT"]["basis_unmodeled"] is True


# ── ISOLATION contract ───────────────────────────────────────────────────────


def test_analyze_frozen_constants_unchanged():
    # These analyze.py constants are the frozen single-asset gate the carry-book
    # lane must NOT perturb. Pinned here so a refactor that touches analyze
    # trips this test. (V11 statistical bars stay frozen; the V60 economic floor
    # is 0.0 since the 2026-06-19 operator removal of the 10%/yr bar.)
    assert analyze.MIN_TRADES_FOR_SIG == 100
    assert analyze.ANNUALIZED_RETURN_PASS_THRESHOLD_PCT == 0.0
    assert analyze.DSR_SIGNIFICANCE_THRESHOLD == 0.90
    assert analyze.DENOM_SQ_BOOTSTRAP_FLOOR == 0.05


def test_carry_book_gate_does_not_call_analyze_gate_functions():
    # Import-graph / no-call assertion: the ONLY thing carry_book_gate may pull
    # from analyze is the shared economic-floor CONSTANT. It must never call
    # analyze's gate functions (statistical_verdict / the DSR / PSR machinery),
    # which would couple the isolated book lane to the frozen single-asset gate.
    src = Path(inspect.getfile(g)).read_text(encoding="utf-8")
    tree = ast.parse(src)

    # 1. The only symbol imported FROM analyze is the economic-floor constant.
    analyze_imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.endswith("analyze"):
            analyze_imports.extend(alias.name for alias in node.names)
    assert analyze_imports == ["ANNUALIZED_RETURN_PASS_THRESHOLD_PCT"], analyze_imports

    # 2. No `analyze.<anything>` attribute access anywhere (no module-qualified
    #    call into analyze's gate functions).
    forbidden = {
        "statistical_verdict", "probabilistic_sharpe_ratio", "deflated_sharpe_ratio",
        "bootstrap_pf_ci", "analyze_run", "economic_gate",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            assert node.value.id != "analyze", f"forbidden analyze.{node.attr} access"
        if isinstance(node, ast.Name):
            assert node.id not in forbidden, f"forbidden gate-fn reference {node.id}"


@pytest.mark.parametrize(
    "name",
    ["MIN_BOOK_SHARPE", "MAX_BOOK_MAXDD_PCT", "MIN_N_LEGS", "DEFAULT_LEVERAGE_CAP"],
)
def test_named_module_constants_exist(name):
    # The thresholds must be named MODULE constants (operator-controlled), not
    # buried magic numbers.
    assert hasattr(g, name)


# ── DB-backed lane (fake asyncpg connection) ─────────────────────────────────


class _FakeConn:
    """Minimal asyncpg-connection stub: serves the run header (fetchrow), the
    equity points + trades (fetch) the carry_book_gate service reads. Dispatch
    is by a substring in the SQL — enough to exercise the real service code."""

    def __init__(self, *, run_row, equity_rows, trade_rows):
        self._run = run_row
        self._equity = equity_rows
        self._trades = trade_rows

    async def fetchrow(self, sql, *args):
        if "FROM backtest_run" in sql:
            return self._run
        return None

    async def fetch(self, sql, *args):
        if "FROM backtest_equity_point" in sql:
            return self._equity
        if "FROM backtest_trade" in sql:
            return self._trades
        return []


@pytest.mark.asyncio
async def test_evaluate_stored_run_passes():
    pts = _book_curve(annual_return=0.12, daily_sd=0.0015, n_days=365 * 3, seed=7)
    run_row = {
        "backtest_run_id": "r1", "strategy_code": "CARRY_BOOK", "status": "COMPLETED",
        "asset": "BOOK", "interval_name": "1d", "start_time": None, "end_time": None,
        "backtest_mode": "PORTFOLIO_CARRY",
        "book_n_legs": 5, "book_leverage": 2.0, "borrow_cost_bps_per_day": 1.0,
    }
    equity_rows = [{"equity_date": d, "total_equity": v} for d, v in pts]
    trade_rows = [
        {"asset": s, "realized_pnl_amount": 1.0, "entry_time": None, "exit_time": None}
        for s in _PASS_SYMBOLS
    ]
    conn = _FakeConn(run_row=run_row, equity_rows=equity_rows, trade_rows=trade_rows)
    from uuid import uuid4
    v = await g.evaluate(conn, backtest_run_id=uuid4())
    assert v["run_found"] is True
    assert v["backtest_mode"] == "PORTFOLIO_CARRY"
    assert v["passed"] is True, v
    assert v["n_legs"] == 5
    assert v["source"] == "stored_run"
    # per-leg attribution grouped by asset, basis flag on SOLUSDT.
    by_sym = {leg["symbol"]: leg for leg in v["per_leg"]}
    assert by_sym["SOLUSDT"]["basis_unmodeled"] is True
    assert by_sym["BTCUSDT"]["basis_unmodeled"] is False


@pytest.mark.asyncio
async def test_evaluate_run_not_found():
    conn = _FakeConn(run_row=None, equity_rows=[], trade_rows=[])
    from uuid import uuid4
    v = await g.evaluate(conn, backtest_run_id=uuid4())
    assert v["passed"] is False
    assert v["reason"] == "backtest_run_not_found"
    assert v["run_found"] is False


# ── API wiring (preview lane through the FastAPI router) ──────────────────────


def _preview_legs(symbols, seed=5):
    rng = random.Random(seed)
    n = 365 * 3
    return [
        {"symbol": s, "daily_returns": [0.0003 + rng.gauss(0.0, 0.0006) for _ in range(n)]}
        for s in symbols
    ]


def test_preview_endpoint_get_and_post(client):
    body = {"legs": _preview_legs(_PASS_SYMBOLS), "leverage": 2.0}
    headers = {"X-Orch-Token": "test-token"}
    r_get = client.request("GET", "/carry-book/preview", json=body, headers=headers)
    assert r_get.status_code == 200, r_get.text
    assert r_get.json()["passed"] is True
    r_post = client.post("/carry-book/preview", json=body, headers=headers)
    assert r_post.status_code == 200
    assert r_post.json()["passed"] is True


def test_preview_endpoint_requires_token(client):
    body = {"legs": _preview_legs(_PASS_SYMBOLS), "leverage": 2.0}
    r = client.post("/carry-book/preview", json=body)
    assert r.status_code == 401
