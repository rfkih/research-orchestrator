"""Levered carry-book approval gate — BOOK-level certification (PR #2).

Companion to PR #1 (trading-engine V188 ``PORTFOLIO_CARRY`` mode +
``PortfolioCarryBacktestService``), which produces a standard ``backtest_run``
carrying ``book_n_legs`` / ``book_leverage`` / ``borrow_cost_bps_per_day``, a
``backtest_equity_point`` book equity curve, and ``backtest_trade`` rows whose
owner codes are restamped to the run code but whose ``asset`` (per-leg symbol)
is preserved.

A delta-neutral multi-leg carry book CANNOT be graded by the single-asset
trade-count V11/DSR gate — it is one portfolio, not 100+ independent trades on
one surface, and its edge lives in the BOOK equity curve, not in any single
leg's PnL. This module is a NEW, ISOLATED approval lane that certifies the book
on book-level metrics (Sharpe / maxDD / net CAGR / leg-count / leverage /
positive-every-year) derived from the recorded equity curve.

ISOLATION CONTRACT
==================
This module REUSES the *math* already proven elsewhere but does NOT touch the
frozen single-asset gate:

  * Book Sharpe / maxDD / net CAGR come from
    ``hedging_gate._strat_metrics_from_equity`` (a real per-day ``total_equity``
    series → annualised Sharpe, true peak-to-trough maxDD, calendar-span CAGR).
  * The positive-every-judged-year stationarity guard reuses
    ``pool.stream_sharpe_validity``'s per-year logic (via the small helper
    ``_positive_every_judged_year`` below, which mirrors its calendar-year
    bucketing + thin-boundary-year exclusion).
  * The economic floor is IMPORTED verbatim from
    ``analyze.ANNUALIZED_RETURN_PASS_THRESHOLD_PCT`` (0.0 since the 2026-06-19
    operator removal of the 10%/yr floor) — NOT redefined here, so it can never
    silently drift from the platform-wide bar.

It deliberately does NOT import or call ``analyze``'s gate functions
(``statistical_verdict``, the DSR check, etc.) — the book gate is a parallel
lane, and analyze.py's frozen constants stay untouched. The only symbol pulled
from analyze is the shared economic-floor constant.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any
from uuid import UUID

import asyncpg

from ..repo import backtest_equity

# The economic floor is the SHARED platform-wide bar — import it, never redefine.
from .analyze import ANNUALIZED_RETURN_PASS_THRESHOLD_PCT
from .hedging_gate import _strat_metrics_from_equity
from .pool import STREAM_MIN_YEAR_DAYS

# ── Pinned book-gate constants (operator-tunable; pin-tested) ───────────────
# These are the BOOK-level bars a levered carry portfolio must clear. They are
# distinct from the single-asset V11/V60 gate and do NOT touch it.
MIN_BOOK_SHARPE: float = 1.0       # book Sharpe must clear this (delta-neutral carry ≫ 1)
MAX_BOOK_MAXDD_PCT: float = -15.0  # max tolerable book drawdown (NEGATIVE; shallower=larger)
MIN_N_LEGS: int = 4                # a "book" needs real breadth, not 1-2 legs
DEFAULT_LEVERAGE_CAP: float = 3.0  # default ceiling on book_leverage (per-call overridable)

# The basis (perp-vs-spot) is only MODELLED for these symbols — perp_basis data
# exists ONLY for these four. A leg outside this set has its carry computed
# without a modelled basis leg, so its economics are approximate; we surface
# that as ``basis_unmodeled=true`` per leg and (optionally) hard-fail on it.
BASIS_COVERED_SYMBOLS: frozenset[str] = frozenset(
    {"BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT"}
)


def _positive_every_judged_year(
    daily_returns: dict[date, float], *, min_year_days: int = STREAM_MIN_YEAR_DAYS
) -> dict[str, Any]:
    """Positive cumulative return in EVERY calendar year with enough data.

    Mirrors the stationarity guard inside ``pool.stream_sharpe_validity``: bucket
    daily returns by calendar year, JUDGE only years with >= ``min_year_days``
    observations (a thin first/last boundary year is statistically meaningless
    for 'positive every year' and must not gate), and require every judged year
    to be net-positive. This is the 'real stationary premium vs regime-dependent'
    hallmark — the same guard the carry stream-admission lane uses."""
    by_year: dict[int, float] = {}
    by_year_days: dict[int, int] = {}
    for d, r in daily_returns.items():
        by_year[d.year] = by_year.get(d.year, 0.0) + r
        by_year_days[d.year] = by_year_days.get(d.year, 0) + 1
    judged = {y: v for y, v in by_year.items() if by_year_days[y] >= min_year_days}
    ok = bool(judged) and all(v > 0 for v in judged.values())
    return {
        "positive_every_year": ok,
        "judged_years": sorted(judged),
        "by_year_return": {str(y): round(v, 6) for y, v in sorted(by_year.items())},
    }


def _per_leg_from_trades(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group ``backtest_trade`` rows by ``asset`` → per-leg PnL contribution.

    PR #1 restamps each trade's owner ``strategy_code`` to the run code, but the
    per-leg ``asset`` (symbol) is preserved — so the symbol is the correct
    grouping key for leg attribution. Each leg's ``return`` is the summed
    ``realized_pnl_amount`` across its trades (raw book-currency contribution).
    ``basis_unmodeled`` flags legs outside the perp_basis coverage set."""
    by_symbol: dict[str, float] = {}
    n_trades_by_symbol: dict[str, int] = {}
    for t in trades:
        sym = t.get("asset") or t.get("symbol")
        if not sym:
            continue
        try:
            pnl = float(t.get("realized_pnl_amount") or 0.0)
        except (TypeError, ValueError):
            pnl = 0.0
        by_symbol[sym] = by_symbol.get(sym, 0.0) + pnl
        n_trades_by_symbol[sym] = n_trades_by_symbol.get(sym, 0) + 1
    out: list[dict[str, Any]] = []
    for sym in sorted(by_symbol):
        out.append({
            "symbol": sym,
            "return": round(by_symbol[sym], 6),
            "n_trades": n_trades_by_symbol[sym],
            "basis_unmodeled": sym not in BASIS_COVERED_SYMBOLS,
        })
    return out


def _thresholds(*, leverage_cap: float, require_basis_coverage: bool) -> dict[str, Any]:
    """The named, pin-tested threshold set echoed into every verdict for audit.

    ``min_net_of_cost_return_pct`` is the IMPORTED economic floor (never a local
    copy), so the verdict self-documents the exact bar it was judged against."""
    return {
        "min_book_sharpe": MIN_BOOK_SHARPE,
        "max_book_maxdd_pct": MAX_BOOK_MAXDD_PCT,
        "min_n_legs": MIN_N_LEGS,
        "min_net_of_cost_return_pct": ANNUALIZED_RETURN_PASS_THRESHOLD_PCT,
        "leverage_cap": leverage_cap,
        "require_basis_coverage": require_basis_coverage,
    }


def evaluate_book(
    *,
    equity_points: list[tuple[date, float]],
    n_legs: int,
    leverage: float,
    per_leg: list[dict[str, Any]],
    leverage_cap: float = DEFAULT_LEVERAGE_CAP,
    require_basis_coverage: bool = False,
) -> dict[str, Any]:
    """Pure PASS/FAIL on a completed carry book — the core gate logic.

    Given the recorded per-day book equity curve, the book's leg count +
    leverage, and the per-leg attribution, compute the book metrics and apply
    the gate. PURE (no I/O) so it is reusable by both the stored-run lane (DB
    fetch → here) and the stateless ``preview`` what-if lane.

    PASS rule — ALL must hold:
      * ``book_sharpe >= MIN_BOOK_SHARPE``
      * ``book_maxdd_pct >= MAX_BOOK_MAXDD_PCT`` (both NEGATIVE; shallower passes)
      * ``n_legs >= MIN_N_LEGS``
      * ``net_of_cost_return_pct >= ANNUALIZED_RETURN_PASS_THRESHOLD_PCT``
      * ``leverage <= leverage_cap``
      * positive cumulative return in every judged calendar year
      * (when ``require_basis_coverage``) no included leg is basis-unmodeled

    Returns the structured verdict dict (see module docstring / API contract)."""
    thresholds = _thresholds(
        leverage_cap=leverage_cap, require_basis_coverage=require_basis_coverage
    )

    if len(equity_points) < 2:
        return {
            "passed": False,
            "book_sharpe": None,
            "book_maxdd_pct": None,
            "n_legs": n_legs,
            "net_of_cost_return_pct": None,
            "leverage_used": leverage,
            "leverage_cap": leverage_cap,
            "per_leg": per_leg,
            "reason": "insufficient_equity_points",
            "thresholds": thresholds,
        }

    eq_dates = [(d.date() if isinstance(d, datetime) else d) for d, _ in equity_points]
    eq_values = [v for _, v in equity_points]
    metrics = _strat_metrics_from_equity(eq_values, eq_dates)
    book_sharpe = metrics["sharpe"]
    book_maxdd_pct = metrics["max_drawdown_pct"]
    net_return_pct = metrics["cagr_pct"]
    # _strat_metrics_from_equity returns maxDD as a POSITIVE magnitude; the gate
    # and the operator-facing knob express drawdown as a NEGATIVE number
    # (−1.9% = a 1.9% drawdown), so normalise the sign here.
    book_maxdd_signed = None if book_maxdd_pct is None else -abs(book_maxdd_pct)

    # Daily book returns keyed by the date each return is realised on, for the
    # positive-every-year stationarity guard. The pct-change for the step ending
    # on ``cur`` is realised on ``cur``'s date — the same day-0-dropping
    # convention hedging_gate uses. Derive it directly from consecutive points
    # (not by zipping the day-0-dropping ``_returns`` list, whose length can
    # differ if a point is ≤0) so the date↔return pairing can never desync.
    daily_by_date: dict[date, float] = {}
    for (_, prev_eq), (cur_d, cur_eq) in zip(equity_points, equity_points[1:], strict=False):
        if prev_eq:
            d = cur_d.date() if isinstance(cur_d, datetime) else cur_d
            daily_by_date[d] = cur_eq / prev_eq - 1.0
    year_guard = _positive_every_judged_year(daily_by_date)
    positive_every_year = year_guard["positive_every_year"]

    n_unmodeled = sum(1 for leg in per_leg if leg.get("basis_unmodeled"))

    # Evaluate each clause; the FIRST failing clause (in this priority order)
    # becomes the headline ``reason`` so the operator gets one actionable cause.
    sharpe_ok = book_sharpe is not None and book_sharpe >= MIN_BOOK_SHARPE
    maxdd_ok = book_maxdd_signed is not None and book_maxdd_signed >= MAX_BOOK_MAXDD_PCT
    legs_ok = n_legs >= MIN_N_LEGS
    return_ok = (
        net_return_pct is not None
        and net_return_pct >= ANNUALIZED_RETURN_PASS_THRESHOLD_PCT
    )
    leverage_ok = leverage <= leverage_cap
    basis_ok = (not require_basis_coverage) or n_unmodeled == 0

    if not sharpe_ok:
        reason = "book_sharpe_below_min"
    elif not maxdd_ok:
        reason = "book_maxdd_too_deep"
    elif not legs_ok:
        reason = "too_few_legs"
    elif not return_ok:
        reason = "net_return_below_floor"
    elif not leverage_ok:
        reason = "leverage_over_cap"
    elif not positive_every_year:
        reason = "not_positive_every_year"
    elif not basis_ok:
        reason = "basis_coverage_required"
    else:
        reason = "passed"

    passed = bool(
        sharpe_ok and maxdd_ok and legs_ok and return_ok
        and leverage_ok and positive_every_year and basis_ok
    )

    return {
        "passed": passed,
        "book_sharpe": None if book_sharpe is None else round(book_sharpe, 4),
        "book_maxdd_pct": None if book_maxdd_signed is None else round(book_maxdd_signed, 4),
        "n_legs": n_legs,
        "net_of_cost_return_pct": None if net_return_pct is None else round(net_return_pct, 4),
        "leverage_used": leverage,
        "leverage_cap": leverage_cap,
        "per_leg": per_leg,
        "positive_every_year": positive_every_year,
        "year_guard": year_guard,
        "n_basis_unmodeled_legs": n_unmodeled,
        "reason": reason,
        "thresholds": thresholds,
    }


def per_leg_from_returns(
    leg_returns: dict[str, list[float]]
) -> list[dict[str, Any]]:
    """Per-leg attribution from raw per-leg daily-return series (preview lane).

    ``leg_returns`` maps symbol → its daily-return list; each leg's ``return``
    is the compounded total over its series. ``basis_unmodeled`` flags legs
    outside the perp_basis coverage set — identical semantics to the
    trade-derived ``_per_leg_from_trades``."""
    out: list[dict[str, Any]] = []
    for sym in sorted(leg_returns):
        cum = 1.0
        for r in leg_returns[sym]:
            cum *= 1.0 + r
        out.append({
            "symbol": sym,
            "return": round(cum - 1.0, 6),
            "n_obs": len(leg_returns[sym]),
            "basis_unmodeled": sym not in BASIS_COVERED_SYMBOLS,
        })
    return out


def preview_from_leg_returns(
    *,
    leg_returns: dict[str, list[float]],
    leverage: float,
    borrow_cost_bps_per_day: float = 0.0,
    start: date | None = None,
    leverage_cap: float = DEFAULT_LEVERAGE_CAP,
    require_basis_coverage: bool = False,
) -> dict[str, Any]:
    """Stateless what-if: build a synthetic book equity curve from raw per-leg
    daily-return series + a leverage + a flat borrow model, then run the SAME
    ``evaluate_book`` gate. No stored run required.

    The book daily return on day ``t`` is the equal-NOTIONAL mean of the legs'
    returns present that day, scaled by ``leverage``, minus the per-day borrow
    drag ``leverage * borrow_cost_bps_per_day / 1e4``. The legs are date-aligned
    by INDEX (the caller supplies aligned, equal-length series — the common
    contract for a what-if); legs of differing length contribute on the days
    they have. The curve starts at 1.0 on ``start`` (default 2020-01-01) and
    advances one calendar day per step so the year guard has real dates."""
    symbols = sorted(leg_returns)
    n_days = max((len(leg_returns[s]) for s in symbols), default=0)
    borrow_daily = leverage * (borrow_cost_bps_per_day / 1e4)

    d0 = start or date(2020, 1, 1)
    equity_points: list[tuple[date, float]] = [(d0, 1.0)]
    equity = 1.0
    for i in range(n_days):
        present = [leg_returns[s][i] for s in symbols if i < len(leg_returns[s])]
        if present:
            book_ret = leverage * (sum(present) / len(present)) - borrow_daily
        else:
            book_ret = 0.0
        equity *= 1.0 + book_ret
        equity_points.append((d0 + timedelta(days=i + 1), equity))

    per_leg = per_leg_from_returns(leg_returns)
    verdict = evaluate_book(
        equity_points=equity_points,
        n_legs=len(symbols),
        leverage=leverage,
        per_leg=per_leg,
        leverage_cap=leverage_cap,
        require_basis_coverage=require_basis_coverage,
    )
    verdict["borrow_cost_bps_per_day"] = borrow_cost_bps_per_day
    verdict["source"] = "preview"
    return verdict


# ── DB-backed lane: read a completed PORTFOLIO_CARRY run + certify it ─────────


async def _fetch_book_run(
    conn: asyncpg.Connection, backtest_run_id: UUID
) -> dict[str, Any] | None:
    """Read the run header + the PR #1 carry columns for one backtest run.

    Mirrors ``tick._fetch_run_metrics`` / ``combination``'s scoped run read.
    Returns None when the run row is absent. ``book_n_legs`` / ``book_leverage``
    / ``borrow_cost_bps_per_day`` are the columns PR #1 added; ``backtest_mode``
    distinguishes a PORTFOLIO_CARRY run from a single-asset one."""
    row = await conn.fetchrow(
        """
        SELECT backtest_run_id, strategy_code, status, asset, interval_name,
               start_time, end_time, backtest_mode,
               book_n_legs, book_leverage, borrow_cost_bps_per_day
        FROM backtest_run
        WHERE backtest_run_id = $1
        """,
        backtest_run_id,
    )
    if row is None:
        return None
    return dict(row)


async def _fetch_book_trades(
    conn: asyncpg.Connection, backtest_run_id: UUID
) -> list[dict[str, Any]]:
    """Per-leg trade rows for leg attribution — ``asset`` + ``realized_pnl_amount``.

    The shared ``trades_repo.fetch_trades`` does NOT select ``asset`` (the
    single-asset analyzer never needed it), so the book gate reads the two
    columns it needs directly. ``asset`` is the preserved per-leg symbol (PR #1
    restamps strategy_code but not asset)."""
    rows = await conn.fetch(
        """
        SELECT asset, realized_pnl_amount, entry_time, exit_time
        FROM backtest_trade
        WHERE backtest_run_id = $1
        ORDER BY entry_time ASC
        """,
        backtest_run_id,
    )
    return [dict(r) for r in rows]


async def evaluate(
    conn: asyncpg.Connection,
    *,
    backtest_run_id: UUID,
    leverage_cap: float = DEFAULT_LEVERAGE_CAP,
    require_basis_coverage: bool = False,
) -> dict[str, Any]:
    """Certify a completed PORTFOLIO_CARRY backtest run on book-level metrics.

    Fetches the run header (+ PR #1 carry columns), the ``backtest_equity_point``
    book curve, and the ``backtest_trade`` rows; computes the per-leg attribution
    + book metrics; and applies ``evaluate_book``. Returns the structured verdict
    augmented with ``backtest_run_id`` / ``backtest_mode`` / a ``run_found`` flag.

    Does NOT write — the API layer owns any audit/journal side effects."""
    run = await _fetch_book_run(conn, backtest_run_id)
    if run is None:
        return {
            "passed": False,
            "reason": "backtest_run_not_found",
            "backtest_run_id": str(backtest_run_id),
            "run_found": False,
            "thresholds": _thresholds(
                leverage_cap=leverage_cap, require_basis_coverage=require_basis_coverage
            ),
        }

    n_legs = int(run.get("book_n_legs") or 0)
    leverage = float(run.get("book_leverage") or 1.0)

    points = await backtest_equity.fetch_equity_points(conn, backtest_run_id)
    trades = await _fetch_book_trades(conn, backtest_run_id)
    per_leg = _per_leg_from_trades(trades)

    verdict = evaluate_book(
        equity_points=points,
        n_legs=n_legs,
        leverage=leverage,
        per_leg=per_leg,
        leverage_cap=leverage_cap,
        require_basis_coverage=require_basis_coverage,
    )
    verdict["backtest_run_id"] = str(backtest_run_id)
    verdict["backtest_mode"] = run.get("backtest_mode")
    verdict["borrow_cost_bps_per_day"] = (
        float(run["borrow_cost_bps_per_day"])
        if run.get("borrow_cost_bps_per_day") is not None
        else None
    )
    verdict["run_found"] = True
    verdict["source"] = "stored_run"
    return verdict
