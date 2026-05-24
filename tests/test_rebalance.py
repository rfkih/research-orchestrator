"""Pure-function tests for the portfolio rebalance service.

Covers:
  * Pin tests for operator-controlled guardrails (MIN/MAX weight,
    MIN_OBS_FOR_REBALANCE, MIN_USDT_NOTIONAL_USD, safety factor).
  * Cross-language pin: Python ``MIN_USDT_NOTIONAL_USD`` matches the
    Java ``TradeConstant.MIN_USDT_NOTIONAL`` literal.
  * ``_apply_guardrails`` — clamp + renormalise round-trip, edge cases.
  * ``_compute_raw_weights`` — single-strategy short-circuit, optimiser
    dispatch.
  * ``_check_min_notional`` — sub-floor detection, edge cases, the
    "kelly upper bound" assumption documented in the source.

DB-touching ``rebalance_book`` end-to-end lives behind the integration
marker (uses pytest-postgresql + Flyway V109) and is not in this file.
"""

from __future__ import annotations

import math
import re
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from orchestrator.services.rebalance import (
    EFFECTIVE_MIN_NOTIONAL_USD,
    MAX_WEIGHT,
    MIN_NOTIONAL_SAFETY_FACTOR,
    MIN_OBS_FOR_REBALANCE,
    MIN_USDT_NOTIONAL_USD,
    MIN_WEIGHT,
    _apply_guardrails,
    _check_min_notional,
    _compute_raw_weights,
)


# ── Pin tests for operator-controlled constants ───────────────────────


def test_rebalance_guardrails_pinned() -> None:
    """Changing MIN/MAX widens or narrows the rebalance batch's bounds.
    Update only with a documented ORCHESTRATOR_CHANGE journal entry.
    MIN_OBS_FOR_REBALANCE controls when HRP falls back to equal-weight."""
    assert MIN_WEIGHT == 0.05
    assert MAX_WEIGHT == 0.50
    assert MIN_OBS_FOR_REBALANCE == 30


def test_min_notional_guardrails_pinned() -> None:
    """Pin the min-notional guard constants.

    ``MIN_USDT_NOTIONAL_USD`` MUST match ``TradeConstant.MIN_USDT_NOTIONAL``
    in the trading JVM — the executor silently rejects below this. The
    safety factor was 2.0 in Phase A.5 to cover the ``kelly=1.0``
    upper-bound approximation; Phase A.6 (2026-05-22) tightened it to
    1.2 once the orchestrator gained a live Kelly read via the
    :class:`TradingJvmClient`. The remaining buffer covers equity/Kelly
    drift between Preview and signal fire, plus float precision around
    the floor. Loosening past 1.0 makes the guard ineffective and
    requires an ORCHESTRATOR_CHANGE journal entry citing rationale."""
    assert MIN_USDT_NOTIONAL_USD == 7.0
    assert MIN_NOTIONAL_SAFETY_FACTOR == 1.2
    assert EFFECTIVE_MIN_NOTIONAL_USD == pytest.approx(8.4, abs=1e-9)


def test_min_notional_pin_matches_java() -> None:
    """Cross-language pin: Python ``MIN_USDT_NOTIONAL_USD`` must equal
    the literal in ``TradeConstant.java:11``.

    Drift = silent failure: if the Java side moves to $10 and the
    Python guard stays at $7, the guard says "fine" for entries the
    executor will reject. If the Python guard moves to $10 and Java
    stays at $7, the guard refuses applies the executor would have
    allowed. Either direction breaks the contract.

    Skipped when the trading JVM repo isn't sibling-checked-out (CI
    runs orchestrator-only). When skipped, the in-process pin above
    is still asserted — the cross-language check is belt-and-braces.
    """
    repo_root = Path(__file__).resolve().parents[2]
    java_path = (
        repo_root
        / "blackheart-trading-engine"
        / "src"
        / "main"
        / "java"
        / "id"
        / "co"
        / "blackheart"
        / "util"
        / "TradeConstant.java"
    )
    if not java_path.exists():
        pytest.skip(f"trading JVM repo not present at {java_path}")
    text = java_path.read_text(encoding="utf-8")
    match = re.search(
        r"MIN_USDT_NOTIONAL\s*=\s*new\s+BigDecimal\(\"([\d.]+)\"\)",
        text,
    )
    assert match is not None, (
        "MIN_USDT_NOTIONAL literal not found in TradeConstant.java — "
        "the Java symbol may have been renamed; update the regex AND "
        "the Python pin to match."
    )
    java_value = float(match.group(1))
    assert MIN_USDT_NOTIONAL_USD == pytest.approx(java_value, abs=1e-9), (
        f"Python MIN_USDT_NOTIONAL_USD ({MIN_USDT_NOTIONAL_USD}) "
        f"drifted from TradeConstant.MIN_USDT_NOTIONAL ({java_value}). "
        f"Update both files together; see services/rebalance.py docstring."
    )


# ── _apply_guardrails ────────────────────────────────────────────────


def test_apply_guardrails_empty_input() -> None:
    weights, diag = _apply_guardrails({})
    assert weights == {}
    assert diag["n_clamped_low"] == 0
    assert diag["n_clamped_high"] == 0


def test_apply_guardrails_passthrough_in_bounds() -> None:
    """Weights already in [MIN, MAX] survive the clamp and renormalise
    to sum=1 trivially since they already do."""
    raw = {"LSR": 0.3, "VCB": 0.3, "VBO": 0.4}
    weights, diag = _apply_guardrails(raw)
    assert diag["n_clamped_low"] == 0
    assert diag["n_clamped_high"] == 0
    assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9)
    for c in raw:
        assert math.isclose(weights[c], raw[c], abs_tol=1e-9)


def test_apply_guardrails_clamps_low_weight() -> None:
    """A weight below MIN_WEIGHT is floored to MIN_WEIGHT, then the
    whole vector is renormalised to sum=1."""
    raw = {"LSR": 0.01, "VCB": 0.495, "VBO": 0.495}
    weights, diag = _apply_guardrails(raw)
    assert diag["n_clamped_low"] == 1
    assert weights["LSR"] >= MIN_WEIGHT * 0.9   # post-renorm may shrink slightly
    assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9)


def test_apply_guardrails_clamps_high_weight() -> None:
    """A weight above MAX_WEIGHT is capped, then renormalised. The
    capped weight is reduced further by the renorm factor."""
    raw = {"LSR": 0.9, "VCB": 0.05, "VBO": 0.05}
    weights, diag = _apply_guardrails(raw)
    assert diag["n_clamped_high"] == 1
    assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9)
    # After clamp the raw sum is 0.5 + 0.05 + 0.05 = 0.6 → renorm by 1/0.6.
    # LSR: 0.5 * (1/0.6) ≈ 0.833 (post-renorm may exceed MAX, that's
    # accepted Phase A — see module docstring).
    assert weights["LSR"] >= MAX_WEIGHT


def test_apply_guardrails_zero_sum_is_safe() -> None:
    """A pathological all-zero raw input cannot renormalise to sum=1.
    The implementation returns the clamped-but-not-renorm version
    rather than dividing by zero."""
    raw = {"LSR": 0.0, "VCB": 0.0}
    weights, diag = _apply_guardrails(raw)
    # All clamped up to MIN_WEIGHT, then renorm. 2 * MIN_WEIGHT = 0.10 → factor 10.
    assert diag["n_clamped_low"] == 2
    assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9)


# ── _compute_raw_weights ─────────────────────────────────────────────


def _make_series(n_days: int, seed: int) -> dict[date, float]:
    """Deterministic synthetic daily-return series for testing."""
    rng = np.random.default_rng(seed)
    base = date(2026, 1, 1)
    return {base + timedelta(days=i): float(rng.normal(0.001, 0.02)) for i in range(n_days)}


def test_compute_raw_weights_single_strategy_shortcircuit() -> None:
    """One strategy → weight 1.0 regardless of optimiser. HRP/MV both
    require N>=2 so the dispatcher must short-circuit before reaching
    them."""
    series_by_code: dict[str, dict[date, float]] = {"LSR": _make_series(60, 1)}
    for opt in ("HRP", "EQUAL_WEIGHT", "MEAN_VARIANCE"):
        w = _compute_raw_weights(
            optimizer=opt,  # type: ignore[arg-type]
            codes=["LSR"],
            series_by_code=series_by_code,
            min_overlap_days=5,
            mu_by_code=None,
            risk_aversion=1.0,
        )
        assert w == {"LSR": 1.0}, opt


def test_compute_raw_weights_equal_weight_dispatch() -> None:
    """EQUAL_WEIGHT should give 1/N — doesn't even need the
    correlation matrix."""
    codes = ["LSR", "VCB", "VBO"]
    series_by_code = {c: _make_series(60, i) for i, c in enumerate(codes)}
    w = _compute_raw_weights(
        optimizer="EQUAL_WEIGHT",
        codes=codes,
        series_by_code=series_by_code,
        min_overlap_days=5,
        mu_by_code=None,
        risk_aversion=1.0,
    )
    assert all(math.isclose(w[c], 1.0 / 3.0, abs_tol=1e-9) for c in codes)


def test_compute_raw_weights_hrp_returns_simplex() -> None:
    """HRP output should be a non-negative, sum-to-1 weighting over
    the provided codes."""
    codes = ["LSR", "VCB", "VBO"]
    series_by_code = {c: _make_series(60, i) for i, c in enumerate(codes)}
    w = _compute_raw_weights(
        optimizer="HRP",
        codes=codes,
        series_by_code=series_by_code,
        min_overlap_days=5,
        mu_by_code=None,
        risk_aversion=1.0,
    )
    assert set(w.keys()) == set(codes)
    assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-6)
    assert all(v >= 0 for v in w.values())


def test_compute_raw_weights_mv_with_mu() -> None:
    """Mean-variance with mu provided should produce a long-only
    simplex weighting."""
    codes = ["LSR", "VCB", "VBO"]
    series_by_code = {c: _make_series(60, i) for i, c in enumerate(codes)}
    mu = {"LSR": 0.001, "VCB": 0.001, "VBO": 0.001}
    w = _compute_raw_weights(
        optimizer="MEAN_VARIANCE",
        codes=codes,
        series_by_code=series_by_code,
        min_overlap_days=5,
        mu_by_code=mu,
        risk_aversion=1.0,
    )
    assert set(w.keys()) == set(codes)
    assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-6)
    assert all(v >= 0 for v in w.values())


# ── _check_min_notional ──────────────────────────────────────────────


def _make_row(
    code: str,
    symbol: str = "BTCUSDT",
    cap_alloc_pct: float = 30.0,
) -> dict[str, Any]:
    """Build a target row the way ``_fetch_target_account_strategies``
    returns one. ``capital_allocation_pct`` is a Decimal in production
    (asyncpg returns Decimal for NUMERIC); tests pass Decimal too so
    the float-coercion in the guard is exercised end-to-end."""
    return {
        "account_strategy_id": f"as-{code}",
        "account_id": "acct-A",
        "strategy_code": code,
        "symbol": symbol,
        "interval_name": "1h",
        "capital_allocation_pct": Decimal(str(cap_alloc_pct)),
        "portfolio_weight": Decimal("1.0"),
        "weight_source": "EQUAL_WEIGHT",
    }


def test_check_min_notional_all_above_floor() -> None:
    """Comfortable equity × cap_alloc × weight stays well above the
    floor → no violations. Sanity check: $10k account, 30% allocation,
    33% weight → $990 projected entry vs $14 floor."""
    rows = [
        _make_row("LSR", cap_alloc_pct=30),
        _make_row("VCB", cap_alloc_pct=30),
        _make_row("VBO", cap_alloc_pct=20),
    ]
    weights = {"LSR": 0.33, "VCB": 0.33, "VBO": 0.34}
    violations = _check_min_notional(rows, weights, available_usdt=10_000.0)
    assert violations == []


def test_check_min_notional_below_floor_flagged() -> None:
    """Small equity drives some strategies below the floor.

    Math at the Phase A.6 effective floor ($8.40):
      LSR: 100 × 0.30 × 0.10 = $3.00 → below → FLAGGED (shortfall 5.40)
      VCB: 100 × 0.30 × 0.50 = $15.00 → above → CLEAN
      VBO: 100 × 0.20 × 0.40 = $8.00 → below → FLAGGED (shortfall 0.40)

    Kelly defaults to 1.0 when no map is passed (the legacy Phase A.5
    fallback path). Both flagged rows carry structured forensics
    including the shortfall + Kelly so the operator can prioritise the
    worst row."""
    rows = [
        _make_row("LSR", cap_alloc_pct=30),
        _make_row("VCB", cap_alloc_pct=30),
        _make_row("VBO", cap_alloc_pct=20),
    ]
    weights = {"LSR": 0.10, "VCB": 0.50, "VBO": 0.40}
    violations = _check_min_notional(rows, weights, available_usdt=100.0)
    flagged = {v["strategy_code"]: v for v in violations}
    assert set(flagged.keys()) == {"LSR", "VBO"}
    assert flagged["LSR"]["projected_entry_usd"] == pytest.approx(3.0, abs=0.01)
    assert flagged["LSR"]["shortfall_usd"] == pytest.approx(5.4, abs=0.01)
    assert flagged["LSR"]["kelly_multiplier"] == 1.0
    assert flagged["VBO"]["projected_entry_usd"] == pytest.approx(8.0, abs=0.01)
    assert flagged["VBO"]["shortfall_usd"] == pytest.approx(0.4, abs=0.01)
    # Every violation carries the floor so the UI doesn't need a
    # second round-trip to format the warning.
    for v in violations:
        assert v["floor_usd"] == EFFECTIVE_MIN_NOTIONAL_USD


def test_check_min_notional_zero_equity_flags_all_with_reason() -> None:
    """Defensive: zero equity flags every weighted row with a
    structured ``non_positive_equity`` reason so the operator can
    distinguish "below floor" from "bad input".

    Pydantic validates ``gt=0`` on the request body, but direct
    callers (cron, tests) might pass zero. We must not divide by
    zero and must not silently return clean."""
    rows = [
        _make_row("LSR", cap_alloc_pct=30),
        _make_row("VCB", cap_alloc_pct=30),
    ]
    weights = {"LSR": 0.5, "VCB": 0.5}
    violations = _check_min_notional(rows, weights, available_usdt=0.0)
    assert {v["strategy_code"] for v in violations} == {"LSR", "VCB"}
    for v in violations:
        assert v["reason"] == "non_positive_equity"
        assert v["projected_entry_usd"] == 0.0


def test_check_min_notional_skips_zero_weight() -> None:
    """A strategy whose post-clamp weight is 0 (or missing) isn't
    going to fire anyway — no violation to flag. The guard exists to
    catch *active* strategies whose entries silently die at the
    executor, not to flag inactive rows."""
    rows = [_make_row("LSR", cap_alloc_pct=30)]
    violations = _check_min_notional(rows, {"LSR": 0.0}, available_usdt=100.0)
    assert violations == []
    violations = _check_min_notional(rows, {}, available_usdt=100.0)
    assert violations == []


def test_check_min_notional_skips_zero_cap_alloc() -> None:
    """``capital_allocation_pct=0`` means the executor's
    ``allocFraction`` returns ZERO → no entry fires regardless of the
    portfolio weight. The guard skips the row so the operator's
    "zero out via cap_alloc" workflow doesn't trigger spurious
    sub-floor flags."""
    rows = [_make_row("LSR", cap_alloc_pct=0)]
    violations = _check_min_notional(rows, {"LSR": 0.5}, available_usdt=100.0)
    assert violations == []


def test_check_min_notional_negative_equity_treated_as_zero() -> None:
    """A negative equity figure is nonsensical; the guard treats it as
    the ``non_positive_equity`` case rather than producing a negative
    projected entry that compares spuriously below the floor."""
    rows = [_make_row("LSR", cap_alloc_pct=30)]
    violations = _check_min_notional(rows, {"LSR": 0.5}, available_usdt=-100.0)
    assert len(violations) == 1
    assert violations[0]["reason"] == "non_positive_equity"


def test_check_min_notional_infinite_weight_skipped() -> None:
    """Defensive: numerical NaN/inf weights (e.g. from a buggy
    upstream solver) shouldn't crash the guard. They're skipped — the
    upstream renormalise should have surfaced them already, but the
    guard fails closed rather than open."""
    rows = [_make_row("LSR", cap_alloc_pct=30)]
    violations = _check_min_notional(
        rows, {"LSR": float("inf")}, available_usdt=100.0
    )
    assert violations == []
    violations = _check_min_notional(
        rows, {"LSR": float("nan")}, available_usdt=100.0
    )
    assert violations == []


def test_check_min_notional_boundary_at_floor() -> None:
    """A projected entry exactly equal to the floor passes; only
    *below* the floor is a violation. This matches the JVM's
    ``compareTo(MIN_USDT_NOTIONAL) < 0`` predicate in
    ``LiveTradingDecisionExecutorService.calculateLongTradeAmount``."""
    # Engineer: equity × 0.10 × 1.0 = 14.0 (exactly the floor at default
    # safety factor) → no violation.
    rows = [_make_row("LSR", cap_alloc_pct=10)]
    equity_at_floor = EFFECTIVE_MIN_NOTIONAL_USD / 0.10 / 1.0
    violations = _check_min_notional(
        rows, {"LSR": 1.0}, available_usdt=equity_at_floor
    )
    assert violations == []
    # One cent below → violation.
    violations = _check_min_notional(
        rows, {"LSR": 1.0}, available_usdt=equity_at_floor - 0.10
    )
    assert len(violations) == 1


def test_check_min_notional_kelly_haircut_flips_clean_to_violated() -> None:
    """Phase A.6: a live Kelly read can flip a clean strategy to a
    violation when Kelly < 1.0. With no Kelly map (legacy Phase A.5
    path) we'd miss this.

    Math:
      kelly=1.0 → 100 × 0.10 × 1.0 × 1.0 = $10.00 → above $8.40, CLEAN
      kelly=0.5 → 100 × 0.10 × 0.5 × 1.0 = $5.00  → below $8.40, FLAGGED

    The guard reports the Kelly value used in the math so the operator
    can correlate with the JVM's /kelly-status admin endpoint."""
    rows = [_make_row("LSR", cap_alloc_pct=10)]
    weights = {"LSR": 1.0}

    no_kelly = _check_min_notional(rows, weights, available_usdt=100.0)
    assert no_kelly == []

    with_kelly = _check_min_notional(
        rows,
        weights,
        available_usdt=100.0,
        kelly_by_account_strategy_id={str(rows[0]["account_strategy_id"]): 0.5},
    )
    assert len(with_kelly) == 1
    v = with_kelly[0]
    assert v["strategy_code"] == "LSR"
    assert v["kelly_multiplier"] == 0.5
    assert v["projected_entry_usd"] == pytest.approx(5.0, abs=0.01)


def test_check_min_notional_kelly_keyed_by_account_strategy_id() -> None:
    """Multi-symbol books have multiple rows sharing one strategy_code;
    Kelly is per ``account_strategy_id`` not per code. The map's keys
    must match what ``_fetch_target_account_strategies`` returns."""
    rows = [
        _make_row("LSR", symbol="BTCUSDT", cap_alloc_pct=30),
        _make_row("LSR", symbol="ETHUSDT", cap_alloc_pct=30),
    ]
    # Same _make_row helper produces a deterministic
    # ``as-LSR`` account_strategy_id for both rows; rebuild with
    # distinct ids so the test is realistic.
    rows[0]["account_strategy_id"] = "as-LSR-BTC"
    rows[1]["account_strategy_id"] = "as-LSR-ETH"
    weights = {"LSR": 1.0}
    kelly_map = {"as-LSR-BTC": 0.5, "as-LSR-ETH": 1.0}
    violations = _check_min_notional(
        rows,
        weights,
        available_usdt=100.0,
        kelly_by_account_strategy_id=kelly_map,
    )
    # BTC row: 100 × 0.30 × 0.5 × 1.0 = $15 → above $8.40, CLEAN.
    # ETH row: 100 × 0.30 × 1.0 × 1.0 = $30 → above $8.40, CLEAN.
    # Both clean despite different Kelly values — the floor is $8.40 not $14.
    assert violations == []

    # Tighten: BTC at kelly=0.1 → $3.00, below floor.
    kelly_map["as-LSR-BTC"] = 0.1
    violations = _check_min_notional(
        rows, weights, available_usdt=100.0, kelly_by_account_strategy_id=kelly_map
    )
    assert len(violations) == 1
    assert violations[0]["symbol"] == "BTCUSDT"
    assert violations[0]["kelly_multiplier"] == 0.1


def test_check_min_notional_kelly_missing_row_falls_back_to_one() -> None:
    """When the Kelly map is provided but a specific row's id is
    absent, the guard uses kelly=1.0 for that row — same semantics as
    the executor's no-op when ``kellySizingEnabled = false``."""
    rows = [_make_row("LSR", cap_alloc_pct=10)]
    weights = {"LSR": 1.0}
    # Map is provided but doesn't contain this row's id.
    violations = _check_min_notional(
        rows,
        weights,
        available_usdt=100.0,
        kelly_by_account_strategy_id={"some-other-id": 0.1},
    )
    # 100 × 0.10 × 1.0 (fallback) × 1.0 = $10 → above $8.40, CLEAN.
    assert violations == []


def test_check_min_notional_kelly_invalid_falls_back_to_one() -> None:
    """Defensive: a NaN / inf / non-positive Kelly read shouldn't
    crash the guard. We treat it as the no-op fallback (kelly=1.0)
    rather than letting the bad value propagate into the math."""
    rows = [_make_row("LSR", cap_alloc_pct=10)]
    weights = {"LSR": 1.0}
    for bad in (float("nan"), float("inf"), -1.0, 0.0):
        violations = _check_min_notional(
            rows,
            weights,
            available_usdt=100.0,
            kelly_by_account_strategy_id={
                str(rows[0]["account_strategy_id"]): bad
            },
        )
        # Falls back to kelly=1.0 → $10 → above $8.40, CLEAN.
        assert violations == [], f"bad kelly={bad} caused unexpected flag"


def test_check_min_notional_carries_symbol_for_ui() -> None:
    """Each violation carries the symbol so the UI can render
    ``LSR/BTCUSDT`` rather than asking the operator to cross-reference
    the strategy_code to a row. Multi-symbol books (post-Phase D) will
    have multiple rows per code, so symbol is required to disambiguate."""
    rows = [
        _make_row("LSR", symbol="BTCUSDT", cap_alloc_pct=30),
        _make_row("LSR", symbol="ETHUSDT", cap_alloc_pct=30),
    ]
    violations = _check_min_notional(rows, {"LSR": 0.05}, available_usdt=100.0)
    # Both rows violate (100 × 0.30 × 0.05 = $1.50 each).
    assert len(violations) == 2
    symbols = {v["symbol"] for v in violations}
    assert symbols == {"BTCUSDT", "ETHUSDT"}
