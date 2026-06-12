"""Pure-function tests for the portfolio correlation gate.

Covers:
  * ``daily_returns_from_trades`` — empty, mixed-day, capital-normalised.
  * ``pearson_corr`` — perfect/zero/anti correlation, thin-overlap None.
  * ``max_abs_correlation`` — absolute value, picks the binding code.
  * ``effective_pf_lo`` — discount math at boundary cases.
  * ``apply_portfolio_gate`` — demote/no-demote, missing baseline, edges.
  * Pin tests for operator-controlled constants.

Async ``fetch_book_baselines`` needs a real DB and lives behind the
integration marker (not in this file).
"""

from __future__ import annotations

import math
from datetime import date, datetime

from orchestrator.services.portfolio import (
    BASELINE_CACHE_TTL_S,
    MIN_OVERLAP_DAYS_FOR_CORR,
    PORTFOLIO_DISCOUNT_K,
    PROTECTED_STRATEGY_CODES,
    _ranks_with_average_ties,
    apply_portfolio_gate,
    daily_returns_from_trades,
    effective_pf_lo,
    max_abs_correlation,
    max_positive_correlation,
    pearson_corr,
    spearman_corr,
)


# ── Pin tests for operator-controlled constants ───────────────────────


def test_portfolio_constants_are_pinned() -> None:
    # Changing K shifts how harshly correlation discounts pf_lo.
    # Changing PROTECTED_STRATEGY_CODES adds/removes book strategies.
    # Changing MIN_OVERLAP_DAYS_FOR_CORR weakens the correlation
    # estimator. Changing BASELINE_MAX_AGE_DAYS changes how stale a
    # baseline backtest can be before it stops counting. Update any of
    # these only with a documented ORCHESTRATOR_CHANGE journal entry.
    from orchestrator.services.portfolio import BASELINE_MAX_AGE_DAYS

    assert PORTFOLIO_DISCOUNT_K == 0.5
    assert PROTECTED_STRATEGY_CODES == ("LSR", "VCB", "VBO")
    # Raised 5 -> 20 (2026-06-12): Spearman on 5 points (SE ~0.7) must
    # not bind a 50%-PF-discount decision.
    assert MIN_OVERLAP_DAYS_FOR_CORR == 20
    assert BASELINE_CACHE_TTL_S == 24 * 3600
    assert BASELINE_MAX_AGE_DAYS == 30


# ── daily_returns_from_trades ─────────────────────────────────────────


def test_daily_returns_empty_returns_empty() -> None:
    assert daily_returns_from_trades([]) == {}


def test_daily_returns_groups_same_day_by_exit() -> None:
    # Bucketing is by exit_time — pnl is realized at exit, not entry.
    e1 = datetime(2026, 1, 1, 9, 30)
    e2 = datetime(2026, 1, 1, 14, 0)
    e3 = datetime(2026, 1, 2, 9, 30)
    out = daily_returns_from_trades(
        [
            {"exit_time": e1, "realized_pnl_amount": "1.0"},
            {"exit_time": e2, "realized_pnl_amount": "2.0"},
            {"exit_time": e3, "realized_pnl_amount": "0.5"},
        ],
        initial_capital=100.0,
    )
    assert out[date(2026, 1, 1)] == 0.03
    assert out[date(2026, 1, 2)] == 0.005


def test_daily_returns_falls_back_to_entry_when_exit_missing() -> None:
    # Some trade rows may be missing exit_time (open at end of backtest);
    # entry_time is the safe fallback.
    et = datetime(2026, 1, 1, 9, 30)
    out = daily_returns_from_trades(
        [{"entry_time": et, "realized_pnl_amount": "1.0"}], initial_capital=100.0
    )
    assert out[date(2026, 1, 1)] == 0.01


def test_daily_returns_uses_exit_time_when_both_present() -> None:
    # The whole point of the change: a 5-day-held trade attributes pnl
    # to exit-day, not entry-day, so candidate and baseline correlation
    # tracks "made money on the same day" rather than "entered on the
    # same day".
    et = datetime(2026, 1, 1, 9, 30)
    xt = datetime(2026, 1, 6, 17, 0)
    out = daily_returns_from_trades(
        [{"entry_time": et, "exit_time": xt, "realized_pnl_amount": "5.0"}],
        initial_capital=100.0,
    )
    assert date(2026, 1, 6) in out
    assert date(2026, 1, 1) not in out


def test_daily_returns_normalises_by_capital() -> None:
    t = datetime(2026, 1, 1, 9, 30)
    raw = [{"exit_time": t, "realized_pnl_amount": "10.0"}]
    out_100 = daily_returns_from_trades(raw, initial_capital=100.0)
    out_1000 = daily_returns_from_trades(raw, initial_capital=1000.0)
    assert out_100[date(2026, 1, 1)] == 0.1
    assert out_1000[date(2026, 1, 1)] == 0.01


def test_daily_returns_skips_invalid_rows() -> None:
    t = datetime(2026, 1, 1, 9, 30)
    out = daily_returns_from_trades(
        [
            {"exit_time": None, "entry_time": None, "realized_pnl_amount": "1.0"},  # no time
            {"exit_time": t, "realized_pnl_amount": "abc"},  # bad pnl
            {"exit_time": t, "realized_pnl_amount": "1.0"},  # OK
        ],
        initial_capital=100.0,
    )
    assert out == {date(2026, 1, 1): 0.01}


def test_daily_returns_handles_zero_or_negative_capital() -> None:
    t = datetime(2026, 1, 1, 9, 30)
    out = daily_returns_from_trades(
        [{"exit_time": t, "realized_pnl_amount": "1.0"}],
        initial_capital=0.0,  # falls back to 100.0
    )
    assert out[date(2026, 1, 1)] == 0.01


# ── pearson_corr ──────────────────────────────────────────────────────


def _series(pairs: list[tuple[int, float]]) -> dict[date, float]:
    return {date(2026, 1, d): v for d, v in pairs}


def _lin(n, f, month: int = 1) -> dict[date, float]:
    """n-day series over 2026-{month} with value f(day). Fixtures need
    >= MIN_OVERLAP_DAYS_FOR_CORR (20) points since the 2026-06-12 raise."""
    return {date(2026, month, d): float(f(d)) for d in range(1, n + 1)}


def test_pearson_corr_perfect_positive_is_one() -> None:
    a = _lin(21, lambda d: d)
    b = _lin(21, lambda d: 2 * d)
    rho = pearson_corr(a, b)
    assert rho is not None
    assert abs(rho - 1.0) < 1e-9


def test_pearson_corr_perfect_negative_is_minus_one() -> None:
    a = _lin(21, lambda d: d)
    b = _lin(21, lambda d: 22 - d)
    rho = pearson_corr(a, b)
    assert rho is not None
    assert abs(rho + 1.0) < 1e-9


def test_pearson_corr_thin_overlap_returns_none() -> None:
    a = _series([(1, 1.0), (2, 2.0)])
    b = _series([(1, 2.0), (2, 3.0)])
    assert pearson_corr(a, b) is None  # overlap=2 < MIN_OVERLAP_DAYS_FOR_CORR


def test_pearson_corr_no_overlap_returns_none() -> None:
    a = _lin(21, lambda d: d)
    b = _lin(21, lambda d: d, month=2)
    assert pearson_corr(a, b) is None


def test_pearson_corr_constant_series_returns_none() -> None:
    a = _lin(21, lambda d: 1.0)
    b = _lin(21, lambda d: d)
    assert pearson_corr(a, b) is None  # zero variance in a → undefined


# ── spearman_corr (robust) ────────────────────────────────────────────


def test_ranks_with_average_ties_simple() -> None:
    assert _ranks_with_average_ties([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]


def test_ranks_with_average_ties_handles_ties() -> None:
    # Two values tied at the bottom → both get rank (1+2)/2 = 1.5
    assert _ranks_with_average_ties([1.0, 1.0, 3.0]) == [1.5, 1.5, 3.0]


def test_spearman_corr_perfect_monotonic_is_one() -> None:
    # Spearman picks up monotonic but non-linear relationships.
    a = _lin(21, lambda d: d)
    b = _lin(21, lambda d: d * d)
    rho = spearman_corr(a, b)
    assert rho is not None
    assert abs(rho - 1.0) < 1e-9


def test_spearman_corr_ignores_outliers_more_than_pearson() -> None:
    # Six matched-rank pairs, then one extreme outlier in B that doesn't
    # change ranks. Pearson swings; Spearman barely moves.
    a = _lin(21, lambda d: d)
    b = _lin(21, lambda d: d if d < 21 else 1000.0)
    rho_s = spearman_corr(a, b)
    rho_p = pearson_corr(a, b)
    assert rho_s is not None and rho_p is not None
    assert abs(rho_s - 1.0) < 1e-9  # ranks still strictly increasing
    assert rho_p < rho_s            # pearson dragged down by the outlier


def test_spearman_corr_thin_overlap_returns_none() -> None:
    a = _series([(1, 1.0), (2, 2.0)])
    b = _series([(1, 2.0), (2, 3.0)])
    assert spearman_corr(a, b) is None


# ── max_abs_correlation ───────────────────────────────────────────────


def test_max_abs_correlation_picks_largest_abs() -> None:
    cand = _lin(21, lambda d: d)
    base = {
        "LSR": _lin(21, lambda d: 22 - d),                        # rho=-1
        "VCB": _lin(21, lambda d: d + (1.5 if d == 3 else 0.0)),  # rho<+1 (one inversion)
        "VBO": _lin(2, lambda d: d, month=2),  # no overlap → None
    }
    # Default method is spearman now; on rank-perfect series, both yield 1.
    max_abs, per_code = max_abs_correlation(cand, base)
    assert max_abs is not None and abs(max_abs - 1.0) < 1e-9
    assert per_code["LSR"] is not None and abs(per_code["LSR"] + 1.0) < 1e-9
    assert per_code["VBO"] is None


def test_max_abs_correlation_method_pearson_explicit() -> None:
    cand = _lin(21, lambda d: d)
    base = {"LSR": _lin(21, lambda d: 2 * d)}
    max_abs, per_code = max_abs_correlation(cand, base, method="pearson")
    assert max_abs is not None and abs(max_abs - 1.0) < 1e-9
    assert per_code["LSR"] is not None and abs(per_code["LSR"] - 1.0) < 1e-9


def test_max_abs_correlation_no_baselines_returns_none() -> None:
    cand = _lin(21, lambda d: d)
    max_abs, per_code = max_abs_correlation(cand, {})
    assert max_abs is None
    assert per_code == {}


# ── max_positive_correlation (2026-05-19 methodology fix) ─────────────


def test_max_positive_correlation_picks_largest_positive() -> None:
    # Candidate is rank-perfectly anti-correlated with LSR (-1.0) and
    # perfectly correlated with VBO (+1.0). max_positive must be +1.0
    # and per_code preserves the SIGNED values (no |abs|).
    cand = _lin(21, lambda d: d)
    base = {
        "LSR": _lin(21, lambda d: 22 - d),
        "VBO": _lin(21, lambda d: d),
    }
    max_positive, per_code = max_positive_correlation(cand, base)
    assert max_positive is not None and abs(max_positive - 1.0) < 1e-9
    assert per_code["LSR"] is not None and abs(per_code["LSR"] + 1.0) < 1e-9
    assert per_code["VBO"] is not None and abs(per_code["VBO"] - 1.0) < 1e-9


def test_max_positive_correlation_all_negative_returns_zero() -> None:
    # Every per-code correlation is negative (pure-diversifier candidate).
    # max_positive returns 0.0 (no positive correlation present), not
    # the |abs| value — i.e. the gate won't discount pf_lo at all.
    cand = _lin(21, lambda d: d)
    base = {
        "LSR": _lin(21, lambda d: 22 - d),
        "VCB": _lin(21, lambda d: 50.0 - 2 * d),
    }
    max_positive, per_code = max_positive_correlation(cand, base)
    assert max_positive == 0.0
    assert all(v is not None and v < 0 for v in per_code.values())


def test_max_positive_correlation_no_baselines_returns_none() -> None:
    cand = _lin(21, lambda d: d)
    max_positive, per_code = max_positive_correlation(cand, {})
    assert max_positive is None
    assert per_code == {}


def test_max_positive_correlation_method_pearson_explicit() -> None:
    cand = _lin(21, lambda d: d)
    base = {"LSR": _lin(21, lambda d: 2 * d)}
    max_positive, per_code = max_positive_correlation(cand, base, method="pearson")
    assert max_positive is not None and abs(max_positive - 1.0) < 1e-9
    assert per_code["LSR"] is not None and abs(per_code["LSR"] - 1.0) < 1e-9


# ── effective_pf_lo ───────────────────────────────────────────────────


def test_effective_pf_lo_zero_corr_unchanged() -> None:
    assert effective_pf_lo(1.5, 0.0) == 1.5


def test_effective_pf_lo_full_corr_halved() -> None:
    # K=0.5; |rho|=1 → multiplier (1 - 0.5*1) = 0.5
    assert effective_pf_lo(1.5, 1.0) == 0.75


def test_effective_pf_lo_negative_input_clamps_to_zero_no_discount() -> None:
    # Methodology fix (2026-05-19): negative correlation = diversification
    # benefit, NOT duplication. effective_pf_lo's input is expected to be
    # ``max_positive_corr`` (>=0); a negative input clamps to 0 (no
    # discount) rather than reintroducing the |abs| semantic that
    # falsely demoted hedges. See operator-escalation journal c58a4b8a.
    assert effective_pf_lo(2.0, -0.6) == 2.0
    # Behaviour mirror: 0.0 also yields no discount.
    assert effective_pf_lo(2.0, 0.0) == 2.0


def test_effective_pf_lo_clamps_corr_above_one() -> None:
    # Caller might pass a noisy estimator slightly >1.0; clamp.
    assert effective_pf_lo(2.0, 1.5) == effective_pf_lo(2.0, 1.0)


# ── apply_portfolio_gate ──────────────────────────────────────────────


def test_apply_gate_no_baseline_skips() -> None:
    out = apply_portfolio_gate({"low": 1.2, "high": 2.0}, max_positive_corr=None)
    assert out["applied"] is False
    assert out["demote"] is False


def test_apply_gate_low_corr_does_not_demote() -> None:
    # pf_lo=1.05, rho=0.1 → eff = 1.05 * (1 - 0.5*0.1) = 0.99825 → demote!
    # Pick rho=0.05 → eff = 1.05 * 0.975 = 1.02375 → cleared.
    out = apply_portfolio_gate({"low": 1.05, "high": 1.5}, 0.05, {"LSR": 0.05})
    assert out["applied"] is True
    assert out["demote"] is False
    assert out["pf_lo_effective"] > 1.0


def test_apply_gate_high_corr_demotes() -> None:
    # pf_lo=1.1, rho=0.6 → eff = 1.1 * (1 - 0.5*0.6) = 1.1 * 0.7 = 0.77 → demote
    out = apply_portfolio_gate({"low": 1.1, "high": 1.8}, 0.6, {"LSR": 0.6})
    assert out["applied"] is True
    assert out["demote"] is True
    assert out["pf_lo_effective"] < 1.0


def test_apply_gate_missing_pf_lo_skips() -> None:
    out = apply_portfolio_gate({"low": None, "high": 1.5}, 0.5)
    assert out["applied"] is False
    assert out["demote"] is False


def test_apply_gate_emits_per_code_breakdown() -> None:
    # max_positive_corr=0.20 (the only positive value; LSR at -0.55 is a
    # hedge, not duplication). Verify both the signed per_code_corr
    # breakdown AND the legacy max_abs_corr field (computed from the
    # signed dict; |-0.55|=0.55) are surfaced for forensics.
    out = apply_portfolio_gate(
        {"low": 1.05, "high": 1.5},
        max_positive_corr=0.20,
        per_code_corr={"LSR": -0.55, "VCB": 0.20, "VBO": None},
    )
    assert out["per_code_corr"]["LSR"] == -0.55
    assert out["per_code_corr"]["VCB"] == 0.20
    assert out["per_code_corr"]["VBO"] is None
    # Legacy forensic field still present (computed from signed dict).
    assert out["max_abs_corr"] == 0.55
    # New binding field.
    assert out["max_positive_corr"] == 0.20


def test_apply_gate_pf_lo_just_above_threshold_demotes_when_corr_high() -> None:
    # Exactly the case the gate exists for: V11 cleared a candidate at
    # pf_lo=1.02, but rho=0.7 with the book makes it redundant.
    out = apply_portfolio_gate({"low": 1.02, "high": 1.4}, 0.7)
    assert out["demote"] is True


def test_apply_gate_anti_correlated_does_not_demote_2026_05_19_fix() -> None:
    # Methodology fix (2026-05-19): a candidate with strongly NEGATIVE
    # correlation against book members is a diversifier (improves
    # portfolio Sharpe), not duplication. Demoting it would falsely
    # reject hedges. The gate now uses max-positive (not |abs|), so an
    # anti-correlated candidate with NO positive correlations clears.
    #
    # Concretely: per_code_corr={"LSR": -0.7} → max_positive_corr = 0.0
    # (no positive entries) → no discount → pf_lo_effective = pf_lo.
    out = apply_portfolio_gate(
        {"low": 1.1, "high": 1.6},
        max_positive_corr=0.0,
        per_code_corr={"LSR": -0.7},
    )
    assert out["applied"] is True
    assert out["demote"] is False, (
        "Anti-correlated candidate must NOT demote — diversification is "
        "a benefit, not a duplication signal. If this assertion fires, "
        "the |corr| bug has regressed (see operator-escalation journal "
        "c58a4b8a)."
    )
    # Forensic: the signed per-code value is preserved.
    assert out["per_code_corr"]["LSR"] == -0.7
    # Legacy max_abs_corr field computed from signed dict for backward
    # compat — readers of old payloads still get a familiar field.
    assert out["max_abs_corr"] == 0.7


def test_apply_gate_mixed_high_pos_and_high_neg_demotes_on_positive() -> None:
    # Mirror of the review.py test: when a candidate has BOTH a
    # high-positive correlation (duplication risk) AND a high-negative
    # correlation (hedge), the positive duplication risk dominates —
    # the hedge against one book member does NOT compensate for being
    # a duplicate of another. pf_lo=1.1, max_positive=0.65 → discount.
    out = apply_portfolio_gate(
        {"low": 1.1, "high": 1.6},
        max_positive_corr=0.65,
        per_code_corr={"LSR": -0.70, "VCB": 0.65, "VBO": 0.10},
    )
    assert out["demote"] is True
    assert out["max_positive_corr"] == 0.65


# ── evaluate_portfolio_gate (wiring integration) ──────────────────────
#
# The hook in tick.py reads run_metrics["initial_capital"] (snake_case
# from Postgres). A typo there would silently disable the gate. These
# tests exercise the exact wiring contract.


class _FakeDb:
    """Minimal Database stand-in: short-circuits fetch_book_baselines
    by returning whatever baselines the test injects via attribute.
    The real Database subclass would acquire a connection and run the
    backtest_run query."""


def _make_trades(days_pnl: dict[int, float]) -> list[dict]:
    """Build trade rows keyed off exit_time day-of-month."""
    return [
        {
            "entry_time": datetime(2026, 1, d, 9, 0),
            "exit_time": datetime(2026, 1, d, 17, 0),
            "realized_pnl_amount": str(pnl),
        }
        for d, pnl in days_pnl.items()
    ]


def test_evaluate_portfolio_gate_reads_snake_case_initial_capital(monkeypatch) -> None:
    # The gate's first decision branch is "did initial_capital normalise
    # the candidate returns correctly". If the field name drifts to
    # camelCase (e.g. JVM JSON convention leaks in), the gate would
    # silently use the 100.0 fallback and produce wrong correlations.
    import asyncio

    from orchestrator.services.portfolio import (
        clear_baseline_cache,
        evaluate_portfolio_gate,
    )

    # Inject a baseline that's perfectly rank-correlated with the
    # candidate. Patch fetch_book_baselines to return it directly.
    candidate_trades = _make_trades({d: float(d) for d in range(1, 22)})
    baseline_returns = {date(2026, 1, d): float(d) for d in range(1, 22)}

    async def fake_fetch(db, codes=None, now_fn=None, **kwargs):
        return {"LSR": baseline_returns}

    monkeypatch.setattr(
        "orchestrator.services.portfolio.fetch_book_baselines", fake_fetch
    )
    clear_baseline_cache()

    gate = asyncio.run(
        evaluate_portfolio_gate(
            db=_FakeDb(),
            run_metrics={"initial_capital": "100.0"},  # snake_case
            trades=candidate_trades,
            pf_ci={"low": 1.05, "high": 1.5},
        )
    )
    # Perfect rank correlation → max_abs_corr near 1.0 → demote.
    assert gate["applied"] is True
    assert gate["demote"] is True
    assert gate["max_abs_corr"] > 0.99


def test_evaluate_portfolio_gate_no_baseline_skips(monkeypatch) -> None:
    import asyncio

    from orchestrator.services.portfolio import (
        clear_baseline_cache,
        evaluate_portfolio_gate,
    )

    async def fake_fetch(db, codes=None, now_fn=None, **kwargs):
        return {}  # no protected backtests on file

    monkeypatch.setattr(
        "orchestrator.services.portfolio.fetch_book_baselines", fake_fetch
    )
    clear_baseline_cache()

    gate = asyncio.run(
        evaluate_portfolio_gate(
            db=_FakeDb(),
            run_metrics={"initial_capital": "100.0"},
            trades=_make_trades({d: float(d) for d in range(1, 22)}),
            pf_ci={"low": 1.05, "high": 1.5},
        )
    )
    assert gate["applied"] is False
    assert gate["demote"] is False


def test_evaluate_portfolio_gate_exposes_both_spearman_and_pearson(monkeypatch) -> None:
    # The gate decision uses Spearman (robust); Pearson is exposed
    # alongside for forensics. This test pins that contract — if a
    # refactor drops one, journal queries and dashboards break.
    import asyncio

    from orchestrator.services.portfolio import (
        clear_baseline_cache,
        evaluate_portfolio_gate,
    )

    candidate_trades = _make_trades({d: float(d) for d in range(1, 22)})
    baseline = {date(2026, 1, d): float(d) for d in range(1, 22)}

    async def fake_fetch(db, codes=None, now_fn=None, **kwargs):
        return {"LSR": baseline}

    monkeypatch.setattr(
        "orchestrator.services.portfolio.fetch_book_baselines", fake_fetch
    )
    clear_baseline_cache()

    gate = asyncio.run(
        evaluate_portfolio_gate(
            db=_FakeDb(),
            run_metrics={"initial_capital": "100.0"},
            trades=candidate_trades,
            pf_ci={"low": 1.05, "high": 1.5},
        )
    )
    assert gate["method"] == "spearman"
    assert "per_code_corr" in gate
    assert "pearson_per_code" in gate
    assert "LSR" in gate["per_code_corr"]
    assert "LSR" in gate["pearson_per_code"]
