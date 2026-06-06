from datetime import date, datetime

from orchestrator.services import hedging_gate


def test_buy_hold_metrics_basic():
    # 5 ascending closes → positive CAGR, finite Sharpe, small drawdown.
    closes = [(date(2024, 1, d), 100.0 + d) for d in range(1, 6)]
    m = hedging_gate.buy_hold_metrics(closes)
    assert m["cagr_pct"] > 0
    assert m["max_drawdown_pct"] >= 0
    assert m["sharpe"] is not None


def test_buy_hold_metrics_drawdown():
    closes = [(date(2024, 1, 1), 100.0), (date(2024, 1, 2), 120.0), (date(2024, 1, 3), 60.0)]
    m = hedging_gate.buy_hold_metrics(closes)
    # peak 120 → trough 60 = 50% drawdown
    assert round(m["max_drawdown_pct"], 1) == 50.0


def test_gate_passes_on_material_sharpe_gain_within_cagr_floor():
    v = hedging_gate.beats_buy_hold_risk_adj(
        strat={"cagr_pct": 30.0, "sharpe": 1.5, "max_drawdown_pct": 20.0},
        bench={"cagr_pct": 32.0, "sharpe": 1.0, "max_drawdown_pct": 40.0},
    )
    assert v["passed"] is True            # CAGR within tol (−2 ≥ −5) AND Sharpe +0.5 ≥ θ
    assert v["reason"]


def test_gate_fails_when_return_floor_breached():
    v = hedging_gate.beats_buy_hold_risk_adj(
        strat={"cagr_pct": 5.0, "sharpe": 3.0, "max_drawdown_pct": 5.0},
        bench={"cagr_pct": 30.0, "sharpe": 1.0, "max_drawdown_pct": 40.0},
    )
    assert v["passed"] is False           # gave up 25% CAGR > tol → fail regardless of Sharpe


def test_gate_fails_without_material_improvement():
    v = hedging_gate.beats_buy_hold_risk_adj(
        strat={"cagr_pct": 31.0, "sharpe": 1.05, "max_drawdown_pct": 39.0},
        bench={"cagr_pct": 32.0, "sharpe": 1.0, "max_drawdown_pct": 40.0},
    )
    assert v["passed"] is False           # +0.05 Sharpe < θ AND −1pt DD < θ_dd


def test_hedging_constants_pinned():
    assert hedging_gate.TOL_CAGR_PCT == 5.0
    assert hedging_gate.THETA_SHARPE == 0.25
    assert hedging_gate.THETA_DD_PCT == 5.0
    assert hedging_gate.BOOTSTRAP_REPS == 1000
    assert hedging_gate.CI_LEVEL == 0.95
    assert hedging_gate.RNG_SEED == 42
    assert hedging_gate.TRADING_DAYS == 252


def test_improvement_significant_when_strat_dominates():
    # strat returns strictly less volatile + higher-mean than bench → improvement CI > 0
    strat = [0.01] * 60
    bench = [0.02, -0.02] * 30
    v = hedging_gate.improvement_significant(strat_returns=strat, bench_returns=bench)
    assert v["sharpe_improvement_significant"] is True


def test_improvement_not_significant_when_identical():
    series = [0.01, -0.005] * 40
    v = hedging_gate.improvement_significant(strat_returns=series, bench_returns=list(series))
    assert v["sharpe_improvement_significant"] is False


def test_improvement_insufficient_overlap():
    v = hedging_gate.improvement_significant(
        strat_returns=[0.01] * 10, bench_returns=[0.0] * 10
    )
    assert v["sharpe_improvement_significant"] is False
    assert v["reason"] == "insufficient_overlap"


# ── Fix 1: bootstrap phase-alignment in evaluate's strat/bench pairing ──────
#
# strat_returns is a calendar grid over [start, end] inclusive (day-0 = the
# start date, a 0/level entry). bench = _returns(closes) DROPS day-0, so the
# k-th bench return is realised on date start+k+1. A plain [:common] truncation
# pairs strat day i with bench day i+1 (a 1-day phase offset).
# _align_returns_by_date keys both by realisation date and intersects, so the
# SAME calendar day's returns are paired.

def test_align_returns_by_date_pairs_same_calendar_day():
    start = date(2024, 1, 1)
    # closes over [start, start+4] → 5 closes → 4 bench returns realised on
    # dates start+1..start+4.
    closes = [(datetime(2024, 1, d), 100.0 + d) for d in range(1, 6)]
    # strat grid over [start, start+4] inclusive → 5 entries; index 0 is the
    # start-date level entry (a sentinel that must be dropped by alignment).
    strat_returns = [0.0, 0.11, 0.12, 0.13, 0.14]
    strat_aligned, bench_aligned = hedging_gate._align_returns_by_date(
        strat_returns=strat_returns, strat_start=start, closes=closes,
    )
    # Common realisation dates are start+1..start+4 (bench has no return for the
    # start date; strat's start-date sentinel is excluded).
    assert len(strat_aligned) == len(bench_aligned) == 4
    # strat values paired are the post-sentinel entries (the same calendar days
    # the bench returns are realised on) — NOT the leading 0.0 sentinel.
    assert strat_aligned == [0.11, 0.12, 0.13, 0.14]
    # bench realised on start+1: 102/101-1, etc.
    assert bench_aligned[0] == 102.0 / 101.0 - 1.0
    assert bench_aligned[-1] == 105.0 / 104.0 - 1.0


def test_align_drops_strat_leading_sentinel_not_a_real_return():
    # The leading strat sentinel (start-date level) must never pair with a bench
    # return — under the OLD [:common] offset it would have been paired with
    # bench[0], smearing a fake same-day relationship.
    start = date(2024, 6, 1)
    closes = [(datetime(2024, 6, d), 200.0) for d in range(1, 4)]  # flat → 0 bench
    strat_returns = [99.0, 0.01, 0.02]  # 99.0 is the bogus sentinel
    strat_aligned, bench_aligned = hedging_gate._align_returns_by_date(
        strat_returns=strat_returns, strat_start=start, closes=closes,
    )
    assert 99.0 not in strat_aligned
    assert strat_aligned == [0.01, 0.02]


def test_phase_offset_would_misalign_lagged_series_old_bug():
    """Demonstrates the OLD bug's failure mode at the alignment layer.

    Build a strat grid that is the bench return series shifted LATER by one day
    (strat day i = bench day i-1). The OLD ``[:common]`` truncation paired strat
    index j with bench index j, i.e. strat(start+1+j-shift) with bench(start+1+j)
    — accidentally re-aligning the lag so a lagged copy looked same-day-paired.

    The date-keyed alignment instead pairs by CALENDAR date, so the lagged strat
    value for date D is correctly paired with the bench value for date D (a
    genuinely different value), exposing the lag rather than hiding it.
    """
    from datetime import timedelta as _td
    start = date(2024, 1, 1)
    n_closes = 10
    closes = [(datetime(2024, 1, 1) + _td(days=d), 100.0 + d * d) for d in range(n_closes)]
    # bench returns realised on dates start+1..start+9
    bench_by_date = {}
    for (_, prev_px), (cd, cur_px) in zip(closes, closes[1:]):
        bench_by_date[cd.date()] = cur_px / prev_px - 1.0
    bench_dates = sorted(bench_by_date)
    # strat grid keyed by date: day D carries the bench return of D-1 (a lag).
    strat_by_date = {start: 0.0}
    for i, cd in enumerate(bench_dates):
        # strat value on date cd = bench value of the PRIOR bench date (lag).
        strat_by_date[cd] = bench_by_date[bench_dates[i - 1]] if i > 0 else 0.123
    # serialise strat into the calendar-grid list evaluate passes.
    strat_returns = [strat_by_date[start + _td(days=k)] for k in range(n_closes)]

    strat_aligned, bench_aligned = hedging_gate._align_returns_by_date(
        strat_returns=strat_returns, strat_start=start, closes=closes,
    )
    # Correct alignment: for each shared date the strat value is the LAGGED one,
    # NOT equal to the same-date bench → the lag is exposed (series differ).
    assert strat_aligned != bench_aligned
    # And specifically the first shared date pairs strat's bogus 0.123 sentinel
    # with bench's real first return — provably a different (lagged) pairing than
    # the old offset, which would have dropped this misalignment.
    assert strat_aligned[0] == 0.123


def test_identical_same_day_strat_and_bench_not_significant():
    """A strat whose per-day returns equal bench's per-day returns (same
    calendar day) must NOT be significant: the paired Sharpe difference is
    exactly 0 on every resample → CI straddles 0.

    We feed bench's EXACT reconstructed returns back as the strat post-sentinel
    grid so there is no float-noise drift between the two paired series.
    """
    from datetime import timedelta as _td
    start = date(2024, 1, 1)
    n_closes = 80
    px = 100.0
    closes = [(datetime(2024, 1, 1), px)]
    for d in range(1, n_closes):
        r = 0.02 if d % 2 == 0 else -0.018
        px *= 1.0 + r
        closes.append((datetime(2024, 1, 1) + _td(days=d), px))
    # bench returns exactly as the helper reconstructs them.
    bench_by_date = {}
    for (_, prev_px), (cd, cur_px) in zip(closes, closes[1:]):
        bench_by_date[cd.date()] = cur_px / prev_px - 1.0
    # strat grid: leading sentinel + the SAME reconstructed bench values keyed to
    # the SAME calendar dates → after alignment strat == bench bit-for-bit.
    strat_returns = [0.0] + [bench_by_date[start + _td(days=k)] for k in range(1, n_closes)]
    strat_aligned, bench_aligned = hedging_gate._align_returns_by_date(
        strat_returns=strat_returns, strat_start=start, closes=closes,
    )
    assert strat_aligned == bench_aligned  # bit-identical, same calendar day
    v = hedging_gate.improvement_significant(
        strat_returns=strat_aligned, bench_returns=bench_aligned,
    )
    assert v["sharpe_improvement_significant"] is False


def test_same_day_real_improvement_is_significant():
    """A strat that genuinely dominates bench on the SAME calendar day (lower
    vol, higher mean) IS significant after phase-alignment — the fix doesn't
    suppress real edge, only corrects the pairing."""
    from datetime import timedelta as _td
    start = date(2024, 1, 1)
    n_closes = 80
    px = 100.0
    closes = [(datetime(2024, 1, 1), px)]
    for d in range(1, n_closes):
        r = 0.03 if d % 2 == 0 else -0.029  # choppy bench, near-zero drift
        px *= 1.0 + r
        closes.append((datetime(2024, 1, 1) + _td(days=d), px))
    # strat: steady small positive same-day returns → much higher Sharpe.
    strat_returns = [0.0] + [0.006 for _ in range(1, n_closes)]
    strat_aligned, bench_aligned = hedging_gate._align_returns_by_date(
        strat_returns=strat_returns, strat_start=start, closes=closes,
    )
    v = hedging_gate.improvement_significant(
        strat_returns=strat_aligned, bench_returns=bench_aligned,
    )
    assert v["sharpe_improvement_significant"] is True


def test_improvement_deterministic_same_seed():
    strat = [0.01, 0.012, -0.003] * 20
    bench = [0.02, -0.02, 0.005] * 20
    a = hedging_gate.improvement_significant(strat_returns=strat, bench_returns=bench)
    b = hedging_gate.improvement_significant(strat_returns=strat, bench_returns=bench)
    assert a["ci_low"] == b["ci_low"]
    assert a["ci_high"] == b["ci_high"]
