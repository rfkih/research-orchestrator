from datetime import date

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


def test_improvement_deterministic_same_seed():
    strat = [0.01, 0.012, -0.003] * 20
    bench = [0.02, -0.02, 0.005] * 20
    a = hedging_gate.improvement_significant(strat_returns=strat, bench_returns=bench)
    b = hedging_gate.improvement_significant(strat_returns=strat, bench_returns=bench)
    assert a["ci_low"] == b["ci_low"]
    assert a["ci_high"] == b["ci_high"]
