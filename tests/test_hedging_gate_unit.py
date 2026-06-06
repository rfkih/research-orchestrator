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
