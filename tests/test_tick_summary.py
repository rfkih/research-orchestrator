"""Unit tests for ``services/tick_summary.compute_tick_summary``.

The runner agent reads ``summary.next_action`` to decide whether to keep
ticking. A bug here would either spin the runner forever (missing a
terminal state) or stop it early (missing edge that needed another tick).
Tests cover every branch in the decision tree.
"""

from __future__ import annotations

from orchestrator.services.tick_summary import compute_tick_summary


# ── outcome=empty_queue ──────────────────────────────────────────────


def test_empty_queue_is_empty_queue() -> None:
    out = compute_tick_summary(
        outcome="empty_queue",
        statistical_verdict=None,
        verdict=None,
        next_actions=None,
    )
    assert out["next_action"] == "EMPTY_QUEUE"
    assert "POST /queue" in out["decision_hint"]


# ── outcome=sweep_exhausted ──────────────────────────────────────────


def test_sweep_exhausted_is_pivot() -> None:
    out = compute_tick_summary(
        outcome="sweep_exhausted",
        statistical_verdict=None,
        verdict=None,
        next_actions=[{"kind": "read_doc"}],
    )
    assert out["next_action"] == "PIVOT"
    assert "axis" in out["decision_hint"] or "archetype" in out["decision_hint"]


# ── outcome=iterated, verdict=PASS ───────────────────────────────────


def test_iterated_pass_is_graduate() -> None:
    out = compute_tick_summary(
        outcome="iterated",
        statistical_verdict="SIGNIFICANT_EDGE",
        verdict="PASS",
        next_actions=[{"kind": "call", "method": "POST", "path": "/walk-forward"}],
        pf=1.45,
        n_trades=132,
    )
    assert out["next_action"] == "GRADUATE"
    assert "PF=1.45" in out["verdict_line"]
    assert "n=132" in out["verdict_line"]
    assert "graduation" in out["decision_hint"].lower()


def test_iterated_pass_without_metrics_still_graduates() -> None:
    out = compute_tick_summary(
        outcome="iterated",
        statistical_verdict="SIGNIFICANT_EDGE",
        verdict="PASS",
        next_actions=[],
    )
    assert out["next_action"] == "GRADUATE"


# ── outcome=iterated, verdict=DISCARD ────────────────────────────────


def test_iterated_discard_is_pivot() -> None:
    out = compute_tick_summary(
        outcome="iterated",
        statistical_verdict="NO_EDGE",
        verdict="DISCARD",
        next_actions=[],
        pf=0.71,
        n_trades=120,
    )
    assert out["next_action"] == "PIVOT"
    assert "DISCARD" in out["verdict_line"]


# ── outcome=iterated, SIG_EDGE but verdict=ITERATE (V60 miss) ───────


def test_sig_edge_without_econ_gate_continues() -> None:
    out = compute_tick_summary(
        outcome="iterated",
        statistical_verdict="SIGNIFICANT_EDGE",
        verdict="ITERATE",
        next_actions=[],
        pf=1.32,
        n_trades=158,
    )
    assert out["next_action"] == "CONTINUE"
    assert "econ" in out["verdict_line"].lower()


# ── outcome=iterated, NO_EDGE / INSUF ────────────────────────────────


def test_no_edge_continues() -> None:
    out = compute_tick_summary(
        outcome="iterated",
        statistical_verdict="NO_EDGE",
        verdict="ITERATE",
        next_actions=[],
        pf=0.94,
        n_trades=140,
    )
    assert out["next_action"] == "CONTINUE"


def test_insuf_evidence_continues() -> None:
    out = compute_tick_summary(
        outcome="iterated",
        statistical_verdict="INSUFFICIENT_EVIDENCE",
        verdict="ITERATE",
        next_actions=[],
        pf=1.10,
        n_trades=68,
    )
    assert out["next_action"] == "CONTINUE"
    assert "INSUF" in out["verdict_line"]


# ── infra-failure branches ───────────────────────────────────────────


def test_infra_with_retry_hint_waits() -> None:
    out = compute_tick_summary(
        outcome="failed",
        statistical_verdict=None,
        verdict=None,
        next_actions=[{"kind": "retry", "wait_s": 30.0}],
    )
    assert out["next_action"] == "WAIT"


def test_infra_without_retry_escalates() -> None:
    out = compute_tick_summary(
        outcome="failed",
        statistical_verdict=None,
        verdict=None,
        next_actions=[{"kind": "contact_human"}],
    )
    assert out["next_action"] == "INFRA_FAIL"


def test_unknown_outcome_treated_as_infra() -> None:
    # Any outcome other than the three known ones routes through the
    # infra branch so the runner never silently treats it as CONTINUE.
    out = compute_tick_summary(
        outcome="some_new_outcome",
        statistical_verdict=None,
        verdict=None,
        next_actions=None,
    )
    assert out["next_action"] == "INFRA_FAIL"


# ── catch-all (iterated but verdict pair unknown) ────────────────────


def test_iterated_unknown_verdict_continues() -> None:
    out = compute_tick_summary(
        outcome="iterated",
        statistical_verdict=None,
        verdict=None,
        next_actions=[],
    )
    assert out["next_action"] == "CONTINUE"


# ── next_action value is one of the documented six ──────────────────


def test_next_action_values_are_pinned() -> None:
    valid = {"CONTINUE", "GRADUATE", "PIVOT", "EMPTY_QUEUE", "WAIT", "INFRA_FAIL"}
    cases = [
        ("empty_queue", None, None, None),
        ("sweep_exhausted", None, None, None),
        ("iterated", "SIGNIFICANT_EDGE", "PASS", None),
        ("iterated", "SIGNIFICANT_EDGE", "ITERATE", None),
        ("iterated", "NO_EDGE", "ITERATE", None),
        ("iterated", "INSUFFICIENT_EVIDENCE", "ITERATE", None),
        ("iterated", "NO_EDGE", "DISCARD", None),
        ("failed", None, None, [{"kind": "retry", "wait_s": 5.0}]),
        ("failed", None, None, [{"kind": "contact_human"}]),
    ]
    for outcome, sv, v, na in cases:
        out = compute_tick_summary(
            outcome=outcome, statistical_verdict=sv, verdict=v, next_actions=na
        )
        assert out["next_action"] in valid, f"unexpected {out} for {outcome}/{sv}/{v}"
