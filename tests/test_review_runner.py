"""Tests for ``services/review_runner.py`` — the auto-reviewer.

The endpoint integration is covered indirectly: drive the service
layer with a fake asyncpg.Connection that records queries and returns
canned rows, then assert the verdict shape that gets persisted.

These tests pin three contracts the researcher relies on:

  1. Plan path: pre_registration + mechanism + axis_not_recently_failed
     run against the right artifacts.
  2. Graduation path: n_trades + PF CI + DSR + cost + regime + portfolio
     + robustness run against the iteration row.
  3. Missing request row → 404 envelope, not a 500.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

import pytest

from orchestrator.errors import OrchestratorError
from orchestrator.services.review_runner import (
    AUTO_REVIEWER_AGENT,
    auto_run_checklist,
)


# ── Fake asyncpg.Connection ───────────────────────────────────────────


class FakeConn:
    """Minimal asyncpg.Connection stand-in.

    Routes ``fetchrow`` / ``fetch`` by inspecting the query string for a
    discriminating substring, returning whichever fixture row(s) the
    test seeded. Strictly enough surface for the auto_run_checklist
    service to work; anything new the service starts calling will fail
    here loudly, which is the point.
    """

    def __init__(self) -> None:
        self.request_row: dict[str, Any] | None = None
        self.hypothesis_row: dict[str, Any] | None = None
        self.prior_outcome_rows: list[dict[str, Any]] = []
        self.iteration_row: dict[str, Any] | None = None
        self.history_rows: list[dict[str, Any]] = []
        self.recorded_inserts: list[dict[str, Any]] = []

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        q = query.strip()
        if "FROM research_journal" in q and "IDEA_BACKLOG" in q and "review_request" in (
            args[0] if args else ""
        ):
            return self.request_row
        if "FROM research_journal" in q and "journal_id = $1" in q:
            return self.hypothesis_row
        if "FROM research_iteration_log" in q and "iteration_id = $1" in q:
            return self.iteration_row
        raise AssertionError(f"Unexpected fetchrow query: {q!r}")

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        q = query.strip()
        if "FROM research_journal" in q and "STRATEGY_OUTCOME" in (args or [None])[0:1] or (
            args and len(args) >= 1 and args[0] == "STRATEGY_OUTCOME"
        ):
            return self.prior_outcome_rows
        if "FROM research_iteration_log" in q and "ORDER BY created_time DESC" in q:
            return self.history_rows
        if "FROM research_journal" in q:
            return self.prior_outcome_rows
        raise AssertionError(f"Unexpected fetch query: {q!r}")


# ── Helpers ───────────────────────────────────────────────────────────


def _plan_request_row(
    *, target_id: str, hypothesis_id: str, strategy_code: str, axis_names: list[str],
    created_time: datetime,
) -> dict[str, Any]:
    return {
        "journal_id": uuid4(),
        "strategy_code": strategy_code,
        "structured_data": {
            "kind": "review_request",
            "target_id": target_id,
            "target_kind": "plan",
            "request_payload": {
                "strategy_code": strategy_code,
                "axis_names": axis_names,
                "hypothesis_id": hypothesis_id,
            },
            "requested_by": "quant-researcher",
        },
        "created_time": created_time,
    }


def _graduation_request_row(
    *, target_id: str, iteration_id: str, strategy_code: str,
    created_time: datetime,
) -> dict[str, Any]:
    return {
        "journal_id": uuid4(),
        "strategy_code": strategy_code,
        "structured_data": {
            "kind": "review_request",
            "target_id": target_id,
            "target_kind": "graduation",
            "request_payload": {
                "strategy_code": strategy_code,
                "iteration_id": iteration_id,
            },
            "requested_by": "quant-researcher",
        },
        "created_time": created_time,
    }


# ── Plan path ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_plan_path_approved_when_all_checks_pass() -> None:
    plan_ts = datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc)
    hyp_id = uuid4()
    conn = FakeConn()
    conn.request_row = _plan_request_row(
        target_id="plan:MMR:hash:" + str(hyp_id),
        hypothesis_id=str(hyp_id),
        strategy_code="MMR",
        axis_names=["ATR_EXT", "RSI_EXT"],
        created_time=plan_ts,
    )
    conn.hypothesis_row = {
        "journal_id": hyp_id,
        "entry_type": "HYPOTHESIS",
        "content": (
            "Tighter ATR + RSI extremes raise MFE/MAE because overshoots in "
            "low-vol regimes have higher snap-back probability than at "
            "default thresholds."
        ),
        "created_time": plan_ts - timedelta(hours=1),
    }
    conn.prior_outcome_rows = []  # No recent failures.

    result = await auto_run_checklist(
        conn, target_id=conn.request_row["structured_data"]["target_id"],
        target_kind="plan",
    )

    assert result["verdict"] == "APPROVED"
    assert result["n_blocker_fails"] == 0
    assert result["n_warning_fails"] == 0
    assert result["target_id"] == conn.request_row["structured_data"]["target_id"]
    assert result["motivating_request_id"] == str(conn.request_row["journal_id"])
    assert result["strategy_code"] == "MMR"


@pytest.mark.asyncio
async def test_plan_path_rejected_when_hypothesis_missing() -> None:
    plan_ts = datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc)
    hyp_id = uuid4()
    conn = FakeConn()
    conn.request_row = _plan_request_row(
        target_id="plan:MMR:hash:" + str(hyp_id),
        hypothesis_id=str(hyp_id),
        strategy_code="MMR",
        axis_names=["ATR_EXT"],
        created_time=plan_ts,
    )
    conn.hypothesis_row = None  # blocker fail on pre_registration
    conn.prior_outcome_rows = []

    result = await auto_run_checklist(
        conn, target_id=conn.request_row["structured_data"]["target_id"],
        target_kind="plan",
    )
    assert result["verdict"] == "REJECTED"
    assert result["n_blocker_fails"] >= 1
    pre_reg = next(c for c in result["checks"] if c["check_name"] == "pre_registration")
    assert pre_reg["passed"] is False


@pytest.mark.asyncio
async def test_plan_path_recent_outcome_demotes_to_conditional() -> None:
    plan_ts = datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc)
    hyp_id = uuid4()
    conn = FakeConn()
    conn.request_row = _plan_request_row(
        target_id="plan:MMR:hash:" + str(hyp_id),
        hypothesis_id=str(hyp_id),
        strategy_code="MMR",
        axis_names=["ATR_EXT", "RSI_EXT"],
        created_time=plan_ts,
    )
    conn.hypothesis_row = {
        "journal_id": hyp_id,
        "entry_type": "HYPOTHESIS",
        "content": (
            "ATR + RSI extremes drive mean-reversion because overshoot "
            "probability rises in low-vol regimes."
        ),
        "created_time": plan_ts - timedelta(hours=2),
    }
    conn.prior_outcome_rows = [
        {
            "journal_id": uuid4(),
            "entry_type": "STRATEGY_OUTCOME",
            "title": "Outcome: MMR ATR/RSI sweep ended NO_EDGE",
            "structured_data": {
                "axis_names": ["ATR_EXT", "RSI_EXT"],
                "verdict": "NO_EDGE",
            },
            "created_time": plan_ts - timedelta(days=3),
        }
    ]

    result = await auto_run_checklist(
        conn, target_id=conn.request_row["structured_data"]["target_id"],
        target_kind="plan",
    )
    # Only the axis_not_recently_failed warning fires -> CONDITIONAL_APPROVAL.
    assert result["verdict"] == "CONDITIONAL_APPROVAL"
    assert result["n_warning_fails"] == 1


@pytest.mark.asyncio
async def test_missing_request_row_raises_404() -> None:
    conn = FakeConn()
    conn.request_row = None  # No matching journal row.

    with pytest.raises(OrchestratorError) as exc_info:
        await auto_run_checklist(conn, target_id="plan:UNKNOWN:x:y", target_kind="plan")

    err = exc_info.value
    assert err.status_code == 404
    assert err.envelope.error_code == "review_request_not_found"


@pytest.mark.asyncio
async def test_bad_target_kind_raises_400() -> None:
    conn = FakeConn()
    with pytest.raises(OrchestratorError) as exc_info:
        await auto_run_checklist(conn, target_id="x", target_kind="garbage")
    assert exc_info.value.status_code == 400
    assert exc_info.value.envelope.error_code == "bad_target_kind"


# ── Graduation path ───────────────────────────────────────────────────


def _passing_iteration_row(*, iteration_id: UUID, strategy_code: str) -> dict[str, Any]:
    """Iteration with metrics that clear every graduation check."""
    return {
        "iteration_id": iteration_id,
        "strategy_code": strategy_code,
        "iteration_number": 5,
        "backtest_run_id": uuid4(),
        "params_snapshot": {"atr_ext": 2.5, "rsi_ext": 30, "stop": 0.02},
        "metrics_snapshot": {
            "profit_factor": "1.8",
            "analysis": {
                "n_trades": 150,
                "dsr": 0.97,
                "dsr_n_trials": 25,
                "slippage_haircut_pnl": {"+50bps": 12.5, "+20bps": 18.0},
                "regimes": {
                    "by_trend_regime": {
                        "TRENDING_UP": {"n": 60, "pnl": 8.0},
                        "TRENDING_DOWN": {"n": 40, "pnl": 5.0},
                        "CHOP": {"n": 50, "pnl": 2.0},
                    }
                },
            },
            "portfolio_corr": {"applied": True, "max_abs_corr": 0.25},
        },
        "confidence_intervals": {"pf_95": {"low": 1.25, "high": 2.40}},
        "verdict": "PASS",
        "statistical_verdict": "SIGNIFICANT_EDGE",
        "sample_size_adequate": True,
    }


@pytest.mark.asyncio
async def test_graduation_path_approved_on_clean_iteration() -> None:
    iteration_id = uuid4()
    conn = FakeConn()
    target_id = f"graduation:{iteration_id}"
    conn.request_row = _graduation_request_row(
        target_id=target_id,
        iteration_id=str(iteration_id),
        strategy_code="MMR",
        created_time=datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc),
    )
    iter_row = _passing_iteration_row(iteration_id=iteration_id, strategy_code="MMR")
    conn.iteration_row = iter_row
    # Sweep history with a Hamming-1 neighbour at PF 1.6 (within 30% of 1.8 optimum).
    conn.history_rows = [
        {**iter_row, "iteration_id": uuid4()},  # the candidate itself
        {
            "iteration_id": uuid4(),
            "strategy_code": "MMR",
            "params_snapshot": {"atr_ext": 2.5, "rsi_ext": 30, "stop": 0.025},
            "metrics_snapshot": {"profit_factor": "1.6"},
        },
    ]

    result = await auto_run_checklist(conn, target_id=target_id, target_kind="graduation")
    assert result["verdict"] == "APPROVED", result
    assert result["n_blocker_fails"] == 0


@pytest.mark.asyncio
async def test_graduation_path_rejected_on_low_dsr() -> None:
    iteration_id = uuid4()
    conn = FakeConn()
    target_id = f"graduation:{iteration_id}"
    conn.request_row = _graduation_request_row(
        target_id=target_id,
        iteration_id=str(iteration_id),
        strategy_code="MMR",
        created_time=datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc),
    )
    bad = _passing_iteration_row(iteration_id=iteration_id, strategy_code="MMR")
    bad["metrics_snapshot"]["analysis"]["dsr"] = 0.80
    conn.iteration_row = bad
    conn.history_rows = [bad]

    result = await auto_run_checklist(conn, target_id=target_id, target_kind="graduation")
    assert result["verdict"] == "REJECTED"
    dsr_check = next(c for c in result["checks"] if c["check_name"] == "dsr_threshold")
    assert dsr_check["passed"] is False


@pytest.mark.asyncio
async def test_graduation_path_404_when_iteration_missing() -> None:
    iteration_id = uuid4()
    conn = FakeConn()
    target_id = f"graduation:{iteration_id}"
    conn.request_row = _graduation_request_row(
        target_id=target_id,
        iteration_id=str(iteration_id),
        strategy_code="MMR",
        created_time=datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc),
    )
    conn.iteration_row = None

    with pytest.raises(OrchestratorError) as exc_info:
        await auto_run_checklist(conn, target_id=target_id, target_kind="graduation")
    assert exc_info.value.status_code == 404
    assert exc_info.value.envelope.error_code == "iteration_not_found"


# ── Identity / pin tests ──────────────────────────────────────────────


def test_auto_reviewer_agent_name_is_pinned() -> None:
    # The audit trail's "who reviewed" stamp must stay stable so historical
    # rows can be filtered consistently.
    assert AUTO_REVIEWER_AGENT == "research-orchestrator-auto-reviewer"
