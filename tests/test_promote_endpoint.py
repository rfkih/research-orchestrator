"""Endpoint-level wiring tests for POST /models/{id}/promote gauntlet re-check.

Drives ``promote_model`` directly with a mocked DB (acquire + transaction
+ fetchrow) so the WIRING is covered — that the gate actually returns 409
on failing metrics, that override_gauntlet bypasses AND persists the
verdict, that a good model passes, and that non-deployment targets
(retired) are never gated. The pure-function gate logic lives in
test_promotion_gate.py; this file guards against a mis-wired endpoint.
"""
from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import pytest
from fastapi import HTTPException

from orchestrator.api.models import PromoteRequest, promote_model

# 6b3b12e4's real metrics (fails median + transferability).
FAIL_METRICS = {
    "auc": 0.6094,
    "walk_forward": {
        "primary_metric": "auc", "primary_mean": 0.5338,
        "primary_median": 0.5109, "primary_std": 0.1142,
        "metric_means": {"ci_max_abs_diff": 0.2949},
    },
}
GOOD_METRICS = {
    "saved_booster": {"auc": 0.6},
    "walk_forward": {
        "primary_metric": "auc", "primary_mean": 0.60,
        "primary_median": 0.58, "primary_std": 0.08,
        "metric_means": {"ci_max_abs_diff": 0.10},
    },
}


def _run(coro):
    return asyncio.run(coro)


class _Tx:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *exc):
        return None


class _FakeConn:
    def __init__(self, model_row: dict[str, Any]):
        self.model_row = model_row
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def transaction(self):
        return _Tx()

    async def fetchrow(self, sql: str, *params: Any):
        self.calls.append((sql, params))
        if "UPDATE model_registry" in sql:
            # mirror update_status RETURNING shape
            return {
                "id": self.model_row["id"], "status": params[1],
                "version": self.model_row["version"], "promoted_by": params[2],
                "promoted_at": None, "reviewer_verdict": params[3],
                "reviewer_run_id": params[4], "updated_time": None,
            }
        return self.model_row  # get_model_by_id


class _Acquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return None


class _Db:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return _Acquire(self._conn)


class _NS:
    pass


def _request(db):
    req = _NS()
    req.app = _NS()
    req.app.state = _NS()
    req.app.state.db = db
    req.app.state.idempotency = None  # unused: idempotency_key is None
    req.state = _NS()
    req.state.agent_name = "test"
    req.state.idempotency_key = None
    return req


def _promote(current_status: str, metrics: dict, target: str, **body_kw):
    row = {
        "id": uuid4(), "status": current_status, "version": 1,
        "metrics": metrics, "promoted_at": None, "promoted_by": None,
    }
    conn = _FakeConn(row)
    req = _request(_Db(conn))
    body = PromoteRequest(target_status=target, **body_kw)
    return _run(promote_model(req, row["id"], body)), conn


def _update_call(conn: _FakeConn):
    return next(c for c in conn.calls if "UPDATE model_registry" in c[0])


# ── The wiring ─────────────────────────────────────────────────────────────


def test_promote_to_staged_blocked_when_metrics_fail():
    with pytest.raises(HTTPException) as ei:
        _promote("trained", FAIL_METRICS, "staged")
    assert ei.value.status_code == 409
    assert "gauntlet" in ei.value.detail.lower()
    assert "median" in ei.value.detail


def test_promote_override_bypasses_and_persists_verdict():
    result, conn = _promote("trained", FAIL_METRICS, "staged", override_gauntlet=True)
    assert result["status"] == "staged"
    assert result["transition_applied"] is True
    # reviewer_verdict (param index 3 of update_status) stamped as override
    assert _update_call(conn)[1][3] == "gauntlet_override"


def test_promote_override_keeps_caller_reviewer_verdict():
    _, conn = _promote(
        "trained", FAIL_METRICS, "staged",
        override_gauntlet=True, reviewer_verdict="manual_ok",
    )
    assert _update_call(conn)[1][3] == "manual_ok"


def test_promote_good_metrics_passes_gate():
    result, conn = _promote("trained", GOOD_METRICS, "staged")
    assert result["status"] == "staged"
    assert _update_call(conn)[1][3] is None  # no override sentinel


def test_promote_to_retired_is_not_gated():
    # retired is a safe downgrade — failing metrics must NOT block it.
    result, _ = _promote("live", FAIL_METRICS, "retired")
    assert result["status"] == "retired"


def test_promote_live_with_failing_metrics_blocked():
    # the exact incident path: cooling_down -> live on the bad model.
    with pytest.raises(HTTPException) as ei:
        _promote("cooling_down", FAIL_METRICS, "live")
    assert ei.value.status_code == 409
