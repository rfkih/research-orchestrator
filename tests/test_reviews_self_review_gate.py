"""Regression tests for the self-approvable review-gate fix (POST /reviews).

Pre-fix, the same self-asserted agent that requested a review could post its
own APPROVED verdict and clear the /queue gate the paired-review loop exists to
enforce. The handler now rejects a verdict whose reviewer equals the latest
request's requester with 409 ``self_review_forbidden``. The deterministic
/reviews/auto-run-checklist path posts as AUTO_REVIEWER_AGENT and bypasses this
handler, so it is unaffected.
"""
from __future__ import annotations

import pytest

from orchestrator.api import reviews as reviews_module

_HEADERS_TOKEN = {"X-Orch-Token": "test-token"}


def _verdict_body() -> dict:
    return {
        "target_id": "plan:LSR:abc123:hyp-1",
        "target_kind": "plan",
        "verdict": "APPROVED",
        "summary_reason": "looks good",
        "strategy_code": "LSR",
    }


def test_self_review_is_rejected_409(client, monkeypatch) -> None:
    async def _requester(_conn, _target_id):
        return "quant-researcher"

    monkeypatch.setattr(
        reviews_module.reviews_repo, "fetch_latest_request_requester", _requester
    )

    resp = client.post(
        "/reviews",
        json=_verdict_body(),
        headers={**_HEADERS_TOKEN, "X-Agent-Name": "quant-researcher"},
    )
    assert resp.status_code == 409
    assert resp.json()["error_code"] == "self_review_forbidden"


def test_distinct_reviewer_is_allowed(client, monkeypatch) -> None:
    async def _requester(_conn, _target_id):
        return "quant-researcher"

    async def _insert(*_args, **_kwargs):
        return "journal-1"

    async def _log(*_args, **_kwargs):
        return None

    monkeypatch.setattr(
        reviews_module.reviews_repo, "fetch_latest_request_requester", _requester
    )
    monkeypatch.setattr(reviews_module.reviews_repo, "insert_review_verdict", _insert)
    monkeypatch.setattr(reviews_module, "log_activity", _log)

    resp = client.post(
        "/reviews",
        json=_verdict_body(),
        headers={**_HEADERS_TOKEN, "X-Agent-Name": "quant-reviewer"},
    )
    assert resp.status_code == 201
    assert resp.json()["verdict"] == "APPROVED"


def test_no_prior_request_is_allowed(client, monkeypatch) -> None:
    # No request exists for the target → requester unknown → cannot be a
    # self-review, so the verdict proceeds (the separate
    # request-existence validation is out of scope for this gate).
    async def _requester(_conn, _target_id):
        return None

    async def _insert(*_args, **_kwargs):
        return "journal-2"

    async def _log(*_args, **_kwargs):
        return None

    monkeypatch.setattr(
        reviews_module.reviews_repo, "fetch_latest_request_requester", _requester
    )
    monkeypatch.setattr(reviews_module.reviews_repo, "insert_review_verdict", _insert)
    monkeypatch.setattr(reviews_module, "log_activity", _log)

    resp = client.post(
        "/reviews",
        json=_verdict_body(),
        headers={**_HEADERS_TOKEN, "X-Agent-Name": "quant-researcher"},
    )
    assert resp.status_code == 201
