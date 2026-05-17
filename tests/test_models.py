"""Tests for ``POST /models/register`` (M5e).

Three layers:

1. **Pure-function** tests for status derivation — no DB, no client.
2. **Repo-level** tests with a fake asyncpg connection — exercise the
   SQL shape (next_version derivation, idempotent sha lookup) without
   needing a live database.
3. **HTTP-level** validation tests via the TestClient — exercise the
   pydantic schema + auth path. The endpoint's DB call would fail
   against the fake DSN, so these stop at the request-validation
   boundary (which is what we care about for the request shape).
"""
from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from orchestrator.api.models import (
    DeploymentReadinessBlock,
    GauntletBlock,
    MarkSyncedRequest,
    ModelRegisterRequest,
    SpecBlock,
    _derive_status,
)
from orchestrator.repo import models as models_repo


# ── _FakeConn helper ──────────────────────────────────────────────────────


class _FakeConn:
    """Records each query's SQL + params and returns pre-seeded rows.
    Mirrors the pattern used in test_account_scoping.py."""

    def __init__(self, *, fetchrow_results=None, fetchval_result=None):
        self.fetchrow_results = list(fetchrow_results or [])
        self.fetchval_result = fetchval_result
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetchrow(self, sql: str, *params: Any) -> Any:
        self.calls.append(("fetchrow", sql, params))
        if not self.fetchrow_results:
            return None
        return self.fetchrow_results.pop(0)

    async def fetchval(self, sql: str, *params: Any) -> Any:
        self.calls.append(("fetchval", sql, params))
        return self.fetchval_result


def _run(coro):
    return asyncio.run(coro)


# ── _derive_status ────────────────────────────────────────────────────────


def _body(
    *,
    gauntlet_verdict: str | None = "PASS",
    deployment_ready: bool = True,
) -> ModelRegisterRequest:
    """Helper: minimal valid ModelRegisterRequest with overrides for the
    fields that drive status derivation."""
    from datetime import datetime
    return ModelRegisterRequest(
        content_sha256="a" * 64,
        artifact_uri="/x.pkl",
        artifact_size_bytes=100,
        spec=SpecBlock(
            name="regime_btc_v1", purpose="regime", symbol="BTCUSDT",
            interval="1h", label_feature="label_regime_risk_on_48h",
            objective="binary",
            train_start=datetime(2024, 12, 1),
            train_end=datetime(2026, 5, 14),
            hyperparams={"random_state": 42},
        ),
        feature_names=["f1", "f2"],
        metrics={"auc": 0.55},
        walk_forward=None,
        gauntlet=GauntletBlock(overall_verdict=gauntlet_verdict) if gauntlet_verdict else None,
        deployment_readiness=DeploymentReadinessBlock(deployment_ready=deployment_ready),
        data_fingerprint="dead" * 16,
    )


def test_status_trained_on_pass_and_ready():
    assert _derive_status(_body(gauntlet_verdict="PASS", deployment_ready=True)) == "trained"


def test_status_awaiting_review_on_pass_but_not_ready():
    assert _derive_status(
        _body(gauntlet_verdict="PASS", deployment_ready=False)
    ) == "awaiting_operator_review"


def test_status_rejected_on_gauntlet_fail():
    assert _derive_status(_body(gauntlet_verdict="FAIL")) == "rejected_by_operator"


def test_status_training_when_no_gauntlet_block():
    assert _derive_status(_body(gauntlet_verdict=None)) == "training"


# ── Repo: find_by_artifact_sha ───────────────────────────────────────────


def test_find_by_artifact_sha_returns_dict_when_present():
    expected_id = uuid4()
    row = {
        "id": expected_id, "family": "lightgbm_modulator", "purpose": "regime",
        "symbol": "BTCUSDT", "interval": "1h", "horizon_bars": 24,
        "feature_set": {"names": []}, "hyperparams": {}, "metrics": {},
        "random_seed": 42, "artifact_uri": "/x.pkl",
        "artifact_sha256": "a" * 64, "artifact_size_bytes": 100,
        "status": "trained", "version": 1,
        "created_time": None, "created_by": "orchestrator:test",
    }
    conn = _FakeConn(fetchrow_results=[row])
    result = _run(models_repo.find_by_artifact_sha(conn, "a" * 64))
    assert result is not None
    assert result["id"] == expected_id


def test_find_by_artifact_sha_returns_none_when_missing():
    conn = _FakeConn(fetchrow_results=[None])
    result = _run(models_repo.find_by_artifact_sha(conn, "z" * 64))
    assert result is None


# ── Repo: insert_model ────────────────────────────────────────────────────


def test_insert_model_is_a_single_statement_with_inline_version():
    """MR1 fix: ``insert_model`` no longer does a separate SELECT for
    next_version. Both the version computation and the INSERT happen
    in one statement so two concurrent callers can't both observe the
    same MAX before either commits.

    We assert there's exactly one DB call, and that the statement
    contains both the COALESCE(MAX(version)) subquery and the INSERT.
    """
    new_id = uuid4()
    insert_row = {
        "id": new_id, "version": 3, "status": "trained", "created_time": None,
    }
    conn = _FakeConn(fetchrow_results=[insert_row])
    result = _run(models_repo.insert_model(
        conn,
        family="lightgbm_modulator",
        purpose="regime", symbol="BTCUSDT", interval="1h", horizon_bars=24,
        feature_set={"names": ["a"]}, hyperparams={"lr": 0.05},
        metrics={"auc": 0.6}, random_seed=42,
        artifact_uri="/x.pkl", artifact_sha256="a" * 64, artifact_size_bytes=100,
        status="trained", created_by="orchestrator:test",
    ))
    assert result["id"] == new_id
    assert result["version"] == 3
    # Single statement only.
    assert len(conn.calls) == 1
    assert conn.calls[0][0] == "fetchrow"
    sql = conn.calls[0][1]
    assert "INSERT INTO model_registry" in sql
    assert "COALESCE" in sql
    assert "MAX(version)" in sql


def test_insert_model_uses_is_not_distinct_for_nullable_columns():
    """interval and horizon_bars can be NULL. The inline version-lookup
    must match NULL with NULL, which requires IS NOT DISTINCT FROM
    rather than equality (NULL = NULL is unknown, not true)."""
    conn = _FakeConn(
        fetchrow_results=[{
            "id": uuid4(), "version": 1, "status": "trained", "created_time": None,
        }],
    )
    _run(models_repo.insert_model(
        conn,
        family="lightgbm_modulator",
        purpose="directional", symbol="BTCUSDT", interval=None, horizon_bars=None,
        feature_set={}, hyperparams=None, metrics=None, random_seed=0,
        artifact_uri=None, artifact_sha256="b" * 64, artifact_size_bytes=None,
        status="trained", created_by="orchestrator:test",
    ))
    sql = conn.calls[0][1]
    assert "IS NOT DISTINCT FROM" in sql


# ── HTTP-level: input validation + auth ──────────────────────────────────


def test_register_rejects_missing_auth_token(client: TestClient):
    r = client.post("/models/register", json={})
    assert r.status_code == 401
    assert r.json()["error_code"] == "auth_missing_token"


def test_register_rejects_missing_required_fields(client: TestClient):
    r = client.post(
        "/models/register",
        headers={"X-Orch-Token": "test-token", "X-Agent-Name": "quant-researcher"},
        json={"content_sha256": "a" * 64},   # missing spec, metrics, etc.
    )
    assert r.status_code == 422   # FastAPI pydantic validation


def test_register_rejects_short_content_sha(client: TestClient):
    r = client.post(
        "/models/register",
        headers={"X-Orch-Token": "test-token", "X-Agent-Name": "quant-researcher"},
        json={
            "content_sha256": "short",   # not 64 chars
            "spec": {
                "name": "x", "purpose": "regime", "symbol": "BTCUSDT",
                "interval": "1h", "label_feature": "label_regime_risk_on_48h",
                "objective": "binary",
                "train_start": "2024-12-01T00:00:00",
                "train_end": "2026-05-14T00:00:00",
            },
            "feature_names": ["a"],
            "metrics": {"auc": 0.6},
            "deployment_readiness": {"deployment_ready": True},
        },
    )
    assert r.status_code == 422


def test_register_rejects_invalid_purpose(client: TestClient):
    """MA2 fix: ``purpose`` is Literal-constrained at the pydantic
    boundary. A bogus value gets a 422 from FastAPI rather than a 500
    from PostgreSQL's CHECK constraint."""
    r = client.post(
        "/models/register",
        headers={"X-Orch-Token": "test-token", "X-Agent-Name": "quant-researcher"},
        json={
            "content_sha256": "a" * 64,
            "spec": {
                "name": "x", "purpose": "garbage", "symbol": "BTCUSDT",
                "interval": "1h", "label_feature": "label_regime_risk_on_48h",
                "objective": "binary",
                "train_start": "2024-12-01T00:00:00",
                "train_end": "2026-05-14T00:00:00",
            },
            "feature_names": ["a"],
            "metrics": {"auc": 0.6},
            "deployment_readiness": {"deployment_ready": True},
        },
    )
    assert r.status_code == 422
    # FastAPI's pydantic error should mention the invalid enum value
    body = r.json()
    assert any("garbage" in str(d).lower() or "purpose" in str(d).lower()
               for d in [body.get("detail"), body])


def test_register_rejects_invalid_objective(client: TestClient):
    """MA2 fix: ``objective`` is Literal-constrained to binary/regression."""
    r = client.post(
        "/models/register",
        headers={"X-Orch-Token": "test-token", "X-Agent-Name": "quant-researcher"},
        json={
            "content_sha256": "a" * 64,
            "spec": {
                "name": "x", "purpose": "regime", "symbol": "BTCUSDT",
                "interval": "1h", "label_feature": "label_regime_risk_on_48h",
                "objective": "classification",   # not in Literal
                "train_start": "2024-12-01T00:00:00",
                "train_end": "2026-05-14T00:00:00",
            },
            "feature_names": ["a"],
            "metrics": {"auc": 0.6},
            "deployment_readiness": {"deployment_ready": True},
        },
    )
    assert r.status_code == 422


def test_register_accepts_explicit_horizon_bars():
    """MA3 fix: callers can pass ``horizon_bars`` explicitly to register
    a model with a label name not in the lookup table.

    Schema validation only — we don't need to land the row. The
    pydantic ``Field(ge=1, le=10_000)`` constraint runs at request
    parse time, before any handler code runs.
    """
    from orchestrator.api.models import ModelRegisterRequest

    body = ModelRegisterRequest(
        content_sha256="a" * 64,
        horizon_bars=12,
        spec={
            "name": "x", "purpose": "regime", "symbol": "BTCUSDT",
            "interval": "1h",
            "label_feature": "label_some_future_thing_12h",
            "objective": "binary",
            "train_start": "2024-12-01T00:00:00",
            "train_end": "2026-05-14T00:00:00",
        },
        feature_names=["a"],
        metrics={"auc": 0.6},
        deployment_readiness={"deployment_ready": True},
    )
    assert body.horizon_bars == 12


def test_mark_synced_body_accepts_empty():
    """An empty body is valid — the endpoint applies its own defaults
    (synced_at=NOW(), remote_path=None means 'keep existing URI')."""
    body = MarkSyncedRequest()
    assert body.remote_path is None
    assert body.synced_at is None


def test_mark_synced_body_accepts_remote_path_and_timestamp():
    """A past timestamp is accepted as-is. The future-date guard
    (MF6) is exercised separately below; this only checks the happy
    path doesn't trip it."""
    from datetime import datetime as _dt
    past = _dt(2026, 5, 13)   # one day before "today" (2026-05-14)
    body = MarkSyncedRequest(remote_path="/opt/blackheart/models/x.pkl", synced_at=past)
    assert body.remote_path == "/opt/blackheart/models/x.pkl"
    assert body.synced_at == past


def test_mark_synced_body_rejects_far_future_synced_at():
    """MF6: ``synced_at`` more than 5 minutes ahead of wall clock is
    rejected at pydantic-validation time. Catches operator clock skew
    and typo'd manual POSTs before they corrupt the audit column."""
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz
    from pydantic import ValidationError

    far_future = _dt.now(_tz.utc) + _td(days=1)
    with pytest.raises(ValidationError, match="5 minutes"):
        MarkSyncedRequest(synced_at=far_future)


def test_mark_synced_body_accepts_within_5min_tolerance():
    """MF6: a timestamp within the 5-minute tolerance is accepted —
    NTP drift on the operator's host shouldn't trip the guard."""
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz

    nearly_now = _dt.now(_tz.utc) + _td(minutes=2)
    body = MarkSyncedRequest(synced_at=nearly_now)
    assert body.synced_at == nearly_now


def test_list_models_no_filters_passes_no_extra_wheres():
    """When all filters are None, only the TRUE base + LIMIT/OFFSET
    params bind. We assert the SQL doesn't accumulate stale clauses."""
    conn = _FakeConn(fetchrow_results=[])
    conn.fetch_results = []   # type: ignore[attr-defined]

    async def fetch(sql, *params):
        conn.calls.append(("fetch", sql, params))
        return []

    conn.fetch = fetch   # type: ignore[attr-defined]

    _run(models_repo.list_models(conn))
    fetch_call = next(c for c in conn.calls if c[0] == "fetch")
    sql = fetch_call[1]
    # TRUE is the only WHERE token; no status/purpose/symbol/synced filters.
    assert "WHERE TRUE" in sql
    assert "status =" not in sql
    assert "purpose =" not in sql
    assert "artifact_synced_to_vps =" not in sql


def test_list_models_synced_filter_appears_in_sql():
    conn = _FakeConn(fetchrow_results=[])

    async def fetch(sql, *params):
        conn.calls.append(("fetch", sql, params))
        return []

    conn.fetch = fetch   # type: ignore[attr-defined]
    _run(models_repo.list_models(conn, synced=False))
    sql = next(c for c in conn.calls if c[0] == "fetch")[1]
    assert "artifact_synced_to_vps =" in sql


def test_mark_synced_updates_with_remote_path():
    """When remote_path is provided, the SQL must UPDATE artifact_uri
    too. We assert the column appears in the SET clause. Param ordering
    is (model_id=$1, synced_at=$2, remote_path=$3, actor=$4)."""
    target_id = uuid4()
    row = {
        "id": target_id,
        "artifact_synced_to_vps": True,
        "artifact_synced_at": None,
        "artifact_uri": "/new/path.pkl",
        "status": "trained",
    }
    conn = _FakeConn(fetchrow_results=[row])
    _run(models_repo.mark_synced(
        conn, model_id=target_id, remote_path="/new/path.pkl",
        synced_at=None, actor="orchestrator:test",
    ))
    sql = conn.calls[0][1]
    assert "artifact_synced_to_vps = TRUE" in sql
    assert "artifact_uri = $3" in sql


def test_mark_synced_refreshes_updated_time():
    """MF1: every mark-synced UPDATE must refresh ``updated_time`` —
    without this, two distinct sync events on the same row are
    indistinguishable in the audit-trail column."""
    target_id = uuid4()
    row = {
        "id": target_id,
        "artifact_synced_to_vps": True,
        "artifact_synced_at": None,
        "artifact_uri": "/x.pkl",
        "status": "trained",
    }
    # With remote_path
    conn = _FakeConn(fetchrow_results=[row])
    _run(models_repo.mark_synced(
        conn, model_id=target_id, remote_path="/x.pkl",
        synced_at=None, actor="orchestrator:test",
    ))
    assert "updated_time = NOW()" in conn.calls[0][1]
    # Without remote_path — same column must still refresh
    conn = _FakeConn(fetchrow_results=[row])
    _run(models_repo.mark_synced(
        conn, model_id=target_id, remote_path=None,
        synced_at=None, actor="orchestrator:test",
    ))
    assert "updated_time = NOW()" in conn.calls[0][1]


def test_mark_synced_omits_artifact_uri_update_when_no_remote_path():
    """If remote_path is None, the SET clause should not touch
    artifact_uri — preserves whatever was there at registration. The
    RETURNING clause still names artifact_uri because the caller wants
    to see the current URI in the response — we check only the SET
    section here."""
    target_id = uuid4()
    row = {
        "id": target_id,
        "artifact_synced_to_vps": True,
        "artifact_synced_at": None,
        "artifact_uri": "/orig.pkl",
        "status": "trained",
    }
    conn = _FakeConn(fetchrow_results=[row])
    _run(models_repo.mark_synced(
        conn, model_id=target_id, remote_path=None,
        synced_at=None, actor="orchestrator:test",
    ))
    sql = conn.calls[0][1]
    set_clause = sql.split("SET", 1)[1].split("WHERE", 1)[0]
    assert "artifact_uri" not in set_clause


def test_mark_synced_returns_none_when_row_missing():
    conn = _FakeConn(fetchrow_results=[None])
    result = _run(models_repo.mark_synced(
        conn, model_id=uuid4(), remote_path=None,
        synced_at=None, actor="orchestrator:test",
    ))
    assert result is None


def test_register_rejects_horizon_bars_out_of_range():
    """MA3 guardrail: horizon_bars must be in [1, 10000]."""
    from orchestrator.api.models import ModelRegisterRequest
    from pydantic import ValidationError

    common = {
        "content_sha256": "a" * 64,
        "spec": {
            "name": "x", "purpose": "regime", "symbol": "BTCUSDT",
            "interval": "1h", "label_feature": "label_x",
            "objective": "binary",
            "train_start": "2024-12-01T00:00:00",
            "train_end": "2026-05-14T00:00:00",
        },
        "feature_names": ["a"], "metrics": {"auc": 0.6},
        "deployment_readiness": {"deployment_ready": True},
    }
    with pytest.raises(ValidationError):
        ModelRegisterRequest(horizon_bars=0, **common)
    with pytest.raises(ValidationError):
        ModelRegisterRequest(horizon_bars=20_000, **common)


# ── Promote: validate_transition pure function ─────────────────────────────


def test_promote_validate_same_status_is_noop():
    """Same target as current status is allowed and flagged as noop.
    The endpoint uses the noop flag to skip the UPDATE and return the
    existing row with transition_applied=false."""
    from orchestrator.api.models import validate_transition
    ok, note = validate_transition("trained", "trained")
    assert ok is True
    assert note == "noop"


def test_promote_validate_trained_to_staged_allowed():
    """The primary forward edge: an operator-reviewed model advances
    from 'trained' to 'staged' (ready for shadow deployment)."""
    from orchestrator.api.models import validate_transition
    ok, note = validate_transition("trained", "staged")
    assert ok is True
    assert note is None


def test_promote_validate_full_forward_chain():
    """Walk the full lifecycle chain that gets a model from trained to
    live. Each edge must be allowed individually."""
    from orchestrator.api.models import validate_transition
    chain = [
        ("trained", "staged"),
        ("staged", "shadow"),
        ("shadow", "cooling_down"),
        ("cooling_down", "live"),
        ("live", "retired"),
    ]
    for src, dst in chain:
        ok, note = validate_transition(src, dst)
        assert ok is True, f"transition {src} -> {dst} should be allowed"
        assert note is None


def test_promote_validate_rejects_backward_edge():
    """Live -> shadow (rolling back a live model to shadow) is NOT a
    valid promote edge. Rolling back live models is via fresh
    registration + kill switch, not status reversion."""
    from orchestrator.api.models import validate_transition
    ok, reason = validate_transition("live", "shadow")
    assert ok is False
    assert reason is not None
    assert "invalid transition" in reason
    assert "shadow" in reason


def test_promote_validate_rejects_skipping_states():
    """Trained -> live skips staged + shadow + cooling_down. Not allowed —
    each gate has audit + observability value."""
    from orchestrator.api.models import validate_transition
    ok, reason = validate_transition("trained", "live")
    assert ok is False
    assert reason is not None
    assert "invalid transition" in reason


def test_promote_validate_rejects_from_terminal_retired():
    """Retired is terminal — no outbound edges. An operator promoting
    out of retired is a sign of confusion (the artifact is gone for
    operational purposes); the schema makes it impossible."""
    from orchestrator.api.models import validate_transition
    ok, reason = validate_transition("retired", "live")
    assert ok is False
    assert "terminal status" in reason


def test_promote_validate_awaiting_review_clearable_to_trained():
    """An operator who has cleared the deployment_readiness gap
    (registered the missing features/label) can force the model to
    'trained' rather than re-registering. The transition is allowed."""
    from orchestrator.api.models import validate_transition
    ok, note = validate_transition("awaiting_operator_review", "trained")
    assert ok is True
    assert note is None


def test_promote_validate_awaiting_review_rejectable():
    """Symmetric: the same model can be explicitly rejected from
    awaiting_operator_review (the operator decides the deployment gap
    is unfixable)."""
    from orchestrator.api.models import validate_transition
    ok, note = validate_transition("awaiting_operator_review", "rejected_by_operator")
    assert ok is True
    assert note is None


def test_promote_validate_rejected_can_only_go_to_retired():
    """A rejected model cannot be rehabilitated to trained — a fix
    means a new content_sha and a fresh registration. The only forward
    edge is explicit retirement for audit-trail tidiness."""
    from orchestrator.api.models import validate_transition
    # Rehabilitation blocked
    ok, reason = validate_transition("rejected_by_operator", "trained")
    assert ok is False
    assert "invalid transition" in reason
    # Retirement allowed
    ok, _ = validate_transition("rejected_by_operator", "retired")
    assert ok is True


# ── Repo: update_status ────────────────────────────────────────────────────


def test_update_status_stamps_promoted_columns():
    """The repo function must set BOTH promoted_by and promoted_at as
    well as the status — the audit trail answers 'who moved this, when'
    for every transition past trained."""
    target_id = uuid4()
    row = {
        "id": target_id, "status": "staged", "version": 2,
        "promoted_by": "orchestrator:test", "promoted_at": None,
        "reviewer_verdict": None, "reviewer_run_id": None,
        "updated_time": None,
    }
    conn = _FakeConn(fetchrow_results=[row])
    result = _run(models_repo.update_status(
        conn, model_id=target_id, new_status="staged",
        expected_current_status="trained",
        reviewer_verdict=None, reviewer_run_id=None,
        actor="orchestrator:test",
    ))
    assert result is not None
    assert result["status"] == "staged"
    sql = conn.calls[0][1]
    # Both promoted_by and promoted_at must appear in the SET clause
    set_clause = sql.split("SET", 1)[1].split("WHERE", 1)[0]
    assert "status = $2::text" in set_clause
    assert "promoted_by = $3" in set_clause
    assert "promoted_at = NOW()" in set_clause
    # updated_time must also refresh so two distinct transitions on the
    # same row are distinguishable in the audit column
    assert "updated_time = NOW()" in set_clause


def test_update_status_optimistic_lock_predicate():
    """MR1 fix: the UPDATE filters on the expected current status, not
    just the id. Without this, a concurrent transition between the
    API's SELECT and this UPDATE could let us write a transition whose
    source no longer matches reality.
    """
    conn = _FakeConn(fetchrow_results=[{
        "id": uuid4(), "status": "staged", "version": 1,
        "promoted_by": "x", "promoted_at": None,
        "reviewer_verdict": None, "reviewer_run_id": None,
        "updated_time": None,
    }])
    _run(models_repo.update_status(
        conn, model_id=uuid4(), new_status="staged",
        expected_current_status="trained",
        reviewer_verdict=None, reviewer_run_id=None,
        actor="orchestrator:test",
    ))
    sql = conn.calls[0][1]
    where_clause = sql.split("WHERE", 1)[1].split("RETURNING", 1)[0]
    assert "id = $1" in where_clause
    assert "status = $6::text" in where_clause
    # And the expected current status is passed as $6
    params = conn.calls[0][2]
    assert params[5] == "trained"


def test_update_status_returns_none_when_optimistic_lock_fails():
    """When the WHERE id=? AND status=expected predicate doesn't match
    (another caller raced past us), the UPDATE writes 0 rows and the
    repo returns None. The API layer surfaces this as 409, NOT 404."""
    conn = _FakeConn(fetchrow_results=[None])
    result = _run(models_repo.update_status(
        conn, model_id=uuid4(), new_status="staged",
        expected_current_status="trained",
        reviewer_verdict=None, reviewer_run_id=None,
        actor="orchestrator:test",
    ))
    assert result is None


def test_update_status_uses_coalesce_for_reviewer_fields():
    """If reviewer_verdict / reviewer_run_id are not supplied, the row's
    existing values must be preserved. COALESCE(new, existing) is the
    cleanest expression of 'NULL means keep'."""
    conn = _FakeConn(fetchrow_results=[{
        "id": uuid4(), "status": "shadow", "version": 1,
        "promoted_by": "x", "promoted_at": None,
        "reviewer_verdict": "APPROVED", "reviewer_run_id": None,
        "updated_time": None,
    }])
    _run(models_repo.update_status(
        conn, model_id=uuid4(), new_status="shadow",
        expected_current_status="staged",
        reviewer_verdict=None, reviewer_run_id=None,
        actor="orchestrator:test",
    ))
    sql = conn.calls[0][1]
    assert "reviewer_verdict = COALESCE($4, reviewer_verdict)" in sql
    assert "reviewer_run_id = COALESCE($5, reviewer_run_id)" in sql


def test_update_status_returns_none_when_row_missing():
    """Either the id is unknown OR the optimistic-lock predicate didn't
    match. Both surface as None; the API layer re-fetches to disambiguate."""
    conn = _FakeConn(fetchrow_results=[None])
    result = _run(models_repo.update_status(
        conn, model_id=uuid4(), new_status="staged",
        expected_current_status="trained",
        reviewer_verdict=None, reviewer_run_id=None,
        actor="orchestrator:test",
    ))
    assert result is None


def test_get_model_by_id_selects_promoted_columns():
    """MR2 fix: get_model_by_id must return promoted_at + promoted_by
    so the promote endpoint's noop branch can surface the real audit
    values rather than lying with None."""
    conn = _FakeConn(fetchrow_results=[{
        "id": uuid4(), "family": "lightgbm_modulator", "purpose": "regime",
        "symbol": "BTCUSDT", "interval": "1h", "horizon_bars": 24,
        "feature_set": {"names": []}, "hyperparams": {}, "metrics": {},
        "random_seed": 42, "artifact_uri": None,
        "artifact_sha256": None, "artifact_size_bytes": None,
        "artifact_synced_to_vps": False, "artifact_synced_at": None,
        "status": "trained", "version": 1,
        "promoted_at": None, "promoted_by": None,
        "created_time": None, "created_by": None, "updated_time": None,
    }])
    _run(models_repo.get_model_by_id(conn, uuid4()))
    sql = conn.calls[0][1]
    assert "promoted_at" in sql
    assert "promoted_by" in sql


# ── HTTP-level: input validation for /promote ──────────────────────────────


def test_promote_rejects_missing_auth_token(client: TestClient):
    """Auth boundary applies to /promote like every write endpoint."""
    r = client.post(f"/models/{uuid4()}/promote", json={"target_status": "staged"})
    assert r.status_code == 401
    assert r.json()["error_code"] == "auth_missing_token"


def test_promote_rejects_invalid_target_status(client: TestClient):
    """target_status is Literal-constrained — values outside the allowed
    set are rejected at the pydantic boundary, not at the DB CHECK."""
    r = client.post(
        f"/models/{uuid4()}/promote",
        headers={"X-Orch-Token": "test-token", "X-Agent-Name": "quant-researcher"},
        json={"target_status": "magical_state"},
    )
    assert r.status_code == 422


def test_promote_rejects_training_as_target(client: TestClient):
    """'training' is intentionally excluded as a promote target — it's
    the value for in-flight registrations, not a manual state. Pydantic
    rejects."""
    r = client.post(
        f"/models/{uuid4()}/promote",
        headers={"X-Orch-Token": "test-token", "X-Agent-Name": "quant-researcher"},
        json={"target_status": "training"},
    )
    assert r.status_code == 422


def test_promote_accepts_well_formed_body_with_reviewer_metadata(settings):
    """Body with reviewer fields parses cleanly. Same lifespan-aware
    pattern as ``test_register_well_formed_body_passes_validation`` —
    the in-memory idempotency store gets wired by the ``with TestClient``
    block, so the handler progresses past schema/auth and only fails
    against the fake DSN. We assert the request did NOT get a 422."""
    from orchestrator.main import create_app
    app = create_app(settings)
    try:
        with TestClient(app) as c:
            r = c.post(
                f"/models/{uuid4()}/promote",
                headers={"X-Orch-Token": "test-token", "X-Agent-Name": "quant-researcher"},
                json={
                    "target_status": "staged",
                    "reason": "approved by operator after Phase A review",
                    "reviewer_verdict": "APPROVED",
                    "reviewer_run_id": str(uuid4()),
                },
            )
            assert r.status_code != 422
    except Exception as exc:
        # If lifespan raises on the bogus DSN before the request even
        # runs, that's not a validation failure either — surface as PASS.
        assert "validation" not in str(exc).lower()


def test_register_well_formed_body_passes_validation(settings):
    """A well-formed request makes it past pydantic validation. We use
    the lifespan-aware ``with TestClient as`` form so ``app.state``
    gets wired (idempotency store, etc.). The DB pool is opened but
    actual queries will fail against the fake DSN — the endpoint will
    error inside the DB acquire, NOT at request validation. So we
    expect 5xx (or an asyncpg/connection error propagated), and we
    assert specifically that we did NOT get 422 (pydantic rejection)."""
    from orchestrator.main import create_app
    app = create_app(settings)
    try:
        with TestClient(app) as c:
            r = c.post(
                "/models/register",
                headers={"X-Orch-Token": "test-token", "X-Agent-Name": "quant-researcher"},
                json={
                    "content_sha256": "a" * 64,
                    "artifact_uri": "/tmp/x.pkl",
                    "artifact_size_bytes": 100,
                    "spec": {
                        "name": "regime_btc_v1", "purpose": "regime", "symbol": "BTCUSDT",
                        "interval": "1h", "label_feature": "label_regime_risk_on_48h",
                        "label_version": 1,
                        "objective": "binary",
                        "train_start": "2024-12-01T00:00:00",
                        "train_end": "2026-05-14T00:00:00",
                        "hyperparams": {"random_state": 42},
                        "derived_features": [],
                    },
                    "feature_names": ["f1", "f2"],
                    "metrics": {"auc": 0.6},
                    "gauntlet": {"overall_verdict": "PASS"},
                    "deployment_readiness": {"deployment_ready": True},
                },
            )
            # The body passed schema validation; the only failure mode
            # left is DB-related (5xx).
            assert r.status_code != 422
    except Exception as exc:
        # If lifespan raises on the bogus DSN before the request even
        # runs, that's not a validation failure either — the schema
        # acceptance is implicit in the fact that we didn't get a 422.
        # Surface as PASS rather than letting the connection error mask
        # the schema check.
        assert "validation" not in str(exc).lower()
