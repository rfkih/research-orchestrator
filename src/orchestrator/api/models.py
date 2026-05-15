"""Model registry write surface (M5e per blueprint § 19).

Endpoint:
  POST /models/register

Accepts artifact metadata (caller posts the relevant payload fields;
the orchestrator does not read the artifact file) and writes a
``model_registry`` row (V66 schema). Derives ``status`` from the
gauntlet verdict and the deployment-readiness flag the training
pipeline emits.

Why caller-posts-metadata rather than orchestrator-reads-disk:

* The artifact pickle requires ``lightgbm`` to load — adding that to
  the orchestrator's dependency tree is heavy for a side-effect-free
  endpoint.
* Cross-host deployments (training on home, orchestrator on VPS) need
  this anyway; rsync of the artifact happens elsewhere.
* The training CLI already has the metadata in memory; passing it via
  HTTP is the natural shape.

The endpoint never reads files. It records ``artifact_uri`` and
``artifact_sha256`` as supplied by the caller. Live inference workers
that load the booster re-verify sha at load time (already enforced by
``blackheart_train.artifacts.read_artifact``).

Status derivation:

  gauntlet PASS  + deployment_ready=True  → "trained"
  gauntlet PASS  + deployment_ready=False → "awaiting_operator_review"
  gauntlet FAIL                           → "rejected_by_operator"
  no gauntlet block                       → "training"

These are values from the V66 ``chk_model_registry_status`` CHECK
constraint. Promotion past ``trained`` (to ``staged`` / ``shadow`` /
``live``) is M5e+ / Phase 4 — a separate operator-action endpoint, not
the auto-register flow.

Idempotency:

* ``Idempotency-Key`` header: same TTL'd cache pattern as
  ``features.py``. A repeated POST with the same key returns the same
  response.
* Plus: content-addressed dedup — if a row with the supplied
  ``content_sha256`` already exists in ``model_registry``, we return
  it (no new insert). This is the load-bearing semantic — registering
  the same artifact twice always lands at the same model_id.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Literal
from uuid import UUID

import asyncpg
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field, field_validator

from orchestrator.repo import models as models_repo

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/models", tags=["models"])


# ── Request / response shapes ─────────────────────────────────────────────


class SpecBlock(BaseModel):
    """Just the fields we read off the spec — the training payload's
    ``spec`` dict has more (val_fraction, hyperparams, etc.) but only
    these are load-bearing for the registry row.

    ``purpose`` and ``objective`` are constrained to Literal values
    mirroring V66's CHECK constraints. Without these, an arbitrary
    string slipped past pydantic and crashed the INSERT with
    ``CheckViolationError`` (500 to the caller). Now pydantic itself
    rejects with a 422 carrying the list of allowed values.
    """

    name: str
    purpose: Literal[
        "regime", "positioning", "flow",
        "directional", "meta_label", "stacker",
    ]
    symbol: str
    interval: str | None = None
    label_feature: str
    label_version: int = 1
    objective: Literal["binary", "regression"]
    train_start: datetime
    train_end: datetime
    hyperparams: dict[str, Any] | None = None
    derived_features: list[str] = Field(default_factory=list)


class GauntletBlock(BaseModel):
    overall_verdict: str   # "PASS" | "FAIL"


class DeploymentReadinessBlock(BaseModel):
    deployment_ready: bool
    unregistered_input_features: list[str] = Field(default_factory=list)
    unregistered_label: str | None = None


class ModelRegisterRequest(BaseModel):
    """Body shape for ``POST /models/register``. Mirrors the training
    payload's top-level fields plus an explicit ``artifact_uri`` /
    ``artifact_size_bytes`` and an optional ``horizon_bars`` override.

    ``horizon_bars`` is forward-window size for the label, in bars. If
    omitted, the endpoint tries the known-label lookup table; an
    unknown label name stores ``NULL`` and logs a WARNING. Callers
    introducing a new label horizon should pass this field explicitly
    so the registry row carries the right key.
    """

    content_sha256: str = Field(..., min_length=64, max_length=64)
    artifact_uri: str | None = None
    artifact_size_bytes: int | None = Field(default=None, ge=0)
    horizon_bars: int | None = Field(
        default=None, ge=1, le=10_000,
        description=(
            "Forward-window size for the label, in bars. Overrides the "
            "_LABEL_HORIZON_BARS lookup. Pass explicitly for new label "
            "names — without it, unknown labels land with horizon_bars=NULL "
            "and lose the per-horizon distinction in the registry's "
            "(purpose, symbol, interval, horizon_bars, version) tuple."
        ),
    )
    spec: SpecBlock
    feature_names: list[str]
    metrics: dict[str, Any]
    walk_forward: dict[str, Any] | None = None
    gauntlet: GauntletBlock | None = None
    deployment_readiness: DeploymentReadinessBlock
    data_fingerprint: str | None = None


# ── Constants ─────────────────────────────────────────────────────────────


# Map our derived/registry label names to their forward-horizon in
# bars. The registry column ``horizon_bars`` is denormalised metadata
# that helps the gauntlet and the inference worker pick the right model
# for a given decision horizon. Keeping the lookup here rather than
# round-tripping through feature_registry lets us register v2 specs
# (whose labels are derived, not registered yet) without a Flyway.
_LABEL_HORIZON_BARS: dict[str, int] = {
    "label_return_7d": 168,
    "label_return_24h": 24,
    "label_meanrev_24h": 24,
    "label_regime_risk_on_48h": 48,
    "label_regime_risk_on_24h": 24,
    "label_triple_barrier": 24,
}


_FAMILY = "lightgbm_modulator"   # all M5b/M5c/M5d sub-models share this family


# ── Status derivation ─────────────────────────────────────────────────────


def _derive_status(body: ModelRegisterRequest) -> str:
    """Translate (gauntlet, deployment_readiness) into a V66
    chk_model_registry_status value. Pure function — easy to unit test.
    """
    if body.gauntlet is None:
        # No gauntlet block: the model was trained but not validated.
        # Per V66, "training" is the default for un-evaluated rows.
        return "training"
    if body.gauntlet.overall_verdict == "PASS":
        if body.deployment_readiness.deployment_ready:
            return "trained"
        # Gauntlet passes but feature set isn't registry-ready —
        # promote-to-live is blocked until features are registered.
        return "awaiting_operator_review"
    # Gauntlet FAIL (or any other non-PASS verdict): the model is on
    # the audit trail but won't be promoted.
    return "rejected_by_operator"


# ── Response shape helpers ────────────────────────────────────────────────


def _response_for_existing(row: dict[str, Any]) -> dict[str, Any]:
    """Shape a response from an existing model_registry row (idempotent
    replay path)."""
    return {
        "model_id": str(row["id"]),
        "content_sha256": row["artifact_sha256"],
        "status": row["status"],
        "version": int(row["version"]),
        "registered_at": row["created_time"].isoformat() if row["created_time"] else None,
        "idempotent_replay": True,
    }


def _response_for_new(
    *, model_id: UUID, content_sha: str, status: str, version: int, created_time
) -> dict[str, Any]:
    return {
        "model_id": str(model_id),
        "content_sha256": content_sha,
        "status": status,
        "version": int(version),
        "registered_at": created_time.isoformat() if created_time else None,
        "idempotent_replay": False,
    }


# ── Endpoint ──────────────────────────────────────────────────────────────


@router.post("/register")
async def register_model(request: Request, body: ModelRegisterRequest):
    """Register a trained sub-model artifact.

    Inserts a row into ``model_registry`` (V66). Status is derived from
    gauntlet + deployment_readiness. Idempotent on
    ``Idempotency-Key`` header and on ``content_sha256``.
    """
    agent = request.state.agent_name
    idempotency_key = request.state.idempotency_key
    store = request.app.state.idempotency
    cache_key = f"model-register:{idempotency_key}" if idempotency_key else None

    # Idempotency layer 1: header-driven replay (covers retries from
    # the same caller within the TTL).
    if cache_key:
        cached = await store.get(agent, cache_key)
        if cached is not None:
            if isinstance(cached, dict) and cached.get("_idempotent_error"):
                raise HTTPException(
                    status_code=cached["status_code"],
                    detail=cached["detail"],
                )
            return cached

    derived_status = _derive_status(body)
    # horizon_bars resolution: explicit body field wins; fall back to
    # the known-label lookup; warn if neither yields a value so the
    # operator sees the per-horizon-key degradation up front.
    if body.horizon_bars is not None:
        horizon_bars = body.horizon_bars
    else:
        horizon_bars = _LABEL_HORIZON_BARS.get(body.spec.label_feature)
        if horizon_bars is None:
            logger.warning(
                "model register: unknown label horizon | agent=%s label=%s "
                "spec=%s — storing horizon_bars=NULL; per-horizon distinction "
                "in (purpose, symbol, interval, horizon_bars, version) "
                "tuple is lost. Pass horizon_bars explicitly to fix.",
                agent, body.spec.label_feature, body.spec.name,
            )

    # Random seed: training pulls from spec.hyperparams["random_state"].
    # If somehow missing (e.g. operator hand-rolled a payload), fall
    # back to 0 — the column is NOT NULL.
    random_seed = 0
    if body.spec.hyperparams:
        rs = body.spec.hyperparams.get("random_state")
        if isinstance(rs, int):
            random_seed = rs

    # feature_set JSONB carries the list of feature names and a flag
    # for derived features so M5e+/M5f can scan it without re-parsing
    # the artifact. The blueprint § 6.1 says feature_set should be
    # JSONB; we don't lock its inner shape here.
    feature_set = {
        "names": list(body.feature_names),
        "derived": list(body.spec.derived_features),
        "label_feature": body.spec.label_feature,
        "label_version": body.spec.label_version,
    }

    # Combine the saved-booster metrics with walk-forward aggregates and
    # the gauntlet verdict into one JSONB blob so downstream consumers
    # (M5e shadow, M5d reviewer) have one place to look. Keep the keys
    # explicit so a future column extraction is mechanical.
    metrics_blob: dict[str, Any] = {
        "saved_booster": dict(body.metrics),
    }
    if body.walk_forward is not None:
        metrics_blob["walk_forward"] = {
            "primary_metric": body.walk_forward.get("primary_metric"),
            "primary_mean": body.walk_forward.get("primary_mean"),
            "primary_median": body.walk_forward.get("primary_median"),
            "primary_std": body.walk_forward.get("primary_std"),
            "n_folds_run": body.walk_forward.get("n_folds_run"),
            "n_folds_valid_metric": body.walk_forward.get("n_folds_valid_metric"),
            "metric_means": body.walk_forward.get("metric_means"),
        }
    if body.gauntlet is not None:
        metrics_blob["gauntlet_verdict"] = body.gauntlet.overall_verdict
    metrics_blob["data_fingerprint"] = body.data_fingerprint
    metrics_blob["deployment_readiness"] = body.deployment_readiness.model_dump()

    async with request.app.state.db.acquire() as conn:
        # Idempotency layer 2: content-sha lookup. The endpoint is
        # idempotent ON THE ARTIFACT, not just on the request key —
        # different agents that POST the same sha must converge on the
        # same model_id.
        existing = await models_repo.find_by_artifact_sha(conn, body.content_sha256)
        if existing is not None:
            logger.info(
                "model register: sha already present | agent=%s sha=%s model_id=%s",
                agent, body.content_sha256[:12], existing["id"],
            )
            response = _response_for_existing(existing)
            if cache_key:
                await store.put(agent, cache_key, response)
            return response

        try:
            row = await models_repo.insert_model(
                conn,
                family=_FAMILY,
                purpose=body.spec.purpose,
                symbol=body.spec.symbol,
                interval=body.spec.interval,
                horizon_bars=horizon_bars,
                feature_set=feature_set,
                hyperparams=body.spec.hyperparams,
                metrics=metrics_blob,
                random_seed=random_seed,
                artifact_uri=body.artifact_uri,
                artifact_sha256=body.content_sha256,
                artifact_size_bytes=body.artifact_size_bytes,
                status=derived_status,
                created_by=f"orchestrator:{agent}",
            )
        except asyncpg.UniqueViolationError as e:
            # Race: another caller inserted the same (purpose, symbol,
            # interval, horizon_bars, version) tuple between our
            # next_version computation and the insert. Convert to 409
            # so the caller retries with a fresh sha or backs off.
            detail = (
                f"unique constraint violation on (purpose={body.spec.purpose}, "
                f"symbol={body.spec.symbol}, interval={body.spec.interval}, "
                f"horizon_bars={horizon_bars}): another registration won the race"
            )
            if cache_key:
                await store.put(
                    agent, cache_key,
                    {"_idempotent_error": True, "status_code": 409, "detail": detail},
                )
            raise HTTPException(status_code=409, detail=detail) from e

    logger.info(
        "model registered | agent=%s sha=%s model_id=%s status=%s version=%d purpose=%s symbol=%s",
        agent, body.content_sha256[:12], row["id"], row["status"], row["version"],
        body.spec.purpose, body.spec.symbol,
    )

    response = _response_for_new(
        model_id=row["id"],
        content_sha=body.content_sha256,
        status=row["status"],
        version=row["version"],
        created_time=row["created_time"],
    )
    if cache_key:
        await store.put(agent, cache_key, response)
    return response


@router.get("")
async def list_models(
    request: Request,
    status: str | None = Query(None, description="Filter by model_registry.status"),
    purpose: str | None = Query(None, description="Filter by purpose (regime/positioning/flow/…)"),
    symbol: str | None = Query(None, description="Filter by trading symbol"),
    synced: bool | None = Query(
        None,
        description=(
            "Filter by artifact_synced_to_vps. ``synced=false`` answers "
            "'what's registered but not yet rsynced to VPS?' which is the "
            "operator's pre-promotion checklist."
        ),
    ),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """List model_registry rows with optional filters.

    Returns the same shape as :func:`get_model` per row, plus a
    ``total`` count so the caller can paginate. Ordered by
    ``created_time DESC`` — newest first matches the operator's
    review workflow.
    """
    async with request.app.state.db.acquire() as conn:
        rows = await models_repo.list_models(
            conn,
            status=status, purpose=purpose, symbol=symbol, synced=synced,
            limit=limit, offset=offset,
        )
        total = await models_repo.count_models(
            conn,
            status=status, purpose=purpose, symbol=symbol, synced=synced,
        )
    return {"models": rows, "total": total, "limit": limit, "offset": offset}


class MarkSyncedRequest(BaseModel):
    """Body for ``POST /models/{id}/mark-synced``. Both fields optional —
    a body of ``{}`` is valid and means "use defaults (now, keep
    existing URI)." This matches the rsync workflow where the operator
    runs the transport manually and just needs the orchestrator to
    record completion.
    """

    remote_path: str | None = Field(
        default=None,
        description=(
            "New value for artifact_uri after sync — typically the VPS path "
            "the rsync target landed at. If omitted, the existing artifact_uri "
            "is preserved (useful when the URI was already in VPS form at "
            "registration time)."
        ),
    )
    synced_at: datetime | None = Field(
        default=None,
        description="Timestamp the sync completed. Defaults to NOW().",
    )

    @field_validator("synced_at")
    @classmethod
    def _reject_far_future_synced_at(cls, v: datetime | None) -> datetime | None:
        """Reject ``synced_at`` more than 5 minutes ahead of wall clock.

        Why: ``mark_synced`` records the moment the rsync transport
        finished — a future timestamp is either a clock-skew bug on the
        operator's host or a typo'd manual POST. Without this guard, a
        bad value lands silently in ``artifact_synced_at`` and corrupts
        downstream "what synced when?" queries. The 5-minute window
        absorbs NTP drift without tolerating obvious mistakes.
        """
        if v is None:
            return v
        now = datetime.now(timezone.utc)
        v_cmp = v if v.tzinfo is not None else v.replace(tzinfo=timezone.utc)
        if v_cmp > now + timedelta(minutes=5):
            raise ValueError(
                f"synced_at must not be more than 5 minutes in the future "
                f"(got {v.isoformat()}, now {now.isoformat()})"
            )
        return v


@router.post("/{model_id}/mark-synced")
async def mark_model_synced(
    request: Request,
    model_id: UUID,
    body: MarkSyncedRequest | None = None,
):
    """Flip ``artifact_synced_to_vps = TRUE`` for the model and set
    ``artifact_synced_at``.

    Idempotent: re-marking a synced model refreshes the timestamp
    (latest wins) and optionally updates the URI. The rsync transport
    itself is out of scope — operators run it manually for now, then
    POST here to record completion.

    Honours ``Idempotency-Key`` like the other write endpoints.
    """
    agent = request.state.agent_name
    idempotency_key = request.state.idempotency_key
    store = request.app.state.idempotency
    cache_key = f"mark-synced:{model_id}:{idempotency_key}" if idempotency_key else None

    if cache_key:
        cached = await store.get(agent, cache_key)
        if cached is not None:
            if isinstance(cached, dict) and cached.get("_idempotent_error"):
                raise HTTPException(status_code=cached["status_code"], detail=cached["detail"])
            return cached

    body = body or MarkSyncedRequest()
    synced_at = body.synced_at or datetime.now(timezone.utc)

    async with request.app.state.db.acquire() as conn:
        row = await models_repo.mark_synced(
            conn,
            model_id=model_id,
            remote_path=body.remote_path,
            synced_at=synced_at,
            actor=f"orchestrator:{agent}",
        )
    if row is None:
        detail = f"model_id {model_id} not found"
        if cache_key:
            await store.put(
                agent, cache_key,
                {"_idempotent_error": True, "status_code": 404, "detail": detail},
            )
        raise HTTPException(status_code=404, detail=detail)

    # Marking a rejected_by_operator model as synced is operationally
    # suspect — rejected artifacts shouldn't be promoted to VPS in the
    # first place. We don't refuse the request (the operator may have a
    # reason, e.g. retaining the file for forensics), but surface the
    # event so the audit trail catches it.
    if row["status"] == "rejected_by_operator":
        logger.warning(
            "mark_synced applied to rejected model | agent=%s model_id=%s "
            "status=rejected_by_operator — rejected artifacts should not "
            "normally be promoted to VPS; verify operator intent.",
            agent, model_id,
        )

    response = {
        "model_id": str(row["id"]),
        "artifact_synced_to_vps": bool(row["artifact_synced_to_vps"]),
        "artifact_synced_at": (
            row["artifact_synced_at"].isoformat() if row["artifact_synced_at"] else None
        ),
        "artifact_uri": row["artifact_uri"],
        "status": row["status"],
    }
    logger.info(
        "model marked synced | agent=%s model_id=%s remote_path=%s",
        agent, model_id, body.remote_path,
    )
    if cache_key:
        await store.put(agent, cache_key, response)
    return response


@router.get("/{model_id}")
async def get_model(request: Request, model_id: UUID):
    """Read one model_registry row by ID. Auth-required like the rest
    of the orchestrator; returns 404 if no such id."""
    async with request.app.state.db.acquire() as conn:
        row = await models_repo.get_model_by_id(conn, model_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"model_id {model_id} not found")
    # asyncpg returns UUID and datetime; the TypedJSONResponse encoder
    # handles them. Just hand it back.
    return row
