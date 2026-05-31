"""Mechanical ML model prescreen — Phase C (2026-05-19).

Pre-flight checks before invoking the quant-ml-judge in-context
template pass. Mirrors the skeptic-prescreen pattern: deterministic
pattern checks against persisted training artifacts that catch the
overfit / leakage / fragility cases we can detect WITHOUT any
qualitative reasoning. If all checks pass clean, the operator's Max-
plan budget is saved (no in-context judge pass needed). If any check
fires, the judge pass runs.

Reads from two tables:

  * ``model_registry`` (V66) — metrics JSONB carries the rich
    train-time scalars: saved_booster.auc, walk_forward.primary_mean,
    walk_forward.primary_std, gauntlet_verdict, deployment_readiness.
    feature_set JSONB carries the feature_names list.

  * ``experiment_run`` (V92) — leakage_report JSONB carries the
    per-feature correlation snapshot for label-leakage detection.
    Joined via experiment_run.content_sha256 = model_registry.
    artifact_sha256.

The 5 checks:

  1. ``label_leakage_signature`` — leakage_report.leaking_features
     non-empty → HARD flag; max_score in [0.80, 0.95) → SOFT flag.
  2. ``train_oof_auc_gap`` — saved_booster.auc minus
     walk_forward.metric_means.auc (the OOF mean). Gap > 0.10 = HARD
     (severe overfit), 0.05-0.10 = SOFT.
  3. ``walk_forward_fold_cv`` — primary_std / primary_mean.
     CV > 0.30 = HARD (fragile across folds), 0.15-0.30 = SOFT.
  4. ``gauntlet_verdict`` — FAIL = HARD, WARN = SOFT, PASS = clean,
     missing = WARN (gauntlet wasn't run; we can't tell).
  5. ``deployment_ready_flag`` — model self-reports
     deployment_ready=False → SOFT (the train pipeline itself said
     this isn't ready; surface it to the judge).

The return shape mirrors skeptic_prescreen:

  {
    "model_id": ...,
    "experiment_run_id": ...,
    "checks": [{check_name, severity, flag, finding, metric, threshold}, ...],
    "any_flag_fired": bool,
    "n_hard_flags": int,
    "n_soft_flags": int,
    "recommendation": "invoke_ml_judge" | "skip_ml_judge",
    "evaluated_at": iso8601,
  }

A HARD flag forces ``invoke_ml_judge``. A SOFT flag also forces it
(low-cost; the judge filters false positives). Only an all-clean
result skips.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import asyncpg


# Tuning knobs. Pinned in tests. Operator-controlled — moving these
# changes WHAT the prescreen flags, not whether the loop honors it.
LEAKAGE_SOFT_MIN: float = 0.80
LEAKAGE_HARD_MIN: float = 0.95  # matches blackheart-train's own threshold
AUC_GAP_SOFT_MIN: float = 0.05
AUC_GAP_HARD_MIN: float = 0.10
WF_CV_SOFT_MIN: float = 0.15
WF_CV_HARD_MIN: float = 0.30


async def ml_prescreen(
    conn: asyncpg.Connection,
    *,
    model_id: UUID | None = None,
    experiment_run_id: UUID | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Run all five mechanical pattern checks. Accepts either
    ``model_id`` (resolves experiment_run via content_sha256) or
    ``experiment_run_id`` directly. If both are given, model_id wins."""
    now = now or datetime.now(timezone.utc)

    model_row, exp_row = await _resolve_artifacts(
        conn, model_id=model_id, experiment_run_id=experiment_run_id
    )

    leakage_check = _check_label_leakage(exp_row)
    auc_gap_check = _check_train_oof_gap(model_row)
    fold_cv_check = _check_fold_cv(model_row)
    gauntlet_check = _check_gauntlet_verdict(model_row)
    deploy_check = _check_deployment_ready(model_row)

    checks = [
        leakage_check, auc_gap_check, fold_cv_check,
        gauntlet_check, deploy_check,
    ]
    n_hard = sum(1 for c in checks if c["severity"] == "HARD")
    n_soft = sum(1 for c in checks if c["severity"] == "SOFT")
    any_flag = (n_hard + n_soft) > 0

    return {
        "model_id": str(model_row["id"]) if model_row else None,
        "experiment_run_id": (
            str(exp_row["run_id"]) if exp_row else None
        ),
        "content_sha256": (
            model_row["artifact_sha256"] if model_row else None
        ),
        "checks": checks,
        "any_flag_fired": any_flag,
        "n_hard_flags": n_hard,
        "n_soft_flags": n_soft,
        "recommendation": (
            "invoke_ml_judge" if any_flag else "skip_ml_judge"
        ),
        "evaluated_at": now.isoformat(),
    }


# ── Resolver ────────────────────────────────────────────────────────


async def _resolve_artifacts(
    conn: asyncpg.Connection,
    *,
    model_id: UUID | None,
    experiment_run_id: UUID | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Fetch (model_registry, experiment_run) rows. model_id wins when
    both are given. Either row may be None if not found — checks handle
    nulls gracefully (most cases produce a "data missing" SOFT flag)."""
    model_row: dict[str, Any] | None = None
    if model_id is not None:
        row = await conn.fetchrow(
            "SELECT id, artifact_sha256, feature_set, metrics, status, "
            "       family, purpose, symbol "
            "FROM model_registry WHERE id = $1",
            model_id,
        )
        model_row = dict(row) if row else None

    exp_row: dict[str, Any] | None = None
    if experiment_run_id is not None:
        row = await conn.fetchrow(
            "SELECT run_id, content_sha256, leakage_report, params, "
            "       summary_metrics, status "
            "FROM experiment_run WHERE run_id = $1",
            experiment_run_id,
        )
        exp_row = dict(row) if row else None
    elif model_row and model_row.get("artifact_sha256"):
        # Resolve via content_sha256. Newest experiment_run wins (a
        # rare case where a sha was re-registered after a code change).
        row = await conn.fetchrow(
            "SELECT run_id, content_sha256, leakage_report, params, "
            "       summary_metrics, status "
            "FROM experiment_run "
            "WHERE content_sha256 = $1 "
            "ORDER BY created_time DESC LIMIT 1",
            model_row["artifact_sha256"],
        )
        exp_row = dict(row) if row else None

    return model_row, exp_row


# ── Per-check implementations ───────────────────────────────────────


def _check_label_leakage(exp_row: dict[str, Any] | None) -> dict[str, Any]:
    """Read ``experiment_run.leakage_report``. The CLI's
    `_check_label_leakage` writes ``leaking_features`` (list) when the
    hard threshold (0.95) is crossed, plus a ``max_score`` for the
    sub-threshold soft-warn band."""
    if exp_row is None:
        return _missing("label_leakage_signature",
                        "experiment_run row not found; cannot verify leakage")
    report = exp_row.get("leakage_report") or {}
    if not isinstance(report, dict) or not report:
        return _missing("label_leakage_signature",
                        "experiment_run.leakage_report is empty; cannot verify")

    leaking = report.get("leaking_features") or []
    max_score = report.get("max_score")
    max_feature = report.get("max_score_feature")
    method = report.get("method", "unknown")

    if leaking:
        return {
            "check_name": "label_leakage_signature",
            "severity": "HARD",
            "flag": True,
            "metric": float(max_score) if max_score is not None else None,
            "threshold": LEAKAGE_HARD_MIN,
            "finding": (
                f"{len(leaking)} feature(s) above {method} threshold "
                f"{LEAKAGE_HARD_MIN}: {leaking[:3]}"
                f"{'…' if len(leaking) > 3 else ''}"
            ),
            "details": {"leaking_features": leaking, "max_feature": max_feature},
        }

    score = float(max_score) if max_score is not None else 0.0
    if score >= LEAKAGE_SOFT_MIN:
        return {
            "check_name": "label_leakage_signature",
            "severity": "SOFT",
            "flag": True,
            "metric": score,
            "threshold": LEAKAGE_SOFT_MIN,
            "finding": (
                f"max {method} {max_feature}={score:.3f} in soft band "
                f"[{LEAKAGE_SOFT_MIN}, {LEAKAGE_HARD_MIN})"
            ),
        }
    return {
        "check_name": "label_leakage_signature",
        "severity": "PASS",
        "flag": False,
        "metric": score,
        "threshold": LEAKAGE_SOFT_MIN,
        "finding": f"max {method} {max_feature}={score:.3f} below soft threshold",
    }


def _check_train_oof_gap(model_row: dict[str, Any] | None) -> dict[str, Any]:
    """saved_booster.auc - walk_forward.metric_means.auc. Positive gap
    means the in-sample fit beat the OOF average; large positive is
    overfit. Handles missing AUC by checking primary metric instead."""
    if model_row is None:
        return _missing("train_oof_auc_gap",
                        "model_registry row not found; cannot compute gap")
    metrics = model_row.get("metrics") or {}
    if not isinstance(metrics, dict):
        return _missing("train_oof_auc_gap",
                        "model_registry.metrics is not a dict")

    saved = metrics.get("saved_booster") or {}
    wf = metrics.get("walk_forward") or {}
    primary_metric = wf.get("primary_metric") or "auc"

    train_value = saved.get(primary_metric)
    oof_means = wf.get("metric_means") or {}
    oof_value = oof_means.get(primary_metric)
    if oof_value is None:
        # Fall back to the aggregate.
        oof_value = wf.get("primary_mean")

    if train_value is None or oof_value is None:
        return _missing("train_oof_auc_gap",
                        f"missing {primary_metric}: train={train_value} oof={oof_value}")

    gap = float(train_value) - float(oof_value)
    if gap >= AUC_GAP_HARD_MIN:
        sev = "HARD"
    elif gap >= AUC_GAP_SOFT_MIN:
        sev = "SOFT"
    else:
        sev = "PASS"
    return {
        "check_name": "train_oof_auc_gap",
        "severity": sev,
        "flag": sev != "PASS",
        "metric": round(gap, 4),
        "threshold": AUC_GAP_HARD_MIN if sev == "HARD" else AUC_GAP_SOFT_MIN,
        "finding": (
            f"{primary_metric} gap train-oof = {gap:.3f} "
            f"(train={train_value:.3f}, oof={oof_value:.3f})"
        ),
    }


def _check_fold_cv(model_row: dict[str, Any] | None) -> dict[str, Any]:
    """Walk-forward fold variance: std/mean of primary metric across
    folds. High CV = the model's edge is fragile across time windows."""
    if model_row is None:
        return _missing("walk_forward_fold_cv",
                        "model_registry row not found; cannot compute CV")
    metrics = model_row.get("metrics") or {}
    wf = metrics.get("walk_forward") or {}
    mean_v = wf.get("primary_mean")
    std_v = wf.get("primary_std")
    if mean_v is None or std_v is None:
        return _missing("walk_forward_fold_cv",
                        f"missing wf stats: mean={mean_v} std={std_v}")

    mean_v = float(mean_v)
    std_v = float(std_v)
    if abs(mean_v) < 1e-9:
        return _missing("walk_forward_fold_cv",
                        f"primary_mean near zero ({mean_v}); CV undefined")

    cv = abs(std_v / mean_v)
    if cv >= WF_CV_HARD_MIN:
        sev = "HARD"
    elif cv >= WF_CV_SOFT_MIN:
        sev = "SOFT"
    else:
        sev = "PASS"
    return {
        "check_name": "walk_forward_fold_cv",
        "severity": sev,
        "flag": sev != "PASS",
        "metric": round(cv, 4),
        "threshold": WF_CV_HARD_MIN if sev == "HARD" else WF_CV_SOFT_MIN,
        "finding": (
            f"walk_forward CV = {cv:.3f} (std={std_v:.4f}, mean={mean_v:.4f}); "
            f"primary={wf.get('primary_metric', '?')}"
        ),
    }


def _check_gauntlet_verdict(model_row: dict[str, Any] | None) -> dict[str, Any]:
    """``model_registry.metrics.gauntlet_verdict``. FAIL is HARD,
    WARN is SOFT, PASS is clean. Missing means the gauntlet wasn't run,
    which is itself a SOFT flag — researcher should know."""
    if model_row is None:
        return _missing("gauntlet_verdict",
                        "model_registry row not found; cannot read verdict")
    metrics = model_row.get("metrics") or {}
    verdict = (metrics.get("gauntlet_verdict") or "").upper().strip()
    if not verdict:
        return {
            "check_name": "gauntlet_verdict",
            "severity": "SOFT",
            "flag": True,
            "metric": None,
            "threshold": None,
            "finding": "gauntlet_verdict missing; gauntlet was not run on this model",
        }
    if verdict == "FAIL":
        return {
            "check_name": "gauntlet_verdict",
            "severity": "HARD",
            "flag": True,
            "metric": verdict,
            "threshold": "PASS",
            "finding": "gauntlet verdict FAIL — the 5-gate sub-model gauntlet rejected this artifact",
        }
    if verdict == "WARN":
        return {
            "check_name": "gauntlet_verdict",
            "severity": "SOFT",
            "flag": True,
            "metric": verdict,
            "threshold": "PASS",
            "finding": "gauntlet verdict WARN — one or more sub-gates raised concerns",
        }
    return {
        "check_name": "gauntlet_verdict",
        "severity": "PASS",
        "flag": False,
        "metric": verdict,
        "threshold": "PASS",
        "finding": f"gauntlet verdict {verdict}",
    }


def _check_deployment_ready(model_row: dict[str, Any] | None) -> dict[str, Any]:
    """``model_registry.deployment_ready`` column (V66) OR
    ``metrics.deployment_readiness.deployment_ready`` (V66 also writes the
    sub-block on register). False means the train pipeline itself
    declared this artifact not ready — SOFT flag so the judge surfaces
    the gap."""
    if model_row is None:
        return _missing("deployment_ready_flag",
                        "model_registry row not found")
    ready_col = model_row.get("deployment_ready")
    metrics = model_row.get("metrics") or {}
    dr_block = metrics.get("deployment_readiness") or {}
    ready_blob = dr_block.get("deployment_ready") if isinstance(dr_block, dict) else None
    # Both sources should agree post-register; prefer the column if set.
    if ready_col is not None:
        ready = bool(ready_col)
    elif ready_blob is not None:
        ready = bool(ready_blob)
    else:
        return {
            "check_name": "deployment_ready_flag",
            "severity": "SOFT",
            "flag": True,
            "metric": None,
            "threshold": True,
            "finding": "deployment_ready flag missing on model_registry row",
        }
    if ready:
        return {
            "check_name": "deployment_ready_flag",
            "severity": "PASS",
            "flag": False,
            "metric": True,
            "threshold": True,
            "finding": "train pipeline marked this artifact deployment_ready=true",
        }
    reasons = (
        dr_block.get("reasons") or dr_block.get("not_ready_reasons") or []
    ) if isinstance(dr_block, dict) else []
    return {
        "check_name": "deployment_ready_flag",
        "severity": "SOFT",
        "flag": True,
        "metric": False,
        "threshold": True,
        "finding": (
            "train pipeline marked deployment_ready=false"
            + (f"; reasons: {reasons[:3]}" if reasons else "")
        ),
    }


def _missing(check_name: str, finding: str) -> dict[str, Any]:
    """Standardized "data missing, cannot evaluate" result. Severity
    is SOFT (not PASS) so the judge sees it — silently passing a
    check we couldn't run is a false-negative."""
    return {
        "check_name": check_name,
        "severity": "SOFT",
        "flag": True,
        "metric": None,
        "threshold": None,
        "finding": finding,
    }
