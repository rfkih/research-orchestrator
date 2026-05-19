"""Pure-function tests for the ml_prescreen mechanical checks + API
auth/validation shape. DB-touching paths (the resolver query) belong
in an integration suite.

Pattern matches test_specialists_api.py: a lenient TestClient that
converts server exceptions to HTTP responses so we can assert on
error envelopes even when the no-DB fixture trips downstream deps.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from orchestrator.config import Settings
from orchestrator.main import create_app
from orchestrator.services.specialists.ml_prescreen import (
    AUC_GAP_HARD_MIN,
    AUC_GAP_SOFT_MIN,
    LEAKAGE_HARD_MIN,
    LEAKAGE_SOFT_MIN,
    WF_CV_HARD_MIN,
    WF_CV_SOFT_MIN,
    _check_deployment_ready,
    _check_fold_cv,
    _check_gauntlet_verdict,
    _check_label_leakage,
    _check_train_oof_gap,
)


@pytest.fixture
def lenient_client(settings: Settings) -> TestClient:
    app = create_app(settings)
    return TestClient(app, raise_server_exceptions=False)


# ── Constants pinned ────────────────────────────────────────────────


def test_thresholds_pinned() -> None:
    """If any of these moves, the change is methodological — operator
    must approve. The check exists to surface accidental edits."""
    assert LEAKAGE_HARD_MIN == 0.95
    assert LEAKAGE_SOFT_MIN == 0.80
    assert AUC_GAP_HARD_MIN == 0.10
    assert AUC_GAP_SOFT_MIN == 0.05
    assert WF_CV_HARD_MIN == 0.30
    assert WF_CV_SOFT_MIN == 0.15


# ── _check_label_leakage ────────────────────────────────────────────


def test_leakage_hard_flag_on_leaking_features() -> None:
    out = _check_label_leakage({
        "leakage_report": {
            "method": "pearson_abs",
            "max_score": 0.98,
            "max_score_feature": "rsi_label_shifted",
            "leaking_features": ["rsi_label_shifted"],
        }
    })
    assert out["severity"] == "HARD"
    assert out["flag"] is True


def test_leakage_soft_flag_in_warning_band() -> None:
    out = _check_label_leakage({
        "leakage_report": {
            "method": "pearson_abs",
            "max_score": 0.85,
            "max_score_feature": "rsi_14",
        }
    })
    assert out["severity"] == "SOFT"
    assert out["flag"] is True


def test_leakage_pass_below_soft_threshold() -> None:
    out = _check_label_leakage({
        "leakage_report": {
            "method": "pearson_abs",
            "max_score": 0.35,
            "max_score_feature": "rsi_14",
        }
    })
    assert out["severity"] == "PASS"
    assert out["flag"] is False


def test_leakage_missing_report_is_soft_flag() -> None:
    """No leakage_report = we can't verify; default to SOFT to surface
    the data gap to the judge."""
    out = _check_label_leakage({"leakage_report": None})
    assert out["severity"] == "SOFT"
    assert out["flag"] is True
    out2 = _check_label_leakage(None)
    assert out2["severity"] == "SOFT"


# ── _check_train_oof_gap ────────────────────────────────────────────


def _model_row(saved_auc: float, oof_auc: float) -> dict:
    return {
        "metrics": {
            "saved_booster": {"auc": saved_auc},
            "walk_forward": {
                "primary_metric": "auc",
                "metric_means": {"auc": oof_auc},
            },
        }
    }


def test_auc_gap_hard_when_severe_overfit() -> None:
    out = _check_train_oof_gap(_model_row(saved_auc=0.85, oof_auc=0.60))
    assert out["severity"] == "HARD"
    assert out["metric"] == pytest.approx(0.25, abs=1e-3)


def test_auc_gap_soft_in_warning_band() -> None:
    out = _check_train_oof_gap(_model_row(saved_auc=0.65, oof_auc=0.58))
    assert out["severity"] == "SOFT"


def test_auc_gap_pass_when_oof_tracks_train() -> None:
    out = _check_train_oof_gap(_model_row(saved_auc=0.62, oof_auc=0.60))
    assert out["severity"] == "PASS"


def test_auc_gap_falls_back_to_primary_mean_when_metric_means_absent() -> None:
    row = {
        "metrics": {
            "saved_booster": {"auc": 0.80},
            "walk_forward": {
                "primary_metric": "auc",
                "primary_mean": 0.55,
            },
        }
    }
    out = _check_train_oof_gap(row)
    assert out["severity"] == "HARD"
    assert out["metric"] == pytest.approx(0.25, abs=1e-3)


def test_auc_gap_missing_metrics_is_soft_flag() -> None:
    out = _check_train_oof_gap({"metrics": {"saved_booster": {}, "walk_forward": {}}})
    assert out["severity"] == "SOFT"


# ── _check_fold_cv ──────────────────────────────────────────────────


def test_fold_cv_hard_when_high_variance() -> None:
    out = _check_fold_cv({
        "metrics": {"walk_forward": {"primary_mean": 0.60, "primary_std": 0.30}}
    })
    assert out["severity"] == "HARD"
    assert out["metric"] == pytest.approx(0.5, abs=1e-3)


def test_fold_cv_soft_in_warning_band() -> None:
    out = _check_fold_cv({
        "metrics": {"walk_forward": {"primary_mean": 0.60, "primary_std": 0.12}}
    })
    assert out["severity"] == "SOFT"
    assert out["metric"] == pytest.approx(0.20, abs=1e-3)


def test_fold_cv_pass_when_stable() -> None:
    out = _check_fold_cv({
        "metrics": {"walk_forward": {"primary_mean": 0.60, "primary_std": 0.05}}
    })
    assert out["severity"] == "PASS"


def test_fold_cv_handles_zero_mean_safely() -> None:
    """Edge case: mean=0 → CV undefined; surface as SOFT not crash."""
    out = _check_fold_cv({
        "metrics": {"walk_forward": {"primary_mean": 0.0, "primary_std": 0.05}}
    })
    assert out["severity"] == "SOFT"
    assert out["flag"] is True


# ── _check_gauntlet_verdict ─────────────────────────────────────────


def test_gauntlet_fail_is_hard() -> None:
    out = _check_gauntlet_verdict({"metrics": {"gauntlet_verdict": "FAIL"}})
    assert out["severity"] == "HARD"


def test_gauntlet_warn_is_soft() -> None:
    out = _check_gauntlet_verdict({"metrics": {"gauntlet_verdict": "WARN"}})
    assert out["severity"] == "SOFT"


def test_gauntlet_pass_is_clean() -> None:
    out = _check_gauntlet_verdict({"metrics": {"gauntlet_verdict": "PASS"}})
    assert out["severity"] == "PASS"


def test_gauntlet_missing_is_soft_flag() -> None:
    """Missing gauntlet = we can't verify any of the 5 sub-gates ran;
    surface to the judge rather than silently passing."""
    out = _check_gauntlet_verdict({"metrics": {}})
    assert out["severity"] == "SOFT"
    assert out["flag"] is True


def test_gauntlet_handles_mixed_case() -> None:
    out = _check_gauntlet_verdict({"metrics": {"gauntlet_verdict": "fail"}})
    assert out["severity"] == "HARD"


# ── _check_deployment_ready ─────────────────────────────────────────


def test_deployment_ready_true_is_pass() -> None:
    out = _check_deployment_ready({"deployment_ready": True, "metrics": {}})
    assert out["severity"] == "PASS"


def test_deployment_ready_false_is_soft() -> None:
    out = _check_deployment_ready({"deployment_ready": False, "metrics": {}})
    assert out["severity"] == "SOFT"
    assert out["flag"] is True


def test_deployment_ready_falls_back_to_metrics_block() -> None:
    out = _check_deployment_ready({
        "deployment_ready": None,
        "metrics": {"deployment_readiness": {"deployment_ready": True}},
    })
    assert out["severity"] == "PASS"


def test_deployment_ready_missing_is_soft_flag() -> None:
    out = _check_deployment_ready({"metrics": {}})
    assert out["severity"] == "SOFT"


# ── API auth / validation ──────────────────────────────────────────


def test_ml_prescreen_requires_token(client: TestClient) -> None:
    r = client.post(
        "/ml-prescreen",
        json={"model_id": "00000000-0000-0000-0000-000000000001"},
    )
    assert r.status_code == 401


def test_ml_prescreen_requires_at_least_one_id(lenient_client: TestClient) -> None:
    """@model_validator enforces model_id OR experiment_run_id. In
    production it surfaces as 422 validation_failed (F2 fix — was 500
    with model_post_init). In the no-DB test fixture, FastAPI resolves
    Depends(get_db_conn) before body validation, so we see 500 here.
    Production path is verified by code review; the test asserts the
    structural reject."""
    r = lenient_client.post(
        "/ml-prescreen",
        headers={"X-Orch-Token": "test-token"},
        json={},
    )
    assert r.status_code >= 400
    assert r.json()["error_code"] in ("validation_failed", "internal_error")


def test_ml_prescreen_body_validator_rejects_empty_body() -> None:
    """Pure-Pydantic check that the @model_validator fires — bypasses
    FastAPI's Depends-before-body ordering."""
    from pydantic import ValidationError

    from orchestrator.api.specialists import MlPrescreenBody

    with pytest.raises(ValidationError):
        MlPrescreenBody()


def test_ml_prescreen_rejects_non_uuid(lenient_client: TestClient) -> None:
    r = lenient_client.post(
        "/ml-prescreen",
        headers={"X-Orch-Token": "test-token"},
        json={"model_id": "not-a-uuid"},
    )
    assert r.status_code >= 400
    assert r.json()["error_code"] in ("validation_failed", "internal_error")
