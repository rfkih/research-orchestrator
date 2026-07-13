"""Tests for the promotion-time gauntlet re-check (services/promotion_gate.py).

Pure-function tests — no DB, no TestClient. The load-bearing case is the
regression fixture ``_METRICS_6B3B12E4``: the exact stored metrics of the
regime model that reached ``live`` on 2026-07-12 and was later shown to
have certified a look-ahead-contaminated book. The gate MUST reject it.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from orchestrator.services.promotion_gate import (
    _EDGE_THRESHOLD,
    _RANDOM_BASELINE,
    _STABILITY_THRESHOLD,
    _TRANSFERABILITY_THRESHOLD,
    GATED_TARGET_STATUSES,
    evaluate_promotion_gate,
)

# The real model_registry.metrics blob for 6b3b12e4 (regime_eth_v2),
# trimmed to the fields the gate reads. mean 0.5338 clears the 0.52 bar,
# but median 0.5109 does not, and ci_max_abs_diff 0.2949 >= 0.15.
_METRICS_6B3B12E4 = {
    "auc": 0.6094,  # walk_forward_last_fold headline — one lucky fold
    "walk_forward": {
        "primary_metric": "auc",
        "primary_mean": 0.5338,
        "primary_median": 0.5109,
        "primary_std": 0.1142,
        "metric_means": {
            "auc": 0.5338,
            "adversarial_auc": 0.9976,
            "ci_max_abs_diff": 0.2949,
        },
    },
}


def _good_metrics() -> dict:
    """A model that legitimately clears every binding gate."""
    return {
        "walk_forward": {
            "primary_metric": "auc",
            "primary_mean": 0.60,
            "primary_median": 0.58,
            "primary_std": 0.08,
            "metric_means": {"ci_max_abs_diff": 0.10, "adversarial_auc": 0.99},
        }
    }


# ── The regression case ────────────────────────────────────────────────────


def test_gate_rejects_the_6b3b12e4_lookahead_model():
    res = evaluate_promotion_gate(_METRICS_6B3B12E4)
    assert res.passed is False
    joined = " | ".join(res.failures)
    assert "median" in joined  # gate 3 median
    assert "transferability" in joined  # gate 6 ci_max_abs_diff
    # adversarial_auc is high but must NOT be the reason (deliberately not gated)
    assert "adversarial" not in joined.lower()


def test_gate_passes_a_genuinely_good_model():
    res = evaluate_promotion_gate(_good_metrics())
    assert res.passed is True
    assert res.failures == []


# ── Individual binding gates ───────────────────────────────────────────────


def test_median_below_bar_fails_even_if_mean_passes():
    m = _good_metrics()
    m["walk_forward"]["primary_mean"] = 0.55  # passes
    m["walk_forward"]["primary_median"] = 0.51  # fails 0.52
    res = evaluate_promotion_gate(m)
    assert res.passed is False
    assert any("median" in f for f in res.failures)


def test_transferability_shift_fails():
    m = _good_metrics()
    m["walk_forward"]["metric_means"]["ci_max_abs_diff"] = 0.20  # >= 0.15
    res = evaluate_promotion_gate(m)
    assert res.passed is False
    assert any("transferability" in f for f in res.failures)


def test_unstable_folds_fail():
    m = _good_metrics()
    m["walk_forward"]["primary_std"] = 0.20  # > 0.15
    res = evaluate_promotion_gate(m)
    assert res.passed is False
    assert any("fold_stability" in f for f in res.failures)


def test_pearson_regression_metric_uses_its_own_thresholds():
    m = {
        "walk_forward": {
            "primary_metric": "pearson_r",
            "primary_mean": 0.08,
            "primary_median": 0.07,
            "primary_std": 0.15,
            "metric_means": {"ci_max_abs_diff": 0.4},  # < 0.5 regression bar
        }
    }
    assert evaluate_promotion_gate(m).passed is True
    m["walk_forward"]["metric_means"]["ci_max_abs_diff"] = 0.6  # >= 0.5
    assert evaluate_promotion_gate(m).passed is False


# ── Fail-closed behaviour ──────────────────────────────────────────────────


def test_no_metrics_fails_closed():
    assert evaluate_promotion_gate(None).passed is False
    assert evaluate_promotion_gate({}).passed is False


def test_no_walk_forward_fails_closed():
    assert evaluate_promotion_gate({"auc": 0.9}).passed is False


def test_missing_median_fails_closed():
    m = _good_metrics()
    del m["walk_forward"]["primary_median"]
    res = evaluate_promotion_gate(m)
    assert res.passed is False


def test_missing_ci_max_fails_closed():
    m = _good_metrics()
    m["walk_forward"]["metric_means"] = {}
    res = evaluate_promotion_gate(m)
    assert res.passed is False


def test_unknown_primary_metric_fails_closed():
    m = {"walk_forward": {"primary_metric": "macro_auc_ovr", "primary_mean": 0.9}}
    res = evaluate_promotion_gate(m)
    assert res.passed is False


# ── Threshold pins — must match blackheart_train (source of truth) ─────────


def test_thresholds_are_pinned_to_blackheart_train():
    # blackheart_train/gauntlet.py::_EDGE_THRESHOLD
    assert _EDGE_THRESHOLD == {"auc": 0.52, "pearson_r": 0.05}
    # blackheart_train/gauntlet.py::_STABILITY_THRESHOLD
    assert _STABILITY_THRESHOLD == {"auc": 0.15, "pearson_r": 0.20}
    # blackheart_train/conditional_invariance.py::PASS_THRESHOLD
    assert _TRANSFERABILITY_THRESHOLD == {"binary": 0.15, "regression": 0.5, "multiclass": 0.15}


def test_gated_targets_are_the_deployment_bearing_statuses():
    assert frozenset(
        {"staged", "shadow", "cooling_down", "live"}
    ) == GATED_TARGET_STATUSES
    # safe downgrades / pre-deploy states are NOT gated
    assert "retired" not in GATED_TARGET_STATUSES
    assert "rejected_by_operator" not in GATED_TARGET_STATUSES
    assert "trained" not in GATED_TARGET_STATUSES


# ── Gate 5 — saved booster above random ────────────────────────────────────


def test_booster_below_random_fails():
    m = _good_metrics()
    m["saved_booster"] = {"auc": 0.48}  # < 0.5
    res = evaluate_promotion_gate(m)
    assert res.passed is False
    assert any("saved_booster_above_random" in f for f in res.failures)


def test_booster_legacy_flat_shape_read():
    m = _good_metrics()
    m["auc"] = 0.42  # legacy flat saved-booster metric < 0.5
    res = evaluate_promotion_gate(m)
    assert any("saved_booster_above_random" in f for f in res.failures)


def test_booster_absent_is_lenient():
    # _good_metrics has no saved_booster and no flat metric -> no booster failure
    res = evaluate_promotion_gate(_good_metrics())
    assert res.passed is True
    assert not any("saved_booster" in f for f in res.failures)


def test_6b3b12e4_does_not_fail_on_booster():
    # its saved booster auc 0.6094 clears random 0.5; the failures are
    # median + transferability only, never the booster gate.
    res = evaluate_promotion_gate(_METRICS_6B3B12E4)
    assert not any("saved_booster" in f for f in res.failures)


# ── Cross-repo drift: thresholds must match blackheart-train source ─────────


def _module_const(path: Path, name: str):
    """Extract a module-level dict constant from a .py file via ast."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        targets = (
            node.targets if isinstance(node, ast.Assign)
            else [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        for t in targets:
            if isinstance(t, ast.Name) and t.id == name and node.value is not None:
                return ast.literal_eval(node.value)
    raise KeyError(f"{name} not found in {path}")


def test_thresholds_match_blackheart_train_source():
    """Best-effort cross-repo pin: when blackheart-train is a sibling
    checkout, assert our duplicated thresholds equal its source of truth.
    Skips in an isolated CI where blackheart-train isn't present."""
    train = (
        Path(__file__).resolve().parents[2]
        / "blackheart-train" / "src" / "blackheart_train"
    )
    gauntlet = train / "gauntlet.py"
    cond_inv = train / "conditional_invariance.py"
    if not gauntlet.exists() or not cond_inv.exists():
        pytest.skip("blackheart-train not a sibling checkout — drift check skipped")
    assert _module_const(gauntlet, "_EDGE_THRESHOLD") == _EDGE_THRESHOLD
    assert _module_const(gauntlet, "_STABILITY_THRESHOLD") == _STABILITY_THRESHOLD
    assert _module_const(gauntlet, "_RANDOM_BASELINE") == _RANDOM_BASELINE
    assert _module_const(cond_inv, "PASS_THRESHOLD") == _TRANSFERABILITY_THRESHOLD
