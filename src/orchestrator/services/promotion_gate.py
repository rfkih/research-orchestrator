"""Promotion-time gauntlet re-check (governance fix, 2026-07-13).

Why this exists
---------------

``POST /models/{id}/promote`` used to validate only the *shape* of a
status transition (``_PROMOTION_TRANSITIONS``) — never the model's
quality. That let a model registered under an OLD gauntlet walk
``trained → staged → shadow → cooling_down → live`` with no
re-evaluation.

That is exactly how regime model ``6b3b12e4`` reached ``live`` on
2026-07-12: it was registered 2026-05-21 under the pre-2026-06-13
5-gate gauntlet (which had no median / transferability checks), and its
stored metrics FAIL the current gauntlet:

* walk-forward **primary_median = 0.5109** (< the 0.52 AUC edge bar) — a
  coin-flip typical fold; one lucky fold's 0.6094 carried the mean.
* **ci_max_abs_diff = 0.2949** (≥ the 0.15 binary transferability bar) —
  the conditional relationship shifts across folds (regime/covariate
  drift). Its DCB pooled-book certification was later shown to be a
  look-ahead artifact (``project_dcb_pooled_book_cert_2026-07-10``).

This module re-runs the *binding* gauntlet gates against the model's
already-stored ``metrics`` JSONB at promotion time, so a model that
fails today's methodology cannot be promoted into a deployment-bearing
status without an explicit, audited operator override.

Design
------

* Pure function of ``metrics`` — no DB, no re-fit, no artifact read.
  Everything it needs (walk-forward mean / median / std, metric_means
  ``ci_max_abs_diff``) is already on the registry row.
* Thresholds MIRROR ``blackheart_train.gauntlet`` +
  ``blackheart_train.conditional_invariance.PASS_THRESHOLD``. They are
  duplicated here (the orchestrator can't import blackheart-train) and
  pinned by ``test_promotion_gate`` so drift surfaces in CI.
* We deliberately do NOT gate ``adversarial_auc`` — it is structurally
  ≈1.0 for hourly + macro bars on rolling walk-forward, so gating it
  would block essentially every model. ``ci_max_abs_diff`` (the
  conditional-shift signal) is the right transferability gate. See the
  note in ``blackheart_train/gauntlet.py::_gate_transferability``.
* Fail-closed: if the metrics needed to affirm quality are absent, the
  gate FAILS (the operator re-trains under the current gauntlet or
  passes an explicit override) rather than waving the model through.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, TypeGuard

# ── Thresholds — keep in sync with blackheart_train (pinned by tests) ──────
# Source of truth: blackheart_train/gauntlet.py (_EDGE_THRESHOLD,
# _STABILITY_THRESHOLD) and conditional_invariance.PASS_THRESHOLD.
_EDGE_THRESHOLD: dict[str, float] = {"auc": 0.52, "pearson_r": 0.05}
_STABILITY_THRESHOLD: dict[str, float] = {"auc": 0.15, "pearson_r": 0.20}
_TRANSFERABILITY_THRESHOLD: dict[str, float] = {
    "binary": 0.15, "regression": 0.5, "multiclass": 0.15,
}

# primary_metric -> objective (for the transferability threshold lookup).
_METRIC_TO_OBJECTIVE: dict[str, str] = {"auc": "binary", "pearson_r": "regression"}

# Statuses where the model can observe or enforce on live decisions.
# Promotion INTO any of these re-runs the gauntlet. `retired` /
# `rejected_by_operator` (safe downgrades) and `trained` are never gated.
GATED_TARGET_STATUSES: frozenset[str] = frozenset(
    {"staged", "shadow", "cooling_down", "live"}
)


@dataclass
class PromotionGateResult:
    """Outcome of the promotion-time gauntlet re-check.

    ``passed`` is the aggregate. ``failures`` is a list of human-readable
    reasons (empty when passed). ``checks`` echoes the values evaluated
    so the API response / audit log can show exactly what was compared.
    """

    passed: bool
    failures: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)


def _finite(v: Any) -> TypeGuard[float]:
    """True (and narrows to float) for a finite real number."""
    return isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))


def evaluate_promotion_gate(metrics: dict[str, Any] | None) -> PromotionGateResult:
    """Re-run the binding gauntlet gates against a model's stored metrics.

    Reproduces gauntlet gates 3 (generalization edge — mean AND median),
    4 (fold stability), and 6 (transferability / conditional invariance)
    from the numbers already on the ``model_registry.metrics`` blob.

    Returns a :class:`PromotionGateResult`. Fail-closed on missing data.
    """
    failures: list[str] = []
    checks: dict[str, Any] = {}

    wf = (metrics or {}).get("walk_forward")
    if not isinstance(wf, dict) or not wf:
        return PromotionGateResult(
            passed=False,
            failures=[
                "no walk_forward metrics on the model — cannot affirm "
                "generalization/stability/transferability; re-train with "
                "--walk-forward or pass override_gauntlet=true"
            ],
            checks={"walk_forward_present": False},
        )

    metric = wf.get("primary_metric")
    checks["primary_metric"] = metric
    if not isinstance(metric, str):
        return PromotionGateResult(
            passed=False,
            failures=[
                f"primary_metric={metric!r} is not a known metric name "
                "— cannot affirm quality"
            ],
            checks=checks,
        )
    edge = _EDGE_THRESHOLD.get(metric)
    stability = _STABILITY_THRESHOLD.get(metric)
    if edge is None or stability is None:
        return PromotionGateResult(
            passed=False,
            failures=[
                f"no promotion-gate threshold for primary_metric={metric!r} "
                f"(known: {sorted(_EDGE_THRESHOLD)}) — refusing to affirm quality"
            ],
            checks=checks,
        )

    mean = wf.get("primary_mean")
    median = wf.get("primary_median")
    std = wf.get("primary_std")
    means = wf.get("metric_means") or {}
    ci_max = means.get("ci_max_abs_diff")
    checks.update(
        {"mean": mean, "median": median, "std": std, "ci_max_abs_diff": ci_max}
    )

    # Gate 3 — generalization edge: BOTH mean and median must clear the
    # bar. Median is the load-bearing add (a single lucky fold can lift
    # the mean while the typical fold sits at random).
    if not _finite(mean) or mean < edge:
        failures.append(
            f"generalization_edge: walk-forward {metric} mean="
            f"{mean if _finite(mean) else 'missing'} < {edge}"
        )
    if not _finite(median) or median < edge:
        failures.append(
            f"generalization_edge: walk-forward {metric} median="
            f"{median if _finite(median) else 'missing'} < {edge} "
            f"(coin-flip typical fold — the exact hole 6b3b12e4 slipped through)"
        )

    # Gate 4 — fold stability.
    if not _finite(std) or std > stability:
        failures.append(
            f"fold_stability: walk-forward {metric} std="
            f"{std if _finite(std) else 'missing'} > {stability}"
        )

    # Gate 6 — transferability (conditional invariance). ci_max_abs_diff
    # is the honest transferability signal (NOT adversarial_auc).
    objective = _METRIC_TO_OBJECTIVE.get(metric)
    transfer = _TRANSFERABILITY_THRESHOLD.get(objective) if objective else None
    checks["objective"] = objective
    if transfer is None:
        failures.append(
            f"transferability: no threshold for objective={objective!r} "
            f"(from metric {metric!r})"
        )
    elif not _finite(ci_max) or ci_max >= transfer:
        failures.append(
            f"transferability: ci_max_abs_diff="
            f"{ci_max if _finite(ci_max) else 'missing'} >= {transfer} "
            f"(conditional relationship shifts across folds — OOS metrics untrustworthy)"
        )

    return PromotionGateResult(passed=not failures, failures=failures, checks=checks)
