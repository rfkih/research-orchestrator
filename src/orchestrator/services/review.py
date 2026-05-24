"""Quant review checklist — adversarial audit for the paired-agent loop.

The quant-reviewer agent invokes these functions to produce a structured,
mechanical verdict on a research plan or graduation candidate. Each check
returns a record with:

    {check_name, severity, passed, finding, details}

Severity levels — fail behaviour is encoded in ``aggregate_verdict``:

    * ``blocker``  — any failure → REJECTED
    * ``warning``  — one failure → CONDITIONAL_APPROVAL; two+ → REJECTED
    * ``info``     — never blocks; for forensics only

This module is pure-function-only. The reviewer agent runs these against
artifacts pulled from research_journal / research_iteration_log via the
existing GET endpoints — no new I/O paths needed here. The orchestrator
gates (``/queue``, ``/walk-forward``) read the verdict the reviewer
posted; they do NOT re-run the checks themselves. That separation is
intentional: the reviewer can apply qualitative judgement on top of the
mechanical checks, and the orchestrator stays a thin enforcement layer.

Reference: industry-standard paired-research roles (PM proposes /
Risk audits) — neither side can graduate a candidate alone, both write
to the same append-only audit log.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

# Operator-controlled constants, pinned in tests.
DSR_THRESHOLD: float = 0.95
PF_CI_LOWER_BOUND_PASS: float = 1.0
MIN_TRADES_PASS: int = 100
COST_REALISM_BPS: int = 50
PORTFOLIO_CORR_THRESHOLD: float = 0.5
ROBUSTNESS_NEIGHBOR_TOLERANCE: float = 0.30  # adjacent cell within 30% of optimum
WARNING_FAIL_REJECT_THRESHOLD: int = 2  # 2 warning fails → REJECTED


# Public — re-exported by api/reviews.py so the request schema and the
# checklist functions agree on the severity enum by definition. Adding
# a level (say ``critical``) must happen in exactly one place.
Severity = Literal["blocker", "warning", "info"]


def _check(
    name: str, severity: Severity, passed: bool, finding: str, **details: Any
) -> dict[str, Any]:
    """Build one check record. Keep keys flat for JSONB-friendliness."""
    return {
        "check_name": name,
        "severity": severity,
        "passed": passed,
        "finding": finding,
        "details": details,
    }


# Plan review checks — run BEFORE the sweep enqueues.
def check_pre_registration(
    hypothesis_entry: dict[str, Any] | None, plan_created_time: datetime | None
) -> dict[str, Any]:
    """The HYPOTHESIS journal entry must exist AND predate the plan.

    Pre-registration is the line between confirmatory research (which
    can clear V11 / DSR) and exploratory dredging (which cannot,
    regardless of how good the metrics look). If a hypothesis was
    written AFTER the plan it's not a hypothesis — it's a post-hoc
    rationalisation. BLOCKER.
    """
    if hypothesis_entry is None:
        return _check(
            "pre_registration",
            "blocker",
            False,
            "No HYPOTHESIS journal entry found. The plan cannot be reviewed "
            "without a pre-registered thesis stating predicted effect, "
            "mechanism, and success metric.",
        )
    h_time = hypothesis_entry.get("created_time")
    if not isinstance(h_time, datetime) or plan_created_time is None:
        return _check(
            "pre_registration",
            "blocker",
            False,
            "HYPOTHESIS or plan timestamp missing — cannot verify pre-registration.",
            hypothesis_id=str(hypothesis_entry.get("journal_id")),
        )
    # Strict ordering: HYPOTHESIS strictly before plan.
    if h_time >= plan_created_time:
        return _check(
            "pre_registration",
            "blocker",
            False,
            f"HYPOTHESIS was written at {h_time.isoformat()} but the plan "
            f"references it at {plan_created_time.isoformat()}. "
            "Hypothesis must predate the plan.",
            hypothesis_time=h_time.isoformat(),
            plan_time=plan_created_time.isoformat(),
        )
    return _check(
        "pre_registration",
        "blocker",
        True,
        "HYPOTHESIS predates the plan.",
        hypothesis_id=str(hypothesis_entry.get("journal_id")),
    )


_MECHANISM_MARKERS = (
    "because",
    "driven by",
    "due to",
    "caused by",
    "mechanism",
    "rationale",
    "explained by",
    "follows from",
)


def check_mechanism_named(hypothesis_entry: dict[str, Any] | None) -> dict[str, Any]:
    """The HYPOTHESIS content must name an economic / structural mechanism.

    "Tighter ATR-extreme + RSI-extreme raises MFE/MAE" with no causal
    story is just curve-fit metric optimisation. A real hypothesis says
    *why* — e.g., "tighter extremes harvest mean-reversion *because*
    overshoots in low-vol regimes have higher snap-back probability."
    WARNING.
    """
    if hypothesis_entry is None:
        return _check(
            "mechanism_named",
            "warning",
            False,
            "No HYPOTHESIS to inspect for mechanism.",
        )
    content = (hypothesis_entry.get("content") or "").lower()
    if len(content) < 60:
        return _check(
            "mechanism_named",
            "warning",
            False,
            f"HYPOTHESIS.content is only {len(content)} chars — too short to "
            "carry a mechanism. Minimum ~60 chars to state effect + cause.",
            content_length=len(content),
        )
    if not any(m in content for m in _MECHANISM_MARKERS):
        return _check(
            "mechanism_named",
            "warning",
            False,
            "HYPOTHESIS.content does not contain any mechanism marker "
            f"({', '.join(_MECHANISM_MARKERS)}). Without a 'why', the "
            "research is parameter dredging.",
            markers_searched=list(_MECHANISM_MARKERS),
        )
    return _check(
        "mechanism_named", "warning", True, "Mechanism marker found in HYPOTHESIS."
    )


def check_axis_not_recently_failed(
    prior_strategy_outcomes: list[dict[str, Any]],
    proposed_axis_names: list[str],
    lookback_days: int = 14,
) -> dict[str, Any]:
    """The orchestrator's re-discovery gate hard-rejects on prior DISCARD.
    The reviewer's check is softer: warn if the same archetype/axis
    combo has produced INSUFFICIENT_EVIDENCE / NO_EDGE recently — the
    plan may not be hopeless, but the agent is fishing.
    """
    if not prior_strategy_outcomes:
        return _check(
            "axis_not_recently_failed",
            "warning",
            True,
            "No prior STRATEGY_OUTCOME entries on this axis combo within "
            f"the last {lookback_days} days.",
        )
    proposed = set(proposed_axis_names)
    near_misses: list[str] = []
    for entry in prior_strategy_outcomes:
        sd = entry.get("structured_data") or {}
        prior_axes = sd.get("axis_names")
        if isinstance(prior_axes, list) and set(prior_axes) == proposed:
            verdict = sd.get("verdict") or entry.get("title", "")
            near_misses.append(
                f"{entry.get('created_time')} · {verdict}"
            )
    if near_misses:
        return _check(
            "axis_not_recently_failed",
            "warning",
            False,
            f"Same axis combo {sorted(proposed)} produced "
            f"{len(near_misses)} prior outcome(s) in the last "
            f"{lookback_days} days. Re-running risks p-hacking even if "
            "the orchestrator's discard gate doesn't fire.",
            near_misses=near_misses,
        )
    return _check(
        "axis_not_recently_failed",
        "warning",
        True,
        "Axis combo not recently exercised on this strategy.",
    )


def plan_review_checklist(
    *,
    hypothesis_entry: dict[str, Any] | None,
    plan_created_time: datetime | None,
    proposed_axis_names: list[str],
    prior_strategy_outcomes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run all plan-review checks. Order matters for human-readable output."""
    return [
        check_pre_registration(hypothesis_entry, plan_created_time),
        check_mechanism_named(hypothesis_entry),
        check_axis_not_recently_failed(prior_strategy_outcomes, proposed_axis_names),
    ]


# Graduation review checks — run BEFORE /walk-forward.
def check_n_trades_ample(iteration_metrics: dict[str, Any]) -> dict[str, Any]:
    """Sanity check on V11. The orchestrator already gates this in
    statistical_verdict — re-checking here catches data drift between
    iteration_log and metrics_snapshot. BLOCKER."""
    n = (iteration_metrics.get("analysis") or {}).get("n_trades")
    if n is None or n < MIN_TRADES_PASS:
        return _check(
            "n_trades_ample",
            "blocker",
            False,
            f"n_trades={n} below threshold {MIN_TRADES_PASS}. Cannot "
            "graduate on insufficient sample.",
            n_trades=n,
            threshold=MIN_TRADES_PASS,
        )
    return _check(
        "n_trades_ample", "blocker", True, f"n_trades={n} >= {MIN_TRADES_PASS}.",
        n_trades=n,
    )


def check_pf_ci_lower_bound(iteration_ci: dict[str, Any]) -> dict[str, Any]:
    """The PF 95% CI lower bound must exceed 1.0 — the V11 contract.
    Re-checked here as a backstop. BLOCKER."""
    pf_95 = iteration_ci.get("pf_95") or {}
    lo = pf_95.get("low")
    if lo is None:
        return _check(
            "pf_ci_lower_bound",
            "blocker",
            False,
            "PF 95% CI lower bound is not computable (n too small).",
        )
    try:
        lo_f = float(lo)
    except (TypeError, ValueError):
        return _check(
            "pf_ci_lower_bound",
            "blocker",
            False,
            f"PF 95% CI lower bound {lo!r} not coercible to float.",
        )
    passed = lo_f > PF_CI_LOWER_BOUND_PASS
    return _check(
        "pf_ci_lower_bound",
        "blocker",
        passed,
        f"PF 95% CI lower bound = {lo_f}; "
        f"{'> 1.0 (passes V11)' if passed else '<= 1.0 (fails V11)'}.",
        pf_lo=lo_f,
    )


def check_dsr_threshold(iteration_metrics: dict[str, Any]) -> dict[str, Any]:
    """DSR (Bailey & López de Prado 2014) with the real cumulative trial
    count must clear the significance threshold. Selection-bias
    deflation matters. BLOCKER."""
    analysis = iteration_metrics.get("analysis") or {}
    dsr = analysis.get("dsr")
    if dsr is None:
        return _check(
            "dsr_threshold",
            "blocker",
            False,
            (
                "DSR is None — closed-form Bailey-LdP denom_sq<=0 AND "
                "bootstrap fallback also unavailable (n_obs<30, zero "
                "bootstrap variance, or pre-bootstrap iteration). "
                "Cannot graduate."
            ),
        )
    try:
        dsr_f = float(dsr)
    except (TypeError, ValueError):
        return _check(
            "dsr_threshold",
            "blocker",
            False,
            f"DSR value {dsr!r} not coercible to float.",
        )
    passed = dsr_f >= DSR_THRESHOLD
    n_trials = analysis.get("dsr_n_trials")
    return _check(
        "dsr_threshold",
        "blocker",
        passed,
        f"DSR = {round(dsr_f, 4)} (n_trials = {n_trials}); "
        f"{'>=' if passed else '<'} threshold {DSR_THRESHOLD}.",
        dsr=dsr_f,
        n_trials=n_trials,
    )


def check_cost_realism(iteration_metrics: dict[str, Any]) -> dict[str, Any]:
    """+50bps slippage scenario must still leave net PnL positive.
    +20bps is V11's gate; +50bps is the reviewer's stress test for
    realistic production turnover (taker-fees + spread + impact).
    WARNING."""
    slip = (iteration_metrics.get("analysis") or {}).get("slippage_haircut_pnl") or {}
    key = f"+{COST_REALISM_BPS}bps"
    val = slip.get(key)
    if val is None:
        return _check(
            "cost_realism",
            "warning",
            False,
            f"Slippage scenario {key} not computed.",
        )
    try:
        val_f = float(val)
    except (TypeError, ValueError):
        return _check(
            "cost_realism",
            "warning",
            False,
            f"Slippage value {val!r} for {key} not coercible to float.",
        )
    passed = val_f > 0
    return _check(
        "cost_realism",
        "warning",
        passed,
        f"Net PnL at {key} = {round(val_f, 4)}; "
        f"{'positive (cost-realistic)' if passed else 'negative (fragile to cost realism)'}.",
        slippage_bps=COST_REALISM_BPS,
        net_pnl=val_f,
    )


def check_regime_concentration(iteration_metrics: dict[str, Any]) -> dict[str, Any]:
    """The edge must show positive expectancy across dominant regimes.
    A strategy that works only in TRENDING and bleeds in CHOP is a
    regime bet, not an edge. WARNING."""
    regimes = (iteration_metrics.get("analysis") or {}).get("regimes") or {}
    by_regime = regimes.get("by_trend_regime") or {}
    if not by_regime:
        return _check(
            "regime_concentration",
            "warning",
            False,
            "No by_trend_regime breakdown; cannot verify regime invariance.",
        )
    losing_regimes: list[str] = []
    for regime, stats in by_regime.items():
        if not isinstance(stats, dict):
            continue
        n = stats.get("n") or 0
        pnl = stats.get("pnl")
        if n < 5:
            # Too few trades in this regime to draw a conclusion; ignore.
            continue
        if pnl is None or pnl <= 0:
            losing_regimes.append(f"{regime} (n={n}, pnl={pnl})")
    if losing_regimes:
        return _check(
            "regime_concentration",
            "warning",
            False,
            f"Regimes with non-positive expectancy: {', '.join(losing_regimes)}. "
            "Edge may be regime-conditional rather than structural.",
            losing_regimes=losing_regimes,
        )
    return _check(
        "regime_concentration",
        "warning",
        True,
        "Positive expectancy across all regimes with n>=5 trades.",
    )


def check_portfolio_fit(iteration_metrics: dict[str, Any]) -> dict[str, Any]:
    """Spearman max-POSITIVE correlation with the protected book must be
    < threshold. Highly redundant candidates (positively correlated with
    existing book members) don't justify a slot. Negative correlations
    are diversification benefits, not penalties — a strategy that hedges
    the book IMPROVES portfolio Sharpe and must pass this check.

    Updated 2026-05-19 (methodology fix): previously used
    ``max_abs_corr`` which treated hedges and duplicates identically,
    producing false REJECTs on legitimate diversifiers. Iter
    ``88b4c6a3-d6d9-4836-91e9-ead4cf67d231`` (DCB BTC 4h, signed corr
    -0.43 vs VCB) was the first such case to surface this drift.
    Operator-escalation journal ``c58a4b8a-3d14-4b37-b1fe-ab5eb45cb3f7``
    documents the diagnosis.

    Methodology: the gate's purpose is to reject candidates that
    DUPLICATE existing book exposure. Duplication = positive
    correlation. Diversification = negative correlation. Only positive
    correlations count against the threshold; negative ones are
    informational (diversification benefit).

    Note on the tick-time portfolio gate
    (``portfolio.evaluate_portfolio_gate``): that gate is still using
    ``|corr|`` and may demote legitimate diversifiers pre-graduation.
    Discovered-during-implementation; tracked as follow-up. This
    reviewer check uses the corrected methodology regardless.

    WARNING."""
    pc = iteration_metrics.get("portfolio_corr") or {}
    if not pc.get("applied"):
        # Gate didn't fire (no baselines or no PF CI). Pass with note.
        return _check(
            "portfolio_fit",
            "warning",
            True,
            "Portfolio gate did not apply (no baselines or no CI). "
            "Reviewer should manually inspect correlation if baselines "
            "are expected.",
            applied=False,
        )

    raw_per_code = pc.get("per_code_corr") or {}
    # Coerce to floats; ignore None and non-numeric entries (forensics
    # only — payload upstream is already rounded).
    signed: dict[str, float] = {}
    for code, value in raw_per_code.items():
        if value is None:
            continue
        try:
            signed[code] = float(value)
        except (TypeError, ValueError):
            continue

    if not signed:
        # No usable per-code correlations. Fall back to legacy
        # ``max_abs_corr`` for diagnostic visibility but pass with
        # explicit note — historical iterations predate per_code
        # surfacing and should not be retroactively rejected.
        legacy_max_abs = pc.get("max_abs_corr")
        return _check(
            "portfolio_fit",
            "warning",
            True,
            f"per_code_corr empty or unusable; legacy max_abs_corr"
            f"={legacy_max_abs}. Cannot compute signed-correlation gate; "
            f"passing with note (likely a pre-2026-05-19 iteration).",
            applied=True,
            per_code=raw_per_code,
        )

    # Max positive correlation = duplication risk (penalized).
    # Negative correlations = diversification benefit (not penalized).
    positives = {code: v for code, v in signed.items() if v > 0}
    max_positive = max(positives.values(), default=0.0)
    most_positive_code = (
        max(positives, key=positives.get) if positives else None
    )
    most_negative = min(signed.values())
    most_negative_code = min(signed, key=signed.get)

    passed = max_positive < PORTFOLIO_CORR_THRESHOLD
    if not positives:
        finding = (
            f"All correlations are non-positive (most-negative: "
            f"{round(most_negative, 4)} vs {most_negative_code}). "
            f"Candidate is a pure diversifier vs the book — no "
            f"duplication risk."
        )
    else:
        finding = (
            f"max_positive_corr = {round(max_positive, 4)} (vs "
            f"{most_positive_code}); "
            f"{'< ' if passed else '>= '}{PORTFOLIO_CORR_THRESHOLD} "
            f"threshold. Most-negative: {round(most_negative, 4)} vs "
            f"{most_negative_code} (diversification benefit, not "
            f"penalized)."
        )
    return _check(
        "portfolio_fit",
        "warning",
        passed,
        finding,
        max_positive_corr=round(max_positive, 4),
        most_negative_corr=round(most_negative, 4),
        most_positive_code=most_positive_code,
        most_negative_code=most_negative_code,
        per_code=raw_per_code,
    )


def check_param_robustness(
    iteration_params: dict[str, Any],
    sweep_history: list[tuple[dict[str, Any], float]],
    tolerance: float = ROBUSTNESS_NEIGHBOR_TOLERANCE,
) -> dict[str, Any]:
    """The optimum cell shouldn't be on a cliff. At least one
    Hamming-distance-1 neighbour in the sweep should have PF within
    ``tolerance`` of the optimum. If the only good cell is the
    optimum, the result is curve-fit. WARNING.

    sweep_history items: ``(params_dict, pf)`` for completed iterations
    on this strategy + axis. Pass them sorted by created_time if you
    want, but order doesn't affect the check.
    """
    if not sweep_history:
        return _check(
            "param_robustness",
            "warning",
            True,
            "No sweep history available to assess robustness; assume neutral.",
        )
    opt_pf = max((pf for _, pf in sweep_history), default=None)
    if opt_pf is None:
        return _check(
            "param_robustness", "warning", True, "No PF values in history."
        )
    opt_params: dict[str, Any] | None = None
    for params, pf in sweep_history:
        if pf == opt_pf:
            opt_params = params
            break
    if opt_params is None:
        return _check(
            "param_robustness", "warning", True, "Could not locate optimum params."
        )
    neighbours: list[tuple[dict[str, Any], float]] = []
    for params, pf in sweep_history:
        diffs = sum(1 for k in opt_params if str(params.get(k)) != str(opt_params.get(k)))
        if diffs == 1:
            neighbours.append((params, pf))
    if not neighbours:
        return _check(
            "param_robustness",
            "warning",
            False,
            "Optimum has no Hamming-1 neighbours in the swept grid. "
            "Cannot rule out cliff. Either sweep is too coarse or "
            "result is a single-cell pick.",
            opt_params=opt_params,
            opt_pf=opt_pf,
        )
    threshold = opt_pf * (1.0 - tolerance)
    near = [(p, pf) for p, pf in neighbours if pf >= threshold]
    if not near:
        return _check(
            "param_robustness",
            "warning",
            False,
            f"Optimum PF={round(opt_pf, 4)} but no Hamming-1 neighbour "
            f"within {int(tolerance * 100)}% (threshold {round(threshold, 4)}). "
            "Looks like a cliff — likely overfit.",
            opt_pf=opt_pf,
            neighbour_pfs=[pf for _, pf in neighbours],
        )
    return _check(
        "param_robustness",
        "warning",
        True,
        f"Optimum PF={round(opt_pf, 4)} has {len(near)} Hamming-1 "
        f"neighbour(s) within {int(tolerance * 100)}%. Plateau-shaped, "
        "not a cliff.",
        opt_pf=opt_pf,
        nearby_neighbours=len(near),
    )


def graduation_review_checklist(
    *,
    iteration_metrics: dict[str, Any],
    iteration_ci: dict[str, Any],
    iteration_params: dict[str, Any],
    sweep_history: list[tuple[dict[str, Any], float]],
) -> list[dict[str, Any]]:
    """Run all graduation-review checks. Order matters for output legibility."""
    return [
        check_n_trades_ample(iteration_metrics),
        check_pf_ci_lower_bound(iteration_ci),
        check_dsr_threshold(iteration_metrics),
        check_cost_realism(iteration_metrics),
        check_regime_concentration(iteration_metrics),
        check_portfolio_fit(iteration_metrics),
        check_param_robustness(iteration_params, sweep_history),
    ]


def aggregate_verdict(checks: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce a list of checks to a verdict.

    Rule:
        any blocker fail               → REJECTED
        ≥ WARNING_FAIL_REJECT_THRESHOLD warning fails → REJECTED
        exactly 1 warning fail         → CONDITIONAL_APPROVAL
        no fails                       → APPROVED
    """
    blocker_fails = [c for c in checks if c["severity"] == "blocker" and not c["passed"]]
    warning_fails = [c for c in checks if c["severity"] == "warning" and not c["passed"]]

    if blocker_fails:
        verdict = "REJECTED"
        reason = (
            f"Blocker failed: "
            + "; ".join(c["check_name"] for c in blocker_fails)
        )
    elif len(warning_fails) >= WARNING_FAIL_REJECT_THRESHOLD:
        verdict = "REJECTED"
        reason = (
            f"{len(warning_fails)} warning checks failed "
            f"(>= {WARNING_FAIL_REJECT_THRESHOLD} → REJECTED): "
            + "; ".join(c["check_name"] for c in warning_fails)
        )
    elif len(warning_fails) == 1:
        verdict = "CONDITIONAL_APPROVAL"
        reason = (
            "1 warning check failed (CONDITIONAL_APPROVAL — researcher "
            "may proceed but must address in follow-up): "
            + warning_fails[0]["check_name"]
        )
    else:
        verdict = "APPROVED"
        reason = "All blocker and warning checks passed."

    return {
        "verdict": verdict,
        "reason": reason,
        "n_checks": len(checks),
        "n_blocker_fails": len(blocker_fails),
        "n_warning_fails": len(warning_fails),
        "checks": checks,
    }
