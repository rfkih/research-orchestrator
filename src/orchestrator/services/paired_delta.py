"""Paired-delta analytics — Phase E (2026-05-19).

Closes the closed-loop ML lifecycle. After a Phase B sweep with an
ML sentinel axis (e.g. ``_ml_gate_enabled: [true, false]``) drains,
this service partitions iteration_log rows into matched pairs holding
all non-ML params constant, computes per-pair deltas on the operator's
chosen metric, applies a paired bootstrap for the CI, and emits a
verdict:

  * ``POSITIVE_DELTA`` — mean delta significantly > 0 (CI lower > 0).
    The model adds positive expected value on this surface.
  * ``NEGATIVE_DELTA`` — mean delta significantly < 0 (CI upper < 0).
    The model DESTROYS value (see Phase 4 Stage H lesson). Researcher
    MUST NOT graduate; reject the model.
  * ``NEUTRAL`` — CI straddles zero AND mean magnitude is small.
    The model is no better than no-gate; defer or pivot.
  * ``INDETERMINATE`` — fewer than ``MIN_PAIRS_FOR_VERDICT`` pairs,
    or CI is too wide to call. More iterations needed.

The verdict is BINDING — the researcher's loop blocks graduation on
NEGATIVE_DELTA. Phase 4 Stage H showed that without this check the
operator manually compared sharpes; that's the failure mode this
service exists to close.

Stats notes
-----------
Backtests are deterministic — running the same (params, seed) twice
produces the same metrics. So per-pair variance is zero; the
statistical signal is whether the mean across pairs is meaningfully
nonzero. Bootstrap is over the pair index, not over within-pair noise.

The default primary metric is Sharpe ratio because Phase 4 Stage H
collapsed Sharpe (-1.23 → -3.13); but the body can name any of
{sharpe, return, profit_factor, win_rate, trade_count}. NEGATIVE on
trade_count alone (gate suppressed trades) is NOT a NEGATIVE_DELTA
verdict — that's just a description; the verdict requires a metric
where direction has clear "good" semantics.
"""

from __future__ import annotations

import random
from typing import Any, Literal
from uuid import UUID

import asyncpg

from ..errors import OrchestratorError


# Tuning knobs. Pinned in tests.
# MIN_PAIRS raised 3 -> 8 (2026-06-12): with n=3 same-sign deltas the
# percentile bootstrap CI excludes zero with probability 1, and under a
# sign-symmetric null P(all 3 same sign) = 25% -- a 12.5% false-positive
# rate PER DIRECTION on a verdict that can block graduation
# (NEGATIVE_DELTA). At n=8 the same accident costs P = 2/2^8 ~ 0.8%.
MIN_PAIRS_FOR_VERDICT: int = 8
DEFAULT_BOOTSTRAP_REPS: int = 1000
DEFAULT_CI_LEVEL: float = 0.95
NEUTRAL_MAGNITUDE_THRESHOLD: float = 0.05  # |mean_delta| below this → NEUTRAL
DEFAULT_SEED: int = 42

# Recognized ML sentinel param names (matches services.sweep.ML_SENTINELS).
_ML_SENTINELS: frozenset[str] = frozenset({
    "_ml_gate_enabled", "_ml_signal_name", "_ml_shadow_mode",
})

# Metric → JSONB path in metrics_snapshot. Trade_count uses a different
# scale; the verdict logic special-cases it (no good/bad direction).
_METRIC_KEYS: dict[str, str] = {
    "sharpe": "sharpe_ratio",
    "return": "return_pct",
    "profit_factor": "profit_factor",
    "win_rate": "win_rate",
    "trade_count": "trade_count",
}

# Metrics where "higher is better" — others (e.g. drawdown) would need
# sign-flipped logic if added later.
_HIGHER_IS_BETTER: frozenset[str] = frozenset(
    {"sharpe", "return", "profit_factor", "win_rate"}
)


PrimaryMetric = Literal["sharpe", "return", "profit_factor", "win_rate", "trade_count"]


async def compute_paired_delta(
    conn: asyncpg.Connection,
    *,
    queue_id: UUID | None = None,
    iteration_ids: list[UUID] | None = None,
    primary_metric: PrimaryMetric = "sharpe",
    treatment_key: str = "_ml_gate_enabled",
    treatment_on_value: Any = True,
    treatment_off_value: Any = False,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    ci_level: float = DEFAULT_CI_LEVEL,
    rng_seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Run the paired-delta analysis.

    Exactly one of ``queue_id`` or ``iteration_ids`` must be set.
    ``treatment_key`` defaults to ``_ml_gate_enabled`` (the most common
    "did the gate help" comparison); pass another sentinel to compare
    model variants (e.g. ``_ml_signal_name`` with on='regime_btc_v4'
    off='regime_btc_v3')."""
    if (queue_id is None) == (iteration_ids is None):
        raise OrchestratorError(
            status_code=400,
            error_code="paired_delta_target_required",
            message=(
                "compute_paired_delta requires exactly one of queue_id, "
                "iteration_ids; got both or neither"
            ),
            retryable=False,
        )
    if treatment_key not in _ML_SENTINELS:
        raise OrchestratorError(
            status_code=400,
            error_code="paired_delta_bad_treatment_key",
            message=(
                f"treatment_key={treatment_key!r} is not a recognized ML "
                f"sentinel; expected one of {sorted(_ML_SENTINELS)}"
            ),
            retryable=False,
        )
    if primary_metric not in _METRIC_KEYS:
        raise OrchestratorError(
            status_code=400,
            error_code="paired_delta_bad_metric",
            message=(
                f"primary_metric={primary_metric!r} unsupported; expected "
                f"one of {sorted(_METRIC_KEYS)}"
            ),
            retryable=False,
        )

    rows = await _fetch_rows(conn, queue_id=queue_id, iteration_ids=iteration_ids)
    pairs, unpaired = _pair_iterations(
        rows,
        treatment_key=treatment_key,
        on_value=treatment_on_value,
        off_value=treatment_off_value,
    )
    # F3 (2026-05-19) — _compute_delta_records bundles the pair
    # metadata with each delta so we can't get out-of-sync with `pairs`.
    # n_pairs counts pairs that produced a usable delta; pairs dropped
    # for missing metrics are summarised in n_dropped_missing_metric.
    delta_records = _compute_delta_records(pairs, primary_metric=primary_metric)
    deltas = [r["delta"] for r in delta_records]
    n_pairs = len(delta_records)
    n_dropped_missing_metric = len(pairs) - n_pairs

    if n_pairs < MIN_PAIRS_FOR_VERDICT:
        verdict = "INDETERMINATE"
        ci_low: float | None = None
        ci_high: float | None = None
        mean_delta: float | None = (
            sum(deltas) / n_pairs if n_pairs > 0 else None
        )
    else:
        mean_delta = sum(deltas) / n_pairs
        ci_low, ci_high = _paired_bootstrap_ci(
            deltas, n_reps=bootstrap_reps, ci_level=ci_level, seed=rng_seed
        )
        verdict = _decide_verdict(
            mean=mean_delta, ci_low=ci_low, ci_high=ci_high,
            primary_metric=primary_metric,
        )

    return {
        "queue_id": str(queue_id) if queue_id else None,
        "iteration_ids": (
            [str(i) for i in iteration_ids] if iteration_ids else None
        ),
        "primary_metric": primary_metric,
        "treatment_key": treatment_key,
        "treatment_on_value": treatment_on_value,
        "treatment_off_value": treatment_off_value,
        "n_pairs": n_pairs,
        "n_unpaired": len(unpaired),
        "n_dropped_missing_metric": n_dropped_missing_metric,
        "mean_delta": (
            round(mean_delta, 6) if mean_delta is not None else None
        ),
        "ci_level": ci_level,
        "ci_low": round(ci_low, 6) if ci_low is not None else None,
        "ci_high": round(ci_high, 6) if ci_high is not None else None,
        "bootstrap_reps": bootstrap_reps if n_pairs >= MIN_PAIRS_FOR_VERDICT else 0,
        "verdict": verdict,
        "pairs": delta_records,
        "unpaired_keys": unpaired,
    }


# ── Repo helper (small, kept here so the service is self-contained) ─


async def _fetch_rows(
    conn: asyncpg.Connection,
    *,
    queue_id: UUID | None,
    iteration_ids: list[UUID] | None,
) -> list[dict[str, Any]]:
    if queue_id is not None:
        rows = await conn.fetch(
            """
            SELECT il.iteration_id, il.params_snapshot, il.metrics_snapshot,
                   il.verdict, il.statistical_verdict
              FROM research_iteration_log il
              JOIN hypothesis_audit ha ON ha.iteration_id = il.iteration_id
             WHERE ha.queue_id = $1
             ORDER BY il.iteration_number
            """,
            queue_id,
        )
    else:
        assert iteration_ids is not None
        rows = await conn.fetch(
            """
            SELECT iteration_id, params_snapshot, metrics_snapshot,
                   verdict, statistical_verdict
              FROM research_iteration_log
             WHERE iteration_id = ANY($1::uuid[])
            """,
            list(iteration_ids),
        )
    return [dict(r) for r in rows]


# ── Pure functions (testable without DB) ────────────────────────────


def _pair_iterations(
    rows: list[dict[str, Any]],
    *,
    treatment_key: str,
    on_value: Any,
    off_value: Any,
) -> tuple[list[tuple[str, dict[str, Any], dict[str, Any]]], list[str]]:
    """Group rows by their non-ML params. For each group with one
    on-row and one off-row, emit a paired triple ``(key, on, off)``.
    Returns the pairs + a list of group keys that DIDN'T form a clean
    pair (for the "unpaired_keys" forensic field).

    "Non-ML params" means: take params_snapshot, drop ALL ML sentinel
    keys (not just the treatment_key) — the rest must match. This
    guards against accidentally pairing rows that vary in
    ``_ml_signal_name`` while we're treating on ``_ml_gate_enabled``.
    """
    by_key: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        params = dict(row.get("params_snapshot") or {})
        treat = params.get(treatment_key)
        non_ml = {k: v for k, v in params.items() if k not in _ML_SENTINELS}
        key = _stable_key(non_ml)
        bucket = by_key.setdefault(key, {"on": [], "off": [], "other": []})
        if treat == on_value:
            bucket["on"].append(row)
        elif treat == off_value:
            bucket["off"].append(row)
        else:
            bucket["other"].append(row)

    pairs: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    unpaired: list[str] = []
    for key, bucket in by_key.items():
        if len(bucket["on"]) == 1 and len(bucket["off"]) == 1:
            pairs.append((key, bucket["on"][0], bucket["off"][0]))
        else:
            unpaired.append(key)
    return pairs, unpaired


def _compute_delta_records(
    pairs: list[tuple[str, dict[str, Any], dict[str, Any]]],
    *,
    primary_metric: PrimaryMetric,
) -> list[dict[str, Any]]:
    """For each (key, on, off) triple, compute on_metric - off_metric
    and bundle it with the pair metadata. Skips pairs where either side
    lacks the metric (rare — failed-iteration rows can have empty
    metrics_snapshot).

    Returns ``list[dict]`` so the API response can ship each delta with
    its parent identifiers. Replaces the older :func:`_compute_deltas`
    which returned a parallel list of floats — that shape forced a
    fragile zip(strict=True) against ``pairs`` and crashed the endpoint
    when any pair was dropped (F3 fix, 2026-05-19)."""
    metric_key = _METRIC_KEYS[primary_metric]
    records: list[dict[str, Any]] = []
    for pair_key, on_row, off_row in pairs:
        on_v = _metric_value(on_row, metric_key)
        off_v = _metric_value(off_row, metric_key)
        if on_v is None or off_v is None:
            continue
        records.append({
            "pair_key": pair_key,
            "on_iteration_id": str(on_row["iteration_id"]),
            "off_iteration_id": str(off_row["iteration_id"]),
            "on_metric": round(on_v, 6),
            "off_metric": round(off_v, 6),
            "delta": round(on_v - off_v, 6),
        })
    return records


def _compute_deltas(
    pairs: list[tuple[str, dict[str, Any], dict[str, Any]]],
    *,
    primary_metric: PrimaryMetric,
) -> list[float]:
    """Backward-compat shim — returns just the delta floats. Kept for
    existing tests that exercised the pure-function shape. New code
    should use :func:`_compute_delta_records`."""
    return [r["delta"] for r in _compute_delta_records(
        pairs, primary_metric=primary_metric
    )]


def _metric_value(row: dict[str, Any], metric_key: str) -> float | None:
    snap = row.get("metrics_snapshot") or {}
    if not isinstance(snap, dict):
        return None
    v = snap.get(metric_key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _paired_bootstrap_ci(
    deltas: list[float],
    *,
    n_reps: int,
    ci_level: float,
    seed: int,
) -> tuple[float, float]:
    """Paired bootstrap on the pair index. Resample WITH replacement
    n_reps times, recompute the mean delta each time, take the
    [α/2, 1−α/2] quantiles. Pure-Python — keeps the dependency footprint
    minimal and works on small N (5-30 pairs is the typical regime)."""
    if not deltas:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(deltas)
    means: list[float] = []
    for _ in range(n_reps):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = 1.0 - ci_level
    lo_idx = int(alpha / 2 * n_reps)
    hi_idx = int((1.0 - alpha / 2) * n_reps) - 1
    lo_idx = max(0, min(n_reps - 1, lo_idx))
    hi_idx = max(0, min(n_reps - 1, hi_idx))
    return means[lo_idx], means[hi_idx]


def _decide_verdict(
    *,
    mean: float,
    ci_low: float,
    ci_high: float,
    primary_metric: str,
) -> str:
    """Map (mean, CI) → verdict.

    Higher-is-better metrics:
      * CI lower > 0 → POSITIVE_DELTA (significant gain)
      * CI upper < 0 → NEGATIVE_DELTA (significant loss)
      * CI straddles 0 AND |mean| < NEUTRAL_MAGNITUDE_THRESHOLD → NEUTRAL
      * Otherwise → INDETERMINATE (wide CI, can't call)

    trade_count is neither good nor bad on its own — we still compute
    the delta and CI, but the verdict is NEUTRAL by default so the
    researcher reads it as descriptive rather than gating."""
    if primary_metric == "trade_count":
        return "NEUTRAL"
    if primary_metric not in _HIGHER_IS_BETTER:
        return "INDETERMINATE"

    if ci_low > 0:
        return "POSITIVE_DELTA"
    if ci_high < 0:
        return "NEGATIVE_DELTA"
    if abs(mean) < NEUTRAL_MAGNITUDE_THRESHOLD:
        return "NEUTRAL"
    return "INDETERMINATE"


def _stable_key(d: dict[str, Any]) -> str:
    """Hashable, comparable representation of a param dict. Sorted
    by key so the same set always produces the same string."""
    items = sorted((str(k), str(v)) for k, v in d.items())
    return "|".join(f"{k}={v}" for k, v in items)
