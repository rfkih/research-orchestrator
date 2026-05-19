"""Mechanical skeptic prescreen — deterministic pattern checks.

This runs BEFORE the LLM skeptic agent (when the operator chooses to
invoke it). Catches the cases where the researcher is exhibiting
fishing patterns we can detect without any qualitative reasoning:

  * **archetype_churn_24h** — researcher has tried N distinct
    archetypes in the last 24 hours. High churn = trial-multiplicity
    inflation that DSR's selection-bias correction may not fully
    capture if the trials weren't all logged in ``hypothesis_audit``.
  * **repeat_axis_attempts** — same (strategy_code, axis_set_hash)
    has been queued more than once in the last 14 days. Each attempt
    bites at the multiple-testing budget.
  * **narrow_param_window** — the candidate iteration's params are
    within ε of a recently-DISCARDED neighbour. Suggests the candidate
    is a re-roll on the same axis with a tiny shift.
  * **post_hoc_hypothesis_edit** — the motivating HYPOTHESIS journal
    row's ``updated_time`` is AFTER the plan was written. Pre-
    registration is no longer pre.
  * **dsr_trial_undercount** — cumulative ``hypothesis_audit`` rows
    for the strategy exceed the iteration's ``dsr_n_trials`` by more
    than 20%. Tells the agent that DSR's deflation is too generous.

The prescreen returns each metric + a per-check ``flag`` boolean. The
researcher's loop uses ``any_flag_fired`` to decide whether to invoke
the LLM skeptic — at typical research cadence ~80% of graduations
have no flag and skip the LLM call entirely, saving Max-plan budget.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import asyncpg

# Thresholds. Pinned in tests; operator-controlled. Changing these
# narrows or widens what the prescreen catches.
ARCHETYPE_CHURN_24H_THRESHOLD: int = 4
REPEAT_AXIS_LOOKBACK_DAYS: int = 14
NARROW_PARAM_REL_TOLERANCE: float = 0.10
DSR_UNDERCOUNT_RATIO: float = 1.20


async def skeptic_prescreen(
    conn: asyncpg.Connection,
    *,
    iteration_id: str,
    strategy_code: str,
    motivating_hypothesis_id: str | None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Run all five mechanical pattern checks. Each returns a structured
    metric block; ``any_flag_fired`` is the dispatch hint."""
    now = now or datetime.now(timezone.utc)

    archetype_churn = await _archetype_churn_24h(conn, now=now)
    repeat_axis = await _repeat_axis_attempts(conn, strategy_code=strategy_code, now=now)
    narrow_window = await _narrow_param_window(
        conn, iteration_id=iteration_id, strategy_code=strategy_code
    )
    post_hoc = await _post_hoc_hypothesis_edit(
        conn, hypothesis_id=motivating_hypothesis_id
    )
    dsr_undercount = await _dsr_trial_undercount(
        conn, iteration_id=iteration_id, strategy_code=strategy_code
    )

    checks = [archetype_churn, repeat_axis, narrow_window, post_hoc, dsr_undercount]
    any_flag = any(c.get("flag") for c in checks)
    n_flags = sum(1 for c in checks if c.get("flag"))

    return {
        "iteration_id": iteration_id,
        "strategy_code": strategy_code,
        "checks": checks,
        "any_flag_fired": any_flag,
        "n_flags": n_flags,
        "recommendation": (
            "invoke_skeptic_agent" if any_flag else "skip_skeptic_agent"
        ),
        "evaluated_at": now.isoformat(),
    }


# ── Per-check implementations ───────────────────────────────────────


async def _archetype_churn_24h(
    conn: asyncpg.Connection, *, now: datetime
) -> dict[str, Any]:
    """How many distinct strategy_codes did the researcher queue in
    the last 24 hours? High count = breadth-first fishing."""
    cutoff = now - timedelta(hours=24)
    row = await conn.fetchrow(
        """
        SELECT COUNT(DISTINCT strategy_code)::int AS n_codes
          FROM research_queue
         WHERE created_time >= $1
        """,
        cutoff,
    )
    n = int(row["n_codes"]) if row and row["n_codes"] is not None else 0
    return {
        "check_name": "archetype_churn_24h",
        "metric": n,
        "threshold": ARCHETYPE_CHURN_24H_THRESHOLD,
        "flag": n > ARCHETYPE_CHURN_24H_THRESHOLD,
        "finding": (
            f"{n} distinct strategy_codes queued in last 24h "
            f"(threshold > {ARCHETYPE_CHURN_24H_THRESHOLD})"
        ),
    }


async def _repeat_axis_attempts(
    conn: asyncpg.Connection, *, strategy_code: str, now: datetime
) -> dict[str, Any]:
    """How many distinct queue rows have the same axis-set hash for
    this strategy in the last N days? > 1 = re-running the same
    dimension after a prior outcome."""
    cutoff = now - timedelta(days=REPEAT_AXIS_LOOKBACK_DAYS)
    row = await conn.fetchrow(
        """
        SELECT COUNT(DISTINCT axis_set_hash)::int AS n_axes,
               COUNT(*)::int AS n_attempts
          FROM hypothesis_audit
         WHERE strategy_code = $1
           AND created_time >= $2
        """,
        strategy_code,
        cutoff,
    )
    n_axes = int(row["n_axes"]) if row and row["n_axes"] is not None else 0
    n_attempts = int(row["n_attempts"]) if row and row["n_attempts"] is not None else 0
    # Flag when attempts exceed axes by more than 1.5x — pure
    # multi-trial work on the same axes.
    flag = n_axes > 0 and n_attempts > int(1.5 * n_axes)
    return {
        "check_name": "repeat_axis_attempts",
        "n_distinct_axes": n_axes,
        "n_attempts": n_attempts,
        "lookback_days": REPEAT_AXIS_LOOKBACK_DAYS,
        "flag": flag,
        "finding": (
            f"{n_attempts} hypothesis_audit rows across {n_axes} distinct "
            f"axis-sets in last {REPEAT_AXIS_LOOKBACK_DAYS}d"
        ),
    }


async def _narrow_param_window(
    conn: asyncpg.Connection, *, iteration_id: str, strategy_code: str
) -> dict[str, Any]:
    """Is the candidate's param vector within ε of a recently-DISCARDED
    neighbour? If yes the candidate may be a noise-driven re-roll."""
    cand = await conn.fetchrow(
        """
        SELECT params_snapshot
          FROM research_iteration_log
         WHERE iteration_id = $1::uuid
        """,
        iteration_id,
    )
    if cand is None or cand["params_snapshot"] is None:
        return {
            "check_name": "narrow_param_window",
            "flag": False,
            "finding": "candidate iteration row missing params_snapshot",
        }
    cand_params = dict(cand["params_snapshot"])
    if not cand_params:
        return {
            "check_name": "narrow_param_window",
            "flag": False,
            "finding": "candidate has no params",
        }
    discards = await conn.fetch(
        """
        SELECT iteration_id, params_snapshot
          FROM research_iteration_log
         WHERE strategy_code = $1
           AND verdict = 'DISCARD'
           AND iteration_id <> $2::uuid
         ORDER BY created_time DESC
         LIMIT 20
        """,
        strategy_code,
        iteration_id,
    )
    near_misses: list[dict[str, Any]] = []
    for r in discards:
        if r["params_snapshot"] is None:
            continue
        prior = dict(r["params_snapshot"])
        if set(prior.keys()) != set(cand_params.keys()):
            continue
        # Numeric-only comparison; non-numeric keys must match exactly.
        too_close = True
        for k, v in cand_params.items():
            pv = prior.get(k)
            try:
                fv = float(v)
                fpv = float(pv)
            except (TypeError, ValueError):
                if pv != v:
                    too_close = False
                    break
                continue
            denom = max(abs(fv), abs(fpv), 1e-9)
            if abs(fv - fpv) / denom > NARROW_PARAM_REL_TOLERANCE:
                too_close = False
                break
        if too_close:
            near_misses.append(
                {"iteration_id": str(r["iteration_id"]), "params": prior}
            )
    return {
        "check_name": "narrow_param_window",
        "n_near_discards": len(near_misses),
        "rel_tolerance": NARROW_PARAM_REL_TOLERANCE,
        "flag": len(near_misses) > 0,
        "finding": (
            f"{len(near_misses)} recently-DISCARDED iteration(s) within "
            f"{int(NARROW_PARAM_REL_TOLERANCE * 100)}% on every param"
        ),
        "near_misses": near_misses[:3],
    }


async def _post_hoc_hypothesis_edit(
    conn: asyncpg.Connection, *, hypothesis_id: str | None
) -> dict[str, Any]:
    """Was the motivating HYPOTHESIS journal row updated AFTER its
    creation? Edit-after-create breaks pre-registration."""
    if not hypothesis_id:
        return {
            "check_name": "post_hoc_hypothesis_edit",
            "flag": False,
            "finding": "no motivating_hypothesis_id provided",
        }
    row = await conn.fetchrow(
        """
        SELECT created_time, updated_time, entry_type
          FROM research_journal
         WHERE journal_id = $1::uuid
        """,
        hypothesis_id,
    )
    if row is None:
        return {
            "check_name": "post_hoc_hypothesis_edit",
            "flag": True,  # missing hypothesis = pre-reg unverifiable = flag
            "finding": f"hypothesis_id={hypothesis_id} not found in journal",
        }
    if row["entry_type"] != "HYPOTHESIS":
        return {
            "check_name": "post_hoc_hypothesis_edit",
            "flag": True,
            "finding": (
                f"motivating_hypothesis_id points at entry_type="
                f"{row['entry_type']!r}, not HYPOTHESIS"
            ),
        }
    created = row["created_time"]
    updated = row["updated_time"]
    # Tolerate near-identical timestamps — Flyway-managed columns may
    # both be set in one transaction with the same NOW(). 1 second is
    # a defensible noise floor; any later edit is a real edit.
    edited = (
        created is not None
        and updated is not None
        and (updated - created).total_seconds() > 1.0
    )
    return {
        "check_name": "post_hoc_hypothesis_edit",
        "created_time": created.isoformat() if created else None,
        "updated_time": updated.isoformat() if updated else None,
        "flag": edited,
        "finding": (
            "HYPOTHESIS row updated after creation — pre-registration broken"
            if edited
            else "HYPOTHESIS row unedited since creation"
        ),
    }


async def _dsr_trial_undercount(
    conn: asyncpg.Connection, *, iteration_id: str, strategy_code: str
) -> dict[str, Any]:
    """Does the iteration's ``dsr_n_trials`` materially under-count the
    real cumulative ``hypothesis_audit`` rows for the strategy?

    DSR's deflation assumes ``dsr_n_trials`` captures every prior
    trial. When the researcher submitted backtests outside the audit
    path (rare but possible — e.g. operator-run smoke tests, ad-hoc
    paired-backtests for ML evaluation), the true multiplicity is
    higher. Flagging this prompts the LLM skeptic to recompute DSR
    against the actual audit count."""
    iter_row = await conn.fetchrow(
        """
        SELECT (metrics_snapshot->'analysis'->>'dsr_n_trials')::int AS dsr_n
          FROM research_iteration_log
         WHERE iteration_id = $1::uuid
        """,
        iteration_id,
    )
    if iter_row is None or iter_row["dsr_n"] is None:
        return {
            "check_name": "dsr_trial_undercount",
            "flag": False,
            "finding": "iteration row missing dsr_n_trials",
        }
    dsr_n = int(iter_row["dsr_n"])
    audit_row = await conn.fetchrow(
        """
        SELECT COUNT(*)::int AS n
          FROM hypothesis_audit
         WHERE strategy_code = $1
        """,
        strategy_code,
    )
    real_n = int(audit_row["n"]) if audit_row and audit_row["n"] is not None else 0
    if dsr_n <= 0:
        return {
            "check_name": "dsr_trial_undercount",
            "dsr_n_trials": dsr_n,
            "real_audit_count": real_n,
            "flag": False,
            "finding": "dsr_n_trials is zero — nothing to compare",
        }
    ratio = real_n / dsr_n
    return {
        "check_name": "dsr_trial_undercount",
        "dsr_n_trials": dsr_n,
        "real_audit_count": real_n,
        "ratio": round(ratio, 3),
        "threshold_ratio": DSR_UNDERCOUNT_RATIO,
        "flag": ratio > DSR_UNDERCOUNT_RATIO,
        "finding": (
            f"hypothesis_audit count {real_n} vs dsr_n_trials {dsr_n} "
            f"(ratio {round(ratio, 3)}); "
            f"{'undercount > threshold' if ratio > DSR_UNDERCOUNT_RATIO else 'within tolerance'}"
        ),
    }
