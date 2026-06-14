"""``hypothesis_audit`` reads + writes.

Schema reference: ``V41__create_hypothesis_audit.sql``. This table is
the orchestrator's ground-truth for "how many trials has this strategy
seen" — drives DSR multiplicity in ``services/analyze.py`` and the
re-discovery gate in ``api/queue.py``.

Hashing convention (must match across writers + readers):

  * ``axis_set_hash``    = sha256(",".join(sorted(param_names)))
  * ``param_combo_hash`` = sha256(",".join(f"{k}={v}" for k, v in
                          sorted(combo.items())))

Hashes are stable across processes because they sort first. Do not
change the join character or the sort order without bumping the
schema — old rows would silently fail to match.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable
from uuid import UUID

import asyncpg


def axis_set_hash(param_names: Iterable[str]) -> str:
    """SHA-256 of the sorted, comma-joined parameter NAMES.

    Identifies the dimension set being explored. Two sweeps with the
    same axes but different value lists collide here on purpose —
    that's what lets the gate say "agent has already explored this
    axis combo".
    """
    canonical = ",".join(sorted(str(n) for n in param_names))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def param_combo_hash(combo: dict[str, Any]) -> str:
    """SHA-256 of the sorted ``key=value`` combo. Identifies the EXACT cell."""
    parts = [f"{k}={_canon(v)}" for k, v in sorted(combo.items())]
    canonical = ",".join(parts)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _canon(v: Any) -> str:
    """Stable string form for hashing — ``json.dumps`` w/ sort_keys.

    Numerics are normalised so that 1, 1.0, and "1" hash identically:
    the agent shouldn't be able to dodge the dedup gate by passing
    "0.5" vs 0.5 vs Decimal("0.5") for the same threshold.
    """
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (dict, list)):
        return json.dumps(v, sort_keys=True, separators=(",", ":"))
    if isinstance(v, (int, float)):
        f = float(v)
        return str(int(f)) if f.is_integer() else repr(f)
    s = str(v).strip()
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else repr(f)
    except (TypeError, ValueError):
        return s


async def insert_audit(
    conn: asyncpg.Connection,
    *,
    strategy_code: str,
    symbol: str,
    interval_name: str,
    params_snapshot: dict[str, Any],
    queue_id: UUID | None,
    created_by: str,
    decision_verdict: str | None = None,
) -> UUID:
    """Record a trial attempt. Called BEFORE the backtest submits so a
    crashed tick still leaves an audit trail. Returns the audit_id so
    the caller can ``update_audit_verdict`` after the iteration logs.

    Empty combos are still recorded (axis_set is empty string hash) —
    a no-param sweep still consumes one trial of multiplicity.

    ``symbol`` + ``interval_name`` denormalise the data-universe identity
    onto the audit row (V93). Required so ``count_data_universe_trials``
    can scope DSR n_trials by (symbol, interval) without a JOIN to
    research_queue. The hard-rule allowlist (BTCUSDT/ETHUSDT × 5m/15m/
    1h/4h) is enforced at the DB layer by V93's CHECK constraints — pass
    junk values and the INSERT will raise.

    ``decision_verdict`` is normally NULL at insert (step 5 backfills it).
    The null-screen passes ``'NULL_SCREEN_DRAW'`` for its completed draws
    so they count toward data-universe multiplicity without an iteration
    row (see ``count_data_universe_trials``).
    """
    names = list(params_snapshot.keys())
    a_hash = axis_set_hash(names)
    c_hash = param_combo_hash(params_snapshot)
    row = await conn.fetchrow(
        """
        INSERT INTO hypothesis_audit (
            audit_id, strategy_code, symbol, interval_name,
            axis_set_hash, param_combo_hash,
            params_snapshot, queue_id, created_by, decision_verdict,
            created_time, updated_time
        ) VALUES (
            gen_random_uuid(), $1, $2, $3, $4, $5, $6, $7, $8, $9, NOW(), NOW()
        )
        RETURNING audit_id
        """,
        strategy_code,
        symbol,
        interval_name,
        a_hash,
        c_hash,
        params_snapshot,  # asyncpg's jsonb codec encodes the dict (infra/db.py)
        queue_id,
        created_by,
        decision_verdict,
    )
    return row["audit_id"]


async def latest_unattached_for_queue(
    conn: asyncpg.Connection, queue_id: UUID
) -> dict[str, Any] | None:
    """Most recent audit row for this queue with no iteration attached.

    The resume path uses it to recover the exact combo a prior (dead)
    attempt submitted: step 3.5 wrote this row immediately before the
    submit, so its ``params_snapshot`` is authoritative for the in-flight
    run — re-deriving the combo would diverge for TPE sweeps, whose
    sampler is not deterministic across processes.
    """
    row = await conn.fetchrow(
        """
        SELECT audit_id, params_snapshot, created_time
          FROM hypothesis_audit
         WHERE queue_id = $1
           AND iteration_id IS NULL
           AND decision_verdict IS DISTINCT FROM 'NULL_SCREEN_DRAW'
         ORDER BY created_time DESC
         LIMIT 1
        """,
        queue_id,
    )
    return dict(row) if row else None


async def update_audit_verdict(
    conn: asyncpg.Connection,
    *,
    audit_id: UUID,
    iteration_id: UUID,
    statistical_verdict: str | None,
    decision_verdict: str | None,
) -> None:
    """Backfill the audit row once the iteration log is written. The
    re-discovery gate keys on ``decision_verdict='DISCARD'``; without
    this update step the gate would never fire."""
    await conn.execute(
        """
        UPDATE hypothesis_audit
           SET iteration_id = $2,
               statistical_verdict = $3,
               decision_verdict = $4,
               updated_time = NOW()
         WHERE audit_id = $1
        """,
        audit_id,
        iteration_id,
        statistical_verdict,
        decision_verdict,
    )


# REMOVED 2026-06-09: ``count_cumulative_trials(strategy_code)``. It scoped DSR
# multiplicity by strategy_code, which RESET the selection-bias counter on every
# archetype pivot — exactly when multiplicity compounds. The S3 swap to
# ``count_data_universe_trials`` (scoped by symbol×interval) is fully live in
# both callers (tick.py:556, walk_forward.py:506), so the per-strategy counter
# was dead code. Deleted to prevent a future caller from re-introducing the
# under-deflation bug. The (symbol, interval) data universe is the honest
# family-wise count — it aggregates trials across ALL archetypes on the same
# bars, which is the real multiple-comparisons surface.


async def count_data_universe_trials(
    conn: asyncpg.Connection,
    symbol: str,
    interval_name: str,
    external_declared: int = 0,
) -> int:
    """COMPLETED trials on this data universe. Scopes DSR ``n_trials`` by
    (symbol, interval_name) — the set of bars the autonomous loop has
    actually inspected — rather than by strategy_code (which resets on
    every archetype pivot, leaking selection bias).

    ``external_declared`` (2026-06-14 re-audit HOLE-1): off-orchestrator
    trials the audit ledger CANNOT see — offline Python sweeps, hand-tuning,
    direct JVM backtests — declared by the caller at enqueue / walk-forward
    submission and ADDED into the count. The DB count is a structural lower
    bound on multiplicity; declaring the off-ledger trials raises SR* and
    deflates DSR, closing the one fail-open hole (undercounted n_trials →
    DSR biased UP → gate too lenient). Clamped ≥ 0 — it can only ever raise
    the trial count, never lower it, so the gate can only TIGHTEN (V11-safe).

    Pre-V93 audit rows whose data-universe identity is NULL are excluded
    implicitly: the equality comparison ``symbol = $1`` returns NULL (not
    true) when ``symbol`` is NULL, so those rows never match. The partial
    index ``idx_hypothesis_audit_data_universe`` (V93) is defined with
    ``symbol IS NOT NULL AND interval_name IS NOT NULL AND iteration_id
    IS NOT NULL`` so the planner can prove the query predicate implies the
    index predicate when $1 / $2 are non-NULL.

    iteration_id IS NOT NULL semantics: crashed/aborted ticks are kept for
    forensics but don't count toward multiplicity (an infra failure isn't
    selection bias).

    Null-screen draws DO count (decision_verdict='NULL_SCREEN_DRAW', no
    iteration row): the screen's K completed backtests SELECT which
    archetypes proceed to a sweep — that is exactly the selection process
    Bailey-LdP n_trials must include. Excluding them (the pre-2026-06-12
    behaviour) silently granted every post-screen sweep ~K uncounted
    trials.

    Callers must add 1 for the in-flight trial they're about to log;
    its audit row has iteration_id=NULL until step 5 backfills it.
    """
    val = await conn.fetchval(
        """
        SELECT COUNT(*)::int FROM hypothesis_audit
         WHERE symbol = $1
           AND interval_name = $2
           AND (iteration_id IS NOT NULL
                OR decision_verdict = 'NULL_SCREEN_DRAW')
        """,
        symbol,
        interval_name,
    )
    try:
        extra = max(0, int(external_declared or 0))
    except (TypeError, ValueError, OverflowError):
        extra = 0
    return int(val or 0) + extra


async def count_axis_trials(
    conn: asyncpg.Connection, strategy_code: str, axis_set_hash_value: str
) -> int:
    """Trials this strategy has run on this exact axis set."""
    val = await conn.fetchval(
        """
        SELECT COUNT(*)::int FROM hypothesis_audit
         WHERE strategy_code = $1 AND axis_set_hash = $2
        """,
        strategy_code,
        axis_set_hash_value,
    )
    return int(val or 0)


async def axis_has_discard(
    conn: asyncpg.Connection,
    strategy_code: str,
    axis_set_hash_value: str,
    *,
    track: str | None = None,
) -> dict[str, Any] | None:
    """Return the most recent DISCARD audit row for this strategy+axis,
    or None. The re-discovery gate uses this to 409 fresh sweeps that
    revisit a dimension the agent already wrote off.

    Why most-recent: lets the response carry the iteration_id +
    created_time so the caller can journal-cite the prior outcome.

    Dual-track scoping (Phase 1): the DISCARD's track lives on the
    originating ``research_queue`` row's ``sweep_config->>'track'`` (the
    audit table has no track column). We LEFT JOIN to it on ``queue_id``
    so the gate can filter per-track:

      * ``track=None`` (legacy global enqueue) ⇒ no track predicate; ANY
        prior discard blocks, exactly as before this change.
      * ``track`` set ⇒ a discard blocks only when its queue row's track
        equals ``track`` OR is NULL (an un-tracked/global discard still
        blocks every track — never weaken the gate). A discard tagged to a
        DIFFERENT track does not block, so a hedging discard won't reject a
        trading axis-set.

    Audit rows whose ``queue_id`` is NULL (e.g. null-screen trials) keep
    their pre-track behaviour: the LEFT JOIN leaves ``rq.sweep_config``
    NULL, which the COALESCE treats as a global discard.
    """
    row = await conn.fetchrow(
        """
        SELECT ha.audit_id, ha.iteration_id, ha.params_snapshot,
               ha.created_time, ha.created_by
          FROM hypothesis_audit ha
          LEFT JOIN research_queue rq ON rq.queue_id = ha.queue_id
         WHERE ha.strategy_code = $1
           AND ha.axis_set_hash = $2
           AND ha.decision_verdict = 'DISCARD'
           AND ($3::text IS NULL OR rq.sweep_config->>'track' IS NULL
                OR rq.sweep_config->>'track' = $3)
         ORDER BY ha.created_time DESC
         LIMIT 1
        """,
        strategy_code,
        axis_set_hash_value,
        track,
    )
    return dict(row) if row else None


async def combo_already_tested(
    conn: asyncpg.Connection, strategy_code: str, combo_hash: str
) -> dict[str, Any] | None:
    """Most recent audit for an EXACT combo. Used for soft warnings —
    the gate doesn't enforce combo-level dedup (sometimes you legit
    want to re-run the same cell, e.g. after a data backfill)."""
    row = await conn.fetchrow(
        """
        SELECT audit_id, iteration_id, decision_verdict, statistical_verdict,
               created_time
          FROM hypothesis_audit
         WHERE strategy_code = $1 AND param_combo_hash = $2
         ORDER BY created_time DESC
         LIMIT 1
        """,
        strategy_code,
        combo_hash,
    )
    return dict(row) if row else None
