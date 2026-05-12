"""Archetype null-screen — pre-sweep edge sniff test.

Before burning a V11 sweep on an archetype, run K random param draws over
the archetype's parameter ranges and inspect the *distribution* of profit
factors. The decision rule is pre-registered (pinned in tests) and cheap
relative to running 5 V11 iterations to discover an archetype is dead.

Why this exists: the autonomous loop's grid sweeps are confirmatory tools
— they produce a single SIGNIFICANT_EDGE / NO_EDGE / INSUFFICIENT verdict
per cell. They do not characterize the *landscape*. An archetype with no
edge produces PFs centered at ~1.0 with no right-tail mass; an archetype
with edge produces a right-skewed distribution where a meaningful fraction
of cells clear PF >= 1.2. The null-screen surfaces the difference in
~K backtests instead of having to discover it via 5+ failed V11 ticks.

Decision rule (pinned):
    EDGE_PRESENT          if  P75(PF) >= 1.2 AND share(PF >= 1.2) >= 0.25
    NO_EDGE_DETECTED      if  P95(PF) <  1.2 OR share(PF >= 1.2) <  0.10
    INCONCLUSIVE          otherwise
    INSUFFICIENT_DATA     if  fewer than ceil(K/2) backtests produced a
                              finite PF (JVM failures, n<5 trades, etc.)

The screen does NOT write to ``hypothesis_audit`` (would inflate the DSR
multiplicity counter for the legitimate sweep that follows) or
``research_iteration_log`` (an exploratory landscape probe is not an
audit-grade trial). Result is journaled as ``NULL_SCREEN_RESULT``.

Pure functions are testable in isolation; the orchestration helper
shells out to the JVM the same way ``services/tick.py`` does.
"""

from __future__ import annotations

import math
import random
import statistics
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import asyncpg

from ..clients.jvm import JvmClient
from ..config import Settings
from ..errors import OrchestratorError
from ..infra.db import Database
from ..logging import get_logger
from .tick import (
    _build_submit_payload,
    _fetch_run_metrics,
    _poll_backtest_status,
    _resolve_account_strategy,
)

log = get_logger(__name__)

# Pinned thresholds — operator-controlled constants. Changing these shifts
# the screen's strictness; do it only with a documented journal entry and
# update the pin tests in tests/test_null_screen.py.
P75_EDGE_THRESHOLD = 1.2
P95_NO_EDGE_THRESHOLD = 1.2
SHARE_ABOVE_EDGE_FRACTION = 0.25
SHARE_ABOVE_NO_EDGE_FRACTION = 0.10
MIN_USEFUL_PF_FRACTION = 0.5  # need >= K/2 finite PFs to decide

# Default trial count. K=8 trades off ~8 × ~5min JVM time ≈ 40min vs the
# ~50min cost of 5 V11 iterations on a dead archetype. Operator can override.
DEFAULT_N_DRAWS = 8

# Bail the screen after this many consecutive failed draws. A dead JVM
# would otherwise burn through all K draws before reporting; bailing
# saves ~30-45min when the infra is broken. Pinned in tests.
MAX_CONSECUTIVE_FAILURES = 3


# ── Random combo generation (pure) ────────────────────────────────────


def generate_random_combos(
    param_ranges: list[dict[str, Any]],
    n_draws: int,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Generate ``n_draws`` random parameter combos from ``param_ranges``.

    Each ``param_ranges`` entry is one of:
      * ``{"name": "X", "type": "float",  "low": "1.0", "high": "3.0"}``
      * ``{"name": "X", "type": "int",    "low": "10",  "high": "40"}``
      * ``{"name": "X", "type": "choice", "values": ["a","b","c"]}``

    BigDecimal-as-string convention matches the JVM side. Numeric draws
    are rounded to 4 decimals to keep param hashing stable across reruns
    of the same seed.
    """
    rng = random.Random(seed)
    out: list[dict[str, str]] = []
    for _ in range(n_draws):
        combo: dict[str, str] = {}
        for spec in param_ranges:
            name = spec["name"]
            ptype = spec.get("type", "float")
            if ptype == "choice":
                vals = spec.get("values") or []
                if not vals:
                    raise OrchestratorError(
                        status_code=400,
                        error_code="bad_param_range",
                        message=f"choice param {name!r} has no values",
                        retryable=False,
                    )
                combo[name] = str(rng.choice(vals))
            elif ptype == "int":
                lo = int(float(spec["low"]))
                hi = int(float(spec["high"]))
                if hi < lo:
                    raise OrchestratorError(
                        status_code=400,
                        error_code="bad_param_range",
                        message=f"int param {name!r}: low > high",
                        retryable=False,
                    )
                combo[name] = str(rng.randint(lo, hi))
            else:  # float
                lo_f = float(spec["low"])
                hi_f = float(spec["high"])
                if hi_f < lo_f:
                    raise OrchestratorError(
                        status_code=400,
                        error_code="bad_param_range",
                        message=f"float param {name!r}: low > high",
                        retryable=False,
                    )
                v = rng.uniform(lo_f, hi_f)
                combo[name] = f"{round(v, 4)}"
        out.append(combo)
    return out


# ── Distribution + verdict (pure) ─────────────────────────────────────


def _percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolation percentile (q in [0,1]). Returns None for empty."""
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo_idx = int(math.floor(pos))
    hi_idx = int(math.ceil(pos))
    if lo_idx == hi_idx:
        return s[lo_idx]
    frac = pos - lo_idx
    return s[lo_idx] * (1 - frac) + s[hi_idx] * frac


def summarize_pf_distribution(pfs: list[float | None]) -> dict[str, Any]:
    """Summary stats on the PF list. ``None`` entries (failed runs) are
    counted in ``n_attempted`` but excluded from the distribution math."""
    finite = [p for p in pfs if p is not None and math.isfinite(p)]
    n_attempted = len(pfs)
    n_finite = len(finite)
    if not finite:
        return {
            "n_attempted": n_attempted,
            "n_finite": 0,
            "mean": None,
            "median": None,
            "p25": None,
            "p75": None,
            "p95": None,
            "share_pf_ge_1_0": None,
            "share_pf_ge_1_2": None,
            "max": None,
        }
    return {
        "n_attempted": n_attempted,
        "n_finite": n_finite,
        "mean": round(statistics.fmean(finite), 4),
        "median": round(statistics.median(finite), 4),
        "p25": round(_percentile(finite, 0.25) or 0.0, 4),
        "p75": round(_percentile(finite, 0.75) or 0.0, 4),
        "p95": round(_percentile(finite, 0.95) or 0.0, 4),
        "share_pf_ge_1_0": round(sum(1 for p in finite if p >= 1.0) / n_finite, 4),
        "share_pf_ge_1_2": round(sum(1 for p in finite if p >= 1.2) / n_finite, 4),
        "max": round(max(finite), 4),
    }


def null_screen_verdict(distribution: dict[str, Any], n_attempted: int) -> dict[str, str]:
    """Map the PF distribution to a verdict. See module docstring for rule."""
    n_finite = int(distribution.get("n_finite") or 0)
    min_useful = max(1, math.ceil(n_attempted * MIN_USEFUL_PF_FRACTION))
    if n_finite < min_useful:
        return {
            "verdict": "INSUFFICIENT_DATA",
            "reason": (
                f"Only {n_finite}/{n_attempted} draws produced a finite PF "
                f"(need >= {min_useful}). Likely JVM failures or n<5 trades."
            ),
        }
    p75 = distribution.get("p75")
    p95 = distribution.get("p95")
    share_edge = distribution.get("share_pf_ge_1_2")
    # Fallback if any field is missing (defensive — summarize_pf_distribution
    # populates them for any non-empty finite list).
    if p75 is None or p95 is None or share_edge is None:
        return {"verdict": "INSUFFICIENT_DATA", "reason": "Distribution missing fields."}
    if p75 >= P75_EDGE_THRESHOLD and share_edge >= SHARE_ABOVE_EDGE_FRACTION:
        return {
            "verdict": "EDGE_PRESENT",
            "reason": (
                f"P75(PF)={p75} >= {P75_EDGE_THRESHOLD} AND share(PF>=1.2)="
                f"{share_edge} >= {SHARE_ABOVE_EDGE_FRACTION}. Right-tail "
                f"mass suggests an exploitable region; sweep justified."
            ),
        }
    if p95 < P95_NO_EDGE_THRESHOLD or share_edge < SHARE_ABOVE_NO_EDGE_FRACTION:
        return {
            "verdict": "NO_EDGE_DETECTED",
            "reason": (
                f"P95(PF)={p95} < {P95_NO_EDGE_THRESHOLD} OR share(PF>=1.2)="
                f"{share_edge} < {SHARE_ABOVE_NO_EDGE_FRACTION}. PF "
                f"distribution centered at noise; archetype likely dead "
                f"on this axis set."
            ),
        }
    return {
        "verdict": "INCONCLUSIVE",
        "reason": (
            f"P75(PF)={p75}, P95(PF)={p95}, share(PF>=1.2)={share_edge}. "
            f"Right-tail too thin for EDGE_PRESENT but P95 not weak enough "
            f"for NO_EDGE_DETECTED. Run a small grid sweep to disambiguate."
        ),
    }


# ── Async orchestration ───────────────────────────────────────────────


async def _run_one_draw(
    *,
    db: Database,
    jvm: JvmClient,
    settings: Settings,
    account_strategy_id: str,
    strategy_code: str,
    instrument: str,
    interval_name: str,
    combo: dict[str, str],
    allow_long: bool,
    allow_short: bool,
) -> dict[str, Any]:
    """Submit one backtest, poll, return draw record. Failures are
    returned as records with ``pf=None`` and ``error`` set; we do NOT
    propagate exceptions because a single bad draw shouldn't abort the
    whole screen."""
    try:
        payload = _build_submit_payload(
            account_strategy_id=account_strategy_id,
            strategy_code=strategy_code,
            asset=instrument,
            interval_name=interval_name,
            overrides=combo,
            settings=settings,
            allow_long=allow_long,
            allow_short=allow_short,
        )
        backtest_run_id = await jvm.submit_backtest(payload)
        final_status = await _poll_backtest_status(db, backtest_run_id)
        if final_status != "COMPLETED":
            return {
                "combo": combo,
                "pf": None,
                "backtest_run_id": backtest_run_id,
                "error": f"backtest_status={final_status}",
            }
        async with db.acquire() as conn:
            metrics = await _fetch_run_metrics(conn, backtest_run_id)
        pf_raw = metrics.get("profit_factor")
        try:
            pf = float(pf_raw) if pf_raw is not None else None
        except (TypeError, ValueError):
            pf = None
        n_trades = int(metrics.get("total_trades") or 0)
        return {
            "combo": combo,
            "pf": pf,
            "n_trades": n_trades,
            "backtest_run_id": backtest_run_id,
            "error": None,
        }
    except OrchestratorError as e:
        return {"combo": combo, "pf": None, "error": e.envelope.error_code}
    except Exception as e:  # noqa: BLE001
        log.warn("null_screen.draw_crashed", combo=combo, error=str(e))
        return {"combo": combo, "pf": None, "error": type(e).__name__}


async def run_null_screen(
    *,
    db: Database,
    jvm: JvmClient,
    settings: Settings,
    agent_name: str,
    strategy_code: str,
    interval_name: str,
    instrument: str,
    param_ranges: list[dict[str, Any]],
    n_draws: int,
    seed: int,
) -> dict[str, Any]:
    """Execute the null screen end-to-end. Returns a structured payload
    with verdict, distribution, and per-draw records. Also writes a
    ``NULL_SCREEN_RESULT`` journal row so the agent can find it later."""
    if not param_ranges:
        raise OrchestratorError(
            status_code=400,
            error_code="empty_param_ranges",
            message="param_ranges must contain at least one entry.",
            retryable=False,
        )
    if n_draws < 4:
        raise OrchestratorError(
            status_code=400,
            error_code="n_draws_too_small",
            message="n_draws must be >= 4 for the distribution to be meaningful.",
            retryable=False,
        )

    # Resolve account_strategy once so we don't repeat the lookup per draw.
    async with db.acquire() as conn:
        as_row = await _resolve_account_strategy(
            conn, strategy_code, settings.research_account_id
        )
    if not as_row:
        raise OrchestratorError(
            status_code=412,
            error_code="account_strategy_missing",
            message=(
                f"No account_strategy row for strategy_code={strategy_code} "
                "(is_deleted=false). Seed one before running a null screen."
            ),
            retryable=False,
        )

    combos = generate_random_combos(param_ranges, n_draws, seed)
    log.info(
        "null_screen.start",
        strategy_code=strategy_code,
        instrument=instrument,
        interval=interval_name,
        n_draws=n_draws,
    )

    draws: list[dict[str, Any]] = []
    consecutive_failures = 0
    early_bail = False
    for combo in combos:
        rec = await _run_one_draw(
            db=db,
            jvm=jvm,
            settings=settings,
            account_strategy_id=as_row["account_strategy_id"],
            strategy_code=strategy_code,
            instrument=instrument,
            interval_name=interval_name,
            combo=combo,
            allow_long=as_row["allow_long"],
            allow_short=as_row["allow_short"],
        )
        draws.append(rec)
        log.info(
            "null_screen.draw_done",
            combo=combo,
            pf=rec.get("pf"),
            error=rec.get("error"),
        )
        if rec.get("pf") is None:
            consecutive_failures += 1
        else:
            consecutive_failures = 0
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            log.warn(
                "null_screen.early_bail",
                strategy_code=strategy_code,
                consecutive_failures=consecutive_failures,
                completed=len(draws),
                planned=n_draws,
            )
            early_bail = True
            break

    pfs = [d["pf"] for d in draws]
    distribution = summarize_pf_distribution(pfs)
    if early_bail:
        # Force the verdict even if the partial distribution looks edgy —
        # we don't have enough data to trust the right tail.
        verdict = {
            "verdict": "INSUFFICIENT_DATA",
            "reason": (
                f"Bailed after {MAX_CONSECUTIVE_FAILURES} consecutive failed "
                f"draws ({len(draws)}/{n_draws} completed). Likely JVM / "
                "data-plane issue; investigate before re-running."
            ),
        }
    else:
        verdict = null_screen_verdict(distribution, n_draws)

    structured = {
        "strategy_code": strategy_code,
        "interval_name": interval_name,
        "instrument": instrument,
        "n_draws": n_draws,
        "seed": seed,
        "early_bail": early_bail,
        "completed_draws": len(draws),
        "param_ranges": param_ranges,
        "distribution": distribution,
        "verdict": verdict,
        "draws": draws,
    }

    # research_journal.entry_type is CHECK-constrained (V1 baseline) to a
    # fixed enum. Map verdicts to the closest allowed type so the row
    # actually inserts AND the agent's cold-boot recipe surfaces it:
    #   * NO_EDGE_DETECTED  → ANTI_PATTERN  (cold-boot reads this filter)
    #   * EDGE_PRESENT      → STRATEGY_OUTCOME
    #   * INCONCLUSIVE      → STRATEGY_OUTCOME
    #   * INSUFFICIENT_DATA → STRATEGY_OUTCOME
    # The structured_data.kind discriminator lets future queries pull
    # only null-screen entries: WHERE structured_data->>'kind' = 'null_screen'.
    journal_entry_type = journal_entry_type_for(verdict["verdict"])
    structured["kind"] = "null_screen"
    title = (
        f"NULL_SCREEN {strategy_code} {instrument} {interval_name} → "
        f"{verdict['verdict']}"
    )
    content = (
        f"Null screen on {strategy_code} ({instrument} {interval_name}, "
        f"n_draws={n_draws}, seed={seed}). Verdict: {verdict['verdict']}. "
        f"{verdict['reason']}"
    )
    async with db.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                """
                INSERT INTO research_journal
                  (journal_id, entry_type, strategy_code, title, content,
                   structured_data, status, created_by, created_time)
                VALUES (gen_random_uuid(), $1, $2, $3, $4,
                        $5, 'ACTIVE', $6, NOW())
                """,
                journal_entry_type,
                strategy_code,
                title,
                content,
                structured,
                agent_name,
            )

    log.info(
        "null_screen.done",
        strategy_code=strategy_code,
        verdict=verdict["verdict"],
        n_finite=distribution.get("n_finite"),
    )

    return {
        "strategy_code": strategy_code,
        "interval_name": interval_name,
        "instrument": instrument,
        "n_draws": n_draws,
        "seed": seed,
        "distribution": distribution,
        "verdict": verdict,
        "draws": draws,
        "next_actions": _next_actions(verdict["verdict"], strategy_code),
    }


def journal_entry_type_for(verdict: str) -> str:
    """Map a null-screen verdict to its ``research_journal.entry_type``.

    Keep in sync with the CHECK constraint on research_journal (V1
    baseline): allowed values are STRATEGY_OUTCOME, CROSS_STRATEGY_FINDING,
    HYPOTHESIS, IDEA_BACKLOG, ANTI_PATTERN, RUN_SUMMARY. NO_EDGE_DETECTED
    deserves the ANTI_PATTERN slot so the agent's cold-boot recipe
    (``GET /journal?status=ACTIVE&entry_type=ANTI_PATTERN``) picks it up
    automatically.
    """
    if verdict == "NO_EDGE_DETECTED":
        return "ANTI_PATTERN"
    return "STRATEGY_OUTCOME"


def _next_actions(verdict: str, strategy_code: str) -> list[dict[str, Any]]:
    """Map verdict → recommended next call. Pure for testability."""
    if verdict == "EDGE_PRESENT":
        return [{"kind": "call", "method": "POST", "path": "/queue",
                 "hint": f"Edge present on {strategy_code}; enqueue a focused sweep."}]
    if verdict == "NO_EDGE_DETECTED":
        return [{"kind": "read_doc", "doc_anchor": "null_screen.no_edge",
                 "hint": "Move on to a different archetype or axis set; "
                         "do NOT spend ticks here."}]
    if verdict == "INCONCLUSIVE":
        return [{"kind": "call", "method": "POST", "path": "/queue",
                 "hint": "Run a small (8-cell) confirmatory grid to "
                         "disambiguate."}]
    return [{"kind": "contact_human", "hint": "Insufficient draws produced PFs."}]
