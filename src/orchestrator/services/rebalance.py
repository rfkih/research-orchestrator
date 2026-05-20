"""Portfolio rebalance — write HRP/MV/equal-weight allocations to live
``account_strategy.portfolio_weight``.

Phase A of scorecard aspect #10 (portfolio construction, 3 → 5). The
math (HRP, mean-variance, equal-weight) already lives in
``services/specialists/portfolio_math.py`` and is exposed read-only via
``POST /portfolio/optimize`` for the quant-portfolio-manager agent. The
gap this module closes: nothing was writing the optimiser's output back
to the live trading state.

Resolver precedence on the JVM live path (see V109 migration):

    effective_capital = capital_allocation_pct
                      * kelly_multiplier
                      * portfolio_weight

Defaults preserve legacy LSR / VCB / VBO behaviour: pre-rebalance rows
carry ``portfolio_weight = 1.0`` and ``weight_source = 'EQUAL_WEIGHT'``,
so the multiplication is a no-op until this service has run.

Guardrails (Phase A):

* **MIN_WEIGHT = 0.05** — never zero out a live strategy via the batch.
  Operators kill a strategy by flipping ``enabled = false`` via the
  admin UI, NEVER by letting the rebalance batch drive its weight to
  zero. A 5% floor preserves a token allocation even when the optimiser
  thinks the strategy adds no marginal value.
* **MAX_WEIGHT = 0.50** — concentration cap. No single strategy can
  carry more than half the book. Forces the optimiser to spread risk
  even when one strategy dominates on sample stats.
* **MANUAL skip** — rows where ``weight_source = 'MANUAL'`` are left
  alone. The admin UI's "Force equal-weight" / "Override weight" flow
  stamps MANUAL so the operator hot-fix is not silently overwritten on
  the next nightly run.

The clamp can violate the simplex constraint (sum = 1) so we
renormalise once after clamping. For N=3 strategies (the protected
book today) one pass is sufficient; if the clamp pushes us outside
bounds again, the renormalise step would re-introduce a deviation
which is acceptable Phase A (the operator can re-run with different
bounds, and Phase B's vol-targeting layer is where this gets a proper
projection-onto-simplex).
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any, Literal
from uuid import UUID

import asyncpg
import numpy as np

from ..logging import get_logger
from .portfolio import (
    PROTECTED_STRATEGY_CODES,
    _fetch_one_baseline,
)
from .specialists.portfolio_math import (
    equal_weight,
    hrp_weights,
    mean_variance_weights,
    realised_vol,
    spearman_corr_matrix,
    summarise_weights,
)

log = get_logger(__name__)

# Operator-controlled guardrails. Changing these widens or narrows the
# rebalance batch's bounds. Pin tests in tests/test_rebalance.py guard
# against accidental drift; document any intentional change as an
# ORCHESTRATOR_CHANGE journal entry citing rationale.
MIN_WEIGHT: float = 0.05
MAX_WEIGHT: float = 0.50

# How many days of backtest-derived daily returns to require before
# attempting to compute a covariance / HRP allocation. Below this the
# job falls back to equal-weight rather than feeding a noisy 2-day
# correlation into HRP.
MIN_OBS_FOR_REBALANCE: int = 30

OptimizerName = Literal["HRP", "EQUAL_WEIGHT", "MEAN_VARIANCE"]


def _apply_guardrails(
    raw_weights: dict[str, float],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Clamp each weight to ``[MIN_WEIGHT, MAX_WEIGHT]`` then renormalise
    so they sum to 1.0. Returns ``(clamped_weights, diagnostics)`` --
    diagnostics includes per-code clamp events so the operator can see
    which weights the bounds bound.

    Empty input → empty output (caller decides what to do; an empty
    book is not a rebalance, it's a no-op).
    """
    if not raw_weights:
        return {}, {"n_clamped_low": 0, "n_clamped_high": 0, "renorm_factor": 1.0}
    n_low = 0
    n_high = 0
    clamped: dict[str, float] = {}
    for code, w in raw_weights.items():
        if w < MIN_WEIGHT:
            n_low += 1
            clamped[code] = MIN_WEIGHT
        elif w > MAX_WEIGHT:
            n_high += 1
            clamped[code] = MAX_WEIGHT
        else:
            clamped[code] = float(w)
    s = sum(clamped.values())
    factor = 1.0 / s if s > 0 else 1.0
    normalised = {c: float(w * factor) for c, w in clamped.items()}
    return normalised, {
        "n_clamped_low": n_low,
        "n_clamped_high": n_high,
        "renorm_factor": round(factor, 6),
    }


def _compute_raw_weights(
    optimizer: OptimizerName,
    codes: list[str],
    series_by_code: dict[str, dict[date, float]],
    min_overlap_days: int,
    mu_by_code: dict[str, float] | None,
    risk_aversion: float,
) -> dict[str, float]:
    """Dispatch to the requested optimiser. Single-strategy edge case
    short-circuits to {code: 1.0} so the guardrails downstream see a
    valid input -- HRP / MV both require N>=2."""
    if len(codes) <= 1:
        return {c: 1.0 for c in codes}
    if optimizer == "EQUAL_WEIGHT":
        return equal_weight(codes)
    _, corr = spearman_corr_matrix(series_by_code, min_overlap=min_overlap_days)
    vol_arr = np.asarray(
        [realised_vol(series_by_code[c]) or 1e-8 for c in codes],
        dtype=np.float64,
    )
    corr_filled = np.where(np.isnan(corr), 0.0, corr)
    np.fill_diagonal(corr_filled, 1.0)
    if optimizer == "HRP":
        return hrp_weights(codes, corr_filled, vol_arr)
    # MEAN_VARIANCE
    sigma = np.outer(vol_arr, vol_arr) * corr_filled
    mu_arr: np.ndarray | None = None
    if mu_by_code:
        mu_arr = np.asarray(
            [mu_by_code.get(c, 0.0) for c in codes], dtype=np.float64
        )
    return mean_variance_weights(
        codes, sigma, mu=mu_arr, risk_aversion=risk_aversion
    )


async def _fetch_target_account_strategies(
    conn: asyncpg.Connection,
    strategy_codes: tuple[str, ...],
    account_id: str,
) -> list[dict[str, Any]]:
    """All live (enabled, non-simulated, non-deleted, non-MANUAL) rows
    matching the candidate strategy_codes WITHIN THE GIVEN ACCOUNT.
    MANUAL rows are filtered out in SQL so the rebalance never
    overwrites an operator hot-fix.

    account_id scoping (added 2026-05-20) is the multi-tenancy
    safety boundary: each user owns their AccountStrategy rows, so
    a rebalance request from user A must never UPDATE user B's
    portfolio_weight. The caller (POST /portfolio/rebalance) requires
    account_id and the proxy / future JWT layer enforces that the
    caller may rebalance only their own account.
    """
    rows = await conn.fetch(
        """
        SELECT account_strategy_id,
               account_id,
               strategy_code,
               symbol,
               interval_name,
               portfolio_weight,
               weight_source
          FROM account_strategy
         WHERE strategy_code = ANY($1::text[])
           AND account_id = $2::uuid
           AND enabled = TRUE
           AND is_deleted = FALSE
           AND simulated = FALSE
           AND weight_source <> 'MANUAL'
        """,
        list(strategy_codes),
        account_id,
    )
    return [dict(r) for r in rows]


async def _write_weights(
    conn: asyncpg.Connection,
    rows: list[dict[str, Any]],
    new_weights_by_code: dict[str, float],
    source: OptimizerName,
) -> int:
    """UPDATE one row per AccountStrategy. Returns count updated.

    Caller wraps in a transaction together with the journal write so
    weights + audit row commit atomically. The batch hits every
    matching row even when multiple AS rows share a strategy_code (e.g.
    same code on different intervals / accounts) -- they all get the
    same weight. Phase D adds per-(symbol, code) weights once the
    universe expands; for the BTC-only protected book today, code-level
    is sufficient.
    """
    if not rows:
        return 0
    n = 0
    for r in rows:
        new_w = new_weights_by_code.get(r["strategy_code"])
        if new_w is None or not math.isfinite(new_w):
            continue
        await conn.execute(
            """
            UPDATE account_strategy
               SET portfolio_weight  = $1,
                   weight_source     = $2,
                   weight_updated_at = NOW()
             WHERE account_strategy_id = $3
            """,
            round(new_w, 5),
            source,
            r["account_strategy_id"],
        )
        n += 1
    return n


async def _journal_rebalance(
    conn: asyncpg.Connection,
    *,
    account_id: str,
    optimizer: OptimizerName,
    weights_before: dict[str, float],
    weights_after: dict[str, float],
    diagnostics: dict[str, Any],
    skipped: dict[str, str],
    n_updated: int,
    dry_run: bool,
    agent_name: str,
) -> UUID:
    """Append an audit row to research_journal.

    Uses entry_type CROSS_STRATEGY_FINDING because rebalancing is by
    definition cross-strategy. A future migration may introduce a
    dedicated PORTFOLIO_REBALANCE entry_type; until then this is the
    closest fit in the existing CHECK enum.
    """
    title = (
        f"Portfolio rebalance ({optimizer})"
        + (" [DRY-RUN]" if dry_run else "")
        + f" — {n_updated} rows updated"
    )
    content_lines = [
        f"Optimizer: {optimizer}",
        f"Dry-run: {dry_run}",
        f"Rows updated: {n_updated}",
        f"Clamp events: low={diagnostics.get('n_clamped_low')} "
        f"high={diagnostics.get('n_clamped_high')} "
        f"renorm={diagnostics.get('renorm_factor')}",
        "",
        "Before:",
        *[f"  {c}: {w:.5f}" for c, w in sorted(weights_before.items())],
        "",
        "After:",
        *[f"  {c}: {w:.5f}" for c, w in sorted(weights_after.items())],
    ]
    if skipped:
        content_lines.append("")
        content_lines.append("Skipped:")
        content_lines.extend(f"  {c}: {r}" for c, r in sorted(skipped.items()))
    content = "\n".join(content_lines)
    structured_data = {
        "account_id": account_id,
        "optimizer": optimizer,
        "dry_run": dry_run,
        "n_updated": n_updated,
        "weights_before": weights_before,
        "weights_after": weights_after,
        "diagnostics": diagnostics,
        "skipped": skipped,
        "guardrails": {"min_weight": MIN_WEIGHT, "max_weight": MAX_WEIGHT},
    }
    row = await conn.fetchrow(
        """
        INSERT INTO research_journal
          (journal_id, entry_type, title, content, structured_data,
           status, created_by, created_time)
        VALUES (gen_random_uuid(), 'CROSS_STRATEGY_FINDING', $1, $2, $3,
                'ACTIVE', $4, NOW())
        RETURNING journal_id
        """,
        title,
        content,
        structured_data,
        agent_name,
    )
    return row["journal_id"]


async def rebalance_book(
    conn: asyncpg.Connection,
    *,
    account_id: str,
    optimizer: OptimizerName = "HRP",
    strategy_codes: tuple[str, ...] = PROTECTED_STRATEGY_CODES,
    min_overlap_days: int = 5,
    mu_by_code: dict[str, float] | None = None,
    risk_aversion: float = 1.0,
    dry_run: bool = False,
    agent_name: str = "orchestrator",
) -> dict[str, Any]:
    """Compute new portfolio_weight per strategy_code from the most-
    recent COMPLETED backtest_run daily returns, apply guardrails, and
    (unless dry_run) UPDATE every live, non-MANUAL account_strategy row
    OWNED BY ``account_id``.

    account_id is required (multi-tenancy boundary, added 2026-05-20):
    each user owns their AccountStrategy rows, so a rebalance is always
    a per-account operation. The legacy "rebalance every matching row
    across all accounts" behaviour was a latent multi-tenant bug fixed
    by this scoping.

    Returns the full forensic payload — caller stashes it on the
    response envelope so the operator sees before/after even when
    dry_run = True.
    """
    target_rows = await _fetch_target_account_strategies(
        conn, strategy_codes, account_id
    )
    codes_in_book = sorted({r["strategy_code"] for r in target_rows})

    skipped: dict[str, str] = {}
    series_by_code: dict[str, dict[date, float]] = {}
    for code in codes_in_book:
        series = await _fetch_one_baseline(conn, code)
        if series is None:
            skipped[code] = "no_recent_backtest_run"
            continue
        if len(series) < MIN_OBS_FOR_REBALANCE:
            skipped[code] = (
                f"insufficient_observations ({len(series)} < {MIN_OBS_FOR_REBALANCE})"
            )
            continue
        series_by_code[code] = series

    eligible_codes = sorted(series_by_code.keys())
    weights_before = {
        r["strategy_code"]: float(r["portfolio_weight"])
        for r in target_rows
        if r["strategy_code"] in eligible_codes
    }

    if not eligible_codes:
        return {
            "applied": False,
            "account_id": account_id,
            "optimizer": optimizer,
            "dry_run": dry_run,
            "codes": [],
            "weights_before": weights_before,
            "weights_after": {},
            "diagnostics": {"reason": "no eligible strategies"},
            "skipped": skipped,
            "n_updated": 0,
            "journal_id": None,
            "computed_at": datetime.utcnow().isoformat() + "Z",
        }

    raw_weights = _compute_raw_weights(
        optimizer=optimizer,
        codes=eligible_codes,
        series_by_code=series_by_code,
        min_overlap_days=min_overlap_days,
        mu_by_code=mu_by_code,
        risk_aversion=risk_aversion,
    )
    new_weights, diagnostics = _apply_guardrails(raw_weights)

    # Weights + journal commit atomically. If either fails the other
    # rolls back -- prevents the "weights written without an audit row"
    # state and the "journal written claiming weights changed when they
    # didn't" state. Dry-run still wraps in a transaction for symmetry;
    # nothing is written under dry_run except the journal entry.
    eligible_target_rows = [
        r for r in target_rows if r["strategy_code"] in eligible_codes
    ]
    async with conn.transaction():
        if dry_run:
            n_updated = 0
        else:
            n_updated = await _write_weights(
                conn, eligible_target_rows, new_weights, optimizer
            )
        journal_id = await _journal_rebalance(
            conn,
            account_id=account_id,
            optimizer=optimizer,
            weights_before=weights_before,
            weights_after=new_weights,
            diagnostics=diagnostics,
            skipped=skipped,
            n_updated=n_updated,
            dry_run=dry_run,
            agent_name=agent_name,
        )

    log.info(
        "portfolio.rebalance",
        account_id=account_id,
        optimizer=optimizer,
        dry_run=dry_run,
        n_eligible=len(eligible_codes),
        n_updated=n_updated,
        skipped=list(skipped.keys()),
    )

    return {
        "applied": not dry_run,
        "account_id": account_id,
        "optimizer": optimizer,
        "dry_run": dry_run,
        "codes": eligible_codes,
        "weights_before": {c: round(w, 5) for c, w in weights_before.items()},
        "weights_after": {c: round(w, 5) for c, w in new_weights.items()},
        "summary": summarise_weights(new_weights),
        "diagnostics": diagnostics,
        "skipped": skipped,
        "n_updated": n_updated,
        "journal_id": str(journal_id) if journal_id else None,
        "guardrails": {"min_weight": MIN_WEIGHT, "max_weight": MAX_WEIGHT},
        "computed_at": datetime.utcnow().isoformat() + "Z",
    }
