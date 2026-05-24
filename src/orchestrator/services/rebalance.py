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

from typing import TYPE_CHECKING

from ..errors import OrchestratorError
from ..logging import get_logger
from .portfolio import (
    PROTECTED_STRATEGY_CODES,
    _fetch_one_baseline,
)

if TYPE_CHECKING:
    from ..clients.trading_jvm import TradingJvmClient
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

# ── Min-notional guard (Phase A.5, scorecard #10) ─────────────────────
#
# The clamp+renorm math above can produce a mathematically valid
# weight vector (sum=1, each in [MIN_WEIGHT, MAX_WEIGHT]) that, on a
# small account, translates into per-strategy entry sizes the Binance
# executor will silently reject. ``LiveTradingDecisionExecutorService.
# applyPortfolioWeight`` multiplies the entry by ``portfolio_weight``;
# ``TradeOpenService`` then rejects any order whose USDT notional is
# below ``TradeConstant.MIN_USDT_NOTIONAL``. A strategy clamped to
# ``MIN_WEIGHT=0.05`` on a $500 account produces an entry far below
# that floor — the dashboard shows 5% allocation while the strategy
# never trades. The guard below catches that *before* the operator
# clicks Apply.

MIN_USDT_NOTIONAL_USD: float = 7.0
"""Mirror of ``TradeConstant.MIN_USDT_NOTIONAL`` in the trading JVM
(``blackheart-trading-engine/src/main/java/id/co/blackheart/util/
TradeConstant.java:11``). ``TradeOpenService.java:378-382`` silently
rejects any LONG order below this; SHORT orders bottom out at the
symbol's LOT_SIZE step. A pin test in ``tests/test_rebalance.py``
reads the Java literal and asserts equality so the two cannot drift
apart. Phase A.6 plumbs this from a JVM endpoint; until then update
both files together."""

MIN_NOTIONAL_SAFETY_FACTOR: float = 1.2
"""Buffer on top of ``MIN_USDT_NOTIONAL_USD`` for the rebalance guard.

Phase A.6 (2026-05-22) tightened this from 2.0 to 1.2 once the
orchestrator gained a live Kelly read via :class:`TradingJvmClient`.
The previous 2.0× was sized to absorb the ``kelly=1.0`` upper-bound
approximation (we couldn't see what Kelly the executor would actually
apply, so we left 50% of the floor as slack). With the live read, the
guard math is bit-identical to ``LiveTradingDecisionExecutorService``
(``available × cap_alloc × kelly × weight``), so the remaining buffer
only needs to cover:

* Equity / Kelly drift between Preview and signal fire (minutes-to-
  hours window).
* Binance price moves that change SHORT-side notional when priced into
  USDT (LONG-side is unaffected — see :func:`_check_min_notional`).
* Floating-point precision around the floor.

1.2× (effective floor $8.40) is the smallest buffer that still leaves
~20% headroom above the JVM's strict $7 reject point. Pin test guards
against drift; loosening past 1.0 makes the guard ineffective and
requires an ORCHESTRATOR_CHANGE journal entry citing rationale."""

EFFECTIVE_MIN_NOTIONAL_USD: float = MIN_USDT_NOTIONAL_USD * MIN_NOTIONAL_SAFETY_FACTOR
"""Per-strategy entry-notional floor enforced by the rebalance guard.
``$7 × 2.0 = $14`` at the pinned defaults. Computed at module load
so the test pinning the literal also pins the derived value."""

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
               capital_allocation_pct,
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


def _check_min_notional(
    eligible_target_rows: list[dict[str, Any]],
    new_weights: dict[str, float],
    available_usdt: float,
    kelly_by_account_strategy_id: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Detect strategies whose post-rebalance projected entry size
    would fall below ``EFFECTIVE_MIN_NOTIONAL_USD``.

    Math mirrors ``LiveTradingDecisionExecutorService.
    calculateLongTradeAmount`` (``blackheart-trading-engine/.../
    LiveTradingDecisionExecutorService.java:677``) AND the in-line
    Kelly + portfolio-weight multiplies that follow in the executor::

        projected_entry_usd = available_usdt
                            × (cap_alloc_pct / 100)
                            × kelly_multiplier
                            × portfolio_weight

    ``available_usdt`` is the *free* USDT balance — the executor sizes
    against ``Portfolio.balance`` (line 334-339 of the executor), which
    is the per-asset free balance, NOT the total marketable equity.

    ``kelly_by_account_strategy_id`` carries the live Kelly multiplier
    per row, fetched from the trading JVM's
    ``/api/v1/internal/orch/kelly/{accountStrategyId}`` endpoint
    (Phase A.6). When None or a code's id is absent from the map, we
    fall back to ``kelly = 1.0`` — the no-op assumption the executor
    itself uses when ``kellySizingEnabled = false``. The safety factor
    on ``MIN_NOTIONAL_SAFETY_FACTOR`` covers drift between the read at
    Preview/Apply time and the live re-computation at signal fire.

    **Known approximations (still present after Phase A.6):**

    1. *Fallback-path sizing only.* This formula models
       ``calculateLongTradeAmount``, which is the executor's FALLBACK
       — invoked only when ``decision.notionalSize`` is empty
       (``executeOpenLong`` line 337-339 prefers the strategy's own
       value when set). Self-sizing strategies (e.g. risk-budget ×
       atr-stop) can produce a different notional that the guard
       cannot predict without per-strategy logic. The guard is still
       useful because ``applyPortfolioWeight`` and ``applyKellySizing``
       multiply regardless of how the base size was computed, so a
       0.05 weight on a strategy whose base was already $30 still
       lands at $1.50 — well below the floor.

    2. *SHORT entries are not modelled.* ``calculateShortTradeAmount``
       multiplies ``baseBalance × alloc`` in the BASE asset (e.g. BTC),
       which then needs the current symbol price to compare against
       MIN_USDT_NOTIONAL. The orchestrator has no live price read
       today. For BTC-only protected-book strategies that are LONG-
       biased this is fine; multi-symbol or SHORT-heavy books will
       under-detect violations on the SHORT side.

    Returns one entry per violating strategy with the structured
    forensics the operator needs to act:

    * ``strategy_code``, ``symbol``, ``weight`` — *which* strategy
      and *what* weight produced the projected sub-floor entry.
    * ``capital_allocation_pct`` — the row's allocation as stored in
      ``account_strategy``, so the operator can see whether to widen
      that knob instead of changing the optimiser output.
    * ``kelly_multiplier`` — the live Kelly value used in the math
      (1.0 when none was provided), so the operator can correlate
      with the JVM's ``GET /kelly-status`` admin endpoint.
    * ``projected_entry_usd`` / ``floor_usd`` / ``shortfall_usd`` —
      magnitude of the violation so the operator can prioritise the
      worst row.

    Empty list = clean — no violations. Caller (``rebalance_book``)
    decides whether to refuse the write or just warn.
    """
    kelly_map = kelly_by_account_strategy_id or {}

    if available_usdt <= 0 or not math.isfinite(available_usdt):
        # Defensive — Pydantic validates ``> 0`` on the request body
        # but direct callers (cron, tests) may pass zero. We flag
        # every strategy with a structured ``non_positive_equity``
        # reason rather than dividing by zero or returning clean.
        return [
            {
                "strategy_code": r["strategy_code"],
                "symbol": r["symbol"],
                "weight": float(new_weights.get(r["strategy_code"], 0.0)),
                "capital_allocation_pct": float(r["capital_allocation_pct"]),
                "kelly_multiplier": float(
                    kelly_map.get(str(r["account_strategy_id"]), 1.0)
                ),
                "projected_entry_usd": 0.0,
                "floor_usd": EFFECTIVE_MIN_NOTIONAL_USD,
                "shortfall_usd": EFFECTIVE_MIN_NOTIONAL_USD,
                "reason": "non_positive_equity",
            }
            for r in eligible_target_rows
            if r["strategy_code"] in new_weights
        ]

    violations: list[dict[str, Any]] = []
    for r in eligible_target_rows:
        code = r["strategy_code"]
        weight = new_weights.get(code)
        if weight is None or not math.isfinite(weight) or weight <= 0:
            # Zero / missing weight means the strategy isn't trading
            # post-rebalance anyway — no violation to flag.
            continue
        cap_alloc = float(r["capital_allocation_pct"])
        if cap_alloc <= 0:
            # capital_allocation_pct=0 means the row is disabled at
            # the allocation layer; the executor returns ZERO from
            # allocFraction (LiveTradingDecisionExecutorService.java
            # :714-720), so no entry ever fires. Skip — the row
            # wasn't going to trade with or without the rebalance.
            continue
        cap_alloc_fraction = cap_alloc / 100.0
        # Kelly defaults to 1.0 when the trading-JVM client was
        # unavailable or returned no entry for this row. Same
        # semantics the executor's ``applyKellySizing`` uses when
        # ``kellySizingEnabled = false`` — a strict no-op.
        kelly = float(kelly_map.get(str(r["account_strategy_id"]), 1.0))
        if not math.isfinite(kelly) or kelly <= 0:
            kelly = 1.0
        projected_entry = (
            available_usdt * cap_alloc_fraction * kelly * float(weight)
        )
        if projected_entry < EFFECTIVE_MIN_NOTIONAL_USD:
            violations.append(
                {
                    "strategy_code": code,
                    "symbol": r["symbol"],
                    "weight": round(float(weight), 5),
                    "capital_allocation_pct": cap_alloc,
                    "kelly_multiplier": round(kelly, 5),
                    "projected_entry_usd": round(projected_entry, 2),
                    "floor_usd": EFFECTIVE_MIN_NOTIONAL_USD,
                    "shortfall_usd": round(
                        EFFECTIVE_MIN_NOTIONAL_USD - projected_entry, 2
                    ),
                }
            )
    return violations


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
    blocked: bool = False,
    violations: list[dict[str, Any]] | None = None,
) -> UUID:
    """Append an audit row to research_journal.

    Uses entry_type CROSS_STRATEGY_FINDING because rebalancing is by
    definition cross-strategy. A future migration may introduce a
    dedicated PORTFOLIO_REBALANCE entry_type; until then this is the
    closest fit in the existing CHECK enum.

    ``blocked=True`` records that the rebalance was refused by the
    min-notional guard. The journal entry IS still written (the
    operator's attempt + the violations need to be auditable) but
    ``n_updated=0`` reflects that no weights moved.
    """
    if blocked:
        title_suffix = " [BLOCKED]"
    elif dry_run:
        title_suffix = " [DRY-RUN]"
    else:
        title_suffix = ""
    title = (
        f"Portfolio rebalance ({optimizer}){title_suffix}"
        f" — {n_updated} rows updated"
    )
    content_lines = [
        f"Optimizer: {optimizer}",
        f"Dry-run: {dry_run}",
        f"Blocked: {blocked}",
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
    if violations:
        content_lines.append("")
        content_lines.append(
            f"Min-notional violations (floor=${EFFECTIVE_MIN_NOTIONAL_USD:.2f}):"
        )
        for v in violations:
            shortfall = v.get("shortfall_usd")
            content_lines.append(
                f"  {v['strategy_code']}/{v['symbol']}: "
                f"projected=${v['projected_entry_usd']:.2f} "
                f"weight={v['weight']:.5f} "
                f"shortfall=${shortfall:.2f}"
                if shortfall is not None
                else (
                    f"  {v['strategy_code']}/{v['symbol']}: "
                    f"projected=${v['projected_entry_usd']:.2f} "
                    f"weight={v['weight']:.5f}"
                )
            )
    if skipped:
        content_lines.append("")
        content_lines.append("Skipped:")
        content_lines.extend(f"  {c}: {r}" for c, r in sorted(skipped.items()))
    content = "\n".join(content_lines)
    structured_data = {
        "account_id": account_id,
        "optimizer": optimizer,
        "dry_run": dry_run,
        "blocked": blocked,
        "n_updated": n_updated,
        "weights_before": weights_before,
        "weights_after": weights_after,
        "diagnostics": diagnostics,
        "skipped": skipped,
        "violations": violations or [],
        "guardrails": {
            "min_weight": MIN_WEIGHT,
            "max_weight": MAX_WEIGHT,
            "min_notional_usd": MIN_USDT_NOTIONAL_USD,
            "min_notional_safety_factor": MIN_NOTIONAL_SAFETY_FACTOR,
            "effective_floor_usd": EFFECTIVE_MIN_NOTIONAL_USD,
        },
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
    available_usdt: float,
    optimizer: OptimizerName = "HRP",
    strategy_codes: tuple[str, ...] = PROTECTED_STRATEGY_CODES,
    min_overlap_days: int = 5,
    mu_by_code: dict[str, float] | None = None,
    risk_aversion: float = 1.0,
    dry_run: bool = False,
    agent_name: str = "orchestrator",
    trading_jvm: "TradingJvmClient | None" = None,
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

    ``available_usdt`` is the caller's view of the FREE USDT balance
    for ``account_id`` — NOT the total marketable equity. The JVM
    executor sizes LONG entries against the account's per-asset free
    balance (``Portfolio.balance``, ``LiveTradingDecisionExecutorService
    .java:334-339``), so the guard must compare against the same number.

    ``trading_jvm`` is the orchestrator's :class:`TradingJvmClient`.
    When present and enabled (Phase A.6, 2026-05-22), this function
    fetches:

    * Live ``available_usdt`` from the trading JVM. Validated against
      the caller-provided value within ±5%; drift is logged. The
      JVM-side value is used as truth in the guard math because that's
      what the executor will see at signal-fire time.
    * Per-strategy ``kelly_multiplier`` from
      ``KellySizingService.getStatus`` so the guard math is bit-
      identical to ``applyKellySizing`` — no upper-bound approximation.

    When ``trading_jvm`` is None or disabled, the function falls back
    to the Phase A.5 contract (caller-provided ``available_usdt``,
    ``kelly=1.0`` upper-bound assumption with the wider safety factor).

    The min-notional guard compares ``available_usdt × cap_alloc_pct/
    100 × kelly_multiplier × proposed_weight`` against
    ``EFFECTIVE_MIN_NOTIONAL_USD``; weights that would produce sub-
    floor entries cause Apply to refuse with 422
    ``below_min_notional_floor`` (dry-run still computes and surfaces
    the violations in ``diagnostics.below_min_notional``).

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
            "diagnostics": {
                "reason": "no eligible strategies",
                "below_min_notional": [],
                "available_usdt": round(float(available_usdt), 2),
                "available_usdt_source": "caller_provided",
                "effective_floor_usd": EFFECTIVE_MIN_NOTIONAL_USD,
                "kelly_source": "fallback_kelly=1.0",
            },
            "skipped": skipped,
            "n_updated": 0,
            "journal_id": None,
            "guardrails": {
                "min_weight": MIN_WEIGHT,
                "max_weight": MAX_WEIGHT,
                "min_notional_usd": MIN_USDT_NOTIONAL_USD,
                "min_notional_safety_factor": MIN_NOTIONAL_SAFETY_FACTOR,
                "effective_floor_usd": EFFECTIVE_MIN_NOTIONAL_USD,
            },
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

    # Min-notional guard (Phase A.5 + A.6). Run AFTER clamp+renorm so
    # the check sees the same numbers the executor would. Violations
    # are always attached to diagnostics; whether they BLOCK depends
    # on dry_run. See ``_check_min_notional`` for the math + assumptions.
    eligible_target_rows = [
        r for r in target_rows if r["strategy_code"] in eligible_codes
    ]

    # Phase A.6: live equity + Kelly read from the trading JVM. When
    # the client is unavailable, fall back to the caller-provided
    # available_usdt + kelly=1.0 (legacy Phase A.5 behaviour). The
    # client itself is responsible for short-circuiting cleanly on a
    # configuration miss (``trading_jvm.enabled`` short-circuit) so
    # cron / direct callers get a deterministic fallback path.
    effective_available_usdt = float(available_usdt)
    kelly_by_account_strategy_id: dict[str, float] = {}
    jvm_used = False
    jvm_available_usdt_raw: float | None = None
    jvm_drift_pct: float | None = None
    if trading_jvm is not None and trading_jvm.enabled:
        try:
            jvm_portfolio = await trading_jvm.get_account_portfolio(account_id)
            jvm_available_usdt_raw = float(jvm_portfolio.available_usdt)
            if not math.isfinite(jvm_available_usdt_raw):
                # NaN/inf is a serialization or arithmetic bug on the
                # JVM side, not a transient network issue. Distinct
                # log line so the operator can file the right ticket
                # (JVM bug vs. retry the request). Fall back to
                # caller-provided for graceful degradation; the
                # alternative is a hard refuse, which a one-off bad
                # row would turn into a hard outage.
                log.warning(
                    "rebalance.jvm_returned_non_finite",
                    account_id=account_id,
                    raw_value=str(jvm_portfolio.available_usdt),
                )
            elif jvm_available_usdt_raw < 0:
                # Negative free balance shouldn't exist. Same
                # treatment as NaN/inf — log loudly, fall back.
                log.warning(
                    "rebalance.jvm_returned_negative",
                    account_id=account_id,
                    raw_value=jvm_available_usdt_raw,
                )
            else:
                # JVM truth wins, INCLUDING when it's zero. Zero free
                # USDT is a legitimate state (all USDT locked in open
                # positions); falling back to a stale caller-provided
                # value here would re-introduce the silent-failure
                # mode the guard exists to prevent. The
                # ``non_positive_equity`` branch in
                # ``_check_min_notional`` will then correctly flag
                # every strategy and Apply gets a 422 the operator
                # can act on ("free up locked USDT").
                effective_available_usdt = jvm_available_usdt_raw
                jvm_used = True
                # Drift between client-provided and JVM-side. Logged
                # but not refused — operator clicked Preview a few
                # minutes before Apply, equity moves are expected.
                if float(available_usdt) > 0:
                    jvm_drift_pct = (
                        (jvm_available_usdt_raw - float(available_usdt))
                        / float(available_usdt)
                    )
                # Per-row Kelly. Skipped when effective equity is zero
                # — non_positive_equity short-circuits the guard math
                # regardless of Kelly, so the N HTTP roundtrips would
                # be wasted. Independent failures per row degrade to
                # kelly=1.0 for that row rather than nuking the whole
                # guard — the safety factor covers a single miss.
                if jvm_available_usdt_raw > 0:
                    for r in eligible_target_rows:
                        try:
                            kelly = await trading_jvm.get_kelly_status(
                                r["account_strategy_id"]
                            )
                            kelly_by_account_strategy_id[
                                str(r["account_strategy_id"])
                            ] = float(kelly.current_multiplier)
                        except OrchestratorError as exc:
                            log.warning(
                                "rebalance.kelly_read_failed",
                                account_strategy_id=str(r["account_strategy_id"]),
                                strategy_code=r["strategy_code"],
                                error_code=exc.envelope.error_code,
                            )
        except OrchestratorError as exc:
            # JVM portfolio read failed — log and fall back to caller-
            # provided number with kelly=1.0. The wider Phase A.5
            # safety factor would have been 2.0; at 1.2 there is less
            # cushion, so the guard is somewhat stricter when JVM is
            # unreachable. Acceptable: the alternative is refusing the
            # rebalance outright, which a transient network blip
            # shouldn't cause.
            log.warning(
                "rebalance.portfolio_read_failed",
                account_id=account_id,
                error_code=exc.envelope.error_code,
            )

    violations = _check_min_notional(
        eligible_target_rows,
        new_weights,
        effective_available_usdt,
        kelly_by_account_strategy_id=kelly_by_account_strategy_id or None,
    )
    diagnostics["below_min_notional"] = violations
    diagnostics["available_usdt"] = round(effective_available_usdt, 2)
    diagnostics["available_usdt_source"] = (
        "trading_jvm" if jvm_used else "caller_provided"
    )
    if jvm_used and jvm_drift_pct is not None:
        diagnostics["available_usdt_drift_pct"] = round(jvm_drift_pct * 100, 2)
        diagnostics["available_usdt_caller_provided"] = round(
            float(available_usdt), 2
        )
    diagnostics["effective_floor_usd"] = EFFECTIVE_MIN_NOTIONAL_USD
    diagnostics["kelly_source"] = (
        "trading_jvm" if kelly_by_account_strategy_id else "fallback_kelly=1.0"
    )
    if kelly_by_account_strategy_id:
        diagnostics["kelly_by_account_strategy_id"] = {
            k: round(v, 5) for k, v in kelly_by_account_strategy_id.items()
        }

    is_blocked = bool(violations) and not dry_run

    # Weights + journal commit atomically. If either fails the other
    # rolls back -- prevents the "weights written without an audit row"
    # state and the "journal written claiming weights changed when they
    # didn't" state. Dry-run still wraps in a transaction for symmetry;
    # nothing is written under dry_run except the journal entry. When
    # the guard blocks, we journal the rejection (operator's attempt +
    # violations are auditable) and then raise 422 AFTER the journal
    # commits so the audit row survives.
    async with conn.transaction():
        if dry_run or is_blocked:
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
            blocked=is_blocked,
            violations=violations,
        )

    log.info(
        "portfolio.rebalance",
        account_id=account_id,
        optimizer=optimizer,
        dry_run=dry_run,
        blocked=is_blocked,
        n_eligible=len(eligible_codes),
        n_updated=n_updated,
        n_violations=len(violations),
        available_usdt=round(effective_available_usdt, 2),
        available_usdt_source=diagnostics["available_usdt_source"],
        available_usdt_drift_pct=diagnostics.get("available_usdt_drift_pct"),
        kelly_source=diagnostics["kelly_source"],
        skipped=list(skipped.keys()),
    )

    if is_blocked:
        raise OrchestratorError(
            status_code=422,
            error_code="below_min_notional_floor",
            message=(
                f"Rebalance would produce {len(violations)} strategy "
                f"row(s) with projected entry size below "
                f"${EFFECTIVE_MIN_NOTIONAL_USD:.2f} "
                f"(MIN_USDT_NOTIONAL ${MIN_USDT_NOTIONAL_USD:.2f} × "
                f"safety {MIN_NOTIONAL_SAFETY_FACTOR:g}). Apply refused; "
                f"weights NOT written. Audit row "
                f"{str(journal_id)[:8]}."
            ),
            retryable=False,
            hint=(
                "Either (a) free up locked USDT by closing open "
                "positions, (b) lower the number of live strategies, "
                "(c) raise the violating row's capital_allocation_pct, "
                "or (d) set weight_source='MANUAL' on the row with an "
                "explicit override. Re-run preview after any of these "
                "to see the updated diff."
            ),
            details={
                "violations": violations,
                "available_usdt": round(effective_available_usdt, 2),
                "available_usdt_source": diagnostics["available_usdt_source"],
                "effective_floor_usd": EFFECTIVE_MIN_NOTIONAL_USD,
                "min_notional_usd": MIN_USDT_NOTIONAL_USD,
                "min_notional_safety_factor": MIN_NOTIONAL_SAFETY_FACTOR,
                "kelly_source": diagnostics["kelly_source"],
                "journal_id": str(journal_id) if journal_id else None,
            },
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
        "guardrails": {
            "min_weight": MIN_WEIGHT,
            "max_weight": MAX_WEIGHT,
            "min_notional_usd": MIN_USDT_NOTIONAL_USD,
            "min_notional_safety_factor": MIN_NOTIONAL_SAFETY_FACTOR,
            "effective_floor_usd": EFFECTIVE_MIN_NOTIONAL_USD,
        },
        "computed_at": datetime.utcnow().isoformat() + "Z",
    }
