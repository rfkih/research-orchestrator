# Signal Pool / House Book — Phase 0 Design of Record

**Status:** in build (2026-06-03)
**Goal:** stop discarding weak-but-real signals one at a time. Admit any
statistically-valid signal that *adds uncorrelated return* into a managed pool
(the House Book), weighted by the existing HRP optimizer. Turns the research
agent's near-misses (e.g. XS_MOM +7.3%/yr, Sharpe 0.33) into a product.

## Core principle: additive, off the hot path, uniform

- **No change to `analyze.py` / `tick.py`.** The V11/V60 standalone gate is
  untouched — solo strategies still need ≥10%/yr. Pool admission is a SEPARATE
  evaluation invoked on an already-validated candidate.
- **Uniform treatment.** No protected/privileged strategies. LSR/VCB/VBO are
  ordinary candidates evaluated by the same `admit()` as XS_MOM. The pool's
  marginal-contribution test is computed against *whatever is currently in the
  pool*, never against a hardcoded baseline. (Do NOT use
  `portfolio.PROTECTED_STRATEGY_CODES` here.)

## What a candidate is

An iteration that is **statistically real but failed standalone-economic**:
- `statistical_verdict = SIGNIFICANT_EDGE` (real edge, PF 95%-CI lower > 1.0),
- `DSR ≥ 0.95` (not overfit, multiplicity-deflated),
- a `walk_forward_run` with `stability_verdict = ROBUST` (generalises OOS),
- but `decision_verdict = ITERATE` because annualised return < 10%/yr.

These are exactly the signals the standalone gate throws away. The validity bar
(the first three) is UNCHANGED — only the standalone-10% requirement is dropped
for pool members.

## The admission rule

```
admit(candidate) =
    valid(candidate)                              # SIGNIFICANT_EDGE + DSR≥0.95 + WF ROBUST  [unchanged rigor]
    AND marginal_sharpe(candidate | pool) > θ     # adds uncorrelated return to the current pool
```

`marginal_sharpe`:
1. Build each active pool member's daily-return series (`portfolio.daily_returns_from_trades`
   over its latest COMPLETED backtest_run).
2. `sharpe_before` = annualised Sharpe of the HRP-weighted combined pool series.
3. Add the candidate, re-run HRP, `sharpe_after` = Sharpe of the new combined series.
4. `marginal = sharpe_after − sharpe_before`.
5. Empty pool → first valid candidate admitted (`marginal = candidate standalone Sharpe`).

A redundant (highly-correlated) add barely moves combined Sharpe → fails the
test naturally; no separate correlation veto needed (we log `max_abs_corr` for
transparency). `θ` is a small positive margin (config, default 0.02) so noise
doesn't pad the book.

## Schema (Flyway V147, trading-engine repo)

`signal_pool` — one row per admitted signal:
`pool_id` PK, `iteration_id` (the validated candidate), `strategy_code`,
`symbol`, `interval_name`, `admitted_at`, `admission_metrics` jsonb
(standalone ann_return, sharpe, dsr, marginal_sharpe, max_abs_corr at
admission), `pool_weight` numeric (set by rebalance, default NULL until first
rebalance), `weight_source`, `weight_updated_at`, `status`
('active'|'evicted'), `evicted_at`, `evicted_reason`, BaseEntity audit cols.
Partial-unique on `(strategy_code, symbol, interval_name) WHERE status='active'`.

## Orchestrator components (all new, additive)

- `services/pool.py`:
  - `marginal_sharpe_contribution(candidate_returns, members_returns)` → dict
    {marginal, sharpe_before, sharpe_after, max_abs_corr, n_overlap}. Reuses
    `portfolio_math.spearman_corr_matrix` / `hrp_weights` / `realised_vol`.
  - `evaluate_admission(conn, iteration_id, theta)` → verdict
    {admit: bool, reason, metrics}. Verifies validity from stored
    iteration metrics + the iteration's walk_forward_run ROBUST row.
- `api/pool.py`:
  - `POST /pool/evaluate` {iteration_id} → ADMIT (insert row) | REJECT_INVALID
    | REJECT_REDUNDANT. Idempotent on iteration_id.
  - `GET /pool` → active members + weights + admission metrics.
  - `POST /pool/rebalance` → HRP-weight active members, write `pool_weight`
    (reuses the optimizer + guardrail clamp pattern from `rebalance.py`).

## Explicit non-goals (avoid slop)

- No change to the standalone V11/V60 gate, tick, or analyze.
- No JVM-side combined live execution yet (the House Book as one tradable
  `account_strategy` whose fills = weighted pool). The pool + admission +
  weighting is the core; live execution of the combined book is a later step
  once the pool has members and proven weights.
- No new specialist agent — admission is a deterministic rule, auditable.

## Build order (each step compiles/tests before next)

1. Design doc (this). 2. Flyway V147 `signal_pool`. 3. `services/pool.py` +
unit tests (marginal-Sharpe math: empty pool, diversifying add, redundant add).
4. `api/pool.py` 3 endpoints + register router. 5. Deploy; route the XS_MOM
sweep's ROBUST-but-sub-10% iterations into `POST /pool/evaluate` as first proof.

## Acceptance gate

XS_MOM (or another sub-10% ROBUST signal) is admitted to the pool, HRP-weighted,
and `GET /pool` shows a combined book whose Sharpe exceeds any single member's.
