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

---

# Phase 1-A — House Book live execution (capital-allocation overlay)

**Status:** shipped 2026-06-03. **Goal:** make the weighted pool *tradable* by
propagating `signal_pool.pool_weight` onto a dedicated **House Book account**'s
`account_strategy.portfolio_weight` (the V109 column the live sizer already
multiplies into every entry via `applyPortfolioWeight`). No new schema, no new
JVM engine.

## Why overlay (Option A), not a synthetic netted engine (Option B)

The members are **stateful, event-driven** TA4j strategies that each manage
their own entry/exit/stop lifecycle — that internal management *is* their edge.
Netting them into one position per symbol (B) has no coherent owner for the
netted position's exit logic, and would require rewriting members as stateless
continuous-weight emitters (throwing away the edge). So each member stays its
own `account_strategy` row; the book is the *weighted envelope* over them.
Reserve B for a future generation of stateless continuous-signal models.

## Boundary (deliberate — mirrors `rebalance.py`)

The orchestrator only **UPDATEs weights on rows that already exist** on the book
account. It never creates live-trading rows and never touches admin's account:

- **`to_sync`** — member has a live row + non-null pool_weight → set
  `portfolio_weight = pool_weight`, `weight_source='HOUSE_BOOK'`.
- **`unmaterialized`** — member has no live row → *reported*, not created. The
  operator provisions it via the trading JVM's normal strategy-creation flow,
  then a re-sync picks it up. (Real-capital enablement stays operator-gated.)
- **`to_zero`** — a book-account row that is no longer an active member and
  still carries weight → soft-disabled (`portfolio_weight→0`, reversible;
  `enabled` left untouched). This is how eviction propagates to live.
- **`needs_rebalance`** — member with NULL pool_weight → run `/pool/rebalance`.

## Components (all new/additive)

- `config.house_book_account_id` (env `ORCH_HOUSE_BOOK_ACCOUNT_ID`). Unset →
  `/pool/sync` returns `not_configured` (no-op).
- `services/house_book.py`: `classify_book(members, live_rows)` (pure
  reconciliation, unit-tested) + `reconcile_plan` + `apply_sync(dry_run=True)`
  (writes weights via the `rebalance._write_weights` pattern + journals to
  `research_journal`).
- `api/pool.py`: `POST /pool/sync {dry_run=true}` (preview-only unless
  `dry_run=false`) and `GET /pool/book` (live composition).

## Operating cadence

1. Loop admits members → `POST /pool/rebalance` (HRP weights).
2. Operator materialises any `unmaterialized` members as `account_strategy`
   rows on the book account (start `simulated=true`/paper).
3. Cron `POST /pool/sync {dry_run:false}` → live weights track the pool;
   evicted members zero out automatically.

## Non-goals (Phase 1-A)

- The orchestrator does NOT create or enable `account_strategy` rows — that's
  the operator via the trading JVM (keeps real-capital provisioning gated).
- No book-level aggregate risk guard yet (per-member guards still apply); a
  book net-exposure cap is Phase 1-B.
- Option B (synthetic netted engine) is explicitly deferred.

---

# Hardening — audit fixes (2026-06-03)

A self-audit of Phase 0 + 1-A found and fixed:

- **#1 (safety)** `house_book` eviction (`to_zero`) now only soft-disables rows
  the book itself wrote (`weight_source ∈ {HOUSE_BOOK, HOUSE_BOOK_EVICTED}`). A
  misconfigured `ORCH_HOUSE_BOOK_ACCOUNT_ID` can no longer wipe another
  account's weights.
- **#2 (correctness)** `annualised_sharpe` now computes on a **calendar-daily**
  grid (zero-filled), so √252 is valid and members of different trade
  frequencies are comparable. (`daily_returns_from_trades` is sparse exit-day.)
- **#3 (correctness)** member return series are pinned to the **admitted**
  backtest (`signal_pool.iteration_id → backtest_run_id`), not "latest for the
  surface" — a later unrelated sweep can't silently re-weight a member.
- **#4 (correctness)** `marginal_sharpe_contribution` measures before/after on
  the **overlapping** date window; non-overlapping series return marginal 0
  (`reason=insufficient_overlap`) instead of inventing diversification.
- **#6** `/pool/sync` reports `deployed_fraction` / `undeployed_fraction` so a
  partially-materialised (under-invested) book isn't read as fully deployed.
- **#7** `/pool/evaluate|rebalance|sync` honour `Idempotency-Key`; admission
  INSERT is atomic (`ON CONFLICT … DO NOTHING`) — no check-then-insert race.
- **#8** `/pool/evaluate` returns `200` (was `201`) — it's an evaluation, not
  always a creation.
- **#11** FK `signal_pool.iteration_id → research_iteration_log` (Flyway V148).

## Remaining known limitations (documented, not yet fixed)

- **#5** Exit-day PnL bucketing (`daily_returns_from_trades`) still approximates
  a true mark-to-market equity curve. The calendar-daily fix (#2) corrects the
  *annualisation*, but precise daily correlation/Sharpe needs a JVM-side
  `daily_equity` series — a cross-repo change, deferred.
- **#9** The tick near-miss route walk-forwards the *last* iteration of the
  exhausted sweep, not the best-DSR cell — `research_iteration_log` has no
  queue linkage to cleanly pick "best of sweep". Walk-forward re-validates
  regardless; low impact.
- **#10** `signal_pool ↔ account_strategy` match is by exact
  `symbol`/`interval_name` strings; relies on consistent conventions (they are
  today). A canonicalisation layer would remove the latent coupling.

---

# Phase 1-B — book risk guard + eviction lifecycle (shipped 2026-06-03)

- **Per-symbol concentration cap** (`config.house_book_max_per_symbol`, default
  0.50). `/pool/rebalance` water-fills weight off any symbol over the cap onto
  under-cap members (so the book diversifies across *symbols*, not just
  members). Infeasible compositions (n_symbols × cap < 1) skip the cap and flag
  rather than under-deploy. Pure `pool.apply_per_symbol_cap`, unit-tested.
- **Eviction by marginal** — `POST /pool/reconcile {dry_run=true, evict_theta=0}`
  re-scores each active member's marginal Sharpe vs the REST of the pool
  (member-as-candidate, reusing the admission math) and evicts members that
  have gone redundant (marginal ≤ evict_theta). Sets `status='evicted'` +
  `evicted_reason`. The sole remaining member is never evicted. Preview-only
  unless `dry_run=false`; idempotent.

# Phase 1-C — book performance + attribution (shipped 2026-06-03)

- **`GET /pool/performance`** — combined HRP-weighted book vs each member
  standalone, per-member attribution (`weight × mean_daily_return`), the
  combined equity curve, and the headline `combined_beats_best_member` — the
  diversification payoff the pool exists to capture.

## Remaining for 1-B / 1-C (deferred — need members and/or other repos)

- **Cron** for periodic `rebalance → sync → reconcile`. Held until the pool has
  members (it would no-op now). One `CronCreate` when ready; cadence documented.
- **Live decay eviction** — the current eviction criterion is *redundancy*
  (marginal vs pool). True performance *decay* needs live `trade_history`,
  available only once the book trades. Wire in then.
- **Frontend product surface** — "House Book" as a subscribable picker entry +
  attribution view + leaderboard row. Lives in the frontend/trading-JVM repos;
  consumes `GET /pool/performance` + `GET /pool/book`. Premature to build before
  member #1.
