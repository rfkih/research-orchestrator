# Research plan — 2026-05-29 (VBO × BTCUSDT × 1h — paper-mode characterization, extended window)

## Premise

Operator-authorized paper-mode session to empirically characterize VBO's edge on its primary surface (BTCUSDT 1h) over the longest viable backtest window now that feature_store has been backfilled to 2022-01-01 (38,583 rows confirmed in `trading_db.feature_store`).

Prior characterization sessions (queues `1b9124a6` + `678153f2`) ran 24 cells on the 17-month window 2024-01-01 → 2026-05-29. Every cell came back `INSUFFICIENT_EVIDENCE` with `n_trades` in {41..47} — the V11 floor is `n>=100`, so the 1h cadence frequency-starves the verdict. Extending the window from 878 → 1580 days (~1.8x calendar) should lift per-cell `n` to ~80–100 at constant trade-rate, finally enabling a real `SIGNIFICANT_EDGE` vs `NO_EDGE` call.

Earlier today the session attempted a 2022-01-01 start (queue `b8154976`) and hit JVM `BacktestDataValidatorService.MIN_COVERAGE_RATIO=0.50` (17,606 of 38,592 expected rows present, 45.6% coverage). That gate has now cleared — feature_store backfill complete to 2022-01-01.

Pending queue `ec69d922` (3x3 grid with `tp1R/adxEntryMax/rvolMin`, `start_time=2023-06-01`) is **CANCELLED** at session start — the 2023-06-01 start was a coverage workaround that is no longer needed, and `rvolMin` is a no-op per prior-session diagnosis (pre-filters already gate that data).

## Constraints reaffirmed

- BTC/ETH only — BTCUSDT in scope.
- 1h interval allowed (`BacktestRunRequest.@Pattern`).
- Production strategies untouchable. VBO is live and produces ~+20%/yr; do NOT modify `VBO.defaults()`, `applyOverrides()`, or `account_strategy` rows.
- V11 + V60 gates immutable. No loosening to surface a positive verdict.
- Research-mode `enabled=false simulated=true`. `do_not_promote=true` stamped on hypothesis structured_data.
- Faithful reporting: if VBO doesn't clear V11+V60 on the extended window, document that finding accurately in the paper.

## Hypothesis

`f73a4352-4204-48a9-a671-5110a1ebaf17` (ALGO kind, status ACTIVE)

Prediction: on the 2022-01-01 → 2026-05-01 extended window (1580 days), the default-anchored cell `(tp1R=1.50, adxEntryMax=22, atrExpansionMin=1.30, rvolMin=1.20)` reaches `n>=100`, allowing a definitive `SIGNIFICANT_EDGE` vs `NO_EDGE` call. The point-estimate PF should track the prior session's ~1.00–1.10 range, but the CI will tighten.

Adjacent-cell predictions:
- `tp1R` (only axis that moved PF in the 17mo window) should produce a smooth gradient — tight stops (1.30) over-tax expansion noise; long stops (1.70+) give back winnings to mean-reversion.
- `adxEntryMax` controls the compression-gate threshold — tighter (20) reduces n but may improve PF if compression really matters; looser (25) admits more entries but probably more noise.

Regime expectation (from prior session iter `e80f5be0` regime decomposition):
- BULL: n=16, pnl=-63.6 (mechanism systematically loses in trend regimes — compression breakout is a mean-revert play that gets steamrolled by trend continuation).
- NEUTRAL: n=13, pnl=+32.9 (the real edge sits here).
- BEAR: n=14, pnl=+2.0 (near-zero — VBO neither helps nor hurts in downtrends).

This will be documented in the paper regardless of final V11/V60 outcome.

Falsification:
- All cells `n>=100` AND all cells `PF CI lower < 1.0` AND all cells `geom@90% < 10%/yr` → VBO has no edge on the 4yr window at 1h cadence; the live edge is either regime-dependent (NEUTRAL-rich window prior) or has decayed. Either is a paper-worthy finding.
- Default cell PF CI lower > 1.0 AND geom >= 10% AND DSR >= 0.95 → SIGNIFICANT_EDGE on the default cell. Walk-forward triggered (note: VBO is already live, so walk-forward is for paper completeness, not promotion).

## Experiment

**Strategy:** VBO
**Symbol:** BTCUSDT
**Interval:** 1h
**Hypothesis ID:** f73a4352-4204-48a9-a671-5110a1ebaf17
**Backtest window:** `start_time=2022-01-01T00:00:00` (end_time defaults to today / hypothesis end 2026-05-01)

**Sweep grid (ALGO, concentrate budget on tp1R per prior-session findings):**

VBO production defaults (the anchor cell, mid of each axis):
- `tp1R=1.50` — take-profit at 1.5R
- `adxEntryMax=22` — compression-gate upper bound
- `atrExpansionMin=1.30` — ATR expansion threshold (pinned; was no-op in prior session)
- `rvolMin=1.20` — relative-volume floor (pinned; was no-op)

Grid (5 × 3 = 15 cells):

- `tp1R`: ["1.20", "1.40", "1.50", "1.60", "1.80"] — 5 values centered on default 1.50, widened to characterize the PF curvature
- `adxEntryMax`: ["20", "22", "25"] — 3 values around default 22 to probe compression-gate sensitivity

iter_budget=15 (one per cell). Drain auto-extends if needed.

**Anchor cell (production defaults):** `tp1R=1.50, adxEntryMax=22` — must be present in the grid (it is, the middle cell). Will be flagged prominently in the paper.

**Success criteria (V11 + V60, immutable):**

For ANY cell to clear `SIGNIFICANT_EDGE`:
- `n_trades >= 100`
- `PF 95% CI lower > 1.0`
- `DSR >= 0.95` (with cumulative trial deflation, Tier 1)
- `annualized_geometric_return_pct_at_alloc_90 >= 10`
- Then `statistical_verdict = SIGNIFICANT_EDGE`

Walk-forward only if at least one cell hits SIGNIFICANT_EDGE; then `stability_verdict = ROBUST` is needed for the paper's "robustness" section. This is a PAPER, not a graduation push — but if it clears we'd treat the result faithfully (GOAL_HIT branch unlikely on a protected live strategy anyway).

**Step 8.5 regime-analysis:** run `POST /regime-analysis/{queue_id}` post-completion for the regime-decomposition table in the paper.

**Branches:**

- COMPLETED (all 15 cells run): regime-analysis → POST /papers/{queue_id}/generate → STRATEGY_OUTCOME journal row. No GOAL_HIT (paper mode + do_not_promote).
- SIG_EDGE on any cell: same — run walk-forward for paper completeness, document the result. Do NOT push to graduation review (this is a protected strategy + paper-mode session).
- INSUFFICIENT_EVIDENCE on all 15 cells with n still under 100: document the frequency-starvation conclusion definitively — at 1h cadence VBO produces under ~25 trades/year and CANNOT clear V11 even over 4 years. That's a publishable structural finding for the research library.
- INFRA_FAIL: retry per drain semantics with 30s backoff (Kafka now healthy per docker ps; this should not recur).

## Execution order this session

1. **DONE** — Pre-register hypothesis `f73a4352` (prior session).
2. **DONE** — Cancel pending queue `ec69d922` (wrong start_time, no-op axis).
3. Write this plan (overwrites prior ATR_MOM HYBRID plan; this session is on a different strategy entirely).
4. POST /reviews/request (target_kind=plan, hypothesis_id=`f73a4352`).
5. POST /reviews/auto-run-checklist.
6. POST /queue with the 5x3 grid, start_time=2022-01-01, motivating_hypothesis_id=`f73a4352`.
7. POST /tick/drain (max_wall_clock_s=1500 per heartbeat ceiling). Re-call until terminal.
8. POST /regime-analysis/{queue_id} after queue terminal.
9. POST /papers/{queue_id}/generate.
10. Journal STRATEGY_OUTCOME RUN_SUMMARY row reflecting the actual outcome (faithful reporting).

## Decision criteria for next session

- If queue COMPLETED + paper generated: this hypothesis is closed; next session's resume protocol falls through to branch 5/6 (active-hypothesis or fresh).
- If SIG_EDGE on any cell + walk-forward ROBUST: this is a protected live strategy + paper-mode; document the result, do NOT push graduation/curator. Hypothesis closes faithfully.
- ATR_MOM iter `3a7479e9` Path C state is **NOT touched** this session (operator instruction). The prior Path C re-assertion row `a1988404` predates this session; next session that needs to resume that work will detect it via journal scan for that iteration_id.
