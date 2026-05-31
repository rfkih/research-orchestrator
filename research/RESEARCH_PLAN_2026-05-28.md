# Research Plan — 2026-05-28 (operator-directed extended-window re-test)

## Premise

Prior session reached ARCHETYPE_EXHAUSTION on ATR_MOM-ETHUSDT-1h at iter e86d9d31 (PF=1.73, n=106, DSR=0.97, ag90=25.22%/yr — SIGNIFICANT_EDGE) followed by walk-forward INSUFFICIENT_EVIDENCE. Post-mortem (FINDING b566f956) identified the in-sample window 2024-01-01 → 2026-05-26 was too narrow: walk-forward fold 1 tested 2023-07-01 → 2024-01-01, a period 12+ months before any training data existed. That is an extrapolation, not an OOS test.

Operator directive (2026-05-28) bypasses the 24h ARCHETYPE_EXHAUSTION lockout and authorises re-running the same sweep with `backtest_window.start_time=2022-01-01T00:00:00`. ETHUSDT 1h feature_store and market_data backfilled to 2022-01-01 (RECOMPUTE_RANGE job 25628314, 14472 rows).

Standing hypothesis: `60fc9473-168b-479d-8f6d-02bb28cf4a4f` (ATR_MOM-ETHUSDT-1h fresh in-sample 2022-01-01 to 2026-05-26).

## Constraints reaffirmed

- BTC + ETH only, intervals 5m/15m/1h/4h only
- Production strategies LSR/VCB/VBO untouchable
- Research-mode strategies only
- V11 + V60 gates: n>=100, PF lower CI > 1.0, DSR>=0.95, ag90>=10%, statistical_verdict=SIGNIFICANT_EDGE
- No promotion without operator say-so

## Experiment 1 — ATR_MOM ETHUSDT 1h extended window

- **Strategy**: ATR_MOM
- **Surface**: ETHUSDT 1h
- **In-sample window**: 2022-01-01T00:00:00 to backtest engine's "yesterday UTC midnight" (~2026-05-27)
- **Sweep grid** (Hamming-1 around the proven e86d9d31 SIG_EDGE point):
  - `atrExpansionMult`: ["1.40", "1.50", "1.60"]
  - `erTrendMin`: ["0.35", "0.40", "0.45"]
  - `tpR`: ["1.75", "2.0", "2.25"]
  - Total = 27 cells; iter_budget = 27
- **Hypothesis**: SIG_EDGE retained on multi-regime window (2022 bear + 2023 sideways + 2024-2026 bull); n>=180; if cleared, walk-forward folds will all be in-distribution and ROBUST verdict is feasible.
- **Success criteria**: ANY cell yields verdict=PASS (SIGNIFICANT_EDGE + ag90>=10 + DSR>=0.95). Best cell graduates.
- **Branches**:
  - SIG_EDGE found → graduation review → specialist reviews (Path C) → exit SPECIALIST_REVIEW_PENDING; next session drives walk-forward.
  - All cells INSUFFICIENT_EVIDENCE or NO_EDGE → FALSIFIED hypothesis: 2022-start window does not retain the 2024-onset edge. Mark hypothesis FALSIFIED. Strong evidence the original SIG_EDGE was bull-regime overfitting; journal as findings paper.
  - INFRA_FAIL or partial drain → re-call /tick/drain in next session via resume protocol.

## Execution order

1. Plan review (auto-checklist).
2. POST /queue with `sweep_config.backtest_window.start_time = "2022-01-01T00:00:00"`, `override_discard_gate=true` (justification: prior 2024-start sweep was bull-only, fresh data window not previously tested).
3. POST /tick/drain (max_iters=27, max_wall_clock_s=1500).
4. If GRADUATE → graduation review → Path C specialist requests → exit SPECIALIST_REVIEW_PENDING.
5. Else → STRATEGY_OUTCOME; if MAX_WALL_CLOCK_REACHED next session resumes.

## Decision criteria for next session

- If this session exits SPECIALIST_REVIEW_PENDING: next session drains specialist verdicts and proceeds to walk-forward with `full_start=2022-01-01`.
- If this session exits with NO_EDGE across all cells: mark hypothesis FALSIFIED; write research paper documenting that ATR_MOM-ETHUSDT-1h was bull-regime overfitting; back to ARCHETYPE_EXHAUSTION review.
- If wall-clock cap mid-drain: resume protocol picks up via queue-pending branch.
