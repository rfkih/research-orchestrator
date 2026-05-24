# Research Plan — Path C ML Gate on DCB-ETH-1h via regime_eth_v2

**Date:** 2026-05-21
**Hypothesis:** `fad76022-38c7-495c-924f-d09a26c6d197` (HYBRID DCB ETH 1h with directional_eth_1h_v1 — Path C)
**Strategy code:** DCB (research-mode account_strategy row)
**Symbol / Interval:** ETHUSDT / 1h
**Operator override:** Yes — resolves HARD_RULE_BLOCK_INFERENCE_STAMPING_2026-05-21

## Context

Prior session journalled `HARD_RULE_BLOCK_INFERENCE_STAMPING_2026-05-21` (RUN_SUMMARY
21f4e296) after diagnosing that the inference sidecar's
`fetch_per_bar_values_at_ts` exact-matches `(symbol, interval, ts)` on
`feature_values`, but macro features (`fear_greed_value`,
`stablecoin_supply_change_*`, `eth_btc_ratio_momentum_20d`) live at
`symbol='', interval=''`. Every prior model's `feature_set` includes those
macro rows → sidecar returns 409 `feature_value_missing` at inference time.

Operator override of that exit instructs **Path C**: author a fresh ML
spec consuming ONLY per-symbol-per-interval-stamped bar-level features,
register, backfill via the sidecar, and paired-backtest the four
V60-passing DCB-ETH-1h iterations.

## What was already done THIS session

1. **`directional_eth_1h_v1` (purpose=directional, 13-gate gauntlet)** —
   FAIL across 3 hyperparam variants (default, macro-excluded,
   bayesian-tuned). Walk-forward AUC mean clustered 0.534-0.538, below
   the 0.55 directional bar. Model registered status=`rejected_by_operator`.
   Falsifies the "directional ML on ETH-1h bar-only features" sub-claim.

2. **`regime_eth_v2` (purpose=regime, 5-gate modulator gauntlet)** —
   **PASS**, all 5 gates. Walk-forward AUC mean 0.5338 (>0.52),
   primary_std 0.1142 (<0.15), saved-booster AUC 0.6094. Status=`trained`.
   model_id `6b3b12e4-34e3-46ba-aa32-37e812bdf56f`. Artifact sha256
   `9eceee2cf2e7cb4cce37c78e52db61b18e30e700b02e8eeba166c7156a829e97`.
   First-ever ETH-1h ML model whose feature_set is SIDECAR-SERVABLE
   (no macro/cross-asset features).

3. **signal_definition `regime_eth_v2`** — created with
   signal_id=`53141c26-76c6-46e8-b503-cc7612eb23fa`, status=shadow.
   12,653 signal_history rows backfilled via the sidecar across
   2024-12-02 → 2026-05-13 (≈17 months of ETH-1h coverage). Value
   range [0.0016, 0.997], mean 0.4957. Distribution looks marginal-edge,
   not collapsed.

## Plan for THIS sweep

Paired-backtest the four V60-passing DCB-ETH-1h iterations with
`_ml_gate_enabled` toggled on/off, using regime_eth_v2 as the
`_ml_signal_name`.

### Axis design (3 axes — meets the "≥3 dimensions" rule)

The four host iterations cover these param values:
- `tpR` ∈ {2.0, 3.5, 5.0}
- `adxEntryMin` ∈ {21, 22}
- `_ml_gate_enabled` ∈ {false, true}   ← treatment axis
- `_ml_signal_name` = "regime_eth_v2"   ← fixed (single value)
- `_ml_shadow_mode` = false              ← fixed (single value)

Three axes vary: `tpR × adxEntryMin × _ml_gate_enabled`. Effective grid
3 × 2 × 2 = 12 cells. Estimated wall-clock ~12 × ~30s/cell ≈ 6-8 min.

### Acceptance criteria

- **PRIMARY:** paired-delta on `annualized_geometric_return_pct_at_alloc_90`
  is POSITIVE (or at least non-negative) across the 6 paired (gate-on vs
  gate-off) cells.
- **GRADUATION GATE:** at least one gate-on cell shows V11+V60 PASS —
  n_trades≥100 AND PF lower 95% CI > 1.0 AND DSR≥0.95 AND ag90≥10.

### Known structural risk

The 4 prior V60-passing cells with v6 directional gate ON had n=47-49
(post-gate). The host strategy's gate-OFF n was higher but typically
still <100 on the 142-day window. Adding regime_eth_v2 (which is also
selective, mean=0.5 means ~50% of bars score below midline) might
shrink n further. If all paired cells have n<100, journal STRATEGY_OUTCOME
and conclude that **DCB-ETH-1h cannot graduate within the available
historical depth even with the new sidecar-servable gate**.

### Falsification conditions

- **NEGATIVE_DELTA verdict** from `/paired-delta` → hard reject per
  Phase 4 Stage H precedent (regime_btc_v3 destroyed value −1.23→−3.13
  Sharpe). Journal STRATEGY_OUTCOME on the ML hypothesis.
- **All gate-on cells n<100** → falsifies the V11-unblock premise of
  the operator brief; pivot to step 5 (alternative archetypes).
- **All cells gate-off n<100** → confirms host-archetype structural
  cap independent of ML; pivot likewise.

### Non-falsifying-but-informative

- POSITIVE_DELTA on ag90 + at least one cell crossing V11 n>=100 →
  graduation candidate → submit graduation review.
- POSITIVE_DELTA + no V11 clear → "the gate adds value but the
  host can't graduate"; journal informative STRATEGY_OUTCOME and
  note the model as a candidate for a different host.

## Resource accounting

- Live trading: untouched. Sweep targets DCB-ETHUSDT-1h research-mode
  account_strategy row (research-agent owned).
- No spec deploys. No live promotion attempts. V11+V60 thresholds
  unchanged.
- Estimated wall-clock 6-8 min for sweep + 1-2 min for /paired-delta.
- Idempotency-Key on all POSTs.
