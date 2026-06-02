# Research Plan — 2026-06-02 (BNB/XRP new-symbol surface)

## Premise
ARCHETYPE_EXHAUSTION fired 2026-06-02 04:30 UTC (row f0fafc54): BTC/ETH x all ALGO
archetypes (DCB/MMR/MRO/TPB/VBO_RESEARCH/ATR_MOM/FCARRY) x {5m,15m,1h,4h} exhausted
with 0 SIG_EDGE; ML stationary-label direction (funding/OFI) falsified on transferability
(adversarial_auc=1.0). BNB + XRP were fully plumbed 2026-06-02 (market_data 4 intervals
2024->now + feature_values 1h/4h, 16 feats; V124 allowlist). This is a genuinely NEW
symbol surface — operator-authorized lockout bypass via fresh HYPOTHESIS 5b27b720
(created 04:51 UTC, after the terminal-fire row). Standing GOAL_HIT candidate 6025e072
(DCB ETHUSDT 1h x regime_eth_v2) remains in the curator admin inbox — unaffected.

## Constraints reaffirmed
- Research universe now includes BNBUSDT, XRPUSDT (plumbed). Intervals 5m/15m/1h/4h only.
- Production strategies LSR/VCB/VBO untouchable. New work enabled=false, simulated=true.
- Profitability bar: annualized_geometric_return_pct_at_alloc_90 >= 10 AND walk-forward ROBUST.
- Stat gates: n>=100, PF 95% CI lower>1.0, DSR>=0.95 (cumulative-trial scaled).
- Iterations traverse >=3 dimensions. Max 2 review rounds per hypothesis.

## Experiments

### E1 — DCB ALGO BNBUSDT 1h
- Hypothesis: 5b27b720. Donchian breakout captures BNB directional moves; BNB
  exchange-token beta differs from BTC/ETH so a re-tuned grid may clear V11+V60.
- Grid (4 axes): donchianPeriod {20,30,40} x adxEntryMin {20,25,30} x tpR {2.0,2.5,3.0}
  x stopAtrMult {1.5,2.0}. (orchestrator down-samples to iter_budget cells)
- iter_budget: 18. Success: >=1 cell SIGNIFICANT_EDGE with geom>=10%.
- Branch: SIG_EDGE -> graduation review -> walk-forward. PIVOT -> regime analysis -> E2.

### E2 — DCB ALGO XRPUSDT 1h
- Same grid + hypothesis. XRP episodic event-driven vol.
- iter_budget: 18. Same success/branch criteria.

### E3 (conditional) — DCB ALGO BNB/XRP 4h
- Fires if E1/E2 both PIVOT. Lower-frequency captures larger trend swings; 4h has less
  data density but cleaner trends. Same axes.

### E4 (conditional) — VBO_RESEARCH or VCB_RESEARCH on BNB/XRP 1h
- Fires if E1-E3 exhaust DCB. Different archetype (breakout-volatility / channel) on the
  new surface.

## Execution order
E1 (BNB 1h) -> E2 (XRP 1h) -> [E3 4h if both pivot] -> [E4 alt-archetype if still no edge].

## Decision criteria for next session
- Any SIG_EDGE -> graduation review -> walk-forward; GOAL_HIT if ROBUST + geom>=10%.
- All BNB/XRP DCB+VBO+VCB surfaces exhausted with 0 SIG_EDGE -> STRATEGY_OUTCOME;
  this becomes the 2nd no-credible-archetype diagnosis -> ARCHETYPE_EXHAUSTION terminal.
- Wall-clock: ~3h30m cumulative budget remaining; WIND_DOWN at 8h, RUN_COMPLETE at 8.5h.
