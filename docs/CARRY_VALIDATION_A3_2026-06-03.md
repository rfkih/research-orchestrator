# Funding-Carry — v2 Validation on Real BTC Data (2022–2026)

> **Scope.** These are real-data validation findings for the **JVM funding-carry engine**
> (`FundingCarryStrategyEngine`, `docs/FUNDING_CARRY_DESIGN.md`). They are the "confirm with v2"
> that the v1 design called for: v1 zeros the price P&L ("no spot needed") and over-states risk;
> this measures the basis with a real spot feed. Produced by a throwaway orchestrator-side Python
> harness (since removed — the JVM engine is the production path; the per-period basis-MtM formula
> `−q·Δ(perp−spot)` is the reference for enhancing it to v2).

**Date:** 2026-06-03/04. **Data:** prod `trading_db` export — `binance_funding_rate_btcusdt`
(4,791 settlements, 8h) × BTC 1h price (38,751 bars), window **2022-01-01 → 2026-06-03** (funding
starts 2021 but price only 2022, so the overlap is 4.4 years). Avg funding = **0.0611 bp / 8h** →
naive always-on gross ≈ 6.69%/yr.

## Results

| Scenario | ann % | Sharpe | vol % | maxDD % | held | round-trips |
|---|---|---|---|---|---|---|
| always-on, 0 bps (gross ceiling) | 6.84 | 12.1 | 0.38 | 0.38 | 4790 | 1 |
| always-on, 2 bps (maker) | 6.82 | 12.1 | 0.38 | 0.38 | 4790 | 1 |
| **always-on, 4 bps (taker)** | **6.80** | 12.0 | 0.38 | 0.38 | 4790 | 1 |
| conditional f≥0, 2 bps | −0.26 | −0.2 | 0.75 | 7.1 | 4023 | 386 |
| conditional f≥0, 4 bps | −6.99 | −4.1 | 1.24 | 32.2 | 4023 | 386 |

Decomposition (taker always-on): funding = +2927 USD, fees = −16, basis = 0 (no spot), on $10k over
1615 days.

## Findings

1. **The premium reproduces.** Always-on BTC carry ≈ **6.8%/yr** net of taker fees — the low end of
   the validated "+8–12%/yr" (the higher figures include 2021's elevated funding, before our perp
   window). Fees are negligible because always-on holds continuously (1 round-trip / 4.4yr). The
   validated premium is confirmed on the new engine. **GO.**

2. **Always-on ≫ conditional.** Gating on funding sign churns 386 round-trips and goes **negative** —
   the small negative-funding periods cost less than the fees to dodge them. Hold through; don't time.

3. **⚠️ Sharpe ~12 / maxDD 0.38% is a v1 MODEL ARTIFACT — do not take at face value.** With no spot
   data the engine zeroes the basis term, so the delta-neutral position looks *riskless* (only
   funding-rate jitter contributes variance). Real carry carries basis-blowout, funding-spike,
   margin/liquidation, and execution risk the v1 model omits → realistic Sharpe ≈ **2–4** (still
   excellent). The `binance_spot` klines source (deferred A3 precision item) is what converts this
   artifact into a measured risk number.

## Update 2026-06-04 — measured basis (path a complete)

Built the free `binance_spot` klines source (6 unit tests) + enhanced the engine with **per-period
basis mark-to-market** (`price P&L = −q·Δ(perp−spot)`, q=notional/entry_perp) so basis *risk* —
not just convergence — shows up in the daily-return variance.

**Two findings during the measurement:**

1. **Prod `market_data` BTCUSDT is SPOT-sourced, not perp.** Verified: prod close (65007.27) matches
   Binance *spot* exactly; genuine fapi perp was 64971.70 (~5.5 bp lower) at the same bar. The funding
   harvest result is unaffected (funding is price-independent), but the perp leg + basis were re-pulled
   from `fapi.binance.com` futures klines. *(Platform note: single-symbol "perp" strategies are being
   backtested on spot prices — worth a separate look.)*

2. **Measured basis (genuine fapi perp − spot):** mean **−2.9 bp**, sd **3.8 bp**, range −21 / +32 bp.
   Tight, as expected for deep-liquidity BTC.

**Risk with measured basis (always-on taker):** return unchanged at **6.81%/yr** (basis nets ~+$7),
vol 0.38% → **0.56%**, Sharpe 12.0 → **8.1**. But carry daily returns are autocorrelated (funding
regimes persist), so √252-on-daily still overstates. Re-annualising on less-autocorrelated buckets:

| annualisation | Sharpe |
|---|---|
| daily ×√252 | 8.1 |
| weekly ×√52 | 6.4 |
| **monthly ×√12 (honest)** | **3.6** |

**The deployable Sharpe is ≈ 3.6** — squarely in the realistic carry range, not the 8–12 artifact.
Still an excellent, low-correlation ~6.8%/yr stream — an ideal pool member.

## Pool-admission decision (deferred to operator)

Return + risk are now both honest. **NOT auto-admitted** — two reasons remain: (i) the engine is
uncommitted/undeployed, and (ii) the pool's `annualised_sharpe` uses daily ×√252, which would weight
this carry on the inflated ~8 rather than the honest ~3.6. Recommended admission path: deploy the
engine, then admit using the **monthly-annualised (autocorrelation-honest) Sharpe** — or add an
autocorrelation haircut to the pool's Sharpe for carry-type (autocorrelated) members.

Reproduction: `.carry_validation/run_a3.py` (+ `fetch_spot.py`, `fetch_perp_fapi.py`; local scratch,
prod/Binance exports not committed).
