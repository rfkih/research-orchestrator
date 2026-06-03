# Research Methodology — Hypothesis-Driven (Fix 2)

**Status:** adopted 2026-06-03. Supersedes brute-force parameter-grid search as the
default research approach. The agent-facing version lives in the research journal
(`IDEA_BACKLOG: RESEARCH_METHODOLOGY_HYPOTHESIS_DRIVEN_2026-06-03`); this is the
canonical operator reference.

## The flaw it fixes

Two `SIGNIFICANT_EDGE` iterations in the program's entire history. Diagnosed root causes:

1. **Low signal diversity** — the 6 archetypes (DCB, VBO, MMR, ATR_MOM, CDC, XS_MOM)
   are all OHLCV/price-action re-skins. "120 surfaces" (6 × 5 symbols × 4 intervals)
   is a handful of correlated ideas tested 120 ways, not 120 independent shots.
2. **Brute-force grids fight the gate** — the DSR multiplicity penalty raises the
   significance bar with every trial. Searching *harder* makes the bar *higher*. This
   is p-hacking, and the gate is built to punish it.
3. **Grids find overfit local optima** that walk-forward rejects — e.g. DCB-BTC-4h:
   PF **1.60 in-sample → 0.80 out-of-sample**. The gate is working; the search is wrong.

**The gates are not the flaw.** Loosening them manufactures losers, not alpha. The
flaw is upstream: searching a narrow, efficient, low-diversity space with a method the
(correct) overfitting gates punish.

## The principle

> Research a **premium that has an economic reason to exist**, not a parameter grid.
> **Thesis → minimal harvest strategy → lean test → escalate.**

Fewer, well-motivated trials → lower multiplicity penalty → real edges can clear.

## The loop

1. Pick the top **unworked, data-ready** hypothesis from the backlog.
2. **Pre-register the thesis**: *why* the premium exists + the harvest mechanism + a
   **lean test** (a few motivated param combos, never a Cartesian explosion).
3. Sweep the **most-liquid surface first** (e.g. BTC-1h).
4. `SIGNIFICANT_EDGE` (even sub-10%/yr) → walk-forward → `POST /pool/evaluate`.
   Orthogonal signals are *ideal* House-Book members.
5. `NO_EDGE` → journal the falsification and move to the next hypothesis.
   **Do not re-grid a dead hypothesis** hoping for a different verdict.

## Rules

- **Lean over exhaustive** — every trial costs multiplicity; spend them deliberately.
- **Orthogonality first** — prioritise signals *uncorrelated with price-action*
  (funding, carry, dispersion, flow, vol). These populate the pool; correlated
  archetypes are near-substitutes the pool's marginal-Sharpe test correctly rejects.
- **Pre-register the economic rationale**, not just the param grid.
- **Falsify and move on** — one clean `NO_EDGE` ends the hypothesis.

## Why this pairs with the Signal Pool

The endgame is not one 20%/yr strategy (efficient markets don't hold one) — it's many
**weak, uncorrelated** signals combined via the pool (`combined Sharpe ≈ s·√N`). That
*requires* orthogonal sources, which is exactly what hypothesis-driven research
produces. A pool fed price-action archetypes is a bad pool (redundant). A pool fed
funding/dispersion/carry/vol is a good one. Fix 1 (orthogonal engines) and Fix 2
(hypothesis-first) are the same plan from two angles.

## Prioritised hypothesis backlog (economic strength × data readiness)

| # | Hypothesis | Economic premium | Data | Status |
|---|---|---|---|---|
| **H1** | **Funding carry (FCARRY)** | leverage-demand funding premium; fade the crowded side | `funding_rate_z` 2022→2026 BTC/ETH | **ACTIVE** (seeded) |
| H2 | Cross-sectional dispersion (XS_MOM) | relative-strength / dispersion premium | needs N>5 (2022 backfill → N=8 in progress) | NEXT |
| H3 | Funding-extreme reversal | crowded-positioning capitulation | funding ready | exploratory variant of H1 |
| H4 | Volatility risk premium | implied vol > realized (sell vol / fade IV) | needs Deribit options history (accumulate) | future |
| H5 | Basis / term-structure carry | perp–spot basis convergence | needs two-leg engine + spot | future / infra |
| H6 | Session / time-of-day effects | intraday/weekend calendar anomalies (a *different angle* on OHLCV) | ready | cheap exploratory |

H1 is live as the first test under this methodology (FCARRY seeded on the research
account, BTC/ETH × 1h/4h). See `HYPOTHESIS_FUNDING_CARRY_FCARRY_2026-06-03` in the
journal for its lean test design.
