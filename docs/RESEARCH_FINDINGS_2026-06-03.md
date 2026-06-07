# Research Findings & Strategic Recommendation — 2026-06-03

Session synthesis. Agent-facing version: `RUN_SUMMARY: SESSION_FINDINGS_2026-06-03`
in the research journal. Read alongside `RESEARCH_METHODOLOGY.md` and
`SIGNAL_POOL_DESIGN.md`.

## What was built (machinery — all live, all proven)

| Component | State |
|---|---|
| **Signal Pool / House Book** | Phase 0 (marginal-Sharpe admission) + 1-A (live weight overlay) + 1-B (per-symbol cap + marginal eviction) + 1-C (performance/attribution API), **two adversarial audit passes**. Deployed, ~610 tests green. **Empty — no member admitted.** |
| **Hypothesis-driven research (Fix 2)** | Thesis → lean test → escalate; prioritized economic-hypothesis backlog in the journal. Committed `RESEARCH_METHODOLOGY.md`. |
| **Data plane** | 1h/4h history extended to **2022** for all 8 symbols; 5m/15m alt-extension in progress. |

## What was tested — 3 hypotheses, all correctly rejected

| # | Hypothesis | Verdict | Why |
|---|---|---|---|
| H1b | FCARRY (funding-extreme reversal) | **FALSIFIED** | Real but **non-stationary** — edge lives only in the 2022 bear-capitulation regime; flat/dead 2024–26. 24/24 INSUFFICIENT_EVIDENCE. |
| H2a | XS_MOM, 7-day lookback | rejected | Cross-sectional momentum **decayed** — strong 2022–24, flat/negative 2025–26 (factor crowding / regime flip). |
| H2b | XS_MOM, 14-day lookback, N=8 | **FALSIFIED** | The event-study *survivor*, but the real backtest is **cost-killed**: PF ~1.02 gross vs **−41% to −75% net** (~50%/yr turnover cost from 1.3k–6.4k trades). |

## The core finding

**The 5-majors × price-action search space is genuinely tapped.** Every edge found
is one of: *non-stationary* (regime artifact), *cost-killed* (turnover eats the thin
gross edge), or simply *thin*. The **V11/V60 + walk-forward gates are working
correctly** — they reject all of it. The constraint is the **search space**
(efficient, low-diversity, crowded), **not the gates and not the effort.** Loosening
the gates would manufacture losers, not alpha.

## Methodology lesson (logged)

Hypothesis-first pre-validation (cheap, read-only event studies) is **excellent for
killing dead hypotheses** — it cheaply falsified FCARRY and 7d-XS_MOM, saving multi-hour
sweeps. **But it can overstate survivors:** a frictionless, daily-sampled event study
showed XS_MOM at +46%/yr gross; the real 1h backtest with costs showed −43%. The cheap
test ignores turnover cost and intraday noise. **Necessary, not sufficient — always
confirm a survivor with a cost-aware backtest before believing it.**

## Strategic recommendation — stop tuning the majors

The machinery (pool + methodology + gates) is **ready and correct**. The platform is
**starved of orthogonal alpha sources, not of machinery.** The unlock is one of:

1. **Orthogonal data + a new engine (highest leverage):**
   - **True funding-basis carry** — a *delta-neutral two-leg* trade harvesting the
     validated +8–12%/yr funding premium (the one FCARRY's directional bet *couldn't*
     capture). Needs a two-leg backtest capability — a real but bounded engine build.
   - **On-chain flow** (exchange netflow) — leading indicator, orthogonal to price.
     Needs paid data + a flow engine.
   - **Options vol-risk-premium** (Deribit IV vs realized) — accumulate history now.
2. **Universe breadth beyond crypto-majors** — less-efficient corners (more symbols /
   other markets) where cross-sectional/momentum can survive costs.
3. **Feed the pool orthogonal signals** — a pool of price-action archetypes is
   redundant (the marginal-Sharpe test rejects near-substitutes). It only compounds
   with genuinely uncorrelated sources (the above).

## Open items / loose ends

- **5m/15m alt backfill to 2022** — still completing in the background (useful for any
  future research regardless).
- **Two cross-sectional Phase-1 integration bugs** found by finally running the path,
  worked around (use a real symbol as the `hypothesis_audit` key instead of the `XS*`
  label; send ISO datetime not date for `backtest_window.start_time`). Proper fixes are
  small follow-ups: allow `XS*` in `chk_hypothesis_audit_symbol`; normalize date→datetime
  in the orchestrator; raise the JVM poll timeout for heavy multi-symbol backtests.
- **Local WIP commits held** (orchestrator + trading-engine), not deployed.
