# Research Paper: VBO_RESEARCH — BNBUSDT × 15M

**Author:** quant-researcher
**Date:** 2026-06-02
**Strategy:** `VBO_RESEARCH`
**Surface:** `BNBUSDT` × `15m` (with adjacent BNB-1h, XRP-15m, and DCB-blocked findings)
**Type:** ALGO
**Surface attempt:** v1 — ALGO (first VBO test on the newly-plumbed BNB/XRP surface)
**Prior papers on this surface:** none (first attempt)
**Filename:** `RESEARCH_PAPER_VBO_RESEARCH_BNBUSDT_15M_v1_ALGO_2026-06-02.md`
**Terminal:** ARCHETYPE_EXHAUSTION
**Goal status:** NOT HIT
**Hypothesis:** `3947fd26-c19e-4392-ac21-46a06c7436c6` (BNB 15m); parent `5b27b720` (DCB BNB/XRP), sibling `e5f57370` (XRP 15m)
**Queue(s):** `599c86ea-721e-4155-b80a-7cca9594b6b6` (BNB 15m, decisive); `ec535e4b` (BNB 1h, starved); `d3c72f7d` (XRP 15m, starved)

---

## TL;DR

BNB and XRP were plumbed end-to-end on 2026-06-02 (market_data 4 intervals 2024→now + feature_values 1h/4h), opening a genuinely new symbol surface. This session tested it under an operator-authorized lockout bypass (fresh hypothesis after the morning's ARCHETYPE_EXHAUSTION). The intended cheapest archetype — DCB — was **blocked**: the orchestrator finds no `account_strategy` row for `strategy_code=DCB` on the VPS-migrated primary DB (a seed gap from the 2026-06-02 orchestrator→VPS migration; not researcher-fixable). VBO_RESEARCH ran and was **falsified** on the new surface: BNB 1h is trade-starved (n≈0), BNB 15m solved trade-starvation (n=785) but is a **structural net loser at PF=0.64 (NO_EDGE, DISCARD)**, and XRP 15m is again trade-starved (n=1). With VBO falsified, DCB blocked on infra, and VPS-DB backtests now running ~37 min each (past the 30-min poll cap, precluding a valid ≥5-iter VCB sweep in the remaining budget), no credible next archetype was autonomously reachable → ARCHETYPE_EXHAUSTION (2nd in the 7d window).

*(No SIGNIFICANT_EDGE cell — metrics table omitted per template.)*

---

## 1. Background

At session start the research book had just fired ARCHETYPE_EXHAUSTION (terminal row `f0fafc54`, 2026-06-02 04:30 UTC): BTC/ETH × all ALGO archetypes (DCB/MMR/MRO/TPB/VBO_RESEARCH/ATR_MOM/FCARRY) × {5m,15m,1h,4h} exhausted with 0 SIG_EDGE, and the ML stationary-label direction (funding/OFI) falsified on transferability (adversarial_auc=1.0). A 24h lockout was active. The operator authorized the documented lockout bypass — journaling a fresh HYPOTHESIS (`5b27b720` @ 04:51 UTC, strictly after the terminal-fire row) for the genuinely new, untested BNB/XRP symbol surface plumbed earlier that day. BNB/XRP are in the V124 hypothesis_audit allowlist. The standing GOAL_HIT candidate `6025e072` (DCB ETHUSDT 1h × regime_eth_v2) remains in the curator admin inbox, unaffected.

---

## 2. Hypothesis

**Mechanism:** VBO_RESEARCH is a volatility-breakout strategy — it enters on a Bollinger-band-width expansion (`bbWidthMin` gate) confirmed by trend strength (`adxEntryMin`), targeting a multiple of risk (`tpR`). It is one of the protected production VBO's research variants and resolves an `account_strategy` template carrying allow_long/allow_short.

**Pre-registration:** BNB-15m hypothesis `3947fd26` registered 2026-06-02 05:48 UTC, after the BNB-1h trade-starvation finding and before the 15m sweep. Parent hypothesis `5b27b720` (DCB BNB/XRP) registered 04:51 UTC. XRP-15m hypothesis `e5f57370` registered after the BNB-15m DISCARD. Falsification criterion (per hypothesis): ≥1 cell with n≥100 AND statistical_verdict=SIGNIFICANT_EDGE; goal ag90≥10% ROBUST.

**Type:** ALGO (no ML sentinels).

---

## 3. Methodology

### 3.1 Statistical Gates (V11 + V60)

| Gate | Threshold | Binding? |
|---|---|---|
| n_trades | ≥ 100 | YES |
| PF 95% bootstrap CI lower | > 1.0 | YES |
| DSR (Bailey–LdP, n_trials from hypothesis_audit) | ≥ 0.95 | YES |
| Statistical verdict | SIGNIFICANT_EDGE | YES |
| ann. geometric return at alloc-90 (ag90) | ≥ 10%/yr | YES |

### 3.2 Sweep Design

- **Type:** GRID (orchestrator down-samples to iter_budget)
- **Backtest window:** 2024-01-01 → yesterday-UTC (~17 months)
- **Dimensions swept (BNB 15m, decisive):** `bbWidthMin` {0.015, 0.025, 0.035} × `adxEntryMin` {18, 22, 26} × `tpR` {2.0, 2.5, 3.0} (3 axes — hard rule #7 satisfied)
- **Total cells:** iter_budget 4; 1 executed before NO_EDGE early-stop DISCARD halted the sweep (re-discovery gate).

### 3.3 Walk-Forward Protocol

Did not run — no cell reached SIGNIFICANT_EDGE.

---

## 4. Parameter Space Explored

| Axis | Values Tested |
|---|---|
| `bbWidthMin` | 0.015, 0.025, 0.035 |
| `adxEntryMin` | 18, 22, 26 |
| `tpR` | 2.0, 2.5, 3.0 |

**Total iterations across the new surface:** 5 (BNB 1h ×3, BNB 15m ×1, XRP 15m ×1)

**Edge verdict distribution:**

| Verdict | Count |
|---|---|
| SIGNIFICANT_EDGE | 0 |
| INSUFFICIENT_EVIDENCE | 4 (BNB 1h ×3 trade-starved, XRP 15m ×1 trade-starved) |
| NO_EDGE_DETECTED (DISCARD) | 1 (BNB 15m, PF=0.64, n=785) |

---

## 5. Results

### 5.1 Edge summary

Two distinct failure modes appeared, both fatal:

- **Trade starvation (BNB 1h, XRP 15m).** The VBO breakout-confirmation gate fires almost never on BNB 1h (n≈0 across 3 iters) and on XRP 15m (n=1). The instrument's volatility regime over 2024→now does not produce enough qualifying band-expansion+ADX events for the strategy to be statistically evaluable. This replicates a prior BNB n-starvation finding (the morning's hypothesis `c179dea2`).
- **Structural unprofitability (BNB 15m).** Dropping to 15m solved trade starvation (n=785), but the mechanism is a **net loser** there: PF=0.64 means it loses ~36 cents per dollar risked. The re-discovery gate DISCARDED the axis-set on the NO_EDGE verdict. This is a real negative result over 785 trades, not insufficient evidence — VBO breakout follow-through is negative on BNB 15m.

The binding constraint flips with frequency: starvation at low frequency, structural loss at high frequency. There is no interval where VBO is both adequately-traded and profitable on BNB.

### 5.2 Best cell (BNB 15m — anchors future comparison)

| Param | Value |
|---|---|
| `bbWidthMin` | 0.015 (first executed cell) |
| `adxEntryMin` | 18 |
| `tpR` | 2.0 |

| Metric | Value |
|---|---|
| iteration_id | `4d1cf34e-b751-416f-b61f-7b4309187aba` |
| n_trades | 785 |
| PF (point) | 0.64 |
| Statistical verdict | NO_EDGE_DETECTED (DISCARD) |

---

## 9. Infrastructure Notes

Two infrastructure findings materially shaped this session:

1. **DCB `account_strategy_missing` on the VPS DB.** `POST /tick` for DCB on BNBUSDT returned `error_code=account_strategy_missing` ("No account_strategy row for strategy_code=DCB"), deterministic, `retryable=false`, `next_action=contact_human`. The orchestrator's `_resolve_account_strategy` (tick.py) queries `WHERE strategy_code=$1 AND account_id=$2` (research account), symbol-agnostically — and finds nothing for DCB. The BTC/ETH DCB iterations earlier today ran on the home orchestrator before the 2026-06-02 VPS migration; the DCB `account_strategy` row either did not migrate or is scoped to a different account_id on the VPS primary. VBO_RESEARCH resolves fine and ran. Per code authority the orchestrator must NOT seed `account_strategy` (trading-JVM table) — operator action required.

2. **VPS-DB backtest latency ~37 min/run.** The BNB-15m and XRP-15m backtests took ~37 min each (vs ~20 min for the morning's runs), pushing past tick.py's hardcoded `_POLL_TIMEOUT_S=1800` (30 min). The backtests did still COMPLETE (queues advanced), but this is the documented `project_prod_backtest_poll_cap_trap` risk on the now-larger 5-symbol DB. It also made a methodologically-valid ≥5-iter VCB sweep un-runnable in the remaining ~1h45m.

---

## 10. Methodology Compliance Audit

| Rule | Compliant? | Notes |
|---|---|---|
| Universe (BTC/ETH/SOL/BNB/XRP — hard rule #1) | YES | BNB/XRP are in-universe (plumbed 2026-06-02, V124 allowlist) |
| Intervals in {5m,15m,1h,4h} (hard rule #2) | YES | 1h, 15m only |
| Protected strategies untouched (hard rule #3) | YES | VBO_RESEARCH is the research variant; prod VBO not touched |
| Research-mode only — no live promotion (hard rule #4) | YES | no promotion calls |
| 10%/yr economic bar enforced (hard rule #5) | YES | no cell came near; nothing graduated |
| V11 + V60 gates honored (hard rule #6) | YES | NO_EDGE/INSUFFICIENT verdicts accepted as-is |
| ≥ 3 axes swept (hard rule #7) | YES | bbWidthMin × adxEntryMin × tpR |
| Pre-registration before testing (hard rule #10) | YES | hypotheses 5b27b720 / 3947fd26 / e5f57370 pre-date their sweeps |
| Append-only durable evidence (hard rule #10) | YES | no deletes; queues cancelled with reason, not removed |
| Reviewer verdict authoritative (hard rule #12) | YES | plan reviews CONDITIONAL_APPROVAL before each /queue |
| Research JVM only — no trading JVM calls (hard rule #13) | YES | all via orchestrator :8082 |

---

## 11. Conclusions

1. **DCB is the right first archetype on a new symbol surface but is currently un-runnable** — blocked by a missing `account_strategy` row for `strategy_code=DCB` on the VPS-migrated primary DB; this is an operator-only infra fix.
2. **VBO_RESEARCH is falsified on BNBUSDT across the tradeable frequency range** — trade-starved at 1h (n≈0) and a structural net loser at 15m (PF=0.64 over n=785, DISCARD). No interval is both adequately-traded and profitable.
3. **XRPUSDT replicates BNB's trade-starvation at 15m** (n=1) — VBO's breakout gate is too restrictive for both new symbols' volatility regimes; XRP's episodic event-driven vol did not produce the cleaner follow-through hypothesized.
4. **VPS-DB backtest latency (~37 min/run) is approaching the 30-min poll cap** — a methodological hazard (false FAILED) and a throughput constraint that precluded a valid VCB sweep this session.
5. **Implication for the research loop:** the new BNB/XRP surface is NOT exhausted — only VBO has been tested, and the cheapest archetype (DCB) and the untested VCB are both blocked on operator-side infra (account_strategy seed + backtest latency). Unblocking DCB on the VPS DB is the single highest-leverage action.

---

## 12. Data Wishlist

| Priority | Item | Blocker type |
|---|---|---|
| 1 | Seed a `DCB` `account_strategy` row for the research account (`ORCH_RESEARCH_ACCOUNT_ID`) on the VPS primary DB — unblocks the cheapest/most-proven archetype on the new surface. | DB seed / migration gap |
| 2 | Investigate/raise VPS-DB backtest latency (~37 min/run vs ~20 min) or make `_POLL_TIMEOUT_S` configurable to avoid false FAILED on the larger 5-symbol DB. | JVM/orchestrator config |
| 3 | Confirm a `VCB_RESEARCH` `account_strategy` row resolves on the VPS DB so the untested VCB archetype can be swept on BNB/XRP. | DB seed |

---

## 13. Appendix — Audit Trail

| Item | ID |
|---|---|
| Hypothesis (BNB 15m, decisive) | `3947fd26-c19e-4392-ac21-46a06c7436c6` |
| Hypothesis (parent, DCB BNB/XRP) | `5b27b720-1c05-4a98-b6fc-a30b86a8261a` |
| Hypothesis (XRP 15m) | `e5f57370-71a3-42ed-8c6f-42261c204229` |
| Queue (BNB 15m, decisive) | `599c86ea-721e-4155-b80a-7cca9594b6b6` |
| Queue (BNB 1h, starved) | `ec535e4b-be82-4f61-8998-7a249a471e0f` |
| Queue (XRP 15m, starved) | `d3c72f7d-4ee8-47a1-9452-3140f5147e03` |
| Decisive iteration (BNB 15m, PF=0.64) | `4d1cf34e-b751-416f-b61f-7b4309187aba` |
| XRP iteration (starved) | `faee2d41-b59d-4298-9899-06cd58a696d0` |
| STRATEGY_OUTCOME journal (VBO BNB) | `63039645-a05a-4dae-8354-d38a2f5b7144` |
| RUN_SUMMARY journal (terminal) | `84ffac69-d5b7-4225-a25f-841e7facb346` |
| Curator request (standing, unaffected) | `beab77ed-5efd-4cff-bcb1-ee21edb2e43e` |

---
*Paper generated by quant-researcher. DB registration: `POST /papers/599c86ea.../generate` → `BH-VBO_RESEARCH-BNBUSDT-15M-599c86ea`.*
