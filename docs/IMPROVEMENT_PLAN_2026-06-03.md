# Research Improvement Plan — From Tapped Majors to an Orthogonal Signal Portfolio

**Author:** quant-research diagnosis, 2026-06-03
**Supersedes:** nothing (additive to `RESEARCH_FINDINGS_2026-06-03.md` + `RESEARCH_METHODOLOGY.md`)
**One-line thesis:** the binding constraint is **orthogonal alpha supply + effective breadth**, not the
gates, not effort, not the strategy code. Every fix below buys breadth or orthogonality; nothing
loosens a gate.

---

## 0. Diagnosis recap (why the researcher kept failing)

Two `SIGNIFICANT_EDGE` hits in the program's entire history is **not** a malfunction. Root causes,
all evidenced in our own run logs:

1. **Breadth ≈ 1–2, not 120.** 6 archetypes × 5 majors × 4 intervals are correlated re-skins of one
   idea (OHLCV momentum/breakout) on ~0.8-correlated names. `IR ≈ IC·√breadth` — we have almost no
   independent breadth, so no information ratio to clear the bar.
2. **Brute-force grids fight the DSR multiplicity penalty.** Searching harder raises the significance
   bar. The gate is punishing p-hacking, correctly. (DCB-BTC-4h: PF 1.60 IS → 0.80 OOS.)
3. **Every edge found was non-stationary or cost-killed.** FCARRY lived only in the 2022 regime;
   XS_MOM 14d was +46%/yr gross / **−43% net** after turnover.
4. **Cheap pre-validation overstates survivors** (frictionless/daily flatters high-turnover signals).

**Conclusion:** the majors × price-action surface is efficiently arbitraged. The platform machinery
(gates, walk-forward, Signal Pool, House Book) is **built and correct but starved of orthogonal
inputs.** The plan is to feed it.

**North Star:** stop hunting one 20%/yr strategy. Build a portfolio of many **weak, genuinely
uncorrelated** signals netted via the pool: `combined Sharpe ≈ s·√N`. √N only pays if the N are
independent — so orthogonality is the whole game.

---

## 1. Workstreams (what to build)

### A — Two-leg delta-neutral carry engine  ⭐ highest leverage
**Why first:** H1a (pure funding carry, long-spot/short-perp) is the only **already-validated**
premium in the backlog — BTC **+8–12%/yr, positive every single year** — and it is `DEFERRED` for
exactly one reason: *the platform has no engine that can hold two legs.* This is a validated return
stream blocked purely by missing infrastructure. Building it also unlocks **H4 (basis/term-structure
carry)** for free — same engine.

**Scope:**
- **A0 (scoping, ~0.5d):** audit data readiness. Confirm we have (a) **spot** market_data for the
  carry legs (we have perp; spot may need plumbing), (b) raw funding accrual series (not just the
  clamped `funding_rate_z`), (c) perp–spot basis. Output: a go/no-go data checklist.
- **A1 (architecture decision, ~0.5d):** delta-neutral carry is a **continuous-accrual** return
  stream, not event-driven TA4j. Decide: extend the JVM backtest engine vs. a **Python
  delta-neutral backtester** in the orchestrator that emits a `backtest_run`-shaped row the existing
  `analyze.py` gates can score. *Leaning Python* — netting/accrual has no coherent TA4j exit-logic
  owner (same reasoning that picked House-Book Option A over a synthetic netted engine).
- **A2 (build, ~3–4d):** two-leg P&L = funding accrual − both legs' fees − basis-convergence drift −
  borrow/financing. Model rebalancing-to-neutral turnover honestly (this is where naive carry
  backtests lie).
- **A3 (validate):** reproduce the +8–12%/yr BTC result out-of-sample, then ETH. Run it through V11
  walk-forward. A delta-neutral stream with low correlation to the price-action book is an **ideal
  House-Book member** even at sub-10%/yr.

**Definition of done:** a carry backtest clears V11 walk-forward ROBUST and is admitted to the
Signal Pool via `POST /pool/evaluate` with positive marginal Sharpe.

---

### B — Orthogonal data plumbing — FREE TIER ONLY (operator directive 2026-06-03)
Ingest needs **zero architecture change** — a source is a drop-in `sources/<name>.py` module.
**No paid data until the free tier is exhausted.** Paid items are parked in §5, not in scope now.

- **B1 — Exhaust already-plumbed free on-chain (F2), $0. ← promoted to lead the orthogonal leg.**
  hashrate / active-addr / txcnt momentum (CoinMetrics community), stablecoin supply (DefiLlama),
  Fear&Greed — all **live but untested**. These are the closest free substitute for the (deferred)
  paid netflow signal. Caveat: daily/global regime proxies, weak as 1h *entry* signals → best as
  **ML-gate inputs**, not standalone alpha. $0 to rule in/out — do this first.
  - Tasks: confirm rows land in `feature_values` → if driving an ML gate, the
    `feature_values→train→signal_history→MLRegimeGateGuard` path; if a direct signal, a JVM patcher
    to `feature_store` (the V142 / `EthBtcCorrelationPatcher` pattern). **LIVE-smoke `POST /pull`
    after any source touch** — CI-green ≠ working (naive-UTC convention + register in `_KNOWN_SOURCES`).
- **B2 — Deribit options forward-accumulation ("plant the seed") — DO NOW, $0.** Free public API,
  no deep free history → ~12–18mo to walk-forward-grade. It's the long pole, so **start the clock
  immediately** even though it pays off last. ~6h build (`sources/deribit_options.py`).
- **B3 — Free Santiment tier, $0.** Deferred even within free tier — thin coverage, heaviest NLP
  lift. Only after B1/B2 are exhausted.

---

### C — Universe breadth (the only price-derived lever worth pulling)
More **independent** names raise real `√breadth` for cross-sectional/dispersion signals. BNB/XRP are
plumbed but **inert** (no `account_strategy` rows). Caveat: majors are ~0.8 correlated, so the
breadth gain is sublinear — this is **lower priority than A/B** and only worth it to give XS/dispersion
a fair shot. Add names per V124; seed XS_MOM rows; re-test dispersion with N≥8 **cost-aware**.

---

### D — Methodology enforcement (cheap, durable, prevents backslide)
Hypothesis-driven research is *adopted* but **not enforced in code/playbook** (per the methodology
doc's own honesty note). Without enforcement the autonomous loop will regress to brute-force grids.

- **D1:** update `GET /agent/playbook` + the quant-researcher spawn mandate to **require** a
  pre-registered economic thesis + lean test (no Cartesian grids) before a sweep is accepted.
- **D2:** fold **turnover cost into the cheap pre-validation event study**, or make a cost-aware
  backtest a **mandatory confirm step** before any `SIGNIFICANT_EDGE` is trusted. XS_MOM (+46% gross
  / −43% net) is the cautionary tale this closes.

---

### E — Pool activation (downstream of A/B producing signals)
The Signal Pool + House Book are built but **empty and inert**. Once ≥1 orthogonal signal walk-forwards
ROBUST: designate the prod book account (`ORCH_HOUSE_BOOK_ACCOUNT_ID`, currently unset → sync is a
no-op), provision member `account_strategy` rows, and take the first real admission. This is where the
`s·√N` thesis finally turns on.

---

## 2. Sequencing (respecting "one brick at a time", parallelizing only what's free)

| Phase | Work | Effort | Rationale |
|---|---|---|---|
| **0 — now, parallel** | B2 (plant Deribit seed) + D1/D2 (enforce methodology) | ~1.5d | B2 is the long pole — start its clock today. D is cheap insurance against grid-regression. Neither blocks A. |
| **1 — the unlock** | A0→A3 two-leg carry engine | ~4–5d | Validated premium, blocked only by infra. Single biggest expected-value move. Unlocks H4 too. |
| **2 — first orthogonal predictor** | B1 free on-chain (F2) → lean ML-gate hypothesis | ~2–3d | First uncorrelated input at **$0**; daily/global → ML-gate, not standalone. |
| **3 — turn on the portfolio** | E pool activation (carry + best free-on-chain signal as first members) | ~1d | `s·√N` goes live once ≥2 orthogonal members exist. |
| **4+ — compounding** | options VRP (B2 matured), C universe breadth, H4 basis carry on A's engine | ongoing | Each adds an independent signal; pool Sharpe compounds. |

**Free-tier note:** the whole Phase 0–3 path is **$0**. The carry engine (A) uses funding + spot data
that are already free; the orthogonal-data leg runs on free on-chain (B1) + the Deribit seed (B2). Paid
data (§5) only gets revisited if/when the free tier is provably exhausted.

---

## 3. Guardrails (do NOT do)

- **Do not loosen V11/V60 or re-grid the majors.** Loosening manufactures losers, not alpha. Every
  hour re-gridding a dead surface is an hour not plumbing orthogonal data — the actual constraint.
- **Do not trust gross PnL on high-turnover signals.** Cost-aware confirm before any `SIGNIFICANT`
  claim (D2).
- **Do not chase `funding_rate_z` > 5** — clamp artifacts (±20 clamp, ~279 BTC-1h bars are noise).
  Keep `minAbsRate8h` floor binding.
- **Respect the ops traps:** backtest windows ≤ ~2.5yr (poll cap), sequential feature_store PATCH,
  don't cancel+requeue mid-drain, live-smoke every new ingest source.

---

## 5. Parked — paid data (revisit only after free tier is exhausted)

Not in scope under the free-tier-first directive. Listed so the option isn't lost:

- **Exchange netflow** — CryptoQuant Advanced ~$29–99 (1mo→backfill→cancel ≈ one-time). Highest
  on-chain orthogonality; the strongest paid candidate. Revisit if free on-chain (B1) proves too
  coarse to clear the gate as a signal/gate input.
- **Options IV/skew history** — Tardis.dev historical Deribit ~$50–200 one-time. Only needed if the
  free B2 forward-accumulation seed is too slow and we want VRP sooner.
- **Sentiment (symbol-scoped)** — Santiment Pro ~$99. P3, heaviest lift.

(Verify prices before any purchase — approximate.)

## 4. Success metric

Not "find a 20%/yr strategy." The plan succeeds when the **Signal Pool holds ≥3 walk-forward-ROBUST
members whose pairwise return correlation is < 0.3**, and the pool's combined out-of-sample Sharpe
exceeds the best standalone member. That is the quant-firm endgame: breadth from orthogonality, not a
single hero trade.
