# Dual-Track Parallel Research — Design Spec

**Date:** 2026-06-06
**Status:** Draft for operator review
**Owner:** quant-research platform
**Scope:** Run two `quant-researcher` loops concurrently — a **trading** track and a **hedging** track — each launched from its own CLI, isolated in the research orchestrator, fronted by a literature/forum-driven alpha-discovery pipeline. Adds a **4-factor neutralization model** and a **signal-level alpha-combination book** so the trading track can harvest weak-but-orthogonal *idiosyncratic* alpha (institutional paradigm) alongside the existing standalone-strategy gate.

---

## 1. Goal & motivation

The single autonomous `quant-researcher` loop works one alpha lifecycle at a time. The strategic state (per memory) is that **majors × price-action is tapped** — the gates work, the search space is exhausted. The unlock is *orthogonal, theory-driven* signals and *more research throughput*, not loosening gates.

This design delivers:

1. **Throughput** — two research loops running at the same time against the same orchestrator + DB.
2. **A second objective** — a *hedging* track judged on risk-adjusted improvement over a held book (not standalone alpha).
3. **Theory-first discovery** — each track is fronted by a multi-agent pipeline that reads academic papers **and** practitioner quant forums, formulates a parameter-light falsifiable signal, and **pre-registers** it before any backtest.
4. **Idiosyncratic-alpha framing** (the Citadel-gap fix) — a 4-factor model strips disguised beta so the researcher measures *idiosyncratic* edge, and a signal-level combination book accumulates weak-but-orthogonal residual alphas that the standalone gate would discard. This moves the trading track from "find one ≥10%/yr winner" toward "combine many weak, neutralized, orthogonal signals" — **alongside**, not replacing, V60.

Non-goals: promoting to live, deploying spec strategies, loosening V11/V60 for the trading track.

---

## 2. Decisions locked with operator

| # | Decision | Choice |
|---|---|---|
| 1 | Meaning of "parallel" | **Throughput** — two loops running at once |
| 2 | Hedging success gate | **Beats buy-hold on risk-adjusted** (lower maxDD and/or higher Sharpe, return within tolerance) |
| 3 | Isolation mechanism | **Approach A** — JSONB `track` tag, no Flyway migration |
| 4 | Launch model | **Per-CLI slash command** `/research-track <trading\|hedging>` |
| 5 | Steering | **Autonomous, but checkpoints to operator** at decision points |
| 6 | Discovery ordering | **Confirmatory** — theory → pre-register → single test |
| 7 | Web access | Granted to **discovery stage only**, not the mechanical loop |
| 8 | Discovery sources | Academic papers **+ practitioner quant forums** |
| 9 | Rollout | **Local-dev validation first, then prod via SSH tunnel** |
| 10 | Alpha-combination framing | **Alongside** V60 — V60 stays frozen for standalone graduates + live picker; a combination book accumulates sub-threshold orthogonal idiosyncratic signals |
| 11 | Factor model | **4-factor crypto:** market (BTC beta), momentum (XS), carry (funding/basis), volatility (DVOL/realized) |
| 12 | Combination weighting | **HRP on factor-neutral residual returns** (reuse shipped pool/house-book machinery) |

---

## 3. Architecture overview

```
  CLI #1  ── /research-track trading ──►  quant-researcher (agent_name=…-trading, track=trading, gate=V60)
  CLI #2  ── /research-track hedging ──►  quant-researcher (agent_name=…-hedging, track=hedging, gate=beats_buy_hold)
                                                   │
                  ┌────────────────────────────────┴───────────────────────────────┐
                  ▼                                                                  ▼
        Discovery pipeline (per track, web fan-out)                       Validation loop (existing)
        D1 source sweep → D2 synthesis → D3 formulate                     queue → backtest → gate →
        → D4 pre-register HYPOTHESIS + queue spec                          walk-forward → graduate
                                                   │
                                                   ▼
                       research-orchestrator (:8082, single uvicorn worker)
                         track-scoped: /agent/state?track= , /tick/drain?track= , /queue
                                                   │
                                  PostgreSQL (research_queue, research_journal, …)
                                  track carried in JSONB: sweep_config / structured_data
```

Two loops share infrastructure; isolation is by a `track` tag. The queue's existing `FOR UPDATE SKIP LOCKED` claim already makes concurrent drains safe — we add a track predicate.

---

## 4. Component 1 — Track isolation (JSONB tag, no schema migration)

**Discriminator.** `track ∈ {"trading","hedging"}` carried in existing JSONB columns:
- `research_queue.sweep_config->>'track'` — stamped on queue insert.
- `research_journal.structured_data->>'track'` — stamped on every journal write (HYPOTHESIS, RUN_SUMMARY, NULL_SCREEN_RESULT, SESSION_CHECKPOINT, specialist requests/verdicts).

**Absent tag ⇒ today's global behavior.** Fully backward-compatible and reversible.

**Scoped reads.** Thread an optional `track` arg into `repo/agent_state.get_state_digest` and its helpers, exposed as `?track=` on the API:
- `GET /agent/state?track=…` filters: `_lockout_state`, `_last_run_summary`, `_active_hypotheses`, `_last_null_screen_per_surface`, `_pending_specialist_reviews`, `_recent_specialist_verdicts`, SESSION_CHECKPOINT resume.
- `POST /tick/drain?track=…` and `POST /tick` — claim/operate only rows whose `sweep_config->>'track'` matches.
- `POST /queue` — require/stamp `track`; the re-discovery DISCARD gate is scoped per-track (a hedging discard must not block a trading axis-set).

**Free isolation.** ML-training budget + idempotency already key on `agent_name`, which differs per loop — those separate automatically.

**Why two loops can't poison each other:** lockout (`ARCHETYPE_EXHAUSTION` 24h), the last-RUN_SUMMARY resume anchor, and null-screen surface-skips all become per-track.

**Files:** `src/orchestrator/repo/agent_state.py`, `repo/queue.py`, `repo/queue_write.py`, `repo/journal.py`, `api/agent.py`, `api/tick.py`, `api/queue.py`, `services/tick.py`, `services/tick_drain.py`.

---

## 5. Component 2 — Hedging gate (`beats_buy_hold_risk_adj`)

A new verdict path in the analyze service, selected when `track=hedging`. The trading track's V60 path is untouched.

- **Benchmark:** buy-hold of the underlying (BTC/ETH) over the same window; reuse the EMA-trend allocation backtest machinery already validated (see `project_ema_trend_allocation_win`).
- **Pass condition (deterministic):** `PASS` iff **(a)** return floor holds: `CAGR_strategy ≥ CAGR_buyhold − tol_cagr`, **AND (b)** at least one *material* risk improvement: `Sharpe_strategy − Sharpe_buyhold ≥ θ_sharpe` **OR** `maxDD_buyhold − maxDD_strategy ≥ θ_dd`. The floor prevents a hedge from "winning" by sitting in cash and killing CAGR. `tol_cagr`, `θ_sharpe`, `θ_dd` are pinned named constants (exact values set in the implementation plan, with pin-tests).
- **ROBUST requirement:** the same walk-forward stability requirement applies — a hedge that only beats buy-hold in one regime is not ROBUST.
- **Constants pinned** as named values with pin-tests, mirroring the V11/V60 discipline.

**Boundary respected:** this is an *additive, separate* objective — not a relaxed V11/V60. V11/V60 stay frozen for the trading track.

**Files:** `src/orchestrator/services/analyze.py` (+ a `services/hedging_gate.py` helper), `api/constants.py`.

---

## 6. Component 3 — `/research-track` launch command

A project slash command run once per CLI:

```
/research-track trading      # CLI #1
/research-track hedging      # CLI #2
```

On invocation:
1. **Resolve track config** → `{agent_name, track, gate}`:
   - `trading` → `quant-researcher-trading`, `track=trading`, gate=V60
   - `hedging` → `quant-researcher-hedging`, `track=hedging`, gate=`beats_buy_hold_risk_adj`
2. **Confirm target** — print whether this CLI points at **prod (tunnel)** or **local dev**; refuse to start a second loop against the empty local dev DB by accident.
3. **Boot the `quant-researcher` sub-agent** scoped to the track — it calls `GET /agent/state?track=…`, drains `/tick/drain?track=…`, stamps every write with the track tag, and consults the discovery pipeline (Component 6) when it needs a new hypothesis.

Each CLI is a top-level session, so `Agent` is available — no nested-spawn restriction.

**Files:** `.claude/commands/research-track.md` (or a skill under `.claude/skills/`).

---

## 7. Component 4 — Checkpoint-to-operator flow

Reuses the **existing async-checkpoint pattern** (the one specialist reviews already use) — no new protocol.

- The track's researcher runs autonomously until a **decision point**: graduation candidate ready, pivot decision, or archetype exhaustion.
- It **writes a track-tagged `SESSION_CHECKPOINT` journal row and returns** to the CLI's main session with a compact checkpoint payload — instead of silently terminating or auto-deciding.
- The CLI **surfaces the checkpoint to the operator** and waits.
- The operator answers (approve graduation / pick pivot / authorize exhaustion bypass).
- The main session **resumes the sub-agent** with the decision via `SendMessage` (context preserved), and the loop continues.

Net: hands-off *between* decision points; stops and asks at the three moments that matter — independently per track.

---

## 8. Component 5 — Alpha-discovery pipeline (per track, multi-agent fan-out)

Runs *ahead* of each track's validation loop; emits a pre-registered hypothesis + queue-ready sweep spec. Implemented **as a Workflow** (deterministic multi-agent fan-out).

- **D1 — Source sweep (parallel, web):**
  - *Academic:* arXiv q-fin, SSRN, Google Scholar, journals.
  - *Practitioner forums:* Quantitative Finance StackExchange, QuantConnect forum, Wilmott, Nuclear Phynance, r/algotrading, Elite Trader.
  - Track-specialized: trading → return-predictive alpha; hedging → allocation / vol-targeting / crisis-hedge methods.
- **D2 — Synthesis + dedup:** collapse to distinct candidate *mechanisms*; keep only those testable on available data (BTC/ETH/SOL/BNB/XRP, `feature_store`, DVOL, funding, on-chain seeds).
- **D3 — Mathematical formulation:** turn each mechanism into a **parameter-light, falsifiable equation** with a **predicted sign**; **pre-register** (hypothesis statement + signal spec + data surface) *before any backtest*.
- **D4 — Handoff:** write the track-tagged `HYPOTHESIS` journal row + a queue-ready sweep with a **declared `n_trials`**; the existing loop runs the single confirmatory test.

**Methodological discipline (hard requirement):** confirmatory only. Pre-registration in D3 precedes any backtest, so the loop's statistical check is a single confirmatory test, not scan-then-formalize. If exploratory scanning is ever used, **every pattern examined must be counted into DSR `n_trials`** and the survivor must clear out-of-sample walk-forward — nothing graduates on in-sample discovery.

**Tool boundary:** `WebSearch`/`WebFetch` live only in D1–D2 (discovery). The mechanical `quant-researcher` loop keeps its existing toolset (`Bash/Read/Grep/Glob/Write/Edit`) and never touches the web — the gates stay web-isolated.

**Files:** a workflow script (discovery pipeline); orchestrator endpoint to accept a pre-registered hypothesis + sweep spec (reuse `POST /queue` + journal HYPOTHESIS write).

---

## 8b. Component 6 — Factor model & neutralization (`services/factor_model.py`)

**Purpose:** isolate *idiosyncratic* alpha from disguised factor exposure. Today a signal whose "edge" is really momentum-factor or carry-premium exposure passes the gate and then dies out-of-regime (the ATR_MOM / FCARRY non-stationarity pattern). This component measures and strips that exposure.

**Factor return series (daily, computed in the orchestrator):**
- **MARKET** — BTC daily return (proxy for the crypto market beta). Cap-weighted basket optional later.
- **MOMENTUM** — cross-sectional momentum factor: long recent winners / short losers across the universe. Reuses the shipped `XS_MOM` / `cross_sectional_rank` machinery (V144 universe column).
- **CARRY** — funding/basis carry factor: long high-funding / short low-funding leg. Sourced from funding data + the spot/perp basis now being plumbed (`binance_spot`).
- **VOLATILITY** — vol factor from DVOL / realized-vol (e.g. low-vol-minus-high-vol return). Uses the Step-4 DVOL features.

**Neutralization:** for a candidate signal's daily return series, run a time-series OLS regression on the 4 factor returns →
`r_t = α + β_mkt·MKT_t + β_mom·MOM_t + β_carry·CARRY_t + β_vol·VOL_t + ε_t`.
The **residual series `ε_t`** (idiosyncratic return) is what feeds the gate and the combination book. Outputs: factor betas (exposure report), idiosyncratic α + its t-stat, residual Sharpe / PSR on residuals.

**Disguised-beta flag:** if the standalone edge largely vanishes after neutralization (idiosyncratic α t-stat below a pinned threshold while raw return was significant), the signal is flagged **DISGUISED_BETA** — it is *not* admitted to the combination book and the verdict notes that the apparent edge was factor exposure.

**Storage:** factor betas + residual metrics go into the existing iteration `metrics_snapshot` JSONB (no Flyway migration). Factor return series are cached per window.

**Reuse:** regression via numpy/statsmodels (already a declared analysis dep); `probabilistic_sharpe_ratio` from `analyze.py` applied to residuals.

**Files:** `src/orchestrator/services/factor_model.py` (new), wired into `services/analyze.py`; pinned constants in `api/constants.py`.

## 8c. Component 7 — Signal-level alpha-combination book (extends `services/pool.py`)

**Purpose:** accumulate the **weak-but-orthogonal idiosyncratic** signals that fail the standalone V60 gate but are exactly what an institutional book is built from. Runs **alongside** V60 — standalone winners still graduate normally; this book collects the rest.

**Admission path (signal-level, on factor-neutral residuals):** a candidate that *fails* V60 standalone is still evaluated for combination admission. Admit iff **all** hold:
1. **Idiosyncratic significance** — residual α t-stat (or PSR on residuals from Component 6) clears a pinned threshold. (Strips disguised beta.)
2. **Predictive content** — information coefficient (rank correlation of signal vs forward return) is significant and the right sign vs the pre-registered hypothesis.
3. **Adds uncorrelated return** — inserting it raises the combination book's HRP-weighted Sharpe by > θ (the existing `marginal_sharpe_contribution`, **fed residual series instead of raw returns**).
4. **Low redundancy** — `max_abs_corr` to existing members below a pinned cap.

**Weighting:** HRP on the residual return series — reuse `_hrp_for` / `hrp_weights` / `spearman_corr_matrix` verbatim; the only change is the input series (residual, not raw).

**Deliverable:** the **combined neutralized book** is itself evaluated for a Sharpe/robustness bar (walk-forward stable). That aggregate book — not any single member — is the trading-track's second-class output, additive to standalone graduates.

**Relationship to the shipped pool:** the existing Signal-Pool / House-Book (Phase 0) admits *whole strategies* on *raw* returns. Component 7 is the same admission math operating one level down — *signals* on *residual* returns. Members are tagged (`kind=signal_combination`) so the two coexist in the same tables without a migration.

**Files:** extend `src/orchestrator/services/pool.py` (residual-fed admission variant) + `api/pool.py`; IC computation helper in `services/analyze.py`; pinned constants in `api/constants.py`.

> **Dependency:** Component 7 consumes Component 6's residual series. Build #6 before #7.

## 9. Testing & rollout

**Unit (pure):**
- Track-scoping predicates in `agent_state` queries (lockout / run-summary / null-screen filtered correctly per track).
- Hedging-gate math (maxDD/Sharpe vs buy-hold + return-tolerance floor) with pinned-threshold tests.
- **Factor model:** regression recovers known betas on synthetic series (e.g. a pure-MARKET series → β_mkt≈1, residual≈0); residual α t-stat math; DISGUISED_BETA flag fires when raw edge is significant but residual α is not — pinned-threshold tests.
- **Combination admission:** residual-fed `marginal_sharpe_contribution` matches the raw-fed result when factors are zero (reuse-equivalence); IC sign/significance check; redundancy cap rejects a near-duplicate signal.

**Integration (pytest-postgresql):**
- Two tracks' queue + journal rows coexisting.
- A hedging `ARCHETYPE_EXHAUSTION` does **not** appear in `GET /agent/state?track=trading`.
- Re-discovery DISCARD gate scoped per-track.
- A V60-failing signal with significant *residual* α is admitted to the combination book; a V60-failing signal whose edge is pure beta is **rejected** (DISGUISED_BETA).
- `kind=signal_combination` members and existing strategy-level pool members coexist without collision.

**Backward-compat:**
- A no-`track` call returns today's global digest unchanged.

**Rollout:**
1. Build + green on **local dev**; validate isolation with two short dummy sweeps (one per track) before any prod run.
2. Point both CLIs at **prod via the SSH tunnel** for real research.
3. Reversible at every step: drop the `track` tag → service behaves exactly as today.

---

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Data-mining via discovery (multiplicity) | Confirmatory ordering + DSR `n_trials` counting; pre-registration before backtest |
| Two loops accidentally on empty local dev DB | Launch command confirms prod-vs-local target before starting |
| Hedging gate "wins" by killing return | Return-tolerance floor in pass condition |
| Soft JSONB tag (no DB constraint) | Acceptable for single-operator service; promote to a real column (Approach B) if tracks grow beyond ~3 |
| Web sources unreliable / paywalled | Forums + arXiv/SSRN are open; discovery degrades gracefully (fewer candidates), never blocks the loop |
| Factor model over-fit on tiny universe | Only 4 economically-motivated factors; betas are time-series (per-asset), not a fitted cross-section; factor set frozen + pin-tested |
| Factor returns themselves leak / are non-PIT | Factor series built from the same PIT feature surface as signals; embargo (the cheap fix flagged separately) applies to factor regressions too |
| Combination book = laundered multiplicity | Each admitted signal still counts into DSR `n_trials`; the *combined* book must pass walk-forward robustness, not just in-sample marginal Sharpe |
| Residual α is real but tiny / uncapacitied | Combination members still pass through capacity-judge before any live consideration |

---

## 11. Out of scope

- Live promotion / deployment of any graduated strategy (operator-gated, unchanged).
- Loosening V11/V60 for the trading track. The combination book is **additive**, not a relaxed V60.
- A real `track` DB column (Approach B) — deferred unless track count grows.
- A second orchestrator instance (Approach C) — rejected.
- A full Barra-style cross-sectional risk model — the 4-factor time-series model is the deliberate minimal version for a 5-symbol universe.
- Purged-CV embargo in walk-forward — a real flaw, but tracked separately as a cheap standalone fix, not part of this spec.
