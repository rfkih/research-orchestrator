# Alpha-Discovery Pipeline (Phase 2b)

**Status:** SHIPPED (workflow authored 2026-06-07). Producer artifact — does not backtest.
**Spec:** `docs/specs/2026-06-06-dual-track-research-design.md` §8 (Component 5).

## What it is
A multi-agent **Workflow** that runs *ahead* of a research track's `/tick` loop and emits **pre-registered, falsifiable hypotheses** sourced from academic literature **and** practitioner quant forums. It is the "theory-first" front end that attacks the strategic dead-end (majors × price-action is tapped) by pulling in *orthogonal*, economically-motivated mechanisms instead of re-tuning the same axes.

It is **not** TDD'd Python — it's an orchestration script (`.claude/workflows/alpha-discovery.js`) that fans out web-search agents. Web tools (`WebSearch`/`WebFetch`) live **only** here; the mechanical `/tick` loop never touches the web, so the gates stay web-isolated.

## Pipeline (D1→D4)
1. **Source sweep** (parallel, web) — two scouts: academic (arXiv q-fin / SSRN / Scholar) and practitioner forums (Quant SE, QuantConnect, Wilmott, Nuclear Phynance, r/algotrading, Elite Trader). Track-specialized objective (trading = return-predictive alpha; hedging = beat buy-hold risk-adjusted).
2. **Synthesize** — collapse to distinct mechanisms, drop near-dupes, keep only those **testable on our data surface** (BTC/ETH/SOL/BNB/XRP + TA + funding/basis/DVOL/on-chain, 1h/4h/1d), orthogonal-first.
3. **Formulate** — turn each into a **parameter-light, falsifiable** signal: exact definition (equation/rule), **predicted sign**, minimal motivated params (not a grid), instrument+interval, and a **declared `n_trials`** (every combo to be examined — feeds DSR multiplicity).
4. **Pre-register pack** — returns the pre-registered hypotheses for handoff.

## How to run
Invoke via the Workflow tool (token-intensive — explicit opt-in):
```
Workflow(name="alpha-discovery", args={ track: "hedging", n_hypotheses: 3 })
```
Returns `{track, generated_count, hypotheses:[...], handoff}`.

## Handoff — CONFIRMATORY discipline (the hard rule)
For **each** returned hypothesis, **before any backtest**:
1. `POST /journal` `{entry_type:"HYPOTHESIS", status:"ACTIVE", title, content:<rationale>, structured_data:{track, mechanism, signal_definition, predicted_sign, declared_n_trials, sources}}` → capture `journal_id`. **This pre-registration must happen first.**
2. `POST /queue` `{track, instrument, interval_name, sweep_config:{...combos, track, n_trials:declared_n_trials}, hypothesis:<journal_id>}`.

Then the existing `/tick` loop runs the **single confirmatory** test. Rules:
- The HYPOTHESIS row MUST exist before `/queue` — this is pre-registration, not scan-then-formalize.
- Do **not** exceed `declared_n_trials` — every extra combo raises the DSR significance bar (anti-p-hacking).
- No orchestrator code was added for the handoff — it reuses `POST /journal` + `POST /queue` (Phase 1 added `track` to `/queue`).

## Relationship to the loops
`/research-track <trading|hedging>` consults this workflow when it needs a fresh hypothesis line, then pre-registers + enqueues per the handoff. Discovery (web) and validation (`/tick`, no web) stay cleanly separated.
