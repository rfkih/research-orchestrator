# Research Strategy Registry — Admin Page (Design Spec)

**Date:** 2026-06-16
**Status:** Approved (hybrid data source; Phase 1 + 2 both in scope)
**Repos touched:** `research-orchestrator` (endpoints), `blackheart` / trading-engine (one Flyway migration only), `Blackridge` / frontend (UI)

## 1. Problem

The operator wants an admin page that shows the ranked roster of every strategy the platform has researched — most-promising → least — with detailed status per strategy (live / lead / parked / data-gated / falsified), the quantitative cells (DSR, walk-forward verdict, annualized return, n_trials), the qualitative verdict ("beta-not-alpha", "real-but-uncertifiable", "2019-21 artifact"), evidence pointers, and a provenance link.

That ranked roster currently exists only as curated analysis (operator memory). The database holds the *quantitative* cells (`research_iteration_log`, `walk_forward_run`, `backtest_run`) and the falsified-reasons (`research_journal` `ANTI_PATTERN` rows), but **not** the promise ranking, the qualitative verdict tags, or the offline leads that never ran through the orchestrator (TOP-TRADER-LSR-FADE, the parked ALT_CAP_FADE re-eval, global-LSR contrarian).

## 2. Chosen approach — Hybrid (curated editorial layer + live metrics join)

A new curated table holds the editorial layer (rank, tier, verdict tag, thesis, detail, provenance, evidence pointers). At read time the orchestrator LEFT-JOINs **live** metrics for rows that carry a `strategy_code` (+ optional symbol/interval), so the numbers stay current and self-updating. Rows with no run (offline leads) still render from the curated layer with a "not-yet-run" badge. A server-computed **divergence flag** surfaces when the curated narrative and a fresh DB number disagree (e.g. marked LIVE but no enabled `account_strategy`; marked FALSIFIED but a live run now clears the 0.90 DSR gate).

Rejected alternatives: **live-only** (mechanical rank, drops offline leads + qualitative verdicts) and **curated-only** (numbers go stale, duplicates orchestrator data).

### 2.1 Why the endpoint lives in the orchestrator (not a new JVM controller)

`ResearchOrchestratorProxyController` (`blackheart-trading-engine`) maps `{"/api/v1/research-orch", "/api/v1/research-orch/**"}` as a **catch-all** that strips the prefix, forwards the rest verbatim to the orchestrator (`:8082`), and injects `X-Orch-Token` + `X-Agent-Name: dashboard` + `X-Viewer-Is-Admin` (resolved from the JWT SecurityContext) server-side. Therefore a new orchestrator endpoint at `/strategy-registry` is reachable from the browser at `/api/v1/research-orch/strategy-registry` with **zero** trading-JVM Java changes, and is already admin-aware. The frontend calls it via `apiClient` (trading JVM `:8080`), exactly like `orchestrator.ts` / `ranking.ts` / `researchPapers.ts`.

## 3. Data model — `strategy_research_registry` (Flyway `V182`)

New migration `blackheart-trading-engine/src/main/resources/db/flyway/V182__strategy_research_registry.sql` (head is V181). The orchestrator is a DML-only client of this table (per orchestrator CLAUDE.md — schema lives in the trading-JVM repo).

| column | type | notes |
|---|---|---|
| `registry_id` | UUID PK | `DEFAULT gen_random_uuid()` |
| `slug` | TEXT UNIQUE NOT NULL | stable kebab id (`top-trader-lsr-fade`) — idempotent seed + deep-link |
| `rank` | INT NULL | global promise rank (1..N) |
| `promise_tier` | TEXT NOT NULL | CHECK in `('TIER_A','TIER_B','TIER_C')` |
| `display_name` | TEXT NOT NULL | e.g. `TOP-TRADER L/S FADE` |
| `signal_family` | TEXT NULL | `positioning` / `trend` / `vrp` / `carry` / `breakout` / `mean_reversion` / `liquidation` / `cross_sectional` / `order_flow` / `basis` / `funding` |
| `strategy_code` | TEXT NULL | links to live data; NULL for pure offline leads |
| `symbol` | TEXT NULL | e.g. `BTCUSDT`; NULL for XS/universe strategies |
| `interval_name` | TEXT NULL | `5m`/`15m`/`1h`/`4h`/`1d` |
| `verdict_tag` | TEXT NOT NULL | `REAL_LEAD` / `REAL_UNCERTIFIABLE` / `BETA_NOT_ALPHA` / `DATA_GATED` / `PARKED` / `FALSIFIED` / `FALSIFIED_OOS` / `EXHAUSTED` |
| `lifecycle_status` | TEXT NOT NULL | `LIVE` / `LEAD` / `PARKED` / `DATA_GATED` / `FALSIFIED` |
| `thesis` | TEXT NOT NULL | one-line "what the edge is" |
| `detail` | TEXT NULL | markdown — the nuance / evidence / why it passes or fails |
| `evidence_iteration_id` | UUID NULL | explicit pin to a `research_iteration_log` row (no FK) |
| `evidence_walk_forward_id` | UUID NULL | explicit pin to a `walk_forward_run` row (no FK) |
| `evidence_backtest_run_id` | UUID NULL | explicit pin to a `backtest_run` row (no FK) |
| `journal_id` | UUID NULL | link to the `ANTI_PATTERN` / `STRATEGY_OUTCOME` row (no FK) |
| `memory_ref` | TEXT NULL | operator-memory filename (provenance) |
| `is_offline_lead` | BOOLEAN NOT NULL DEFAULT false | true = never ran through the orchestrator |
| `archived` | BOOLEAN NOT NULL DEFAULT false | soft delete |
| `created_time` | TIMESTAMPTZ NOT NULL DEFAULT now() | |
| `updated_time` | TIMESTAMPTZ NOT NULL DEFAULT now() | |
| `updated_by` | TEXT NULL | stamped from `X-Agent-Name` on mutation |

Indexes: `(archived, promise_tier, rank)`; `(strategy_code, symbol, interval_name)`.
**No foreign keys** to the append-only audit tables — the join is a tolerant LEFT JOIN, and evidence rows differ per environment.

### 3.1 Seed (idempotent, environment-agnostic)

The migration seeds the **top-20** as curated rows (editorial fields + `strategy_code`/`symbol`/`interval_name` + `memory_ref` + `slug`), with **evidence pointers left NULL** — live metrics resolve by `(strategy_code, symbol, interval)` lookup so the seed carries no environment-specific UUIDs. Offline leads set `is_offline_lead=true` and may use a synthetic `strategy_code` (or NULL).

Seed rows (rank · tier · slug · display · code/symbol/interval · verdict · status · family · offline · memory_ref):

```
A 1  top-trader-lsr-fade   TOP-TRADER L/S FADE        TOPTRADER_LSR_FADE / —(top-50) /1d  REAL_LEAD          LEAD       positioning      offline  project_positioning_xs_factors_2026-06-16
A 2  ema-band-btc          Slow-EMA trend (EMA_BAND)  EMA_BAND_BTC /BTCUSDT/1d            BETA_NOT_ALPHA     LIVE       trend            -        project_ema_trend_allocation_win
A 3  funding-carry-dn      Funding carry (Δ-neutral)  FUNDING_CARRY /BTCUSDT/1h           REAL_UNCERTIFIABLE LEAD       carry            -        project_funding_carry_engine
A 4  vrp-btc               VRP vol-timing (BTC)       VRP_BTC /BTCUSDT/1d                 REAL_UNCERTIFIABLE LIVE       vrp              -        project_vrp_btc_beta_alpha_decomp_2026-06-15
A 5  global-lsr-fade       Global-LSR contrarian      GLOBAL_LSR_FADE / —(top-50) /1d     REAL_LEAD          LEAD       positioning      offline  project_positioning_xs_factors_2026-06-16
A 6  liq-fade              LIQ_FADE (cascade fade)    LIQ_FADE /ETHUSDT/5m               DATA_GATED         DATA_GATED liquidation      offline  project_oi_quadrant_lifecycle
B 7  vrp-eth               VRP vol-timing (ETH)       VRP_ETH /ETHUSDT/1d                 REAL_UNCERTIFIABLE LIVE       vrp              -        project_v102_live_gate_and_dsr_trial_tax
B 8  vmt-btc               Vol-managed trend          VMT_BTC /BTCUSDT/1d                 BETA_NOT_ALPHA     PARKED     trend            -        project_oi_quadrant_lifecycle
B 9  dcb-eth-1h            DCB-ETH 1h (tpR-5.0)       DCB /ETHUSDT/1h                     REAL_UNCERTIFIABLE LIVE       breakout         -        project_dcb_eth_1h_live_param_swap_2026-06-15
B 10 alt-cap-fade          ALT capitulation fade      ALT_CAP_FADE / —(alts) /1h          FALSIFIED_OOS      PARKED     mean_reversion   -        project_alt_cap_fade_crosssectional_reeval_2026-06-15
B 11 xs-funding-carry      XS funding carry           XS_FCARRY / —(top-50) /1d           EXHAUSTED          PARKED     carry            offline  project_positioning_xs_factors_2026-06-16
C 12 ema-band-resp-btc     EMA_BAND_RESP              EMA_BAND_RESP_BTC /BTCUSDT/1d       FALSIFIED          FALSIFIED  trend            -        project_ema_band_responsive_hedge_2026-06-07
C 13 oi-quadrant-btc       OI_QUADRANT               OI_QUADRANT /BTCUSDT/4h             FALSIFIED          FALSIFIED  positioning      -        project_oi_quadrant_lifecycle
C 14 dcb-eth-4h            DCB-ETH 4h                DCB /ETHUSDT/4h                     FALSIFIED          FALSIFIED  breakout         -        project_dcb_eth_4h_falsified_2026-06-14
C 15 basis-mom             Spot-perp basis / BASIS_MOM BASIS_MOM /BTCUSDT/1d             FALSIFIED          FALSIFIED  basis            -        project_oi_quadrant_lifecycle
C 16 ofi-cvd-continuous    Taker/OFI/CVD continuous   OFI_CVD /BTCUSDT/1h                 FALSIFIED          FALSIFIED  order_flow       -        project_cvd_momentum_lead_2026-06-14
C 17 vbo-btc               VBO (vol breakout)         VBO /BTCUSDT/1h                     FALSIFIED          FALSIFIED  breakout         -        project_vbo_falsified_prefix_backtests_suspect
C 18 intraday-ema-trend    Intraday EMA trend (15m)   INTRADAY_EMA /BTCUSDT/15m           FALSIFIED          FALSIFIED  trend            -        project_intraday_ema_trend_falsified_2026-06-14
C 19 xs-dispersion         XS dispersion (IVRV/VoV)   XS_IVRV / —(5-name) /1d             FALSIFIED          FALSIFIED  cross_sectional  -        project_alpha_pack_2026-06-11
C 20 funding-z-trend       Funding-z trend filter     FUNDING_Z /BTCUSDT/1d              FALSIFIED          FALSIFIED  funding          -        project_funding_divergence_research_2026-06-11
```

`thesis` + `detail` for each row are filled from the corresponding memory note (one-liner + the nuance paragraph). Exact prose authored in the migration.

## 4. Backend — orchestrator (`research-orchestrator`)

New files mirroring the existing `api/rankings.py` + `repo/rankings.py` pattern (FastAPI `APIRouter`, `Depends(get_db_conn)`, asyncpg repo; JSONB read via `metrics_snapshot->'analysis'->>'dsr'` etc.). Register the router in `src/orchestrator/main.py`.

- `src/orchestrator/api/strategy_registry.py`
- `src/orchestrator/repo/strategy_registry.py`
- `tests/test_strategy_registry.py`

### 4.1 Endpoints (proxied at `/api/v1/research-orch/...`)

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/strategy-registry` | authed | list merged rows + tier/status counts; filters `tier`, `status`, `family`, `search`, `include_archived` |
| GET | `/strategy-registry/{id}` | authed | single merged row (full detail) |
| POST | `/strategy-registry` | **admin** | create curated row |
| PATCH | `/strategy-registry/{id}` | **admin** | partial update |
| DELETE | `/strategy-registry/{id}` | **admin** | soft delete (`archived=true`) |

Admin gating: mutations require header `X-Viewer-Is-Admin: true` (injected by the proxy from the JWT). Missing/false → `OrchestratorError` 403 `admin_required`. `updated_by` stamped from `X-Agent-Name`.

### 4.2 Live-metrics resolution (per row, LEFT-JOIN, tolerant)

1. If `evidence_iteration_id` set and the row exists → pull `dsr`, `psr`, `annualized_geometric_return_pct_at_alloc_90`, `sharpe_annualized`, `n_trades`, `pf_point_estimate`, `statistical_verdict` from that `research_iteration_log.metrics_snapshot`; `resolvedFrom="pointer"`.
2. Else if `strategy_code` (+ symbol/interval) set → resolve the best completed run for that key using the same join the ranking query uses (`research_iteration_log` → `backtest_run` on `backtest_run_id`, filter `backtest_run.asset`/`interval_name`, pick best by annualized return / most recent); `resolvedFrom="lookup"`.
3. Else → no live metrics (`resolvedFrom=null`), `isOfflineLead` drives the "not-yet-run" badge.

Walk-forward verdict: latest `walk_forward_run.stability_verdict` by `evidence_walk_forward_id`, else by `(strategy_code, instrument, interval_name)`.
Live flag: `EXISTS` an `account_strategy` row for `strategy_code` (+symbol) with `enabled=true` (and not `simulated`).

### 4.3 Divergence flag (server-computed)

- `lifecycle_status='LIVE'` but `isLive=false` → `"marked LIVE but no enabled account_strategy"`.
- `verdict_tag IN ('FALSIFIED','FALSIFIED_OOS','PARKED','EXHAUSTED')` but resolved `dsr >= 0.90` → `"marked {tag} but a live run now clears the 0.90 DSR gate — re-examine"`.
- else `flag=false`.

(The 0.90 gate constant is read-only context, not a gate change — methodology contract untouched.)

### 4.4 Response shape (camelCase, matches `strategy-ranking`)

```json
{
  "items": [{
    "registryId": "uuid", "slug": "vrp-btc", "rank": 4,
    "promiseTier": "TIER_A", "displayName": "VRP vol-timing (BTC)",
    "signalFamily": "vrp", "strategyCode": "VRP_BTC", "symbol": "BTCUSDT", "intervalName": "1d",
    "verdictTag": "REAL_UNCERTIFIABLE", "lifecycleStatus": "LIVE",
    "thesis": "...", "detail": "...markdown...",
    "memoryRef": "project_vrp_btc_beta_alpha_decomp_2026-06-15",
    "isOfflineLead": false, "archived": false,
    "evidence": {"iterationId": null, "walkForwardId": null, "backtestRunId": null, "journalId": null},
    "live": {"dsr": 0.053, "psr": 0.997, "annualizedReturnPct": 46.0, "sharpeAnnualized": 1.21,
             "nTrades": 84, "profitFactor": null, "statisticalVerdict": "INSUFFICIENT_EVIDENCE",
             "walkForwardVerdict": "INSUFFICIENT_EVIDENCE", "isLive": true,
             "resolvedFrom": "lookup", "backtestRunId": "uuid"},
    "divergence": {"flag": false, "reason": null},
    "updatedTime": "2026-06-16T...", "updatedBy": null
  }],
  "total": 20,
  "tierCounts": {"TIER_A": 6, "TIER_B": 5, "TIER_C": 9},
  "statusCounts": {"LIVE": 4, "LEAD": 3, "PARKED": 3, "DATA_GATED": 1, "FALSIFIED": 9}
}
```

## 5. Frontend (`Blackridge`)

- **Route:** `src/app/(dashboard)/admin/research-strategies/page.tsx` — `'use client'`, gated by `useIsAdmin()` + `useAuthHydrated()` (existing admin pattern).
- **API module:** `src/lib/api/researchRegistry.ts` — uses `apiClient`, `BASE = '/api/v1/research-orch/strategy-registry'` (research-orch goes through the trading-JVM proxy = `apiClient`, like `orchestrator.ts`). Functions: `listRegistry(filters)`, `getRegistryEntry(id)`, `createRegistryEntry(body)`, `updateRegistryEntry(id, patch)`, `archiveRegistryEntry(id)`. Coerce wire numbers with `toNumOrNull` (`@/lib/api/coerce`).
- **Types:** `src/types/research.ts` — `ResearchRegistryEntry`, `RegistryLiveMetrics`, `PromiseTier`, `VerdictTag`, `LifecycleStatus`, `RegistryListResponse`. (`Backend*` wire types stay in the api module.)
- **Hook:** `useQuery(['research-registry', filters], …, { staleTime: 60_000 })`; mutations via `useMutation` + invalidate.
- **Layout:** `PageHeader` (eyebrow "Research" / title "Strategy Research Registry") → `StatCard` row (Live / Leads / Parked / Data-gated / Falsified) → filter bar (tier · status · family `Select` + debounced search via `useDebouncedSearchPage`, filters sent server-side — no client-side filtering) → **table grouped by Tier A/B/C** with group headers. Columns: rank · name (+family badge) · status badge · verdict-tag badge · DSR (mono) · WF verdict · ann-return% (mono, profit/loss color) · n_trades · symbol/interval · ⚠ divergence.
- **Detail dialog:** thesis + `detail` markdown + live metrics block + evidence ids (copyable; link `strategy_code` to the leaderboard filtered by code where a route exists) + `memoryRef`.
- **Edit/create dialog (Phase 2):** RHF controlled-`<div>` form (no `<form>` element). Fields: rank, tier, name, family, code/symbol/interval, verdict, status, thesis, detail, memoryRef, isOfflineLead, evidence ids. Admin-only; surfaced via "Add" button in `PageHeader` and a per-row "Edit"/"Archive" action.
- **Nav:** add a sidebar link under the admin group (`Sidebar.tsx`).
- **Theming:** strictly token-driven (`bg-bg-surface`, `text-text-primary`, semantic `--color-*`/`--tint-*`), IBM Plex, `font-mono tabular-nums` for numbers. No hardcoded hex.

## 6. Phasing

- **Phase 1:** migration + seed + `GET` endpoints + read-only page (grouped table, stat cards, filters, detail dialog, live join, divergence flag, nav link).
- **Phase 2:** `POST`/`PATCH`/`DELETE` endpoints (admin-gated) + create/edit/archive dialog + row actions.

## 7. Testing

- **Orchestrator (`PYTHONPATH=src pytest -q`):** pure-function tests (divergence calc, tier/status counting, DTO mapping) run anywhere; DB tests behind `@pytest.mark.integration` (`pytest-postgresql`): list returns seeded rows, filters, live-join resolves metrics for a row with a matching `backtest_run`, offline lead → null live, divergence cases, admin gating (POST without `X-Viewer-Is-Admin` → 403), create/patch/archive round-trip.
- **Migration:** test DB applies `V182`; assert 20 seeded rows + `slug` uniqueness + tier CHECK.
- **Frontend:** vitest for `researchRegistry.ts` (param building + mapping) and component render (table tiers/badges/divergence; dialog). `pnpm tsc --noEmit`. Build verified via the CRLF-trap workaround (typecheck + vitest + temp eslint-ignore build) — do not trust local `pnpm lint`/`next build` on a CRLF tree.

## 8. Rollout

- Work on `dev` in all three repos (already on `dev`); merge to `master` only after tests pass; deploy follows `master` via GitHub Actions (test → GHCR → VPS). Apply to dev too (it is dev).
- **Ordering:** `V182` must be applied before the orchestrator endpoint queries the table. Deploy trading-engine (migration) first/with; then orchestrator; then frontend. Locally, run the migration before exercising the orchestrator.
- Update catalogs in-PR: trading-engine `docs/agent-context/SCHEMA.md` + `MIGRATIONS.md` (new table + V182), orchestrator `README`/`GET /agent/playbook` if it enumerates endpoints.

## 9. Non-goals (YAGNI)

- No WebSocket live updates (60s `staleTime` poll is enough for ~20 rows).
- No auto-sync from operator-memory files (curation is manual/editorial).
- No FK constraints to append-only audit tables.
- No pagination (return all; ≤ ~50 rows expected).
