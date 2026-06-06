# Dual-Track Research — Phase 3 (Factor Model + Combination Book) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** The Citadel-gap fixes. (#6) A **4-factor neutralization model** that strips disguised beta from a candidate's returns, exposing *idiosyncratic* alpha and flagging `DISGUISED_BETA`. (#7) A **signal-level combination book** that admits weak-but-orthogonal idiosyncratic signals (failing the standalone V60 gate) by running the existing pool admission math on **factor-neutral residual** returns. Both run **alongside** the frozen V11/V60 — additive, never loosening.

**Architecture:** New `services/factor_model.py` builds four daily factor-return series from `market_data` (MARKET, MOMENTUM) and `macro_raw` (CARRY=funding, VOL=DVOL), regresses a candidate's daily equity returns on them (numpy OLS) → betas + idiosyncratic α (+ t-stat) + residual series. New admission path in `services/pool.py` (residual-fed) admits a candidate to the combination book iff residual-α significant **and** IC significant **and** it adds uncorrelated residual return (existing `marginal_sharpe_contribution`, fed residuals) **and** low redundancy. Members tagged `kind=signal_combination` coexist with the existing strategy-level pool. No Flyway migrations.

**Tech Stack:** Python 3.11, asyncpg, numpy, pytest + pytest-postgresql (Phase-1 `tests/integration/` harness). Prefix every pytest run: `ORCH_AUTH_TOKEN=dev-sentinel-not-for-prod ORCH_DB_DSN="postgresql://x:y@127.0.0.1:5432/none" PYTHONPATH=src python -m pytest <args>`.

**Repo/branch:** `blackheart-research-orchestrator`, worktree `C:\Project\.wt-dualtrack-orch`, branch `feat/dual-track-phase3` (off `feat/dual-track-phase2`). DML-only; **no migrations**.

**Spec:** `docs/specs/2026-06-06-dual-track-research-design.md` §8b/§8c (Components 6, 7). Build #6 fully before #7 (#7 consumes residuals).

**Reuse map (do NOT re-implement):**
- `repo/market_data.py:fetch_daily_closes` (Phase 2) — daily closes per symbol.
- `services/portfolio.py:daily_returns_from_trades` + `services/pool.py:_calendar_fill`, `annualised_sharpe`, `realised_vol`, `spearman_corr_matrix`, `hrp_weights`, `marginal_sharpe_contribution` — equity series + HRP/corr/marginal-Sharpe (feed RESIDUAL series instead of raw).
- `services/hedging_gate.py:_returns`, `_sharpe` — daily-return helpers (or factor it into a shared util if cleaner).
- `services/analyze.py:probabilistic_sharpe_ratio`, `bootstrap_pf_ci` — significance patterns.
- Data: `market_data(symbol, interval, start_time, close_price)`; `macro_raw(source, series_id, symbol, event_time, value)` — funding `series_id='funding_rate'` (per-symbol), DVOL `series_id IN ('deribit_btc_dvol','deribit_eth_dvol')`.

**Universe constant:** `FACTOR_SYMBOLS = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT"]` (the plumbed set).

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `src/orchestrator/repo/macro_raw.py` | Fetch funding + DVOL daily series from `macro_raw` | Create |
| `src/orchestrator/services/factor_model.py` | Build 4 factor series; OLS-neutralize; DISGUISED_BETA | Create |
| `src/orchestrator/services/combination_book.py` | IC + residual-fed admission decision (orchestrates pool fns) | Create |
| `src/orchestrator/services/pool.py` | (only if a residual variant needs a tiny seam) | Modify if needed |
| `src/orchestrator/api/combination.py` | `POST /combination/evaluate` (thin handoff) | Create |
| `src/orchestrator/main.py` | register the new router | Modify |
| `tests/integration/test_factor_model.py`, `tests/test_factor_model_unit.py`, `tests/integration/test_combination_book.py` | tests | Create |

---

## Task 1: `macro_raw` series repo (funding + DVOL)

**Files:** Create `src/orchestrator/repo/macro_raw.py`; test `tests/integration/test_factor_model.py`.

- [ ] **Step 1: Add `macro_raw` to the integration schema** — append a minimal `macro_raw` table to `tests/integration/schema_phase1.sql` (cols the query reads: `source VARCHAR(80)`, `series_id VARCHAR(120)`, `symbol VARCHAR(20)`, `event_time TIMESTAMPTZ`, `value NUMERIC(28,10)`; a simple `id BIGSERIAL PRIMARY KEY` is fine — skip the partitioning/composite PK from prod, irrelevant to reads). Source types from Flyway `V66__add_ml_sentiment_schema.sql`.

- [ ] **Step 2: Failing test** — seed `macro_raw` funding rows for BTCUSDT, assert `fetch_series(conn, series_id="funding_rate", symbol="BTCUSDT", start, end)` returns ascending `[(event_time, value)]`.

```python
# tests/integration/test_factor_model.py
import pytest
from datetime import datetime, timezone
pytestmark = pytest.mark.integration

@pytest.mark.asyncio
async def test_fetch_series_ascending(db_conn):
    from orchestrator.repo import macro_raw
    rows = [("binance_macro","funding_rate","BTCUSDT",
             datetime(2024,1,d,tzinfo=timezone.utc), 0.0001*d) for d in range(1,5)]
    await db_conn.executemany(
        "INSERT INTO macro_raw (source, series_id, symbol, event_time, value, source_uri, content_hash, ingestion_time) "
        "VALUES ($1,$2,$3,$4,$5,'u','h',$4)", rows)
    out = await macro_raw.fetch_series(db_conn, series_id="funding_rate", symbol="BTCUSDT",
                                       start=datetime(2024,1,1,tzinfo=timezone.utc),
                                       end=datetime(2024,1,5,tzinfo=timezone.utc))
    assert [round(v,4) for _,v in out] == [0.0001,0.0002,0.0003]
    assert out == sorted(out)
```
> Keep the integration-schema `macro_raw` columns and the test INSERT consistent (add `source_uri`/`content_hash`/`ingestion_time` NOT NULLs or give them defaults).

- [ ] **Step 3: Run → fail.**

- [ ] **Step 4: Implement** `fetch_series(conn, *, series_id, symbol=None, start, end)` → ascending `[(event_time, float(value))]`, half-open `[start,end)`, `WHERE series_id=$1 AND ($2::text IS NULL OR symbol=$2) AND event_time>=$3 AND event_time<$4 ORDER BY event_time ASC`. Params bound.

- [ ] **Step 5: Run → pass.** Full suite both markers green.

- [ ] **Step 6: Commit** — `git commit -m "feat(factor): macro_raw series repo (funding + DVOL)"`

---

## Task 2: Build the four factor return series (pure, given inputs)

To stay unit-testable, the factor math is PURE functions over already-fetched price/macro series; a thin async `build_factors(conn, start, end)` fetches + assembles.

**Files:** Create `src/orchestrator/services/factor_model.py`; unit tests `tests/test_factor_model_unit.py`; integration `tests/integration/test_factor_model.py`.

- [ ] **Step 1: Failing unit tests for the pure factor builders**

```python
# tests/test_factor_model_unit.py
from datetime import date
from orchestrator.services import factor_model as fm

def test_market_factor_is_btc_returns():
    closes = {date(2024,1,1):100.0, date(2024,1,2):110.0, date(2024,1,3):99.0}
    mkt = fm.market_factor(closes)          # {date: return}
    assert round(mkt[date(2024,1,2)],4) == 0.1
    assert round(mkt[date(2024,1,3)],4) == -0.1

def test_momentum_factor_long_top_short_bottom():
    # 2 symbols, A trending up, B down → long A short B → positive on up days
    closes_by_sym = {
      "A": {date(2024,1,d): 100.0+d for d in range(1,6)},
      "B": {date(2024,1,d): 100.0-d for d in range(1,6)},
    }
    mom = fm.momentum_factor(closes_by_sym, lookback=2, top_k=1)
    assert any(v != 0 for v in mom.values())
```

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3: Implement the four pure builders** in `factor_model.py`:
  - `market_factor(btc_closes: dict[date,float]) -> dict[date,float]` — daily simple returns of BTC.
  - `momentum_factor(closes_by_sym, *, lookback, top_k) -> dict[date,float]` — each day, rank symbols by trailing `lookback`-day return; daily factor return = mean(top_k next-day returns) − mean(bottom_k next-day returns). Point-in-time: rank uses returns up to t-1, realises t's return (no look-ahead).
  - `carry_factor(funding_by_sym: dict[str,dict[date,float]], returns_by_sym) -> dict[date,float]` — each day long the high-funding symbols / short low-funding; factor return = spread of next-day symbol returns. (Funding is the sort key; the factor's *return* is realised from price moves of the sorted legs.)
  - `vol_factor(dvol_series: dict[date,float]) -> dict[date,float]` — daily change in DVOL (Δlog or pct), the implied-vol factor. (Single series; BTC DVOL as the market-vol proxy.)
  Pin constants: `MOM_LOOKBACK=20`, `MOM_TOP_K=1`, with pin-test.

- [ ] **Step 4: Run → pass.**

- [ ] **Step 5: Integration test for `build_factors`** — seed `market_data` (5 syms) + `macro_raw` (funding + dvol); assert `await fm.build_factors(conn, start, end)` returns `{"MARKET":{...},"MOMENTUM":{...},"CARRY":{...},"VOL":{...}}`, each a date→return dict over the window. Degrade gracefully: if a factor's source data is absent, that factor is omitted (logged) — never crash.

- [ ] **Step 6: Implement `build_factors`** (fetches via `repo.market_data` + `repo.macro_raw`, calls the pure builders). **Step 7: Run → pass. Step 8: Commit** — `git commit -m "feat(factor): build MARKET/MOM/CARRY/VOL daily factor series"`

---

## Task 3: OLS neutralization + DISGUISED_BETA flag (pure)

**Files:** extend `factor_model.py`; unit tests `tests/test_factor_model_unit.py`.

- [ ] **Step 1: Failing tests**

```python
def test_neutralize_recovers_pure_market_beta():
    # candidate == 2x market exactly → beta_mkt≈2, residual≈0, alpha≈0
    from datetime import date
    mkt = {date(2024,1,d): 0.01*((-1)**d) for d in range(1,40)}
    factors = {"MARKET": mkt}
    cand = {d: 2.0*r for d,r in mkt.items()}
    res = fm.neutralize(cand, factors)
    assert abs(res["betas"]["MARKET"] - 2.0) < 1e-6
    assert abs(res["alpha"]) < 1e-6
    assert max(abs(x) for x in res["residuals"].values()) < 1e-6
    assert res["disguised_beta"] is True   # raw edge is pure beta

def test_neutralize_keeps_real_alpha():
    # candidate = market + constant idiosyncratic drift → alpha>0, residual nonzero
    ...
```

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3: Implement** `neutralize(candidate_returns: dict[date,float], factors: dict[str,dict[date,float]]) -> dict`:
  - Align candidate + factor series on common dates; build design matrix `X` (intercept + each factor), `y` = candidate returns; numpy `lstsq` → `alpha` (intercept), `betas` (per factor); `residuals` = y − Xβ as a date→value dict.
  - `alpha_tstat`: `alpha / se(alpha)` from OLS standard errors (compute `se` via residual variance · diag((XᵀX)⁻¹)).
  - `disguised_beta`: True iff the RAW candidate Sharpe is significant (PSR-ish or |annualised Sharpe|>some pinned floor) BUT `|alpha_tstat| < ALPHA_TSTAT_MIN`. Pin `ALPHA_TSTAT_MIN=2.0`.
  - Return `{alpha, alpha_tstat, betas, residuals, disguised_beta, n_obs}`. Guard `n_obs < MIN_FACTOR_OBS (=60)` → `{disguised_beta:False, reason:"insufficient_obs", residuals:{}}` (never neutralize on thin data).

- [ ] **Step 4: Run → pass. Step 5: pin-test `ALPHA_TSTAT_MIN`, `MIN_FACTOR_OBS`, `MOM_LOOKBACK`, `MOM_TOP_K`. Step 6: Commit** — `git commit -m "feat(factor): OLS neutralization + idiosyncratic alpha t-stat + DISGUISED_BETA"`

---

## Task 4: IC helper + residual-fed combination admission (pure-ish)

**Files:** Create `src/orchestrator/services/combination_book.py`; unit tests `tests/test_combination_book_unit.py`.

- [ ] **Step 1: Failing tests** for `information_coefficient(signal_vals, fwd_returns)` (Spearman rank corr; sign + significance) and `admit(candidate_residuals, members_residuals, ic)` returning a structured decision.

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3: Implement**:
  - `information_coefficient(signal: list[float], fwd_returns: list[float]) -> {ic, significant, sign}` — Spearman rank corr; `significant` via the standard `t = ic·√((n−2)/(1−ic²))` vs a pinned `IC_TSTAT_MIN=2.0`.
  - `admit(*, candidate_residuals: dict[date,float], members_residuals: dict[str,dict[date,float]], ic: dict, alpha_tstat: float) -> dict` — admit iff ALL: (1) `alpha_tstat >= ALPHA_TSTAT_MIN` (idiosyncratic significance, from Task 3); (2) `ic["significant"]` and sign matches the pre-registered direction (pass the expected sign in); (3) `marginal = pool.marginal_sharpe_contribution(candidate_residuals, members_residuals)` with `marginal["marginal"] > MARGINAL_SHARPE_MIN` (reuse the shipped fn, fed RESIDUALS); (4) `marginal["max_abs_corr"] is None or < MAX_CORR` (pinned `MAX_CORR=0.80`). Return `{admitted, reasons:{...}, marginal, ic}`. Reuse `MARGINAL_SHARPE_MIN` from `pool.py` (`_MIN_MARGINAL_SHARPE`) — import it, do not redefine.

- [ ] **Step 4: Run → pass. Step 5: pin-test `IC_TSTAT_MIN`, `MAX_CORR`. Step 6: Commit** — `git commit -m "feat(combination): IC + residual-fed admission decision"`

---

## Task 5: Combination-book persistence (coexist with the strategy pool)

**Files:** `combination_book.py` (async persistence helpers); integration `tests/integration/test_combination_book.py`.

- [ ] **Step 1:** Determine how the existing Signal Pool persists members (read `services/house_book.py` + `repo` for the pool table / journal kind). Combination members are tagged `structured_data.kind='signal_combination'` (+ `track`) so they coexist without a migration. If the pool uses a journal entry, mirror it; if a dedicated table, reuse it with the kind discriminator.

- [ ] **Step 2: Failing integration test** — admit a residual signal, persist it, read it back filtered by `kind='signal_combination'`; assert it does NOT appear in the existing strategy-level pool reads and vice-versa.

- [ ] **Step 3: Implement** `add_member` / `list_members(kind='signal_combination')` (+ the aggregate book's HRP weights via `pool._hrp_for` on residual series). **Step 4: Run → pass. Step 5: Commit** — `git commit -m "feat(combination): persist signal_combination members alongside strategy pool"`

---

## Task 6: Endpoint + wiring — `POST /combination/evaluate`

**Files:** Create `src/orchestrator/api/combination.py`; register in `main.py`; integration test.

- [ ] **Step 1:** Read how an existing router is registered in `main.py` and how `api/pool.py` is shaped (mirror its auth/deps).

- [ ] **Step 2: Failing integration test** (via `integration_client`) — `POST /combination/evaluate {iteration_id|backtest_run_id, strategy_code, expected_sign, track}` → 200 with `{neutralization, ic, admission}`; a residual-with-real-alpha-and-orthogonality admits; a disguised-beta candidate is rejected with `disguised_beta=true`.

- [ ] **Step 3: Implement** the handler: fetch the candidate's daily equity returns (`portfolio.daily_returns_from_trades` + `_calendar_fill`), `build_factors` over the window, `neutralize`, compute `ic` (signal vs fwd returns from the iteration's data), load existing combination members' residuals, `admit`. Return the composite. Add a playbook capability entry. **Do not** auto-promote — this is evaluation only (operator/curator path unchanged).

- [ ] **Step 4: Run → pass (both markers). Step 5: playbook + README note. Step 6: Commit** — `git commit -m "feat(combination): POST /combination/evaluate (factor-neutral admission)"`

---

## Phase 3 Done When
- A candidate's returns can be neutralized vs the 4 factors; pure beta → `DISGUISED_BETA`, real idiosyncratic α survives.
- A V60-failing trading signal with significant residual α + IC + orthogonality + low corr is admitted to the `signal_combination` book (HRP on residuals); disguised beta is rejected.
- All factor/admission constants pinned + pin-tested. V11/V60 + trading path provably unchanged (existing suite green). No migrations.

## Follow-ups / integration (out of scope here, note for later)
- Hook `/combination/evaluate` at the tick `pool_candidate` near-miss branch so the loop auto-evaluates V60-failers (currently a manual/endpoint call).
- The combined book's aggregate must pass walk-forward robustness before any live consideration (spec) — wire to `/walk-forward`.
- Factor series caching per window (perf).
