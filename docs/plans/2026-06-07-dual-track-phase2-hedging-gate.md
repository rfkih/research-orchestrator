# Dual-Track Research — Phase 2 (Hedging Gate + Discovery) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Give the `hedging` track its own graduation objective — **beats buy-hold on a risk-adjusted, equity-level basis** — replacing the trading track's trade-level V11 + V60-economic gates for `track=hedging` ONLY. Then add the literature/forum **alpha-discovery** pipeline.

**Architecture:** When a tick's queue row is `track=hedging`, the verdict path computes a **buy-hold benchmark** (from `market_data` closes over the backtest window) and the strategy's **daily equity-return series** (from its `backtest_trade` rows), then applies a NEW equity-level gate: (a) a **bootstrap significance** test on the risk-adjusted improvement vs buy-hold (replaces V11 for hedging), and (b) a **decision gate** `beats_buy_hold_risk_adj` (replaces V60 for hedging). ROBUST walk-forward still required. **V11 + V60 are unchanged for `track=trading` — not loosened, just not applied to hedging; the hedging gate is a separate, additive, operator-authorized objective.**

**Tech Stack:** Python 3.11, asyncpg, numpy, pytest + pytest-postgresql (Phase-1 `tests/integration/` harness). Prefix every pytest run: `ORCH_AUTH_TOKEN=dev-sentinel-not-for-prod ORCH_DB_DSN="postgresql://x:y@127.0.0.1:5432/none" PYTHONPATH=src python -m pytest <args>`.

**Repo:** `blackheart-research-orchestrator`, worktree `C:\Project\.wt-dualtrack-orch`, branch `feat/dual-track-phase2` (off `feat/dual-track-phase1`). DML-only; **no Flyway migrations**.

**Spec:** `docs/specs/2026-06-06-dual-track-research-design.md` (Component 2; Component 5 = discovery). **Builds on Phase 1** (`docs/plans/2026-06-06-dual-track-phase1-plumbing.md`) — the `track` tag on queue rows is the selector.

**Reuse map (do NOT re-implement):**
- `services/portfolio.py:daily_returns_from_trades(trades)` — trades → sparse `{date: return}`.
- `services/pool.py:_calendar_fill`, `annualised_sharpe`, `realised_vol` — calendar grid + √252 Sharpe.
- `repo/trades.py:fetch_trades(conn, backtest_run_id)` — `backtest_trade` rows.
- `services/regime_analysis.py:_classify_from_market_data` — the `market_data` read pattern (`symbol`, `interval`, `start_time`, `close_price`).
- `services/analyze.py:bootstrap_pf_ci` / `_bootstrap_dsr` — bootstrap CI patterns; `ANNUALIZED_RETURN_PASS_THRESHOLD_PCT`.
- Phase-1 integration harness `tests/integration/conftest.py` (`db_conn`, `integration_client`).

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `src/orchestrator/repo/market_data.py` | Fetch underlying daily closes for [instrument, start, end] | Create |
| `src/orchestrator/services/hedging_gate.py` | Buy-hold benchmark metrics + equity-level significance + `beats_buy_hold_risk_adj` decision | Create |
| `src/orchestrator/api/constants.py` (or `services/hedging_gate.py`) | Pinned constants (`tol_cagr`, `θ_sharpe`, `θ_dd`, bootstrap reps, CI level) | Modify/Create |
| `src/orchestrator/services/tick.py` | Branch the economic/statistical gate on `track=hedging` | Modify |
| `tests/integration/test_hedging_gate.py` | Unit + integration tests | Create |
| `docs/plans/2026-06-07-dual-track-phase2-discovery.md` | Discovery-pipeline plan (Phase 2b — see §below) | Create later |

---

## Task 1: `market_data` buy-hold series repo

**Files:** Create `src/orchestrator/repo/market_data.py`; test in `tests/integration/test_hedging_gate.py`.

- [ ] **Step 1: Add `market_data` to the integration schema**

Append the `market_data` columns the query reads to `tests/integration/schema_phase1.sql` (copy types from Flyway V1 baseline): `symbol VARCHAR(10)`, `interval VARCHAR(5)`, `start_time TIMESTAMP`, `close_price NUMERIC(24,12)` (+ a `BIGSERIAL id PRIMARY KEY`). Minimal subset is fine.

- [ ] **Step 2: Write the failing test**

```python
# tests/integration/test_hedging_gate.py
import pytest
from datetime import datetime

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_fetch_daily_closes_returns_ordered_series(db_conn):
    from orchestrator.repo import market_data
    rows = [("BTCUSDT", "1d", datetime(2024,1,d), 100.0 + d) for d in range(1, 6)]
    await db_conn.executemany(
        "INSERT INTO market_data (symbol, interval, start_time, end_time, open_price, close_price, high_price, low_price, volume, trade_count, quote_asset_volume, taker_buy_base_volume, taker_buy_quote_volume) "
        "VALUES ($1,$2,$3,$3,0,$4,0,0,0,0,0,0,0)",
        rows,
    )
    series = await market_data.fetch_daily_closes(
        db_conn, symbol="BTCUSDT", interval="1d",
        start=datetime(2024,1,1), end=datetime(2024,1,6),
    )
    assert [c for _, c in series] == [101.0, 102.0, 103.0, 104.0, 105.0]
    assert series == sorted(series)  # ascending by date
```

> The integration schema's `market_data` must include the NOT NULL columns the INSERT lists; if Step 1's minimal subset omitted some, either add them or make the test INSERT only the columns present. Keep the test INSERT and the schema consistent.

- [ ] **Step 3: Run it, see it fail** — `... pytest tests/integration/test_hedging_gate.py::test_fetch_daily_closes_returns_ordered_series -v -m integration` → FAIL (module missing).

- [ ] **Step 4: Implement**

```python
"""Read underlying price series from market_data for the buy-hold benchmark."""
from __future__ import annotations
from datetime import datetime
import asyncpg


async def fetch_daily_closes(
    conn: asyncpg.Connection, *, symbol: str, interval: str,
    start: datetime, end: datetime,
) -> list[tuple[datetime, float]]:
    """Ascending [(start_time, close_price), ...] for the buy-hold leg.
    Half-open [start, end). Mirrors the regime_analysis market_data read."""
    rows = await conn.fetch(
        """
        SELECT start_time, close_price
        FROM market_data
        WHERE symbol = $1 AND interval = $2
          AND start_time >= $3 AND start_time < $4
        ORDER BY start_time ASC
        """,
        symbol, interval, start, end,
    )
    return [(r["start_time"], float(r["close_price"])) for r in rows]
```

- [ ] **Step 5: Run it, see it pass.** Then full suite `-m integration` + `-m "not integration"` green.

- [ ] **Step 6: Commit** — `git add src/orchestrator/repo/market_data.py tests/integration/test_hedging_gate.py tests/integration/schema_phase1.sql && git commit -m "feat(hedging): market_data daily-close repo for buy-hold benchmark"`

---

## Task 2: Buy-hold benchmark metrics (pure functions)

**Files:** Create `src/orchestrator/services/hedging_gate.py`; test in `tests/integration/test_hedging_gate.py` (these are PURE — no `db_conn` needed; can be plain `@pytest.mark.asyncio`-free unit tests, but keep them in the same file under a non-integration marker OR a separate `tests/test_hedging_gate_unit.py`).

> Put PURE-function tests in `tests/test_hedging_gate_unit.py` (no DB, runs in the default suite). Only DB-touching tests go under `tests/integration/`.

- [ ] **Step 1: Failing test — buy-hold metrics from a close series**

```python
# tests/test_hedging_gate_unit.py
from datetime import date
from orchestrator.services import hedging_gate


def test_buy_hold_metrics_basic():
    # 5 ascending closes → positive CAGR, finite Sharpe, small drawdown.
    closes = [(date(2024,1,d), 100.0 + d) for d in range(1, 6)]
    m = hedging_gate.buy_hold_metrics(closes)
    assert m["cagr_pct"] > 0
    assert m["max_drawdown_pct"] >= 0
    assert m["sharpe"] is not None


def test_buy_hold_metrics_drawdown():
    closes = [(date(2024,1,1),100.0),(date(2024,1,2),120.0),(date(2024,1,3),60.0)]
    m = hedging_gate.buy_hold_metrics(closes)
    # peak 120 → trough 60 = 50% drawdown
    assert round(m["max_drawdown_pct"], 1) == 50.0
```

- [ ] **Step 2: Run, fail.** `... PYTHONPATH=src python -m pytest tests/test_hedging_gate_unit.py -v`

- [ ] **Step 3: Implement `buy_hold_metrics`**

```python
"""Hedging-track gate: beats-buy-hold on a risk-adjusted, equity-level basis.

Separate, additive objective for track=hedging. Does NOT touch V11/V60, which
stay frozen for track=trading. Operator-authorized 2026-06-07.
"""
from __future__ import annotations
from datetime import date
from typing import Any
import math

# ── Pinned gate constants (operator-tunable; pin-tested below) ──────────
TOL_CAGR_PCT: float = 5.0       # hedge may give up at most this much CAGR vs buy-hold
THETA_SHARPE: float = 0.25      # a "material" Sharpe improvement
THETA_DD_PCT: float = 5.0       # a "material" maxDD reduction (percentage points)
BOOTSTRAP_REPS: int = 1000
CI_LEVEL: float = 0.95
RNG_SEED: int = 42
TRADING_DAYS: int = 252


def _returns(values: list[float]) -> list[float]:
    out = []
    for a, b in zip(values, values[1:]):
        if a:
            out.append(b / a - 1.0)
    return out


def _max_drawdown_pct(equity: list[float]) -> float:
    peak = equity[0] if equity else 0.0
    mdd = 0.0
    for v in equity:
        peak = max(peak, v)
        if peak > 0:
            mdd = max(mdd, (peak - v) / peak)
    return mdd * 100.0


def _sharpe(rets: list[float]) -> float | None:
    if len(rets) < 2:
        return None
    mu = sum(rets) / len(rets)
    var = sum((r - mu) ** 2 for r in rets) / (len(rets) - 1)
    sd = math.sqrt(var)
    if sd == 0:
        return None
    return (mu / sd) * math.sqrt(TRADING_DAYS)


def buy_hold_metrics(closes: list[tuple[date, float]]) -> dict[str, Any]:
    """CAGR%, annualised Sharpe, maxDD% of holding the underlying."""
    px = [c for _, c in closes]
    if len(px) < 2:
        return {"cagr_pct": None, "sharpe": None, "max_drawdown_pct": None}
    rets = _returns(px)
    years = max(len(px) / 365.0, 1e-9)
    cagr_pct = ((px[-1] / px[0]) ** (1.0 / years) - 1.0) * 100.0
    return {
        "cagr_pct": cagr_pct,
        "sharpe": _sharpe(rets),
        "max_drawdown_pct": _max_drawdown_pct(px),
    }
```

- [ ] **Step 4: Run, pass.**

- [ ] **Step 5: Commit** — `git commit -m "feat(hedging): buy-hold benchmark metrics (CAGR/Sharpe/maxDD)"`

---

## Task 3: The decision gate `beats_buy_hold_risk_adj` (pure)

**Files:** extend `src/orchestrator/services/hedging_gate.py`; test in `tests/test_hedging_gate_unit.py`.

- [ ] **Step 1: Failing tests for the deterministic boolean**

```python
def test_gate_passes_on_material_sharpe_gain_within_cagr_floor():
    v = hedging_gate.beats_buy_hold_risk_adj(
        strat={"cagr_pct": 30.0, "sharpe": 1.5, "max_drawdown_pct": 20.0},
        bench={"cagr_pct": 32.0, "sharpe": 1.0, "max_drawdown_pct": 40.0},
    )
    assert v["passed"] is True            # CAGR within tol (−2 ≥ −5) AND Sharpe +0.5 ≥ θ
    assert v["reason"]


def test_gate_fails_when_return_floor_breached():
    v = hedging_gate.beats_buy_hold_risk_adj(
        strat={"cagr_pct": 5.0, "sharpe": 3.0, "max_drawdown_pct": 5.0},
        bench={"cagr_pct": 30.0, "sharpe": 1.0, "max_drawdown_pct": 40.0},
    )
    assert v["passed"] is False           # gave up 25% CAGR > tol → fail regardless of Sharpe


def test_gate_fails_without_material_improvement():
    v = hedging_gate.beats_buy_hold_risk_adj(
        strat={"cagr_pct": 31.0, "sharpe": 1.05, "max_drawdown_pct": 39.0},
        bench={"cagr_pct": 32.0, "sharpe": 1.0, "max_drawdown_pct": 40.0},
    )
    assert v["passed"] is False           # +0.05 Sharpe < θ AND −1pt DD < θ_dd
```

- [ ] **Step 2: Run, fail.**

- [ ] **Step 3: Implement**

```python
def beats_buy_hold_risk_adj(*, strat: dict[str, Any], bench: dict[str, Any]) -> dict[str, Any]:
    """Deterministic: PASS iff CAGR floor holds AND a material risk improvement.
    floor:  strat.cagr >= bench.cagr - TOL_CAGR_PCT
    edge:   (strat.sharpe - bench.sharpe) >= THETA_SHARPE
            OR (bench.maxDD - strat.maxDD) >= THETA_DD_PCT
    """
    def _n(x): return None if x is None else float(x)
    s_cagr, b_cagr = _n(strat.get("cagr_pct")), _n(bench.get("cagr_pct"))
    s_sh, b_sh = _n(strat.get("sharpe")), _n(bench.get("sharpe"))
    s_dd, b_dd = _n(strat.get("max_drawdown_pct")), _n(bench.get("max_drawdown_pct"))
    if None in (s_cagr, b_cagr, s_sh, b_sh, s_dd, b_dd):
        return {"passed": False, "reason": "insufficient metrics for hedging gate"}
    floor_ok = s_cagr >= b_cagr - TOL_CAGR_PCT
    sharpe_gain = s_sh - b_sh
    dd_cut = b_dd - s_dd
    material = (sharpe_gain >= THETA_SHARPE) or (dd_cut >= THETA_DD_PCT)
    passed = bool(floor_ok and material)
    return {
        "passed": passed,
        "floor_ok": floor_ok,
        "sharpe_gain": round(sharpe_gain, 4),
        "dd_cut_pct": round(dd_cut, 4),
        "reason": (
            "beats buy-hold risk-adjusted" if passed
            else ("CAGR floor breached" if not floor_ok else "no material risk improvement")
        ),
    }
```

- [ ] **Step 4: Run, pass.**

- [ ] **Step 5: Pin-test the constants** (mirrors V11/V60 discipline):

```python
def test_hedging_constants_pinned():
    assert hedging_gate.TOL_CAGR_PCT == 5.0
    assert hedging_gate.THETA_SHARPE == 0.25
    assert hedging_gate.THETA_DD_PCT == 5.0
```

- [ ] **Step 6: Commit** — `git commit -m "feat(hedging): deterministic beats_buy_hold_risk_adj decision + pinned constants"`

---

## Task 4: Equity-level bootstrap significance (replaces V11 for hedging)

The decision gate says "is the improvement big enough"; this says "is it *real* (not noise)". Block-bootstrap the paired daily series; require the improvement's CI to clear 0.

**Files:** extend `services/hedging_gate.py`; test in `tests/test_hedging_gate_unit.py`.

- [ ] **Step 1: Failing test**

```python
def test_improvement_significant_when_strat_dominates():
    # strat returns strictly less volatile + higher-mean than bench → improvement CI > 0
    strat = [0.01]*60
    bench = [0.02, -0.02]*30
    v = hedging_gate.improvement_significant(strat_returns=strat, bench_returns=bench)
    assert v["sharpe_improvement_significant"] is True

def test_improvement_not_significant_when_identical():
    series = [0.01, -0.005]*40
    v = hedging_gate.improvement_significant(strat_returns=series, bench_returns=list(series))
    assert v["sharpe_improvement_significant"] is False
```

- [ ] **Step 2: Run, fail.**

- [ ] **Step 3: Implement** a deterministic (seeded) stationary block bootstrap of paired daily returns; for each resample compute `sharpe(strat) - sharpe(bench)`; the improvement is significant iff the `CI_LEVEL` lower bound > 0. Use `random.Random(RNG_SEED)` for determinism (no `Math.random`-style nondeterminism), block length ≈ `max(1, round(n**(1/3)))`. Return `{sharpe_improvement_significant: bool, ci_low: float, ci_high: float, n_obs: int}`. Mirror the resampling shape of `analyze.py:_bootstrap_dsr`.

> If `n_obs` is below a minimum (e.g. < 30 paired days), return `sharpe_improvement_significant=False` with `reason="insufficient_overlap"` — never pass on thin data.

- [ ] **Step 4: Run, pass.** Add a determinism test (same seed → identical `ci_low`).

- [ ] **Step 5: Commit** — `git commit -m "feat(hedging): bootstrap significance of risk-adj improvement vs buy-hold"`

---

## Task 5: Orchestrator entry point — assemble the hedging verdict

**Files:** extend `services/hedging_gate.py` with an async `evaluate(...)`; test in `tests/integration/test_hedging_gate.py`.

- [ ] **Step 1: Failing integration test** — seed `market_data` (buy-hold) + `backtest_trade` rows (strategy), call `hedging_gate.evaluate(conn, backtest_run_id=..., symbol=..., interval=..., start=..., end=...)`, assert it returns `{passed, decision, significance, benchmark, strategy}` and that a clearly-superior strategy passes while a buy-hold-equivalent one fails.

- [ ] **Step 2: Run, fail.**

- [ ] **Step 3: Implement `evaluate`** — orchestrates:
  1. `closes = await market_data.fetch_daily_closes(conn, symbol, interval=daily_interval, start, end)` (use the instrument's daily interval, e.g. `"1d"`; if the strategy interval isn't daily, still benchmark on daily closes).
  2. `bench = buy_hold_metrics(closes)`.
  3. `trades = await repo.trades.fetch_trades(conn, backtest_run_id)`; `strat_daily = pool._calendar_fill(portfolio.daily_returns_from_trades(trades), start.date(), end.date())`; derive `strat = {cagr_pct, sharpe, max_drawdown_pct}` from the strat daily equity curve (build equity by cumulative-compounding the daily returns).
  4. `bench_returns` = daily returns from `closes`; align both to the common date window.
  5. `decision = beats_buy_hold_risk_adj(strat, bench)`; `sig = improvement_significant(strat_returns, bench_returns)`.
  6. `passed = decision["passed"] and sig["sharpe_improvement_significant"]`.
  Return the composite dict.

- [ ] **Step 4: Run, pass.** Full suite green both markers.

- [ ] **Step 5: Commit** — `git commit -m "feat(hedging): evaluate() assembles benchmark + decision + significance"`

---

## Task 6: Wire into `tick.py` — select the gate by track

**Files:** `src/orchestrator/services/tick.py`; test in `tests/integration/test_hedging_gate.py`.

- [ ] **Step 1: Read the V60 decision block** (`tick.py` ~L900-990, where `decision_verdict == "PASS"` is set from SIGNIFICANT_EDGE + `annualized_geom >= ANNUALIZED_RETURN_PASS_THRESHOLD_PCT`). Identify where `track` is available (the claimed queue row's `sweep_config->>'track'`).

- [ ] **Step 2: Failing integration test** — drive a tick (or the decision helper) on a `track=hedging` queue row whose backtest beats buy-hold; assert it reaches the hedging PASS path (not the V60 ≥10% path); and a `track=trading` row still uses V60 unchanged.

- [ ] **Step 3: Implement the branch** — when the queue row's `track == "hedging"`:
  - SKIP the trade-level V11 `statistical_verdict` PASS requirement AND the V60 ≥10% economic check.
  - Instead call `hedging_gate.evaluate(...)` with the backtest window/instrument/run_id; set `decision_verdict = "PASS"` iff `evaluate(...).passed` (still requires walk-forward ROBUST downstream — unchanged).
  - Stash the hedging verdict dict on `metrics_snapshot.hedging_gate` for audit (like `portfolio_corr`).
  - For `track in (None, "trading")`: the existing V11+V60 path is **untouched**.
  - Keep the change surgical and clearly commented; do not alter the trading path's behavior.

- [ ] **Step 4: Run, pass.** Then the FULL suite (`-m "not integration"` 636+ green proving trading path unchanged; `-m integration` green).

- [ ] **Step 5: Update playbook** — note in the `/tick` + `drain_tick_queue` purpose that `track=hedging` rows graduate on the equity-level beats-buy-hold gate, not V11/V60.

- [ ] **Step 6: Commit** — `git commit -m "feat(hedging): tick selects equity-level beats-buy-hold gate for track=hedging"`

---

## Phase 2a Done When
- `track=hedging` candidates graduate on beats-buy-hold (equity-level significance + decision + ROBUST WF); `track=trading` still on V11+V60, **provably unchanged** (existing suite green).
- All hedging-gate constants pinned + pin-tested.
- Hedging verdict stashed on `metrics_snapshot.hedging_gate` for audit.

## Phase 2b — Alpha-discovery pipeline (separate plan)
The discovery pipeline (Component 5: lit/forum web fan-out → synthesis → math formulation → pre-registered HYPOTHESIS + queue spec) is implemented as a **Workflow script** (multi-agent, needs `WebSearch`/`WebFetch`) plus a thin orchestrator path to accept a pre-registered hypothesis (reuse `POST /journal` HYPOTHESIS + `POST /queue` with declared `n_trials`). It is a different artifact class than the gate (orchestration, not TDD'd Python), so it gets its own plan `docs/plans/2026-06-07-dual-track-phase2-discovery.md` and is built/iterated after the gate lands. Confirmatory discipline (pre-register before backtest) and the web-tools-only-in-discovery boundary are hard requirements (spec §8).
