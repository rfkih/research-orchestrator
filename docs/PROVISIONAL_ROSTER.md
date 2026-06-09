# Provisional Trading-Roster Floor

**Status:** Orchestrator side SHIPPED (master `e17e9d3`, 2026-06-09). JVM side
WRITTEN but UNRECONCILED — see [§7 JVM reconciliation checklist](#7-jvm-reconciliation-checklist).
Feature is INERT until the JVM companion ships **and** the operator opts in.

---

## 1. Why

The graduation gate is deliberately strict — and after the 2026-06-09 metric
corrections (per-observation DSR + Bailey-LdP kurtosis fix) it is *stricter still*.
The honest consequence is that the live catalogue can sit below a usable number
of tradeable strategies: users have nothing to trade while research grinds.

The provisional floor keeps a **minimum of 5 active TRADING strategies** live by
auto-activating the best *near-misses* PROVISIONALLY until real graduates cover
the floor.

**This is NOT a loosening of the gate.** V11/V60 and the graduation contract are
untouched. Full qualifiers still graduate the normal way and are *not*
provisional. The floor is an **additive** product mechanism that only fills
*empty* roster slots, and only with candidates that have a *statistically real*
edge and make money. It never puts noise or capital-destroyers live.

## 2. Operator decisions (2026-06-09)

| Decision | Choice |
|---|---|
| Floor | **5** active trading strategies (hedging excluded) |
| Provisional treatment | **Marked `provisional=true`, FULL size** (not risk-capped) |
| Activation | **Fully automatic** when active count < floor |
| Ranking | **Balanced composite** = annualised return × overfit-quality |
| Demotion | Real graduates always reclaim slots; weakest provisional revoked first |

## 3. Architecture

```
 research JVM            research-orchestrator (8082)            trading JVM (8080)
 (backtests)             [ DECISION ENGINE — shipped ]           [ EXECUTION — to build ]
      │                          │                                        │
      │  iterations             │  GET /provisional-roster                │ @Scheduled tick
      │  (SIG_EDGE etc.)        │     ?active_count=N&floor=5&exclude=…    │  (disabled by default)
      └────────► shared Postgres ◄───────── ranks near-misses ───────────►│
                                 │  returns fills[] (best-first)          │  demote-then-fill:
                                 │                                        │   revoke weak provisional,
                                 │                                        │   activate fills as
                                 │                                        │   PROVISIONAL approvals
                                 │                                        │   (V102 gate bypassed)
```

**Source-of-truth split:** the JVM owns the live catalogue, so it owns "how many
are active" (`active_count`) and the activation itself. The orchestrator owns
"which near-miss is best to add." Neither can do the other's job — the
orchestrator is research-only and cannot put anything live.

## 4. Orchestrator side (shipped)

### Endpoint — `GET /provisional-roster`
Auth: `X-Orch-Token`. Read-only; never mutates the live book.

| Query param | Meaning |
|---|---|
| `active_count` (required) | currently-active TRADING approvals (JVM-supplied) |
| `floor` (default 5) | the minimum-roster target — pass it so both sides compute the same gap |
| `exclude` (repeatable) | strategy_codes already live, to not re-propose |

Response:
```json
{ "floor": 5, "active_count": 2, "gap": 3,
  "fills": [ { "strategy_code": "...", "symbol": "BTCUSDT", "interval_name": "1d",
              "backtest_run_id": "…", "ann_return_pct": 18.2, "dsr": 0.71,
              "overfit_quality": 0.71, "provisional_score": 12.9, "provisional": true } ],
  "next_actions": [ { "kind": "note", "hint": "…" } ] }
```

### Eligibility (a fill must satisfy ALL)
- **TRADING track** (hedging excluded — different gate)
- `statistical_verdict == SIGNIFICANT_EDGE` (a real edge, not noise)
- `annualised return > 0` (no money-losers, even provisionally)
- no walk-forward verdict of `OVERFIT` / `NO_EDGE`

### Ranking — balanced composite
```
overfit_quality = geometric_mean( dsr,
                                   fold_positive_pct/100  [if walk-forwarded],
                                   1.0 plateau / 0.3 cliff [if robustness known] )
composite_score = annualised_return_pct × overfit_quality      (0 if return ≤ 0)
```
Geometric mean rewards *corroborating* evidence; a single near-zero sub-score
drags quality down. Selection de-dupes by `strategy_code` (one strategy can't
take multiple slots) and fills only `max(0, floor − active_count)`.

### Files
- `services/provisional_roster.py` — pure decision engine (`overfit_quality`,
  `composite_score`, `is_eligible`, `rank_provisional_candidates`,
  `select_roster_fill`).
- `repo/provisional_roster.py` — candidate-pool query (trading-track SIG_EDGE,
  positive return, latest walk-forward verdict; track via
  `hypothesis_audit → research_queue.sweep_config`).
- `api/provisional_roster.py` — the endpoint.
- Tests: `tests/test_provisional_roster.py` (22).

## 5. JVM side (written, unreconciled)

Files (in `blackheart-trading-engine`, **uncommitted, uncompiled**):

| File | Role |
|---|---|
| `db/flyway/V166__symbol_strategy_approval_provisional.sql` | `provisional`, `provisional_score`, `provisional_reason` cols + partial index |
| `model/SymbolStrategyApproval.java` | entity fields for the provisional lane |
| `repository/SymbolStrategyApprovalRepository.java` | `countBy…ProvisionalFalse/True`, `findActiveProvisionalWeakestFirst` |
| `dto/research/ProvisionalRosterResponse.java` | Jackson mapping of the endpoint |
| `client/research/OrchestratorRosterClient.java` | `GET /provisional-roster` (fails soft — never blocks live) |
| `service/approval/ProvisionalRosterService.java` | `@Scheduled` demote-then-fill engine |
| `service/approval/SymbolStrategyApprovalService.java` | **edited**: provisional→real upgrade in `create()` |
| `application.properties` | config knobs (disabled by default) |

### Reconcile flow (`ProvisionalRosterService.reconcile`, each tick)
1. **Demote** — keep only the `floor − realTradingCount` *best* provisional rows;
   revoke the rest weakest-first (by `provisional_score`). Real graduates always
   reclaim slots. Returns the survivors so the exclude set is built from
   post-demotion state.
2. **Fill** — if still below floor, `fetchRoster(activeTrading, floor, excludeCodes)`,
   then for each fill activate a `SymbolStrategyApproval` with `provisional=true`,
   full size, **bypassing `gateService.evaluate()`** but citing the real
   `backtest_run_id` (so `chk_evidence_required_when_not_grandfathered` holds).
   Skips a pair that already has an active approval.

### Provisional → real upgrade (review finding #1)
A provisional row holds the active `(symbol, strategy_code)` slot via the partial
unique index. So `SymbolStrategyApprovalService.create()` now, **after** the gate
passes, revokes any active *provisional* row for that pair (`saveAndFlush` before
the insert) so a genuine graduation can take the slot instead of colliding.

### Config (all env-overridable; DISABLED by default)
```properties
app.roster.provisional-floor.enabled=false   # ROSTER_PROVISIONAL_FLOOR_ENABLED — opt-in (real capital!)
app.roster.provisional-floor.min=5           # ROSTER_PROVISIONAL_FLOOR_MIN
app.roster.provisional-floor.cron=0 0 * * * *# ROSTER_PROVISIONAL_FLOOR_CRON (hourly UTC)
app.roster.hedging-strategy-codes=EMA_BAND_BTC,EMA_BAND_RESP_BTC  # excluded from the trading count
```

## 6. Safety invariants

- **Opt-in.** `@ConditionalOnProperty(... havingValue="true")` — the bean does not
  even load unless explicitly enabled. It activates REAL-CAPITAL strategies.
- **No noise / no money-losers.** Orchestrator only returns SIGNIFICANT_EDGE,
  positive-return, non-OVERFIT trading candidates.
- **Real always wins.** Demotion runs before fill; a genuine graduate of any pair
  revokes its provisional twin (upgrade path).
- **Fail-soft.** The orchestrator call and the scheduled tick swallow errors —
  research-infra problems never block or break live trading.
- **Marked.** `provisional=true` + `provisional_reason` on every floor row; the
  UI must badge these so users know they are experimental.
- **Gate untouched.** V11/V60/DSR thresholds are not altered; this is a parallel
  lane, not a relaxation.

## 7. JVM reconciliation checklist

⚠️ The trading-engine checkout used to write this is **stale** (its CLAUDE.md head
says Flyway V103; `src` migrations stop at V146; prod is V165). The Java was
**never compiled or tested**. Before it ships:

- [ ] **Re-apply against the real prod codebase** (not the stale checkout).
- [ ] **Renumber the migration** to the next free version on prod HEAD (≥ V166).
- [ ] Confirm `symbol_strategy_approval` columns/constraints match prod
      (esp. `chk_evidence_required_when_not_grandfathered` — which evidence
      fields are required when `backtest_run_id` is set).
- [ ] Verify `BaseEntity` exposes `setCreatedBy` / `setUpdatedBy`.
- [ ] Verify the `@Qualifier("restTemplate")` bean name matches prod's
      `RestTemplateConfig` (there are ≥2 RestTemplate beans).
- [ ] Verify TRADING-vs-HEDGING counting: I used a config-driven hedging-code set
      (`app.roster.hedging-strategy-codes`) rather than an unmapped
      `StrategyKindResolver`. Confirm that's the right kind source and seed the
      real hedge codes.
- [ ] Decide whether provisional activate/revoke should fire
      `LeaderboardEventPublisher` (the real `create`/`revoke` paths do; the
      provisional path currently does not).
- [ ] Add JVM tests: demote-then-fill, the provisional→real upgrade (unique-index
      collision avoided), fail-soft on orchestrator-down, floor-mismatch guard.
- [ ] Compile + run the JVM suite green.
- [ ] Apply to dev first (operator rule), verify, then prod.

## 8. Operational runbook

**Enable:** set `ROSTER_PROVISIONAL_FLOOR_ENABLED=true` on the trading JVM (after
the JVM companion ships). The hourly tick reconciles automatically.
**Monitor:** rows where `provisional=true AND revoked_at IS NULL` in
`symbol_strategy_approval`; log lines `provisionally activated …` /
`demoted provisional …`.
**Disable:** set the flag back to `false`. In-flight provisional rows stay live
until a real graduate demotes them, or revoke them manually.

## 9. Related

- Metric corrections that made the gate stricter (the reason this exists):
  `services/analyze.py` (#1–#5, master `86651af`).
- Bear-coverage walk-forward: `services/walk_forward.py` (`window_covers_bear`).
- Graduation review escalation: `services/review.py` (`_HEDGING_BLOCKER_CHECKS`).
