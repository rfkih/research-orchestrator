# Research Orchestrator — agent-facing front door

FastAPI service in Python that mediates between the quant-researcher agent (and the `/research` frontend dashboard) and the Blackheart research JVM. Replaces the legacy bash drivers (`research-tick.sh`, `walk-forward.sh`, `queue-strategy.sh`, `analyze-run.py`) with typed, idempotent, transactional HTTP.

## Topology

```
quant-researcher agent / cron / dashboard
            │   X-Orch-Token + X-Agent-Name [+ Idempotency-Key]
            ▼
research-orchestrator (uvicorn, single worker, 127.0.0.1:8082)
   │ asyncpg as blackheart_research                 │ httpx
   ▼                                                 ▼
PostgreSQL (V28+ schemas owned by trading JVM)      Research JVM (8081)
```

**Loopback only.** Shared-secret token is defense-in-depth, not the boundary.

## Repo layout

```
research-orchestrator/
├── src/orchestrator/
│   ├── api/           # FastAPI routers (queue, iterations, journal, tick, walk_forward, agent)
│   ├── services/      # tick, analyze, walk_forward, sweep, polling
│   ├── repo/          # asyncpg DML — queue, iterations, journal, hypothesis_audit, idempotency
│   ├── infra/         # db.py (asyncpg pool + JSONB codec), jvm.py (httpx client)
│   ├── errors.py      # OrchestratorError + NextAction envelope
│   └── settings.py    # profile (dev/prod), DSN, token, JVM auth
├── tests/             # pytest; run with PYTHONPATH=src pytest -q
├── README.md          # operator-facing
└── RUNBOOK.md         # incident response
```

## Hard contract — DO NOT change without user approval

| Boundary | Why frozen |
|---|---|
| **V11 statistical gate** (`MIN_TRADES_FOR_SIG`, PF 95% CI, PSR/DSR thresholds, +20bps slippage, walk-forward stability cutoffs) | Methodology contract. Loosening to fit a candidate is fraud. |
| **Auth shape** (`X-Orch-Token` header, `Settings.assert_prod_safe()`, dev-token sentinel) | Security boundary. |
| **Settings defaults** (DB role, port, host binding `127.0.0.1`, profile gating) | Loopback-only is deliberate ops choice. |
| **DB schema** | Migrations live in `blackheart/src/main/resources/db/flyway/`. Orchestrator is a DML-only client. New table = new Flyway migration in the trading-JVM repo. |
| **Idempotency contract** (`Idempotency-Key` header semantics, `idempotency_record` schema, replay shape, ~24h TTL) | Cron-driven callers depend on it. |
| **Append-only tables** — `research_iteration_log`, `research_journal`, `research_queue` rows. | Audit evidence. |

## Editable surface

- `src/orchestrator/` — handlers/services/repos/clients
- `tests/` — unit + integration
- `pyproject.toml` — analysis-lib deps (scipy, statsmodels) when an experiment needs them
- `README.md` / `RUNBOOK.md` — keep docs honest after API changes

## Conventions

### asyncpg + JSONB
The codec at `src/orchestrator/infra/db.py:_init_connection` registers `encoder=_json_dumps, decoder=json.loads` for jsonb. **Pass dicts directly to `$N` params for JSONB columns** — do NOT call `json.dumps()` first (it double-encodes; you'll store `"{\"k\":1}"` instead of `{"k":1}`).

Peer pattern: see `repo/queue_write.insert_queue` line 60 — `sweep_config` (a dict) is passed verbatim. The lone exception is the idempotency repo, which does `json.dumps(value, default=str)` AND uses an explicit `$3::jsonb` cast.

### Idempotency
Every state-changing POST honours `Idempotency-Key`. Pattern:

```python
idempotency_key = getattr(request.state, "idempotency_key", None)
store = request.app.state.idempotency
if idempotency_key:
    cached = await store.get(agent, f"<route>:{idempotency_key}")
    if cached is not None:
        return cached
# ... do the work ...
if idempotency_key:
    await store.put(agent, f"<route>:{idempotency_key}", response)
return response
```

Dev profile uses in-memory store; prod uses `PostgresIdempotencyStore` reading V28 `idempotency_record` (TTL ~24h).

### Error envelope
All non-2xx responses use `OrchestratorError` → `{error_code, message, retryable, hint, next_action, details}`. `error_code` is stable; `message` is prose and may change. Callers branch on `error_code` and `retryable`.

### Queue claim
`SELECT … FOR UPDATE SKIP LOCKED` on PENDING rows ordered `(priority ASC, created_time ASC)`. Concurrent `/tick` calls are safe — DB-level lock prevents double-claim. Stuck-row reaper resets RUNNING rows older than the poll cap back to PENDING (`recover_stuck`).

### Tick lifecycle (V11 + Tier 1)
1. Claim queue row.
2. Derive next param combo from `sweep_config`.
3. Resolve account_strategy (412 if missing).
3.5. **Insert `hypothesis_audit` row** (Tier 1, 2026-05-03) — records trial BEFORE backtest submission. `iteration_id` NULL until step 5 backfills. Drives DSR `n_trials` for selection-bias deflation. `count_cumulative_trials` filters `iteration_id IS NOT NULL` so infra failures don't inflate multiplicity.
4. Submit backtest, poll to terminal.
5. Analyze (`analyze_run` with `cumulative_trials`), log iteration, attach to queue, **backfill audit verdicts**.
6. Decide next state (PENDING for next iter, PARKED for SIGNIFICANT_EDGE awaiting walk-forward, COMPLETED).

### Re-discovery gate (Tier 1, 2026-05-03)
`POST /queue` rejects sweeps whose axis-set already produced a `decision_verdict='DISCARD'` audit row on the same strategy. Returns 409 `axis_previously_discarded` with `details.prior_iteration_id`. Bypass via `override_discard_gate: true` only with a documented journal entry. Prevents the autonomous loop from p-hacking by re-running the same dimensions under a fresh hypothesis line.

## Test discipline

```bash
PYTHONPATH=src pytest -q                # all tests
PYTHONPATH=src pytest tests/test_tier1_quant_grade.py -v
```

Pure-function tests run anywhere. DB-touching tests use `pytest-postgresql` and live behind `@pytest.mark.integration`.

When adding analytical functions (DSR, bootstrap CI, regime stratify): pin operator-controlled constants with explicit pin tests (e.g. `test_dsr_threshold_constant_is_pinned`) so accidental drift surfaces.

## Links

- Project root: `../blackheart/CLAUDE.md` (Java JVM, schema owner)
- Architecture context: `../blackheart/docs/agent-context/ARCHITECTURE.md`
- Migration history: `../blackheart/docs/agent-context/MIGRATIONS.md`
- Agent-facing playbook: `../blackheart/research/agent-playbooks/quant-researcher-workflow.md`
- Live API contract for callers: `GET /agent/playbook` (auth-free; the canonical source — keep it in sync with what handlers actually do)
