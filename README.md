# research-orchestrator

Agent-first FastAPI service that replaces `research-tick.sh` and friends.
Primary caller is the **quant-researcher Claude agent**; secondary callers
are the trading-JVM dashboard and ad-hoc operator curl. The orchestrator
runs long-lived under uvicorn — cron jobs collapse to thin curl callers
hitting `POST /tick`.

## Why this exists

The bash orchestrator (`research-tick.sh`, `analyze-run.py`, `queue-strategy.sh`)
hit a structural ceiling: no type safety, no transactions, no testability,
no observability beyond logs. None of those are bash failings — they're
inherent to using bash for a service this stateful.

This service owns:
- claiming the next research-queue row (`SKIP LOCKED`)
- submitting backtests to the research JVM and polling for completion
- writing iteration_log + journal entries
- running walk-forward + verdict logic (port of `analyze-run.py`)
- exposing read endpoints the agent and dashboard share

## Agent-first design

The agent is treated as a first-class user, not an afterthought:

- **Discoverable contract** — `GET /agent/playbook` returns the auth scheme,
  required headers, error shape, and recipe for cold-boot.
- **Stable error envelope** — every non-2xx is `{error_code, message,
  retryable, hint, next_action, details}`. Agents branch on `error_code`,
  never on prose.
- **Identity stamping** — `X-Agent-Name` is required on writes and lands
  in `created_by` columns so the audit trail names the agent that wrote
  the row.
- **Idempotency** — `Idempotency-Key` makes retries safe on `POST /tick`
  and other writes. The orchestrator dedupes server-side.
- **One-shot state** — `GET /agent/state` returns "what should I do next?"
  in a single response so the agent doesn't fan out to five endpoints just
  to orient.

## Quickstart (dev)

```bash
cd blackheart-research-orchestrator
uv venv && uv pip install -e ".[dev]"
cp .env.example .env       # then edit ORCH_AUTH_TOKEN, ORCH_DB_DSN
python -m orchestrator     # serves on 127.0.0.1:8082
```

In another shell:

```bash
# Public — no token needed
curl http://127.0.0.1:8082/healthz
curl http://127.0.0.1:8082/agent/playbook | jq

# Protected — needs the shared secret + an agent identity
curl -H "X-Orch-Token: $(grep ^ORCH_AUTH_TOKEN .env | cut -d= -f2)" \
     -H "X-Agent-Name: quant-researcher" \
     http://127.0.0.1:8082/agent/state
```

## Running tests

```bash
uv run pytest -q
```

Smoke tests don't need a DB or the JVM — they assert the auth middleware,
error envelope, and playbook shape. Phase 2/3 tests (read endpoints, tick)
will use `pytest-postgresql` for a per-test DB and `respx` to fake the JVM.

## Operating notes

- **Loopback only.** The service binds to `127.0.0.1` by default. Don't
  expose it. The shared-secret header is defense-in-depth, not perimeter.
- **Single worker.** `uvicorn` runs `workers=1`. The claim-loop (Phase 3)
  and idempotency registry assume one process. Scale by sharding the
  queue, not by adding workers.
- **Prod profile refuses dev secrets.** `Settings.assert_prod_safe` blocks
  startup if `ORCH_PROFILE=prod` and the auth token is the dev sentinel,
  or `ORCH_JVM_AUTH_MODE=dev_bypass`.
- **Migrations live in `blackheart-trading-engine/`.** This service is a DML-only client
  of the trading JVM's schema. Do not add Flyway here. Reads/writes go
  through the `blackheart_research` role (V14).

## Layout

```
src/orchestrator/
  __init__.py        — version
  __main__.py        — `python -m orchestrator`
  main.py            — app factory
  config.py          — Pydantic Settings
  auth.py            — X-Orch-Token + X-Agent-Name middleware
  errors.py          — ErrorEnvelope + exception handlers
  logging.py         — structlog JSON config
  api/
    health.py        — /healthz, /readyz
    agent.py         — /agent/playbook, /agent/state
    queue.py         — /queue, /queue/{queue_id}
    iterations.py    — /iterations, /iterations/{id}, /leaderboard
    journal.py       — /journal, /journal/{id}
    reviews.py       — /reviews/request, /reviews, /reviews/by-target,
                       /reviews/auto-run-checklist (server-side reviewer)
    tick.py          — /tick, /tick/drain (server-side runner)
    pagination.py    — opaque cursor encode/decode
    json_response.py — Decimal/UUID/datetime-aware encoder
    deps.py          — DB connection + agent-name dependencies
  repo/              — raw-SQL repositories (queue, iterations, journal)
  clients/
    jvm.py           — research JVM httpx client
  infra/
    db.py            — asyncpg pool
  tasks/
    lifespan.py      — startup/shutdown hooks
tests/
  conftest.py
  test_health.py     — smoke
```
