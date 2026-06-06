# Dual-Track Research — Phase 1 (Dual-Loop Plumbing) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let two `quant-researcher` loops (a `trading` track and a `hedging` track) run concurrently against the same orchestrator + DB without poisoning each other's session state, each launched from its own CLI and checkpointing to the operator at decision points.

**Architecture:** Carry a `track` string in existing JSONB (`research_queue.sweep_config->>'track'`, `research_journal.structured_data->>'track'`). Thread an optional `track` arg through the collision-prone reads (`get_state_digest` + helpers) and the queue claim (`/tick`, `/tick/drain`). Absent `track` ⇒ today's global behavior (backward compatible). A `/research-track <trading|hedging>` slash command boots the scoped researcher; decision points write a track-tagged `SESSION_CHECKPOINT` and return to the CLI for operator input.

**Tech Stack:** Python 3.11, FastAPI, asyncpg, pytest + pytest-postgresql. Tests run `PYTHONPATH=src pytest -q` from the worktree root. DB-touching tests use `@pytest.mark.integration`.

**⚠️ Test-run env (confirmed Task 0):** `src/orchestrator/main.py` builds the app at import time, so `Settings()` requires env vars or collection fails. Prefix EVERY pytest command with:
`ORCH_AUTH_TOKEN=dev-sentinel-not-for-prod ORCH_DB_DSN="postgresql://x:y@127.0.0.1:5432/none" PYTHONPATH=src pytest ...`
Integration tests connect via `postgresql_noproc` to the dev server on `127.0.0.1:5432` (creds `postgres`/`admin`; no pg binaries on PATH so `postgresql_proc` is NOT available on this host).

**Repo:** All code paths below are relative to `blackheart-research-orchestrator/`. This repo is a DML-only client — **no Flyway migrations** (schema is owned by `blackheart-trading-engine`). The `track` discriminator lives in JSONB precisely to avoid a migration.

**Spec:** `docs/specs/2026-06-06-dual-track-research-design.md` (Components 1, 3, 4).

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `src/orchestrator/repo/agent_state.py` | Per-track filtering in the state-digest helpers | Modify |
| `src/orchestrator/api/agent.py` | Expose `?track=` on `GET /agent/state` | Modify |
| `src/orchestrator/repo/queue_write.py` | Stamp `track` into `sweep_config` on insert | Modify |
| `src/orchestrator/repo/queue.py` | Add `track` predicate to the claim query | Modify |
| `src/orchestrator/api/queue.py` | Accept/forward `track`; scope re-discovery gate | Modify |
| `src/orchestrator/services/tick.py` | Thread `track` into `run_tick`'s claim | Modify |
| `src/orchestrator/services/tick_drain.py` | Thread `track` through `drain_ticks` | Modify |
| `src/orchestrator/api/tick.py` | Accept `track` on `/tick` and `/tick/drain` | Modify |
| `tests/test_track_isolation.py` | Unit + integration tests for all of the above | Create |
| `.claude/commands/research-track.md` | Per-CLI launch command (in the **monorepo root** `.claude/`) | Create |

> **Note on the command file location:** the `.claude/commands/` dir is at the **monorepo root** (`C:\Project\.claude\commands\`), not inside the orchestrator repo, because the slash command is invoked from the top-level CLI session. Confirm with `ls C:/Project/.claude/commands` before writing.

---

## Task 0: pytest-postgresql integration harness (real DB)

The repo declares `pytest-postgresql>=6.1` + `pytest-asyncio` (`asyncio_mode="auto"`) and CLAUDE.md says "DB-touching tests use pytest-postgresql behind `@pytest.mark.integration`" — but no test actually does. The operator chose **real** integration tests, so this task stands up the live-Postgres fixture + a minimal schema bootstrap for the three tables Phase 1 touches. All later `*_track` tests are `@pytest.mark.integration` and depend on the `db_conn` fixture created here.

**Files:**
- Create: `tests/integration/conftest.py`
- Create: `tests/integration/schema_phase1.sql`
- Create: `tests/integration/test_harness_smoke.py`

- [ ] **Step 1: Write the minimal schema bootstrap**

Create `tests/integration/schema_phase1.sql` with the EXACT column subset Phase 1 queries read, copied from the Flyway V1 baseline (`blackheart-trading-engine/.../V1__baseline.sql`). Deliberately minimal, test-local (documented drift risk: keep the used columns in sync with V1):

```sql
-- Minimal subset of the Flyway V1 baseline for Phase 1 track-isolation tests.
-- Source of truth: blackheart-trading-engine/src/main/resources/db/flyway/V1__baseline.sql
CREATE TABLE IF NOT EXISTS research_journal (
    journal_id      UUID         NOT NULL DEFAULT gen_random_uuid(),
    entry_type      VARCHAR(40)  NOT NULL,
    strategy_code   VARCHAR(60),
    title           VARCHAR(300) NOT NULL,
    content         TEXT         NOT NULL,
    structured_data JSONB        NOT NULL DEFAULT '{}',
    status          VARCHAR(20)  NOT NULL DEFAULT 'ACTIVE',
    created_time    TIMESTAMP    NOT NULL DEFAULT NOW(),
    CONSTRAINT pk_research_journal PRIMARY KEY (journal_id)
);

CREATE TABLE IF NOT EXISTS research_queue (
    queue_id        UUID        NOT NULL DEFAULT gen_random_uuid(),
    priority        INTEGER     NOT NULL DEFAULT 100,
    strategy_code   VARCHAR(60) NOT NULL,
    interval_name   VARCHAR(20) NOT NULL,
    instrument      VARCHAR(30) NOT NULL DEFAULT 'BTCUSDT',
    sweep_config    JSONB       NOT NULL,
    hypothesis      TEXT,
    status          VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    iteration_number INTEGER    NOT NULL DEFAULT 0,
    iter_budget     INTEGER     NOT NULL DEFAULT 5,
    created_time    TIMESTAMP   NOT NULL DEFAULT NOW(),
    CONSTRAINT pk_research_queue PRIMARY KEY (queue_id)
);

CREATE TABLE IF NOT EXISTS research_iteration_log (
    iteration_id        UUID        NOT NULL DEFAULT gen_random_uuid(),
    strategy_code       VARCHAR(60) NOT NULL,
    iteration_number    INTEGER     NOT NULL,
    params_snapshot     JSONB       NOT NULL DEFAULT '{}',
    metrics_snapshot    JSONB       NOT NULL DEFAULT '{}',
    verdict             VARCHAR(20) NOT NULL,
    statistical_verdict VARCHAR(40),
    created_time        TIMESTAMP   NOT NULL DEFAULT NOW(),
    CONSTRAINT pk_research_iteration_log PRIMARY KEY (iteration_id),
    CONSTRAINT uq_research_iteration_strategy_n UNIQUE (strategy_code, iteration_number)
);
```

> `gen_random_uuid()` needs `pgcrypto` on older PG; on PG13+ it is built-in. If the smoke test errors on it, prepend `CREATE EXTENSION IF NOT EXISTS pgcrypto;` to this file.

- [ ] **Step 2: Write the integration conftest with a live asyncpg `db_conn` fixture**

Create `tests/integration/conftest.py`. pytest-postgresql provides the server + applies nothing; we open a parallel `asyncpg` connection (repo code is all asyncpg) and register the project's JSONB codec (mirror `orchestrator/infra/db.py:_init_connection` so dicts round-trip).

```python
"""Live-Postgres fixtures for Phase 1 track-isolation integration tests.

Default: pytest-postgresql spawns an isolated server (postgresql_proc) — needs
initdb/pg_ctl on PATH. If unavailable on this host, switch the marked line to
postgresql_noproc against the dev server on 127.0.0.1:5432 (empty dev DB).
"""
from __future__ import annotations

import json
from pathlib import Path

import asyncpg
import pytest_asyncio
from pytest_postgresql import factories

# NOTE (no pg binaries on PATH): replace with
#   postgresql_proc = factories.postgresql_noproc(
#       host="127.0.0.1", port=5432, user="postgres", password="postgres")
postgresql_proc = factories.postgresql_proc()
postgresql_db = factories.postgresql("postgresql_proc", dbname="orch_phase1_test")

_SCHEMA = (Path(__file__).parent / "schema_phase1.sql").read_text(encoding="utf-8")


@pytest_asyncio.fixture
async def db_conn(postgresql_db):
    p = postgresql_db.info  # psycopg ConnectionInfo
    conn = await asyncpg.connect(
        host=p.host, port=p.port, user=p.user,
        password=p.password, database=p.dbname,
    )
    await conn.set_type_codec(
        "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog",
    )
    await conn.execute(_SCHEMA)
    try:
        yield conn
    finally:
        await conn.close()
```

- [ ] **Step 3: Write the harness smoke test**

```python
# tests/integration/test_harness_smoke.py
import pytest

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_db_conn_roundtrips_jsonb(db_conn):
    await db_conn.execute(
        """
        INSERT INTO research_journal (entry_type, title, content, structured_data)
        VALUES ('HYPOTHESIS', 't', 'c', $1)
        """,
        {"track": "trading"},  # dict → jsonb via the registered codec
    )
    row = await db_conn.fetchrow(
        "SELECT structured_data FROM research_journal WHERE title = 't'"
    )
    assert row["structured_data"] == {"track": "trading"}
```

- [ ] **Step 4: Run the smoke test**

Run: `PYTHONPATH=src pytest tests/integration/test_harness_smoke.py -v -m integration`
Expected: PASS. If it errors "no initdb/pg_ctl", apply the `postgresql_noproc` NOTE in Step 2 and re-run. If `gen_random_uuid` is missing, apply the `pgcrypto` note in Step 1.

- [ ] **Step 5: Confirm the default suite still excludes integration**

Run: `PYTHONPATH=src pytest -q -m "not integration"`
Expected: the full existing suite passes; the new integration test is deselected (default/CI runs don't require Postgres).

- [ ] **Step 6: Commit**

```bash
git add tests/integration/conftest.py tests/integration/schema_phase1.sql tests/integration/test_harness_smoke.py
git commit -m "test(track): pytest-postgresql integration harness + Phase1 schema bootstrap"
```

> **For all subsequent tasks:** put the `*_track` tests in `tests/integration/test_track_isolation.py` (NOT `tests/test_track_isolation.py`), mark the module `pytestmark = pytest.mark.integration`, and use the `db_conn` fixture from this task. The `INSERT`/`SELECT` test bodies in Tasks 1–6 are correct against this real `db_conn`. For the API tests in Tasks 3 & 5, add an `integration_client` fixture to this conftest — a `TestClient` whose `app.state.db` wraps a real asyncpg pool to this test DB (follow `tests/conftest.py:client` but do NOT override `get_db_conn` to None). Run each task's tests with `-m integration`.

---

## Task 1: Track-scoped helpers in `agent_state.py` (lockout + run-summary)

The two most dangerous cross-contaminations: a hedging `ARCHETYPE_EXHAUSTION` must not lock out the trading loop (`_lockout_state`), and each track must resume its own checkpoint (`_last_run_summary`). Both read `research_journal`; we add an optional `track` filter on `structured_data->>'track'`. `track=None` keeps the global query (backward compatible).

**Files:**
- Modify: `src/orchestrator/repo/agent_state.py`
- Test: `tests/test_track_isolation.py`

- [ ] **Step 1: Write the failing test (lockout is per-track)**

```python
# tests/test_track_isolation.py
import pytest
from datetime import datetime, timezone

from orchestrator.repo import agent_state


pytestmark = pytest.mark.integration


async def _insert_run_summary(conn, *, title, track, content="x"):
    await conn.execute(
        """
        INSERT INTO research_journal
            (entry_type, strategy_code, title, content, structured_data, status, created_time)
        VALUES ('RUN_SUMMARY', 'TRK_TEST', $1, $2, $3, 'ACTIVE', NOW())
        """,
        title, content, {"track": track} if track else {},
    )


@pytest.mark.asyncio
async def test_lockout_is_scoped_to_track(db_conn):
    # A hedging-track exhaustion must NOT lock out the trading track.
    await _insert_run_summary(
        db_conn, title="ARCHETYPE_EXHAUSTION_HEDGE", track="hedging",
    )

    trading = await agent_state._lockout_state(db_conn, track="trading")
    hedging = await agent_state._lockout_state(db_conn, track="hedging")

    assert trading["in_lockout"] is False, "trading must not see hedging's lockout"
    assert hedging["in_lockout"] is True, "hedging must see its own lockout"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_track_isolation.py::test_lockout_is_scoped_to_track -v`
Expected: FAIL — `_lockout_state()` got an unexpected keyword argument `track`.

- [ ] **Step 3: Add the `track` filter to `_lockout_state`**

Modify `_lockout_state` to accept `track: str | None = None` and add a track predicate to the `terminal` CTE's `WHERE`. The filter must treat NULL track as "match all" so existing callers are unchanged:

```python
async def _lockout_state(conn: asyncpg.Connection, *, track: str | None = None) -> dict[str, Any]:
    row = await conn.fetchrow(
        """
        WITH terminal AS (
            SELECT title, created_time,
                   CASE
                       WHEN title LIKE 'ARCHETYPE_EXHAUSTION_%' THEN INTERVAL '24 hours'
                       WHEN title LIKE 'OPERATOR_ESCALATION_%'   THEN INTERVAL '12 hours'
                   END AS window_iv
            FROM research_journal
            WHERE entry_type = 'RUN_SUMMARY'
              AND (title LIKE 'ARCHETYPE_EXHAUSTION_%'
                   OR title LIKE 'OPERATOR_ESCALATION_%')
              AND ($1::text IS NULL OR structured_data->>'track' = $1)
            ORDER BY created_time DESC
            LIMIT 1
        )
        SELECT t.title,
               (t.created_time + t.window_iv) AS expires_at,
               (NOW() < t.created_time + t.window_iv) AS within_window,
               EXISTS(
                   SELECT 1 FROM research_journal h
                   WHERE h.entry_type = 'HYPOTHESIS' AND h.status = 'ACTIVE'
                     AND h.created_time > t.created_time
                     AND ($1::text IS NULL OR h.structured_data->>'track' = $1)
               ) AS bypass_available
        FROM terminal t
        WHERE t.window_iv IS NOT NULL
        """,
        track,
    )
    if row is None:
        return {"in_lockout": False, "terminal_title": None,
                "lockout_expires_at": None, "bypass_available": False}
    bypass = bool(row["bypass_available"])
    within = bool(row["within_window"])
    expires_at = row["expires_at"]
    return {
        "in_lockout": bool(within and not bypass),
        "terminal_title": row["title"],
        "lockout_expires_at": expires_at.isoformat() if expires_at else None,
        "bypass_available": bypass,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_track_isolation.py::test_lockout_is_scoped_to_track -v`
Expected: PASS

- [ ] **Step 5: Write the failing test (last-run-summary is per-track)**

```python
@pytest.mark.asyncio
async def test_last_run_summary_is_scoped_to_track(db_conn):
    await _insert_run_summary(db_conn, title="SESSION_CHECKPOINT_TRADING", track="trading", content="t")
    await _insert_run_summary(db_conn, title="SESSION_CHECKPOINT_HEDGING", track="hedging", content="h")

    trading = await agent_state._last_run_summary(db_conn, track="trading")
    hedging = await agent_state._last_run_summary(db_conn, track="hedging")

    assert trading["title"] == "SESSION_CHECKPOINT_TRADING"
    assert hedging["title"] == "SESSION_CHECKPOINT_HEDGING"
```

- [ ] **Step 6: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_track_isolation.py::test_last_run_summary_is_scoped_to_track -v`
Expected: FAIL — unexpected keyword argument `track`.

- [ ] **Step 7: Add the `track` filter to `_last_run_summary`**

```python
async def _last_run_summary(conn: asyncpg.Connection, *, track: str | None = None) -> dict[str, Any] | None:
    row = await conn.fetchrow(
        """
        SELECT journal_id, strategy_code, title, content,
               structured_data, created_time
        FROM research_journal
        WHERE entry_type = 'RUN_SUMMARY'
          AND ($1::text IS NULL OR structured_data->>'track' = $1)
        ORDER BY created_time DESC
        LIMIT 1
        """,
        track,
    )
    if row is None:
        return None
    return {
        "journal_id": str(row["journal_id"]),
        "strategy_code": row["strategy_code"],
        "title": row["title"],
        "content": row["content"],
        "structured_data": row["structured_data"],
        "created_time": row["created_time"],
    }
```

- [ ] **Step 8: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_track_isolation.py::test_last_run_summary_is_scoped_to_track -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add tests/test_track_isolation.py src/orchestrator/repo/agent_state.py
git commit -m "feat(track): per-track lockout + run-summary in agent_state"
```

---

## Task 2: Track-scope the remaining digest helpers + `get_state_digest`

Apply the same `($N::text IS NULL OR structured_data->>'track' = $N)` filter to `_active_hypotheses`, `_last_null_screen_per_surface`, `_pending_specialist_reviews`, `_recent_specialist_verdicts`, then thread `track` through `get_state_digest`. (`_queue_counts` and `_last_iterations` are addressed in Task 4 via the queue's `sweep_config` tag — leave them here.)

**Files:**
- Modify: `src/orchestrator/repo/agent_state.py`
- Test: `tests/test_track_isolation.py`

- [ ] **Step 1: Write the failing test (digest threads track to all journal helpers)**

```python
@pytest.mark.asyncio
async def test_get_state_digest_threads_track(db_conn):
    # Active hypotheses are per-track.
    await db_conn.execute(
        """
        INSERT INTO research_journal
            (entry_type, strategy_code, title, content, structured_data, status, created_time)
        VALUES ('HYPOTHESIS', 'TRK_TEST', 'h-trading', 'x', '{"track":"trading"}', 'ACTIVE', NOW()),
               ('HYPOTHESIS', 'TRK_TEST', 'h-hedging', 'x', '{"track":"hedging"}', 'ACTIVE', NOW())
        """
    )
    digest = await agent_state.get_state_digest(db_conn, track="trading")
    titles = {h["title"] for h in digest["active_hypotheses"]}
    assert "h-trading" in titles
    assert "h-hedging" not in titles
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_track_isolation.py::test_get_state_digest_threads_track -v`
Expected: FAIL — `get_state_digest()` got an unexpected keyword argument `track`.

- [ ] **Step 3: Add `track` filter to `_active_hypotheses`**

```python
async def _active_hypotheses(conn: asyncpg.Connection, *, track: str | None = None) -> list[dict[str, Any]]:
    rows = await conn.fetch(
        """
        SELECT journal_id, strategy_code, title, created_time
        FROM research_journal
        WHERE entry_type = 'HYPOTHESIS' AND status = 'ACTIVE'
          AND ($1::text IS NULL OR structured_data->>'track' = $1)
        ORDER BY created_time DESC
        """,
        track,
    )
    return [
        {
            "journal_id": str(r["journal_id"]),
            "strategy_code": r["strategy_code"],
            "title": r["title"],
            "created_time": r["created_time"],
        }
        for r in rows
    ]
```

- [ ] **Step 4: Add `track` filter to `_last_null_screen_per_surface`**

Add `, *, track: str | None = None` to the signature and insert the predicate into the `WHERE`:

```python
        WHERE entry_type = 'NULL_SCREEN_RESULT'
          AND ($1::text IS NULL OR structured_data->>'track' = $1)
```

Pass `track` as the single `$1` param to `conn.fetch`. (The `DISTINCT ON` ordering is unchanged.)

- [ ] **Step 5: Add `track` filter to `_pending_specialist_reviews` and `_recent_specialist_verdicts`**

Both already filter `structured_data->>'kind'`. Add the track predicate as an additional `AND` and append `track` as the trailing positional param. For `_pending_specialist_reviews` (currently `$1` = hard_cap → becomes `$2`):

```python
         WHERE entry_type = 'IDEA_BACKLOG'
           AND status = 'ACTIVE'
           AND structured_data->>'kind' = 'specialist_review_request'
           AND ($2::text IS NULL OR structured_data->>'track' = $2)
         ORDER BY created_time ASC
         LIMIT $1
```
called with `conn.fetch(sql, hard_cap, track)`. Apply the analogous change to `_recent_specialist_verdicts` (its params are `window_hours, hard_cap` → add `track` as `$3`).

- [ ] **Step 6: Thread `track` through `get_state_digest`**

Add `track: str | None = None` to the `get_state_digest` signature (after `agent_name`), and pass it to each scoped helper call:

```python
async def get_state_digest(
    conn: asyncpg.Connection,
    *,
    sig_edge_window_days: int = 7,
    last_n_iterations: int = 5,
    agent_name: str | None = None,
    track: str | None = None,
    ml_training_cap: int = 4,
    ml_training_window_hours: int = 24,
) -> dict[str, Any]:
    queue_counts = await _queue_counts(conn, track=track)            # Task 4 adds the param
    last_iters = await _last_iterations(conn, last_n_iterations)     # NOT track-scoped — see note
    sig_edge_ids = await _recent_sig_edge_ids(conn, sig_edge_window_days)  # NOT track-scoped
    hypotheses = await _active_hypotheses(conn, track=track)
    run_summary = await _last_run_summary(conn, track=track)
    null_screens = await _last_null_screen_per_surface(conn, track=track)
    pending_specialist_reviews = await _pending_specialist_reviews(conn, track=track)
    recent_specialist_verdicts = await _recent_specialist_verdicts(conn, track=track)
    lockout_state = await _lockout_state(conn, track=track)
    # ... ml_budget / pending_ml_runs unchanged (keyed by agent_name) ...
```

> **Sequencing + design note:** `_queue_counts` gains a `track` param in Task 4 (it reads `research_queue.sweep_config`, which carries the tag). For THIS task call it without `track` and add the kwarg in Task 4 — no TODO marker needed since the call site is updated in the very next task. `_last_iterations` and `_recent_sig_edge_ids` are **intentionally left global**: `research_iteration_log` has **no `queue_id`/track column** (confirmed against Flyway V1), so an iteration cannot be linked to a track. These two fields are informational digest content; cross-track bleed there is cosmetic and cannot cause a wrong lockout, mis-resume, or double-claim. Document this with a one-line code comment above each call: `# global by design: iteration_log has no track linkage (see Phase 1 plan Task 4)`.

- [ ] **Step 7: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_track_isolation.py::test_get_state_digest_threads_track -v`
Expected: PASS

- [ ] **Step 8: Backward-compat test — no track returns global**

```python
@pytest.mark.asyncio
async def test_digest_no_track_is_global(db_conn):
    await db_conn.execute(
        """
        INSERT INTO research_journal
            (entry_type, strategy_code, title, content, structured_data, status, created_time)
        VALUES ('HYPOTHESIS', 'TRK_TEST', 'h-untagged', 'x', '{}', 'ACTIVE', NOW()),
               ('HYPOTHESIS', 'TRK_TEST', 'h-tagged',   'x', '{"track":"trading"}', 'ACTIVE', NOW())
        """
    )
    digest = await agent_state.get_state_digest(db_conn)  # no track → global
    titles = {h["title"] for h in digest["active_hypotheses"]}
    assert {"h-untagged", "h-tagged"} <= titles
```

- [ ] **Step 9: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_track_isolation.py::test_digest_no_track_is_global -v`
Expected: PASS

- [ ] **Step 10: Commit**

```bash
git add tests/test_track_isolation.py src/orchestrator/repo/agent_state.py
git commit -m "feat(track): thread track through get_state_digest journal helpers"
```

---

## Task 3: Expose `?track=` on `GET /agent/state`

**Files:**
- Modify: `src/orchestrator/api/agent.py` (the `GET /agent/state` handler — search for `get_state_digest`)
- Test: `tests/test_track_isolation.py`

- [ ] **Step 1: Locate the handler**

Run: `PYTHONPATH=src python -c "import re,sys; print('grep agent/state handler')"` then open `src/orchestrator/api/agent.py` and find the `@router.get("/state")` handler that calls `agent_state_repo.get_state_digest(...)`.

- [ ] **Step 2: Write the failing test (API forwards track)**

```python
@pytest.mark.asyncio
async def test_agent_state_endpoint_accepts_track(async_client):
    # async_client is the repo's httpx test client fixture against the app.
    resp = await async_client.get("/agent/state?track=trading",
                                  headers={"X-Orch-Token": "dev-token"})
    assert resp.status_code == 200
    # The digest shape is unchanged; we only assert the param is accepted.
    assert "queue_counts" in resp.json()
```

> If the repo has no `async_client` fixture, reuse the pattern from `tests/test_agent_state_phase2.py` (it already exercises `GET /agent/state`); copy its client/fixture setup.

- [ ] **Step 3: Run test to verify it fails (or 422)**

Run: `PYTHONPATH=src pytest tests/test_track_isolation.py::test_agent_state_endpoint_accepts_track -v`
Expected: FAIL — the handler ignores `track` (param not wired) or the fixture is missing.

- [ ] **Step 4: Add the query param and forward it**

In the handler signature add `track: str | None = None` (FastAPI reads it from the query string automatically), and pass it through:

```python
@router.get("/state", response_model=...)
async def agent_state(request: Request, track: str | None = None) -> ...:
    db = request.app.state.db
    agent = getattr(request.state, "agent_name", None)
    async with db.acquire() as conn:
        digest = await agent_state_repo.get_state_digest(
            conn, agent_name=agent, track=track,
        )
    # ... existing db_ok/jvm_ok envelope unchanged ...
```

Match the existing handler's exact envelope construction (db_ok/jvm_ok). Only the `track=track` kwarg is new.

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_track_isolation.py::test_agent_state_endpoint_accepts_track -v`
Expected: PASS

- [ ] **Step 6: Update the playbook capability text**

In `api/agent.py`, the `agent_state` `PlaybookCapability.purpose` string: append a sentence so the researcher knows the param exists:

```
" Pass ?track=trading|hedging to scope lockout/run-summary/hypotheses/"
"null-screen/specialist rows to one research track (omit for global)."
```

- [ ] **Step 7: Commit**

```bash
git add tests/test_track_isolation.py src/orchestrator/api/agent.py
git commit -m "feat(track): GET /agent/state accepts ?track= and forwards it"
```

---

## Task 4: Stamp `track` on queue insert + scope the claim and queue-count reads

This is the loop side: a tick on the trading track must only claim trading rows. The claim is `FOR UPDATE SKIP LOCKED` (already concurrency-safe); we add a `WHERE` predicate on `sweep_config->>'track'`.

**Files:**
- Modify: `src/orchestrator/repo/queue_write.py` (`insert_queue`)
- Modify: `src/orchestrator/repo/queue.py` (the claim query + count helpers)
- Modify: `src/orchestrator/repo/agent_state.py` (`_queue_counts`, `_last_iterations` — remove Task 2's TODO markers)
- Test: `tests/test_track_isolation.py`

- [ ] **Step 1: Write the failing test (claim is track-scoped)**

```python
@pytest.mark.asyncio
async def test_claim_only_picks_matching_track(db_conn):
    from orchestrator.repo import queue as queue_repo, queue_write

    await queue_write.insert_queue(
        db_conn, strategy_code="TRK_A", interval_name="1h", instrument="BTCUSDT",
        sweep_config={"strategy": "grid", "params": [], "track": "hedging"},
        hypothesis="h", iter_budget=1, track="hedging",
    )
    claimed = await queue_repo.claim_next_pending(db_conn, track="trading")
    assert claimed is None, "trading claim must skip a hedging-tagged row"

    claimed_h = await queue_repo.claim_next_pending(db_conn, track="hedging")
    assert claimed_h is not None and claimed_h["strategy_code"] == "TRK_A"
```

> Use the **actual** claim function name in `repo/queue.py` (it may be `claim_next_pending`, `claim_pending`, or inline in `tick.py`). If the claim is inline in `services/tick.py`, extract it into `repo/queue.py:claim_next_pending(conn, *, track=None)` first (pure move, no behavior change) and update `tick.py` to call it — then this test targets the extracted function.

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_track_isolation.py::test_claim_only_picks_matching_track -v`
Expected: FAIL — `insert_queue`/`claim_next_pending` got an unexpected keyword argument `track`.

- [ ] **Step 3: Stamp `track` in `insert_queue`**

In `repo/queue_write.py:insert_queue`, accept `track: str | None = None`. The `sweep_config` dict is passed verbatim to the JSONB column (per CLAUDE.md — do NOT `json.dumps`). Stamp the tag into it before insert so it lives where the claim reads it:

```python
async def insert_queue(conn, *, strategy_code, interval_name, instrument,
                       sweep_config, hypothesis, iter_budget, track=None, **kw):
    if track is not None:
        sweep_config = {**sweep_config, "track": track}  # don't mutate caller's dict
    # ... existing INSERT, passing sweep_config to the jsonb $N param verbatim ...
```

- [ ] **Step 4: Add the `track` predicate to the claim query**

In the claim SQL (the `SELECT ... FOR UPDATE SKIP LOCKED` on `status='PENDING'` ordered by `(priority ASC, created_time ASC)`), add:

```sql
  AND ($N::text IS NULL OR sweep_config->>'track' = $N)
```

where `$N` is a new `track` param. NULL track = claim any row (backward compatible — existing untagged rows still claim).

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_track_isolation.py::test_claim_only_picks_matching_track -v`
Expected: PASS

- [ ] **Step 6: Scope `_queue_counts` by track; wire it into `get_state_digest`**

In `repo/agent_state.py`, give `_queue_counts` a `track: str | None = None` param and filter on the queue's JSONB tag:

```python
async def _queue_counts(conn: asyncpg.Connection, *, track: str | None = None) -> dict[str, int]:
    rows = await conn.fetch(
        """
        SELECT status, COUNT(*)::int AS n
        FROM research_queue
        WHERE ($1::text IS NULL OR sweep_config->>'track' = $1)
        GROUP BY status
        """,
        track,
    )
    counts = {r["status"]: int(r["n"]) for r in rows}
    for k in ("PENDING", "RUNNING", "PARKED", "COMPLETED", "FAILED"):
        counts.setdefault(k, 0)
    return counts
```

Then in `get_state_digest`, change the queue-counts call to `await _queue_counts(conn, track=track)`. Leave `_last_iterations` and `_recent_sig_edge_ids` calls global (per the Task 2 design note — no track linkage on `research_iteration_log`).

- [ ] **Step 7: Write + run the queue-counts isolation test**

```python
@pytest.mark.asyncio
async def test_queue_counts_scoped_to_track(db_conn):
    from orchestrator.repo import queue_write
    await queue_write.insert_queue(
        db_conn, strategy_code="TRK_C", interval_name="1h", instrument="BTCUSDT",
        sweep_config={"strategy": "grid", "params": []}, hypothesis="h",
        iter_budget=1, track="hedging",
    )
    counts = await agent_state._queue_counts(db_conn, track="trading")
    assert counts["PENDING"] == 0
    counts_h = await agent_state._queue_counts(db_conn, track="hedging")
    assert counts_h["PENDING"] >= 1
```

Run: `PYTHONPATH=src pytest tests/test_track_isolation.py::test_queue_counts_scoped_to_track -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add tests/test_track_isolation.py src/orchestrator/repo/queue_write.py src/orchestrator/repo/queue.py src/orchestrator/repo/agent_state.py
git commit -m "feat(track): stamp track on enqueue, scope claim + queue digest reads"
```

---

## Task 5: Accept `track` on `POST /queue`, `/tick`, `/tick/drain`; scope the re-discovery gate

**Files:**
- Modify: `src/orchestrator/api/queue.py` (`POST /queue` body model + re-discovery gate)
- Modify: `src/orchestrator/services/tick.py` (`run_tick` → pass track to claim)
- Modify: `src/orchestrator/services/tick_drain.py` (`drain_ticks` signature)
- Modify: `src/orchestrator/api/tick.py` (`/tick` + `/tick/drain` body models)
- Test: `tests/test_track_isolation.py`

- [ ] **Step 1: Add `track` to the `POST /queue` request model**

In `api/queue.py`, add an optional `track: str | None = None` (Literal-validated to `{"trading","hedging"}` if a model is used) to the enqueue body model, and pass it to `insert_queue(..., track=body.track)`.

- [ ] **Step 2: Scope the re-discovery DISCARD gate per-track**

Find the re-discovery check (returns `409 axis_previously_discarded`). The prior-DISCARD lookup must be filtered to the same track so a hedging discard doesn't block a trading axis-set. Add the track predicate to that query's `WHERE` (it reads `hypothesis_audit`/iteration rows joined to the queue — filter the queue side on `sweep_config->>'track'`, NULL = global as elsewhere).

- [ ] **Step 3: Write the failing test (cross-track discard does not block)**

```python
@pytest.mark.asyncio
async def test_rediscovery_gate_is_per_track(db_conn, async_client):
    # Seed a DISCARD audit on the hedging track for axis-set {rsi_period}.
    # Then enqueue the SAME axis-set on the trading track → must NOT 409.
    # (Construct the seed via the same helper the gate reads; assert POST /queue
    #  returns 200/201 for trading and 409 for a repeat on hedging.)
    ...
```

> Fill the seed using the repo's existing discard-audit fixture/helper (see `tests/` for how `axis_previously_discarded` is currently tested). The assertion: trading enqueue succeeds, hedging repeat is `409 axis_previously_discarded`.

- [ ] **Step 4: Run test to verify it fails, then implement, then passes**

Run: `PYTHONPATH=src pytest tests/test_track_isolation.py::test_rediscovery_gate_is_per_track -v`
Expected: FAIL → implement Step 2 predicate → PASS.

- [ ] **Step 5: Thread `track` through `run_tick` and `drain_ticks`**

`services/tick.py:run_tick` — add `track: str | None = None`, pass to the claim call (`claim_next_pending(conn, track=track)`).
`services/tick_drain.py:drain_ticks` — add `track: str | None = None` (after `session_id`), pass `track=track` into each interior `run_tick(...)` call (the loop at line ~136).

- [ ] **Step 6: Accept `track` on the `/tick` and `/tick/drain` endpoints**

In `api/tick.py`, add `track: str | None = None` to both endpoints' request body models and forward to `run_tick(..., track=body.track)` / `drain_ticks(..., track=body.track)`.

- [ ] **Step 7: Write + run the drain-scope test**

```python
@pytest.mark.asyncio
async def test_drain_only_drains_its_track(db_conn, async_client):
    # Enqueue one trading row and one hedging row; POST /tick/drain {track:"trading"};
    # assert the hedging row stays PENDING afterward.
    ...
```

Run: `PYTHONPATH=src pytest tests/test_track_isolation.py -v`
Expected: PASS (all track-isolation tests).

- [ ] **Step 8: Update playbook text for `/queue`, `/tick`, `/tick/drain`**

Append to each capability's `purpose`: "Pass `track` ('trading'|'hedging') to scope this to one research loop; omit for the legacy global queue."

- [ ] **Step 9: Commit**

```bash
git add tests/test_track_isolation.py src/orchestrator/api/queue.py src/orchestrator/api/tick.py src/orchestrator/services/tick.py src/orchestrator/services/tick_drain.py
git commit -m "feat(track): track param on /queue, /tick, /tick/drain + per-track re-discovery gate"
```

---

## Task 6: Full-suite regression + isolation integration test

**Files:**
- Test: `tests/test_track_isolation.py`

- [ ] **Step 1: Add the end-to-end isolation test**

```python
@pytest.mark.asyncio
async def test_hedging_exhaustion_invisible_to_trading_digest(db_conn):
    # The headline guarantee: a hedging ARCHETYPE_EXHAUSTION never appears
    # in the trading digest, and vice-versa.
    await _insert_run_summary(db_conn, title="ARCHETYPE_EXHAUSTION_X", track="hedging")
    trading = await agent_state.get_state_digest(db_conn, track="trading")
    hedging = await agent_state.get_state_digest(db_conn, track="hedging")
    assert trading["lockout_state"]["in_lockout"] is False
    assert hedging["lockout_state"]["in_lockout"] is True
```

- [ ] **Step 2: Run the whole suite**

Run: `PYTHONPATH=src pytest -q`
Expected: all green, including the pre-existing `tests/test_agent_state_phase2.py`, `tests/test_tick_drain.py`, `tests/test_tick_summary.py` (proves backward-compatibility — no existing test regressed).

- [ ] **Step 3: Commit**

```bash
git add tests/test_track_isolation.py
git commit -m "test(track): end-to-end per-track isolation guarantee"
```

---

## Task 7: `/research-track` launch command

**Files:**
- Create: `C:\Project\.claude\commands\research-track.md` (monorepo-root `.claude/`, NOT the orchestrator repo)

- [ ] **Step 1: Confirm the commands directory**

Run: `ls C:/Project/.claude/commands`
Expected: existing command files (or an empty dir). If the dir does not exist, create it.

- [ ] **Step 2: Write the command file**

```markdown
---
description: Launch one quant-researcher loop scoped to a research track (trading|hedging)
argument-hint: <trading|hedging>
---

You are launching ONE track of the dual-track research system. The track is: **$ARGUMENTS**.

## Step 1 — Validate the track
`$ARGUMENTS` MUST be exactly `trading` or `hedging`. If it is anything else (or empty), STOP and tell the operator the valid values. Do not guess.

## Step 2 — Resolve track config
- `trading` → agent_name=`quant-researcher-trading`, track=`trading`, gate=V60 (≥10%/yr ROBUST standalone).
- `hedging` → agent_name=`quant-researcher-hedging`, track=`hedging`, gate=`beats_buy_hold_risk_adj` (Phase 2).

## Step 3 — Confirm the target DB (prod vs local)
Read the orchestrator base URL the loop will use. Per `project_prod_research_from_local_tunnel`, the LOCAL orchestrator (:8082) runs against an EMPTY dev DB. Before booting:
- Confirm whether :8082 is tunneled to prod or is local dev.
- If it is local dev, WARN the operator and require explicit confirmation before continuing (two loops on an empty dev DB produce nothing).

## Step 4 — Boot the scoped researcher
Spawn the `quant-researcher` sub-agent (Agent tool — available because this is a top-level CLI session). In its prompt, pass:
- `X-Agent-Name: <agent_name from Step 2>` on every orchestrator call.
- Every `GET /agent/state` call MUST include `?track=<track>`.
- Every `POST /queue`, `/tick`, `/tick/drain` MUST include `track=<track>` in the body.
- Every journal write (HYPOTHESIS, RUN_SUMMARY, NULL_SCREEN_RESULT, SESSION_CHECKPOINT, specialist requests) MUST stamp `structured_data.track = "<track>"`.

## Step 5 — Checkpoint-to-operator contract
The researcher runs autonomously BETWEEN decision points. At each of these three decision points it MUST:
1. Write a track-tagged `SESSION_CHECKPOINT` journal row (entry_type=RUN_SUMMARY, title prefix `SESSION_CHECKPOINT_<TRACK>`, structured_data.track set, plus a `checkpoint_kind` of `graduation_candidate` | `pivot` | `archetype_exhaustion`).
2. RETURN to this CLI session with a compact summary of the decision and its recommendation.
3. WAIT — do not auto-decide. The operator answers, and this session resumes the sub-agent (SendMessage) with the decision.

Decision points: (a) a graduation candidate is ready, (b) a pivot decision (axis/archetype exhausted), (c) an archetype-exhaustion terminal.
```

- [ ] **Step 3: Smoke-test the command resolves**

Run (in a CLI): `/research-track trading` → it should validate, print the resolved config + the prod/local target, and (on confirmation) boot the scoped sub-agent. `/research-track bogus` → it should refuse.

- [ ] **Step 4: Commit (monorepo root repo)**

```bash
cd C:/Project
git add .claude/commands/research-track.md
git commit -m "feat(track): /research-track per-CLI launch command"
```

> The root repo is on branch `fix/ingest-test-db-mock` with unrelated working-tree changes. Create/checkout a dedicated branch first if the operator wants this isolated: `git checkout -b feat/dual-track-command` before the commit.

---

## Phase 1 Done When
- `PYTHONPATH=src pytest -q` is green (new isolation tests + all pre-existing tests).
- A hedging `ARCHETYPE_EXHAUSTION` is invisible in `GET /agent/state?track=trading`.
- A `/tick/drain {track:"trading"}` leaves hedging-tagged PENDING rows untouched.
- A no-`track` call returns the unchanged global digest (backward-compat proven by the existing suite passing).
- `/research-track trading` and `/research-track hedging` boot scoped, checkpoint-aware researchers from two separate CLIs.

## Hand-off to Phase 2
Phase 2 (`docs/plans/2026-06-06-dual-track-phase2-*.md`, to be written) adds the `beats_buy_hold_risk_adj` hedging gate (`services/analyze.py`) and the alpha-discovery workflow. The hedging track's gate is a stub returning the V60 path until Phase 2 lands — confirm the hedging loop still runs (it just uses V60 thresholds) so Phase 1 is independently shippable.
