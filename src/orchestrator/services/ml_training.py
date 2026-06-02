"""Spawn + manage blackheart-train subprocesses.

POST /ml/training-runs creates a row, kicks off the subprocess via
:func:`start_training_run`, and returns immediately. A background asyncio
task (:func:`_run_subprocess`) owns the lifecycle: marks RUNNING when
the process has a pid, awaits termination with a wall-clock cap, parses
the final stdout JSON for ``content_sha256`` + ``registration.model_id``,
then writes one of {COMPLETED, FAILED, TIMED_OUT} via
``repo.ml_training_runs.mark_terminal``.

The lifespan reaper (:func:`reap_orphaned_on_startup`) flips any rows
still in RUNNING from a prior process to ORPHANED. The subprocesses
themselves may still be alive — that's a forensic concern for the
operator, not a correctness concern for the next run.

Budget cap is enforced at the request boundary (:func:`enforce_daily_cap`)
so the subprocess is never spawned for a request over budget. FAILED
rows do not count, so a cluster of infra-failures don't lock the
researcher out.

Why subprocess and not in-process import: blackheart-train pulls in
LightGBM, scikit-learn, scipy — heavy and slow at import time. Keeping
it out-of-process means the orchestrator's hot path stays light and a
training crash never takes the orchestrator down.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit
from uuid import UUID

import asyncpg

from ..config import Settings
from ..errors import OrchestratorError
from ..repo import ml_training_runs as repo

logger = logging.getLogger(__name__)


# ── Subprocess environment propagation (2026-06-02) ─────────────────
#
# Root cause of INFRA_FAIL_2026-06-02: the train subprocess inherited
# the orchestrator's environment but blackheart_train reads its DB
# config from ``TRAIN_*`` env vars (env_prefix="TRAIN_", see
# blackheart-train/src/blackheart_train/settings.py), which the
# orchestrator does NOT set. With no TRAIN_DB_* present the train
# Settings fell back to host="localhost" — inside the orchestrator
# container Postgres is a separate compose service, so every connect
# died with "connection refused" (loader.py:495). Separately, the
# train ``--register`` / experiment-tracking client read
# TRAIN_ORCHESTRATOR_TOKEN (default = dev sentinel) which does not
# match the prod orchestrator's X-Orch-Token → HTTP 401 auth_bad_token
# on tracking.
#
# Fix: derive the TRAIN_* vars from the orchestrator's already-resolved
# Settings (db_dsn + auth_token + port) and inject them into the child
# env. This is pure subprocess wiring — it does NOT change Settings
# defaults, the auth shape, or the statistical contract. The child gets
# the SAME DB the orchestrator itself connects to, and a tracking token
# the orchestrator will accept.


def _dsn_to_train_db_env(dsn: str) -> dict[str, str]:
    """Parse a libpq/asyncpg postgres DSN into the TRAIN_DB_* env vars
    blackheart_train.settings.Settings reads. Returns only the keys we
    can resolve from the DSN; absent components are left to the child's
    own defaults rather than emitting empty overrides."""
    parts = urlsplit(dsn)
    env: dict[str, str] = {}
    if parts.hostname:
        env["TRAIN_DB_HOST"] = parts.hostname
    if parts.port:
        env["TRAIN_DB_PORT"] = str(parts.port)
    # urlsplit keeps the leading "/" on the path; the db name is the rest.
    dbname = parts.path.lstrip("/")
    if dbname:
        env["TRAIN_DB_NAME"] = dbname
    if parts.username:
        env["TRAIN_DB_USER"] = unquote(parts.username)
    if parts.password is not None:
        env["TRAIN_DB_PASSWORD"] = unquote(parts.password)
    return env


def build_subprocess_env(settings: Settings) -> dict[str, str]:
    """Construct the environment for the blackheart_train subprocess.

    Inherits the orchestrator's own ``os.environ`` (so PATH, venv,
    locale, etc. survive) and overlays the ``TRAIN_*`` vars derived from
    the orchestrator's resolved config:

    - ``TRAIN_DB_*``         from ``settings.db_dsn`` — same DB the
                             orchestrator connects to.
    - ``TRAIN_ORCHESTRATOR_URL``   the loopback orchestrator
                             (``--register`` + tracking POST target).
    - ``TRAIN_ORCHESTRATOR_TOKEN`` the orchestrator's X-Orch-Token so
                             registration + experiment tracking auth.

    Pure function (modulo reading ``os.environ`` once) so the contract
    is unit-testable without spawning anything.
    """
    env = dict(os.environ)
    env.update(_dsn_to_train_db_env(settings.db_dsn.get_secret_value()))
    # The train worker posts to the orchestrator itself for --register
    # and experiment tracking. Point it at this orchestrator's loopback
    # bind + the secret it will accept.
    env["TRAIN_ORCHESTRATOR_URL"] = f"http://127.0.0.1:{settings.port}"
    env["TRAIN_ORCHESTRATOR_TOKEN"] = settings.auth_token.get_secret_value()
    # Artifact output dir. blackheart-train.Settings.artifact_dir defaults
    # to the Windows host path C:/Project/blackheart-train/artifacts, which
    # is invalid inside the Linux container (the train repo is mounted
    # read-only at /train:ro; `C:` does not exist) → write_artifact dies
    # with OSError [Errno 30] Read-only file system AFTER a fully
    # successful train+gauntlet. Only override when the operator has set a
    # writable container path — empty string leaves the child's own
    # default untouched so dev-host runs are unaffected (same conditional
    # pattern as _dsn_to_train_db_env).
    artifact_dir = settings.ml_training_artifact_dir.strip()
    if artifact_dir:
        env["TRAIN_ARTIFACT_DIR"] = artifact_dir
    return env


# ── Public entry points ─────────────────────────────────────────────


async def enforce_daily_cap(
    conn: asyncpg.Connection,
    *,
    settings: Settings,
    agent_name: str,
) -> None:
    """Raise 429 if the agent has hit its rolling daily cap. FAILED and
    TIMED_OUT rows don't count — see V99 schema comments."""
    n = await repo.count_active_in_window(
        conn,
        agent_name=agent_name,
        window_hours=settings.ml_training_daily_window_hours,
    )
    if n >= settings.ml_training_daily_cap:
        raise OrchestratorError(
            status_code=429,
            error_code="ml_training_budget_exceeded",
            message=(
                f"agent={agent_name} has {n} training runs in the last "
                f"{settings.ml_training_daily_window_hours}h "
                f"(cap={settings.ml_training_daily_cap}). Wait for the "
                f"window to roll forward or escalate to the operator."
            ),
            retryable=True,
            hint=(
                "Failed runs don't count. If recent attempts crashed, "
                "those don't consume budget."
            ),
            details={
                "active_in_window": n,
                "cap": settings.ml_training_daily_cap,
                "window_hours": settings.ml_training_daily_window_hours,
            },
        )


def build_cli_args(
    *,
    spec_name: str,
    walk_forward: bool,
    gauntlet: bool,
    folds: int | None,
    bayesian: bool,
    do_register: bool,
    allow_leakage: bool,
    no_write: bool,
) -> list[str]:
    """Translate the request body into ``blackheart-train`` argparse
    flags. Pure function so tests can pin the contract.

    ``do_register`` maps to the CLI's ``--register`` flag — Pydantic
    renamed because ``register`` shadows ``BaseModel`` machinery."""
    args = ["--model", spec_name]
    if walk_forward:
        args.append("--walk-forward")
    if gauntlet:
        args.append("--gauntlet")
    if folds is not None:
        args.extend(["--folds", str(folds)])
    if bayesian:
        args.append("--bayesian")
    if do_register:
        args.append("--register")
    if allow_leakage:
        args.append("--allow-leakage")
    if no_write:
        args.append("--no-write")
    return args


def start_training_run(
    *,
    settings: Settings,
    db_pool: asyncpg.Pool,
    training_run_id: UUID,
    cli_args: list[str],
    agent_name: str,
) -> asyncio.Task:
    """Spawn the subprocess as a background task. Returns the task
    handle (mostly for tests). Production callers fire-and-forget;
    completion writes land via the background task itself."""
    return asyncio.create_task(
        _run_subprocess(
            settings=settings,
            db_pool=db_pool,
            training_run_id=training_run_id,
            cli_args=cli_args,
            agent_name=agent_name,
        ),
        name=f"ml-training-{training_run_id}",
    )


async def reap_orphaned_on_startup(db_pool: asyncpg.Pool) -> int:
    """Lifespan startup hook. Any row left in RUNNING from a prior
    orchestrator process is now ORPHANED. Returns the count."""
    async with db_pool.acquire() as conn:
        n = await repo.reap_orphaned(conn)
    if n:
        logger.warning("reaped %d orphaned ml_training_runs row(s)", n)
    return n


# ── Private: the actual subprocess driver ───────────────────────────


async def _run_subprocess(
    *,
    settings: Settings,
    db_pool: asyncpg.Pool,
    training_run_id: UUID,
    cli_args: list[str],
    agent_name: str,
) -> None:
    """The background task. Owns the row from PENDING to terminal.

    Catches every exception and writes a FAILED row rather than letting
    the asyncio task die silently — the row staying in PENDING with no
    finished_at would be the worst possible failure mode for the
    researcher (would wait forever for a status change that never
    comes)."""
    started_monotonic = time.monotonic()
    proc: asyncio.subprocess.Process | None = None
    try:
        # Sanity-check the python binary BEFORE we mark RUNNING. A
        # missing interpreter means we never spawned; better to record
        # FAILED with a clear error than mark RUNNING and crash.
        if not os.path.isfile(settings.ml_training_python_path):
            await _terminal(
                db_pool,
                training_run_id=training_run_id,
                status="FAILED",
                exit_code=None,
                duration_seconds=0,
                stdout_tail=None,
                stderr_tail=None,
                experiment_run_id=None,
                content_sha256=None,
                model_id=None,
                error_envelope={
                    "error_code": "python_interpreter_missing",
                    "message": (
                        f"ml_training_python_path does not exist: "
                        f"{settings.ml_training_python_path}"
                    ),
                },
                updated_by=agent_name,
            )
            return

        cmd = [
            settings.ml_training_python_path,
            "-m",
            "blackheart_train.cli",
            *cli_args,
        ]
        # F5 (2026-05-19) — Redirect stdout/stderr to per-run temp
        # files instead of PIPE. blackheart-train can produce 100MB+ of
        # logs over a multi-hour run; the OS pipe buffer (typically
        # 64KB / 4KB depending on platform) fills, the subprocess
        # blocks on its next write, and proc.communicate() — which
        # doesn't drain until exit — deadlocks the whole thing. Files
        # have unbounded capacity. We read the tail at terminal time.
        stdio_dir = Path(tempfile.gettempdir()) / "blackheart-train"
        stdio_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = stdio_dir / f"{training_run_id}.stdout"
        stderr_path = stdio_dir / f"{training_run_id}.stderr"
        logger.info(
            "spawning ml training | run_id=%s cmd=%s cwd=%s stdout=%s stderr=%s",
            training_run_id, cmd, settings.ml_training_repo_path,
            stdout_path, stderr_path,
        )
        # Open file handles for the subprocess to own. We pass file
        # descriptors so the kernel writes directly — no buffering
        # in our process.
        stdout_f = open(stdout_path, "wb")
        stderr_f = open(stderr_path, "wb")
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=settings.ml_training_repo_path,
                stdout=stdout_f,
                stderr=stderr_f,
                # 2026-06-02 — inject TRAIN_* env so the child reaches the
                # orchestrator's DB (not localhost) and authenticates its
                # registration/tracking POSTs. See build_subprocess_env.
                env=build_subprocess_env(settings),
            )
        finally:
            # The subprocess inherits the fds via dup(); we can close
            # our copies immediately. Writes by the child still land
            # on disk.
            stdout_f.close()
            stderr_f.close()

        async with db_pool.acquire() as conn:
            await repo.mark_running(
                conn,
                training_run_id=training_run_id,
                pid=proc.pid or 0,
                updated_by=agent_name,
            )

        try:
            await asyncio.wait_for(
                proc.wait(),
                timeout=settings.ml_training_max_seconds,
            )
        except asyncio.TimeoutError:
            # Wall-clock cap blown. Kill the subprocess and mark
            # TIMED_OUT. The files capture whatever was already
            # flushed before the kill; we still read their tails.
            with _quiet_kill(proc):
                proc.kill()
            # Give the OS a beat to flush the kill-time writes.
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass
            duration = int(time.monotonic() - started_monotonic)
            await _terminal(
                db_pool,
                training_run_id=training_run_id,
                status="TIMED_OUT",
                exit_code=None,
                duration_seconds=duration,
                stdout_tail=_read_file_tail(
                    stdout_path, settings.ml_training_stdio_tail_bytes
                ),
                stderr_tail=_read_file_tail(
                    stderr_path, settings.ml_training_stdio_tail_bytes
                ),
                experiment_run_id=None,
                content_sha256=None,
                model_id=None,
                error_envelope={
                    "error_code": "wall_clock_exceeded",
                    "message": (
                        f"subprocess exceeded "
                        f"{settings.ml_training_max_seconds}s; killed"
                    ),
                    "duration_seconds": duration,
                },
                updated_by=agent_name,
            )
            return

        duration = int(time.monotonic() - started_monotonic)
        exit_code = proc.returncode if proc.returncode is not None else -1
        # Read the full stdout for summary parsing; tail-cap both for
        # the audit row. Stdout is needed in full to find the JSON
        # summary blackheart-train dumps at the end.
        stdout_text = _read_file_full(stdout_path)
        stdout_tail = _tail_text(stdout_text, settings.ml_training_stdio_tail_bytes)
        stderr_tail = _read_file_tail(
            stderr_path, settings.ml_training_stdio_tail_bytes
        )

        if exit_code != 0:
            await _terminal(
                db_pool,
                training_run_id=training_run_id,
                status="FAILED",
                exit_code=exit_code,
                duration_seconds=duration,
                stdout_tail=stdout_tail,
                stderr_tail=stderr_tail,
                experiment_run_id=None,
                content_sha256=None,
                model_id=None,
                error_envelope={
                    "error_code": "subprocess_nonzero_exit",
                    "message": f"blackheart-train exited with code {exit_code}",
                },
                updated_by=agent_name,
            )
            return

        # Parse the final JSON summary. blackheart-train dumps the
        # whole summary to stdout right before exit, so the LAST
        # JSON object in the stream is what we want. Be robust to
        # log lines interleaved with the JSON tail.
        summary = _parse_summary(stdout_text)
        content_sha256, experiment_run_id, model_id = _extract_ids(summary)

        await _terminal(
            db_pool,
            training_run_id=training_run_id,
            status="COMPLETED",
            exit_code=exit_code,
            duration_seconds=duration,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
            experiment_run_id=experiment_run_id,
            content_sha256=content_sha256,
            model_id=model_id,
            error_envelope=None,
            updated_by=agent_name,
        )

    except Exception as exc:  # noqa: BLE001 — never let the task die silently
        logger.exception("ml training run %s crashed", training_run_id)
        try:
            await _terminal(
                db_pool,
                training_run_id=training_run_id,
                status="FAILED",
                exit_code=None,
                duration_seconds=int(time.monotonic() - started_monotonic),
                stdout_tail=None,
                stderr_tail=None,
                experiment_run_id=None,
                content_sha256=None,
                model_id=None,
                error_envelope={
                    "error_code": "background_task_crashed",
                    "message": f"{type(exc).__name__}: {exc}",
                },
                updated_by=agent_name,
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "failed to mark training run %s as FAILED after task crash",
                training_run_id,
            )


# ── Internal helpers ────────────────────────────────────────────────


async def _terminal(
    db_pool: asyncpg.Pool,
    *,
    training_run_id: UUID,
    status: str,
    exit_code: int | None,
    duration_seconds: int | None,
    stdout_tail: str | None,
    stderr_tail: str | None,
    experiment_run_id: UUID | None,
    content_sha256: str | None,
    model_id: UUID | None,
    error_envelope: dict[str, Any] | None,
    updated_by: str,
) -> None:
    async with db_pool.acquire() as conn:
        await repo.mark_terminal(
            conn,
            training_run_id=training_run_id,
            status=status,
            exit_code=exit_code,
            duration_seconds=duration_seconds,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
            experiment_run_id=experiment_run_id,
            content_sha256=content_sha256,
            model_id=model_id,
            error_envelope=error_envelope,
            updated_by=updated_by,
        )


def _tail(b: bytes, cap_bytes: int) -> str:
    if not b:
        return ""
    return _tail_text(b.decode("utf-8", errors="replace"), cap_bytes)


def _tail_text(s: str, cap_bytes: int) -> str:
    encoded = s.encode("utf-8")
    if len(encoded) <= cap_bytes:
        return s
    # Trim from the front so the tail is the most useful part. Decode
    # forgivingly to handle a partial multi-byte cut.
    return encoded[-cap_bytes:].decode("utf-8", errors="replace")


def _read_file_tail(path: Path, cap_bytes: int) -> str:
    """Read the last ``cap_bytes`` of a stdio capture file. Returns
    empty string if the file is missing (e.g. subprocess never ran)
    or unreadable. F5 (2026-05-19) — replaces the bytes-from-PIPE path
    that deadlocked on long training runs."""
    try:
        size = path.stat().st_size
    except OSError:
        return ""
    if size == 0:
        return ""
    try:
        with open(path, "rb") as f:
            if size > cap_bytes:
                f.seek(size - cap_bytes)
            data = f.read()
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace")


def _read_file_full(path: Path) -> str:
    """Read the entire stdio capture. Used for summary parsing on
    COMPLETED runs where we need the trailing JSON. Returns empty
    string on read failure rather than raising — caller handles
    missing-summary path."""
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace")


def _parse_summary(stdout_text: str) -> dict[str, Any] | None:
    """blackheart-train ends with one ``json.dump(...)`` of the summary.
    Find the LAST top-level JSON object in stdout — it may be preceded
    by log lines. Returns None when no parseable JSON tail is found."""
    if not stdout_text:
        return None
    # Walk backward from EOF looking for a brace-balanced JSON object.
    # The summary is multi-line + indent=2, so we search for the last
    # closing brace and then back up to its matching opener.
    end_idx = stdout_text.rfind("}")
    if end_idx == -1:
        return None
    depth = 0
    start_idx: int | None = None
    for i in range(end_idx, -1, -1):
        c = stdout_text[i]
        if c == "}":
            depth += 1
        elif c == "{":
            depth -= 1
            if depth == 0:
                start_idx = i
                break
    if start_idx is None:
        return None
    candidate = stdout_text[start_idx : end_idx + 1]
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return None
    return None


def _extract_ids(
    summary: dict[str, Any] | None,
) -> tuple[str | None, UUID | None, UUID | None]:
    """Pull (content_sha256, experiment_run_id, model_id) out of the
    train summary. All three are independently optional — the artifact
    may exist without registration, registration may exist without an
    experiment_run id reaching stdout, etc."""
    if not summary:
        return None, None, None
    artifact = summary.get("artifact") or {}
    registration = summary.get("registration") or {}
    content_sha = artifact.get("content_sha256") if isinstance(artifact, dict) else None
    model_id_raw = (
        registration.get("model_id") if isinstance(registration, dict) else None
    )
    exp_run_raw = summary.get("experiment_run_id") or summary.get("run_id")
    return (
        content_sha if isinstance(content_sha, str) else None,
        _maybe_uuid(exp_run_raw),
        _maybe_uuid(model_id_raw),
    )


def _maybe_uuid(value: Any) -> UUID | None:
    if not value:
        return None
    if isinstance(value, UUID):
        return value
    if isinstance(value, str):
        try:
            return UUID(value)
        except (ValueError, AttributeError):
            return None
    return None


class _quiet_kill:
    """Context manager around proc.kill() that swallows ProcessLookupError
    when the process has already exited between the timeout and the
    kill call. Without this guard, the wait_for-then-kill race produces
    a spurious traceback for what is otherwise a clean timeout path."""

    def __init__(self, proc: asyncio.subprocess.Process | None):
        self._proc = proc

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        return exc_type is ProcessLookupError
