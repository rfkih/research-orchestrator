"""asyncpg queries against ``experiment_run`` + ``experiment_metric`` (V92).

R1.A of the ML/research uplift plan. Adds first-class run tracking — every
blackheart-train invocation registers a row here, regardless of whether it
ultimately produces a registered model artifact. The previous trail
(``model_registry``) only captured successful + gauntlet-passing runs;
failures and aborts left nothing to look back at.

Write surface, in order of typical use:

* :func:`insert_run`     — POST /experiments (start), status='running'
* :func:`set_params`     — PATCH /experiments/{id}/params (one-shot snapshot)
* :func:`set_tags`       — PATCH /experiments/{id}/tags
* :func:`log_metric`     — PATCH /experiments/{id}/metrics (one scalar)
* :func:`log_metrics`    — PATCH /experiments/{id}/metrics (batch)
* :func:`finish_run`     — POST /experiments/{id}/finish — sets terminal state

Read surface:

* :func:`get_run`        — by run_id
* :func:`list_runs`      — filtered, paginated
* :func:`leaderboard`    — sorted by a named summary metric

JSONB columns (``params``, ``summary_metrics``, ``tags``, ``leakage_report``)
are passed as Python dicts — the asyncpg codec at
``infra/db.py:_init_connection`` registers the encoder so we don't double-
encode. Mirrors peer ``repo/models.py`` style: raw SQL, explicit type casts
on the ``$N`` params where asyncpg's prepared-statement inference might
otherwise clash.
"""
from __future__ import annotations

from typing import Any
from uuid import UUID


# ─────────────────────────────────────────────────────────────────────────
# Write surface
# ─────────────────────────────────────────────────────────────────────────


async def insert_run(
    conn,
    *,
    spec_name: str,
    spec_version: str | None,
    spec_symbol: str | None,
    spec_interval: str | None,
    spec_horizon_bars: int | None,
    git_sha: str | None,
    dataset_sha: str | None,
    params: dict[str, Any] | None,
    tags: dict[str, Any] | None,
    created_by: str,
) -> dict[str, Any]:
    """Open one ``experiment_run`` row with status='running'.

    ``params`` and ``tags`` are optional at start time — callers commonly
    snapshot params after spec resolution (post-CLI-argparse) via
    :func:`set_params`, and tags accumulate over a run's lifetime.

    Returns ``{run_id, started_at, status}`` — the run_id is the handle the
    train-side client carries for all subsequent PATCHes.
    """
    row = await conn.fetchrow(
        """
        INSERT INTO experiment_run (
            spec_name, spec_version, spec_symbol, spec_interval, spec_horizon_bars,
            git_sha, dataset_sha,
            params, tags,
            created_by
        ) VALUES (
            $1::text, $2::text, $3::text, $4::text, $5::int,
            $6::text, $7::text,
            COALESCE($8, '{}'::jsonb),
            COALESCE($9, '{}'::jsonb),
            $10::text
        )
        RETURNING run_id, started_at, status
        """,
        spec_name, spec_version, spec_symbol, spec_interval, spec_horizon_bars,
        git_sha, dataset_sha,
        params, tags,
        created_by,
    )
    return dict(row)


async def set_params(
    conn,
    *,
    run_id: UUID,
    params: dict[str, Any],
    actor: str,
) -> dict | None:
    """Replace the run's ``params`` JSONB blob wholesale.

    Why replace, not merge: hyperparams snapshots are point-in-time. The
    train-side client sends one final dict after argparse + search;
    incremental merges would conflate config sources. Use :func:`set_tags`
    for accumulating annotations.

    Returns the updated row's metadata. ``None`` if ``run_id`` is unknown.
    """
    row = await conn.fetchrow(
        """
        UPDATE experiment_run
        SET params = $2,
            updated_by = $3,
            updated_time = NOW()
        WHERE run_id = $1
        RETURNING run_id, params, updated_time
        """,
        run_id, params, actor,
    )
    return dict(row) if row else None


async def set_tags(
    conn,
    *,
    run_id: UUID,
    tags: dict[str, Any],
    merge: bool,
    actor: str,
) -> dict | None:
    """Set or merge the run's ``tags`` JSONB blob.

    ``merge=True`` shallow-merges on top of the existing tags (PostgreSQL's
    JSONB concatenation operator ``||`` overwrites on key collision —
    rightmost wins, so the new ``tags`` argument takes precedence).
    ``merge=False`` replaces wholesale.

    Returns the updated row's tags. ``None`` if ``run_id`` is unknown.
    """
    if merge:
        sql = """
            UPDATE experiment_run
            SET tags = tags || $2,
                updated_by = $3,
                updated_time = NOW()
            WHERE run_id = $1
            RETURNING run_id, tags, updated_time
        """
    else:
        sql = """
            UPDATE experiment_run
            SET tags = $2,
                updated_by = $3,
                updated_time = NOW()
            WHERE run_id = $1
            RETURNING run_id, tags, updated_time
        """
    row = await conn.fetchrow(sql, run_id, tags, actor)
    return dict(row) if row else None


async def log_metric(
    conn,
    *,
    run_id: UUID,
    metric_name: str,
    value: float,
    fold_idx: int | None = None,
    step: int | None = None,
) -> int:
    """Append one metric row.

    When ``fold_idx IS NULL`` the metric is also mirrored into
    ``experiment_run.summary_metrics`` so the leaderboard SELECT can read it
    without joining. Per-fold metrics (fold_idx >= 0) stay only in
    ``experiment_metric``.

    Returns the new ``metric_id``. The chk_experiment_metric_value_finite
    constraint will raise asyncpg.CheckViolationError on NaN/Inf — callers
    must sanitize per cli.py's allow_nan=False convention.
    """
    # One transaction: the metric INSERT and the summary-mirror UPDATE must
    # land together. Split-autocommit left metric rows committed with no
    # JSONB mirror on a crash between the statements — and leaderboard()
    # filters on ``summary_metrics ? metric``, so the run silently vanished
    # from rankings while the metric row said otherwise.
    async with conn.transaction():
        metric_id = await conn.fetchval(
            """
            INSERT INTO experiment_metric (run_id, fold_idx, metric_name, value, step)
            VALUES ($1, $2::int, $3::text, $4::double precision, $5::int)
            RETURNING metric_id
            """,
            run_id, fold_idx, metric_name, value, step,
        )
        if fold_idx is None:
            # Mirror run-level scalars into the header for JOIN-free leaderboards.
            # jsonb_build_object + concat: existing keys are overwritten.
            await conn.execute(
                """
                UPDATE experiment_run
                SET summary_metrics = summary_metrics || jsonb_build_object($2::text, $3::double precision),
                    updated_time = NOW()
                WHERE run_id = $1
                """,
                run_id, metric_name, value,
            )
    return int(metric_id)


async def log_metrics(
    conn,
    *,
    run_id: UUID,
    metrics: list[dict[str, Any]],
) -> int:
    """Batch-insert metrics. Each dict: ``{name, value, fold_idx?, step?}``.

    Run-level entries (fold_idx omitted or None) are additionally mirrored
    into ``experiment_run.summary_metrics`` in a single follow-up UPDATE
    (one concat, not N). Returns the count inserted.

    Empty list is a no-op (returns 0) — convenient when a fold loop emits
    nothing for a given run.

    **Bug-#5 semantic — duplicate metric names:** ``experiment_metric`` is
    append-only at the DB level (no UNIQUE constraint on
    (run_id, metric_name, fold_idx)), so logging the same name twice on a
    run leaves two rows. The ``summary_metrics`` JSONB mirror uses
    Postgres ``||`` which is rightmost-wins on key collision, so the
    JSONB carries only the latest value. Practical implication: the
    leaderboard (reading from JSONB) shows the most recent log, but a
    direct query of experiment_metric returns the full history. This
    matches the operator pattern of "log run-level scalars once per
    run"; if you need accumulating sequence metrics (e.g. epoch-by-epoch
    learning curves), pass a ``step`` index and the rows stay
    distinguishable in the normalized table.
    """
    if not metrics:
        return 0

    # Sanity: every entry needs name + value. fold_idx/step optional.
    rows = [
        (
            run_id,
            m.get("fold_idx"),
            m["name"],
            float(m["value"]),
            m.get("step"),
        )
        for m in metrics
    ]
    # One transaction — see log_metric: a crash between the INSERTs and the
    # mirror UPDATE desyncs the leaderboard from the metric table.
    async with conn.transaction():
        await conn.executemany(
            """
            INSERT INTO experiment_metric (run_id, fold_idx, metric_name, value, step)
            VALUES ($1, $2::int, $3::text, $4::double precision, $5::int)
            """,
            rows,
        )

        # Collapse run-level entries into a single JSONB object and merge.
        run_level = {
            m["name"]: float(m["value"])
            for m in metrics
            if m.get("fold_idx") is None
        }
        if run_level:
            await conn.execute(
                """
                UPDATE experiment_run
                SET summary_metrics = summary_metrics || $2::jsonb,
                    updated_time = NOW()
                WHERE run_id = $1
                """,
                run_id, run_level,
            )
    return len(rows)


async def finish_run(
    conn,
    *,
    run_id: UUID,
    status: str,
    content_sha256: str | None,
    leakage_report: dict[str, Any] | None,
    error_message: str | None,
    dataset_sha: str | None,
    actor: str,
) -> dict | None:
    """Transition the run to a terminal state.

    Computes ``duration_seconds`` server-side from ``started_at`` so callers
    can't lie about wall-clock. Optimistic guard: only transitions from
    ``status='running'`` — if the row is already terminal this returns
    ``None`` and the API layer surfaces 409.

    ``content_sha256`` should be the model_registry.artifact_sha256 the run
    registered (or NULL for failed / --no-write runs). The V92 FK enforces
    referential integrity if present.

    Returns the updated row, or ``None`` on stale-snapshot / unknown id.
    """
    if status not in ("completed", "failed", "aborted"):
        raise ValueError(f"finish_run status must be terminal; got {status!r}")
    row = await conn.fetchrow(
        """
        UPDATE experiment_run
        SET status = $2::text,
            finished_at = NOW(),
            duration_seconds = EXTRACT(EPOCH FROM (NOW() - started_at))::int,
            content_sha256 = COALESCE($3::text, content_sha256),
            leakage_report = COALESCE($4, leakage_report),
            error_message = COALESCE($5::text, error_message),
            dataset_sha = COALESCE($6::text, dataset_sha),
            updated_by = $7::text,
            updated_time = NOW()
        WHERE run_id = $1
          AND status = 'running'
        RETURNING run_id, status, started_at, finished_at, duration_seconds,
                  content_sha256, dataset_sha, summary_metrics
        """,
        run_id, status, content_sha256, leakage_report, error_message,
        dataset_sha, actor,
    )
    return dict(row) if row else None


# ─────────────────────────────────────────────────────────────────────────
# Read surface
# ─────────────────────────────────────────────────────────────────────────


async def get_run(conn, run_id: UUID) -> dict | None:
    """Single-row fetch by run_id. Returns the full denormalized header
    plus the per-fold metric rows nested under ``metrics``.
    """
    header = await conn.fetchrow(
        """
        SELECT run_id, spec_name, spec_version, spec_symbol, spec_interval,
               spec_horizon_bars, status, started_at, finished_at,
               duration_seconds, git_sha, dataset_sha, content_sha256,
               params, summary_metrics, tags, leakage_report, error_message,
               created_time, created_by, updated_time, updated_by
        FROM experiment_run
        WHERE run_id = $1
        """,
        run_id,
    )
    if header is None:
        return None
    metrics = await conn.fetch(
        """
        SELECT metric_id, fold_idx, metric_name, value, step, logged_at
        FROM experiment_metric
        WHERE run_id = $1
        ORDER BY metric_name, fold_idx NULLS FIRST, step
        """,
        run_id,
    )
    out = dict(header)
    out["metrics"] = [dict(m) for m in metrics]
    return out


def _build_run_filters(
    *,
    spec_name: str | None,
    status: str | None,
    symbol: str | None,
    has_artifact: bool | None,
) -> tuple[list[str], list[object]]:
    """Build WHERE-clause fragments + ordered params. Same pattern as
    ``repo/models.py:_build_filters`` — first fragment is the literal
    ``TRUE`` so the caller always ``' AND '.join(wheres)`` cleanly.
    """
    wheres = ["TRUE"]
    params: list[object] = []
    n = 1
    for val, col in (
        (spec_name, "spec_name"),
        (status, "status"),
        (symbol, "spec_symbol"),
    ):
        if val is not None:
            wheres.append(f"{col} = ${n}::text")
            params.append(val)
            n += 1
    if has_artifact is True:
        wheres.append("content_sha256 IS NOT NULL")
    elif has_artifact is False:
        wheres.append("content_sha256 IS NULL")
    return wheres, params


async def list_runs(
    conn,
    *,
    spec_name: str | None = None,
    status: str | None = None,
    symbol: str | None = None,
    has_artifact: bool | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    """List ``experiment_run`` headers with optional filters.

    Ordered by ``started_at DESC`` — the operator's "what ran recently?"
    default. Per-fold metrics are NOT joined in (use :func:`get_run` for
    that); summary_metrics on the header carries enough for a list view.
    """
    wheres, params = _build_run_filters(
        spec_name=spec_name, status=status, symbol=symbol, has_artifact=has_artifact,
    )
    n = len(params) + 1
    params += [limit, offset]
    rows = await conn.fetch(
        f"""
        SELECT run_id, spec_name, spec_version, spec_symbol, spec_interval,
               status, started_at, finished_at, duration_seconds,
               git_sha, dataset_sha, content_sha256,
               summary_metrics, tags,
               created_by
        FROM experiment_run
        WHERE {' AND '.join(wheres)}
        ORDER BY started_at DESC, run_id DESC
        LIMIT ${n} OFFSET ${n + 1}
        """,
        *params,
    )
    return [dict(r) for r in rows]


async def count_runs(
    conn,
    *,
    spec_name: str | None = None,
    status: str | None = None,
    symbol: str | None = None,
    has_artifact: bool | None = None,
) -> int:
    """Mirror of ``list_runs`` filters for paginated total."""
    wheres, params = _build_run_filters(
        spec_name=spec_name, status=status, symbol=symbol, has_artifact=has_artifact,
    )
    row = await conn.fetchrow(
        f"SELECT COUNT(*) AS n FROM experiment_run WHERE {' AND '.join(wheres)}",
        *params,
    )
    return int(row["n"])


async def leaderboard(
    conn,
    *,
    spec_name: str,
    metric: str,
    order: str = "desc",
    status: str | None = "completed",
    limit: int = 20,
) -> list[dict]:
    """Top runs for ``spec_name`` ordered by a named summary metric.

    Reads from ``experiment_run.summary_metrics`` JSONB (not the normalized
    table) — that's the whole reason the header carries a mirror. The
    JSONB-cast-to-numeric path is index-friendly via the GIN index on
    summary_metrics for the existence check; the ORDER BY itself is a
    sequential pass over the filtered set, which is fine for ≤10⁴ runs per
    spec.

    Runs missing ``metric`` from their summary_metrics are filtered out
    (``? metric`` operator). ``status`` default 'completed' avoids surfacing
    running / failed rows in the leaderboard.
    """
    if order not in ("asc", "desc"):
        raise ValueError(f"order must be 'asc' or 'desc'; got {order!r}")
    wheres = ["spec_name = $1::text", "summary_metrics ? $2::text"]
    params: list[object] = [spec_name, metric]
    if status is not None:
        wheres.append("status = $3::text")
        params.append(status)
    direction = "DESC" if order == "desc" else "ASC"
    params.append(limit)
    rows = await conn.fetch(
        f"""
        SELECT run_id, spec_name, spec_version, spec_symbol, spec_interval,
               status, started_at, finished_at, duration_seconds,
               content_sha256,
               (summary_metrics ->> $2::text)::double precision AS metric_value,
               summary_metrics, tags
        FROM experiment_run
        WHERE {' AND '.join(wheres)}
        ORDER BY (summary_metrics ->> $2::text)::double precision {direction} NULLS LAST
        LIMIT ${len(params)}
        """,
        *params,
    )
    return [dict(r) for r in rows]
