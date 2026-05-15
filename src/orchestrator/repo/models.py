"""asyncpg queries against ``model_registry`` (V66 table).

Read + write — the orchestrator is the registry's write surface for
trained sub-model artifacts (M5e). Connects as ``blackheart_research``
which has INSERT + UPDATE on ``model_registry`` per V66 grants.

Two operations matter here:

* :func:`find_by_artifact_sha` — content-addressable lookup, the
  primary idempotency anchor for ``POST /models/register``.
* :func:`insert_model` — atomic insert that derives the next ``version``
  for the ``(purpose, symbol, interval, horizon_bars)`` tuple. The V66
  ``uq_model_registry_directional`` UNIQUE constraint guarantees we
  never write two rows with the same tuple+version.

JSONB columns (``feature_set``, ``hyperparams``, ``metrics``) are passed
as Python dicts — the asyncpg codec at ``infra/db.py`` already registers
the encoder so we don't double-encode. Mirror peer pattern in
``repo/queue_write.py``.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID


async def find_by_artifact_sha(conn, sha: str) -> dict | None:
    """Return the existing ``model_registry`` row whose
    ``artifact_sha256`` equals ``sha``, or ``None``. Used as the
    idempotency anchor on ``POST /models/register``.
    """
    row = await conn.fetchrow(
        """
        SELECT id, family, purpose, symbol, interval, horizon_bars,
               feature_set, hyperparams, metrics,
               random_seed, artifact_uri, artifact_sha256, artifact_size_bytes,
               status, version, created_time, created_by
        FROM model_registry
        WHERE artifact_sha256 = $1
        """,
        sha,
    )
    return dict(row) if row else None


async def insert_model(
    conn,
    *,
    family: str,
    purpose: str,
    symbol: str,
    interval: str | None,
    horizon_bars: int | None,
    feature_set: dict[str, Any],
    hyperparams: dict[str, Any] | None,
    metrics: dict[str, Any] | None,
    random_seed: int,
    artifact_uri: str | None,
    artifact_sha256: str,
    artifact_size_bytes: int | None,
    status: str,
    created_by: str,
) -> dict[str, Any]:
    """Insert one ``model_registry`` row. Version is computed inline in
    the same statement as the INSERT — closing the race window where
    two concurrent callers' ``SELECT MAX(version)`` could each return
    the same value before either's INSERT committed.

    ``IS NOT DISTINCT FROM`` in the subquery matches NULL values
    naturally so the optional ``interval`` / ``horizon_bars`` columns
    don't need special-case branches.

    Returns the new row's ``id, version, status, created_time``. If the
    V66 ``uq_model_registry_directional`` UNIQUE constraint trips
    (manual SQL, operator running two CLIs simultaneously through
    different connections), asyncpg raises ``UniqueViolationError``
    which the API layer converts to 409.
    """
    # Explicit casts on every parameter referenced in BOTH the INSERT
    # VALUES and the inline subquery. asyncpg's prepared-statement
    # type inference deduces from the first usage site, and the
    # subquery's varchar-column comparison clashes with the VALUES
    # clause's text inference — surfaces as
    # ``AmbiguousParameterError: inconsistent types deduced for
    # parameter $N — text versus character varying``. The cast pins
    # the type at parse time so the inference is unambiguous.
    row = await conn.fetchrow(
        """
        INSERT INTO model_registry (
            family, purpose, symbol, interval, horizon_bars,
            feature_set, hyperparams, metrics,
            random_seed,
            artifact_uri, artifact_sha256, artifact_size_bytes,
            status,
            version,
            created_by
        ) VALUES (
            $1, $2::text, $3::text, $4::text, $5::int,
            $6, $7, $8,
            $9,
            $10, $11, $12,
            $13,
            COALESCE(
                (SELECT MAX(version) FROM model_registry
                 WHERE purpose = $2::text
                   AND symbol = $3::text
                   AND interval IS NOT DISTINCT FROM $4::text
                   AND horizon_bars IS NOT DISTINCT FROM $5::int),
                0
            ) + 1,
            $14
        )
        RETURNING id, version, status, created_time
        """,
        family, purpose, symbol, interval, horizon_bars,
        feature_set, hyperparams, metrics,
        random_seed,
        artifact_uri, artifact_sha256, artifact_size_bytes,
        status,
        created_by,
    )
    return dict(row)


async def get_model_by_id(conn, model_id: UUID) -> dict | None:
    """Read one row by primary key. Used by the GET sibling endpoint
    and by tests."""
    row = await conn.fetchrow(
        """
        SELECT id, family, purpose, symbol, interval, horizon_bars,
               feature_set, hyperparams, metrics,
               random_seed, artifact_uri, artifact_sha256, artifact_size_bytes,
               artifact_synced_to_vps, artifact_synced_at,
               status, version, created_time, created_by, updated_time
        FROM model_registry
        WHERE id = $1
        """,
        model_id,
    )
    return dict(row) if row else None


def _build_filters(
    *,
    status: str | None,
    purpose: str | None,
    symbol: str | None,
    synced: bool | None,
) -> tuple[list[str], list[object]]:
    """Build the WHERE-clause fragments and ordered params for
    ``model_registry`` list/count queries. Returns ``(wheres, params)``
    where the first fragment is the literal ``TRUE`` so the caller can
    always ``' AND '.join(wheres)`` without an empty-list special case.

    Positional refs in the returned fragments start at $1. Callers that
    append more params (``LIMIT``/``OFFSET``) compute their own $N from
    ``len(params) + 1``.
    """
    wheres = ["TRUE"]
    params: list[object] = []
    n = 1
    for val, col in (
        (status, "status"),
        (purpose, "purpose"),
        (symbol, "symbol"),
    ):
        if val is not None:
            wheres.append(f"{col} = ${n}::text")
            params.append(val)
            n += 1
    if synced is not None:
        wheres.append(f"artifact_synced_to_vps = ${n}")
        params.append(synced)
        n += 1
    return wheres, params


async def list_models(
    conn,
    *,
    status: str | None = None,
    purpose: str | None = None,
    symbol: str | None = None,
    synced: bool | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    """List ``model_registry`` rows with optional filters. Ordered by
    ``created_time DESC`` so the operator's "what's new since last
    check?" query is the natural default.

    Filters compose with AND. ``synced=False`` is useful operationally
    — answers "what's registered but not yet on VPS?" The operator
    rsyncs those, then POSTs ``/models/{id}/mark-synced`` for each.
    """
    wheres, params = _build_filters(
        status=status, purpose=purpose, symbol=symbol, synced=synced,
    )
    n = len(params) + 1
    params += [limit, offset]
    rows = await conn.fetch(
        f"""
        SELECT id, family, purpose, symbol, interval, horizon_bars,
               feature_set, hyperparams, metrics,
               random_seed, artifact_uri, artifact_sha256, artifact_size_bytes,
               artifact_synced_to_vps, artifact_synced_at,
               status, version, created_time, created_by, updated_time
        FROM model_registry
        WHERE {' AND '.join(wheres)}
        ORDER BY created_time DESC
        LIMIT ${n} OFFSET ${n + 1}
        """,
        *params,
    )
    return [dict(r) for r in rows]


async def count_models(
    conn,
    *,
    status: str | None = None,
    purpose: str | None = None,
    symbol: str | None = None,
    synced: bool | None = None,
) -> int:
    """Mirror of ``list_models`` filters for paginated total."""
    wheres, params = _build_filters(
        status=status, purpose=purpose, symbol=symbol, synced=synced,
    )
    row = await conn.fetchrow(
        f"SELECT COUNT(*) AS n FROM model_registry WHERE {' AND '.join(wheres)}",
        *params,
    )
    return int(row["n"])


async def mark_synced(
    conn,
    *,
    model_id: UUID,
    remote_path: str | None,
    synced_at,
    actor: str,
) -> dict | None:
    """Flip ``artifact_synced_to_vps=true`` + set ``artifact_synced_at``
    on one row. Optionally update ``artifact_uri`` to the new remote
    path (the on-VPS location). Returns the updated row or None if
    ``model_id`` didn't exist.

    Idempotent: re-marking a synced model just refreshes
    ``artifact_synced_at`` (latest wins). ``updated_by`` records who
    flipped the flag for audit; ``updated_time = NOW()`` keeps the
    audit-trail column in sync (without this, two distinct mark-synced
    events on the same row are indistinguishable in
    ``updated_time``).

    SQL is built from a single template with the ``artifact_uri`` clause
    appended only when ``remote_path`` is supplied — keeps the two paths
    on one statement so a future SET-clause addition lands once.
    """
    set_clauses = [
        "artifact_synced_to_vps = TRUE",
        "artifact_synced_at = $2",
    ]
    params: list[Any] = [model_id, synced_at]
    if remote_path is not None:
        params.append(remote_path)
        set_clauses.append(f"artifact_uri = ${len(params)}")
    params.append(actor)
    set_clauses.append(f"updated_by = ${len(params)}")
    set_clauses.append("updated_time = NOW()")

    sql = f"""
        UPDATE model_registry
        SET {', '.join(set_clauses)}
        WHERE id = $1
        RETURNING id, artifact_synced_to_vps, artifact_synced_at, artifact_uri, status
    """
    row = await conn.fetchrow(sql, *params)
    return dict(row) if row else None
