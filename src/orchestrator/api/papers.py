"""``/papers`` — research paper library.

Backed by ``research_paper`` + ``research_paper_trade_data`` (V118).
One paper per strategy × symbol × interval arc. Papers are generated
(or refreshed) by ``POST /papers/{queue_id}/generate``; the /tick
service calls this automatically when a queue reaches COMPLETED or PARKED.

Endpoints
---------
GET  /papers                      — paginated library index
GET  /papers/{paper_id}           — full paper (all sections)
GET  /papers/{paper_id}/chart-data — equity curve + trade list (heavy JSONB)
POST /papers/{queue_id}/generate  — trigger paper generation / refresh
GET  /papers/{paper_id}/export/latex — download compilable .tex source
"""

from __future__ import annotations

import subprocess
import tempfile
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from uuid import UUID

import asyncpg
from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response

from .json_response import TypedJSONResponse

from ..errors import OrchestratorError
from ..repo import papers as papers_repo
from ..services.paper_generator import build_paper_sections, generate_paper
from ..services.latex_export import render_latex
from .deps import get_agent_name, get_db_conn
from .pagination import Page, decode_cursor_ext, encode_cursor_ext

router = APIRouter(prefix="/papers", tags=["papers"])

_VALID_STATUS = {"WORKING_PAPER", "FINALIZED"}


# ---------------------------------------------------------------------------
# GET /papers — library index
# ---------------------------------------------------------------------------

_VALID_SORT_BY = {
    "annualized_return_pct", "profit_factor", "win_rate",
    "n_trades", "sharpe_ratio", "created_time", "updated_time",
}
_TIMESTAMP_SORTS = {"created_time", "updated_time"}


@router.get("", response_model=Page)
async def list_papers(
    paper_status: str | None = Query(
        None,
        description="Filter by status: WORKING_PAPER | FINALIZED",
    ),
    strategy_code: str | None = Query(None),
    instrument: str | None = Query(None, description="e.g. BTCUSDT"),
    interval_name: str | None = Query(None, description="e.g. 4H, 1H, 15m"),
    sort_by: str = Query(
        "created_time",
        description=(
            "Sort field: annualized_return_pct | profit_factor | win_rate "
            "| n_trades | sharpe_ratio | created_time"
        ),
    ),
    sort_dir: str = Query("desc", description="asc | desc"),
    cursor: str | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    conn: asyncpg.Connection = Depends(get_db_conn),
) -> Page:
    if paper_status is not None and paper_status not in _VALID_STATUS:
        raise OrchestratorError(
            status_code=400,
            error_code="bad_paper_status",
            message=f"paper_status must be one of {sorted(_VALID_STATUS)}.",
            retryable=False,
        )
    if sort_by not in _VALID_SORT_BY:
        raise OrchestratorError(
            status_code=400,
            error_code="bad_sort_by",
            message=f"sort_by must be one of {sorted(_VALID_SORT_BY)}.",
            retryable=False,
        )
    sort_dir = sort_dir.lower()
    if sort_dir not in ("asc", "desc"):
        raise OrchestratorError(
            status_code=400,
            error_code="bad_sort_dir",
            message="sort_dir must be 'asc' or 'desc'.",
            retryable=False,
        )

    after_created_time: datetime | None = None
    after_id: str | None = None
    after_metric_val: Decimal | None = None
    has_metric_cursor: bool = False
    if cursor is not None:
        decoded = decode_cursor_ext(cursor)
        raw_t = decoded.get("t")
        after_created_time = datetime.fromisoformat(raw_t) if raw_t else None
        after_id = decoded.get("id")
        # Metric sort cursors encode the sort column value under "sv".
        # Use Decimal (not float) so asyncpg maps it to NUMERIC and avoids
        # float64 precision errors when comparing against NUMERIC columns.
        if "sv" in decoded:
            raw_sv = decoded["sv"]
            if raw_sv is not None:
                try:
                    after_metric_val = Decimal(str(raw_sv))
                except InvalidOperation:
                    after_metric_val = None
            has_metric_cursor = True

    rows = await papers_repo.list_papers(
        conn,
        paper_status=paper_status,
        strategy_code=strategy_code,
        instrument=instrument,
        interval_name=interval_name,
        after_created_time=after_created_time,
        after_id=after_id,
        after_metric_val=after_metric_val,
        has_metric_cursor=has_metric_cursor,
        limit=limit + 1,
        sort_by=sort_by,
        sort_dir=sort_dir,
    )

    has_more = len(rows) > limit
    items = rows[:limit]
    next_cursor: str | None = None
    if has_more:
        last = items[-1]
        if sort_by in _TIMESTAMP_SORTS:
            # Timestamp sorts: store the sort column value in "t" so the repo
            # can use (sort_col, paper_id) as the keyset.
            ts_val = last.get(sort_by)
            cursor_data: dict[str, Any] = {
                "t": ts_val.isoformat() if ts_val else last["created_time"].isoformat(),
                "id": str(last["paper_id"]),
            }
        else:
            # Metric sorts: store created_time as tie-breaker "t" and the
            # sort column value as "sv" (Decimal-safe via str() encoding).
            raw_sv = last.get(sort_by)
            cursor_data = {
                "t": last["created_time"].isoformat(),
                "id": str(last["paper_id"]),
                "sv": float(raw_sv) if raw_sv is not None else None,
            }
        next_cursor = encode_cursor_ext(cursor_data)

    next_actions: list[dict[str, Any]] = []
    if has_more:
        next_actions.append(
            {"kind": "call", "method": "GET", "path": f"/papers?cursor={next_cursor}"}
        )

    return Page(items=items, next_cursor=next_cursor, next_actions=next_actions)


# ---------------------------------------------------------------------------
# GET /papers/{paper_id} — full paper
# ---------------------------------------------------------------------------

@router.get("/{paper_id}")
async def get_paper(
    paper_id: str,
    conn: asyncpg.Connection = Depends(get_db_conn),
) -> dict[str, Any]:
    paper = await papers_repo.get_paper(conn, paper_id)
    if paper is None:
        raise OrchestratorError(
            status_code=404,
            error_code="paper_not_found",
            message=f"No research_paper with paper_id={paper_id!r}.",
            retryable=False,
            hint="Use GET /papers to list available papers.",
        )

    enriched = await build_paper_sections(conn, paper)

    return {
        **enriched,
        "next_actions": [
            {
                "kind": "call",
                "method": "GET",
                "path": f"/papers/{paper_id}/chart-data",
                "description": "Fetch equity curve + trades for chart rendering.",
            },
            {
                "kind": "call",
                "method": "GET",
                "path": f"/papers/{paper_id}/export/latex",
                "description": "Download compilable LaTeX source.",
            },
        ],
    }


# ---------------------------------------------------------------------------
# GET /papers/{paper_id}/chart-data
# ---------------------------------------------------------------------------

@router.get("/{paper_id}/chart-data")
async def get_chart_data(
    paper_id: str,
    conn: asyncpg.Connection = Depends(get_db_conn),
) -> dict[str, Any]:
    paper = await papers_repo.get_paper(conn, paper_id)
    if paper is None:
        raise OrchestratorError(
            status_code=404,
            error_code="paper_not_found",
            message=f"No research_paper with paper_id={paper_id!r}.",
            retryable=False,
        )

    data = await papers_repo.get_chart_data(conn, paper_id)
    if data is None:
        raise OrchestratorError(
            status_code=404,
            error_code="chart_data_not_found",
            message=(
                f"No chart data stored for paper {paper_id!r}. "
                "The paper may still be generating."
            ),
            retryable=True,
            hint="Retry after POST /papers/{queue_id}/generate completes.",
        )

    return {
        "paper_id": paper_id,
        "iteration_id": str(data["iteration_id"]),
        "equity_curve": data["equity_curve"],
        "trades": data["trades"],
        "wf_equity_curve": data["wf_equity_curve"],
        "stored_at": data["created_time"],
    }


# ---------------------------------------------------------------------------
# POST /papers/{queue_id}/generate — trigger / refresh
# ---------------------------------------------------------------------------

@router.post("/{queue_id}/generate")
async def generate(
    queue_id: UUID,
    conn: asyncpg.Connection = Depends(get_db_conn),
    agent: str = Depends(get_agent_name),
) -> dict[str, Any]:
    """Generate or refresh the research paper for a completed/parked sweep.

    Idempotent — safe to call multiple times. Each call increments
    ``version`` so the frontend can detect when to invalidate its cache.
    Returns 201 on first creation, 200 on refresh.
    """
    paper = await generate_paper(conn, queue_id, agent)
    paper_id = paper["paper_id"]
    # version == 1 → first creation (201); version > 1 → refresh (200)
    status_code = 201 if paper.get("version", 1) == 1 else 200
    body = {
        **paper,
        "next_actions": [
            {
                "kind": "call",
                "method": "GET",
                "path": f"/papers/{paper_id}",
                "description": "Read the full paper.",
            },
            {
                "kind": "call",
                "method": "GET",
                "path": f"/papers/{paper_id}/export/latex",
                "description": "Download LaTeX source for journal submission.",
            },
        ],
    }
    return TypedJSONResponse(content=body, status_code=status_code)


# ---------------------------------------------------------------------------
# GET /papers/{paper_id}/export/latex — LaTeX download
# ---------------------------------------------------------------------------

@router.get("/{paper_id}/export/latex")
async def export_latex(
    paper_id: str,
    conn: asyncpg.Connection = Depends(get_db_conn),
) -> Response:
    """Return a compilable .tex file for the paper.

    The file is a complete ``article``-class LaTeX document that compiles
    with pdflatex / xelatex without modification. For venue-specific
    templates (IEEE, ACM, Springer LNCS) copy the section bodies into
    the venue skeleton and add the provided bibliography entries.
    """
    paper_row = await papers_repo.get_paper(conn, paper_id)
    if paper_row is None:
        raise OrchestratorError(
            status_code=404,
            error_code="paper_not_found",
            message=f"No research_paper with paper_id={paper_id!r}.",
            retryable=False,
        )

    paper_data = await build_paper_sections(conn, paper_row)
    latex_source = render_latex(paper_data)
    filename = f"{paper_id}.tex"

    return Response(
        content=latex_source.encode("utf-8"),
        media_type="application/x-tex",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Paper-Version": str(paper_row.get("version", 1)),
        },
    )


# ---------------------------------------------------------------------------
# GET /papers/{paper_id}/export/pdf — PDF compilation
# ---------------------------------------------------------------------------

@router.get("/{paper_id}/export/pdf")
async def export_pdf(
    paper_id: str,
    conn: asyncpg.Connection = Depends(get_db_conn),
) -> Response:
    """Return a compiled PDF of the research paper via pdflatex.

    Compiles the LaTeX source to PDF using pdflatex in a temporary directory.
    If pdflatex is not available or compilation fails, returns an error.
    """
    paper_row = await papers_repo.get_paper(conn, paper_id)
    if paper_row is None:
        raise OrchestratorError(
            status_code=404,
            error_code="paper_not_found",
            message=f"No research_paper with paper_id={paper_id!r}.",
            retryable=False,
        )

    paper_data = await build_paper_sections(conn, paper_row)
    latex_source = render_latex(paper_data)

    with tempfile.TemporaryDirectory() as tmpdir:
        tex_name = f"{paper_id}.tex"
        tex_path = Path(tmpdir) / tex_name
        tex_path.write_text(latex_source, encoding="utf-8")
        pdf_path = Path(tmpdir) / f"{paper_id}.pdf"

        # Two passes — first builds .aux, second resolves \ref{tab:...}
        # cross-references.  Each pass capped at 60s; total wall-time
        # bounded at 120s.  -halt-on-error left OFF so non-fatal warnings
        # (overfull boxes, unresolved refs on pass 1) don't abort.
        proc = None
        try:
            for _ in range(2):
                proc = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", tex_name],
                    cwd=tmpdir,
                    capture_output=True,
                    timeout=60,
                )
                if proc.returncode != 0:
                    break
        except FileNotFoundError:
            raise OrchestratorError(
                status_code=503,
                error_code="pdflatex_not_available",
                message="pdflatex is not available on this server.",
                retryable=False,
            )
        except subprocess.TimeoutExpired:
            raise OrchestratorError(
                status_code=504,
                error_code="pdf_compilation_timeout",
                message="PDF compilation timed out after 60 seconds.",
                retryable=True,
            )

        if not pdf_path.exists():
            # pdflatex writes its real diagnostics to STDOUT (the .log file
            # mirror); stderr is usually empty. Surface stdout + .log tail
            # so the operator can see the actual LaTeX error.
            stdout_tail = proc.stdout.decode("utf-8", errors="replace")[-1500:] if proc else ""
            log_path = Path(tmpdir) / f"{paper_id}.log"
            log_tail = (
                log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
                if log_path.exists()
                else ""
            )
            raise OrchestratorError(
                status_code=500,
                error_code="pdf_compilation_failed",
                message="PDF compilation failed.",
                retryable=False,
                details={"pdflatex_stdout": stdout_tail, "pdflatex_log": log_tail},
            )

        pdf_bytes = pdf_path.read_bytes()

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{paper_id}.pdf"',
            "X-Paper-Version": str(paper_row.get("version", 1)),
            "Cache-Control": "no-store",
        },
    )
