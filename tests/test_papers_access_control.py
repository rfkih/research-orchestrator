"""Access-control tests for the papers library.

Regular users may only see FINALIZED ("completed") papers; in-progress
WORKING_PAPER drafts are admin-only. The role is supplied by the trading-JVM
proxy via the ``X-Viewer-Is-Admin`` header (the orchestrator has no user
identity of its own). These tests pin the two pure pieces that enforce it:

- ``get_viewer_is_admin`` — header parsing, fails CLOSED.
- ``_ensure_viewable``    — 404 (not 403) for a non-admin reading a non-final
                            paper, so an in-progress paper isn't discoverable.
"""

from __future__ import annotations

import pytest
from starlette.requests import Request

from orchestrator.api.deps import get_viewer_is_admin
from orchestrator.api.papers import _ensure_viewable
from orchestrator.errors import OrchestratorError


def _request_with_headers(headers: dict[str, str]) -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    return Request({"type": "http", "method": "GET", "path": "/papers",
                    "headers": raw, "query_string": b""})


# ── get_viewer_is_admin — header parsing, fail-closed ────────────────


def test_admin_header_true_is_admin() -> None:
    assert get_viewer_is_admin(_request_with_headers({"X-Viewer-Is-Admin": "true"})) is True


def test_admin_header_is_case_insensitive() -> None:
    # Starlette headers are case-insensitive; value compare is .lower()'d.
    assert get_viewer_is_admin(_request_with_headers({"x-viewer-is-admin": "TRUE"})) is True


@pytest.mark.parametrize("value", ["false", "", "1", "yes", "admin", " true "])
def test_non_true_values_are_not_admin(value: str) -> None:
    # Only a literal "true" (trimmed/lowered) grants admin. " true " trims to
    # "true" → admin; everything else is non-admin. Keep the surprising one honest:
    expected = value.strip().lower() == "true"
    assert get_viewer_is_admin(_request_with_headers({"X-Viewer-Is-Admin": value})) is expected


def test_absent_header_fails_closed_to_non_admin() -> None:
    # No header at all (e.g. a direct loopback call) → non-admin, so a proxy bug
    # can only ever hide papers, never leak in-progress ones.
    assert get_viewer_is_admin(_request_with_headers({})) is False


# ── _ensure_viewable — per-paper read guard ──────────────────────────


def test_non_admin_blocked_from_working_paper() -> None:
    with pytest.raises(OrchestratorError) as exc:
        _ensure_viewable({"paper_status": "WORKING_PAPER"}, is_admin=False, paper_id="p1")
    assert exc.value.status_code == 404
    assert exc.value.envelope.error_code == "paper_not_found"  # indistinguishable from "doesn't exist"


def test_non_admin_allowed_finalized() -> None:
    _ensure_viewable({"paper_status": "FINALIZED"}, is_admin=False, paper_id="p1")  # no raise


def test_admin_allowed_working_paper() -> None:
    _ensure_viewable({"paper_status": "WORKING_PAPER"}, is_admin=True, paper_id="p1")  # no raise


def test_non_admin_blocked_when_status_missing() -> None:
    # Defensive: a row without a status is treated as not-finalized for users.
    with pytest.raises(OrchestratorError):
        _ensure_viewable({}, is_admin=False, paper_id="p1")
