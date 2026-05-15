"""Structured error envelope.

Every non-2xx response — from validation failures, our own raises, or
unhandled exceptions — is serialized through ``ErrorEnvelope``. The shape
is stable so the quant-researcher agent can branch on ``error_code`` and
``retryable`` without parsing prose.

``hint`` is for the agent to read; ``next_action`` is a structured pointer
the agent can act on (a follow-up endpoint, a doc anchor, a wait + retry).
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

_unhandled_logger = logging.getLogger(__name__)


class NextAction(BaseModel):
    """A machine-readable suggestion for what the agent should do next.

    ``kind`` is the discriminator. Concrete shapes:

    - ``retry``        — call the same endpoint again, optionally after ``wait_s``
    - ``call``         — call ``method`` ``path`` (e.g. GET /agent/state)
    - ``read_doc``     — read ``doc_anchor`` (key into /agent/playbook)
    - ``contact_human``— escalate; the agent should not retry blindly
    """

    kind: Literal["retry", "call", "read_doc", "contact_human"]
    method: str | None = None
    path: str | None = None
    wait_s: float | None = None
    doc_anchor: str | None = None


class ErrorEnvelope(BaseModel):
    """Stable error shape every non-2xx response uses.

    The agent's contract: branch on ``error_code`` (never on HTTP status
    alone) and on ``retryable`` (never on prose).
    """

    error_code: str = Field(
        ...,
        description="Stable, snake_case identifier. Safe to switch on.",
        examples=["auth_missing_token", "validation_failed", "jvm_unreachable"],
    )
    message: str = Field(..., description="Human-readable summary; do not parse.")
    retryable: bool = Field(..., description="True if the same call may succeed later.")
    hint: str | None = Field(None, description="Agent-readable advice on what went wrong.")
    next_action: NextAction | None = None
    details: dict[str, Any] | None = Field(
        None, description="Optional structured context (validation paths, IDs, etc.)."
    )


class OrchestratorError(Exception):
    """Raise from handlers when you want a specific envelope."""

    def __init__(
        self,
        *,
        status_code: int,
        error_code: str,
        message: str,
        retryable: bool = False,
        hint: str | None = None,
        next_action: NextAction | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.envelope = ErrorEnvelope(
            error_code=error_code,
            message=message,
            retryable=retryable,
            hint=hint,
            next_action=next_action,
            details=details,
        )


def _envelope_response(status_code: int, env: ErrorEnvelope) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=env.model_dump(exclude_none=True))


def _sanitize_validation_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip non-JSON-serializable values from pydantic ``errors()`` output.

    When a ``@field_validator`` raises ``ValueError(...)``, pydantic v2
    attaches the raw exception object under ``ctx.error``. That breaks
    ``JSONResponse`` rendering (the exception isn't JSON-serializable)
    and surfaces as an unhandled 500 from inside the 422 handler — a
    confusing failure mode where a validator that should reject with
    422 explodes with 500 instead. We drop the raw exception but keep
    its message (already present in the top-level ``msg`` field) so
    the response remains informative without crashing the encoder.
    """
    out: list[dict[str, Any]] = []
    for e in errors:
        e = dict(e)
        ctx = e.get("ctx")
        if isinstance(ctx, dict) and "error" in ctx:
            e["ctx"] = {k: v for k, v in ctx.items() if k != "error"}
        out.append(e)
    return out


def register_exception_handlers(app: FastAPI) -> None:
    """Wire every exception path through ``ErrorEnvelope``.

    The agent sees one shape regardless of whether the failure was
    validation (422), auth (401), our own raise, or an unhandled crash (500).
    """

    @app.exception_handler(OrchestratorError)
    async def _orch_error(_: Request, exc: OrchestratorError) -> JSONResponse:
        return _envelope_response(exc.status_code, exc.envelope)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        return _envelope_response(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            ErrorEnvelope(
                error_code="validation_failed",
                message="Request body or query parameters failed validation.",
                retryable=False,
                hint="Fix the fields listed in details.errors and resubmit.",
                next_action=NextAction(kind="read_doc", doc_anchor="errors.validation_failed"),
                details={"errors": _sanitize_validation_errors(exc.errors())},
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        # Map plain HTTPException raises into the same envelope. Most of our
        # handlers should raise OrchestratorError instead; this is the safety net.
        return _envelope_response(
            exc.status_code,
            ErrorEnvelope(
                error_code=f"http_{exc.status_code}",
                message=str(exc.detail) if exc.detail else "HTTP error.",
                retryable=exc.status_code in (429, 502, 503, 504),
            ),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Last resort. logger.exception lands on the
        # ErrorReporterLoggingHandler installed by configure_logging(),
        # which ships the row through the reporter — single path, single
        # row per fingerprint. Don't call reporter.report_exception_async
        # here too: that would compute a different logger_name and write
        # a second row for the same crash.
        _unhandled_logger.exception(
            "unhandled exception in %s %s",
            request.method,
            request.url.path,
        )
        # Envelope is intentionally opaque — agents must not retry on
        # unknown 500s without a hint.
        return _envelope_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            ErrorEnvelope(
                error_code="internal_error",
                message="The service crashed handling this request.",
                retryable=False,
                hint="Check orchestrator logs for the traceback.",
                next_action=NextAction(kind="contact_human"),
                details={"exception_class": type(exc).__name__},
            ),
        )
