"""Python ``logging.Handler`` that ships ERROR-level records through the
shared error_reporter pipeline.

This is the orchestrator's analogue of the JVM's ``DbErrorAppender``: any
``logger.error(...)`` (or ``logger.exception(...)``) anywhere in the
service — repos, services, lifespans, FastAPI's own loggers — funnels into
``error_log`` with ``source=research-orch``.

Wired into the root logger by ``configure_logging`` so it covers every
loggerName, including ``uvicorn`` and ``asyncpg``. The reporter itself
guards against re-entry (its own warnings go through
``logger.warning`` not ``error``), so this can never recurse on its own
failure path.
"""

from __future__ import annotations

import json
import logging
import traceback

from .error_reporter import ErrorReporter


class ErrorReporterLoggingHandler(logging.Handler):
    """Forwards ERROR/CRITICAL records to the trading-JVM error_log."""

    def __init__(self, reporter: ErrorReporter) -> None:
        # WARNING is intentionally excluded — only ERROR+ rows are persisted
        # by the trading JVM's ingest service, so shipping WARNs would just
        # be wasted requests.
        super().__init__(level=logging.ERROR)
        self._reporter = reporter

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        # Be paranoid in this handler — ANY exception escaping `emit` would
        # be logged by Python's logging framework itself and could create
        # a loop if our reporter emitted another error.
        try:
            if not self._reporter.enabled:
                return

            # Don't recurse on our own warnings/errors.
            if record.name.startswith("orchestrator.observability"):
                return
            if record.name.startswith("httpx") or record.name.startswith("httpcore"):
                # The reporter uses httpx; if its lib logs an error during
                # our POST, shipping that would loop straight back.
                return

            exc_class: str | None = None
            stack: str | None = None
            if isinstance(record.msg, dict):
                # structlog event routed via wrap_for_formatter: record.msg
                # is the event dict. Extract the event name as the message
                # and the (already-rendered) exception text as the stack.
                event_dict = dict(record.msg)
                stack = event_dict.pop("exception", None)
                event_dict.pop("_record", None)
                event_dict.pop("_from_structlog", None)
                message = json.dumps(event_dict, default=str)
            else:
                message = record.getMessage()
            if record.exc_info:
                exc_type, exc_value, exc_tb = record.exc_info
                if exc_type is not None:
                    exc_class = exc_type.__name__
                stack = "".join(
                    traceback.format_exception(exc_type, exc_value, exc_tb)
                )

            mdc: dict[str, str] = {
                "thread": record.threadName or "?",
                "module": record.module or "?",
                "func": record.funcName or "?",
                "line": str(record.lineno),
            }

            self._reporter.report(
                message=message,
                logger_name=record.name,
                level=record.levelname,
                exception_class=exc_class,
                stack_trace=stack,
                mdc=mdc,
            )
        except Exception:  # noqa: BLE001
            self.handleError(record)
