"""structlog → JSON config.

Every log line is a single JSON object. Fields are stable: ``ts``, ``level``,
``event``, plus whatever the call site bound. Agents tail this and grep on
``event=jvm.unreachable`` rather than parsing prose.

Standard-library loggers are routed through structlog so ``uvicorn`` and
``asyncpg`` emit the same shape.
"""

from __future__ import annotations

import logging
import sys

import structlog

from .config import Settings
from .observability.error_reporter import init_reporter
from .observability.log_handler import ErrorReporterLoggingHandler


def configure_logging(settings: Settings) -> None:
    """Idempotent. Safe to call from both the app factory and tests.

    structlog is routed THROUGH the stdlib logging tree
    (``structlog.stdlib.LoggerFactory`` + ``wrap_for_formatter``), not
    printed directly. The pre-2026-06-12 ``PrintLoggerFactory(stdout)``
    bypassed stdlib entirely, so the ErrorReporterLoggingHandler attached
    to the root logger NEVER saw a structlog ``log.error(...)`` — which is
    nearly every error in the service (tick pipeline, JVM clients,
    background tasks). Errors printed JSON to stdout and alerted nobody.
    """

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True, key="ts")

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            # Hand the event dict to the stdlib record (record.msg = dict);
            # the ProcessorFormatter below renders it to one JSON line.
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.log_level)
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # One stdout handler renders BOTH structlog events and foreign stdlib
    # records (uvicorn, asyncpg) to the same JSON-line shape.
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
            foreign_pre_chain=shared_processors,
        )
    )
    root = logging.getLogger()
    handlers: list[logging.Handler] = [handler]

    # Initialise the error_log reporter and attach a stdlib logging handler
    # that ships every ERROR-level record (ours, uvicorn's, asyncpg's,
    # FastAPI's) to the trading JVM's /api/v1/errors. Empty url → reporter
    # is created but `enabled=False`, so the handler is a no-op — keeps
    # the wiring simple without a feature-flag conditional everywhere.
    reporter = init_reporter(
        ingest_url=settings.error_ingest_url or None,
        timeout_s=settings.error_ingest_timeout_s,
    )
    handlers.append(ErrorReporterLoggingHandler(reporter))

    root.handlers = handlers
    root.setLevel(settings.log_level)

    for noisy in ("uvicorn.access",):
        # Access logs are useful but doubled by our own request middleware.
        # Drop their level a notch to keep the JSON stream clean.
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name) if name else structlog.get_logger()
