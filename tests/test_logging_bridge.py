"""H7 regression (2026-06-12): structlog must route through the stdlib
logging tree so ERROR-level events reach handlers attached to the root
logger — in production that is ErrorReporterLoggingHandler (error_log +
Telegram). The old PrintLoggerFactory(stdout) wiring bypassed stdlib
entirely: every ``log.error`` from the tick pipeline / JVM clients /
background tasks printed JSON and alerted nobody.
"""

from __future__ import annotations

import json
import logging

import structlog

from orchestrator.config import Settings
from orchestrator.logging import configure_logging, get_logger
from orchestrator.observability.log_handler import ErrorReporterLoggingHandler


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _settings(**overrides) -> Settings:
    base = dict(
        profile="dev",
        db_dsn="postgresql://x:y@localhost:5432/test",
        auth_token="dev-sentinel-not-for-prod",
    )
    base.update(overrides)
    return Settings(**base)


def test_structlog_error_reaches_root_stdlib_handlers(capsys) -> None:
    configure_logging(_settings())
    root = logging.getLogger()
    cap = _Capture()
    root.addHandler(cap)
    try:
        log = get_logger("orchestrator.services.test_bridge")
        log.error("bridge_test_event", queue_id="q-123")
    finally:
        root.removeHandler(cap)

    assert len(cap.records) == 1, (
        "structlog .error() did not traverse the stdlib root logger — "
        "the ErrorReporter would never see it (H7 regression)"
    )
    rec = cap.records[0]
    assert rec.levelno == logging.ERROR
    assert rec.name == "orchestrator.services.test_bridge"
    # The stdout line must still be one JSON object with the stable keys.
    out_lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    payload = json.loads(out_lines[-1])
    assert payload["event"] == "bridge_test_event"
    assert payload["level"] == "error"
    assert payload["queue_id"] == "q-123"
    assert "ts" in payload


def test_error_reporter_handler_attached_to_root() -> None:
    configure_logging(_settings())
    root = logging.getLogger()
    assert any(
        isinstance(h, ErrorReporterLoggingHandler) for h in root.handlers
    )


def test_foreign_stdlib_records_still_render_json(capsys) -> None:
    configure_logging(_settings())
    logging.getLogger("uvicorn.error").error("plain stdlib message")
    out_lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    payload = json.loads(out_lines[-1])
    assert payload["event"] == "plain stdlib message"
    assert payload["level"] == "error"
