"""research-orchestrator → trading JVM error reporter.

POSTs to ``/api/v1/errors`` with ``source=research-orch`` so unhandled
exceptions, FastAPI 500s, and ERROR-level log records all land in the
shared ``error_log`` table. Same dedup, same severity classifier, same
Telegram alerting that backs the JVM-side ``DbErrorAppender``.

Design:

- Uses ``httpx.AsyncClient`` for the actual POST. The reporter exposes a
  sync ``report`` entrypoint and an async ``report_async`` so both
  contexts (logging handlers — sync; FastAPI exception handlers — async)
  call without ceremony.
- In-flight dedup latch (fingerprint → expiry) so a tight error loop in
  the tick thread doesn't fire 60 reports/sec for the same payload.
  Server-side dedup is authoritative; this is just network courtesy.
- Never raises. Failures are demoted to a single ``logger.warning`` line
  so a broken endpoint can't take the orchestrator down. Critically,
  the reporter NEVER calls ``logger.error`` — that would re-enter the
  logging handler that called us.
- No env URL → no-op. ``error_ingest_url`` must be set explicitly so a
  misconfigured deploy doesn't silently spray to whatever happens to be
  on localhost:8080.

The fingerprint hash is SHA-256 of (loggerName + exceptionClass + first
5 stack lines) — same shape as the JVM-side ``DbErrorAppender`` and the
controller's server-side fallback, so identical errors across services
collide on a single row when their logger names match.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
import traceback
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_DEDUP_TTL_S = 60.0
_MESSAGE_CHAR_LIMIT = 5_000
_STACK_CHAR_LIMIT = 16_000
_FINGERPRINT_STACK_LINES = 5
_DEFAULT_LOGGER = "orch.unknown"
_SOURCE = "research-orch"


class ErrorReporter:
    """Singleton-style reporter. One instance per app, set during lifespan."""

    def __init__(self, *, ingest_url: str | None, timeout_s: float = 5.0) -> None:
        self._ingest_url = ingest_url
        self._timeout_s = timeout_s
        self._inflight: dict[str, float] = {}
        self._lock = threading.Lock()
        # Lazily-built async client. We don't open it eagerly because the
        # event loop may not exist at construction time (this can be built
        # before the FastAPI lifespan opens).
        self._async_client: httpx.AsyncClient | None = None
        self._sync_client: httpx.Client | None = None

    @property
    def enabled(self) -> bool:
        return bool(self._ingest_url)

    async def aclose(self) -> None:
        if self._async_client is not None:
            try:
                await self._async_client.aclose()
            except Exception:  # noqa: BLE001 — close failures don't matter
                pass
        if self._sync_client is not None:
            try:
                self._sync_client.close()
            except Exception:  # noqa: BLE001
                pass

    # ── Public API ──────────────────────────────────────────────────────

    def report(
        self,
        *,
        message: str,
        logger_name: str | None = None,
        level: str = "ERROR",
        exception_class: str | None = None,
        stack_trace: str | None = None,
        fingerprint: str | None = None,
        mdc: dict[str, str] | None = None,
    ) -> None:
        """Sync entrypoint. Used by the logging handler. Fire-and-forget —
        spawns a background thread for the HTTP POST so the caller never
        blocks waiting for the trading JVM."""
        if not self.enabled:
            return
        body = self._build_body(
            message=message,
            logger_name=logger_name,
            level=level,
            exception_class=exception_class,
            stack_trace=stack_trace,
            fingerprint=fingerprint,
            mdc=mdc,
        )
        if body is None:
            return
        thread = threading.Thread(target=self._post_sync, args=(body,), daemon=True)
        thread.start()

    async def report_async(
        self,
        *,
        message: str,
        logger_name: str | None = None,
        level: str = "ERROR",
        exception_class: str | None = None,
        stack_trace: str | None = None,
        fingerprint: str | None = None,
        mdc: dict[str, str] | None = None,
    ) -> None:
        """Async entrypoint. Used by FastAPI exception handlers."""
        if not self.enabled:
            return
        body = self._build_body(
            message=message,
            logger_name=logger_name,
            level=level,
            exception_class=exception_class,
            stack_trace=stack_trace,
            fingerprint=fingerprint,
            mdc=mdc,
        )
        if body is None:
            return
        try:
            client = self._get_async_client()
            await client.post(self._ingest_url or "", json=body)
        except Exception as exc:  # noqa: BLE001 — never raise out of reporter
            logger.warning(
                "[error-reporter] async POST failed: %s",
                exc.__class__.__name__,
            )

    def report_exception(
        self,
        exc: BaseException,
        *,
        logger_name: str | None = None,
        mdc: dict[str, str] | None = None,
    ) -> None:
        """Convenience: stack-format an exception and route to ``report``."""
        self.report(
            message=str(exc) or exc.__class__.__name__,
            logger_name=logger_name,
            exception_class=exc.__class__.__name__,
            stack_trace="".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ),
            mdc=mdc,
        )

    async def report_exception_async(
        self,
        exc: BaseException,
        *,
        logger_name: str | None = None,
        mdc: dict[str, str] | None = None,
    ) -> None:
        await self.report_async(
            message=str(exc) or exc.__class__.__name__,
            logger_name=logger_name,
            exception_class=exc.__class__.__name__,
            stack_trace="".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ),
            mdc=mdc,
        )

    # ── Internals ───────────────────────────────────────────────────────

    def _build_body(
        self,
        *,
        message: str,
        logger_name: str | None,
        level: str,
        exception_class: str | None,
        stack_trace: str | None,
        fingerprint: str | None,
        mdc: dict[str, str] | None,
    ) -> dict[str, Any] | None:
        msg = _truncate(message, _MESSAGE_CHAR_LIMIT)
        if not msg:
            return None
        stack = _truncate(stack_trace, _STACK_CHAR_LIMIT)
        logger_ = logger_name or _DEFAULT_LOGGER
        fp = fingerprint or _compute_fingerprint(logger_, exception_class, stack)

        if not self._claim_fingerprint(fp):
            return None

        return {
            "source": _SOURCE,
            "loggerName": logger_,
            "level": level,
            "message": msg,
            "exceptionClass": exception_class,
            "stackTrace": stack,
            "fingerprint": fp,
            "mdc": _build_mdc(mdc),
        }

    def _claim_fingerprint(self, fp: str) -> bool:
        """Returns True if we should send (cache miss); False to skip (recent dup)."""
        now = time.monotonic()
        with self._lock:
            if len(self._inflight) >= 64:
                self._inflight = {k: v for k, v in self._inflight.items() if v > now}
            expiry = self._inflight.get(fp)
            if expiry is not None and expiry > now:
                return False
            self._inflight[fp] = now + _DEDUP_TTL_S
            return True

    def _post_sync(self, body: dict[str, Any]) -> None:
        try:
            client = self._get_sync_client()
            client.post(self._ingest_url or "", json=body)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[error-reporter] sync POST failed: %s",
                exc.__class__.__name__,
            )

    def _get_async_client(self) -> httpx.AsyncClient:
        # Lock prevents two concurrent callers from each constructing a
        # client and leaking the loser's connection pool until GC runs.
        with self._lock:
            if self._async_client is None:
                self._async_client = httpx.AsyncClient(
                    timeout=self._timeout_s,
                    headers={"Content-Type": "application/json"},
                )
            return self._async_client

    def _get_sync_client(self) -> httpx.Client:
        with self._lock:
            if self._sync_client is None:
                self._sync_client = httpx.Client(
                    timeout=self._timeout_s,
                    headers={"Content-Type": "application/json"},
                )
            return self._sync_client


# ── Helpers ─────────────────────────────────────────────────────────────


def _compute_fingerprint(
    logger_name: str, exception_class: str | None, stack: str | None
) -> str:
    parts = [logger_name or "?", exception_class or "?"]
    if stack:
        lines = [ln.strip() for ln in stack.splitlines()[:_FINGERPRINT_STACK_LINES]]
        parts.append("|".join(lines))
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()


def _truncate(s: str | None, max_len: int) -> str | None:
    if s is None:
        return None
    if len(s) <= max_len:
        return s
    return f"{s[:max_len]}\n…[truncated]"


def _build_mdc(extra: dict[str, str] | None) -> dict[str, str]:
    base: dict[str, str] = {
        "service": "research-orchestrator",
        "pid": str(os.getpid()),
    }
    if extra:
        for k, v in extra.items():
            if v is None:
                continue
            s = v if isinstance(v, str) else str(v)
            base[k] = s if len(s) <= 500 else f"{s[:500]}…"
    return base


# ── Module-level singleton ──────────────────────────────────────────────

_reporter: ErrorReporter | None = None


def init_reporter(*, ingest_url: str | None, timeout_s: float = 5.0) -> ErrorReporter:
    """Idempotent. Returns the existing reporter on subsequent calls so
    repeat invocations of ``configure_logging`` (tests, hot reload) don't
    leak the previous reporter's httpx client pool. Tests that need to
    swap the reporter must call ``reset_reporter()`` explicitly first."""
    global _reporter
    if _reporter is None:
        _reporter = ErrorReporter(ingest_url=ingest_url, timeout_s=timeout_s)
    return _reporter


def reset_reporter() -> None:
    """Drop the module-level reporter. Httpx clients are best-effort closed
    via ``aclose`` if you are inside an event loop; this helper does not
    await — call ``await reporter.aclose()`` first if you care about the
    connection pool draining cleanly. Intended for test fixtures."""
    global _reporter
    _reporter = None


def get_reporter() -> ErrorReporter | None:
    return _reporter
