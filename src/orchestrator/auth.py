"""Auth + agent identity middleware.

Two headers per request:

- ``X-Orch-Token`` (required) — shared secret. Compared with constant-time
  ``secrets.compare_digest``. Missing/wrong → ``auth_*`` envelope, no DB hit.
- ``X-Agent-Name`` (optional) — agent identity, e.g. ``quant-researcher``.
  Defaulted to ``anonymous`` so curl-from-laptop still works during dev.
  Stamped onto ``request.state.agent_name`` for downstream handlers to copy
  into ``created_by`` columns.

Idempotency-Key is parsed here too so handlers can read it off
``request.state`` instead of digging into headers.
"""

from __future__ import annotations

import secrets
from collections.abc import Awaitable, Callable

from fastapi import Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware

from .config import Settings
from .errors import ErrorEnvelope, NextAction

# Endpoints reachable without the shared-secret header. Health probes need
# to work from the systemd readiness check; /agent/playbook is intentionally
# discoverable so a fresh agent can read the contract before authenticating.
_PUBLIC_PATHS = frozenset({"/healthz", "/readyz", "/agent/playbook"})

# Agent name is allow-listed: snake_case alpha + dash, max 64. Prevents log
# injection and stops typos from creating "fingerprint" rows under garbage
# identities.
_MAX_AGENT_NAME_LEN = 64


def _is_valid_agent_name(name: str) -> bool:
    if not name or len(name) > _MAX_AGENT_NAME_LEN:
        return False
    return all(c.isalnum() or c in "-_" for c in name)


class AuthMiddleware(BaseHTTPMiddleware):
    """Validates ``X-Orch-Token`` and pins agent identity to request state."""

    def __init__(self, app, settings: Settings) -> None:
        super().__init__(app)
        self._expected = settings.auth_token.get_secret_value()

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.url.path in _PUBLIC_PATHS:
            request.state.agent_name = "anonymous"
            request.state.idempotency_key = None
            return await call_next(request)

        token = request.headers.get("X-Orch-Token", "")
        if not token:
            return _envelope(
                status.HTTP_401_UNAUTHORIZED,
                ErrorEnvelope(
                    error_code="auth_missing_token",
                    message="X-Orch-Token header is required.",
                    retryable=False,
                    hint="Set the header to ORCH_AUTH_TOKEN from the service env.",
                    next_action=NextAction(kind="read_doc", doc_anchor="auth.shared_secret"),
                ),
            )
        if not secrets.compare_digest(token, self._expected):
            return _envelope(
                status.HTTP_401_UNAUTHORIZED,
                ErrorEnvelope(
                    error_code="auth_bad_token",
                    message="X-Orch-Token did not match the configured secret.",
                    retryable=False,
                    hint="Confirm the agent is reading the same .env the service booted with.",
                    next_action=NextAction(kind="contact_human"),
                ),
            )

        agent = request.headers.get("X-Agent-Name", "anonymous").strip() or "anonymous"
        if not _is_valid_agent_name(agent):
            return _envelope(
                status.HTTP_400_BAD_REQUEST,
                ErrorEnvelope(
                    error_code="auth_bad_agent_name",
                    message="X-Agent-Name must be 1-64 chars of [A-Za-z0-9_-].",
                    retryable=False,
                    hint="Use a stable identifier like 'quant-researcher' or 'operator-curl'.",
                ),
            )
        request.state.agent_name = agent
        request.state.idempotency_key = request.headers.get("Idempotency-Key") or None
        return await call_next(request)


def _envelope(status_code: int, env: ErrorEnvelope) -> Response:
    # Local import to keep the auth module free of FastAPI response classes
    # at import time — middleware is loaded before the app builds.
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=status_code, content=env.model_dump(exclude_none=True))
