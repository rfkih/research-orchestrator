"""Research-JVM HTTP client.

Owns:
  * the long-lived ``httpx.AsyncClient``
  * the JWT mint flow (``dev_bypass`` → ``/api/v1/dev/login-as``;
    ``service_account`` → ``/api/v1/users/login``)
  * tenacity-driven retries on transport-level failures (NOT on 4xx)
  * a ``health_probe`` for ``/readyz``

Quant-relevant calls (submit backtest, poll, fetch metrics) live on this
class as concrete methods so handlers don't ad-hoc URL strings. Endpoints
beyond ``health_probe`` are stubbed and filled in during Phase 3 (tick).
"""

from __future__ import annotations

from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..config import Settings
from ..errors import NextAction, OrchestratorError
from ..logging import get_logger

log = get_logger(__name__)


class JvmClient:
    """Wraps the research JVM (port 8081)."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: httpx.AsyncClient | None = None
        self._jwt: str | None = None

    async def open(self) -> None:
        if self._client is not None:
            return
        self._client = httpx.AsyncClient(
            base_url=self._settings.jvm_base_url,
            timeout=self._settings.jvm_request_timeout_s,
            # Loopback only — the research JVM is on the same box.
            transport=httpx.AsyncHTTPTransport(retries=0),
        )

    async def close(self) -> None:
        if self._client is None:
            return
        await self._client.aclose()
        self._client = None
        self._jwt = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("JvmClient is not open. Call open() first.")
        return self._client

    async def _mint_jwt(self) -> str:
        """Mint a JWT once per process. Re-mint on 401 from a downstream call."""
        if self._settings.jvm_auth_mode == "static_jwt":
            # Pre-minted static JWT. Used when the research JVM runs without
            # dev-tooling (no /api/v1/dev/login-as endpoint) and no service
            # account credentials are available. Caller is responsible for
            # ensuring the token is valid and not expired.
            static = self._settings.jvm_static_jwt
            if not static:
                raise OrchestratorError(
                    status_code=502,
                    error_code="jvm_auth_failed",
                    message="jvm_auth_mode=static_jwt but ORCH_JVM_STATIC_JWT is not set.",
                    retryable=False,
                    next_action=NextAction(kind="contact_human"),
                )
            return static.get_secret_value()
        if self._settings.jvm_auth_mode == "dev_bypass":
            # Dev backdoor — research JVM exposes /api/v1/dev/login-as on the
            # research profile only. Refused on prod by Settings.assert_prod_safe.
            # Empty body → JVM picks the first user (any user works for
            # research; the orchestrator only reads/writes research tables
            # which have no per-user authorisation). The endpoint's
            # LoginAsRequest DTO has only `email`; sending an unknown field
            # like `username` triggers a Jackson 500.
            payload: dict[str, Any] = {}
            if self._settings.jvm_dev_login_email:
                payload["email"] = self._settings.jvm_dev_login_email
            r = await self.client.post("/api/v1/dev/login-as", json=payload)
        else:
            user = self._settings.jvm_service_user
            pw = self._settings.jvm_service_password
            assert user is not None and pw is not None  # validated at startup
            r = await self.client.post(
                "/api/v1/users/login",
                json={"email": user, "password": pw.get_secret_value()},
            )
        if r.status_code != 200:
            raise OrchestratorError(
                status_code=502,
                error_code="jvm_auth_failed",
                message=f"JVM auth returned {r.status_code}.",
                retryable=False,
                hint=(
                    "If profile=prod check ORCH_JVM_SERVICE_USER/PASSWORD; "
                    "if dev check the research JVM is on with profile=research."
                ),
                next_action=NextAction(kind="contact_human"),
                details={"status": r.status_code, "body": r.text[:500]},
            )
        body = r.json()
        envelope = body.get("data") if isinstance(body.get("data"), dict) else body
        token = envelope.get("token") or envelope.get("accessToken")
        if not token:
            raise OrchestratorError(
                status_code=502,
                error_code="jvm_auth_no_token",
                message="JVM auth response did not include a token field.",
                retryable=False,
            )
        return token

    async def _authed_headers(self) -> dict[str, str]:
        if self._jwt is None:
            self._jwt = await self._mint_jwt()
        return {"Authorization": f"Bearer {self._jwt}"}

    async def health_probe(self) -> bool:
        try:
            # `/actuator/health/{liveness,readiness}` are explicitly carved
            # out of the admin gate in SecurityConfig — both the trading and
            # research JVMs return them unauthenticated. Full /actuator/health
            # is admin-only and would 401 here.
            r = await self.client.get("/actuator/health/liveness")
            return r.status_code == 200
        except (httpx.HTTPError, RuntimeError):
            # RuntimeError = the `client` property guard when the client was
            # never .open()ed. A health probe must report False, never throw
            # (matches Database.health_probe's contract); a caller asking "is
            # the JVM healthy?" before open() should get False, not a crash.
            return False

    async def get(self, path: str, **kwargs: Any) -> httpx.Response:
        return await self._request("GET", path, **kwargs)

    async def post(self, path: str, **kwargs: Any) -> httpx.Response:
        return await self._request("POST", path, **kwargs)

    async def submit_backtest(self, payload: dict[str, Any]) -> str:
        """POST /api/v1/backtest. Returns the new ``backtestRunId``."""
        r = await self.post("/api/v1/backtest", json=payload)
        if r.status_code >= 400:
            raise OrchestratorError(
                status_code=502,
                error_code="jvm_backtest_submit_failed",
                message=f"JVM rejected backtest submission ({r.status_code}).",
                retryable=r.status_code >= 500,
                hint="Check the strategy_code, account_strategy_id, and date range in the queue row.",
                details={"status": r.status_code, "body": r.text[:1000]},
            )
        body = r.json()
        run_id = (
            (body.get("data") or {}).get("backtestRunId")
            or (body.get("data") or {}).get("id")
            or body.get("backtestRunId")
        )
        if not run_id:
            raise OrchestratorError(
                status_code=502,
                error_code="jvm_backtest_no_run_id",
                message="JVM returned 200 but no backtestRunId field.",
                retryable=False,
                details={"body": body},
            )
        return str(run_id)

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        # Retry only transport-level failures; 4xx/5xx from the JVM are
        # business errors and bubble up to the handler so the agent can
        # branch on the envelope.
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            retry=retry_if_exception_type(httpx.TransportError),
            reraise=True,
        ):
            with attempt:
                headers = kwargs.pop("headers", {}) | await self._authed_headers()
                r = await self.client.request(method, path, headers=headers, **kwargs)
                if r.status_code == 401:
                    # Token expired or rotated — mint once and retry on the
                    # next attempt by clearing the cached JWT.
                    self._jwt = None
                    raise httpx.TransportError("JVM 401 — re-mint and retry")
                return r
        # Unreachable — tenacity reraise=True
        raise RuntimeError("JvmClient._request retry loop fell through")
