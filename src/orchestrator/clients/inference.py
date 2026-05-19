"""blackheart-inference sidecar HTTP client.

Mirrors the JvmClient shape: long-lived ``httpx.AsyncClient``, structured
error-envelope passthrough, health probe. The inference sidecar runs
loopback-only on ``:8000`` and the orchestrator is the only legitimate
caller — the researcher reaches inference only through this proxy.

Error-envelope passthrough: when the sidecar returns 4xx/5xx with our
``ErrorEnvelope`` shape (the inference service uses an identical
envelope), we re-raise as ``OrchestratorError`` carrying the same
``error_code`` and ``details`` so callers branch the same way regardless
of which service originated the failure.
"""

from __future__ import annotations

from typing import Any

import httpx

from ..config import Settings
from ..errors import NextAction, OrchestratorError
from ..logging import get_logger

log = get_logger(__name__)


class InferenceClient:
    """Wraps the blackheart-inference sidecar (port 8000)."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: httpx.AsyncClient | None = None

    async def open(self) -> None:
        if self._client is not None:
            return
        self._client = httpx.AsyncClient(
            base_url=self._settings.inference_base_url,
            timeout=self._settings.inference_request_timeout_s,
            transport=httpx.AsyncHTTPTransport(retries=0),
        )

    async def close(self) -> None:
        if self._client is None:
            return
        await self._client.aclose()
        self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("InferenceClient.open() must be called before use.")
        return self._client

    def _headers(self, agent_name: str) -> dict[str, str]:
        return {
            "X-Inference-Token": self._settings.inference_auth_token.get_secret_value(),
            "X-Agent-Name": agent_name,
            "Content-Type": "application/json",
        }

    async def health_probe(self) -> bool:
        try:
            r = await self.client.get("/healthz")
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def run(self, *, body: dict[str, Any], agent_name: str) -> dict[str, Any]:
        return await self._post("/inference/run", body, agent_name)

    async def backfill(self, *, body: dict[str, Any], agent_name: str) -> dict[str, Any]:
        return await self._post("/inference/backfill", body, agent_name)

    async def _post(
        self, path: str, body: dict[str, Any], agent_name: str
    ) -> dict[str, Any]:
        try:
            resp = await self.client.post(path, json=body, headers=self._headers(agent_name))
        except httpx.ConnectError as exc:
            raise OrchestratorError(
                status_code=503,
                error_code="inference_service_unreachable",
                message=(
                    f"blackheart-inference at {self._settings.inference_base_url} "
                    "is not reachable. Is the sidecar running?"
                ),
                retryable=True,
                hint="Start the sidecar with `python -m blackheart_inference`.",
                next_action=NextAction(kind="retry", wait_s=30.0),
            ) from exc
        except httpx.HTTPError as exc:
            raise OrchestratorError(
                status_code=502,
                error_code="inference_transport_error",
                message=f"Transport-level failure talking to inference sidecar: {exc}",
                retryable=True,
                next_action=NextAction(kind="retry", wait_s=30.0),
            ) from exc

        if resp.status_code < 400:
            return resp.json()

        # Forward the sidecar's envelope if it spoke our schema.
        try:
            env = resp.json()
        except ValueError:
            env = {}
        raise OrchestratorError(
            status_code=resp.status_code,
            error_code=env.get("error_code", f"inference_http_{resp.status_code}"),
            message=env.get("message", f"Inference returned HTTP {resp.status_code}."),
            retryable=bool(env.get("retryable", resp.status_code in (429, 502, 503, 504))),
            hint=env.get("hint"),
            details=env.get("details"),
        )
