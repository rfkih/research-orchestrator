"""Trading-JVM HTTP client (Phase A.6, 2026-05-22).

Distinct from :class:`orchestrator.clients.jvm.JvmClient` which targets
the research JVM (port 8081). This client targets the trading JVM (port
8080) and authenticates via the **shared X-Orch-Token** rather than a
JWT — the trading JVM exposes a small set of orchestrator-internal
endpoints under ``/api/v1/internal/orch/**`` allowlisted by the
:class:`OrchInternalAuthFilter` (Spring filter, validates the token).

Two surfaces today:

* :meth:`get_account_portfolio` — live USDT balance for a given
  ``account_id``. Refreshed from Binance server-side before return so
  the orchestrator's min-notional guard math compares against the same
  number the executor will see when the next signal fires.
* :meth:`get_kelly_status` — current Kelly multiplier for a given
  ``account_strategy_id``. Wraps the trading JVM's
  ``KellySizingService.getStatus`` so the guard math uses the same
  PSR-weighted half-Kelly fraction the executor's ``applyKellySizing``
  multiplies in — no upper-bound assumption, no math drift between
  orchestrator and JVM.

Failure mode: when ``trading_jvm_base_url`` is empty or the trading
JVM is unreachable, the client raises :class:`OrchestratorError` with
``error_code='trading_jvm_unreachable'``. Callers decide whether to
fall back to caller-provided values (the legacy Phase A.5 path) or to
surface the failure to the operator.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from uuid import UUID

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


class TradingJvmPortfolio:
    """Subset of the trading-JVM ``PortfolioBalanceResponse`` the
    orchestrator actually consumes. Decimal-typed so the guard math
    stays exact through to the comparison against ``MIN_USDT_NOTIONAL``.
    """

    __slots__ = ("account_id", "available_usdt", "total_usdt", "locked_usdt")

    def __init__(
        self,
        account_id: UUID | None,
        available_usdt: Decimal,
        total_usdt: Decimal,
        locked_usdt: Decimal,
    ) -> None:
        self.account_id = account_id
        self.available_usdt = available_usdt
        self.total_usdt = total_usdt
        self.locked_usdt = locked_usdt

    def __repr__(self) -> str:  # pragma: no cover — debug only
        return (
            f"TradingJvmPortfolio(account_id={self.account_id}, "
            f"available_usdt={self.available_usdt}, "
            f"total_usdt={self.total_usdt}, "
            f"locked_usdt={self.locked_usdt})"
        )


class TradingJvmKelly:
    """Wire shape of the orchestrator-internal Kelly endpoint. Mirrors
    ``OrchestratorInternalController.KellyResponse``."""

    __slots__ = (
        "account_strategy_id",
        "enabled",
        "current_multiplier",
        "max_fraction",
        "qualifying_runs",
        "reason",
    )

    def __init__(
        self,
        account_strategy_id: UUID,
        enabled: bool,
        current_multiplier: Decimal,
        max_fraction: Decimal | None,
        qualifying_runs: int,
        reason: str,
    ) -> None:
        self.account_strategy_id = account_strategy_id
        self.enabled = enabled
        self.current_multiplier = current_multiplier
        self.max_fraction = max_fraction
        self.qualifying_runs = qualifying_runs
        self.reason = reason


def _coerce_decimal(value: Any, fallback: Decimal = Decimal("0")) -> Decimal:
    """Jackson serialises BigDecimal as a JSON number; httpx hands us a
    Python ``float``. Round-trip via ``str`` keeps the precision sane
    (``Decimal(float_value)`` materialises binary-float noise; via
    ``str`` we get the operator-visible literal).
    """
    if value is None:
        return fallback
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


class TradingJvmClient:
    """Wraps the trading JVM (port 8080) for orchestrator-internal reads."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: httpx.AsyncClient | None = None

    @property
    def enabled(self) -> bool:
        """When ``trading_jvm_base_url`` is empty the client is a no-op.
        Callers branch on this rather than catching exceptions to keep
        the "client disabled" path explicit + fast."""
        return bool(self._settings.trading_jvm_base_url)

    async def open(self) -> None:
        if self._client is not None or not self.enabled:
            return
        self._client = httpx.AsyncClient(
            base_url=self._settings.trading_jvm_base_url,
            timeout=self._settings.trading_jvm_request_timeout_s,
            # Loopback only — pin retries off; tenacity wraps the
            # request-level retry policy below so we can branch on
            # transport vs. business errors.
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
            raise RuntimeError(
                "TradingJvmClient is not open. Call open() first "
                "(typically wired into the FastAPI lifespan)."
            )
        return self._client

    def _auth_headers(self) -> dict[str, str]:
        """X-Orch-Token shared secret. Same value as ``ORCH_AUTH_TOKEN``
        (the orchestrator's inbound token) — the trading JVM's
        ``app.research.orchestrator.token`` property mirrors it."""
        return {"X-Orch-Token": self._settings.auth_token.get_secret_value()}

    async def get_account_portfolio(
        self, account_id: UUID | str
    ) -> TradingJvmPortfolio:
        """Live USDT-balance snapshot for ``account_id``, refreshed from
        Binance server-side. Raises :class:`OrchestratorError` on any
        non-2xx response or transport failure — callers decide the
        fallback policy."""
        path = f"/api/v1/internal/orch/portfolio/{account_id}"
        body = await self._fetch_json(path)
        data = body.get("data") if isinstance(body.get("data"), dict) else body
        return TradingJvmPortfolio(
            account_id=_parse_uuid(data.get("accountId")),
            available_usdt=_coerce_decimal(data.get("availableUsdt")),
            total_usdt=_coerce_decimal(data.get("totalUsdt")),
            locked_usdt=_coerce_decimal(data.get("lockedUsdt")),
        )

    async def get_kelly_status(
        self, account_strategy_id: UUID | str
    ) -> TradingJvmKelly:
        """Current Kelly multiplier + diagnostics for a single
        ``account_strategy`` row. ``current_multiplier == 1.0`` means
        "no adjustment" (Kelly disabled or no qualifying backtest runs);
        anything in [0.05, kelly_max_fraction] means the executor's
        ``applyKellySizing`` will scale entry size by that fraction."""
        path = f"/api/v1/internal/orch/kelly/{account_strategy_id}"
        body = await self._fetch_json(path)
        data = body.get("data") if isinstance(body.get("data"), dict) else body
        return TradingJvmKelly(
            account_strategy_id=_parse_uuid(data.get("accountStrategyId"))
            or UUID(str(account_strategy_id)),
            enabled=bool(data.get("enabled", False)),
            current_multiplier=_coerce_decimal(
                data.get("currentMultiplier"), Decimal("1")
            ),
            max_fraction=(
                _coerce_decimal(data["maxFraction"])
                if data.get("maxFraction") is not None
                else None
            ),
            qualifying_runs=int(data.get("qualifyingRuns") or 0),
            reason=str(data.get("reason") or ""),
        )

    async def _fetch_json(self, path: str) -> dict[str, Any]:
        if not self.enabled:
            raise OrchestratorError(
                status_code=503,
                error_code="trading_jvm_disabled",
                message=(
                    "TradingJvmClient is disabled "
                    "(ORCH_TRADING_JVM_BASE_URL is empty). The "
                    "min-notional guard cannot read live equity/Kelly "
                    "without this client."
                ),
                retryable=False,
                next_action=NextAction(kind="contact_human"),
            )
        # Retry only transport-level failures; 4xx/5xx from the JVM are
        # business errors we surface to the caller so it can branch.
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            retry=retry_if_exception_type(httpx.TransportError),
            reraise=True,
        ):
            with attempt:
                try:
                    r = await self.client.get(path, headers=self._auth_headers())
                except httpx.TransportError:
                    raise
                if r.status_code == 401:
                    raise OrchestratorError(
                        status_code=502,
                        error_code="trading_jvm_unauthorized",
                        message=(
                            "Trading JVM rejected X-Orch-Token. Check "
                            "that ORCH_AUTH_TOKEN matches "
                            "app.research.orchestrator.token on the "
                            "trading JVM."
                        ),
                        retryable=False,
                        next_action=NextAction(kind="contact_human"),
                        details={"path": path, "status": r.status_code},
                    )
                if r.status_code >= 400:
                    raise OrchestratorError(
                        status_code=502,
                        error_code="trading_jvm_error",
                        message=(
                            f"Trading JVM returned {r.status_code} on "
                            f"{path}."
                        ),
                        retryable=r.status_code >= 500,
                        details={
                            "path": path,
                            "status": r.status_code,
                            "body": r.text[:500],
                        },
                    )
                return r.json()
        # Unreachable — tenacity reraise=True
        raise RuntimeError("TradingJvmClient._fetch_json retry loop fell through")


def _parse_uuid(value: Any) -> UUID | None:
    if value is None:
        return None
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (ValueError, TypeError):
        return None
