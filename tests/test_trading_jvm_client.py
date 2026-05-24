"""Tests for :class:`TradingJvmClient` (Phase A.6, 2026-05-22).

Mocks the trading JVM via :mod:`respx`. Covers:

* Happy-path portfolio + Kelly reads.
* Decimal precision (Jackson serialises BigDecimal as a JSON number;
  the client must round-trip through ``str`` to keep operator-visible
  literals exact).
* X-Orch-Token header is sent.
* 401 from the JVM → structured :class:`OrchestratorError` rather than
  raw httpx exception bleeding through.
* 5xx → structured retryable error.
* ``enabled = False`` when base URL is empty.
"""

from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

import httpx
import pytest
import respx

from orchestrator.clients.trading_jvm import TradingJvmClient
from orchestrator.config import DEV_TOKEN_SENTINEL, Settings
from orchestrator.errors import OrchestratorError


def _build_settings(trading_base_url: str = "http://127.0.0.1:8080") -> Settings:
    """Construct Settings with only the fields :class:`TradingJvmClient`
    reads. Other required fields use harmless defaults — we never call
    .assert_prod_safe() in these tests."""
    return Settings(
        auth_token=DEV_TOKEN_SENTINEL,
        db_dsn="postgresql://x:x@localhost:5432/x",
        trading_jvm_base_url=trading_base_url,
        trading_jvm_request_timeout_s=5.0,
    )


@pytest.mark.asyncio
async def test_client_disabled_when_base_url_empty() -> None:
    """``enabled`` short-circuit lets callers branch fast without
    catching exceptions. With an empty base URL the client refuses any
    request rather than dialing nowhere."""
    settings = _build_settings(trading_base_url="")
    client = TradingJvmClient(settings)
    assert client.enabled is False
    await client.open()  # no-op when disabled
    with pytest.raises(OrchestratorError) as ei:
        await client.get_account_portfolio(uuid4())
    assert ei.value.envelope.error_code == "trading_jvm_disabled"
    assert ei.value.envelope.retryable is False


@pytest.mark.asyncio
async def test_get_account_portfolio_happy_path() -> None:
    """End-to-end: client opens, fires GET, parses
    PortfolioBalanceResponse into Decimal-typed dataclass, sends the
    X-Orch-Token header."""
    account_id = uuid4()
    settings = _build_settings()
    client = TradingJvmClient(settings)
    await client.open()

    captured_headers: dict[str, str] = {}

    async with respx.mock(base_url=settings.trading_jvm_base_url) as mock:
        route = mock.get(f"/api/v1/internal/orch/portfolio/{account_id}").mock(
            return_value=httpx.Response(
                200,
                json={
                    "responseCode": "20000",
                    "data": {
                        "accountId": str(account_id),
                        "totalUsdt": 10000.55,
                        "availableUsdt": 7500.25,
                        "lockedUsdt": 2500.30,
                        "assets": [],
                    },
                },
            )
        )

        portfolio = await client.get_account_portfolio(account_id)

        # Header capture from the recorded call.
        call = route.calls[0]
        captured_headers = dict(call.request.headers)

    assert portfolio.account_id == account_id
    assert portfolio.available_usdt == Decimal("7500.25")
    assert portfolio.total_usdt == Decimal("10000.55")
    assert portfolio.locked_usdt == Decimal("2500.30")
    assert captured_headers.get("x-orch-token") == DEV_TOKEN_SENTINEL

    await client.close()


@pytest.mark.asyncio
async def test_get_kelly_status_happy_path() -> None:
    """Kelly endpoint returns the executor's exact multiplier — no
    upper-bound assumption. The client coerces BigDecimal-via-JSON-
    number through str to keep precision sane."""
    account_strategy_id = uuid4()
    settings = _build_settings()
    client = TradingJvmClient(settings)
    await client.open()

    async with respx.mock(base_url=settings.trading_jvm_base_url) as mock:
        mock.get(f"/api/v1/internal/orch/kelly/{account_strategy_id}").mock(
            return_value=httpx.Response(
                200,
                json={
                    "responseCode": "20000",
                    "data": {
                        "accountStrategyId": str(account_strategy_id),
                        "enabled": True,
                        "currentMultiplier": 0.4321,
                        "maxFraction": 0.25,
                        "qualifyingRuns": 4,
                        "reason": "computed from 4 run(s)",
                    },
                },
            )
        )

        kelly = await client.get_kelly_status(account_strategy_id)

    assert kelly.enabled is True
    assert kelly.current_multiplier == Decimal("0.4321")
    assert kelly.max_fraction == Decimal("0.25")
    assert kelly.qualifying_runs == 4
    assert "computed from 4 run(s)" in kelly.reason

    await client.close()


@pytest.mark.asyncio
async def test_get_kelly_status_disabled_returns_one() -> None:
    """When Kelly sizing is disabled, the JVM returns ``enabled=False``
    + ``currentMultiplier=1.0``. The client surfaces both fields so the
    guard math can multiply by 1.0 (the no-op fallback) and the UI can
    label the row accurately."""
    account_strategy_id = uuid4()
    settings = _build_settings()
    client = TradingJvmClient(settings)
    await client.open()

    async with respx.mock(base_url=settings.trading_jvm_base_url) as mock:
        mock.get(f"/api/v1/internal/orch/kelly/{account_strategy_id}").mock(
            return_value=httpx.Response(
                200,
                json={
                    "responseCode": "20000",
                    "data": {
                        "accountStrategyId": str(account_strategy_id),
                        "enabled": False,
                        "currentMultiplier": 1.0,
                        "maxFraction": 0.25,
                        "qualifyingRuns": 0,
                        "reason": "kelly disabled",
                    },
                },
            )
        )

        kelly = await client.get_kelly_status(account_strategy_id)

    assert kelly.enabled is False
    assert kelly.current_multiplier == Decimal("1")

    await client.close()


@pytest.mark.asyncio
async def test_unauthorized_raises_structured_error() -> None:
    """A 401 from the JVM means X-Orch-Token mismatch — almost always
    a config skew between ORCH_AUTH_TOKEN and the JVM's
    app.research.orchestrator.token. Surface a specific error code so
    the operator can fix the right env var."""
    account_id = uuid4()
    settings = _build_settings()
    client = TradingJvmClient(settings)
    await client.open()

    async with respx.mock(base_url=settings.trading_jvm_base_url) as mock:
        mock.get(f"/api/v1/internal/orch/portfolio/{account_id}").mock(
            return_value=httpx.Response(401, json={"errorMessage": "Invalid X-Orch-Token"})
        )

        with pytest.raises(OrchestratorError) as ei:
            await client.get_account_portfolio(account_id)

    assert ei.value.envelope.error_code == "trading_jvm_unauthorized"
    assert ei.value.envelope.retryable is False
    assert "ORCH_AUTH_TOKEN" in (ei.value.envelope.hint or ei.value.envelope.message)

    await client.close()


@pytest.mark.asyncio
async def test_5xx_raises_retryable_error() -> None:
    """JVM 5xx is a transient failure (DB blip, restart) — the
    envelope carries ``retryable=True`` so callers / cron policies can
    back off and retry."""
    account_id = uuid4()
    settings = _build_settings()
    client = TradingJvmClient(settings)
    await client.open()

    async with respx.mock(base_url=settings.trading_jvm_base_url) as mock:
        mock.get(f"/api/v1/internal/orch/portfolio/{account_id}").mock(
            return_value=httpx.Response(503, json={"errorMessage": "Database unavailable"})
        )

        with pytest.raises(OrchestratorError) as ei:
            await client.get_account_portfolio(account_id)

    assert ei.value.envelope.error_code == "trading_jvm_error"
    assert ei.value.envelope.retryable is True
    assert (ei.value.envelope.details or {}).get("status") == 503

    await client.close()


@pytest.mark.asyncio
async def test_4xx_raises_non_retryable_error() -> None:
    """A 404 (unknown accountId) is a client-side mistake — caller
    should fix its inputs, not back off. envelope.retryable=False."""
    account_id = uuid4()
    settings = _build_settings()
    client = TradingJvmClient(settings)
    await client.open()

    async with respx.mock(base_url=settings.trading_jvm_base_url) as mock:
        mock.get(f"/api/v1/internal/orch/portfolio/{account_id}").mock(
            return_value=httpx.Response(404, json={"errorMessage": "Not found"})
        )

        with pytest.raises(OrchestratorError) as ei:
            await client.get_account_portfolio(account_id)

    assert ei.value.envelope.error_code == "trading_jvm_error"
    assert ei.value.envelope.retryable is False
    assert (ei.value.envelope.details or {}).get("status") == 404

    await client.close()
