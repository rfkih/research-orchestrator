"""Endpoint-contract tests for ``POST /signal-screen``.

Auth, validation envelopes, signal resolution (feature_store vs
feature_values vs 404), journal-row emission (kind='signal_screen',
multiplicity note), and Idempotency-Key replay -- all with the repo layer
monkeypatched, no DB. Mirrors the conftest ``client`` conventions.
"""

from __future__ import annotations

import math
import random
import uuid
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

import orchestrator.services.signal_screen as svc

AUTH = {"X-Orch-Token": "test-token", "X-Agent-Name": "quant-researcher"}
T0 = datetime(2024, 1, 1)


def _bars(n: int, seed: int = 7) -> list[tuple[datetime, float]]:
    rng = random.Random(seed)
    closes = [100.0]
    for _ in range(n - 1):
        closes.append(closes[-1] * math.exp(rng.gauss(0, 0.01)))
    return [(T0 + timedelta(hours=k), closes[k]) for k in range(n)]


@pytest.fixture
def mocked_repos(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Patch every repo call run_signal_screen makes. The default shape:
    'rsi' is a feature_store column; 'macro_x' is a feature_values series;
    anything else is unknown. Captures journal inserts."""
    bars = _bars(120)
    state: dict = {"journal_inserts": [], "bars": bars, "prior_screens": 0}

    async def list_numeric_columns(conn):
        return {"rsi", "atr_pct"}

    async def fetch_closes(conn, *, symbol, interval, start, end):
        return state["bars"]

    async def fetch_column_series(conn, *, column, symbol, interval, start, end):
        rng = random.Random(99)
        return [(ts, rng.gauss(0, 1)) for ts, _ in state["bars"]]

    async def fetch_feature_series(conn, *, name, symbol, interval, start, end):
        if name != "macro_x":
            return [], {"resolved": False}
        rng = random.Random(98)
        points = [
            (ts - timedelta(minutes=30), rng.gauss(0, 1)) for ts, _ in state["bars"]
        ]
        return points, {
            "resolved": True, "version": 1,
            "scope_symbol": "", "scope_interval": "",
        }

    async def count_signal_screens(conn, *, signal_family):
        return state["prior_screens"]

    async def insert_journal(conn, **kwargs):
        state["journal_inserts"].append(kwargs)
        return {"journal_id": uuid.uuid4(), **kwargs}

    monkeypatch.setattr(
        svc.feature_store_repo, "list_numeric_columns", list_numeric_columns
    )
    monkeypatch.setattr(
        svc.feature_store_repo, "fetch_column_series", fetch_column_series
    )
    monkeypatch.setattr(svc.market_data_repo, "fetch_closes", fetch_closes)
    monkeypatch.setattr(
        svc.features_repo, "fetch_feature_series", fetch_feature_series
    )
    monkeypatch.setattr(
        svc.journal_repo, "count_signal_screens", count_signal_screens
    )
    monkeypatch.setattr(svc.journal_repo, "insert_journal", insert_journal)
    return state


def _body(**overrides) -> dict:
    body = {
        "signal": "rsi",
        "instruments": ["BTCUSDT"],
        "interval": "1h",
        "horizons_bars": [1, 6],
        "transform": "zscore",
    }
    body.update(overrides)
    return body


# -- auth + validation envelopes ----------------------------------------


def test_signal_screen_requires_token(client: TestClient) -> None:
    r = client.post("/signal-screen", json=_body())
    assert r.status_code == 401
    assert r.json()["error_code"] == "auth_missing_token"


def test_signal_screen_rejects_bad_interval(client: TestClient) -> None:
    r = client.post("/signal-screen", json=_body(interval="2h"), headers=AUTH)
    assert r.status_code == 400
    assert r.json()["error_code"] == "bad_interval"


def test_signal_screen_rejects_bad_transform(client: TestClient) -> None:
    r = client.post("/signal-screen", json=_body(transform="log"), headers=AUTH)
    assert r.status_code in (400, 422)


def test_signal_screen_rejects_injection_shaped_signal(client: TestClient) -> None:
    r = client.post(
        "/signal-screen",
        json=_body(signal='rsi"; DROP TABLE feature_store; --'),
        headers=AUTH,
    )
    assert r.status_code in (400, 422)


def test_signal_screen_rejects_empty_instruments(client: TestClient) -> None:
    r = client.post("/signal-screen", json=_body(instruments=[]), headers=AUTH)
    assert r.status_code in (400, 422)


def test_signal_screen_rejects_bad_horizon(client: TestClient) -> None:
    r = client.post("/signal-screen", json=_body(horizons_bars=[0]), headers=AUTH)
    assert r.status_code in (400, 422)


# -- resolution + happy path ----------------------------------------------


def test_screen_feature_store_column_happy_path(
    client: TestClient, mocked_repos: dict
) -> None:
    r = client.post("/signal-screen", json=_body(), headers=AUTH)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["resolved_source"] == "feature_store"
    assert out["overall_verdict"] in {"PROMISING", "WEAK", "DEAD", "INSUFFICIENT_DATA"}
    # One row per (instrument, horizon); single instrument -> no POOLED.
    assert len(out["rows"]) == 2
    horizons = {row["horizon_bars"] for row in out["rows"]}
    assert horizons == {1, 6}
    for row in out["rows"]:
        assert set(row) >= {
            "instrument", "horizon_bars", "n_obs", "ic", "nw_tstat",
            "quantile_spread", "bootstrap_ci", "verdict", "rank_score",
        }
    # Ranked by score desc.
    scores = [row["rank_score"] for row in out["rows"]]
    assert scores == sorted(scores, reverse=True)
    assert "V11/V60" in out["advisory"]


def test_screen_falls_back_to_feature_values(
    client: TestClient, mocked_repos: dict
) -> None:
    r = client.post("/signal-screen", json=_body(signal="macro_x"), headers=AUTH)
    assert r.status_code == 200, r.text
    assert r.json()["resolved_source"] == "feature_values"


def test_screen_unknown_signal_is_404(
    client: TestClient, mocked_repos: dict
) -> None:
    r = client.post("/signal-screen", json=_body(signal="nope"), headers=AUTH)
    assert r.status_code == 404
    assert r.json()["error_code"] == "signal_not_found"
    # 404 must NOT journal anything.
    assert mocked_repos["journal_inserts"] == []


# -- journal emission --------------------------------------------------------


def test_screen_journals_one_signal_screen_row(
    client: TestClient, mocked_repos: dict
) -> None:
    r = client.post("/signal-screen", json=_body(), headers=AUTH)
    assert r.status_code == 200
    inserts = mocked_repos["journal_inserts"]
    assert len(inserts) == 1
    row = inserts[0]
    sd = row["structured_data"]
    assert sd["kind"] == "signal_screen"
    assert sd["signal"] == "rsi"
    assert sd["overall_verdict"] == r.json()["overall_verdict"]
    assert sd["rows"] == r.json()["rows"]
    assert row["title"].startswith("SIGNAL_SCREEN rsi 1h")
    assert row["created_by"] == "quant-researcher"
    assert row["status"] == "ACTIVE"
    # entry_type must be inside the DB CHECK set (no SIGNAL_SCREEN value).
    assert row["entry_type"] in {"ANTI_PATTERN", "CROSS_STRATEGY_FINDING"}
    assert r.json()["journal_id"]


def test_screen_multiplicity_note_counts_prior_screens(
    client: TestClient, mocked_repos: dict
) -> None:
    mocked_repos["prior_screens"] = 3
    r = client.post("/signal-screen", json=_body(), headers=AUTH)
    assert r.status_code == 200
    mult = r.json()["multiplicity"]
    assert mult["prior_screens_same_family"] == 3
    assert "screen #4" in mult["note"]
    # Persisted in the journal row too.
    sd = mocked_repos["journal_inserts"][0]["structured_data"]
    assert sd["multiplicity"]["prior_screens_same_family"] == 3


# -- idempotency ---------------------------------------------------------------


def test_screen_idempotency_key_replays_without_recompute(
    client: TestClient, mocked_repos: dict
) -> None:
    headers = {**AUTH, "Idempotency-Key": "screen-abc"}
    r1 = client.post("/signal-screen", json=_body(), headers=headers)
    r2 = client.post("/signal-screen", json=_body(), headers=headers)
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json() == r2.json()
    # Second call replayed the cache -- only ONE journal row written.
    assert len(mocked_repos["journal_inserts"]) == 1
