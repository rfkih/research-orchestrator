"""R1.A session-3 smoke — exercise ExperimentClient end-to-end.

Two layers:

1. Pure-function tests for extract_run_metrics + derive_summary_tags on a
   synthetic blackheart-train payload. No HTTP.

2. Full client lifecycle (start_run -> set_params -> set_tags -> log_metrics
   -> finish_run) against the in-process orchestrator via a urllib-urlopen
   monkey-patch that routes calls through TestClient. Verifies a real
   experiment_run row + per-fold experiment_metric rows land in the DB with
   the expected summary_metrics + tags + content_sha256 link.

Run with: PYTHONPATH=src python scripts/_smoke_tracking.py
"""
from __future__ import annotations

import io
import json
import os
import sys
import urllib.error
import urllib.request
import uuid
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv  # type: ignore[import-not-found]

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# Make blackheart-train importable alongside the orchestrator.
TRAIN_SRC = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "blackheart-train", "src"),
)
if TRAIN_SRC not in sys.path:
    sys.path.insert(0, TRAIN_SRC)

from fastapi.testclient import TestClient  # noqa: E402

from blackheart_train.tracking import (  # noqa: E402
    ExperimentClient,
    derive_summary_tags,
    extract_run_metrics,
)
from orchestrator.main import create_app  # noqa: E402


TOKEN = os.environ["ORCH_AUTH_TOKEN"]


def _step(label: str) -> None:
    print(f"\n-- {label}")


# ─────────────────────────────────────────────────────────────────────────
# urllib monkey-patch routing through TestClient
# ─────────────────────────────────────────────────────────────────────────


class _StreamResp:
    """Minimal urllib HTTPResponse stand-in for the with-block in
    _request. Only ``read()`` and ``__enter__``/``__exit__`` are touched
    by ExperimentClient.
    """

    def __init__(self, content: bytes, status: int = 200) -> None:
        self._content = content
        self.status = status

    def __enter__(self) -> "_StreamResp":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._content


def make_urlopen_shim(http: TestClient):
    """Build a urlopen replacement that routes through ``http``."""
    def _shim(req, timeout=None):  # noqa: ANN001 — matches urllib.request.urlopen signature
        method = req.get_method()
        path = urlparse(req.full_url).path
        headers = {k: v for k, v in req.headers.items()}
        json_body: Any = None
        if req.data:
            json_body = json.loads(req.data.decode("utf-8"))
        r = http.request(method, path, headers=headers, json=json_body)
        if r.status_code >= 400:
            # HTTPError signature: url, code, msg, hdrs, fp
            raise urllib.error.HTTPError(
                req.full_url, r.status_code, r.reason_phrase,
                dict(r.headers), io.BytesIO(r.content),
            )
        return _StreamResp(r.content, r.status_code)
    return _shim


# ─────────────────────────────────────────────────────────────────────────
# Pure-function tests
# ─────────────────────────────────────────────────────────────────────────


def test_pure_functions() -> None:
    payload = {
        "metrics": {"auc": 0.61, "log_loss": 0.612, "accuracy": 0.55, "ignored_str": "x"},
        "walk_forward": {
            "primary_metric": "wf_auc",
            "primary_mean": 0.59,
            "primary_median": 0.60,
            "primary_std": 0.04,
            "metric_means": {"auc": 0.58, "log_loss": 0.62},
            "folds": [
                {"fold": 0, "metrics": {"auc": 0.55, "log_loss": 0.65}},
                {"fold": 1, "metrics": {"auc": 0.60, "log_loss": 0.61}},
                {"fold": 2, "metrics": {"auc": 0.62, "log_loss": 0.60}},
            ],
        },
        "gauntlet": {"overall_verdict": "PASS"},
        "deployment_readiness": {"deployment_ready": True},
        "eval_kind": "walk_forward_last_fold",
    }
    metrics = extract_run_metrics(payload)
    names = sorted({m["name"] for m in metrics})
    assert "auc" in names and "log_loss" in names, names
    assert "wf_primary_mean" in names and "wf_mean_auc" in names, names
    # Per-fold present.
    fold_rows = [m for m in metrics if m.get("fold_idx") is not None]
    assert len(fold_rows) == 6, f"expected 6 fold rows, got {len(fold_rows)}"
    # No non-numeric leakage (ignored_str must be filtered).
    assert "ignored_str" not in names

    tags = derive_summary_tags(payload)
    assert tags == {
        "gauntlet_verdict": "PASS",
        "deployment_ready": True,
        "eval_kind": "walk_forward_last_fold",
    }, tags
    print("   ok extract_run_metrics + derive_summary_tags shape correct")


# ─────────────────────────────────────────────────────────────────────────
# Full client lifecycle against in-process orchestrator
# ─────────────────────────────────────────────────────────────────────────


def test_client_lifecycle(http: TestClient) -> str:
    original_urlopen = urllib.request.urlopen
    urllib.request.urlopen = make_urlopen_shim(http)
    try:
        client = ExperimentClient(
            orchestrator_url="http://stub-not-used",  # urlopen is shimmed
            auth_token=TOKEN,
            agent_name="r1-track-smoke",
            timeout_s=5.0,
        )

        _step("1. start_run")
        run_id = client.start_run(
            spec_name="regime_btc_v3_track_smoke",
            spec_symbol="BTCUSDT",
            spec_interval="1h",
            spec_horizon_bars=24,
            git_sha="abc123",
            dataset_sha="d00d",
            params={"num_leaves": 32},
            tags={"lifecycle": "r1-track-smoke", "agent": "r1-track-smoke"},
            idempotency_key="r1-track-smoke-001",
        )
        assert run_id is not None
        uuid.UUID(run_id)
        assert client.run_id == run_id
        print(f"   ok run_id={run_id[:8]}...")

        _step("2. set_params replaces snapshot")
        client.set_params({"num_leaves": 64, "learning_rate": 0.03})
        # Verify via GET — exercises the read path too.
        detail = http.get(f"/experiments/{run_id}", headers={"X-Orch-Token": TOKEN, "X-Agent-Name": "r1-track-smoke"})
        assert detail.json()["params"]["num_leaves"] == 64
        print("   ok params updated to 64")

        _step("3. set_tags merges")
        client.set_tags({"gauntlet_verdict": "PASS"}, merge=True)
        detail = http.get(f"/experiments/{run_id}", headers={"X-Orch-Token": TOKEN, "X-Agent-Name": "r1-track-smoke"})
        tags = detail.json()["tags"]
        assert tags.get("lifecycle") == "r1-track-smoke" and tags.get("gauntlet_verdict") == "PASS", tags
        print(f"   ok merged tags: {sorted(tags)}")

        _step("4. log_metric (single, run-level)")
        client.log_metric("oof_dsr", 0.91)
        detail = http.get(f"/experiments/{run_id}", headers={"X-Orch-Token": TOKEN, "X-Agent-Name": "r1-track-smoke"})
        assert detail.json()["summary_metrics"]["oof_dsr"] == 0.91
        print("   ok run-level scalar mirrored into summary_metrics")

        _step("5. log_metrics batch (mixed run + fold)")
        payload_metrics = [
            {"name": "oof_auc", "value": 0.62},
            {"name": "fold_auc", "value": 0.55, "fold_idx": 0},
            {"name": "fold_auc", "value": 0.60, "fold_idx": 1},
            {"name": "fold_auc", "value": 0.62, "fold_idx": 2},
            # Non-finite filter — client drops, server never sees it.
            {"name": "nan_drop", "value": float("nan")},
        ]
        client.log_metrics(payload_metrics)
        detail = http.get(f"/experiments/{run_id}", headers={"X-Orch-Token": TOKEN, "X-Agent-Name": "r1-track-smoke"})
        all_metrics = detail.json()["metrics"]
        fold_aucs = sorted(m["value"] for m in all_metrics if m["metric_name"] == "fold_auc")
        assert fold_aucs == [0.55, 0.60, 0.62], fold_aucs
        names = {m["metric_name"] for m in all_metrics}
        assert "nan_drop" not in names, "NaN metric must be filtered client-side"
        assert detail.json()["summary_metrics"]["oof_auc"] == 0.62
        print(f"   ok 4 metrics inserted, NaN dropped, {len(all_metrics)} total rows")

        _step("6. finish_run with no content_sha256 (failed-run case)")
        client.finish_run("completed")
        detail = http.get(f"/experiments/{run_id}", headers={"X-Orch-Token": TOKEN, "X-Agent-Name": "r1-track-smoke"})
        assert detail.json()["status"] == "completed"
        assert detail.json()["content_sha256"] is None
        print("   ok status=completed, content_sha256=NULL")

        _step("7. second finish_run is a no-op (client-side idempotency)")
        client.finish_run("completed")  # Should not raise + should not 409
        # Verify the underlying server still has status=completed (not aborted etc).
        detail = http.get(f"/experiments/{run_id}", headers={"X-Orch-Token": TOKEN, "X-Agent-Name": "r1-track-smoke"})
        assert detail.json()["status"] == "completed"
        print("   ok client._finished prevented duplicate POST")

        return run_id

    finally:
        urllib.request.urlopen = original_urlopen


# ─────────────────────────────────────────────────────────────────────────
# Tolerant-mode test (orchestrator unreachable)
# ─────────────────────────────────────────────────────────────────────────


def test_tolerant_mode_on_failure() -> None:
    """When the orchestrator is unreachable, client logs + degrades but
    does NOT raise. Subsequent calls become no-ops.
    """
    client = ExperimentClient(
        orchestrator_url="http://127.0.0.1:1",  # nothing listens here
        auth_token=TOKEN,
        agent_name="r1-track-smoke",
        timeout_s=1.0,
        max_attempts=1,
        backoff_base_s=0.0,
    )
    run_id = client.start_run(spec_name="unreachable_test")
    assert run_id is None, "tolerant mode must return None on failure"
    assert client.enabled is False
    assert client.degraded is True
    # Subsequent calls no-op cleanly.
    client.log_metric("x", 1.0)
    client.set_tags({"a": "b"})
    client.finish_run("completed")
    print("   ok unreachable orchestrator -> degraded=True, enabled=False, calls no-op")


# ─────────────────────────────────────────────────────────────────────────
# Strict-mode test
# ─────────────────────────────────────────────────────────────────────────


def test_strict_mode_raises() -> None:
    from blackheart_train.tracking import TrackingError

    client = ExperimentClient(
        orchestrator_url="http://127.0.0.1:1",
        auth_token=TOKEN,
        agent_name="r1-track-smoke",
        timeout_s=1.0,
        max_attempts=1,
        backoff_base_s=0.0,
        strict=True,
    )
    try:
        client.start_run(spec_name="strict_test")
    except TrackingError:
        print("   ok strict mode raised TrackingError")
        return
    raise AssertionError("strict mode must raise on unreachable orchestrator")


# ─────────────────────────────────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────────────────────────────────


def _delete_via_docker(ids: list[str]) -> int:
    """Cleanup goes through docker exec psql, not the orchestrator's asyncpg
    pool. The pool is bound to the TestClient's event loop; reaching it from
    asyncio.run() in the same process is a no-go (different loop = pool
    can't be acquired). Keeping cleanup out-of-band side-steps that entirely.
    """
    import subprocess
    sql = f"DELETE FROM experiment_run WHERE run_id IN ({','.join(repr(i) for i in ids)});"
    r = subprocess.run(
        ["docker", "exec", "blackheart-postgres", "psql", "-U", "postgres",
         "-d", "trading_db", "-t", "-A", "-c", sql],
        capture_output=True, text=True, check=True,
    )
    # psql -t -A outputs just "DELETE <n>" on stdout when no rows are returned
    out = (r.stdout or "").strip()
    if out.startswith("DELETE"):
        return int(out.split()[-1])
    return 0


def main() -> int:
    _step("A. pure-function tests")
    test_pure_functions()

    _step("B. tolerant mode + strict mode")
    test_tolerant_mode_on_failure()
    test_strict_mode_raises()

    _step("C. full client lifecycle via in-process orchestrator")
    app = create_app()
    with TestClient(app) as http:
        run_id = test_client_lifecycle(http)

    _step("D. cleanup")
    deleted = _delete_via_docker([run_id])
    print(f"   ok deleted {deleted} smoke run(s)")

    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
