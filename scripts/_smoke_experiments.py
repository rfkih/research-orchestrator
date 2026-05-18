"""R1.A session-2 smoke test — exercise the /experiments router end-to-end.

Uses FastAPI TestClient against a freshly-created app so we don't disturb
the running orchestrator on :8082. The app is wired to the real local DB
(asyncpg connects to blackheart_research role via the .env DSN).

Rows are cleaned up at the end via a final DELETE. Run with:

    PYTHONPATH=src python scripts/_smoke_experiments.py

Exit code 0 = all checks pass. Exit code 1 = first failed assertion.
"""
from __future__ import annotations

import os
import sys
import uuid

# Reuse the running orchestrator's .env file so we hit the same DSN +
# token. The TestClient honours these via the standard Settings load.
from dotenv import load_dotenv  # type: ignore[import-not-found]

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# Import after env is loaded so Settings() picks up the values.
from fastapi.testclient import TestClient  # noqa: E402

from orchestrator.main import create_app  # noqa: E402


TOKEN = os.environ["ORCH_AUTH_TOKEN"]
HEADERS = {"X-Orch-Token": TOKEN, "X-Agent-Name": "r1-smoke"}


def _step(label: str) -> None:
    print(f"\n-- {label}")


def main() -> int:
    app = create_app()
    with TestClient(app) as client:
        # 1. Auth — bad token rejected.
        _step("1. auth: bad token -> 401")
        r = client.post("/experiments", json={"spec_name": "x"}, headers={"X-Orch-Token": "wrong"})
        assert r.status_code == 401, r.status_code
        assert r.json()["error_code"] == "auth_bad_token"
        print("   ok 401")

        # 2. Start a run.
        _step("2. POST /experiments — start run")
        r = client.post(
            "/experiments",
            json={
                "spec_name": "regime_btc_v3_smoke",
                "spec_version": "v0",
                "spec_symbol": "BTCUSDT",
                "spec_interval": "1h",
                "spec_horizon_bars": 24,
                "git_sha": "deadbeef",
                "dataset_sha": "feedface",
                "params": {"num_leaves": 64, "learning_rate": 0.05},
                "tags": {"lifecycle": "r1-smoke"},
            },
            headers=HEADERS,
        )
        assert r.status_code == 200, (r.status_code, r.text)
        run_id = r.json()["run_id"]
        uuid.UUID(run_id)  # parses
        assert r.json()["status"] == "running"
        print(f"   ok run_id={run_id[:8]}...")

        # 3. Idempotency replay — same key returns same run_id.
        _step("3. POST /experiments — idempotency replay")
        idem_headers = {**HEADERS, "Idempotency-Key": "smoke-replay-001"}
        r1 = client.post("/experiments", json={"spec_name": "idem_test"}, headers=idem_headers)
        r2 = client.post("/experiments", json={"spec_name": "idem_test"}, headers=idem_headers)
        assert r1.status_code == 200 and r2.status_code == 200
        assert r1.json()["run_id"] == r2.json()["run_id"], "idempotency-key replay must match"
        print("   ok same run_id across two POSTs")

        # 4. Patch params + tags.
        _step("4. PATCH /experiments/{id}/params + /tags")
        r = client.patch(
            f"/experiments/{run_id}/params",
            json={"params": {"num_leaves": 128, "learning_rate": 0.03}},
            headers=HEADERS,
        )
        assert r.status_code == 200, r.text
        assert r.json()["params"]["num_leaves"] == 128
        r = client.patch(
            f"/experiments/{run_id}/tags",
            json={"tags": {"note": "patched"}, "merge": True},
            headers=HEADERS,
        )
        assert r.status_code == 200
        tags = r.json()["tags"]
        assert tags.get("lifecycle") == "r1-smoke" and tags.get("note") == "patched", tags
        print("   ok params replaced, tags merged")

        # 5. Log metrics — run-level mirror + per-fold.
        _step("5. PATCH /experiments/{id}/metrics")
        r = client.patch(
            f"/experiments/{run_id}/metrics",
            json={"metrics": [
                {"name": "oof_auc", "value": 0.62},
                {"name": "oof_dsr", "value": 0.87},
                {"name": "fold_auc", "value": 0.58, "fold_idx": 0},
                {"name": "fold_auc", "value": 0.64, "fold_idx": 1},
            ]},
            headers=HEADERS,
        )
        assert r.status_code == 200, r.text
        assert r.json()["inserted"] == 4
        print("   ok 4 metrics inserted")

        # 6. NaN protection: httpx refuses to serialize NaN at request
        #    build time (allow_nan=False), so the Pydantic validator is
        #    belt-and-suspenders behind that. The DB CHECK constraint was
        #    smoke-tested in session 1. Nothing to exercise here.
        _step("6. NaN handling — covered by upstream serializer + V92 CHECK")
        print("   skipped (httpx blocks NaN before request leaves the process)")

        # 7. GET run detail — header + per-fold metrics.
        _step("7. GET /experiments/{id}")
        r = client.get(f"/experiments/{run_id}", headers=HEADERS)
        assert r.status_code == 200
        detail = r.json()
        assert detail["summary_metrics"].get("oof_auc") == 0.62
        assert detail["summary_metrics"].get("oof_dsr") == 0.87
        metric_names = sorted({m["metric_name"] for m in detail["metrics"]})
        assert metric_names == ["fold_auc", "oof_auc", "oof_dsr"], metric_names
        print(f"   ok metric rows={len(detail['metrics'])}, summary keys mirrored")

        # 8. Leaderboard.
        _step("8. GET /experiments/leaderboard")
        r = client.get(
            "/experiments/leaderboard",
            params={"spec_name": "regime_btc_v3_smoke", "metric": "oof_dsr"},
            headers=HEADERS,
        )
        # Leaderboard filters status='completed' by default, so the
        # still-running smoke row shouldn't show up.
        assert r.status_code == 200, r.text
        body = r.json()
        assert all(row["run_id"] != run_id for row in body["rows"]), \
            "running rows must not appear in default leaderboard"
        print(f"   ok leaderboard filtered out running runs ({len(body['rows'])} historical)")

        # 9. Finish the run — include dataset_sha (R1 close-out).
        _step("9. POST /experiments/{id}/finish (with dataset_sha)")
        r = client.post(
            f"/experiments/{run_id}/finish",
            json={"status": "completed", "dataset_sha": "abc123" * 10 + "def0"},
            headers=HEADERS,
        )
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "completed"
        assert r.json()["duration_seconds"] is not None
        assert r.json()["dataset_sha"] == "abc123" * 10 + "def0"
        # GET back the row and confirm the dataset_sha persisted.
        r2 = client.get(f"/experiments/{run_id}", headers=HEADERS)
        assert r2.json()["dataset_sha"] == "abc123" * 10 + "def0", r2.json()
        print(f"   ok status=completed, dataset_sha persisted to column")

        # 10. Second finish without idempotency-key → 409.
        _step("10. second /finish without key -> 409")
        r = client.post(
            f"/experiments/{run_id}/finish",
            json={"status": "completed"},
            headers=HEADERS,
        )
        assert r.status_code == 409, (r.status_code, r.text)
        print("   ok 409 race-guard")

        # 11. Leaderboard now includes the completed run.
        _step("11. leaderboard includes completed run")
        r = client.get(
            "/experiments/leaderboard",
            params={"spec_name": "regime_btc_v3_smoke", "metric": "oof_dsr"},
            headers=HEADERS,
        )
        assert r.status_code == 200
        found = [row for row in r.json()["rows"] if row["run_id"] == run_id]
        assert len(found) == 1, "completed run should appear"
        assert found[0]["metric_value"] == 0.87
        print(f"   ok run present, metric_value={found[0]['metric_value']}")

        # 12. List filters.
        _step("12. GET /experiments with filters")
        r = client.get("/experiments",
                       params={"spec_name": "regime_btc_v3_smoke", "status": "completed"},
                       headers=HEADERS)
        assert r.status_code == 200
        assert r.json()["total"] >= 1
        print(f"   ok total={r.json()['total']}")

        # 13. 404 on unknown run_id.
        _step("13. 404 on unknown id")
        fake_id = "00000000-0000-0000-0000-000000000000"
        r = client.patch(f"/experiments/{fake_id}/params",
                         json={"params": {}}, headers=HEADERS)
        assert r.status_code == 404, r.text
        print("   ok 404")

        # Cleanup goes through docker-exec psql, not the orchestrator's
        # asyncpg pool. The pool is bound to the TestClient's event loop;
        # reaching it from a fresh asyncio.run() (different loop) raises
        # ConnectionDoesNotExistError. Out-of-band cleanup side-steps that.
        _step("cleanup")
        ids = [run_id, r1.json()["run_id"]]
        import subprocess

        sql = f"DELETE FROM experiment_run WHERE run_id IN ({','.join(repr(i) for i in ids)});"
        result = subprocess.run(
            ["docker", "exec", "blackheart-postgres", "psql", "-U", "postgres",
             "-d", "trading_db", "-t", "-A", "-c", sql],
            capture_output=True, text=True, check=True,
        )
        out = (result.stdout or "").strip()
        deleted = int(out.split()[-1]) if out.startswith("DELETE") else 0
        print(f"   ok deleted {deleted} smoke runs")

    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
