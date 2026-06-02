"""Health probes.

``/healthz`` is liveness — does the process answer? No I/O, fast, public.
``/readyz`` is readiness — DB + JVM both reachable. Public so the systemd /
Windows-service readiness check works without baking a token into the unit
file. Both endpoints intentionally bypass ``AuthMiddleware`` (see
``auth.py`` ``_PUBLIC_PATHS``).
"""

from __future__ import annotations

import asyncio
import os

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class Health(BaseModel):
    status: str
    version: str


class Ready(BaseModel):
    status: str
    db: bool
    jvm: bool


class MlReady(BaseModel):
    """ML-training infra readiness. All four sub-checks must be true before
    POST /ml/training-runs can succeed. Lets the researcher refuse a doomed
    ML session up-front instead of consuming a budget slot to discover a
    missing mount / unimportable lightgbm / read-only artifact dir."""

    status: str
    all_ok: bool
    python_ok: bool
    repo_ok: bool
    artifact_dir_ok: bool
    lgbm_importable: bool
    detail: str


@router.get("/healthz", response_model=Health)
async def healthz() -> Health:
    from .. import __version__

    return Health(status="ok", version=__version__)


@router.get("/readyz", response_model=Ready)
async def readyz(request: Request) -> Ready:
    db_ok = await request.app.state.db.health_probe()
    jvm_ok = await request.app.state.jvm.health_probe()
    status = "ok" if (db_ok and jvm_ok) else "degraded"
    return Ready(status=status, db=db_ok, jvm=jvm_ok)


@router.get("/ml/readyz", response_model=MlReady)
async def ml_readyz(request: Request) -> MlReady:
    """Probe the ML-training subprocess environment. Public (no DB/JVM I/O
    beyond a quick interpreter spawn) so the researcher can call it before
    forming a HYBRID/ML hypothesis."""
    s = request.app.state.settings
    python_ok = os.path.isfile(s.ml_training_python_path)
    repo_ok = os.path.isdir(s.ml_training_repo_path)
    # Empty artifact dir setting = use train repo default (always fine).
    artifact_dir_ok = (
        not s.ml_training_artifact_dir
        or os.path.isdir(s.ml_training_artifact_dir)
    )

    lgbm_importable = False
    detail = ""
    if python_ok:
        try:
            proc = await asyncio.create_subprocess_exec(
                s.ml_training_python_path, "-c", "import lightgbm",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
            lgbm_importable = proc.returncode == 0
            if not lgbm_importable:
                detail = stderr.decode("utf-8", "replace")[-300:]
        except (asyncio.TimeoutError, OSError) as exc:
            detail = f"lightgbm import probe failed: {exc.__class__.__name__}"
    else:
        detail = f"python interpreter not found at {s.ml_training_python_path}"

    all_ok = python_ok and repo_ok and artifact_dir_ok and lgbm_importable
    return MlReady(
        status="ok" if all_ok else "degraded",
        all_ok=all_ok,
        python_ok=python_ok,
        repo_ok=repo_ok,
        artifact_dir_ok=artifact_dir_ok,
        lgbm_importable=lgbm_importable,
        detail=detail or "ml training infra ready",
    )
