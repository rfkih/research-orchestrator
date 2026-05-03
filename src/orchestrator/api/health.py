"""Health probes.

``/healthz`` is liveness — does the process answer? No I/O, fast, public.
``/readyz`` is readiness — DB + JVM both reachable. Public so the systemd /
Windows-service readiness check works without baking a token into the unit
file. Both endpoints intentionally bypass ``AuthMiddleware`` (see
``auth.py`` ``_PUBLIC_PATHS``).
"""

from __future__ import annotations

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
