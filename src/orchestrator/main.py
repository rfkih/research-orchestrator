"""FastAPI app factory.

``create_app`` is the single seam for tests: pass a custom ``Settings`` to
swap in a fixture DSN / fake JVM. The uvicorn entrypoint
(``__main__.py``) calls it with no args so it reads ``.env`` like normal.
"""

from __future__ import annotations

from functools import partial

from fastapi import FastAPI

from .api.activity import router as activity_router
from .api.agent import router as agent_router
from .api.cross_window import router as cross_window_router
from .api.features import router as features_router
from .api.health import router as health_router
from .api.iterations import router as iterations_router
from .api.journal import router as journal_router
from .api.json_response import TypedJSONResponse
from .api.models import router as models_router
from .api.null_screen import router as null_screen_router
from .api.queue import router as queue_router
from .api.raw import router as raw_router
from .api.reviews import router as reviews_router
from .api.tick import router as tick_router
from .api.verdict_drift import router as verdict_drift_router
from .api.walk_forward import router as walk_forward_router
from .auth import AuthMiddleware
from .config import Settings
from .errors import register_exception_handlers
from .logging import configure_logging
from .tasks.lifespan import lifespan_for


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()  # type: ignore[call-arg]
    configure_logging(settings)

    app = FastAPI(
        title="Blackheart research orchestrator",
        version="0.1.0",
        description=(
            "Agent-first FastAPI service. Primary caller is the quant-researcher "
            "Claude agent. See GET /agent/playbook for the contract."
        ),
        # Default error responses go through register_exception_handlers,
        # so we want FastAPI's own validation handler off the path.
        lifespan=partial(lifespan_for, settings),
        # Repo handlers return dicts straight out of asyncpg (Decimal, UUID,
        # datetime). TypedJSONResponse encodes them; the default does not.
        default_response_class=TypedJSONResponse,
    )

    register_exception_handlers(app)
    app.add_middleware(AuthMiddleware, settings=settings)

    app.include_router(health_router)
    app.include_router(agent_router)
    app.include_router(queue_router)
    app.include_router(iterations_router)
    app.include_router(journal_router)
    app.include_router(tick_router)
    app.include_router(walk_forward_router)
    app.include_router(cross_window_router)
    app.include_router(verdict_drift_router)
    app.include_router(null_screen_router)
    app.include_router(reviews_router)
    app.include_router(activity_router)
    app.include_router(features_router)
    app.include_router(models_router)
    app.include_router(raw_router)
    return app


# Module-level app for `uvicorn orchestrator.main:app` invocations.
app = create_app()
