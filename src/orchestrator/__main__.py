"""``python -m orchestrator`` entrypoint.

Same code path whether invoked from systemd, a Windows service wrapper,
``uv run``, or a developer's terminal. Reads host/port from Settings so
operators only edit ``.env``.
"""

from __future__ import annotations

import uvicorn
from dotenv import load_dotenv

from .config import Settings


def main() -> None:
    load_dotenv()  # populate os.environ from .env so non-Settings modules can read it
    settings = Settings()  # type: ignore[call-arg]
    uvicorn.run(
        "orchestrator.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        # Reload off — this process runs as a service; restart via systemd.
        reload=False,
        # Single worker — claim-loop and Idempotency-Key registry will both
        # assume one process. Scale by sharding the queue, not by adding
        # workers.
        workers=1,
    )


if __name__ == "__main__":
    main()
