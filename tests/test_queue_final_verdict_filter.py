"""Pin: the /queue list supports a server-side ``final_verdict`` filter.

The walk-forward-candidate panel previously fetched PARKED rows with a limit and
post-filtered ``final_verdict == 'SIGNIFICANT_EDGE'`` client-side, so a truncated
page silently dropped qualifying candidates. The filter is now applied
server-side; these pins keep it wired through both layers.
"""
from __future__ import annotations

import inspect

from orchestrator.api import queue as queue_api
from orchestrator.repo import queue as queue_repo


def test_repo_list_queue_filters_on_final_verdict() -> None:
    src = inspect.getsource(queue_repo.list_queue)
    assert "final_verdict: str | None" in src, "list_queue must accept a final_verdict filter param"
    assert "final_verdict = $" in src, "list_queue must add final_verdict to the WHERE clause"


def test_api_list_queue_forwards_final_verdict() -> None:
    src = inspect.getsource(queue_api.list_queue)
    assert "final_verdict" in src, "the /queue endpoint must expose a final_verdict query param"
    assert "final_verdict=final_verdict" in src, "the endpoint must forward final_verdict to the repo"
