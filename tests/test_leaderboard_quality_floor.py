"""Regression guards for the leaderboard quality floor (2026-06-14).

Bug class: ``GET /leaderboard`` defaulted to ``sort=pf`` with no minimum
trade count and no PF-sentinel handling. A backtest with zero losing trades
gets ``profit_factor = 9999`` (``analyze.PF_INFINITY_SENTINEL``); a single
lucky 1-trade run therefore lands at the TOP of a naive ``PF DESC`` board,
ranking a degenerate artifact above every real strategy (observed: VBO and
CARRY n=1 rows ranking #1). The verdict column already said
``INSUFFICIENT_EVIDENCE`` — only the *rank position* was misleading.

Two display-only fixes (neither touches the V11/V60 gate):
  * the ``pf`` sort wraps the metric in ``NULLIF(..., 9999)`` so the no-loss
    sentinel sorts LAST, never above a finite PF, and
  * a ``min_trades`` floor (route default 20) keeps thin n=1-2 rows off the
    board entirely.

Sibling of ``test_iterations_trade_count_key.py`` (same board-read bug class).
"""
from __future__ import annotations

import inspect

from orchestrator.api import iterations as iterations_api
from orchestrator.repo import iterations as iterations_repo
from orchestrator.repo.iterations import (
    _LEADERBOARD_SORTS,
    _PF_INFINITY_SENTINEL,
)


def test_pf_sort_nulls_out_the_infinity_sentinel() -> None:
    expr = _LEADERBOARD_SORTS["pf"]
    assert "NULLIF" in expr, (
        "pf sort must NULLIF the no-loss 9999 sentinel so a 1-trade PF=9999 "
        f"cannot outrank a real finite PF; got {expr!r}."
    )
    assert _PF_INFINITY_SENTINEL in expr, (
        "pf sort must reference the PF infinity sentinel value "
        f"({_PF_INFINITY_SENTINEL}); got {expr!r}."
    )


def test_pf_sentinel_matches_analyzer() -> None:
    # The board sentinel must track the analyzer's PF_INFINITY_SENTINEL.
    from orchestrator.services.analyze import PF_INFINITY_SENTINEL

    assert _PF_INFINITY_SENTINEL == str(int(PF_INFINITY_SENTINEL)), (
        "leaderboard _PF_INFINITY_SENTINEL drifted from "
        "analyze.PF_INFINITY_SENTINEL — they must stay in lockstep."
    )


def test_repo_leaderboard_has_min_trades_param() -> None:
    sig = inspect.signature(iterations_repo.leaderboard)
    assert "min_trades" in sig.parameters, (
        "repo.leaderboard must accept a min_trades floor."
    )


def test_repo_leaderboard_filters_on_total_trades() -> None:
    src = inspect.getsource(iterations_repo.leaderboard)
    assert "min_trades > 0" in src, (
        "repo.leaderboard must apply the min_trades floor only when > 0 "
        "(min_trades=0 must show every row)."
    )
    assert "(metrics_snapshot->>'total_trades')::int >= $" in src, (
        "min_trades floor must compare against the real total_trades key."
    )


def test_route_default_min_trades_is_a_sane_floor() -> None:
    sig = inspect.signature(iterations_api.leaderboard)
    default = sig.parameters["min_trades"].default
    # FastAPI Query(...) default — pull the wrapped value.
    floor = getattr(default, "default", default)
    assert floor == 20, (
        "leaderboard route default min_trades must be 20 (display sanity "
        f"floor, well below the V11 n>=100 gate); got {floor!r}."
    )
