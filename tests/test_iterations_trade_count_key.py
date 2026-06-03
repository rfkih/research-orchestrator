"""Regression guard for the iterations / agent_state ``trade_count`` key drift.

``tick.py`` builds ``metrics_snapshot = {**run_metrics, "analysis": analysis}``.
The run-level trade count is the TOP-LEVEL ``total_trades`` key (sourced from
``backtest_run.total_trades``); there is no ``trade_count`` key anywhere in the
snapshot. (The analyzer's own count lives nested at
``metrics_snapshot->'analysis'->>'n_trades'``.)

Reading ``metrics_snapshot->>'trade_count'`` therefore always yielded NULL, so:
  * the leaderboard ``sort=trade_count`` ordering silently no-op'd, and
  * ``agent_state._last_iterations`` reported ``n_trades=None`` for every row.

These pins assert all three read sites use the real ``total_trades`` key and
that the phantom ``trade_count`` key is gone. Sibling of
``test_rankings_jsonb_path.py`` (same key-drift bug class).
"""
from __future__ import annotations

import inspect

from orchestrator.repo import agent_state, iterations
from orchestrator.repo.iterations import _LEADERBOARD_SORTS


def test_leaderboard_sort_expression_uses_total_trades() -> None:
    expr = _LEADERBOARD_SORTS["trade_count"]
    assert "total_trades" in expr, (
        "leaderboard trade_count sort must order on the real ``total_trades`` "
        f"key; got {expr!r}."
    )
    assert "'trade_count'" not in expr, (
        "leaderboard sort still references the phantom ``trade_count`` JSONB key."
    )


def test_leaderboard_select_reads_total_trades() -> None:
    src = inspect.getsource(iterations.leaderboard)
    assert "metrics_snapshot->>'total_trades'" in src
    assert "metrics_snapshot->>'trade_count'" not in src, (
        "iterations.leaderboard still SELECTs the phantom ``trade_count`` key "
        "(always NULL); read ``total_trades`` instead."
    )


def test_agent_state_last_iterations_reads_total_trades() -> None:
    src = inspect.getsource(agent_state._last_iterations)
    assert "metrics_snapshot->>'total_trades'" in src
    assert "metrics_snapshot->>'trade_count'" not in src, (
        "agent_state._last_iterations still reads the phantom ``trade_count`` "
        "key (always NULL); read ``total_trades`` instead."
    )
