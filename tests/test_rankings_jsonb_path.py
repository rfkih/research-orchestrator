"""Regression guard for the rankings JSONB key-path bug.

``tick.py`` builds ``metrics_snapshot = {**run_metrics, "analysis": analysis}``,
so the analyzer-computed metrics (annualized return, DSR, PSR, Sharpe,
pf_point_estimate, n_trades) live NESTED under ``metrics_snapshot->'analysis'``.

The ranking query previously read them at the top level, so the
``WHERE ...->>'annualized_geometric_return_pct_at_alloc_90' IS NOT NULL``
filter excluded every row and ``/rankings`` returned ``[]``. These pins assert
the analyzer keys are read through ``->'analysis'->>`` and that ``win_rate``
(a run-level column, not in the analysis dict) stays top-level.
"""
from __future__ import annotations

from orchestrator.repo.rankings import _RANKING_SQL

# Analyzer-computed metrics that live under metrics_snapshot->'analysis'.
_ANALYSIS_KEYS = [
    "annualized_geometric_return_pct_at_alloc_90",
    "sharpe_annualized",
    "max_drawdown_pct",
    "n_trades",
    "pf_point_estimate",
    "dsr",
    "psr",
]


def test_analyzer_metrics_read_from_analysis_subobject() -> None:
    for key in _ANALYSIS_KEYS:
        nested = f"metrics_snapshot->'analysis'->>'{key}'"
        assert nested in _RANKING_SQL, (
            f"ranking SQL must read analyzer key {key!r} from the 'analysis' "
            f"sub-object; expected substring {nested!r}."
        )


def test_no_analyzer_metric_read_at_top_level() -> None:
    # The bug was reading these directly off metrics_snapshot. Ensure no
    # top-level ``metrics_snapshot->>'<analyzer key>'`` extraction remains.
    for key in _ANALYSIS_KEYS:
        top_level = f"metrics_snapshot->>'{key}'"
        assert top_level not in _RANKING_SQL, (
            f"ranking SQL still reads analyzer key {key!r} at the top level "
            f"({top_level!r}); it lives under ->'analysis'->> and will be NULL."
        )


def test_win_rate_stays_top_level() -> None:
    # win_rate is a run-level column (not present in the analyzer dict), so it
    # must be read top-level, NOT under 'analysis'.
    assert "metrics_snapshot->>'win_rate'" in _RANKING_SQL
    assert "metrics_snapshot->'analysis'->>'win_rate'" not in _RANKING_SQL


def test_null_filter_uses_nested_path() -> None:
    # The WHERE filter that previously excluded every row must use the nested
    # path so real rows survive.
    assert (
        "metrics_snapshot->'analysis'->>'annualized_geometric_return_pct_at_alloc_90'"
        ") IS NOT NULL" in _RANKING_SQL
    )
