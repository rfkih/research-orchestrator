"""Phase 3 unit tests — sweep derivation.

The /tick path is integration-shaped (DB + JVM); those tests live behind
the ``integration`` marker and require pytest-postgresql + respx, both
deferred. The pieces below are pure functions and pin the bash-parity
logic. Verdict tests moved to test_phase4.py once analyze.py replaced
the placeholder ``services/verdict.py``.
"""

from __future__ import annotations

from orchestrator.services.sweep import derive_combo, total_combos


# ── sweep ──────────────────────────────────────────────────────────────


def test_sweep_grid_first_combo_is_first_value_per_param() -> None:
    cfg = {"params": [{"name": "a", "values": [1, 2]}, {"name": "b", "values": ["x", "y"]}]}
    assert derive_combo(cfg, 0) == {"a": 1, "b": "x"}


def test_sweep_grid_iterates_left_to_right_first_param_slowest() -> None:
    # itertools.product order: a outer, b inner.
    cfg = {"params": [{"name": "a", "values": [1, 2]}, {"name": "b", "values": ["x", "y"]}]}
    combos = [derive_combo(cfg, i) for i in range(4)]
    assert combos == [
        {"a": 1, "b": "x"},
        {"a": 1, "b": "y"},
        {"a": 2, "b": "x"},
        {"a": 2, "b": "y"},
    ]


def test_sweep_grid_returns_none_when_exhausted() -> None:
    cfg = {"params": [{"name": "a", "values": [1, 2]}]}
    assert derive_combo(cfg, 2) is None


def test_sweep_no_params_returns_empty_combo_only_for_iter_zero() -> None:
    assert derive_combo({}, 0) == {}
    assert derive_combo({}, 1) is None


def test_total_combos_multiplies_param_values() -> None:
    cfg = {"params": [{"name": "a", "values": [1, 2, 3]}, {"name": "b", "values": ["x", "y"]}]}
    assert total_combos(cfg) == 6
    assert total_combos({}) == 1
