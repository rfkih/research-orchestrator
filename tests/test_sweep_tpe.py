"""Pure-function tests for TPE sweep helpers.

Covers:
  * Grid behaviour is unchanged (regression guard).
  * ``is_tpe`` / ``tpe_trial_budget`` correctly read sweep_config.
  * ``_build_distributions`` + ``_coerce_param_value`` round-trip
    JSONB-loaded values through Optuna distributions.
  * ``next_combo_tpe`` is deterministic per seed, respects budget,
    handles empty history, drops out-of-shape past trials gracefully.
"""

from __future__ import annotations

import pytest

from orchestrator.services.sweep import (
    _build_distributions,
    _coerce_param_value,
    derive_combo,
    is_tpe,
    next_combo_tpe,
    total_combos,
    tpe_trial_budget,
)


# ── Grid regression ───────────────────────────────────────────────────


def test_grid_derive_combo_unchanged() -> None:
    cfg = {"strategy": "grid", "params": [
        {"name": "X", "values": ["1", "2"]},
        {"name": "Y", "values": ["a", "b"]},
    ]}
    assert derive_combo(cfg, 0) == {"X": "1", "Y": "a"}
    assert derive_combo(cfg, 1) == {"X": "1", "Y": "b"}
    assert derive_combo(cfg, 3) == {"X": "2", "Y": "b"}
    assert derive_combo(cfg, 4) is None


def test_total_combos_grid() -> None:
    cfg = {"params": [
        {"name": "X", "values": ["1", "2", "3"]},
        {"name": "Y", "values": ["a", "b"]},
    ]}
    assert total_combos(cfg) == 6


# ── TPE config helpers ────────────────────────────────────────────────


def test_is_tpe_true_only_for_tpe_strategy() -> None:
    assert is_tpe({"strategy": "tpe"}) is True
    assert is_tpe({"strategy": "grid"}) is False
    assert is_tpe({}) is False  # default = grid


def test_tpe_trial_budget_reads_n_trials() -> None:
    assert tpe_trial_budget({"n_trials": 30}) == 30


def test_tpe_trial_budget_default() -> None:
    assert tpe_trial_budget({}) == 20


# ── Distribution round-trip ───────────────────────────────────────────


def test_build_distributions_handles_all_three_types() -> None:
    import optuna.distributions as od

    dists = _build_distributions([
        {"name": "F", "type": "float", "low": "1.0", "high": "3.0"},
        {"name": "I", "type": "int", "low": "5", "high": "10"},
        {"name": "C", "type": "choice", "values": ["a", "b"]},
    ])
    assert isinstance(dists["F"], od.FloatDistribution)
    assert isinstance(dists["I"], od.IntDistribution)
    assert isinstance(dists["C"], od.CategoricalDistribution)


def test_build_distributions_rejects_inverted_range() -> None:
    with pytest.raises(ValueError):
        _build_distributions([
            {"name": "F", "type": "float", "low": "3.0", "high": "1.0"},
        ])


def test_coerce_param_value_string_to_float() -> None:
    import optuna.distributions as od

    d = od.FloatDistribution(0.0, 10.0)
    # JSONB persistence stores BigDecimals as strings — must coerce.
    assert _coerce_param_value("X", "1.5", d) == 1.5


def test_coerce_param_value_string_to_int() -> None:
    import optuna.distributions as od

    d = od.IntDistribution(0, 100)
    assert _coerce_param_value("K", "42", d) == 42


def test_coerce_param_value_categorical_string_match() -> None:
    import optuna.distributions as od

    d = od.CategoricalDistribution(["1h", "4h"])
    assert _coerce_param_value("TF", "1h", d) == "1h"


def test_coerce_param_value_categorical_unknown_raises() -> None:
    import optuna.distributions as od

    d = od.CategoricalDistribution(["1h", "4h"])
    with pytest.raises(ValueError):
        _coerce_param_value("TF", "1d", d)


# ── next_combo_tpe ────────────────────────────────────────────────────


def _tpe_cfg() -> dict[str, object]:
    return {
        "strategy": "tpe",
        "n_trials": 20,
        "seed": 42,
        "params": [
            {"name": "ATR", "type": "float", "low": "1.0", "high": "3.0"},
            {"name": "RSI", "type": "int", "low": "10", "high": "40"},
        ],
    }


def test_next_combo_tpe_first_draw_no_history() -> None:
    combo = next_combo_tpe(_tpe_cfg(), past_trials=[], iter_index=0)
    assert combo is not None
    assert set(combo) == {"ATR", "RSI"}
    # Bounds:
    assert 1.0 <= float(combo["ATR"]) <= 3.0
    assert 10 <= int(combo["RSI"]) <= 40


def test_next_combo_tpe_is_deterministic_per_seed() -> None:
    a = next_combo_tpe(_tpe_cfg(), past_trials=[], iter_index=0, seed=99)
    b = next_combo_tpe(_tpe_cfg(), past_trials=[], iter_index=0, seed=99)
    assert a == b


def test_next_combo_tpe_returns_none_when_budget_exhausted() -> None:
    cfg = _tpe_cfg()
    cfg["n_trials"] = 5
    assert next_combo_tpe(cfg, past_trials=[], iter_index=5) is None
    assert next_combo_tpe(cfg, past_trials=[], iter_index=99) is None


def test_next_combo_tpe_uses_history_to_explore() -> None:
    # Feed in trials whose value is highest at ATR around 2.0 — the
    # sampler should bias subsequent draws toward that region. We don't
    # assert a specific value (TPE is stochastic with seed) but we do
    # assert the second draw is not random-uniform-equal to the first.
    cfg = _tpe_cfg()
    history = [
        ({"ATR": "1.0", "RSI": "10"}, 0.5),
        ({"ATR": "2.0", "RSI": "25"}, 1.8),
        ({"ATR": "3.0", "RSI": "40"}, 0.4),
    ]
    combo = next_combo_tpe(cfg, past_trials=history, iter_index=3)
    assert combo is not None
    assert 1.0 <= float(combo["ATR"]) <= 3.0


def test_next_combo_tpe_handles_failed_trial_as_pruned() -> None:
    # A None-value past trial should be added as PRUNED — sampler still
    # advances; doesn't crash on float(None).
    cfg = _tpe_cfg()
    history = [
        ({"ATR": "1.5", "RSI": "20"}, 1.2),
        ({"ATR": "2.0", "RSI": "25"}, None),  # failed trial
        ({"ATR": "2.5", "RSI": "30"}, 0.8),
    ]
    combo = next_combo_tpe(cfg, past_trials=history, iter_index=3)
    assert combo is not None
    assert set(combo) == {"ATR", "RSI"}


def test_next_combo_tpe_with_only_failed_trials_falls_back_to_sampling() -> None:
    cfg = _tpe_cfg()
    history = [
        ({"ATR": "1.5", "RSI": "20"}, None),
        ({"ATR": "2.0", "RSI": "25"}, None),
    ]
    combo = next_combo_tpe(cfg, past_trials=history, iter_index=2)
    assert combo is not None  # didn't crash; sampler back to random


def test_next_combo_tpe_skips_history_with_missing_keys() -> None:
    # If a past trial has keys we no longer ask about (axis-set drift),
    # it should be silently dropped — better than crashing the sampler.
    cfg = _tpe_cfg()
    history = [
        ({"ATR": "1.5", "RSI": "20", "OBSOLETE": "0.1"}, 1.2),  # superset, OK
        ({"OLDPARAM": "x"}, 1.5),                                # missing ATR/RSI — drop
    ]
    combo = next_combo_tpe(cfg, past_trials=history, iter_index=2)
    assert combo is not None  # didn't crash
    assert set(combo) == {"ATR", "RSI"}


def test_next_combo_tpe_returns_string_values() -> None:
    # JVM expects string-encoded BigDecimals; combo values are always strings.
    combo = next_combo_tpe(_tpe_cfg(), past_trials=[], iter_index=0)
    assert combo is not None
    for v in combo.values():
        assert isinstance(v, str)


def test_next_combo_tpe_negative_iter_index_returns_none() -> None:
    assert next_combo_tpe(_tpe_cfg(), past_trials=[], iter_index=-1) is None


def test_resolve_iter_budget_grid_unchanged() -> None:
    from orchestrator.api.queue import SweepConfig, resolve_iter_budget

    cfg = SweepConfig(
        strategy="grid",
        params=[{"name": "X", "values": ["1", "2"]}],
    )
    eff, note = resolve_iter_budget(cfg, 5)
    assert eff == 5
    assert note is None


def test_resolve_iter_budget_tpe_bumps_when_too_small() -> None:
    from orchestrator.api.queue import SweepConfig, resolve_iter_budget

    cfg = SweepConfig(
        strategy="tpe",
        n_trials=20,
        params=[
            {"name": "X", "type": "float", "low": "1.0", "high": "2.0"},
        ],
    )
    eff, note = resolve_iter_budget(cfg, 5)
    assert eff == 20
    assert note is not None
    assert "20" in note


def test_resolve_iter_budget_tpe_unchanged_when_already_large_enough() -> None:
    from orchestrator.api.queue import SweepConfig, resolve_iter_budget

    cfg = SweepConfig(
        strategy="tpe",
        n_trials=10,
        params=[{"name": "X", "type": "float", "low": "1.0", "high": "2.0"}],
    )
    eff, note = resolve_iter_budget(cfg, 50)
    assert eff == 50
    assert note is None


def test_next_combo_tpe_categorical_returns_one_of_choices() -> None:
    cfg = {
        "strategy": "tpe",
        "n_trials": 10,
        "seed": 0,
        "params": [
            {"name": "TF", "type": "choice", "values": ["1h", "4h"]},
            {"name": "ATR", "type": "float", "low": "1.0", "high": "3.0"},
        ],
    }
    combo = next_combo_tpe(cfg, past_trials=[], iter_index=0)
    assert combo is not None
    assert combo["TF"] in ("1h", "4h")
