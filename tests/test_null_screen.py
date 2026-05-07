"""Pure-function tests for the null-screen service.

Covers:
  * ``generate_random_combos`` — deterministic w/ seed, respects type
    bounds, surfaces bad ranges as OrchestratorError.
  * ``summarize_pf_distribution`` — handles empty / all-None / mixed.
  * ``null_screen_verdict`` — pinned thresholds; pin tests guard against
    accidental drift of the constants.
  * ``_next_actions`` — verdict → next_action mapping is stable.

Async DB-touching paths (run_null_screen) need pytest-postgresql + a fake
JVM and live behind the integration marker — not in this file.
"""

from __future__ import annotations

import math

import pytest

from orchestrator.errors import OrchestratorError
from orchestrator.api.journal import _VALID_TYPE as _JOURNAL_VALID_TYPE
from orchestrator.services.null_screen import (
    MAX_CONSECUTIVE_FAILURES,
    MIN_USEFUL_PF_FRACTION,
    P75_EDGE_THRESHOLD,
    P95_NO_EDGE_THRESHOLD,
    SHARE_ABOVE_EDGE_FRACTION,
    SHARE_ABOVE_NO_EDGE_FRACTION,
    _next_actions,
    generate_random_combos,
    journal_entry_type_for,
    null_screen_verdict,
    summarize_pf_distribution,
)


# ── Pin tests for operator-controlled constants ───────────────────────


def test_null_screen_thresholds_are_pinned() -> None:
    # Changing these values shifts the screen's strictness — operator-
    # controlled constants. If you intentionally tune one, update both
    # the constant AND this pin test, and journal an ORCHESTRATOR_CHANGE.
    assert P75_EDGE_THRESHOLD == 1.2
    assert P95_NO_EDGE_THRESHOLD == 1.2
    assert SHARE_ABOVE_EDGE_FRACTION == 0.25
    assert SHARE_ABOVE_NO_EDGE_FRACTION == 0.10
    assert MIN_USEFUL_PF_FRACTION == 0.5
    assert MAX_CONSECUTIVE_FAILURES == 3


# ── generate_random_combos ────────────────────────────────────────────


def test_generate_random_combos_is_deterministic() -> None:
    ranges = [
        {"name": "ATR_EXT", "type": "float", "low": "1.0", "high": "3.0"},
        {"name": "RSI_EXT", "type": "int", "low": "10", "high": "40"},
    ]
    a = generate_random_combos(ranges, n_draws=5, seed=42)
    b = generate_random_combos(ranges, n_draws=5, seed=42)
    assert a == b


def test_generate_random_combos_different_seeds_diverge() -> None:
    ranges = [{"name": "X", "type": "float", "low": "0", "high": "1"}]
    a = generate_random_combos(ranges, n_draws=5, seed=1)
    b = generate_random_combos(ranges, n_draws=5, seed=2)
    assert a != b


def test_generate_random_combos_respects_int_bounds() -> None:
    ranges = [{"name": "K", "type": "int", "low": "5", "high": "9"}]
    out = generate_random_combos(ranges, n_draws=20, seed=42)
    for combo in out:
        v = int(combo["K"])
        assert 5 <= v <= 9


def test_generate_random_combos_respects_float_bounds() -> None:
    ranges = [{"name": "X", "type": "float", "low": "1.5", "high": "2.5"}]
    out = generate_random_combos(ranges, n_draws=20, seed=42)
    for combo in out:
        v = float(combo["X"])
        assert 1.5 <= v <= 2.5


def test_generate_random_combos_choice_picks_from_values() -> None:
    ranges = [{"name": "TF", "type": "choice", "values": ["1h", "4h"]}]
    out = generate_random_combos(ranges, n_draws=20, seed=42)
    seen = {c["TF"] for c in out}
    assert seen.issubset({"1h", "4h"})
    assert len(seen) >= 1  # some variability — w/ 20 draws basically guaranteed both


def test_generate_random_combos_returns_string_values() -> None:
    # JVM expects string-encoded BigDecimals; combo values are always strings
    # so param_combo_hash is stable across writers.
    ranges = [
        {"name": "X", "type": "float", "low": "1.0", "high": "2.0"},
        {"name": "Y", "type": "int", "low": "10", "high": "20"},
        {"name": "Z", "type": "choice", "values": [0.1, 0.2]},
    ]
    out = generate_random_combos(ranges, n_draws=3, seed=42)
    for combo in out:
        for val in combo.values():
            assert isinstance(val, str)


def test_generate_random_combos_rejects_inverted_float_range() -> None:
    ranges = [{"name": "X", "type": "float", "low": "3.0", "high": "1.0"}]
    with pytest.raises(OrchestratorError) as exc:
        generate_random_combos(ranges, n_draws=3, seed=42)
    assert exc.value.envelope.error_code == "bad_param_range"


def test_generate_random_combos_rejects_empty_choice_values() -> None:
    ranges = [{"name": "TF", "type": "choice", "values": []}]
    with pytest.raises(OrchestratorError) as exc:
        generate_random_combos(ranges, n_draws=3, seed=42)
    assert exc.value.envelope.error_code == "bad_param_range"


# ── summarize_pf_distribution ─────────────────────────────────────────


def test_summarize_pf_empty_distribution() -> None:
    out = summarize_pf_distribution([])
    assert out["n_attempted"] == 0
    assert out["n_finite"] == 0
    assert out["mean"] is None
    assert out["share_pf_ge_1_2"] is None


def test_summarize_pf_all_failed() -> None:
    out = summarize_pf_distribution([None, None, None])
    assert out["n_attempted"] == 3
    assert out["n_finite"] == 0
    assert out["mean"] is None


def test_summarize_pf_mixed_finite_and_failed() -> None:
    out = summarize_pf_distribution([1.0, 1.5, 2.0, None, None])
    assert out["n_attempted"] == 5
    assert out["n_finite"] == 3
    assert out["mean"] == round((1.0 + 1.5 + 2.0) / 3, 4)
    assert out["max"] == 2.0


def test_summarize_pf_share_pf_ge_1_2() -> None:
    # Three at or above 1.2; two below.
    out = summarize_pf_distribution([0.8, 0.9, 1.2, 1.5, 1.8])
    assert out["share_pf_ge_1_2"] == 0.6


def test_summarize_pf_drops_non_finite() -> None:
    # NaN and infinity should be excluded — math.isfinite filters them.
    out = summarize_pf_distribution([1.0, 1.5, math.nan, math.inf])
    assert out["n_finite"] == 2


# ── null_screen_verdict ───────────────────────────────────────────────


def test_verdict_insufficient_data_when_too_few_finite() -> None:
    # K=8 attempts, only 3 finite (need ceil(K * 0.5) = 4)
    dist = summarize_pf_distribution([1.0, 1.5, 1.2, None, None, None, None, None])
    out = null_screen_verdict(dist, n_attempted=8)
    assert out["verdict"] == "INSUFFICIENT_DATA"


def test_verdict_edge_present_when_right_tail_strong() -> None:
    # Right-skewed: P75 well above 1.2, share>=1.2 well above 0.25
    pfs = [0.9, 1.0, 1.1, 1.3, 1.5, 1.8, 2.0, 2.5]
    dist = summarize_pf_distribution(pfs)
    out = null_screen_verdict(dist, n_attempted=len(pfs))
    assert out["verdict"] == "EDGE_PRESENT"


def test_verdict_no_edge_when_p95_weak_and_share_low() -> None:
    # All centered around 0.95; P95 well below 1.2, almost none above 1.2
    pfs = [0.85, 0.9, 0.92, 0.95, 0.98, 1.0, 1.02, 1.05]
    dist = summarize_pf_distribution(pfs)
    out = null_screen_verdict(dist, n_attempted=len(pfs))
    assert out["verdict"] == "NO_EDGE_DETECTED"


def test_verdict_inconclusive_in_grey_zone() -> None:
    # P95 above 1.2 (so not NO_EDGE), but P75 below 1.2 / share below 0.25
    # so not EDGE_PRESENT either.
    pfs = [0.8, 0.9, 1.0, 1.05, 1.1, 1.15, 1.18, 1.4]
    dist = summarize_pf_distribution(pfs)
    out = null_screen_verdict(dist, n_attempted=len(pfs))
    assert out["verdict"] == "INCONCLUSIVE"


# ── _next_actions ─────────────────────────────────────────────────────


def test_next_actions_edge_present_points_to_queue() -> None:
    out = _next_actions("EDGE_PRESENT", "MMR")
    assert out[0]["method"] == "POST"
    assert out[0]["path"] == "/queue"


def test_next_actions_no_edge_does_not_point_to_queue() -> None:
    # The whole point of the screen: NO_EDGE means do NOT enqueue.
    out = _next_actions("NO_EDGE_DETECTED", "MMR")
    paths = [a.get("path") for a in out]
    assert "/queue" not in paths


def test_next_actions_inconclusive_recommends_small_grid() -> None:
    out = _next_actions("INCONCLUSIVE", "MMR")
    assert out[0]["method"] == "POST"
    assert out[0]["path"] == "/queue"


# ── Journal entry_type mapping (CHECK constraint compliance) ──────────


def test_journal_entry_type_no_edge_is_anti_pattern() -> None:
    # NO_EDGE_DETECTED maps to ANTI_PATTERN so the agent's cold-boot
    # recipe naturally surfaces it without an extra query.
    assert journal_entry_type_for("NO_EDGE_DETECTED") == "ANTI_PATTERN"


def test_journal_entry_type_other_verdicts_are_strategy_outcome() -> None:
    for verdict in ("EDGE_PRESENT", "INCONCLUSIVE", "INSUFFICIENT_DATA"):
        assert journal_entry_type_for(verdict) == "STRATEGY_OUTCOME"


def test_journal_entry_types_are_in_check_constraint() -> None:
    # The DB rejects entry_types not in research_journal's CHECK list.
    # If this test fails after a verdict mapping change, add the new type
    # via Flyway migration in the trading JVM repo before changing here.
    for verdict in ("NO_EDGE_DETECTED", "EDGE_PRESENT", "INCONCLUSIVE", "INSUFFICIENT_DATA"):
        assert journal_entry_type_for(verdict) in _JOURNAL_VALID_TYPE
