"""Pure-function tests for the /signal-screen IC math.

Covers: pinned constants, transforms, forward log returns, the
point-in-time alignment no-look-ahead invariant, Spearman/rank parity
with combination_book, Newey-West known answers, quantile spread,
seeded block bootstrap, verdict mapping, the compute core (perfect
signal -> IC ~= 1 PROMISING; noise -> not PROMISING), and the journal
entry_type CHECK-constraint compliance.

DB-touching paths (run_signal_screen) are exercised via the endpoint
tests with mocked repos in tests/test_signal_screen_api.py.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta

import numpy as np
import pytest

from orchestrator.api.journal import _VALID_TYPE as _JOURNAL_VALID_TYPE
from orchestrator.services.combination_book import _rank as _rank_loop
from orchestrator.services.signal_screen import (
    BOOTSTRAP_DRAWS,
    CI_LEVEL,
    MIN_OBS,
    QUANTILES_DEFAULT,
    QUANTILES_SMALL,
    RNG_SEED,
    SMALL_N_FOR_TERCILES,
    T_DEAD,
    T_PROMISING,
    _next_actions,
    _rank_vec,
    align_point_in_time,
    block_bootstrap_ic_ci,
    compute_screen_rows,
    forward_log_returns,
    journal_entry_type_for,
    newey_west_tstat,
    overall_verdict,
    quantile_spread,
    row_verdict,
    signal_family,
    transform_series,
)

T0 = datetime(2024, 1, 1)


def _bars(closes: list[float]) -> list[tuple[datetime, float]]:
    return [(T0 + timedelta(hours=k), c) for k, c in enumerate(closes)]


def _random_walk(n: int, seed: int = 7) -> list[float]:
    rng = random.Random(seed)
    out = [100.0]
    for _ in range(n - 1):
        out.append(out[-1] * math.exp(rng.gauss(0, 0.01)))
    return out


# -- Pinned operator-controlled constants ------------------------------


def test_signal_screen_constants_are_pinned() -> None:
    # Changing these shifts the screen's strictness / reproducibility.
    # If you intentionally tune one, update the constant AND this pin
    # test, and journal the change.
    assert T_PROMISING == 2.0
    assert T_DEAD == 1.0
    assert BOOTSTRAP_DRAWS == 1000
    assert CI_LEVEL == 0.95
    assert RNG_SEED == 42
    assert MIN_OBS == 30
    assert QUANTILES_DEFAULT == 5
    assert QUANTILES_SMALL == 3
    assert SMALL_N_FOR_TERCILES == 200


# -- signal_family ------------------------------------------------------


def test_signal_family_strips_digit_runs() -> None:
    assert signal_family("ema_100") == "ema"
    assert signal_family("ema_150") == "ema"
    assert signal_family("btc_dvol_zscore_30d") == "btc_dvol_zscore_d"


def test_signal_family_is_stable_for_no_digit_names() -> None:
    assert signal_family("funding_rate") == "funding_rate"


# -- transform_series ---------------------------------------------------


def test_transform_zscore_standardizes() -> None:
    out = transform_series([1.0, 2.0, 3.0, 4.0], "zscore")
    a = np.asarray(out)
    assert abs(a.mean()) < 1e-12
    assert abs(a.std() - 1.0) < 1e-12


def test_transform_zscore_constant_series_is_zeros() -> None:
    assert transform_series([5.0, 5.0, 5.0], "zscore") == [0.0, 0.0, 0.0]


def test_transform_rank_is_monotone_unit_interval() -> None:
    out = transform_series([10.0, -3.0, 7.0], "rank")
    assert out[1] == 0.0 and out[0] == 1.0  # min -> 0, max -> 1
    assert 0.0 < out[2] < 1.0


def test_transform_raw_is_identity() -> None:
    vals = [3.0, 1.0, 2.0]
    assert transform_series(vals, "raw") == vals


def test_transform_unknown_kind_raises() -> None:
    with pytest.raises(ValueError):
        transform_series([1.0], "log")


# -- forward_log_returns -------------------------------------------------


def test_forward_log_returns_known_answer() -> None:
    out = forward_log_returns([100.0, 110.0, 121.0], 1)
    assert out[0] == pytest.approx(math.log(1.1))
    assert out[1] == pytest.approx(math.log(1.1))
    assert out[2] is None  # no future bar


def test_forward_log_returns_horizon_spans_bars() -> None:
    out = forward_log_returns([100.0, 105.0, 121.0], 2)
    assert out[0] == pytest.approx(math.log(1.21))
    assert out[1] is None and out[2] is None


def test_forward_log_returns_nonpositive_close_is_none() -> None:
    out = forward_log_returns([100.0, 0.0, 121.0], 1)
    assert out[0] is None  # forward close is 0
    assert out[1] is None  # base close is 0


# -- align_point_in_time: the no-look-ahead invariant --------------------


def test_pit_value_stamped_after_bar_start_is_invisible() -> None:
    # REGRESSION TEST (non-negotiable): a macro value stamped one
    # microsecond AFTER the bar start must NOT be visible at that bar.
    bar = datetime(2024, 1, 1, 10, 0, 0)
    later = bar + timedelta(microseconds=1)
    out = align_point_in_time([bar, bar + timedelta(hours=1)], [(later, 42.0)])
    assert out[0] is None          # not yet published at bar start
    assert out[1] == 42.0          # visible from the next bar on


def test_pit_uses_latest_value_at_or_before_bar_start() -> None:
    bars = [datetime(2024, 1, 1, h) for h in (8, 9, 10)]
    points = [
        (datetime(2024, 1, 1, 7, 30), 1.0),
        (datetime(2024, 1, 1, 9, 0), 2.0),   # ts == bar start -> visible
        (datetime(2024, 1, 1, 9, 30), 3.0),
    ]
    assert align_point_in_time(bars, points) == [1.0, 2.0, 3.0]


def test_pit_bars_before_first_point_are_none() -> None:
    bars = [datetime(2024, 1, 1), datetime(2024, 1, 2)]
    points = [(datetime(2024, 1, 2), 9.0)]
    assert align_point_in_time(bars, points) == [None, 9.0]


def test_pit_carries_stale_value_forward() -> None:
    bars = [datetime(2024, 1, d) for d in (1, 2, 3, 4)]
    points = [(datetime(2024, 1, 1), 5.0)]
    assert align_point_in_time(bars, points) == [5.0, 5.0, 5.0, 5.0]


# -- _rank_vec parity with combination_book._rank ------------------------


def test_rank_vec_matches_combination_book_rank() -> None:
    rng = np.random.default_rng(3)
    a = rng.normal(size=200)
    a[10:20] = a[0]  # inject ties
    assert np.allclose(_rank_vec(a), _rank_loop(a))


# -- newey_west_tstat -----------------------------------------------------


def test_nw_tstat_perfect_signal_is_capped_huge() -> None:
    rng = random.Random(1)
    y = [rng.gauss(0, 1) for _ in range(100)]
    t = newey_west_tstat(y, y, lag=1)
    assert t == pytest.approx(1e6)  # zero residual variance -> +cap


def test_nw_tstat_pure_noise_is_small() -> None:
    rng = random.Random(2)
    x = [rng.gauss(0, 1) for _ in range(500)]
    y = [rng.gauss(0, 1) for _ in range(500)]
    t = newey_west_tstat(x, y, lag=6)
    assert t is not None and abs(t) < 2.0


def test_nw_tstat_strong_linear_signal_is_significant() -> None:
    rng = random.Random(3)
    x = [rng.gauss(0, 1) for _ in range(400)]
    y = [xi + rng.gauss(0, 0.5) for xi in x]
    t = newey_west_tstat(x, y, lag=6)
    assert t is not None and t > 2.0


def test_nw_lag_widens_se_under_positive_autocorrelation() -> None:
    # Overlapping forward windows induce positive serial correlation in
    # the residuals; the Bartlett kernel must WIDEN the standard error
    # (shrink |t|) relative to the iid lag=0 case.
    rng = random.Random(4)
    n, h = 400, 12
    shocks = [rng.gauss(0, 1) for _ in range(n + h)]
    # h-overlapping moving sums: classic overlapping-window errors.
    y = [sum(shocks[i:i + h]) for i in range(n)]
    x = [rng.gauss(0, 1) * 0.001 + yi for yi in y]  # near-collinear + noise
    t_iid = newey_west_tstat(x, y, lag=0)
    t_nw = newey_west_tstat(x, y, lag=h)
    assert t_iid is not None and t_nw is not None
    assert abs(t_nw) <= abs(t_iid)


def test_nw_tstat_undefined_on_thin_or_constant_input() -> None:
    assert newey_west_tstat([1.0] * 5, [1.0] * 5, lag=1) is None  # n < 8
    assert newey_west_tstat([2.0] * 50, list(range(50)), lag=1) is None  # sxx = 0


# -- quantile_spread ------------------------------------------------------


def test_quantile_spread_positive_for_predictive_signal() -> None:
    n = 300  # >= SMALL_N_FOR_TERCILES -> quintiles
    sig = [float(i) for i in range(n)]
    fwd = [s / 100.0 for s in sig]  # higher signal -> higher return
    out = quantile_spread(sig, fwd)
    assert out["q"] == QUANTILES_DEFAULT
    assert out["top_mean"] > out["bottom_mean"]
    assert out["spread"] > 0


def test_quantile_spread_uses_terciles_below_small_n() -> None:
    n = SMALL_N_FOR_TERCILES - 1
    out = quantile_spread([float(i) for i in range(n)], [0.0] * n)
    assert out["q"] == QUANTILES_SMALL


def test_quantile_spread_too_few_obs_returns_none_means() -> None:
    out = quantile_spread([1.0, 2.0], [0.1, 0.2])
    assert out["top_mean"] is None and out["spread"] is None


def test_quantile_spread_zero_for_flat_returns() -> None:
    n = 250
    out = quantile_spread([float(i) for i in range(n)], [0.01] * n)
    assert out["spread"] == pytest.approx(0.0)


# -- block_bootstrap_ic_ci -------------------------------------------------


def test_bootstrap_is_deterministic() -> None:
    rng = random.Random(5)
    sig = [rng.gauss(0, 1) for _ in range(80)]
    fwd = [rng.gauss(0, 1) for _ in range(80)]
    a = block_bootstrap_ic_ci([(sig, fwd)], horizon=6)
    b = block_bootstrap_ic_ci([(sig, fwd)], horizon=6)
    assert a == b


def test_bootstrap_perfect_signal_ci_excludes_zero() -> None:
    rng = random.Random(6)
    y = [rng.gauss(0, 1) for _ in range(100)]
    out = block_bootstrap_ic_ci([(y, y)], horizon=1)
    assert out["excludes_zero"] is True
    assert out["ci_low"] > 0.9


def test_bootstrap_noise_ci_includes_zero() -> None:
    rng = random.Random(7)
    sig = [rng.gauss(0, 1) for _ in range(200)]
    fwd = [rng.gauss(0, 1) for _ in range(200)]
    out = block_bootstrap_ic_ci([(sig, fwd)], horizon=6)
    assert out["excludes_zero"] is False
    assert out["ci_low"] < 0.0 < out["ci_high"]


def test_bootstrap_block_len_tracks_horizon() -> None:
    rng = random.Random(8)
    sig = [rng.gauss(0, 1) for _ in range(60)]
    fwd = [rng.gauss(0, 1) for _ in range(60)]
    out = block_bootstrap_ic_ci([(sig, fwd)], horizon=24)
    assert out["block_len"] == 24


def test_bootstrap_insufficient_obs_fails_closed() -> None:
    out = block_bootstrap_ic_ci([([1.0] * 5, [1.0] * 5)], horizon=1)
    assert out["excludes_zero"] is False
    assert out["reason"] == "insufficient_obs"


# -- verdict mapping --------------------------------------------------------


def test_row_verdict_promising_needs_t_and_ci() -> None:
    assert row_verdict(n_obs=100, nw_tstat=2.5, ci_excludes_zero=True) == "PROMISING"
    assert row_verdict(n_obs=100, nw_tstat=2.5, ci_excludes_zero=False) == "WEAK"
    assert row_verdict(n_obs=100, nw_tstat=1.5, ci_excludes_zero=True) == "WEAK"


def test_row_verdict_dead_needs_low_t_and_ci_zero() -> None:
    assert row_verdict(n_obs=100, nw_tstat=0.4, ci_excludes_zero=False) == "DEAD"
    assert row_verdict(n_obs=100, nw_tstat=None, ci_excludes_zero=False) == "DEAD"
    assert row_verdict(n_obs=100, nw_tstat=0.4, ci_excludes_zero=True) == "WEAK"


def test_row_verdict_insufficient_below_min_obs() -> None:
    v = row_verdict(n_obs=MIN_OBS - 1, nw_tstat=5.0, ci_excludes_zero=True)
    assert v == "INSUFFICIENT_DATA"


def test_overall_verdict_aggregation() -> None:
    assert overall_verdict(["DEAD", "PROMISING", "WEAK"]) == "PROMISING"
    assert overall_verdict(["DEAD", "DEAD"]) == "DEAD"
    assert overall_verdict(["DEAD", "INSUFFICIENT_DATA"]) == "DEAD"
    assert overall_verdict(["DEAD", "WEAK"]) == "WEAK"
    assert overall_verdict(["INSUFFICIENT_DATA"]) == "INSUFFICIENT_DATA"
    assert overall_verdict([]) == "INSUFFICIENT_DATA"


def test_journal_entry_types_are_in_check_constraint() -> None:
    # The DB CHECK constraint (V1 baseline + V127) rejects unknown
    # entry_types -- SIGNAL_SCREEN is NOT one, hence the kind= JSONB
    # discriminator. The mapped types must stay inside the allowed set.
    for verdict in ("PROMISING", "WEAK", "DEAD", "INSUFFICIENT_DATA"):
        assert journal_entry_type_for(verdict) in _JOURNAL_VALID_TYPE


def test_journal_entry_type_dead_is_anti_pattern() -> None:
    assert journal_entry_type_for("DEAD") == "ANTI_PATTERN"
    assert journal_entry_type_for("PROMISING") == "CROSS_STRATEGY_FINDING"


# -- compute_screen_rows (the pure core) -------------------------------------


def test_compute_perfect_signal_ic_near_one_and_promising() -> None:
    closes = _random_walk(400)
    bars = _bars(closes)
    h = 6
    sig = forward_log_returns(closes, h)  # signal IS the future return
    rows = compute_screen_rows(
        bars_by_instrument={"BTCUSDT": bars},
        signal_by_instrument={"BTCUSDT": sig},
        horizons_bars=[h],
        transform="zscore",
    )
    assert len(rows) == 1
    r = rows[0]
    assert r["ic"] == pytest.approx(1.0)
    assert r["verdict"] == "PROMISING"
    assert r["quantile_spread"]["spread"] > 0


def test_compute_noise_signal_is_not_promising() -> None:
    rng = random.Random(11)
    closes = _random_walk(400)
    bars = _bars(closes)
    noise = [rng.gauss(0, 1) for _ in range(400)]
    rows = compute_screen_rows(
        bars_by_instrument={"BTCUSDT": bars},
        signal_by_instrument={"BTCUSDT": noise},
        horizons_bars=[1, 6],
        transform="zscore",
    )
    for r in rows:
        assert r["verdict"] != "PROMISING"
        assert abs(r["nw_tstat"]) < 2.0


def test_compute_rows_are_ranked_by_score() -> None:
    closes = _random_walk(300)
    bars = _bars(closes)
    rng = random.Random(12)
    sig = [rng.gauss(0, 1) for _ in range(300)]
    rows = compute_screen_rows(
        bars_by_instrument={"BTCUSDT": bars},
        signal_by_instrument={"BTCUSDT": sig},
        horizons_bars=[1, 6, 24],
        transform="zscore",
    )
    scores = [r["rank_score"] for r in rows]
    assert scores == sorted(scores, reverse=True)


def test_compute_emits_pooled_row_for_multiple_instruments() -> None:
    closes_a = _random_walk(200, seed=21)
    closes_b = _random_walk(200, seed=22)
    rng = random.Random(23)
    sig_a = [rng.gauss(0, 1) for _ in range(200)]
    sig_b = [rng.gauss(0, 1) for _ in range(200)]
    rows = compute_screen_rows(
        bars_by_instrument={"BTCUSDT": _bars(closes_a), "ETHUSDT": _bars(closes_b)},
        signal_by_instrument={"BTCUSDT": sig_a, "ETHUSDT": sig_b},
        horizons_bars=[6],
        transform="zscore",
    )
    by_instrument = {r["instrument"] for r in rows}
    assert by_instrument == {"BTCUSDT", "ETHUSDT", "POOLED"}
    pooled = next(r for r in rows if r["instrument"] == "POOLED")
    assert pooled["n_obs"] > max(
        r["n_obs"] for r in rows if r["instrument"] != "POOLED"
    )


def test_compute_no_pooled_row_for_single_instrument() -> None:
    closes = _random_walk(100)
    rows = compute_screen_rows(
        bars_by_instrument={"BTCUSDT": _bars(closes)},
        signal_by_instrument={"BTCUSDT": [1.0] * 100},
        horizons_bars=[1],
        transform="raw",
    )
    assert all(r["instrument"] != "POOLED" for r in rows)


def test_compute_ic_is_transform_invariant() -> None:
    # Spearman IC is rank-based -> identical across monotonic transforms.
    closes = _random_walk(300, seed=31)
    rng = random.Random(32)
    sig = [rng.gauss(0, 1) for _ in range(300)]
    ics = {}
    for tr in ("zscore", "rank", "raw"):
        rows = compute_screen_rows(
            bars_by_instrument={"BTCUSDT": _bars(closes)},
            signal_by_instrument={"BTCUSDT": sig},
            horizons_bars=[6],
            transform=tr,
        )
        ics[tr] = rows[0]["ic"]
    assert ics["zscore"] == pytest.approx(ics["rank"], abs=1e-9)
    assert ics["zscore"] == pytest.approx(ics["raw"], abs=1e-9)


def test_compute_handles_missing_signal_values_pairwise() -> None:
    # None signal slots (PIT gap) are dropped pairwise, not imputed.
    closes = _random_walk(120, seed=41)
    sig: list = [None] * 60 + [1.0 * k for k in range(60)]
    rows = compute_screen_rows(
        bars_by_instrument={"BTCUSDT": _bars(closes)},
        signal_by_instrument={"BTCUSDT": sig},
        horizons_bars=[1],
        transform="zscore",
    )
    assert rows[0]["n_obs"] <= 59  # 60 non-None minus last-bar fwd gap


def test_compute_insufficient_when_signal_empty() -> None:
    closes = _random_walk(100)
    rows = compute_screen_rows(
        bars_by_instrument={"BTCUSDT": _bars(closes)},
        signal_by_instrument={"BTCUSDT": []},
        horizons_bars=[1],
        transform="zscore",
    )
    assert rows[0]["verdict"] == "INSUFFICIENT_DATA"
    assert rows[0]["n_obs"] == 0


# -- _next_actions ------------------------------------------------------------


def test_next_actions_promising_points_to_hypothesis() -> None:
    out = _next_actions("PROMISING", "btc_dvol")
    assert out[0]["path"] == "/journal"


def test_next_actions_dead_does_not_register_hypothesis() -> None:
    out = _next_actions("DEAD", "btc_dvol")
    paths = [a.get("path") for a in out]
    assert "/journal" not in paths and "/queue" not in paths
