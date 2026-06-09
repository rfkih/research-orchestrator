"""Pin tests for the walk-forward bear-coverage gate (2026-06-09).

A walk-forward is the graduation gate's only out-of-sample evidence. If its
window has no bear regime, a ROBUST verdict proves nothing for a trend hedge
(which cannot fail a bull by construction). These tests pin the constants and
the ``window_covers_bear`` predicate so the gate can't silently regress to the
old bull-only default.
"""

from __future__ import annotations

from datetime import date

from orchestrator.services.walk_forward import (
    DEFAULT_FULL_START,
    KNOWN_BEAR_REGIMES,
    MIN_BEAR_OVERLAP_DAYS,
    window_covers_bear,
)


def test_bear_coverage_constants_are_pinned() -> None:
    # Operator-controlled. Change only with a documented ORCHESTRATOR_CHANGE
    # journal entry — loosening this re-opens the bull-only-window blind spot.
    assert MIN_BEAR_OVERLAP_DAYS == 45
    assert DEFAULT_FULL_START == date(2018, 1, 1)
    assert KNOWN_BEAR_REGIMES == (
        (date(2018, 1, 17), date(2018, 12, 15)),
        (date(2020, 2, 19), date(2020, 3, 23)),
        (date(2021, 11, 10), date(2022, 12, 31)),
    )


def test_old_default_window_2024_is_rejected() -> None:
    # The exact window that motivated this fix: 2024-01-01 onward is bull-only.
    assert window_covers_bear(date(2024, 1, 1), date(2026, 6, 8)) is False


def test_full_history_window_covers_bear() -> None:
    # The new default spans all three pinned bears.
    assert window_covers_bear(DEFAULT_FULL_START, date(2026, 6, 8)) is True


def test_window_covering_2022_bear_passes() -> None:
    assert window_covers_bear(date(2021, 6, 1), date(2023, 1, 1)) is True


def test_window_covering_only_2020_covid_passes() -> None:
    # Short-history instrument (e.g. SOL from 2020) still gets the COVID crash
    # and the 2022 bear.
    assert window_covers_bear(date(2020, 1, 1), date(2023, 6, 1)) is True


def test_clipping_last_week_of_a_bear_does_not_count() -> None:
    # Window starts 2022-12-25 — only 6 days overlap with the 2021-22 bear that
    # ends 2022-12-31. Below the 45-day materiality floor → not covered.
    assert window_covers_bear(date(2022, 12, 25), date(2024, 1, 1)) is False


def test_overlap_threshold_boundary() -> None:
    # 2018 bear starts 2018-01-17. Overlap is measured in days from that start.
    # 44-day overlap is below the 45-day floor (fails); 45 days clears it.
    from datetime import timedelta

    bear_start = date(2018, 1, 17)
    assert window_covers_bear(bear_start, bear_start + timedelta(days=44)) is False
    assert window_covers_bear(bear_start, bear_start + timedelta(days=45)) is True


def test_gap_between_bears_is_not_covered() -> None:
    # 2019 was a recovery year — no pinned bear. A 2019-only window fails.
    assert window_covers_bear(date(2019, 1, 1), date(2019, 12, 1)) is False
