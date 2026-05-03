"""Curated regime-labeled epochs for cross-window backtesting.

Phase 5 — multi-window validation. Each entry is a *named, regime-tagged*
date range that a strategy must hold up across. Distinct from walk-forward
folds, which are chronological and can all share the same regime (e.g.
6 rolling folds inside a single bull run).

Quants edit this file directly when curating new windows. Bump
``CATALOG_VERSION`` whenever adding/removing/relabeling — the version is
persisted on every cross_window_run row so we can tell which catalog a
historical verdict came from.

Selection criteria for a window:
  * Window length: ~6-12 months. Shorter than that and per-window trade
    counts are too thin to judge; longer and you're back to chronological
    folds.
  * Regime tag: BULL | BEAR | CHOP | TRANSITION. We label by hand based on
    BTC price structure, not auto-derived — avoids circularity with
    feature_store regime columns the strategy itself may use.
  * Coverage: at least one window per regime tag. Otherwise the verdict
    "ROBUST_CROSS_WINDOW" doesn't actually prove cross-regime robustness.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

# Bump when the window list changes in any way (add / remove / relabel /
# adjust dates). Persisted on cross_window_run.windows_catalog_version.
# v2 — bumped 2026-05-01 to fix end-date convention (now exclusive,
# first-of-next-period). Previous v1 windows ended on the last calendar
# day, which silently dropped that day from the JVM payload.
CATALOG_VERSION = "2026.05.01-v2"


@dataclass(frozen=True)
class RegimeWindow:
    """A regime-labeled epoch.

    End-date convention: ``end`` is **exclusive**. To cover all of 2024
    use ``start=2024-01-01, end=2025-01-01`` — matches the JVM payload's
    ``endTime=<end>T00:00:00`` semantics (trade with entry_time < end).
    """

    label: str       # Stable identifier — never reused even if a window is dropped.
    start: date
    end: date
    regime_tag: str  # BULL | BEAR | CHOP | TRANSITION


# Hand-curated. Keep this list ordered chronologically; the orchestrator
# preserves order when reporting per-window results.
WINDOWS: tuple[RegimeWindow, ...] = (
    RegimeWindow(
        label="2022-bear",
        start=date(2022, 1, 1),
        end=date(2023, 1, 1),
        regime_tag="BEAR",
    ),
    RegimeWindow(
        label="2023-chop",
        start=date(2023, 1, 1),
        end=date(2023, 10, 1),
        regime_tag="CHOP",
    ),
    RegimeWindow(
        label="2023-q4-recovery",
        start=date(2023, 10, 1),
        end=date(2024, 1, 1),
        regime_tag="TRANSITION",
    ),
    RegimeWindow(
        label="2024-bull",
        start=date(2024, 1, 1),
        end=date(2025, 1, 1),
        regime_tag="BULL",
    ),
    RegimeWindow(
        label="2025-bull-cont",
        start=date(2025, 1, 1),
        end=date(2026, 1, 1),
        regime_tag="BULL",
    ),
    RegimeWindow(
        label="2026-ytd",
        start=date(2026, 1, 1),
        end=date(2026, 5, 1),
        regime_tag="TRANSITION",
    ),
)


def windows_for(instrument: str) -> tuple[RegimeWindow, ...]:
    """Return the window catalog for an instrument.

    Currently all instruments share the BTC-derived regime calendar — the
    macro regime is set by BTC and ETH/etc move with it. If we later add
    an asset that runs on its own cycle, fork by instrument here.
    """
    _ = instrument
    return WINDOWS
