"""Shared validation constants for API request bodies.

Each value here is referenced by more than one router. Centralising
removes the drift risk of two routers carrying slightly different copies
of the same domain enum (e.g. one with ``"30m"`` added, one without)
and keeps the canonical spelling co-located with a single docstring.

Service-layer code does NOT validate against these — the API boundary
is the contract enforcement point; services trust their inputs.
"""

from __future__ import annotations

# Bar intervals the research JVM ticks at. 5m is the finest granularity
# the backtest engine emits; sub-5m would silently miss bars. The same
# four-value set gates /queue, /null-screen, /walk-forward, /cross-window.
VALID_INTERVAL_NAMES: frozenset[str] = frozenset({"5m", "15m", "1h", "4h"})

# Review verdicts that pass the paired-agent gate (i.e. unlock /queue or
# /walk-forward). Complement of {"REJECTED"} in the full verdict enum.
# CONDITIONAL_APPROVAL passes because the researcher has acknowledged the
# warnings — the reviewer raised non-blocking concerns, not a veto.
REVIEW_VERDICTS_PASSING_GATE: frozenset[str] = frozenset(
    {"APPROVED", "CONDITIONAL_APPROVAL"}
)
