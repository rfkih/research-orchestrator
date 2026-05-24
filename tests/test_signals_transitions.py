"""Unit tests pinning the signal_definition status-transition matrix.

Pure tests — the API layer also re-validates server-side; this just
asserts the transition map itself stays put. Mirrors the model
promotion-transition pin tests in spirit.
"""
from __future__ import annotations

import pytest

from orchestrator.api.signals import _SIGNAL_STATUS_TRANSITIONS


def test_shadow_can_go_to_active_or_retired() -> None:
    assert _SIGNAL_STATUS_TRANSITIONS["shadow"] == frozenset({"active", "retired"})


def test_active_can_go_to_shadow_or_retired() -> None:
    """Active → shadow is the demotion path operators reach for when
    paired-delta turns negative. Active → retired is the kill switch."""
    assert _SIGNAL_STATUS_TRANSITIONS["active"] == frozenset({"shadow", "retired"})


def test_retired_is_terminal() -> None:
    """Retired signals are dead — rebirth is via a new signal_definition
    insert, not a status flip. Prevents accidental resurrection of a
    signal whose data plumbing has rotted while it was off."""
    assert _SIGNAL_STATUS_TRANSITIONS["retired"] == frozenset()


@pytest.mark.parametrize("status", ["shadow", "active", "retired"])
def test_self_transition_not_in_map(status: str) -> None:
    """No status lists itself as an allowed target — the API layer treats
    a same-state PUT as a no-op return, not a transition. Keeping it out
    of the map means the explicit ``409 signal_transition_forbidden``
    code never fires on a no-op."""
    assert status not in _SIGNAL_STATUS_TRANSITIONS[status]


def test_no_unknown_states_in_map() -> None:
    """The map's keys + reachable values must be a subset of the V66
    ``chk_signal_definition_status`` CHECK enum. If you add a new state
    here, you owe a Flyway migration on the trading-JVM side."""
    allowed = {"shadow", "active", "retired"}
    assert set(_SIGNAL_STATUS_TRANSITIONS.keys()) <= allowed
    reachable = set().union(*_SIGNAL_STATUS_TRANSITIONS.values())
    assert reachable <= allowed
