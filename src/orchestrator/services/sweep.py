"""Sweep-combo derivation.

Ports the inline Python from ``research-tick.sh`` lines 280-294. Given a
sweep_config of the form ``{"params": [{"name": ..., "values": [...]}, ...]}``
and an iteration index, returns the cross-product entry at that index — or
``None`` if the iteration index has overrun the sweep.

Order is ``itertools.product`` left-to-right, which means the first
parameter cycles slowest. The bash script and the agent's queue-strategy
helper both assume this order; do not change it.
"""

from __future__ import annotations

from itertools import product
from typing import Any


def derive_combo(sweep_config: dict[str, Any], iter_index: int) -> dict[str, Any] | None:
    """Return the combo for ``iter_index`` (0-based), or None if exhausted."""
    params = sweep_config.get("params", [])
    if not params:
        return {} if iter_index == 0 else None
    keys = [p["name"] for p in params]
    values = [p["values"] for p in params]
    combos = list(product(*values))
    if iter_index < 0 or iter_index >= len(combos):
        return None
    return dict(zip(keys, combos[iter_index], strict=True))


def total_combos(sweep_config: dict[str, Any]) -> int:
    params = sweep_config.get("params", [])
    if not params:
        return 1
    n = 1
    for p in params:
        n *= len(p["values"])
    return n
