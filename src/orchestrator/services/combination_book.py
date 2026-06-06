"""Phase 3 combination book — IC helper + residual-fed admission decision.

The signal-level combination book (spec §8c, Component 7) admits a weak-but-
orthogonal *idiosyncratic* signal — one that fails the standalone V60 10%/yr
gate — by re-running the EXISTING pool admission math on the candidate's
**factor-neutral residual** returns (from ``factor_model.neutralize``) rather
than its raw equity series. This is additive to the frozen V11/V60; it never
loosens them.

This module is pure-ish: ``information_coefficient`` is a pure function; ``admit``
orchestrates the already-shipped ``pool.marginal_sharpe_contribution`` over
residual series and applies the four admission conditions. No DB, no I/O — the
caller fetches residuals/members and hands them in.

Admit iff ALL of:
  1. idiosyncratic alpha is significant — ``alpha_tstat >= ALPHA_TSTAT_MIN``;
  2. the information coefficient is significant AND its sign matches the
     pre-registered direction (``ic["significant"]`` and ``ic["sign"] ==
     expected_sign``);
  3. the candidate ADDS uncorrelated residual return — the marginal Sharpe
     uplift of inserting it into the residual book exceeds ``MARGINAL_SHARPE_MIN``;
  4. it is not redundant — its max |corr| with any member is below ``MAX_CORR``.
"""
from __future__ import annotations

import math
from datetime import date
from typing import Any
from uuid import UUID

import asyncpg
import numpy as np

from . import pool
from .factor_model import ALPHA_TSTAT_MIN  # reuse — do NOT redefine

# Discriminator that separates the residual combination book from the strategy
# pool inside the shared ``signal_pool`` table. The strategy pool's rows default
# to ``signal_pool`` (V147), so the two books coexist without a new table.
COMBINATION_KIND: str = "signal_combination"

# ── Pinned constants (operator-tunable; pin-tested) ──────────────────────────
# IC significance bar — same t-stat convention as the rest of the platform
# (t = ic·sqrt((n−2)/(1−ic²)) compared to this floor).
IC_TSTAT_MIN: float = 2.0
# Redundancy ceiling — a candidate whose max |corr| with any existing member is
# at/above this is too correlated to add diversifying value.
MAX_CORR: float = 0.80

# Minimum marginal-Sharpe uplift to admit. Reused from the strategy pool's
# admission threshold (``pool.DEFAULT_THETA`` — "Minimum marginal Sharpe uplift
# to admit") so the residual book shares one tunable, not a divergent copy.
#
# NOTE: the plan text names this symbol ``pool._MIN_MARGINAL_SHARPE``; that
# symbol does not exist in the shipped ``pool.py`` (the threshold is
# ``DEFAULT_THETA``). We bind to the real symbol to honour the intent
# ("import, do not redefine the threshold").
MARGINAL_SHARPE_MIN: float = pool.DEFAULT_THETA

# Minimum paired observations for a defensible Spearman IC.
_IC_MIN_OBS: int = 3


def information_coefficient(
    signal: list[float], fwd_returns: list[float]
) -> dict[str, Any]:
    """Spearman rank correlation between a signal and its forward returns.

    Returns ``{"ic", "significant", "sign"}`` where:
      * ``ic``          — Spearman ρ (rank-then-Pearson) over the paired values.
      * ``significant`` — True iff ``|t| >= IC_TSTAT_MIN`` for the standard
        ``t = ic·sqrt((n−2)/(1−ic²))``.
      * ``sign``        — "+", "-", or "0" (the latter for a ~zero / unjudgeable IC).

    Computed directly with NumPy (rank, then Pearson on the ranks) rather than
    via ``pool.spearman_corr_matrix`` — that helper takes date-keyed series and
    returns a pairwise matrix, which doesn't cleanly express a single paired
    (signal, fwd_returns) IC. Guards: n < ``_IC_MIN_OBS`` or a degenerate fit
    (constant ranks, or |ic| == 1 making the t-stat undefined) → not judgeable.
    """
    n = min(len(signal), len(fwd_returns))
    if n < _IC_MIN_OBS:
        return {"ic": 0.0, "significant": False, "sign": "0"}

    a = np.asarray(signal[:n], dtype=float)
    b = np.asarray(fwd_returns[:n], dtype=float)
    ra = _rank(a)
    rb = _rank(b)
    sd_a = ra.std()
    sd_b = rb.std()
    if sd_a == 0 or sd_b == 0:
        # A constant rank vector → correlation undefined.
        return {"ic": 0.0, "significant": False, "sign": "0"}
    cov = ((ra - ra.mean()) * (rb - rb.mean())).mean()
    ic = float(np.clip(cov / (sd_a * sd_b), -1.0, 1.0))

    sign = "+" if ic > 0 else ("-" if ic < 0 else "0")

    # |ic| == 1 makes (1 − ic²) == 0 → t undefined; a perfect monotonic relation
    # over n >= _IC_MIN_OBS is unambiguously significant.
    if abs(ic) >= 1.0:
        significant = True
    else:
        t = ic * math.sqrt((n - 2) / (1.0 - ic * ic))
        significant = abs(t) >= IC_TSTAT_MIN

    return {"ic": round(ic, 6), "significant": bool(significant), "sign": sign}


def _rank(values: np.ndarray) -> np.ndarray:
    """Average-rank assignment with tie handling (mirrors the pool's helper)."""
    n = values.size
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def admit(
    *,
    candidate_residuals: dict[date, float],
    members_residuals: dict[str, dict[date, float]],
    ic: dict[str, Any],
    alpha_tstat: float | None,
    expected_sign: str,
) -> dict[str, Any]:
    """Residual-fed admission decision for the combination book.

    Admit a candidate iff ALL four conditions hold (see module docstring). The
    marginal-Sharpe / redundancy math is the already-shipped
    ``pool.marginal_sharpe_contribution`` fed the candidate's RESIDUAL series and
    the existing members' residuals — so the book measures uncorrelated
    *idiosyncratic* return, not raw beta-laden return.

    Returns ``{"admitted", "reasons": {per-condition bool}, "marginal", "ic"}``.
    """
    # (1) idiosyncratic alpha significance.
    alpha_significant = alpha_tstat is not None and alpha_tstat >= ALPHA_TSTAT_MIN

    # (2) IC significant AND sign matches the pre-registered direction.
    ic_significant_and_signed = bool(ic.get("significant")) and (
        ic.get("sign") == expected_sign
    )

    # (3) + (4) marginal-Sharpe uplift + redundancy, on RESIDUAL series.
    marginal = pool.marginal_sharpe_contribution(candidate_residuals, members_residuals)
    marginal_positive = marginal["marginal"] > MARGINAL_SHARPE_MIN
    max_abs_corr = marginal["max_abs_corr"]
    low_corr = max_abs_corr is None or max_abs_corr < MAX_CORR

    reasons = {
        "alpha_significant": bool(alpha_significant),
        "ic_significant_and_signed": ic_significant_and_signed,
        "marginal_positive": bool(marginal_positive),
        "low_corr": bool(low_corr),
    }
    return {
        "admitted": all(reasons.values()),
        "reasons": reasons,
        "marginal": marginal,
        "ic": ic,
    }


# ── Persistence — coexist with the strategy pool (Task 5) ────────────────────
#
# Combination members live in the SAME ``signal_pool`` table the strategy pool
# uses (services/pool.py + api/pool.py), discriminated by
# ``kind='signal_combination'`` (+ a ``track`` tag). This mirrors the existing
# pool's dedicated-table mechanism rather than inventing a parallel scheme; the
# only addition is the kind/track discriminator so both books coexist without a
# separate table. The strategy pool's writes default ``kind='signal_pool'``,
# so its reads (when filtered to that kind) never see a combination member and
# vice-versa.


async def add_member(
    conn: asyncpg.Connection,
    *,
    iteration_id: UUID,
    strategy_code: str,
    symbol: str,
    interval_name: str,
    track: str | None,
    residual_metrics: dict[str, Any],
    created_by: str,
) -> UUID | None:
    """Persist an admitted combination member into ``signal_pool`` tagged
    ``kind='signal_combination'`` (+ ``track``).

    ``residual_metrics`` (alpha_tstat, ic, marginal, max_abs_corr, …) is the
    residual-fed admission snapshot — the residual analogue of the strategy
    pool's ``admission_metrics``. Atomic + idempotent per active surface via the
    partial-unique index on ``(kind, strategy_code, symbol, interval_name)
    WHERE status='active'``: a concurrent/duplicate admit ON CONFLICT DO NOTHING
    returns NULL (already a member). Returns the new ``pool_id`` on insert.
    """
    return await conn.fetchval(
        """
        INSERT INTO signal_pool
            (iteration_id, strategy_code, symbol, interval_name,
             kind, track, admission_metrics, status, created_by, updated_by)
        VALUES ($1, $2, $3, $4, $5, $6, $7, 'active', $8, $8)
        ON CONFLICT (kind, strategy_code, symbol, interval_name)
            WHERE status = 'active'
        DO NOTHING
        RETURNING pool_id
        """,
        iteration_id, strategy_code, symbol, interval_name,
        COMBINATION_KIND, track, residual_metrics, created_by,
    )


async def list_members(
    conn: asyncpg.Connection, *, track: str | None = None
) -> list[dict[str, Any]]:
    """List active combination-book members (``kind='signal_combination'``),
    optionally filtered to a single ``track``.

    The ``kind`` filter is what enforces coexistence isolation: this read can
    NEVER return a strategy-pool (``kind='signal_pool'``) row.
    """
    rows = await conn.fetch(
        """
        SELECT pool_id, iteration_id, strategy_code, symbol, interval_name,
               kind, track, admitted_at, admission_metrics, pool_weight,
               weight_source
          FROM signal_pool
         WHERE status = 'active'
           AND kind = $1
           AND ($2::text IS NULL OR track = $2)
         ORDER BY strategy_code, symbol, interval_name
        """,
        COMBINATION_KIND, track,
    )
    return [
        {
            "pool_id": str(r["pool_id"]),
            "iteration_id": str(r["iteration_id"]),
            "surface": f"{r['strategy_code']}:{r['symbol']}:{r['interval_name']}",
            "strategy_code": r["strategy_code"],
            "symbol": r["symbol"],
            "interval_name": r["interval_name"],
            "kind": r["kind"],
            "track": r["track"],
            "admitted_at": r["admitted_at"].isoformat() if r["admitted_at"] else None,
            "admission_metrics": r["admission_metrics"],
            "pool_weight": float(r["pool_weight"]) if r["pool_weight"] is not None else None,
            "weight_source": r["weight_source"],
        }
        for r in rows
    ]


def aggregate_book_weights(
    members_residuals: dict[str, dict[date, float]],
) -> dict[str, float]:
    """HRP weights for the aggregate combination book, computed on members'
    RESIDUAL return series via the shipped ``pool._hrp_for`` (so the book is
    diversified on idiosyncratic — not beta-laden — return). Keyed by member
    surface; sums to ~1. Empty input → empty dict."""
    return pool._hrp_for(members_residuals)
