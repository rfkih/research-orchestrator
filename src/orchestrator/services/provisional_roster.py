"""Provisional trading-roster floor (2026-06-09).

Problem: the graduation gate is (correctly) strict, so the live book can sit
below a usable number of tradeable strategies — users have nothing to trade
while research grinds. This module keeps a MINIMUM live roster by selecting the
best *near-miss* trading strategies to run PROVISIONALLY until real qualifiers
graduate.

NOT a loosening of the gate. The V11/V60 graduation contract is untouched —
fully-qualifying strategies still graduate the normal way and are NOT
provisional. This is an ADDITIVE product floor that only fills EMPTY roster
slots, and only with candidates that have a *statistically real* edge
(SIGNIFICANT_EDGE) and make money (annualised return > 0). We never put noise or
capital-destroyers live, even provisionally.

Decisions (operator, 2026-06-09):
  * Floor = 5 active TRADING strategies (hedging excluded).
  * Provisional strategies are MARKED ``provisional=True`` but trade FULL size.
  * Selection is FULLY AUTOMATIC (the JVM auto-activates the top picks when the
    active count drops below the floor; see companion JVM change).
  * Ranking = BALANCED COMPOSITE: annualised return × overfit-quality, where
    overfit-quality aggregates DSR / walk-forward fold-positive% / param
    robustness (whatever evidence the candidate has).

Pure functions only — no I/O. The endpoint composes these with a repo query for
the candidate pool and the JVM-supplied count of currently-active trading
strategies.
"""

from __future__ import annotations

from typing import Any

# Operator-controlled, pin-tested. Change only with a documented journal entry.
MIN_ACTIVE_TRADING_STRATEGIES = 5
# Eligibility: a provisional pick must have a statistically real edge. We do NOT
# promote INSUFFICIENT_EVIDENCE / NO_EDGE — those are noise, not near-misses.
ELIGIBLE_STAT_VERDICT = "SIGNIFICANT_EDGE"
# A walk-forward that already judged the candidate OVERFIT disqualifies it.
_DISQUALIFYING_WF_VERDICTS = frozenset({"OVERFIT", "NO_EDGE"})
# Param-robustness sub-score when the candidate sits on a cliff (vs a plateau).
_PARAM_CLIFF_PENALTY = 0.3


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def overfit_quality(
    dsr: float | None,
    fold_positive_pct: float | None = None,
    param_robust: bool | None = None,
) -> float:
    """A [0,1] "least-overfit" quality score — higher = more trustworthy.

    Geometric mean of whatever evidence exists, so a candidate is rewarded for
    having MORE corroborating validation, not just a high single number:

      * ``dsr`` — deflated Sharpe P(SR>0 after selection-bias), already in [0,1].
        The primary signal; present for every SIGNIFICANT_EDGE candidate.
      * ``fold_positive_pct`` — walk-forward positive-fold % (0-100), if the
        candidate has been walk-forwarded. Normalised to [0,1].
      * ``param_robust`` — whether the optimum sits on a plateau (True) or a
        cliff (False), from the robustness check. Cliff → heavy penalty.

    Geometric mean (not arithmetic) so a single near-zero sub-score drags the
    whole quality down — one strong overfit signal shouldn't be averaged away.
    Returns 0.0 when no evidence is available (cannot certify → no quality).
    """
    subs: list[float] = []
    if dsr is not None:
        subs.append(_clamp01(dsr))
    if fold_positive_pct is not None:
        subs.append(_clamp01(fold_positive_pct / 100.0))
    if param_robust is not None:
        subs.append(1.0 if param_robust else _PARAM_CLIFF_PENALTY)
    if not subs:
        return 0.0
    prod = 1.0
    for s in subs:
        prod *= max(s, 1e-9)
    return prod ** (1.0 / len(subs))


def composite_score(ann_return_pct: float | None, quality: float) -> float:
    """Balanced composite = annualised return × overfit-quality.

    A money-loser (return ≤ 0) or a zero-quality candidate scores 0 → never
    selected. Otherwise return and quality multiply, so the pick must be BOTH a
    strong earner AND trustworthy — neither alone wins a slot.
    """
    if ann_return_pct is None or ann_return_pct <= 0.0:
        return 0.0
    return ann_return_pct * quality


def is_eligible(candidate: dict[str, Any]) -> bool:
    """A candidate may fill a provisional slot iff:

      * it is a TRADING-track strategy (hedging excluded — different gate),
      * its statistical_verdict is SIGNIFICANT_EDGE (a real edge, not noise),
      * its annualised return is strictly positive (don't run money-losers),
      * no walk-forward has judged it OVERFIT / NO_EDGE.

    The whole point is to surface NEAR-MISSES — strategies with a proven edge
    that merely missed the 10% economic bar or haven't been walk-forwarded —
    not to relax what counts as an edge.
    """
    track = candidate.get("track")
    if track is not None and track != "trading":
        return False
    if candidate.get("statistical_verdict") != ELIGIBLE_STAT_VERDICT:
        return False
    ann = candidate.get("ann_return_pct")
    if ann is None or float(ann) <= 0.0:
        return False
    if candidate.get("walk_forward_verdict") in _DISQUALIFYING_WF_VERDICTS:
        return False
    return True


def rank_provisional_candidates(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return eligible candidates annotated with ``provisional_score`` /
    ``overfit_quality``, sorted best-first. Ineligible candidates are dropped.

    Deterministic tie-break: higher score, then higher DSR, then strategy_code
    (so the ordering is stable across calls for the same input).
    """
    ranked: list[dict[str, Any]] = []
    for c in candidates:
        if not is_eligible(c):
            continue
        dsr = c.get("dsr")
        dsr_f = float(dsr) if dsr is not None else None
        quality = overfit_quality(
            dsr_f,
            fold_positive_pct=c.get("fold_positive_pct"),
            param_robust=c.get("param_robust"),
        )
        score = composite_score(
            float(c["ann_return_pct"]) if c.get("ann_return_pct") is not None else None,
            quality,
        )
        ranked.append({**c, "overfit_quality": round(quality, 6),
                       "provisional_score": round(score, 6),
                       # cache a float dsr for a type-consistent sort key (raw
                       # dsr may be Decimal from asyncpg).
                       "_dsr_sort": dsr_f if dsr_f is not None else -1.0})
    ranked.sort(
        key=lambda r: (
            r["provisional_score"],
            r["_dsr_sort"],
            r.get("strategy_code") or "",
        ),
        reverse=True,
    )
    for r in ranked:
        r.pop("_dsr_sort", None)
    return ranked


def roster_gap(active_count: int, floor: int = MIN_ACTIVE_TRADING_STRATEGIES) -> int:
    """How many provisional slots to fill = max(0, floor − active_count)."""
    return max(0, floor - int(active_count))


def select_roster_fill(
    active_count: int,
    candidates: list[dict[str, Any]],
    *,
    floor: int = MIN_ACTIVE_TRADING_STRATEGIES,
    exclude_strategy_codes: frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    """Pick the top-N eligible near-misses to fill the roster to ``floor``.

    ``exclude_strategy_codes`` removes strategies already live (so the floor
    counts them and we don't double-promote). De-dupes by strategy_code — only
    the best-scoring iteration per strategy fills a slot, so 5 cells of the same
    strategy can't monopolise the roster.
    """
    gap = roster_gap(active_count, floor)
    if gap <= 0:
        return []
    excl = exclude_strategy_codes or frozenset()
    picks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for c in rank_provisional_candidates(candidates):
        code = c.get("strategy_code") or ""
        if code in excl or code in seen:
            continue
        seen.add(code)
        picks.append({**c, "provisional": True})
        if len(picks) >= gap:
            break
    return picks
