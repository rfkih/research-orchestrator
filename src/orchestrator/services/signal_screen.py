"""Feature-level IC screen -- the cheapest rung of the cost ladder.

    IC screen (minutes)  ->  null-screen (~1h)  ->  sweep (hours)  ->  walk-forward

``POST /signal-screen`` triages a candidate SIGNAL (a feature_store column
or a feature_values series) BEFORE anyone registers a hypothesis or builds
a JVM engine for it. Per (instrument, horizon) it computes:

  * Spearman rank IC between the signal at bar t and the forward log-return
    over t+1..t+h bars (close[t] -> close[t+h]);
  * a Newey-West t-stat (Bartlett kernel, lag = horizon) on the slope of
    the forward-return-on-transformed-signal regression -- forward returns
    at horizon h overlap h-1 neighbours, so iid OLS errors overstate
    significance by ~sqrt(h);
  * the top-vs-bottom quantile forward-return spread (quintiles; terciles
    below SMALL_N_FOR_TERCILES observations);
  * a seeded moving-block bootstrap 95% CI on the IC (block length ~
    horizon, BOOTSTRAP_DRAWS resamples -- same conventions as
    services/hedging_gate.improvement_significant).

Plus one POOLED row per horizon (all instruments' transformed pairs
concatenated; bootstrap blocks never cross an instrument boundary).

Verdict per row (pinned):
    INSUFFICIENT_DATA   n_obs < MIN_OBS
    PROMISING           |NW t| >= T_PROMISING AND bootstrap CI excludes 0
    DEAD                |NW t| <  T_DEAD      AND bootstrap CI includes 0
    WEAK                otherwise

ADVISORY ONLY. This screen does NOT touch the frozen V11/V60 gate logic in
tick.py / analyze.py, writes no hypothesis_audit row (it must not inflate
DSR multiplicity for the confirmatory sweep that may follow), and its
verdict binds nobody. It journals one row per call (structured_data.kind =
'signal_screen' -- the research_journal entry_type CHECK constraint has no
SIGNAL_SCREEN value, so we map DEAD -> ANTI_PATTERN, else
CROSS_STRATEGY_FINDING, mirroring services/null_screen.py) including a
running multiplicity note: N prior screens on the same signal family.

Point-in-time alignment (non-negotiable): for feature_values series the
signal at a bar is the LATEST value with ts <= bar start. A value stamped
after the bar start is invisible at that bar -- no look-ahead. For
feature_store columns the join is exact on (symbol, interval, start_time):
the per-bar store is computed from that bar and known at its close, which
is exactly when the forward-return clock starts.
"""
from __future__ import annotations

import math
import random
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import asyncpg
import numpy as np

from ..errors import NextAction, OrchestratorError
from ..logging import get_logger
from ..repo import feature_store as feature_store_repo
from ..repo import features as features_repo
from ..repo import journal as journal_repo
from ..repo import market_data as market_data_repo
from .combination_book import information_coefficient

log = get_logger(__name__)

# -- Pinned constants (operator-tunable; pin-tested in tests/test_signal_screen.py)
T_PROMISING: float = 2.0        # |NW t| floor for PROMISING (with CI excluding 0)
T_DEAD: float = 1.0             # |NW t| ceiling for DEAD (with CI including 0)
BOOTSTRAP_DRAWS: int = 1000     # moving-block bootstrap resamples (>= 1000 per spec)
CI_LEVEL: float = 0.95          # bootstrap CI level
RNG_SEED: int = 42              # same seed convention as hedging_gate
MIN_OBS: int = 30               # min paired obs per (instrument, horizon) row
QUANTILES_DEFAULT: int = 5      # quintile spread ...
QUANTILES_SMALL: int = 3        # ... terciles when n < SMALL_N_FOR_TERCILES
SMALL_N_FOR_TERCILES: int = 200
# Cap for a degenerate-but-real t (zero residual variance, e.g. signal ==
# future return). Mirrors hedging_gate._CONST_SHARPE_CAP's convention.
T_CAP: float = 1e6
# How far before the first bar we fetch feature_values so the PIT join has
# a value for the earliest bars (stale-but-PIT-safe carry-forward).
FEATURE_LOOKBACK_DAYS: int = 90

_DEFAULT_START = datetime(2017, 1, 1)

VALID_TRANSFORMS: frozenset[str] = frozenset({"zscore", "rank", "raw"})


# ====================================================================
# Pure math helpers (unit-tested in isolation)
# ====================================================================


def signal_family(signal: str) -> str:
    """Canonical family key for multiplicity counting.

    Lowercase, digit-runs stripped, underscores collapsed:
    ``btc_dvol_zscore_30d`` -> ``btc_dvol_zscore_d``; ``ema_100`` and
    ``ema_150`` -> ``ema``. Imperfect by design -- what matters is that
    the SAME canonicalization counts prior screens, so re-screens of
    window-variants accumulate into one family's multiplicity note.
    """
    fam = re.sub(r"\d+", "", signal.lower())
    fam = re.sub(r"_+", "_", fam).strip("_")
    return fam or signal.lower()


def transform_series(values: list[float], kind: str) -> list[float]:
    """Apply the requested transform to a signal vector.

    * ``zscore`` -- (x - mean) / std (population std). Constant series
      (std == 0) maps to all-zeros rather than NaN.
    * ``rank``   -- average ranks rescaled to [0, 1] (n == 1 -> [0.5]).
    * ``raw``    -- identity copy.

    Spearman IC is invariant to any of these (all monotonic); the
    transform matters for the Newey-West regression slope and above all
    for POOLING -- per-instrument zscore/rank puts instruments on a
    common scale before their pairs are concatenated.
    """
    if kind not in VALID_TRANSFORMS:
        raise ValueError(f"unknown transform {kind!r}")
    if kind == "raw":
        return list(values)
    a = np.asarray(values, dtype=float)
    if kind == "zscore":
        sd = float(a.std())
        if sd == 0.0 or not math.isfinite(sd):
            return [0.0] * len(values)
        return [float(v) for v in (a - a.mean()) / sd]
    # rank
    n = a.size
    if n == 0:
        return []
    if n == 1:
        return [0.5]
    ranks = _rank_vec(a)
    return [float(r) for r in (ranks - 1.0) / (n - 1.0)]


def forward_log_returns(closes: list[float], horizon: int) -> list[float | None]:
    """``out[t] = ln(close[t+horizon] / close[t])`` -- the forward log
    return over bars t+1..t+h. The last ``horizon`` slots are None (no
    future bar yet), as is any slot with a non-positive close."""
    n = len(closes)
    out: list[float | None] = [None] * n
    for t in range(n - horizon):
        c0, c1 = closes[t], closes[t + horizon]
        if c0 is not None and c1 is not None and c0 > 0 and c1 > 0:
            out[t] = math.log(c1 / c0)
    return out


def align_point_in_time(
    bar_starts: list[datetime], points: list[tuple[datetime, float]]
) -> list[float | None]:
    """For each bar start, the LATEST feature value with ts <= bar start.

    THE no-look-ahead invariant of the screen: a value stamped strictly
    after a bar start is invisible at that bar (regression-tested).
    Bars before the first point get None. Both inputs must be ascending;
    a single merge pass keeps this O(n + m).
    """
    out: list[float | None] = []
    i = 0
    last: float | None = None
    m = len(points)
    for bar_ts in bar_starts:
        while i < m and points[i][0] <= bar_ts:
            last = points[i][1]
            i += 1
        out.append(last)
    return out


def _rank_vec(a: np.ndarray) -> np.ndarray:
    """Vectorized average ranks (ties share the mean rank), 1-based.

    Same semantics as ``combination_book._rank`` -- parity is pinned by a
    test -- but O(n log n) numpy instead of a Python while-loop, because
    the bootstrap calls this BOOTSTRAP_DRAWS times per row.
    """
    n = a.size
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    return (sums / counts)[inv]


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    """Spearman rho via Pearson on average ranks. None when undefined."""
    if x.size < 3:
        return None
    rx = _rank_vec(x)
    ry = _rank_vec(y)
    sx, sy = rx.std(), ry.std()
    if sx == 0 or sy == 0:
        return None
    cov = ((rx - rx.mean()) * (ry - ry.mean())).mean()
    return float(np.clip(cov / (sx * sy), -1.0, 1.0))


def newey_west_tstat(x: list[float], y: list[float], lag: int) -> float | None:
    """t-stat of the slope in ``y = a + b*x`` with Newey-West (Bartlett
    kernel) HAC standard errors, truncation ``lag``.

    The IC-bearing regression: x is the transformed signal, y the forward
    log return. Forward returns at horizon h are built from overlapping
    windows -- adjacent errors share h-1 bars -- so the caller passes
    ``lag = horizon`` and the kernel absorbs that serial correlation
    instead of letting iid OLS overstate t by ~sqrt(h).

    Returns None when undefined (n < 8, zero signal variance); a zero
    residual-variance fit with a non-zero slope (signal == future return)
    returns +/-T_CAP rather than dividing by zero.
    """
    n = min(len(x), len(y))
    if n < 8:
        return None
    xa = np.asarray(x[:n], dtype=float)
    ya = np.asarray(y[:n], dtype=float)
    xa = xa - xa.mean()
    ya = ya - ya.mean()
    sxx = float((xa * xa).sum())
    if sxx <= 0.0:
        return None
    b = float((xa * ya).sum()) / sxx
    resid = ya - b * xa
    score = xa * resid  # the slope estimating-equation scores
    big_s = float((score * score).sum())
    lag_eff = max(0, min(int(lag), n - 1))
    for ell in range(1, lag_eff + 1):
        w = 1.0 - ell / (lag_eff + 1.0)
        big_s += 2.0 * w * float((score[:-ell] * score[ell:]).sum())
    big_s = max(big_s, 0.0)
    var_b = big_s / (sxx * sxx)
    if var_b <= 0.0:
        if b == 0.0:
            return 0.0
        return T_CAP if b > 0 else -T_CAP
    t = b / math.sqrt(var_b)
    return float(np.clip(t, -T_CAP, T_CAP))


def quantile_spread(signal: list[float], fwd: list[float]) -> dict[str, Any]:
    """Top-vs-bottom quantile forward-return spread.

    Buckets bars by signal rank into Q equal-count quantiles (quintiles by
    default; terciles when n < SMALL_N_FOR_TERCILES). Spread = mean
    forward return of the top bucket minus the bottom bucket. Returns
    Nones when there are too few obs to fill 2 buckets.
    """
    n = min(len(signal), len(fwd))
    if n == 0:
        return {"q": None, "top_mean": None, "bottom_mean": None, "spread": None}
    q = QUANTILES_DEFAULT if n >= SMALL_N_FOR_TERCILES else QUANTILES_SMALL
    if n < 2 * q:
        return {"q": q, "top_mean": None, "bottom_mean": None, "spread": None}
    sig = np.asarray(signal[:n], dtype=float)
    ret = np.asarray(fwd[:n], dtype=float)
    order = np.argsort(sig, kind="mergesort")
    bucket_of = np.empty(n, dtype=int)
    bucket_of[order] = (np.arange(n) * q) // n
    top = ret[bucket_of == q - 1]
    bottom = ret[bucket_of == 0]
    top_mean = float(top.mean())
    bottom_mean = float(bottom.mean())
    return {
        "q": q,
        "top_mean": round(top_mean, 8),
        "bottom_mean": round(bottom_mean, 8),
        "spread": round(top_mean - bottom_mean, 8),
        "n_top": int(top.size),
        "n_bottom": int(bottom.size),
    }


def block_bootstrap_ic_ci(
    pairs_by_instrument: list[tuple[list[float], list[float]]],
    *,
    horizon: int,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = RNG_SEED,
) -> dict[str, Any]:
    """Seeded moving-block bootstrap CI_LEVEL CI on the Spearman IC.

    Each draw resamples contiguous blocks (with wraparound) of the paired
    (signal, fwd_return) series -- per instrument, so blocks never cross
    an instrument boundary -- concatenates them, and computes the Spearman
    IC of the resample. Block length ~ horizon (the overlapping-window
    dependence scale), floored at 1 and capped at each instrument's n.
    Deterministic: ``random.Random(seed)``, never global random --
    mirrors services/hedging_gate.improvement_significant.
    """
    clean: list[tuple[np.ndarray, np.ndarray]] = []
    for sig, fwd in pairs_by_instrument:
        n = min(len(sig), len(fwd))
        if n > 0:
            clean.append(
                (np.asarray(sig[:n], dtype=float), np.asarray(fwd[:n], dtype=float))
            )
    total_n = sum(s.size for s, _ in clean)
    if total_n < MIN_OBS:
        return {
            "ci_low": None, "ci_high": None, "excludes_zero": False,
            "draws": 0, "block_len": None, "seed": seed,
            "reason": "insufficient_obs",
        }
    rng = random.Random(seed)
    block = max(1, int(horizon))
    ics: list[float] = []
    for _ in range(draws):
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for sig, fwd in clean:
            n = sig.size
            blk = min(block, n)
            idx = np.empty(n, dtype=int)
            filled = 0
            while filled < n:
                start = rng.randrange(n)
                take = min(blk, n - filled)
                idx[filled:filled + take] = (start + np.arange(take)) % n
                filled += take
            xs.append(sig[idx])
            ys.append(fwd[idx])
        ic = _spearman(np.concatenate(xs), np.concatenate(ys))
        if ic is not None:
            ics.append(ic)
    if not ics:
        return {
            "ci_low": None, "ci_high": None, "excludes_zero": False,
            "draws": 0, "block_len": block, "seed": seed,
            "reason": "no_finite_resamples",
        }
    ics.sort()
    alpha = (1.0 - CI_LEVEL) / 2.0
    lo_idx = int(alpha * len(ics))
    hi_idx = max(int((1.0 - alpha) * len(ics)) - 1, 0)
    ci_low = ics[lo_idx]
    ci_high = ics[hi_idx]
    return {
        "ci_low": round(ci_low, 6),
        "ci_high": round(ci_high, 6),
        "excludes_zero": bool(ci_low > 0.0 or ci_high < 0.0),
        "draws": len(ics),
        "block_len": block,
        "seed": seed,
    }


def row_verdict(
    *, n_obs: int, nw_tstat: float | None, ci_excludes_zero: bool
) -> str:
    """Pinned advisory verdict for one (instrument, horizon) row."""
    if n_obs < MIN_OBS:
        return "INSUFFICIENT_DATA"
    if nw_tstat is not None and abs(nw_tstat) >= T_PROMISING and ci_excludes_zero:
        return "PROMISING"
    if (nw_tstat is None or abs(nw_tstat) < T_DEAD) and not ci_excludes_zero:
        return "DEAD"
    return "WEAK"


def overall_verdict(row_verdicts: list[str]) -> str:
    """Aggregate: any PROMISING wins; all-INSUFFICIENT is INSUFFICIENT_DATA;
    DEAD requires every judgeable row DEAD; else WEAK."""
    if not row_verdicts:
        return "INSUFFICIENT_DATA"
    if "PROMISING" in row_verdicts:
        return "PROMISING"
    judgeable = [v for v in row_verdicts if v != "INSUFFICIENT_DATA"]
    if not judgeable:
        return "INSUFFICIENT_DATA"
    if all(v == "DEAD" for v in judgeable):
        return "DEAD"
    return "WEAK"


def journal_entry_type_for(verdict: str) -> str:
    """Map the overall screen verdict to a research_journal.entry_type
    allowed by the CHECK constraint (V1 baseline + V127). DEAD earns the
    ANTI_PATTERN slot (the cold-boot recipe surfaces those); everything
    else is a CROSS_STRATEGY_FINDING -- a feature-level result is not tied
    to one strategy. Same pattern as null_screen.journal_entry_type_for."""
    if verdict == "DEAD":
        return "ANTI_PATTERN"
    return "CROSS_STRATEGY_FINDING"


def compute_screen_rows(
    *,
    bars_by_instrument: dict[str, list[tuple[datetime, float]]],
    signal_by_instrument: dict[str, list[float | None]],
    horizons_bars: list[int],
    transform: str,
) -> list[dict[str, Any]]:
    """Pure computation core: per (instrument, horizon) stats + POOLED rows.

    ``signal_by_instrument[i][k]`` is the PIT-aligned signal value at bar k
    of instrument i (None = no value known at that bar). Bars where either
    the signal or the forward return is missing are dropped pairwise. The
    transform is applied PER INSTRUMENT over its observed values, so pooled
    rows compare like with like. Returns rows ranked by |IC| x |NW t| desc.
    """
    rows: list[dict[str, Any]] = []
    for horizon in horizons_bars:
        pooled_pairs: list[tuple[list[float], list[float]]] = []
        pooled_count = 0
        for instrument, bars in bars_by_instrument.items():
            closes = [px for _, px in bars]
            fwd = forward_log_returns(closes, horizon)
            sig_raw = signal_by_instrument.get(instrument) or []
            paired_sig: list[float] = []
            paired_fwd: list[float] = []
            paired_ts: list[datetime] = []
            for k in range(min(len(bars), len(sig_raw))):
                s, f = sig_raw[k], fwd[k]
                if s is None or f is None:
                    continue
                if not (math.isfinite(s) and math.isfinite(f)):
                    continue
                paired_sig.append(float(s))
                paired_fwd.append(float(f))
                paired_ts.append(bars[k][0])
            sig_t = transform_series(paired_sig, transform)
            rows.append(
                _stat_row(
                    instrument=instrument,
                    horizon=horizon,
                    sig=sig_t,
                    fwd=paired_fwd,
                    coverage=(paired_ts[0], paired_ts[-1]) if paired_ts else None,
                    pairs_by_instrument=[(sig_t, paired_fwd)],
                )
            )
            if sig_t:
                pooled_pairs.append((sig_t, paired_fwd))
                pooled_count += 1
        if pooled_count > 1:
            pooled_sig = [v for sig, _ in pooled_pairs for v in sig]
            pooled_fwd = [v for _, fwd in pooled_pairs for v in fwd]
            rows.append(
                _stat_row(
                    instrument="POOLED",
                    horizon=horizon,
                    sig=pooled_sig,
                    fwd=pooled_fwd,
                    coverage=None,
                    pairs_by_instrument=pooled_pairs,
                )
            )
    rows.sort(key=lambda r: r["rank_score"], reverse=True)
    return rows


def _stat_row(
    *,
    instrument: str,
    horizon: int,
    sig: list[float],
    fwd: list[float],
    coverage: tuple[datetime, datetime] | None,
    pairs_by_instrument: list[tuple[list[float], list[float]]],
) -> dict[str, Any]:
    n_obs = min(len(sig), len(fwd))
    ic_info = information_coefficient(sig, fwd)  # reuse, do NOT re-derive
    nw_t = newey_west_tstat(sig, fwd, lag=horizon) if n_obs >= MIN_OBS else None
    boot = block_bootstrap_ic_ci(pairs_by_instrument, horizon=horizon)
    spread = quantile_spread(sig, fwd)
    verdict = row_verdict(
        n_obs=n_obs, nw_tstat=nw_t, ci_excludes_zero=boot["excludes_zero"]
    )
    ic = float(ic_info["ic"])
    rank_score = abs(ic) * abs(nw_t) if nw_t is not None else 0.0
    return {
        "instrument": instrument,
        "horizon_bars": horizon,
        "n_obs": n_obs,
        "coverage_start": coverage[0].isoformat() if coverage else None,
        "coverage_end": coverage[1].isoformat() if coverage else None,
        "ic": ic,
        "ic_sign": ic_info["sign"],
        "nw_tstat": round(nw_t, 4) if nw_t is not None else None,
        "nw_lag": horizon,
        "quantile_spread": spread,
        "bootstrap_ci": boot,
        "verdict": verdict,
        "rank_score": round(rank_score, 6),
    }


# ====================================================================
# Orchestration (DB-touching)
# ====================================================================


async def run_signal_screen(
    conn: asyncpg.Connection,
    *,
    agent_name: str,
    signal: str,
    instruments: list[str],
    interval: str,
    horizons_bars: list[int],
    transform: str,
    start: datetime | None,
    end: datetime | None,
) -> dict[str, Any]:
    """Resolve the signal, fetch bars + series, compute the ranked IC
    table, journal one signal_screen row, and return the payload."""
    start = start or _DEFAULT_START
    end = end or datetime.now(timezone.utc).replace(tzinfo=None)
    # market_data/feature tables use naive TIMESTAMP columns -- normalize
    # tz-aware request datetimes to naive UTC so asyncpg accepts them.
    if start.tzinfo is not None:
        start = start.astimezone(timezone.utc).replace(tzinfo=None)
    if end.tzinfo is not None:
        end = end.astimezone(timezone.utc).replace(tzinfo=None)
    if start >= end:
        raise OrchestratorError(
            status_code=400,
            error_code="bad_window",
            message="start must be strictly before end.",
            retryable=False,
        )

    # 1. Resolve the signal: feature_store column first, else feature_values.
    fs_columns = await feature_store_repo.list_numeric_columns(conn)
    resolved_source = "feature_store" if signal in fs_columns else None

    # 2. Per-instrument bar series + PIT-aligned signal series.
    bars_by_instrument: dict[str, list[tuple[datetime, float]]] = {}
    signal_by_instrument: dict[str, list[float | None]] = {}
    resolution_meta: dict[str, Any] = {}
    for instrument in instruments:
        bars = await market_data_repo.fetch_closes(
            conn, symbol=instrument, interval=interval, start=start, end=end
        )
        bars_by_instrument[instrument] = bars
        if not bars:
            signal_by_instrument[instrument] = []
            continue
        if resolved_source == "feature_store":
            points = await feature_store_repo.fetch_column_series(
                conn, column=signal, symbol=instrument, interval=interval,
                start=start, end=end,
            )
            by_ts = dict(points)
            signal_by_instrument[instrument] = [by_ts.get(ts) for ts, _ in bars]
        else:
            points, meta = await features_repo.fetch_feature_series(
                conn,
                name=signal,
                symbol=instrument,
                interval=interval,
                start=start - timedelta(days=FEATURE_LOOKBACK_DAYS),
                end=end,
            )
            if meta.get("resolved"):
                resolved_source = resolved_source or "feature_values"
                resolution_meta[instrument] = meta
                # NO LOOK-AHEAD: latest value with ts <= bar start.
                signal_by_instrument[instrument] = align_point_in_time(
                    [ts for ts, _ in bars], points
                )
            else:
                signal_by_instrument[instrument] = []

    if resolved_source is None:
        raise OrchestratorError(
            status_code=404,
            error_code="signal_not_found",
            message=(
                f"signal {signal!r} is neither a feature_store column nor a "
                "feature_values feature_name with rows."
            ),
            retryable=False,
            next_action=NextAction(kind="call", method="GET", path="/features"),
            details={"hint": "GET /features lists registered feature_values names."},
        )

    # 3. Ranked stats table (pure).
    rows = compute_screen_rows(
        bars_by_instrument=bars_by_instrument,
        signal_by_instrument=signal_by_instrument,
        horizons_bars=horizons_bars,
        transform=transform,
    )
    overall = overall_verdict([r["verdict"] for r in rows])

    # 4. Multiplicity note -- count PRIOR screens on the same family.
    family = signal_family(signal)
    prior = await journal_repo.count_signal_screens(conn, signal_family=family)
    multiplicity = {
        "signal_family": family,
        "prior_screens_same_family": prior,
        "note": (
            f"{prior} prior SIGNAL_SCREEN row(s) for family {family!r} -- this is "
            f"screen #{prior + 1}. Borderline t-stats accumulate selection bias "
            "across repeated screens of one family; weigh accordingly."
        ),
    }

    # 5. Journal one row (DML only; entry_type mapped into the CHECK set).
    best = rows[0] if rows else None
    structured = {
        "kind": "signal_screen",
        "signal": signal,
        "signal_family": family,
        "resolved_source": resolved_source,
        "resolution_meta": resolution_meta or None,
        "transform": transform,
        "interval": interval,
        "instruments": instruments,
        "horizons_bars": horizons_bars,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "overall_verdict": overall,
        "rows": rows,
        "multiplicity": multiplicity,
        "advisory": "ADVISORY ONLY -- does not alter V11/V60 gate logic.",
    }
    title = f"SIGNAL_SCREEN {signal} {interval} -> {overall}"
    if best is not None:
        best_line = (
            f"Best row: {best['instrument']} h={best['horizon_bars']} "
            f"IC={best['ic']} NW_t={best['nw_tstat']} ({best['verdict']})."
        )
    else:
        best_line = "No computable rows."
    content = (
        f"IC screen on signal {signal!r} ({resolved_source}) over "
        f"{', '.join(instruments)} @ {interval}, horizons={horizons_bars}, "
        f"transform={transform}. Overall: {overall}. {best_line} "
        f"{multiplicity['note']} Advisory only -- V11/V60 untouched."
    )
    journal_row = await journal_repo.insert_journal(
        conn,
        entry_type=journal_entry_type_for(overall),
        title=title[:300],
        content=content,
        strategy_code=None,
        interval_name=interval,
        instrument=instruments[0] if len(instruments) == 1 else None,
        structured_data=structured,
        status="ACTIVE",
        iteration_id_refs=None,
        related_journal_ids=None,
        created_by=agent_name,
    )

    log.info(
        "signal_screen.done",
        signal=signal,
        interval=interval,
        overall=overall,
        n_rows=len(rows),
        prior_screens=prior,
    )

    return {
        "signal": signal,
        "signal_family": family,
        "resolved_source": resolved_source,
        "transform": transform,
        "interval": interval,
        "instruments": instruments,
        "horizons_bars": horizons_bars,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "overall_verdict": overall,
        "rows": rows,
        "multiplicity": multiplicity,
        "journal_id": str(journal_row["journal_id"]),
        "advisory": "ADVISORY ONLY -- does not alter V11/V60 gate logic.",
        "next_actions": _next_actions(overall, signal),
    }


def _next_actions(verdict: str, signal: str) -> list[dict[str, Any]]:
    """Verdict -> recommended next call. Pure for testability."""
    if verdict == "PROMISING":
        return [{
            "kind": "call", "method": "POST", "path": "/journal",
            "hint": (
                f"IC screen PROMISING on {signal!r}: pre-register a HYPOTHESIS "
                "journal row, then POST /null-screen before any sweep."
            ),
        }]
    if verdict == "DEAD":
        return [{
            "kind": "read_doc", "doc_anchor": "signal_screen.dead",
            "hint": (
                f"{signal!r} is DEAD on all screened horizons/instruments: the "
                "screen row is journaled -- pivot WITHOUT registering a "
                "hypothesis or building an engine."
            ),
        }]
    if verdict == "WEAK":
        return [{
            "kind": "read_doc", "doc_anchor": "signal_screen.weak",
            "hint": (
                "Weak/ambiguous IC. Consider a different interval/horizon or "
                "more history before spending a hypothesis on it."
            ),
        }]
    return [{
        "kind": "contact_human",
        "hint": "Insufficient observations -- check coverage/backfill first.",
    }]
