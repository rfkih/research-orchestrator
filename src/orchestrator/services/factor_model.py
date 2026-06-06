"""Phase 3 factor model — four daily factor-return series + a thin assembler.

The factor math lives in PURE functions over already-fetched price/macro series
so it is unit-testable without a DB. A thin async ``build_factors`` fetches the
underlying data (``repo.market_data`` + ``repo.macro_raw``) and assembles the
four series, degrading gracefully (omit + log) when a source is absent/thin.

Four factors (spec §8b, Component 6):
  - MARKET   : daily simple returns of BTC (the market proxy).
  - MOMENTUM : cross-sectional, point-in-time long-top / short-bottom of the
               universe ranked on trailing return.
  - CARRY    : cross-sectional long-high-funding / short-low-funding.
  - VOL      : daily change in Deribit BTC DVOL (implied-vol factor).

No look-ahead: MOMENTUM/CARRY rank on information available at t-1 (trailing
returns / funding known by t-1); the factor RETURN realised at t comes from t's
price move only.
"""
from __future__ import annotations

import logging
import math
from datetime import date, datetime, timezone

import asyncpg
import numpy as np

from ..repo import macro_raw, market_data
from . import hedging_gate

logger = logging.getLogger(__name__)

# ── Pinned constants (operator-tunable; pin-tested) ─────────────────────────
MOM_LOOKBACK: int = 20          # trailing days used to rank momentum
MOM_TOP_K: int = 1              # legs long the top-k / short the bottom-k

# OLS neutralization (Task 3).
ALPHA_TSTAT_MIN: float = 2.0    # |idiosyncratic alpha t-stat| floor for "real"
MIN_FACTOR_OBS: int = 60        # never neutralize on thinner overlap than this
# Raw-Sharpe floor above which a candidate has a "meaningful" edge → if that
# edge is NOT backed by a significant alpha t-stat, it is disguised beta.
_DISGUISED_SHARPE_FLOOR: float = 1.0
# R² above which a non-trivial-variance candidate is deemed factor-explained
# (catches pure beta that nets to ~zero standalone Sharpe).
_DISGUISED_R2_FLOOR: float = 0.90

# The plumbed universe (matches the trading-side LIVE_SYMBOLS set).
FACTOR_SYMBOLS: list[str] = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]

# BTC is the MARKET proxy; its DVOL is the VOL proxy.
_MARKET_SYMBOL = "BTCUSDT"
_FUNDING_SERIES_ID = "funding_rate"
_DVOL_SERIES_ID = "deribit_btc_dvol"
_DAILY_INTERVAL = "1d"


# ── Small shared helpers ─────────────────────────────────────────────────────
def _simple_returns(closes: dict[date, float]) -> dict[date, float]:
    """Day-over-day simple returns of a date→level series.

    Emits a return on every date that has an immediately-preceding observation
    in the sorted series. The first date (no prior) is omitted. A non-positive
    prior level is skipped (undefined return). Same convention as
    ``hedging_gate._returns`` but keyed by date so factors stay date-aligned.
    """
    out: dict[date, float] = {}
    days = sorted(closes)
    for prev, cur in zip(days, days[1:]):
        p = closes[prev]
        if p:
            out[cur] = closes[cur] / p - 1.0
    return out


def _daily_last(series: list[tuple[datetime, float]]) -> dict[date, float]:
    """Collapse an intraday/8h macro series to ONE value per calendar date.

    Aggregation choice: LAST observation of each UTC calendar date (the value
    "as known by end of day"). ``macro_raw`` funding/DVOL arrive at intraday or
    8h cadence; the factor model operates daily, so we take the closing reading
    per day. Input is ascending (repo guarantees ``ORDER BY event_time ASC``),
    so a plain dict assignment keeps the last write per date.
    """
    out: dict[date, float] = {}
    for ts, val in series:
        d = ts.astimezone(timezone.utc).date() if ts.tzinfo else ts.date()
        out[d] = val
    return out


# ── The four pure factor builders ────────────────────────────────────────────
def market_factor(btc_closes: dict[date, float]) -> dict[date, float]:
    """MARKET factor = BTC daily simple returns."""
    return _simple_returns(btc_closes)


def momentum_factor(
    closes_by_sym: dict[str, dict[date, float]], *, lookback: int, top_k: int
) -> dict[date, float]:
    """Cross-sectional momentum: each day rank symbols by their trailing
    ``lookback``-day return, go long the top-``top_k`` / short the bottom-``top_k``,
    and realise the factor return from the NEXT day's moves.

    Point-in-time (no look-ahead): on realised day ``t`` the rank is computed
    from the trailing return measured over ``[t-1-lookback, t-1]`` — i.e. only
    information available by the close of ``t-1``. The factor RETURN at ``t`` is
    then ``mean(top_k legs' return on t) − mean(bottom_k legs' return on t)``,
    which uses t's price move only.
    """
    if not closes_by_sym:
        return {}

    rets_by_sym = {s: _simple_returns(c) for s, c in closes_by_sym.items()}
    # Candidate realised days = dates for which every symbol has a return.
    all_days: set[date] = set.intersection(
        *[set(r) for r in rets_by_sym.values()]
    ) if rets_by_sym else set()
    days = sorted(all_days)

    factor: dict[date, float] = {}
    for i, t in enumerate(days):
        # Need ``lookback`` realised-return days strictly BEFORE t (i.e. up to
        # t-1) to form the ranking signal. days[i-lookback : i] are those days.
        if i < lookback:
            continue
        window = days[i - lookback:i]  # trailing, ends at t-1 (no look-ahead)
        trailing: dict[str, float] = {}
        for s in rets_by_sym:
            # cumulative trailing return over the window (compounded)
            cum = 1.0
            for d in window:
                cum *= 1.0 + rets_by_sym[s][d]
            trailing[s] = cum - 1.0
        ranked = sorted(trailing, key=lambda s: trailing[s], reverse=True)
        k = min(top_k, len(ranked) // 2) if len(ranked) >= 2 else 0
        if k == 0:
            continue
        longs, shorts = ranked[:k], ranked[-k:]
        long_ret = sum(rets_by_sym[s][t] for s in longs) / k    # realised at t
        short_ret = sum(rets_by_sym[s][t] for s in shorts) / k
        factor[t] = long_ret - short_ret
    return factor


def carry_factor(
    funding_by_sym: dict[str, dict[date, float]],
    returns_by_sym: dict[str, dict[date, float]],
) -> dict[date, float]:
    """Cross-sectional carry: long the high-funding symbols / short the
    low-funding ones; the factor RETURN is the spread of the next day's symbol
    price returns.

    Point-in-time (no look-ahead): funding is the SORT key and is taken as the
    value known by ``t-1``; the factor return at ``t`` is realised from t's
    price moves (``returns_by_sym``). Funding levels carry no return themselves
    here — they only choose the legs.
    """
    if not funding_by_sym or not returns_by_sym:
        return {}

    common_syms = [s for s in funding_by_sym if s in returns_by_sym]
    if len(common_syms) < 2:
        return {}

    # Realised days = dates present in every symbol's RETURN series.
    ret_days: set[date] = set.intersection(
        *[set(returns_by_sym[s]) for s in common_syms]
    )
    days = sorted(ret_days)

    factor: dict[date, float] = {}
    for t in days:
        # Sort key: the most recent funding observation STRICTLY before t (known
        # by t-1). Symbols without such a prior funding reading are excluded.
        sort_key: dict[str, float] = {}
        for s in common_syms:
            prior = [d for d in funding_by_sym[s] if d < t]
            if prior:
                sort_key[s] = funding_by_sym[s][max(prior)]
        if len(sort_key) < 2:
            continue
        ranked = sorted(sort_key, key=lambda s: sort_key[s], reverse=True)
        k = min(MOM_TOP_K, len(ranked) // 2)
        if k == 0:
            continue
        longs, shorts = ranked[:k], ranked[-k:]
        long_ret = sum(returns_by_sym[s][t] for s in longs) / k    # realised t
        short_ret = sum(returns_by_sym[s][t] for s in shorts) / k
        factor[t] = long_ret - short_ret
    return factor


def vol_factor(dvol_series: dict[date, float]) -> dict[date, float]:
    """VOL factor = daily change in (BTC) implied vol (DVOL).

    Change convention: simple percent change ``dvol[t]/dvol[t-1] − 1`` (same
    ``_simple_returns`` helper as the price factors, for a consistent scale).
    Single series → BTC DVOL as the market-vol proxy.
    """
    return _simple_returns(dvol_series)


# ── OLS neutralization + DISGUISED_BETA (pure) ───────────────────────────────
def neutralize(
    candidate_returns: dict[date, float],
    factors: dict[str, dict[date, float]],
) -> dict:
    """Regress a candidate's daily returns on the factor series → idiosyncratic
    alpha, per-factor betas, residual series, alpha t-stat, DISGUISED_BETA flag.

    The candidate and every factor series are aligned on their COMMON dates
    (sorted). The design matrix ``X`` is ``[1 | f1 | f2 | …]`` (intercept first),
    ``y`` the candidate returns on those dates; OLS via ``numpy.linalg.lstsq``.

      - ``alpha``       = intercept coefficient.
      - ``betas``       = ``{factor_name: coefficient}``.
      - ``residuals``   = ``{date: y - X·coef}`` over the common dates.
      - ``alpha_tstat`` = ``alpha / se_alpha`` where
        ``se_alpha = sqrt(resid_var · (XᵀX)⁻¹[0,0])`` and
        ``resid_var = SSR / (n − k)`` (k = number of coefs incl. intercept).
        Guarded → ``None`` when ``se_alpha == 0`` or ``n <= k``.
      - ``disguised_beta`` = True iff the RAW candidate has a meaningful edge
        BUT ``|alpha_tstat| < ALPHA_TSTAT_MIN`` — i.e. the standalone edge is
        explained away by factor exposure. "Meaningful edge" = the candidate has
        a meaningful annualised Sharpe (``|sharpe| > _DISGUISED_SHARPE_FLOOR``
        via ``hedging_gate._sharpe``) OR its return variance is non-trivial and
        almost entirely tracked by the factors (``R² >= _DISGUISED_R2_FLOOR``).
        The R² leg catches pure-beta candidates whose factor-mirrored returns
        net to a ~zero standalone Sharpe yet are wholly factor-driven.

    Thin data (``n_obs < MIN_FACTOR_OBS``) is NEVER neutralized: returns a
    sentinel ``{disguised_beta: False, reason: "insufficient_obs", residuals:{},
    alpha: None, betas: {}, alpha_tstat: None, n_obs}``.
    """
    factor_names = sorted(factors)
    # Common dates across candidate + every factor.
    common: set[date] = set(candidate_returns)
    for name in factor_names:
        common &= set(factors[name])
    dates = sorted(common)
    n_obs = len(dates)

    if n_obs < MIN_FACTOR_OBS:
        return {
            "alpha": None,
            "alpha_tstat": None,
            "betas": {},
            "residuals": {},
            "disguised_beta": False,
            "reason": "insufficient_obs",
            "n_obs": n_obs,
        }

    y = np.array([candidate_returns[d] for d in dates], dtype=float)
    cols = [np.ones(n_obs)]
    for name in factor_names:
        cols.append(np.array([factors[name][d] for d in dates], dtype=float))
    X = np.column_stack(cols)
    k = X.shape[1]

    coef, _resid_ss, _rank, _sv = np.linalg.lstsq(X, y, rcond=None)
    alpha = float(coef[0])
    betas = {name: float(coef[1 + i]) for i, name in enumerate(factor_names)}

    fitted = X @ coef
    resid = y - fitted
    residuals = {d: float(resid[i]) for i, d in enumerate(dates)}

    # OLS standard error of the intercept.
    alpha_tstat: float | None = None
    # ``alpha_significant`` is True only when a finite t-stat clears the floor.
    # ``alpha_judgeable`` is True whenever we CAN rule on significance — which
    # includes the perfect-fit (se==0) case: zero residual variance with a
    # ~zero alpha means the candidate is wholly factor-explained and carries no
    # significant idiosyncratic edge. Only the degenerate ``n<=k`` case (no real
    # fit) leaves significance unjudgeable.
    alpha_significant = False
    alpha_judgeable = n_obs > k
    if n_obs > k:
        ssr = float(resid @ resid)
        resid_var = ssr / (n_obs - k)
        # Treat an effectively-zero residual variance as a PERFECT fit: a tiny
        # SSR is floating-point dust (a candidate that is an exact linear combo
        # of the factors), not real idiosyncratic variation. Otherwise alpha ≈
        # 1e-18 divided by se ≈ 1e-18 yields a meaningless ~2 t-stat. Tolerance
        # is scaled to the candidate's own variance so it is unit-agnostic.
        y_scale = float(y @ y) / n_obs
        perfect_fit = resid_var <= 1e-20 * max(y_scale, 1.0)
        try:
            xtx_inv = np.linalg.inv(X.T @ X)
            var_alpha = resid_var * float(xtx_inv[0, 0])
        except np.linalg.LinAlgError:
            var_alpha = 0.0
        if not perfect_fit and var_alpha > 0:
            se_alpha = math.sqrt(var_alpha)
            alpha_tstat = alpha / se_alpha
            alpha_significant = abs(alpha_tstat) >= ALPHA_TSTAT_MIN
        # else: se==0 → perfect fit → alpha not significant (alpha_tstat stays
        #       None) but significance IS judged (alpha_significant=False).

    # Raw candidate Sharpe (annualised) over the same common dates.
    raw_rets = [candidate_returns[d] for d in dates]
    raw_sharpe = hedging_gate._sharpe(raw_rets)
    meaningful_sharpe = (
        raw_sharpe is not None and abs(raw_sharpe) > _DISGUISED_SHARPE_FLOOR
    )
    # R² of the factor fit (1 − SSR/SST); a pure-beta candidate nets to ~0
    # Sharpe but is wholly factor-explained → still disguised beta.
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - (float(resid @ resid) / sst) if sst > 0 else 0.0
    factor_explained = sst > 0 and r2 >= _DISGUISED_R2_FLOOR
    has_meaningful_edge = meaningful_sharpe or factor_explained
    disguised_beta = bool(
        has_meaningful_edge and alpha_judgeable and not alpha_significant
    )

    return {
        "alpha": alpha,
        "alpha_tstat": alpha_tstat,
        "betas": betas,
        "residuals": residuals,
        "disguised_beta": disguised_beta,
        "n_obs": n_obs,
    }


# ── Thin async assembler ─────────────────────────────────────────────────────
async def build_factors(
    conn: asyncpg.Connection, start: datetime, end: datetime
) -> dict[str, dict[date, float]]:
    """Fetch the underlying data and assemble the four daily factor series.

    Returns ``{"MARKET":{date:ret}, "MOMENTUM":{...}, "CARRY":{...}, "VOL":{...}}``
    over the half-open window ``[start, end)``. **Degrades gracefully:** if a
    factor's source data is absent or too thin to compute, that factor key is
    OMITTED (and logged) rather than crashing the whole build.
    """
    # 1. Daily closes per symbol (MARKET + MOMENTUM share these).
    closes_by_sym: dict[str, dict[date, float]] = {}
    for sym in FACTOR_SYMBOLS:
        try:
            rows = await market_data.fetch_daily_closes(
                conn, symbol=sym, interval=_DAILY_INTERVAL, start=start, end=end
            )
        except Exception:  # pragma: no cover - defensive; never crash a build
            logger.exception("factor build: market_data fetch failed for %s", sym)
            continue
        if rows:
            closes_by_sym[sym] = {ts.date(): px for ts, px in rows}

    factors: dict[str, dict[date, float]] = {}

    # MARKET — BTC returns.
    btc = closes_by_sym.get(_MARKET_SYMBOL, {})
    mkt = market_factor(btc)
    if mkt:
        factors["MARKET"] = mkt
    else:
        logger.info("factor build: MARKET omitted (no BTC closes)")

    # MOMENTUM — needs >= 2 symbols.
    if len(closes_by_sym) >= 2:
        mom = momentum_factor(closes_by_sym, lookback=MOM_LOOKBACK, top_k=MOM_TOP_K)
        if mom:
            factors["MOMENTUM"] = mom
        else:
            logger.info(
                "factor build: MOMENTUM omitted (thin — < %d trailing days)",
                MOM_LOOKBACK,
            )
    else:
        logger.info("factor build: MOMENTUM omitted (< 2 symbols with closes)")

    # CARRY — per-symbol funding + the same symbols' daily returns.
    funding_by_sym: dict[str, dict[date, float]] = {}
    for sym in FACTOR_SYMBOLS:
        try:
            rows = await macro_raw.fetch_series(
                conn, series_id=_FUNDING_SERIES_ID, symbol=sym, start=start, end=end
            )
        except Exception:  # pragma: no cover - defensive
            logger.exception("factor build: funding fetch failed for %s", sym)
            continue
        if rows:
            funding_by_sym[sym] = _daily_last(rows)
    returns_by_sym = {s: _simple_returns(c) for s, c in closes_by_sym.items()}
    if funding_by_sym:
        carry = carry_factor(funding_by_sym, returns_by_sym)
        if carry:
            factors["CARRY"] = carry
        else:
            logger.info("factor build: CARRY omitted (insufficient funding/return overlap)")
    else:
        logger.info("factor build: CARRY omitted (no funding rows)")

    # VOL — BTC DVOL (single series; symbol filter left open).
    try:
        dvol_rows = await macro_raw.fetch_series(
            conn, series_id=_DVOL_SERIES_ID, symbol=None, start=start, end=end
        )
    except Exception:  # pragma: no cover - defensive
        logger.exception("factor build: DVOL fetch failed")
        dvol_rows = []
    if dvol_rows:
        vol = vol_factor(_daily_last(dvol_rows))
        if vol:
            factors["VOL"] = vol
        else:
            logger.info("factor build: VOL omitted (thin DVOL — < 2 daily points)")
    else:
        logger.info("factor build: VOL omitted (no DVOL rows)")

    return factors
