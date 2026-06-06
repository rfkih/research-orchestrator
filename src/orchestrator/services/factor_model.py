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
from datetime import date, datetime, timezone

import asyncpg

from ..repo import macro_raw, market_data

logger = logging.getLogger(__name__)

# ── Pinned constants (operator-tunable; pin-tested) ─────────────────────────
MOM_LOOKBACK: int = 20          # trailing days used to rank momentum
MOM_TOP_K: int = 1              # legs long the top-k / short the bottom-k

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
