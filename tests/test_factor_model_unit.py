"""Pure-function unit tests for the Phase 3 factor model builders.

No DB — these exercise the deterministic factor math over already-fetched
price/macro series. The thin async ``build_factors`` is covered separately in
``tests/integration/test_factor_model.py`` (marked integration).
"""
from datetime import date

from orchestrator.services import factor_model as fm


# ── Step 1 (plan) — MARKET + MOMENTUM ───────────────────────────────────────
def test_market_factor_is_btc_returns():
    closes = {date(2024, 1, 1): 100.0, date(2024, 1, 2): 110.0, date(2024, 1, 3): 99.0}
    mkt = fm.market_factor(closes)          # {date: return}
    assert round(mkt[date(2024, 1, 2)], 4) == 0.1
    assert round(mkt[date(2024, 1, 3)], 4) == -0.1
    # first day has no prior close → no return emitted
    assert date(2024, 1, 1) not in mkt


def test_momentum_factor_long_top_short_bottom():
    # 2 symbols, A trending up, B down → long A short B → positive on up days
    closes_by_sym = {
        "A": {date(2024, 1, d): 100.0 + d for d in range(1, 6)},
        "B": {date(2024, 1, d): 100.0 - d for d in range(1, 6)},
    }
    mom = fm.momentum_factor(closes_by_sym, lookback=2, top_k=1)
    assert any(v != 0 for v in mom.values())


def test_momentum_factor_is_point_in_time_long_winner_short_loser():
    # A strictly rising, B strictly falling. Ranking on trailing returns up to
    # t-1 → A is the top (long), B is the bottom (short). The realised factor
    # return at t = A's return(t) − B's return(t). A up, B down → strictly > 0.
    closes_by_sym = {
        "A": {date(2024, 1, d): 100.0 * (1.05 ** d) for d in range(1, 8)},
        "B": {date(2024, 1, d): 100.0 * (0.95 ** d) for d in range(1, 8)},
    }
    mom = fm.momentum_factor(closes_by_sym, lookback=2, top_k=1)
    assert mom  # non-empty
    # every realised day is long-winner/short-loser → positive
    assert all(v > 0 for v in mom.values())


def test_carry_factor_high_funding_long_low_short():
    # A carries high funding, B low. Sort key = funding known at t-1; realised
    # factor return at t = A_ret(t) − B_ret(t). Make A rise and B fall so the
    # high-funding long leg outperforms → positive carry factor.
    funding_by_sym = {
        "A": {date(2024, 1, d): 0.01 for d in range(1, 8)},
        "B": {date(2024, 1, d): -0.01 for d in range(1, 8)},
    }
    returns_by_sym = {
        "A": {date(2024, 1, d): 0.02 for d in range(2, 8)},
        "B": {date(2024, 1, d): -0.02 for d in range(2, 8)},
    }
    carry = fm.carry_factor(funding_by_sym, returns_by_sym)
    assert carry
    assert all(v > 0 for v in carry.values())


def test_vol_factor_is_daily_change_in_dvol():
    # rising DVOL → positive change; falling → negative
    dvol = {date(2024, 1, 1): 50.0, date(2024, 1, 2): 55.0, date(2024, 1, 3): 49.5}
    vf = fm.vol_factor(dvol)
    assert date(2024, 1, 1) not in vf      # first day has no prior
    assert vf[date(2024, 1, 2)] > 0        # vol up
    assert vf[date(2024, 1, 3)] < 0        # vol down


def test_empty_inputs_degrade_to_empty_dicts():
    assert fm.market_factor({}) == {}
    assert fm.market_factor({date(2024, 1, 1): 100.0}) == {}  # single point
    assert fm.momentum_factor({}, lookback=2, top_k=1) == {}
    assert fm.carry_factor({}, {}) == {}
    assert fm.vol_factor({}) == {}


# ── Pin tests (Step 3) ──────────────────────────────────────────────────────
def test_momentum_constants_are_pinned():
    assert fm.MOM_LOOKBACK == 20
    assert fm.MOM_TOP_K == 1


def test_factor_symbols_constant_is_pinned():
    assert fm.FACTOR_SYMBOLS == [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"
    ]
