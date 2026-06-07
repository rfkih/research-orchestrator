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


# ── Task 3: OLS neutralization + DISGUISED_BETA ─────────────────────────────
def test_neutralize_recovers_pure_market_beta():
    # candidate == 2x market exactly → beta_mkt≈2, residual≈0, alpha≈0.
    # Use >= 60 obs so it clears MIN_FACTOR_OBS.
    mkt = {date(2024, 1, 1) + __import__("datetime").timedelta(days=i):
           0.01 * ((-1) ** i) for i in range(70)}
    factors = {"MARKET": mkt}
    cand = {d: 2.0 * r for d, r in mkt.items()}
    res = fm.neutralize(cand, factors)
    assert abs(res["betas"]["MARKET"] - 2.0) < 1e-6
    assert abs(res["alpha"]) < 1e-6
    assert max(abs(x) for x in res["residuals"].values()) < 1e-6
    assert res["disguised_beta"] is True   # raw edge is pure beta
    assert res["n_obs"] >= 60


def test_neutralize_keeps_real_alpha():
    # candidate = market + positive idiosyncratic drift + small noise →
    # alpha>0, alpha_tstat>=2, residuals nonzero, disguised_beta False.
    import random
    from datetime import timedelta
    rng = random.Random(7)
    base = date(2024, 1, 1)
    mkt = {base + timedelta(days=i): 0.01 * ((-1) ** i) for i in range(120)}
    factors = {"MARKET": mkt}
    drift = 0.003
    cand = {d: 1.0 * r + drift + rng.gauss(0.0, 0.0005)
            for d, r in mkt.items()}
    res = fm.neutralize(cand, factors)
    assert res["alpha"] > 0
    assert res["alpha_tstat"] is not None and res["alpha_tstat"] >= 2.0
    assert max(abs(x) for x in res["residuals"].values()) > 0
    assert res["disguised_beta"] is False


def test_neutralize_insufficient_obs():
    from datetime import timedelta
    base = date(2024, 1, 1)
    mkt = {base + timedelta(days=i): 0.01 for i in range(40)}
    factors = {"MARKET": mkt}
    cand = {d: 2.0 * r for d, r in mkt.items()}
    res = fm.neutralize(cand, factors)
    assert res["reason"] == "insufficient_obs"
    assert res["disguised_beta"] is False
    assert res["residuals"] == {}
    assert res["alpha"] is None
    assert res["betas"] == {}
    assert res["alpha_tstat"] is None
    assert res["n_obs"] == 40


def test_neutralize_rank_deficient_factors_does_not_false_flag_alpha():
    # Two perfectly-collinear factors → XᵀX is singular/rank-deficient, so the
    # alpha t-stat cannot be computed reliably. A candidate with GENUINE
    # idiosyncratic alpha must NOT be wrongly flagged disguised_beta in that
    # un-judgeable case — we return reason='rank_deficient' and keep it.
    import random
    from datetime import timedelta
    rng = random.Random(11)
    base = date(2024, 1, 1)
    mkt = {base + timedelta(days=i): 0.01 * ((-1) ** i) for i in range(120)}
    # DUP is an exact copy of MARKET → the two factor columns are collinear.
    dup = dict(mkt)
    drift = 0.003
    cand = {d: 1.0 * r + drift + rng.gauss(0.0, 0.0005) for d, r in mkt.items()}
    res = fm.neutralize(cand, {"MARKET": mkt, "DUP": dup})
    assert res["reason"] == "rank_deficient"
    assert res["disguised_beta"] is False   # un-judgeable → never false-flag
    assert res["alpha_tstat"] is None
    assert res["n_obs"] >= 60


def test_neutralize_high_r2_with_alpha_is_kept():
    # High factor exposure (R² well above the 0.90 floor) WITH a real,
    # significant idiosyncratic alpha must be KEPT (not flagged disguised_beta).
    # candidate = 3x MARKET (→ very high R²) + significant drift + tiny noise.
    import random
    from datetime import timedelta
    rng = random.Random(3)
    base = date(2024, 1, 1)
    mkt = {base + timedelta(days=i): 0.02 * ((-1) ** i) for i in range(120)}
    factors = {"MARKET": mkt}
    drift = 0.004
    cand = {d: 3.0 * r + drift + rng.gauss(0.0, 0.0003) for d, r in mkt.items()}
    res = fm.neutralize(cand, factors)
    # high factor exposure → R² above the floor
    sst = sum((v - sum(cand.values()) / len(cand)) ** 2 for v in cand.values())
    ssr = sum(v ** 2 for v in res["residuals"].values())
    r2 = 1.0 - ssr / sst
    assert r2 >= fm._DISGUISED_R2_FLOOR, f"expected high R², got {r2}"
    # but a real significant alpha → KEEP
    assert res["alpha_tstat"] is not None
    assert abs(res["alpha_tstat"]) >= fm.ALPHA_TSTAT_MIN
    assert res["disguised_beta"] is False


# ── Pin tests (Step 3 / Task 3 Step 5) ──────────────────────────────────────
def test_momentum_constants_are_pinned():
    assert fm.MOM_LOOKBACK == 20
    assert fm.MOM_TOP_K == 1


def test_neutralize_constants_are_pinned():
    assert fm.ALPHA_TSTAT_MIN == 2.0
    assert fm.MIN_FACTOR_OBS == 60


def test_disguised_beta_floors_are_pinned():
    # These two floors are load-bearing for the DISGUISED_BETA decision —
    # pin their exact values so accidental drift surfaces (mirrors the
    # ALPHA_TSTAT_MIN / MIN_FACTOR_OBS pin-test above).
    assert fm._DISGUISED_R2_FLOOR == 0.90
    assert fm._DISGUISED_SHARPE_FLOOR == 1.0


def test_factor_symbols_constant_is_pinned():
    assert fm.FACTOR_SYMBOLS == [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"
    ]
