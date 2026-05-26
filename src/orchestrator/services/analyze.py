"""Quant-grade analysis — port of ``research/scripts/analyze-run.py``.

For one backtest_run the orchestrator computes:

  * Bootstrap 95% confidence interval on Profit Factor (n_resamples=2000).
    Hedge-fund discipline: a strategy "has edge" only when the CI lower
    bound > 1.0, not when the point estimate > 1.0.
  * Probabilistic Sharpe Ratio (Bailey & Lopez de Prado 2014). Returns
    the probability that the true Sharpe is > 0 after correcting for
    skew, kurtosis, sample size, and selection bias (Bonferroni N tests).
  * Slippage sensitivity: net PnL after +5/+10/+20/+50 bps round-trip
    haircut on each trade.
  * Regime-stratified breakdown by ``entry_trend_regime`` and calendar
    quarter.
  * Statistical verdict: SIGNIFICANT_EDGE / INSUFFICIENT_EVIDENCE /
    NO_EDGE based on n>=100 AND PF CI excluding 1.0.

Pure functions only — no I/O. The trades+run dicts are passed in. The
caller (tick service) handles the DB fetch.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from typing import Any, Iterable

# Below this many trades, statistical_verdict is INSUFFICIENT_EVIDENCE
# regardless of point estimates. Matches research/CLAUDE.md "Profitability bar".
MIN_TRADES_FOR_SIG = 100
PF_INFINITY_SENTINEL = 9999.0

# Annualized compounded-return threshold for the economic PASS gate. Strategies
# clearing the V11 statistical gate must ALSO compound to at least this much
# per year at 90% sizing to be worth keeping. Mirrors CLAUDE.md's headline
# "10%/yr net after fees+slippage" profitability bar — a strategy below this
# can have real edge yet still not deserve real capital.
ANNUALIZED_RETURN_PASS_THRESHOLD_PCT = 10.0

# Calendar days per year. Crypto markets trade 24/7 so we use 365 rather
# than 252 trading days. Plain integer per operator decision — leap-year
# drift is sub-percent and doesn't move the PASS/ITERATE boundary.
DAYS_PER_YEAR = 365


def _f(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, Decimal):
        return float(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def profit_factor(pnls: list[float]) -> float | None:
    """PF = gross profit / |gross loss|. None for n<2; sentinel for no losses."""
    if len(pnls) < 2:
        return None
    wins = sum(p for p in pnls if p > 0)
    losses = sum(-p for p in pnls if p < 0)
    if losses == 0:
        return PF_INFINITY_SENTINEL if wins > 0 else None
    return wins / losses


def bootstrap_pf_ci(
    pnls: list[float],
    n_resamples: int = 2000,
    conf: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap CI on profit factor.

    Sentinel-PF resamples (no losers in the resample) are dropped — we
    don't want infinity-PF samples skewing the upper percentile. With
    very small N this can leave us with too few finite resamples; the
    caller treats ``low is None`` as "not computable".
    """
    if len(pnls) < 5:
        return {"low": None, "high": None, "mean": None, "n_resamples": 0,
                "note": "n too small for bootstrap"}
    rng = random.Random(seed)
    pf_samples: list[float] = []
    for _ in range(n_resamples):
        sample = [rng.choice(pnls) for _ in pnls]
        pf = profit_factor(sample)
        if pf is not None and pf < PF_INFINITY_SENTINEL:
            pf_samples.append(pf)
    if not pf_samples:
        return {"low": None, "high": None, "mean": None, "n_resamples": 0,
                "note": "no resamples produced finite PF"}
    pf_samples.sort()
    alpha = (1 - conf) / 2
    lo_idx = int(alpha * len(pf_samples))
    hi_idx = max(int((1 - alpha) * len(pf_samples)) - 1, 0)
    return {
        "low": round(pf_samples[lo_idx], 4),
        "high": round(pf_samples[hi_idx], 4),
        "mean": round(sum(pf_samples) / len(pf_samples), 4),
        "n_resamples": len(pf_samples),
        "note": None,
    }


def sharpe_ratio(returns: list[float], ann_factor: float | None = None) -> float | None:
    if len(returns) < 2:
        return None
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    if var <= 0:
        return None
    sr = mean / math.sqrt(var)
    return sr * math.sqrt(ann_factor) if ann_factor else sr


def sortino_ratio(
    returns: list[float], ann_factor: float | None = None, target: float = 0.0
) -> float | None:
    if len(returns) < 2:
        return None
    mean_excess = (sum(returns) / len(returns)) - target
    downside = [r - target for r in returns if r < target]
    if len(downside) < 2:
        return None
    var = sum(d**2 for d in downside) / len(downside)
    if var <= 0:
        return None
    sortino = mean_excess / math.sqrt(var)
    return sortino * math.sqrt(ann_factor) if ann_factor else sortino


def skewness(values: list[float]) -> float:
    if len(values) < 3:
        return 0.0
    mean = sum(values) / len(values)
    m2 = sum((v - mean) ** 2 for v in values) / len(values)
    m3 = sum((v - mean) ** 3 for v in values) / len(values)
    if m2 <= 0:
        return 0.0
    return m3 / (m2**1.5)


def excess_kurtosis(values: list[float]) -> float:
    if len(values) < 4:
        return 0.0
    mean = sum(values) / len(values)
    m2 = sum((v - mean) ** 2 for v in values) / len(values)
    m4 = sum((v - mean) ** 4 for v in values) / len(values)
    if m2 <= 0:
        return 0.0
    return m4 / (m2**2) - 3.0


def _norm_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _norm_inv(p: float) -> float:
    """Beasley-Springer-Moro normal inverse CDF, ~1e-9 accurate."""
    if p <= 0:
        return -float("inf")
    if p >= 1:
        return float("inf")
    a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
         1.383577518672690e2, -3.066479806614716e1, 2.506628277459239e0]
    b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
         6.680131188771972e1, -1.328068155288572e1]
    c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838e0,
         -2.549732539343734e0, 4.374664141464968e0, 2.938163982698783e0]
    d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996e0,
         3.754408661907416e0]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def probabilistic_sharpe_ratio(
    sr: float | None,
    n_obs: int,
    skew: float,
    kurt: float,
    n_strategies_tested: int = 5,
) -> float | None:
    """Deflated Sharpe (Bailey & Lopez de Prado 2014).

    Returns P(true SR > 0 | sample). PSR > 0.95 is the quant-grade
    significance bar; below 0.5 the Sharpe is indistinguishable from zero.

    NOTE: ``n_strategies_tested`` defaults to 5 for back-compat with the
    hardcoded V11 contract. Production callers should use
    ``deflated_sharpe_ratio`` and pass the actual cumulative trial
    count from ``hypothesis_audit`` — see analyze_run().
    """
    if sr is None or n_obs < 30:
        return None
    z_alpha = _norm_inv(1 - 0.05 / n_strategies_tested)
    sr_star = z_alpha / math.sqrt(n_obs)
    denom_sq = 1 - skew * sr + ((kurt - 1) / 4) * (sr**2)
    if denom_sq <= 0:
        return None
    z = (sr - sr_star) * math.sqrt((n_obs - 1) / denom_sq)
    return _norm_cdf(z)


def deflated_sharpe_ratio(
    sr: float | None,
    n_obs: int,
    skew: float,
    kurt: float,
    n_trials: int,
    pnls: list[float] | None = None,
    bootstrap_B: int = 1000,
    bootstrap_seed: int = 17,
) -> float | None:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014), using the
    REAL cumulative trial count rather than the V11 hardcoded N=5.

    Closed-form (default): the Bailey-LdP Gaussian-asymptotic standard
    error of the sample Sharpe is

        SE^2 = (1 - skew·SR + ((kurt-1)/4)·SR^2) / (n_obs - 1)

    DSR = Phi((SR - SR*) / SE), where SR* = Phi^-1(1 - 0.05/n_trials) /
    sqrt(n_obs) is the selection-bias-adjusted threshold. The
    ``denom_sq`` numerator can go non-positive when the return
    distribution is heavily skewed (skew·SR > 1 + ((kurt-1)/4)·SR^2),
    in which case the closed-form is undefined.

    Bootstrap fallback (this commit, 2026-05-21): when ``denom_sq <= 0``
    AND a ``pnls`` series is supplied, fall back to the non-parametric
    bootstrap DSR procedure recommended in Bailey & López de Prado
    (2014) §5 *Empirical Estimation of the Sampling Distribution of the
    Sharpe Ratio*: resample ``pnls`` with replacement ``bootstrap_B``
    times, compute the per-resample Sharpe (annualised consistently
    with the input ``sr``), use the empirical std of that distribution
    as SE, and recompute z = (SR - SR*) / SE → DSR = Phi(z). This is
    methodology-faithful — the V11 DSR threshold is unchanged; only
    the *computation* extends to cover the case the closed-form cannot.
    If ``pnls`` is None on a denom_sq<=0 path the function still
    returns None so the gate correctly fails-closed.

    Reference: Bailey, López de Prado (2014), *The Deflated Sharpe
    Ratio: Correcting for Selection Bias, Backtest Overfitting, and
    Non-Normality*. The HLZ (Harvey, Liu, Zhu 2016) data-mining
    adjustment is the same idea applied to factor research — both
    converge on "scale your significance bar by sqrt(2 log N)".

    n_trials clamps to >= 1; passing 0 would otherwise blow up the log.
    """
    if sr is None or n_obs < 30:
        return None
    n_trials = max(1, int(n_trials))
    z_alpha = _norm_inv(1 - 0.05 / n_trials)
    sr_star = z_alpha / math.sqrt(n_obs)
    denom_sq = 1 - skew * sr + ((kurt - 1) / 4) * (sr**2)
    if denom_sq > 0:
        z = (sr - sr_star) * math.sqrt((n_obs - 1) / denom_sq)
        return _norm_cdf(z)
    # Closed-form degenerate (non-Gaussian shape). Try the Bailey-LdP
    # bootstrap fallback if we have the underlying PnL series.
    if pnls is None or len(pnls) < 30:
        return None
    return _bootstrap_dsr(
        pnls=pnls,
        sr_observed=sr,
        sr_star=sr_star,
        n_obs=n_obs,
        B=bootstrap_B,
        seed=bootstrap_seed,
    )


def _bootstrap_dsr(
    pnls: list[float],
    sr_observed: float,
    sr_star: float,
    n_obs: int,
    B: int = 1000,
    seed: int = 17,
) -> float | None:
    """Non-parametric DSR via bootstrap (Bailey & LdP 2014 §5).

    Both ``sr_observed`` and ``sr_star`` are passed in on the
    *annualised* basis (matching closed-form). The bootstrap is run on
    the per-sample SR (unannualised — ``ann_factor=None``); we recover
    the annualisation factor from the observed pair (annualised SR vs
    the per-sample SR of the original ``pnls``) and apply it to the
    bootstrap SE so the z-statistic lives on the same basis as the
    closed-form would.

    Returns ``None`` (gate fails closed) on:
      * ``pnls`` shorter than 30 (insufficient resampling base)
      * fewer than 30 finite bootstrap resamples
      * bootstrap variance zero (degenerate)
      * per-sample observed SR uncomputable (e.g. zero variance pnls)
    """
    if not pnls or len(pnls) < 30:
        return None
    sr_per_sample_observed = sharpe_ratio(pnls, ann_factor=None)
    if sr_per_sample_observed is None or not math.isfinite(sr_per_sample_observed):
        return None
    # Annualisation factor implied by the (annualised, per-sample) pair.
    # Used to project the bootstrap per-sample SE up to the annualised
    # basis where sr_observed and sr_star live.
    if abs(sr_per_sample_observed) < 1e-12:
        # SR ~ 0 — annualisation is undefined; can't certify on this
        # path. Methodology-faithful: fail closed.
        return None
    sqrt_ann_factor = sr_observed / sr_per_sample_observed
    if not math.isfinite(sqrt_ann_factor) or sqrt_ann_factor <= 0:
        return None

    rng = random.Random(seed)
    n = len(pnls)
    sr_samples: list[float] = []
    for _ in range(B):
        sample = [rng.choice(pnls) for _ in range(n)]
        sr_b = sharpe_ratio(sample, ann_factor=None)
        if sr_b is not None and math.isfinite(sr_b):
            sr_samples.append(sr_b)
    if len(sr_samples) < 30:
        return None
    mean_b = sum(sr_samples) / len(sr_samples)
    var_b = sum((s - mean_b) ** 2 for s in sr_samples) / (len(sr_samples) - 1)
    if var_b <= 0:
        return None
    sd_b_per_sample = math.sqrt(var_b)
    # Project per-sample SE to the annualised basis.
    se_annualised = sd_b_per_sample * sqrt_ann_factor
    if se_annualised <= 0:
        return None
    z = (sr_observed - sr_star) / se_annualised
    return _norm_cdf(z)


def slippage_sensitivity(
    trades: Iterable[dict[str, Any]],
    scenarios: tuple[int, ...] = (5, 10, 20, 50),
) -> dict[str, float]:
    """Project net PnL across slippage haircut scenarios.

    Notional source (updated 2026-05-20):
      * Primary: ``trade.notional_size`` (the per-trade quote-notional
        the engine actually sized to — exact, not a proxy).
      * Fallback: ``max(|pnl|, $10) × 50`` legacy proxy for trades
        missing notional_size (older partial-fill rows or pre-fix
        callers that don't surface the column).

    Background: the legacy proxy overstated notional 5-50x at small
    initial_capital (e.g. $100-capital backtests with $1-10 trade PnL
    floored to $10 × 50 = $500 minimum notional vs realistic $50-100).
    That overstatement made ``check_cost_realism`` produce false
    REJECTs on otherwise-passing V11+V60 candidates — see memory
    ``project_retired_cost_gate_methodology_bug`` and operator-
    escalation journal ``c58a4b8a-3d14-4b37-b1fe-ab5eb45cb3f7``.
    """
    trades_list = list(trades)
    pnls = [_f(t.get("realized_pnl_amount")) or 0.0 for t in trades_list]
    # Prefer the engine-sized notional; fall back to the legacy proxy
    # only when notional_size is missing (partial fills / legacy rows).
    notionals: list[float] = []
    for trade, pnl in zip(trades_list, pnls, strict=True):
        n = _f(trade.get("notional_size"))
        if n is not None and n > 0:
            notionals.append(n)
        else:
            notionals.append(max(abs(pnl), 10.0) * 50.0)
    out: dict[str, float] = {}
    for slip_bps in scenarios:
        total = 0.0
        for pnl, notional in zip(pnls, notionals, strict=True):
            haircut = notional * (slip_bps / 10000.0)
            total += pnl - haircut
        out[f"+{slip_bps}bps"] = round(total, 4)
    return out


def regime_stratify(trades: Iterable[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    """Split by entry_trend_regime + calendar quarter."""
    by_regime: dict[str, dict[str, float]] = defaultdict(
        lambda: {"n": 0, "pnl": 0.0, "wins": 0}
    )
    by_quarter: dict[str, dict[str, float]] = defaultdict(
        lambda: {"n": 0, "pnl": 0.0, "wins": 0}
    )
    for t in trades:
        pnl = _f(t.get("realized_pnl_amount")) or 0.0
        regime = t.get("entry_trend_regime") or "UNKNOWN"
        by_regime[regime]["n"] += 1
        by_regime[regime]["pnl"] += pnl
        if pnl > 0:
            by_regime[regime]["wins"] += 1
        et = t.get("entry_time")
        if isinstance(et, datetime):
            q = f"{et.year}Q{(et.month - 1) // 3 + 1}"
            by_quarter[q]["n"] += 1
            by_quarter[q]["pnl"] += pnl
            if pnl > 0:
                by_quarter[q]["wins"] += 1

    def fmt(d: dict[str, dict[str, float]]) -> dict[str, dict[str, Any]]:
        return {
            k: {
                "n": int(v["n"]),
                "pnl": round(v["pnl"], 4),
                "win_rate": round(v["wins"] / v["n"], 4) if v["n"] else None,
            }
            for k, v in d.items()
        }

    return {"by_trend_regime": fmt(by_regime), "by_quarter": fmt(by_quarter)}


def annualize_geometric_return(
    geometric_return_pct: float | None, days: int | None
) -> float | None:
    """Annualize a cumulative geometric return over a backtest window.

    ``geometric_return_pct`` is the persisted
    ``backtest_run.geometric_return_pct_at_alloc_90`` — the compounded equity
    multiplier minus one, in percent, assuming every trade was sized at 90%
    of equity. We extrapolate the same compounding rate to one year:

    .. code-block:: text

        multiplier = 1 + geom/100
        annualized_multiplier = multiplier ^ (365 / days)
        annualized_geom_pct = (annualized_multiplier - 1) * 100

    Edge cases:

    * ``None`` input or ``days <= 0`` → ``None`` (caller must decide; we don't
      fabricate a number).
    * ``multiplier <= 0`` (ruin: at least one trade compounded the equity
      through zero) → ``-100.0`` regardless of window. A ruined strategy is
      ruined whether the window was a month or a decade.
    * ``days < 30`` produces wildly extrapolated values — caller is responsible
      for not gating on those (the V11 ``n>=100`` requirement makes <30 day
      backtests rare anyway, but we don't clamp here).
    """
    if geometric_return_pct is None or days is None or days <= 0:
        return None
    multiplier = 1.0 + (geometric_return_pct / 100.0)
    if multiplier <= 0.0:
        return -100.0
    years = days / DAYS_PER_YEAR
    if years <= 0:
        return None
    annualized_multiplier = multiplier ** (1.0 / years)
    return (annualized_multiplier - 1.0) * 100.0


DSR_SIGNIFICANCE_THRESHOLD = 0.95


def statistical_verdict(
    n: int, pf_ci: dict[str, Any], dsr: float | None = None
) -> dict[str, str]:
    """SIGNIFICANT_EDGE / INSUFFICIENT_EVIDENCE / NO_EDGE per V11+ rules,
    plus a DSR multiplicity gate (Tier 1 quant-grade fix).

    The PF-CI gate is necessary but not sufficient: a strategy whose 95%
    CI excludes 1.0 may still be a noise pass once you account for the
    cumulative trial count. If DSR < 0.95, demote SIGNIFICANT_EDGE to
    INSUFFICIENT_EVIDENCE with a reason that names the multiplicity.

    DSR is None for n<30 (insufficient observations for the PSR formula);
    in that case we fall back to PF-only behaviour rather than blocking
    valid early-stage iterations.
    """
    if n < MIN_TRADES_FOR_SIG:
        return {"verdict": "INSUFFICIENT_EVIDENCE",
                "reason": f"n={n} below n>={MIN_TRADES_FOR_SIG} threshold"}
    lo, hi = pf_ci.get("low"), pf_ci.get("high")
    if lo is None or hi is None:
        return {"verdict": "INSUFFICIENT_EVIDENCE",
                "reason": "PF confidence interval not computable"}
    if lo > 1.0:
        if dsr is not None and dsr < DSR_SIGNIFICANCE_THRESHOLD:
            return {
                "verdict": "INSUFFICIENT_EVIDENCE",
                "reason": (
                    f"PF 95% CI [{lo}, {hi}] favorable but DSR={round(dsr, 4)} "
                    f"< {DSR_SIGNIFICANCE_THRESHOLD} — selection-bias deflation "
                    f"rejects after multiplicity correction."
                ),
            }
        return {"verdict": "SIGNIFICANT_EDGE",
                "reason": f"95% CI [{lo}, {hi}] excludes 1.0 favorably"}
    if hi < 1.0:
        return {"verdict": "NO_EDGE",
                "reason": f"95% CI [{lo}, {hi}] excludes 1.0 adversely"}
    return {"verdict": "INSUFFICIENT_EVIDENCE",
            "reason": f"95% CI [{lo}, {hi}] spans 1.0"}


def analyze_run(
    run: dict[str, Any],
    trades: list[dict[str, Any]],
    n_strategies_tested: int = 5,
    slippage_scenarios: tuple[int, ...] = (5, 10, 20, 50),
    cumulative_trials: int | None = None,
) -> dict[str, Any]:
    """Run the full analysis. Returns the structured payload that maps
    directly into ``research_iteration_log.metrics_snapshot`` and
    ``confidence_intervals``.

    ``n_strategies_tested`` is preserved for the legacy PSR field. Pass
    ``cumulative_trials`` (from ``hypothesis_audit``) to also compute
    DSR with the real selection-bias multiplicity. When omitted, DSR
    falls back to ``n_strategies_tested`` so behaviour matches PSR.
    """
    pnls = [_f(t.get("realized_pnl_amount")) or 0.0 for t in trades]
    n = len(pnls)

    pf_pt = profit_factor(pnls)
    pf_ci = bootstrap_pf_ci(pnls)

    # Annualisation factor: trades/year. Approximation matches the bash analyzer.
    ann_factor: float | None = None
    window_days: int | None = None
    start = run.get("start_time")
    end = run.get("end_time")
    if isinstance(start, datetime) and isinstance(end, datetime):
        window_days = (end - start).days
        if window_days > 0 and n > 0:
            ann_factor = (DAYS_PER_YEAR / window_days) * n

    sr = sharpe_ratio(pnls, ann_factor)
    sortino = sortino_ratio(pnls, ann_factor)
    sk = skewness(pnls)
    kt = excess_kurtosis(pnls)
    psr = probabilistic_sharpe_ratio(sr, n, sk, kt, n_strategies_tested)

    # DSR with real cumulative trial count from hypothesis_audit. Falls
    # back to n_strategies_tested so test fixtures and legacy callers
    # that don't pass cumulative_trials get PSR-equivalent output.
    # ``pnls`` is forwarded so the Bailey-LdP bootstrap fallback can
    # fire when the closed-form denom_sq <= 0 on non-Gaussian
    # distributions (e.g. ML-gate-induced high-skew long-tail PnLs).
    dsr_n_trials = cumulative_trials if cumulative_trials is not None else n_strategies_tested
    dsr = deflated_sharpe_ratio(sr, n, sk, kt, dsr_n_trials, pnls=pnls)

    return_pct = _f(run.get("return_pct"))
    max_dd = _f(run.get("max_drawdown_pct"))
    calmar = (return_pct / max_dd) if (return_pct and max_dd and max_dd > 0) else None

    # V60 — sizing-independent compounding metric. Annualise so the gate
    # operates on a window-invariant number (a 6-month backtest at +6%
    # geom maps to ~12.4%/yr; a 2-year backtest at +25% maps to ~11.8%/yr).
    geom_return_pct = _f(run.get("geometric_return_pct_at_alloc_90"))
    annualized_geom_pct = annualize_geometric_return(geom_return_pct, window_days)

    slippage = slippage_sensitivity(trades, slippage_scenarios)
    regimes = regime_stratify(trades)
    stat = statistical_verdict(n, pf_ci, dsr)

    return {
        "n_trades": n,
        "pf_point_estimate": round(pf_pt, 4) if isinstance(pf_pt, float) else pf_pt,
        "pf_95_ci": pf_ci,
        "sharpe_annualized": round(sr, 4) if sr is not None else None,
        "sortino_annualized": round(sortino, 4) if sortino is not None else None,
        "calmar": round(calmar, 4) if calmar is not None else None,
        "psr": round(psr, 4) if psr is not None else None,
        "dsr": round(dsr, 4) if dsr is not None else None,
        "dsr_n_trials": dsr_n_trials,
        "skewness": round(sk, 4),
        "excess_kurtosis": round(kt, 4),
        "max_drawdown_pct": max_dd,
        "slippage_haircut_pnl": slippage,
        "statistical_verdict": stat,
        "regimes": regimes,
        "ann_factor": round(ann_factor, 4) if ann_factor else None,
        "window_days": window_days,
        "geometric_return_pct_at_alloc_90": (
            round(geom_return_pct, 4) if geom_return_pct is not None else None
        ),
        "annualized_geometric_return_pct_at_alloc_90": (
            round(annualized_geom_pct, 4) if annualized_geom_pct is not None else None
        ),
    }


def decision_verdict(
    stat_verdict: str,
    n: int,
    annualized_geom_return_pct: float | None,
) -> str:
    """Map statistical verdict + annualized compounded return → decision verdict.

    PASS requires:

    1. ``n >= 100`` trades (statistical floor).
    2. ``statistical_verdict == SIGNIFICANT_EDGE`` (PF 95% CI lower > 1.0
       AND DSR >= 0.95).
    3. ``annualized_geometric_return_pct_at_alloc_90 >= 10.0`` — the
       economic threshold. Replaces the prior ``+20bps slippage net > 0``
       gate. A statistically-significant edge that compounds below 10%/yr
       at 90% sizing isn't worth the operational overhead of promoting.

    The geometric return number is window-invariant (annualised), sizing-
    independent (per-trade rate, not capital-based), and inherits the
    ruin clamp from the underlying compounding — a strategy with even one
    -100%+ step gets ``-100%`` annualised and fails the gate cleanly.

    Outcomes:

    * ``n < 100`` → ``ITERATE`` (more data needed).
    * ``NO_EDGE`` → ``DISCARD``.
    * ``SIGNIFICANT_EDGE`` and annualized geom ``>= 10%`` → ``PASS``.
    * ``SIGNIFICANT_EDGE`` but annualized geom ``< 10%`` (incl. ``None``)
      → ``ITERATE`` — the edge is real but the magnitude doesn't justify
      promotion at 90% sizing; refining parameters may lift it.
    * Anything else (e.g. ``INSUFFICIENT_EVIDENCE``) → ``ITERATE``.
    """
    if n < MIN_TRADES_FOR_SIG:
        return "ITERATE"
    if stat_verdict == "NO_EDGE":
        return "DISCARD"
    if stat_verdict == "SIGNIFICANT_EDGE":
        if (
            annualized_geom_return_pct is None
            or annualized_geom_return_pct < ANNUALIZED_RETURN_PASS_THRESHOLD_PCT
        ):
            return "ITERATE"
        return "PASS"
    return "ITERATE"


def sample_size_adequate(n: int) -> bool:
    return n >= MIN_TRADES_FOR_SIG


def ml_gate_shadow_analysis(
    shadow_rows: list[dict[str, Any]],
    trades: list[dict[str, Any]],
    join_window_seconds: int = 300,
) -> dict[str, Any] | None:
    """Phase E — counterfactual analysis of the ML regime gate in shadow mode.

    Joins ml_shadow_log rows against live trades to measure what the gate
    WOULD have done if shadow_mode had been False. For each gate check that
    would have blocked a trade, the nearest trade opened within
    ``join_window_seconds`` on the same (account_strategy_id, side) is
    matched. Matched pairs reveal whether the gate was blocking profitable
    trades (gate hurts) or loss trades (gate helps).

    Returns None when there is insufficient data (<5 shadow rows or no trades).
    ``gate_adds_value`` is True when correct_block_rate > 0.5 AND the sum of
    blocked-trade PnLs is negative — meaning the gate would have avoided net
    losses. Use this as the Phase E sign-off criterion before flipping
    shadow_mode=False.
    """
    if not shadow_rows or len(shadow_rows) < 5 or not trades:
        return None

    trade_by_key: dict[tuple, list] = defaultdict(list)
    for t in trades:
        as_id = str(t.get("account_strategy_id") or "")
        direction = (t.get("direction") or "").upper()
        entry_time = t.get("entry_time")
        pnl = _f(t.get("realized_pnl_amount")) or 0.0
        if as_id and direction and entry_time:
            trade_by_key[(as_id, direction)].append((entry_time, pnl))
    for lst in trade_by_key.values():
        lst.sort(key=lambda x: x[0])

    total_checks = len(shadow_rows)
    would_block = 0
    matched_blocked: list[float] = []
    unmatched_blocked = 0
    allowed_pnls: list[float] = []
    claimed_trades: set[tuple] = set()

    def _find_nearest_trade(as_id: str, side: str, evaluated_at: Any) -> float | None:
        key = (as_id, side)
        for entry_time, pnl in trade_by_key.get(key, []):
            try:
                diff = (entry_time - evaluated_at).total_seconds()
            except TypeError:
                continue
            if 0 <= diff <= join_window_seconds:
                trade_key = (as_id, side, entry_time)
                if trade_key not in claimed_trades:
                    claimed_trades.add(trade_key)
                    return pnl
        return None

    for row in shadow_rows:
        gate_allowed = row.get("gate_allowed")
        if gate_allowed is None:
            continue
        as_id = str(row.get("account_strategy_id") or "")
        side = (row.get("side") or "").upper()
        evaluated_at = row.get("evaluated_at")

        if not gate_allowed:
            would_block += 1
            matched_pnl = _find_nearest_trade(as_id, side, evaluated_at)
            if matched_pnl is not None:
                matched_blocked.append(matched_pnl)
            else:
                unmatched_blocked += 1
        else:
            matched_pnl = _find_nearest_trade(as_id, side, evaluated_at)
            if matched_pnl is not None:
                allowed_pnls.append(matched_pnl)

    n_blocked = len(matched_blocked)
    n_allowed = len(allowed_pnls)

    correct_blocks = sum(1 for p in matched_blocked if p < 0)
    correct_block_rate = correct_blocks / n_blocked if n_blocked > 0 else None
    counterfactual_pnl = sum(matched_blocked) if n_blocked > 0 else None
    avg_blocked_pnl = counterfactual_pnl / n_blocked if n_blocked > 0 else None
    avg_allowed_pnl = sum(allowed_pnls) / n_allowed if n_allowed > 0 else None

    gate_adds_value = bool(
        correct_block_rate is not None
        and correct_block_rate > 0.5
        and counterfactual_pnl is not None
        and counterfactual_pnl < 0
    )

    return {
        "total_checks": total_checks,
        "would_block": would_block,
        "would_block_rate": round(would_block / total_checks, 4) if total_checks else None,
        "n_blocked_matched_to_trade": n_blocked,
        "n_blocked_unmatched": unmatched_blocked,
        "n_allowed_matched_to_trade": n_allowed,
        "blocked_trades_pnl_sum": round(counterfactual_pnl, 4) if counterfactual_pnl is not None else None,
        "blocked_trades_avg_pnl": round(avg_blocked_pnl, 4) if avg_blocked_pnl is not None else None,
        "allowed_trades_avg_pnl": round(avg_allowed_pnl, 4) if avg_allowed_pnl is not None else None,
        "correct_block_rate": round(correct_block_rate, 4) if correct_block_rate is not None else None,
        "gate_adds_value": gate_adds_value,
        "recommendation": (
            "Flip shadow_mode=False: gate blocks net-loss trades at rate >50%."
            if gate_adds_value else
            "Keep shadow_mode=True: insufficient evidence or gate removes profitable trades."
        ),
    }
