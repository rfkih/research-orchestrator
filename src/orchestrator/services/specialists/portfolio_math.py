"""Portfolio-construction math — pure NumPy.

Three optimisers exposed by ``POST /portfolio/optimize``:

  * **equal_weight** — 1/N. Baseline reference. Always finite, always
    feasible. The other two have to beat it on Sharpe-times-something
    to justify themselves.

  * **mean_variance_weights** — classic Markowitz minimum-variance for
    a given target Sharpe (or unconstrained, in which case it returns
    the inverse-variance solution). Closed-form, no solver.

  * **hrp_weights** — López de Prado's Hierarchical Risk Parity
    (Bailey & LdP 2016). Robust to ill-conditioned covariance matrices
    where mean-variance produces concentrated, brittle weights. Single-
    linkage hierarchical clustering implemented in ~25 LOC of NumPy so
    we don't pay scipy as a new dep for one function.

The agent (quant-portfolio-manager) reads the three outputs side-by-side
and picks one — that's the judgment layer that lives in the .md
prompt, not here. This module is pure-functions-only so it's testable
and reproducible.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np


_MIN_OBS_FOR_CORR = 5  # Mirrors services/portfolio.MIN_OVERLAP_DAYS_FOR_CORR


def _ranks_with_average_ties(values: np.ndarray) -> np.ndarray:
    """Average-rank assignment for ties. Matches scipy.stats.rankdata
    ``method='average'`` for our use case. Pure NumPy so we don't
    pull scipy in for one function."""
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


def spearman_corr_matrix(
    series_by_code: dict[str, dict[date, float]],
    *,
    min_overlap: int = _MIN_OBS_FOR_CORR,
) -> tuple[list[str], np.ndarray]:
    """Build a pairwise Spearman correlation matrix over the date-overlap
    of each (code_i, code_j) pair.

    Returns ``(codes, matrix)`` where ``codes`` is the order used for
    rows/columns of ``matrix``. Self-correlation is 1.0; entries with
    insufficient overlap (< ``min_overlap`` shared dates) are NaN so
    downstream consumers can decide whether to drop the pair or fall
    back to a baseline.

    Skipping NaN handling here (rather than imputing 0) is deliberate —
    silent zero-imputation would look like "uncorrelated" and let a
    redundant strategy slip past the gate. The portfolio agent's
    prompt explicitly directs it to refuse to allocate when a row is
    >50% NaN.
    """
    codes = sorted(series_by_code.keys())
    n = len(codes)
    matrix = np.full((n, n), np.nan, dtype=np.float64)
    for i in range(n):
        matrix[i, i] = 1.0
        a = series_by_code[codes[i]]
        for j in range(i + 1, n):
            b = series_by_code[codes[j]]
            overlap = sorted(set(a) & set(b))
            if len(overlap) < min_overlap:
                continue
            xa = np.asarray([a[d] for d in overlap], dtype=np.float64)
            xb = np.asarray([b[d] for d in overlap], dtype=np.float64)
            ra = _ranks_with_average_ties(xa)
            rb = _ranks_with_average_ties(xb)
            sd_a = ra.std()
            sd_b = rb.std()
            if sd_a == 0 or sd_b == 0:
                continue
            cov = ((ra - ra.mean()) * (rb - rb.mean())).mean()
            rho = cov / (sd_a * sd_b)
            # Numerical-precision clip — Spearman is mathematically in
            # [-1, 1] but ranks-of-tiny-floats can drift outside.
            matrix[i, j] = matrix[j, i] = float(np.clip(rho, -1.0, 1.0))
    return codes, matrix


def equal_weight(codes: list[str]) -> dict[str, float]:
    """1/N over the provided codes. Reference baseline."""
    if not codes:
        return {}
    w = 1.0 / len(codes)
    return {c: w for c in codes}


def mean_variance_weights(
    codes: list[str],
    sigma: np.ndarray,
    *,
    mu: np.ndarray | None = None,
    risk_aversion: float = 1.0,
) -> dict[str, float]:
    """Classic Markowitz allocation.

    When ``mu`` is provided: ``w ∝ Σ⁻¹ μ`` (unconstrained, scaled by
    ``risk_aversion``), then projected onto the simplex (long-only, sum
    to 1). When ``mu`` is None: returns the global minimum-variance
    portfolio ``w ∝ Σ⁻¹ 1``.

    Long-only constraint enforced via clamp-then-renormalise. Not the
    classical inequality-constrained QP — that needs a solver. For an
    N≤10 protected-book + candidate scenario, the heuristic is fine;
    the agent inspects the weights and can call out concentration.

    NaN entries in ``sigma`` are imputed with the mean off-diagonal
    value BEFORE inversion — the alternative (skipping the asset)
    would silently drop the candidate from the basket.
    """
    n = len(codes)
    if n == 0:
        return {}
    sigma = np.array(sigma, dtype=np.float64, copy=True)
    if np.isnan(sigma).any():
        off_diag = sigma[~np.eye(n, dtype=bool)]
        finite = off_diag[np.isfinite(off_diag)]
        fill = float(finite.mean()) if finite.size else 0.0
        sigma = np.where(np.isnan(sigma), fill, sigma)
    # Ridge for numerical stability when sigma is near-singular.
    sigma = sigma + 1e-8 * np.eye(n)
    try:
        inv_sigma = np.linalg.inv(sigma)
    except np.linalg.LinAlgError:
        return equal_weight(codes)
    if mu is None:
        raw = inv_sigma @ np.ones(n)
    else:
        raw = (1.0 / risk_aversion) * (inv_sigma @ np.asarray(mu, dtype=np.float64))
    # Long-only + simplex projection: clamp negatives, renormalise.
    raw = np.maximum(raw, 0.0)
    s = raw.sum()
    if s == 0:
        return equal_weight(codes)
    weights = raw / s
    return {c: float(weights[i]) for i, c in enumerate(codes)}


def _cov_from_corr(corr: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """Σ = D · C · D where D = diag(vol). Used by HRP."""
    return np.outer(vol, vol) * corr


def _correlation_distance(corr: np.ndarray) -> np.ndarray:
    """LdP's correlation distance: d_ij = sqrt(0.5 · (1 - C_ij)).
    Maps perfectly-correlated → 0, anti-correlated → 1. Required
    metric property for hierarchical clustering."""
    d2 = np.clip(0.5 * (1.0 - corr), 0.0, 1.0)
    return np.sqrt(d2)


def _single_linkage_order(dist: np.ndarray) -> list[int]:
    """Return an ordered index list via single-linkage hierarchical
    clustering, leaves-only. Pure NumPy implementation — covers the
    HRP quasi-diagonalisation step without scipy.cluster.hierarchy.

    Algorithm: iteratively merge the closest two clusters, tracking
    each cluster's member ordering so the final root carries a
    bisectable index sequence. For N≤20 (the realistic protected-book
    size plus candidates), the O(N²·log N) cost is negligible.
    """
    n = dist.shape[0]
    if n == 1:
        return [0]
    # Each "cluster" is a list of original indices. Distance between
    # two clusters is the min pairwise distance (single linkage).
    clusters: list[list[int]] = [[i] for i in range(n)]
    cluster_dist: list[list[float]] = [list(row) for row in dist]
    active = list(range(n))
    while len(active) > 1:
        best = (float("inf"), -1, -1)
        for ai in range(len(active)):
            for aj in range(ai + 1, len(active)):
                i = active[ai]
                j = active[aj]
                d = cluster_dist[i][j]
                if d < best[0]:
                    best = (d, i, j)
        _, i, j = best
        # Merge j into i. Single-linkage update: dist(merged, k) =
        # min(dist(i, k), dist(j, k)).
        merged = clusters[i] + clusters[j]
        for k in active:
            if k == i or k == j:
                continue
            new_d = min(cluster_dist[i][k], cluster_dist[j][k])
            cluster_dist[i][k] = new_d
            cluster_dist[k][i] = new_d
        clusters[i] = merged
        active.remove(j)
    return clusters[active[0]]


def hrp_weights(
    codes: list[str],
    corr: np.ndarray,
    vol: np.ndarray,
) -> dict[str, float]:
    """Hierarchical Risk Parity (López de Prado, Bailey 2016).

    Three steps:
      1. Quasi-diagonalise the covariance via correlation-distance +
         single-linkage clustering — assets that move together cluster
         together.
      2. Recursive bisection: split the ordered index list in half,
         assign inverse-variance weights between halves.
      3. Repeat until each "cluster" is a single asset.

    Robust to ill-conditioned covariance because we never invert the
    full matrix. Closed-form: no solver, no convergence to debug.

    NaN handling: NaN entries in ``corr`` get imputed with 0 (treated
    as "no information"). That's a defensive choice — over-correlation
    fears get priority over fear-of-missing-correlation.
    """
    n = len(codes)
    if n == 0:
        return {}
    if n == 1:
        return {codes[0]: 1.0}
    corr = np.array(corr, dtype=np.float64, copy=True)
    vol = np.array(vol, dtype=np.float64, copy=True)
    # NaN imputation.
    corr = np.where(np.isnan(corr), 0.0, corr)
    np.fill_diagonal(corr, 1.0)
    # Zero vol → epsilon so inverse-variance is defined.
    vol = np.where((vol <= 0) | np.isnan(vol), 1e-8, vol)
    cov = _cov_from_corr(corr, vol)

    dist = _correlation_distance(corr)
    order = _single_linkage_order(dist)

    # Recursive bisection on the ordered indices.
    weights = np.ones(n, dtype=np.float64)

    def _bisect(items: list[int]) -> None:
        if len(items) <= 1:
            return
        mid = len(items) // 2
        left = items[:mid]
        right = items[mid:]
        # Inverse-variance weight per cluster, normalised to 1 within
        # the cluster. Then the cluster's "portfolio variance" is
        # w_c.T · Σ_c · w_c.
        def _cluster_var(idx: list[int]) -> float:
            sub_cov = cov[np.ix_(idx, idx)]
            iv = 1.0 / np.diag(sub_cov)
            iv /= iv.sum()
            return float(iv @ sub_cov @ iv)

        var_l = _cluster_var(left)
        var_r = _cluster_var(right)
        if var_l + var_r == 0:
            alpha = 0.5
        else:
            alpha = 1.0 - var_l / (var_l + var_r)
        weights[left] *= alpha
        weights[right] *= 1.0 - alpha
        _bisect(left)
        _bisect(right)

    _bisect(order)
    s = weights.sum()
    if s == 0:
        return equal_weight(codes)
    weights = weights / s
    return {codes[i]: float(weights[i]) for i in range(n)}


def realised_vol(series: dict[date, float], min_obs: int = 5) -> float | None:
    """Sample standard deviation of a daily-return series. Returns None
    if too few observations or zero variance."""
    if len(series) < min_obs:
        return None
    vals = np.fromiter(series.values(), dtype=np.float64)
    v = float(vals.std(ddof=1))
    return v if v > 0 else None


def summarise_weights(weights: dict[str, float]) -> dict[str, Any]:
    """Diagnostic summary the agent reads alongside the raw weights —
    catches concentration without needing to re-derive HHI."""
    if not weights:
        return {"n_assets": 0}
    vals = list(weights.values())
    return {
        "n_assets": len(weights),
        "max_weight": round(float(max(vals)), 4),
        "min_weight": round(float(min(vals)), 4),
        "concentration_hhi": round(float(sum(v * v for v in vals)), 4),
        "n_effective": round(1.0 / max(sum(v * v for v in vals), 1e-12), 2),
    }
