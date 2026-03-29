"""
Statistical significance tests for forecast evaluation.

Provides:
- Binomial confidence intervals on hit rates
- Diebold-Mariano test for forecast comparison
- Block bootstrap confidence intervals for information ratio
- Benjamini-Hochberg FDR correction for multiple testing
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy import stats


def binomial_ci(
    hits: int,
    n: int,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Wilson score interval for a binomial proportion.

    Returns (point_estimate, lower, upper).
    More accurate than the normal approximation at extreme proportions
    or small sample sizes.
    """
    if n == 0:
        return 0.0, 0.0, 1.0
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = hits / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return p_hat, max(0.0, centre - margin), min(1.0, centre + margin)


def diebold_mariano(
    errors_1: np.ndarray,
    errors_2: np.ndarray,
    h: int = 1,
    power: int = 2,
) -> Tuple[float, float]:
    """
    Diebold-Mariano test for equal predictive accuracy.

    Tests H0: E[L(e1)] = E[L(e2)] where L is the loss function.

    Parameters
    ----------
    errors_1 : forecast errors from model 1 (ours)
    errors_2 : forecast errors from model 2 (market)
    h : forecast horizon (for Newey-West HAC bandwidth)
    power : 1 = absolute loss, 2 = squared loss

    Returns
    -------
    (dm_statistic, p_value)
    Negative statistic → model 1 is better.
    """
    d = np.abs(errors_2) ** power - np.abs(errors_1) ** power
    n = len(d)
    if n < 5:
        return 0.0, 1.0

    d_mean = np.mean(d)

    # Newey-West HAC variance estimator
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0.0
    bandwidth = max(1, h - 1)
    for k in range(1, bandwidth + 1):
        gamma_k = np.mean((d[k:] - d_mean) * (d[:-k] - d_mean))
        gamma_sum += 2 * gamma_k

    var_d = (gamma_0 + gamma_sum) / n
    if var_d <= 0:
        return 0.0, 1.0

    dm_stat = d_mean / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p_value)


def bootstrap_ci(
    values: np.ndarray,
    statistic_fn=np.mean,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    block_size: int = 10,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Block bootstrap confidence interval for a statistic.

    Uses circular block bootstrap to preserve serial correlation.

    Returns (point_estimate, lower, upper).
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    if n < block_size:
        point = float(statistic_fn(values))
        return point, point, point

    point = float(statistic_fn(values))
    boot_stats = np.empty(n_bootstrap)
    n_blocks = int(np.ceil(n / block_size))

    for b in range(n_bootstrap):
        block_starts = rng.integers(0, n, size=n_blocks)
        indices = np.concatenate([
            np.arange(s, s + block_size) % n for s in block_starts
        ])[:n]
        boot_stats[b] = statistic_fn(values[indices])

    alpha = (1 - confidence) / 2
    lower = float(np.percentile(boot_stats, alpha * 100))
    upper = float(np.percentile(boot_stats, (1 - alpha) * 100))
    return point, lower, upper


def benjamini_hochberg(
    p_values: List[float],
    alpha: float = 0.05,
) -> List[bool]:
    """
    Benjamini-Hochberg FDR correction for multiple testing.

    Returns a list of booleans: True = reject H0 (significant at FDR alpha).
    """
    m = len(p_values)
    if m == 0:
        return []

    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    significant = [False] * m

    max_k = -1
    for k, (orig_idx, p) in enumerate(indexed, 1):
        if p <= k * alpha / m:
            max_k = k

    if max_k > 0:
        for k in range(max_k):
            orig_idx = indexed[k][0]
            significant[orig_idx] = True

    return significant


def effective_sample_size(
    values: np.ndarray,
    max_lag: int = 50,
) -> float:
    """
    Estimate effective sample size accounting for serial correlation.

    Uses the initial positive sequence estimator (Geyer 1992) to
    truncate the autocorrelation sum.
    """
    n = len(values)
    if n < 3:
        return float(n)

    centered = values - np.mean(values)
    var = np.var(centered)
    if var < 1e-12:
        return float(n)

    max_lag = min(max_lag, n // 2)
    rho_sum = 0.0
    for lag in range(1, max_lag + 1):
        rho = np.mean(centered[lag:] * centered[:-lag]) / var
        if rho < 0:
            break
        rho_sum += rho

    tau = 1 + 2 * rho_sum
    return max(1.0, n / tau)
