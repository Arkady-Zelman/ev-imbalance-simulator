"""
Fleet availability model.

Generates stochastic plug-in rates for each settlement period using
Beta distributions coupled through a Gaussian copula, then applies
binomial dispatch-success and customer-override draws.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from src.config import (
    CHARGER_CAPACITY_KW,
    NUM_SETTLEMENT_PERIODS,
    build_correlation_matrix,
    build_sp_beta_params,
)


def _cholesky(decay: float = 0.3) -> np.ndarray:
    """Pre-compute the lower-triangular Cholesky factor (48×48)."""
    corr = build_correlation_matrix(NUM_SETTLEMENT_PERIODS, decay)
    return np.linalg.cholesky(corr)


# Module-level cache so we only factorise once per decay value
_CHOLESKY_CACHE: dict[float, np.ndarray] = {}


def _get_cholesky(decay: float = 0.3) -> np.ndarray:
    if decay not in _CHOLESKY_CACHE:
        _CHOLESKY_CACHE[decay] = _cholesky(decay)
    return _CHOLESKY_CACHE[decay]


def generate_plugin_rates(
    n_runs: int,
    alphas: np.ndarray | None = None,
    betas: np.ndarray | None = None,
    correlated: bool = True,
    decay: float = 0.3,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Draw plug-in rates: shape (n_runs, 48).

    Uses a Gaussian copula to introduce correlation between adjacent
    settlement periods, then transforms marginals through the inverse
    Beta CDF for each SP.
    """
    if rng is None:
        rng = np.random.default_rng()

    if alphas is None or betas is None:
        alphas, betas = build_sp_beta_params()

    n_sp = len(alphas)

    if correlated:
        L = _get_cholesky(decay)
        z = rng.standard_normal((n_runs, n_sp))  # independent normals
        z_corr = z @ L.T                          # correlate
        u = stats.norm.cdf(z_corr)                # → uniform marginals
    else:
        u = rng.uniform(size=(n_runs, n_sp))

    # Clip to avoid 0/1 boundary issues with ppf
    u = np.clip(u, 1e-6, 1 - 1e-6)

    plugin = np.empty_like(u)
    for sp in range(n_sp):
        plugin[:, sp] = stats.beta.ppf(u[:, sp], alphas[sp], betas[sp])

    return plugin


def apply_dispatch_and_override(
    plugin_rates: np.ndarray,
    fleet_size: int,
    dispatch_rate: float = 0.95,
    override_rate: float = 0.03,
    rng: np.random.Generator | None = None,
    dispatch_rate_per_run: np.ndarray | None = None,
) -> np.ndarray:
    """
    Given plug-in rates (n_runs, 48), return delivered MW (n_runs, 48).

    Steps per run per SP:
      1. plugged_in = fleet_size × plug_in_rate
      2. dispatched  ~ Binomial(plugged_in, dispatch_rate)
      3. overridden  ~ Binomial(dispatched,  override_rate)
      4. responding  = dispatched - overridden
      5. delivered_MW = responding × CHARGER_CAPACITY_KW / 1000

    For large fleet sizes we use Normal approximation to the Binomial
    for performance (valid when n*p > 5 and n*(1-p) > 5).

    If dispatch_rate_per_run is provided (shape n_runs,), it overrides
    the scalar dispatch_rate for each run (used for stress coupling).
    """
    if rng is None:
        rng = np.random.default_rng()

    n_runs, n_sp = plugin_rates.shape
    plugged_in = plugin_rates * fleet_size  # (n_runs, 48) float

    if dispatch_rate_per_run is not None:
        dr = dispatch_rate_per_run[:, np.newaxis]  # (n_runs, 1)
    else:
        dr = dispatch_rate

    # Normal approximation to Binomial for dispatch
    mu_d = plugged_in * dr
    sigma_d = np.sqrt(plugged_in * dr * (1 - dr))
    dispatched = np.maximum(0, rng.normal(mu_d, np.maximum(sigma_d, 1e-9)))
    dispatched = np.minimum(dispatched, plugged_in)

    # Normal approximation for override
    mu_o = dispatched * override_rate
    sigma_o = np.sqrt(dispatched * override_rate * (1 - override_rate))
    overridden = np.maximum(0, rng.normal(mu_o, np.maximum(sigma_o, 1e-9)))
    overridden = np.minimum(overridden, dispatched)

    responding = dispatched - overridden
    delivered_mw = responding * CHARGER_CAPACITY_KW / 1000.0
    return delivered_mw


def theoretical_max_mw(fleet_size: int) -> float:
    """Maximum possible MW if every charger is plugged in and responding."""
    return fleet_size * CHARGER_CAPACITY_KW / 1000.0
