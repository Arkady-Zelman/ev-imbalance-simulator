"""
Vectorised Monte Carlo simulation engine.

Orchestrates portfolio availability draws, position sizing, and
imbalance cost calculation for a single parameter set.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (
    CHARGER_CAPACITY_KW,
    DEFAULT_DA_PRICE,
    DAYTYPE_MULTIPLIERS,
    NUM_SETTLEMENT_PERIODS,
    PLUGIN_CLUSTERS,
    SEASONAL_MONTHLY,
    SIP_STRESS_DISPATCH_PENALTY,
    SIP_STRESS_PLUGIN_FACTOR,
    build_sp_beta_params,
    build_sp_means,
)
from src.models.portfolio import (
    apply_dispatch_and_override,
    generate_plugin_rates,
)

logger = logging.getLogger(__name__)


@dataclass
class SimulationParams:
    fleet_size: int = 20_000
    dispatch_rate: float = 0.95
    override_rate: float = 0.03
    n_runs: int = 5_000
    risk_percentile: int = 80
    correlation_decay: float = 0.3
    ewma_alpha: float = 0.05
    day_type: str = "weekday"
    month: int = 1
    sip_stress_coupling: bool = True
    seed: Optional[int] = None

    # Plug-in rate overrides: list of (mean, concentration) per cluster.
    # If None, use config defaults (PLUGIN_CLUSTERS).
    plugin_overrides: Optional[List[Tuple[float, float]]] = None

    # DA price noise (lognormal sigma); 0.0 = deterministic revenue
    da_noise_sigma: float = 0.05

    # SIP-stress coupling factors (only used when sip_stress_coupling=True)
    sip_stress_plugin_factor: float = SIP_STRESS_PLUGIN_FACTOR
    sip_stress_dispatch_penalty: float = SIP_STRESS_DISPATCH_PENALTY

    # Day-type multiplier overrides; if None, use config defaults
    daytype_multipliers: Optional[Dict[str, float]] = None
    # Monthly seasonal factor overrides; if None, use config defaults
    seasonal_monthly: Optional[Dict[int, float]] = None


@dataclass
class SimulationResult:
    """Container for all Monte Carlo outputs."""
    params: SimulationParams
    delivered_mw: np.ndarray = field(repr=False)
    plugin_rates: np.ndarray = field(repr=False)
    traded_mw: np.ndarray = field(repr=False)
    imbalance_mw: np.ndarray = field(repr=False)
    sip: np.ndarray = field(repr=False)
    sip_matrix: np.ndarray = field(repr=False)        # (n_runs, 48) — per-run SIP draws
    da_price: np.ndarray = field(repr=False)
    daily_revenue: np.ndarray = field(repr=False)
    daily_imbalance_cost: np.ndarray = field(repr=False)
    daily_pnl: np.ndarray = field(repr=False)
    sp_imbalance_cost: np.ndarray = field(repr=False)


def run_simulation(
    params: SimulationParams,
    sip_by_sp: np.ndarray,
    da_price: float | np.ndarray = DEFAULT_DA_PRICE,
) -> SimulationResult:
    """
    Execute a full Monte Carlo run.

    Parameters
    ----------
    params : SimulationParams
    sip_by_sp : ndarray of shape (48,) or (n_days, 48)
    da_price : float or ndarray(48,)
    """
    t0 = time.perf_counter()
    logger.info(
        "Starting MC simulation: n_runs=%d, fleet=%d, dispatch=%.3f, override=%.3f, risk_pct=%d, seed=%s",
        params.n_runs, params.fleet_size, params.dispatch_rate, params.override_rate,
        params.risk_percentile, params.seed,
    )

    rng = np.random.default_rng(params.seed)

    if params.plugin_overrides is not None:
        alphas = np.zeros(NUM_SETTLEMENT_PERIODS)
        betas = np.zeros(NUM_SETTLEMENT_PERIODS)
        base_means = np.zeros(NUM_SETTLEMENT_PERIODS)
        for i, cluster in enumerate(PLUGIN_CLUSTERS):
            mean, conc = params.plugin_overrides[i]
            start, end = cluster.sp_range
            alphas[start: end + 1] = mean * conc
            betas[start: end + 1] = (1.0 - mean) * conc
            base_means[start: end + 1] = mean
    else:
        alphas, betas = build_sp_beta_params()
        base_means = build_sp_means()

    daytype_map = params.daytype_multipliers or DAYTYPE_MULTIPLIERS
    seasonal_map = params.seasonal_monthly or SEASONAL_MONTHLY
    daytype_mult = daytype_map.get(params.day_type, 1.0)
    seasonal_mult = seasonal_map.get(params.month, 1.0)
    combined_mult = daytype_mult * seasonal_mult
    if abs(combined_mult - 1.0) > 1e-6:
        concentrations = alphas + betas
        adjusted_means = np.clip(base_means * combined_mult, 0.01, 0.99)
        alphas = adjusted_means * concentrations
        betas = (1.0 - adjusted_means) * concentrations

    plugin_rates = generate_plugin_rates(
        params.n_runs, alphas, betas,
        correlated=True, decay=params.correlation_decay, rng=rng,
    )

    # SIP draw: bootstrap from historical days
    if sip_by_sp.ndim == 2:
        n_days_avail = sip_by_sp.shape[0]
        day_indices = rng.integers(0, n_days_avail, size=params.n_runs)
        sip_matrix = sip_by_sp[day_indices, :]
    else:
        sip_matrix = np.broadcast_to(sip_by_sp, (params.n_runs, NUM_SETTLEMENT_PERIODS)).copy()

    # SIP-availability stress coupling: for runs where SIP is in the top
    # quintile (system stress), degrade plug-in rates and dispatch success.
    # This creates the adverse correlation the real market exhibits.
    dispatch_rate_per_run = np.full(params.n_runs, params.dispatch_rate)
    if params.sip_stress_coupling and sip_matrix.shape[0] > 1:
        mean_sip_per_run = sip_matrix.mean(axis=1)
        sip_p80 = np.percentile(mean_sip_per_run, 80)
        stressed = mean_sip_per_run >= sip_p80
        plugin_rates[stressed] *= params.sip_stress_plugin_factor
        plugin_rates = np.clip(plugin_rates, 0.0, 1.0)
        dispatch_rate_per_run[stressed] *= params.sip_stress_dispatch_penalty

    # Apply dispatch and override with per-run dispatch rates
    delivered_mw = apply_dispatch_and_override(
        plugin_rates, params.fleet_size,
        params.dispatch_rate, params.override_rate, rng=rng,
        dispatch_rate_per_run=dispatch_rate_per_run,
    )

    traded_mw = np.percentile(delivered_mw, params.risk_percentile, axis=0)

    imbalance_mw = traded_mw[np.newaxis, :] - delivered_mw

    sp_imbalance_cost = imbalance_mw * 0.5 * sip_matrix

    if isinstance(da_price, (int, float)):
        da_vec = np.full(NUM_SETTLEMENT_PERIODS, float(da_price))
    else:
        da_vec = np.asarray(da_price)

    da_sigma_log = params.da_noise_sigma
    da_noise = rng.lognormal(
        mean=-0.5 * da_sigma_log**2,
        sigma=max(da_sigma_log, 1e-12),
        size=params.n_runs,
    )
    da_matrix = da_vec[np.newaxis, :] * da_noise[:, np.newaxis]  # (n_runs, 48)

    daily_revenue = np.sum(traded_mw[np.newaxis, :] * 0.5 * da_matrix, axis=1)

    daily_imbalance_cost = sp_imbalance_cost.sum(axis=1)
    daily_pnl = daily_revenue - daily_imbalance_cost

    sip_representative = (
        sip_by_sp.mean(axis=0) if sip_by_sp.ndim == 2 else sip_by_sp
    )

    elapsed = time.perf_counter() - t0
    logger.info("MC simulation completed in %.2fs — mean P&L £%.0f, std £%.0f",
                elapsed, float(np.mean(daily_pnl)), float(np.std(daily_pnl)))

    return SimulationResult(
        params=params,
        delivered_mw=delivered_mw,
        plugin_rates=plugin_rates,
        traded_mw=traded_mw,
        imbalance_mw=imbalance_mw,
        sip=sip_representative,
        sip_matrix=sip_matrix,
        da_price=da_vec,
        daily_revenue=daily_revenue,
        daily_imbalance_cost=daily_imbalance_cost,
        daily_pnl=daily_pnl,
        sp_imbalance_cost=sp_imbalance_cost,
    )


def prepare_sip_matrix(sip_df: pd.DataFrame) -> tuple[np.ndarray, bool]:
    """
    Convert a DataFrame of ELEXON SIP records into an (n_days, 48) matrix.

    Returns (matrix, is_fallback).  If is_fallback is True, the matrix is
    a dummy flat £{DEFAULT_DA_PRICE}/MWh with zero volatility — the caller
    MUST warn the user.
    """
    if sip_df.empty:
        logger.warning("SIP DataFrame is empty — falling back to flat £%.0f/MWh dummy data", DEFAULT_DA_PRICE)
        return np.full((1, NUM_SETTLEMENT_PERIODS), DEFAULT_DA_PRICE), True

    col = "systemBuyPrice" if "systemBuyPrice" in sip_df.columns else "systemSellPrice"
    pivot = sip_df.pivot_table(
        index="settlementDate",
        columns="settlementPeriod",
        values=col,
        aggfunc="first",
    )
    pivot = pivot.dropna(axis=0, thresh=46)
    pivot = pivot.reindex(columns=range(1, 49)).ffill(axis=1).bfill(axis=1)
    if pivot.empty:
        logger.warning("SIP pivot is empty after cleaning — falling back to flat £%.0f/MWh dummy data", DEFAULT_DA_PRICE)
        return np.full((1, NUM_SETTLEMENT_PERIODS), DEFAULT_DA_PRICE), True
    return pivot.values.astype(float), False
