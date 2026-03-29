"""
SIP (System Imbalance Price) generative models.

Provides regime-switching and custom-scenario SIP matrix generation,
extracted from inline logic that was previously in app.py and
tab_scenario_comparison.py.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import NUM_SETTLEMENT_PERIODS, SIP_REGIME_DEFAULTS, SIPRegimeParams

logger = logging.getLogger(__name__)


def derive_da_price_from_mip(mip_df: pd.DataFrame) -> float | None:
    """
    Compute mean MIP from fetched data.  Returns None if data is
    insufficient, letting the caller fall back to the config default.
    """
    if mip_df is None or mip_df.empty:
        return None
    col = "price" if "price" in mip_df.columns else None
    if col is None:
        return None
    vals = mip_df[col].dropna()
    if len(vals) < 10:
        return None
    return round(float(vals.mean()), 2)


def generate_regime_switching_sip(
    n_days: int,
    n_sp: int = NUM_SETTLEMENT_PERIODS,
    params: SIPRegimeParams = SIP_REGIME_DEFAULTS,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate a synthetic (n_days, n_sp) SIP matrix using a two-regime model.

    Normal regime  : Normal(mean, std)
    Spike regime   : LogNormal(mean_log, std_log)
    Transition     : independent Bernoulli per SP with P(spike)
    """
    rng = np.random.default_rng(seed)
    is_spike = rng.random((n_days, n_sp)) < params.spike_probability
    normal_prices = rng.normal(params.normal_mean, params.normal_std, (n_days, n_sp))
    spike_prices = rng.lognormal(params.spike_mean_log, params.spike_std_log, (n_days, n_sp))
    return np.where(is_spike, spike_prices, normal_prices)


def fit_regime_params(sip_df: pd.DataFrame, spike_percentile: float = 95.0) -> SIPRegimeParams:
    """
    Fit regime-switching parameters from historical ELEXON SIP data.

    Splits observations into normal (below spike_percentile) and spike regimes,
    then fits Normal and LogNormal distributions to each.  Falls back to
    SIP_REGIME_DEFAULTS if the data is too sparse.
    """
    col = "systemBuyPrice" if "systemBuyPrice" in sip_df.columns else "systemSellPrice"
    if col not in sip_df.columns or sip_df.empty:
        logger.warning("Cannot fit regime params — no SIP price column; using defaults")
        return SIP_REGIME_DEFAULTS

    prices = sip_df[col].dropna().values.astype(float)
    if len(prices) < 50:
        logger.warning("Only %d SIP observations — too few to fit; using defaults", len(prices))
        return SIP_REGIME_DEFAULTS

    threshold = float(np.percentile(prices, spike_percentile))
    normal_prices = prices[prices <= threshold]
    spike_prices = prices[prices > threshold]

    normal_mean = float(np.mean(normal_prices)) if len(normal_prices) > 0 else SIP_REGIME_DEFAULTS.normal_mean
    normal_std = float(np.std(normal_prices, ddof=1)) if len(normal_prices) > 1 else SIP_REGIME_DEFAULTS.normal_std
    normal_std = max(normal_std, 1.0)

    spike_probability = len(spike_prices) / len(prices) if len(prices) > 0 else SIP_REGIME_DEFAULTS.spike_probability
    spike_probability = max(spike_probability, 0.001)

    if len(spike_prices) >= 3:
        pos_spikes = spike_prices[spike_prices > 0]
        if len(pos_spikes) >= 3:
            log_spikes = np.log(pos_spikes)
            spike_mean_log = float(np.mean(log_spikes))
            spike_std_log = max(float(np.std(log_spikes, ddof=1)), 0.1)
        else:
            spike_mean_log = SIP_REGIME_DEFAULTS.spike_mean_log
            spike_std_log = SIP_REGIME_DEFAULTS.spike_std_log
    else:
        spike_mean_log = SIP_REGIME_DEFAULTS.spike_mean_log
        spike_std_log = SIP_REGIME_DEFAULTS.spike_std_log

    fitted = SIPRegimeParams(
        normal_mean=round(normal_mean, 2),
        normal_std=round(normal_std, 2),
        spike_mean_log=round(spike_mean_log, 3),
        spike_std_log=round(spike_std_log, 3),
        spike_probability=round(spike_probability, 4),
    )
    logger.info("Fitted regime params from %d observations: %s", len(prices), fitted)
    return fitted


def generate_custom_scenario_sip(
    n_days: int,
    normal_mean: float = 60.0,
    normal_std: float = 30.0,
    spike_prob: float = 0.05,
    spike_mean: float = 500.0,
    spike_std: float = 300.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate a custom-scenario SIP matrix.

    Uses lognormal for spikes (always positive, heavy-tailed)
    rather than abs(normal) for consistency.
    """
    rng = np.random.default_rng(seed)
    n_sp = NUM_SETTLEMENT_PERIODS
    is_spike = rng.random((n_days, n_sp)) < spike_prob
    normal_prices = rng.normal(normal_mean, max(normal_std, 1.0), (n_days, n_sp))
    mu_log = np.log(spike_mean**2 / np.sqrt(spike_std**2 + spike_mean**2))
    sigma_log = np.sqrt(np.log(1 + spike_std**2 / spike_mean**2))
    spike_prices = rng.lognormal(mu_log, sigma_log, (n_days, n_sp))
    return np.where(is_spike, spike_prices, normal_prices)
