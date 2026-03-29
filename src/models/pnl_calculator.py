"""
Shared P&L calculation for a given position against delivered MW and SIP.

Consolidates the duplicated imbalance-cost logic that was previously
copy-pasted across tab_monte_carlo, tab_risk_analysis, and
tab_scenario_comparison.
"""

from __future__ import annotations

import numpy as np

from src.config import NUM_SETTLEMENT_PERIODS


def compute_pnl_for_position(
    delivered_mw: np.ndarray,
    traded_mw: np.ndarray,
    sip_matrix: np.ndarray,
    da_price: float,
    n_runs: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Compute daily P&L array for a single traded position.

    Parameters
    ----------
    delivered_mw : (n_runs, 48)
    traded_mw    : (48,)
    sip_matrix   : (n_days, 48) or (48,)  — bootstrap pool or single day
    da_price     : scalar DA price assumption (£/MWh)
    n_runs       : number of simulation runs to price
    seed         : RNG seed for SIP day sampling

    Returns
    -------
    pnl : (n_runs,) daily P&L in £
    """
    rng = np.random.default_rng(seed)
    imbalance = traded_mw[np.newaxis, :] - delivered_mw

    if sip_matrix.ndim == 2 and sip_matrix.shape[0] > 1:
        idx = rng.integers(0, sip_matrix.shape[0], size=n_runs)
        sip_runs = sip_matrix[idx, :]
    else:
        flat = sip_matrix if sip_matrix.ndim == 1 else sip_matrix[0]
        sip_runs = np.broadcast_to(flat, (n_runs, NUM_SETTLEMENT_PERIODS))

    imbalance_cost = np.sum(imbalance * 0.5 * sip_runs, axis=1)
    revenue = float(np.sum(traded_mw * 0.5 * da_price))
    return np.full(n_runs, revenue) - imbalance_cost
