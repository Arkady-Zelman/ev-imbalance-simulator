"""
Risk and performance metrics for the imbalance exposure model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy.stats import kurtosis, skew

from src.config import DEFAULT_DA_PRICE, NUM_SETTLEMENT_PERIODS


@dataclass
class RiskSummary:
    mean_pnl: float
    median_pnl: float
    std_pnl: float
    skew_pnl: float
    kurtosis_pnl: float
    var_95: float
    cvar_95: float
    capture_ratio_mean: float
    reward_to_risk: float         # mean(P&L) / std(P&L) — within-day signal-to-noise, not annualised Sharpe
    max_loss: float
    max_gain: float


def compute_var(pnl: np.ndarray, confidence: float = 0.95) -> float:
    """Value-at-Risk: the (1-confidence) percentile of P&L."""
    return float(np.percentile(pnl, (1 - confidence) * 100))


def compute_cvar(pnl: np.ndarray, confidence: float = 0.95) -> float:
    """Conditional VaR (Expected Shortfall): mean of losses beyond VaR."""
    var = compute_var(pnl, confidence)
    tail = pnl[pnl <= var]
    return float(tail.mean()) if len(tail) > 0 else var


def compute_capture_ratios(
    delivered_mw: np.ndarray,
    traded_mw: np.ndarray,
    da_price: float | np.ndarray = DEFAULT_DA_PRICE,
    sip_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """
    Capture ratio per simulation run.

    Benchmark = revenue if you had traded exactly the delivered volume at DA.
    Actual    = DA revenue on traded volume +/- imbalance cost at SIP.
    """
    if isinstance(da_price, (int, float)):
        da_vec = np.full(NUM_SETTLEMENT_PERIODS, float(da_price))
    else:
        da_vec = np.asarray(da_price)

    benchmark = np.sum(delivered_mw * 0.5 * da_vec[np.newaxis, :], axis=1)
    actual_da = np.sum(traded_mw[np.newaxis, :] * 0.5 * da_vec[np.newaxis, :], axis=1)

    imbalance = traded_mw[np.newaxis, :] - delivered_mw
    if sip_matrix is not None:
        imb_cost = np.sum(imbalance * 0.5 * sip_matrix, axis=1)
    else:
        imb_cost = np.sum(imbalance * 0.5 * da_vec[np.newaxis, :], axis=1)

    actual = actual_da - imb_cost
    safe_benchmark = np.where(np.abs(benchmark) < 1e-6, 1e-6, benchmark)
    return actual / safe_benchmark


def compute_risk_summary(pnl: np.ndarray, capture_ratios: np.ndarray) -> RiskSummary:
    """Aggregate all risk metrics into a single summary object."""
    return RiskSummary(
        mean_pnl=float(np.mean(pnl)),
        median_pnl=float(np.median(pnl)),
        std_pnl=float(np.std(pnl)),
        skew_pnl=float(skew(pnl)),
        kurtosis_pnl=float(kurtosis(pnl)),
        var_95=compute_var(pnl),
        cvar_95=compute_cvar(pnl),
        capture_ratio_mean=float(np.mean(capture_ratios)),
        reward_to_risk=float(np.mean(pnl) / max(np.std(pnl), 1e-9)),
        max_loss=float(np.min(pnl)),
        max_gain=float(np.max(pnl)),
    )


def sensitivity_sweep(
    run_fn,
    param_name: str,
    param_values: List[float],
    base_params: dict,
) -> List[Dict]:
    """
    Run simulations across a range of one parameter, collecting risk metrics.

    run_fn should accept **base_params and return (pnl_array, capture_ratios).
    """
    results = []
    for val in param_values:
        kwargs = {**base_params, param_name: val}
        pnl, cr = run_fn(**kwargs)
        summary = compute_risk_summary(pnl, cr)
        results.append({
            "param_value": val,
            "mean_pnl": summary.mean_pnl,
            "var_95": summary.var_95,
            "cvar_95": summary.cvar_95,
            "capture_ratio": summary.capture_ratio_mean,
            "sharpe": summary.reward_to_risk,
        })
    return results
