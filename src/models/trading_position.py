"""
Trading position sizing utilities.

Given Monte Carlo delivered-MW arrays, compute the traded MW curve
at various risk-appetite percentiles.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from src.config import RISK_APPETITES


def compute_traded_positions(
    delivered_mw: np.ndarray,
    percentiles: Dict[str, int] | None = None,
) -> Dict[str, np.ndarray]:
    """
    Return {label: traded_mw_array(48,)} for each risk-appetite tier.
    """
    if percentiles is None:
        percentiles = RISK_APPETITES

    return {
        label: np.percentile(delivered_mw, pct, axis=0)
        for label, pct in percentiles.items()
    }


def position_imbalance(
    delivered_mw: np.ndarray,
    traded_mw: np.ndarray,
) -> np.ndarray:
    """
    Imbalance per run per SP.  Positive = short (under-delivered).
    Shape: (n_runs, 48).
    """
    return traded_mw[np.newaxis, :] - delivered_mw
