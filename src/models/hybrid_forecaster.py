"""
Hybrid LSTM + XGBoost ensemble forecaster.

Combines predictions from both models using inverse-MAE weighted averaging,
matching the methodology in manuhup/LSTM-XGBoost-Hybrid-Forecasting.

    weight_lstm = (1/lstm_val_mae) / (1/lstm_val_mae + 1/xgb_val_mae)
    weight_xgb  = 1 - weight_lstm
    combined    = weight_lstm * lstm_pred + weight_xgb * xgb_pred

If only one model is available, that model's predictions are returned
directly (effective weight = 1.0).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class HybridIntraday:
    """Result of combining two 48-SP intraday forecasts."""
    combined:    Dict[str, np.ndarray]   # {lookback: np.ndarray(48,)}
    lstm_fc:     Dict[str, np.ndarray]
    xgb_fc:      Dict[str, np.ndarray]
    lstm_weight: float                   # global weight applied to LSTM
    xgb_weight:  float                   # global weight applied to XGBoost


@dataclass
class HybridForward:
    """Result of combining two 14-day forward forecasts."""
    combined:    Dict[str, Dict[int, float]]
    lstm_fc:     Dict[str, Dict[int, float]]
    xgb_fc:      Dict[str, Dict[int, float]]
    lstm_weight: float
    xgb_weight:  float


# ── Weight computation ────────────────────────────────────────────────────────

def _compute_weights(
    xgb_val_mae: Optional[float],
    lstm_val_mae: Optional[float],
) -> tuple[float, float]:
    """
    Return (w_xgb, w_lstm) normalised to sum to 1.

    Rules:
    - Both available: inverse-MAE weighting.
    - Only XGBoost:  (1.0, 0.0)
    - Only LSTM:     (0.0, 1.0)
    - Both missing:  (0.5, 0.5)
    """
    valid_xgb  = xgb_val_mae  is not None and np.isfinite(xgb_val_mae)  and xgb_val_mae  > 0
    valid_lstm = lstm_val_mae is not None and np.isfinite(lstm_val_mae) and lstm_val_mae > 0

    if valid_xgb and valid_lstm:
        w_xgb  = 1.0 / xgb_val_mae
        w_lstm = 1.0 / lstm_val_mae
        total  = w_xgb + w_lstm
        return w_xgb / total, w_lstm / total

    if valid_xgb and not valid_lstm:
        return 1.0, 0.0
    if valid_lstm and not valid_xgb:
        return 0.0, 1.0
    return 0.5, 0.5


def _best_val_mae(trained) -> Optional[float]:
    """
    Extract the best (lowest) validation MAE/score across all cells from a
    TrainedXGBModels or TrainedLSTMModels instance.  Returns None if unavailable.
    """
    scores = getattr(trained, "best_scores", {})
    all_scores = [
        s for lb_dict in scores.values()
        for s in lb_dict.values()
        if s is not None and np.isfinite(s) and s > 0
    ]
    return float(np.min(all_scores)) if all_scores else None


# ── Ensemble functions ────────────────────────────────────────────────────────

def combine_intraday(
    xgb_fc: Dict[str, np.ndarray],
    lstm_fc: Dict[str, np.ndarray],
    xgb_trained=None,
    lstm_trained=None,
    xgb_val_mae: Optional[float] = None,
    lstm_val_mae: Optional[float] = None,
) -> HybridIntraday:
    """
    Weighted average of two 48-SP intraday forecast dicts.

    Parameters
    ----------
    xgb_fc / lstm_fc   : {lookback_label: np.ndarray(48,)} from each model.
    xgb_trained / lstm_trained : Optional trained model objects; used to
                          auto-extract val_mae if xgb_val_mae / lstm_val_mae
                          are not provided directly.
    xgb_val_mae / lstm_val_mae : Override validation MAEs.
    """
    if xgb_val_mae is None and xgb_trained is not None:
        xgb_val_mae = _best_val_mae(xgb_trained)
    if lstm_val_mae is None and lstm_trained is not None:
        lstm_val_mae = _best_val_mae(lstm_trained)

    w_xgb, w_lstm = _compute_weights(xgb_val_mae, lstm_val_mae)

    all_lbs = sorted(set(list(xgb_fc.keys()) + list(lstm_fc.keys())))
    combined: Dict[str, np.ndarray] = {}
    for lb in all_lbs:
        xgb_arr  = xgb_fc.get(lb)
        lstm_arr = lstm_fc.get(lb)
        if xgb_arr is not None and lstm_arr is not None:
            combined[lb] = w_xgb * xgb_arr + w_lstm * lstm_arr
        elif xgb_arr is not None:
            combined[lb] = xgb_arr.copy()
        elif lstm_arr is not None:
            combined[lb] = lstm_arr.copy()

    return HybridIntraday(
        combined=combined,
        lstm_fc=lstm_fc,
        xgb_fc=xgb_fc,
        lstm_weight=float(w_lstm),
        xgb_weight=float(w_xgb),
    )


def combine_forward(
    xgb_fc: Dict[str, Dict[int, float]],
    lstm_fc: Dict[str, Dict[int, float]],
    xgb_trained=None,
    lstm_trained=None,
    xgb_val_mae: Optional[float] = None,
    lstm_val_mae: Optional[float] = None,
) -> HybridForward:
    """
    Weighted average of two 14-day forward forecast dicts.
    {lookback_label: {day: value}}.
    """
    if xgb_val_mae is None and xgb_trained is not None:
        xgb_val_mae = _best_val_mae(xgb_trained)
    if lstm_val_mae is None and lstm_trained is not None:
        lstm_val_mae = _best_val_mae(lstm_trained)

    w_xgb, w_lstm = _compute_weights(xgb_val_mae, lstm_val_mae)

    all_lbs = sorted(set(list(xgb_fc.keys()) + list(lstm_fc.keys())))
    combined: Dict[str, Dict[int, float]] = {}
    for lb in all_lbs:
        xgb_days  = xgb_fc.get(lb,  {})
        lstm_days = lstm_fc.get(lb, {})
        all_days  = sorted(set(list(xgb_days.keys()) + list(lstm_days.keys())))
        day_dict: Dict[int, float] = {}
        for day in all_days:
            xv  = xgb_days.get(day)
            lv  = lstm_days.get(day)
            if xv is not None and lv is not None:
                day_dict[day] = w_xgb * xv + w_lstm * lv
            elif xv is not None:
                day_dict[day] = xv
            elif lv is not None:
                day_dict[day] = lv
        combined[lb] = day_dict

    return HybridForward(
        combined=combined,
        lstm_fc=lstm_fc,
        xgb_fc=xgb_fc,
        lstm_weight=float(w_lstm),
        xgb_weight=float(w_xgb),
    )
