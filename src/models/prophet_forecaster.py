"""
NeuralProphet-based forecasters for SIP and Demand.

NeuralProphet combines AR-Net (autoregression) with Facebook Prophet's
decomposition of trend + seasonality.  It handles sub-daily data natively
and captures both the half-hourly and weekly patterns in electricity prices.

The interface mirrors the naive forecasters in forecaster.py / xgb_forecaster.py
so it can be dispatched from the rolling backtest engine.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from neuralprophet import NeuralProphet, set_log_level

    _HAS_PROPHET = True
    set_log_level("ERROR")
except ImportError:
    _HAS_PROPHET = False
    NeuralProphet = None  # type: ignore[assignment,misc]

_FIT_EPOCHS = 15
_FREQ = "30min"


def _values_to_df(values: np.ndarray, end_idx: int, lookback: int) -> pd.DataFrame:
    """Convert a numpy array slice into the ds/y DataFrame NeuralProphet needs."""
    start = max(0, end_idx - lookback)
    y = values[start:end_idx].astype(float)
    ds = pd.date_range(end="2025-01-01", periods=len(y), freq=_FREQ)
    return pd.DataFrame({"ds": ds, "y": y})


def _neuralprophet_forecast(
    target_values: np.ndarray,
    origin_idx: int,
    lookback_sps: int,
    horizons: List[int],
) -> Dict[int, float]:
    """
    Fit a NeuralProphet model on the lookback window and predict at
    the requested horizons.  Falls back to TOD-mean if NeuralProphet
    is not installed or fitting fails.
    """
    if not _HAS_PROPHET:
        logger.warning("neuralprophet not installed — falling back to TOD mean")
        from src.models.forecaster import _tod_mean_forecast
        return _tod_mean_forecast(target_values, origin_idx, lookback_sps, horizons)

    max_h = max(horizons)
    n = len(target_values)
    forecasts: Dict[int, float] = {}

    df = _values_to_df(target_values, origin_idx, lookback_sps)
    if len(df) < 48:
        from src.models.forecaster import _tod_mean_forecast
        return _tod_mean_forecast(target_values, origin_idx, lookback_sps, horizons)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = NeuralProphet(
                n_forecasts=max_h,
                n_lags=min(48 * 7, len(df) // 2),
                yearly_seasonality=False,
                weekly_seasonality=True,
                daily_seasonality=True,
                epochs=_FIT_EPOCHS,
                learning_rate=0.1,
                batch_size=min(64, len(df)),
            )
            m.fit(df, freq=_FREQ)

            future = m.make_future_dataframe(df, periods=max_h, n_historic_predictions=0)
            pred = m.predict(future)

        yhat_cols = [c for c in pred.columns if c.startswith("yhat")]
        if not yhat_cols:
            raise ValueError("NeuralProphet returned no yhat columns")

        yhat_cols_sorted = sorted(yhat_cols, key=lambda c: int(c.replace("yhat", "") or "1"))

        for h in horizons:
            if origin_idx + h >= n:
                continue
            col_idx = h - 1
            if col_idx < len(yhat_cols_sorted):
                col = yhat_cols_sorted[col_idx]
                val = pred[col].dropna()
                if not val.empty:
                    forecasts[h] = float(val.iloc[-1])

    except Exception as exc:
        logger.warning("NeuralProphet fit failed: %s — falling back to TOD mean", exc)
        from src.models.forecaster import _tod_mean_forecast
        return _tod_mean_forecast(target_values, origin_idx, lookback_sps, horizons)

    return forecasts
