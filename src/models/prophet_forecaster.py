"""
NeuralProphet-based forecasters for SIP, MIP, and Demand.

NeuralProphet combines AR-Net (autoregression) with Facebook Prophet's
decomposition of trend + seasonality. It handles sub-daily data natively
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

_FREQ = "30min"
_DEFAULT_EPOCHS = 15


def _values_to_df(
    values: np.ndarray,
    end_idx: int,
    lookback: int,
    end_date=None,
) -> pd.DataFrame:
    """
    Convert a numpy array slice into the ds/y DataFrame NeuralProphet expects.

    Parameters
    ----------
    end_date : The actual timestamp of values[end_idx - 1]. When provided, the
               date index uses real settlement dates so NeuralProphet's weekday
               seasonality is correctly aligned. Defaults to a synthetic anchor
               (2025-01-01) when unknown, which may mis-align weekly patterns.
    """
    start = max(0, end_idx - lookback)
    y  = values[start:end_idx].astype(float)
    anchor = pd.Timestamp(end_date) if end_date is not None else pd.Timestamp("2025-01-01")
    ds = pd.date_range(end=anchor, periods=len(y), freq=_FREQ)
    return pd.DataFrame({"ds": ds, "y": y})


def _neuralprophet_forecast(
    target_values: np.ndarray,
    origin_idx: int,
    lookback_sps: int,
    horizons: List[int],
    np_params: Optional[Dict] = None,
) -> Dict[int, float]:
    """
    Fit a NeuralProphet model on the lookback window and predict at the
    requested horizons. Falls back to TOD-mean if NeuralProphet is not
    installed or fitting fails.

    Parameters
    ----------
    np_params : Optional dict of NeuralProphet constructor kwargs to override
                the defaults (e.g. tuned params from grid search).
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

    # Base config; np_params overrides any of these
    base_config = dict(
        n_forecasts=max_h,
        n_lags=min(48 * 7, len(df) // 2),
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=True,
        epochs=_DEFAULT_EPOCHS,
        learning_rate=0.1,
        batch_size=min(64, len(df)),
    )
    if np_params:
        base_config.update(np_params)
        # n_forecasts must always match max_h (not overridable)
        base_config["n_forecasts"] = max_h
        # batch_size must not exceed dataset length
        base_config["batch_size"] = min(base_config.get("batch_size", 64), len(df))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = NeuralProphet(**base_config)
            m.fit(df, freq=_FREQ)
            future = m.make_future_dataframe(df, periods=max_h, n_historic_predictions=0)
            pred = m.predict(future)

        yhat_cols = sorted(
            [c for c in pred.columns if c.startswith("yhat")],
            key=lambda c: int(c.replace("yhat", "") or "1"),
        )
        if not yhat_cols:
            raise ValueError("NeuralProphet returned no yhat columns")

        for h in horizons:
            if origin_idx + h >= n:
                continue
            col_idx = h - 1
            if col_idx < len(yhat_cols):
                val = pred[yhat_cols[col_idx]].dropna()
                if not val.empty:
                    forecasts[h] = float(val.iloc[-1])

    except Exception as exc:
        logger.warning("NeuralProphet fit failed: %s — falling back to TOD mean", exc)
        from src.models.forecaster import _tod_mean_forecast
        return _tod_mean_forecast(target_values, origin_idx, lookback_sps, horizons)

    return forecasts
