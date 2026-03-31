"""
XGBoost-based forecasters for SIP and Demand.

For each forecast origin and horizon, we train a fresh XGBoost model on the
lookback window (online/rolling fit — no lookahead).  Features include
time-of-day/day-of-week encodings, recent lags, rolling statistics, and
cross-series interactions.

The interface matches the naive forecasters in forecaster.py so they can
be dispatched interchangeably from the rolling backtest engine.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb

    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    xgb = None  # type: ignore[assignment]


def _cyclical_encode(value: float, period: float) -> tuple[float, float]:
    angle = 2 * np.pi * value / period
    return float(np.sin(angle)), float(np.cos(angle))


def _build_features(
    target_values: np.ndarray,
    idx: int,
    horizon: int,
    mip_values: Optional[np.ndarray] = None,
    aux_values: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    Build a feature vector for the observation at *idx* predicting *horizon*
    SPs ahead.  Only uses data at indices < idx (strict walk-forward).

    Returns None if insufficient history.
    """
    sp_of_day = idx % 48
    day_of_week = (idx // 48) % 7

    sp_sin, sp_cos = _cyclical_encode(sp_of_day, 48)
    dow_sin, dow_cos = _cyclical_encode(day_of_week, 7)
    h_sin, h_cos = _cyclical_encode(horizon % 48, 48)

    feats: list[float] = [sp_sin, sp_cos, dow_sin, dow_cos, h_sin, h_cos]

    target_sp = (idx + horizon) % 48
    tsp_sin, tsp_cos = _cyclical_encode(target_sp, 48)
    feats.extend([tsp_sin, tsp_cos])

    lag_offsets = [48, 96, 336]  # 1d, 2d, 7d
    for off in lag_offsets:
        if idx - off >= 0:
            feats.append(target_values[idx - off])
        else:
            feats.append(np.nan)

    same_sp_offsets = [48, 96, 336]
    for off in same_sp_offsets:
        look = idx - off
        if look >= 0:
            feats.append(target_values[look])
        else:
            feats.append(np.nan)

    window_48 = target_values[max(0, idx - 48):idx]
    if len(window_48) > 0:
        feats.extend([
            float(np.nanmean(window_48)),
            float(np.nanstd(window_48)),
            float(np.nanmax(window_48)),
            float(np.nanmin(window_48)),
        ])
    else:
        feats.extend([np.nan, np.nan, np.nan, np.nan])

    window_336 = target_values[max(0, idx - 336):idx]
    if len(window_336) > 0:
        feats.extend([float(np.nanmean(window_336)), float(np.nanstd(window_336))])
    else:
        feats.extend([np.nan, np.nan])

    if mip_values is not None:
        if idx > 0:
            feats.append(float(mip_values[idx - 1]))
        else:
            feats.append(np.nan)
        mip_win = mip_values[max(0, idx - 48):idx]
        if len(mip_win) > 0:
            feats.append(float(np.nanmean(mip_win)))
        else:
            feats.append(np.nan)
    else:
        feats.extend([np.nan, np.nan])

    if aux_values is not None:
        for off in [48, 96]:
            if idx - off >= 0:
                feats.append(aux_values[idx - off])
            else:
                feats.append(np.nan)
        aux_win = aux_values[max(0, idx - 48):idx]
        if len(aux_win) > 0:
            feats.append(float(np.nanmean(aux_win)))
        else:
            feats.append(np.nan)
        if idx - 48 >= 0 and aux_values is not None:
            feats.append(target_values[max(0, idx - 48)] * aux_values[max(0, idx - 48)])
        else:
            feats.append(np.nan)
    else:
        feats.extend([np.nan, np.nan, np.nan, np.nan])

    return np.array(feats, dtype=np.float32)


_DEFAULT_XGB_PARAMS: Dict[str, object] = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0.1,
    "min_child_weight": 5,
    "max_delta_step": 1,
    "scale_pos_weight": 5,
    "base_score": 0.5,
}


def _xgb_forecast(
    sip_values: np.ndarray,
    origin_idx: int,
    lookback_sps: int,
    horizons: List[int],
    mip_values: Optional[np.ndarray] = None,
    demand_values: Optional[np.ndarray] = None,
    xgb_params: Optional[Dict[str, object]] = None,
) -> Dict[int, float]:
    """
    XGBoost SIP forecast: trains one model per horizon on the lookback window,
    then predicts the value at origin + horizon.

    Parameters
    ----------
    xgb_params : Optional dict of XGBRegressor kwargs. When provided, these
                 override the hardcoded defaults. Pass tuned params from grid search.
    """
    if not _HAS_XGB:
        logger.warning("xgboost not installed — falling back to TOD mean")
        from src.models.forecaster import _tod_mean_forecast

        return _tod_mean_forecast(sip_values, origin_idx, lookback_sps, horizons)

    params = dict(_DEFAULT_XGB_PARAMS)
    if xgb_params is not None:
        params.update(xgb_params)

    n = len(sip_values)
    forecasts: Dict[int, float] = {}
    _FEATURE_HISTORY = 48 + 336  # minimum indices needed for feature builder
    _MIN_SAMPLES = 30

    for h in horizons:
        if origin_idx + h >= n:
            continue

        min_window = _FEATURE_HISTORY + h + _MIN_SAMPLES
        effective_lb = max(lookback_sps, min_window)
        start = max(0, origin_idx - effective_lb)

        X_rows: list[np.ndarray] = []
        y_rows: list[float] = []

        train_start = max(start, _FEATURE_HISTORY)
        for i in range(train_start, origin_idx):
            if i + h >= origin_idx:
                break
            feat = _build_features(sip_values, i, h, mip_values, demand_values)
            if feat is None:
                continue
            X_rows.append(feat)
            y_rows.append(float(sip_values[i + h]))

        if len(X_rows) < 20:
            continue

        X = np.vstack(X_rows)
        y = np.array(y_rows, dtype=np.float32)

        nan_mask = np.isnan(X)
        col_means = np.nanmean(X, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        for c in range(X.shape[1]):
            X[nan_mask[:, c], c] = col_means[c]

        model = xgb.XGBRegressor(
            **params,
            random_state=42,
            verbosity=0,
            n_jobs=1,
        )
        model.fit(X, y)

        x_pred = _build_features(sip_values, origin_idx, h, mip_values, demand_values)
        if x_pred is None:
            continue
        x_pred = np.where(np.isnan(x_pred), col_means, x_pred)
        forecasts[h] = float(model.predict(x_pred.reshape(1, -1))[0])

    return forecasts


def _xgb_demand_forecast(
    demand_values: np.ndarray,
    origin_idx: int,
    lookback_sps: int,
    horizons: List[int],
    sip_values: Optional[np.ndarray] = None,
    xgb_params: Optional[Dict[str, object]] = None,
) -> Dict[int, float]:
    """
    XGBoost demand forecast: trains one model per horizon on the lookback
    window using demand as target and SIP as auxiliary feature.
    """
    if not _HAS_XGB:
        logger.warning("xgboost not installed — falling back to TOD mean for demand")
        from src.models.forecaster import _tod_mean_forecast

        return _tod_mean_forecast(demand_values, origin_idx, lookback_sps, horizons)

    return _xgb_forecast(
        sip_values=demand_values,
        origin_idx=origin_idx,
        lookback_sps=lookback_sps,
        horizons=horizons,
        mip_values=None,
        demand_values=sip_values,
        xgb_params=xgb_params,
    )


def _xgb_mip_forecast(
    mip_values: np.ndarray,
    origin_idx: int,
    lookback_sps: int,
    horizons: List[int],
    sip_values: Optional[np.ndarray] = None,
    demand_values: Optional[np.ndarray] = None,
    xgb_params: Optional[Dict[str, object]] = None,
) -> Dict[int, float]:
    """
    XGBoost wholesale (MIP) forecast: trains one model per horizon on the
    lookback window using MIP as target and SIP + demand as auxiliary features.
    """
    if not _HAS_XGB:
        logger.warning("xgboost not installed — falling back to TOD mean for MIP")
        from src.models.forecaster import _tod_mean_forecast

        return _tod_mean_forecast(mip_values, origin_idx, lookback_sps, horizons)

    return _xgb_forecast(
        sip_values=mip_values,
        origin_idx=origin_idx,
        lookback_sps=lookback_sps,
        horizons=horizons,
        mip_values=None,
        demand_values=sip_values if sip_values is not None else demand_values,
        xgb_params=xgb_params,
    )
