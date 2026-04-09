"""
XGBoost-based forecasters for SIP, MIP, and Demand.

For each forecast origin and horizon, a fresh XGBoost model is trained on the
lookback window (online/rolling fit — no lookahead). Features include
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cyclical_encode(value: float, period: float) -> tuple[float, float]:
    angle = 2 * np.pi * value / period
    return float(np.sin(angle)), float(np.cos(angle))


def _fill_nans(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Impute NaN values with per-column means. Returns (X_filled, col_means).
    Columns that are entirely NaN are filled with 0.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # silence "mean of empty slice"
        col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    X = np.where(np.isnan(X), col_means[np.newaxis, :], X)
    return X, col_means


def _build_features(
    target_values: np.ndarray,
    idx: int,
    horizon: int,
    mip_values: Optional[np.ndarray] = None,
    aux_values: Optional[np.ndarray] = None,
    exog_dict: Optional[Dict[str, np.ndarray]] = None,
) -> Optional[np.ndarray]:
    """
    Build a feature vector for the observation at *idx* predicting *horizon*
    SPs ahead. Only uses data at indices < idx (strict walk-forward, no lookahead).

    Features
    --------
    - Cyclical encoding of current SP-of-day, day-of-week, horizon, target SP-of-day
    - Lags at 1d / 2d / 7d (48 / 96 / 336 SPs back)
    - Rolling mean/std/max/min over last 48 SPs (24h window)
    - Rolling mean/std over last 336 SPs (7d window)
    - MIP: last value + 24h rolling mean (if provided)
    - Auxiliary series (demand for SIP; SIP for demand/MIP): lags at 1d/2d + 24h mean + interaction term

    Returns None is not applicable (idx < 0 etc.), but always returns a vector
    with NaNs for missing entries so the caller can impute.
    """
    sp_of_day  = idx % 48
    day_of_week = (idx // 48) % 7

    sp_sin, sp_cos    = _cyclical_encode(sp_of_day, 48)
    dow_sin, dow_cos  = _cyclical_encode(day_of_week, 7)
    h_sin, h_cos      = _cyclical_encode(horizon % 48, 48)
    tsp_sin, tsp_cos  = _cyclical_encode((idx + horizon) % 48, 48)

    feats: list[float] = [sp_sin, sp_cos, dow_sin, dow_cos, h_sin, h_cos, tsp_sin, tsp_cos]

    # Target lags: 1d / 2d / 7d / 14d / 28d
    lag_1d = target_values[idx - 48]  if idx - 48  >= 0 else np.nan
    lag_2d = target_values[idx - 96]  if idx - 96  >= 0 else np.nan
    lag_7d = target_values[idx - 336] if idx - 336 >= 0 else np.nan
    lag_14d = target_values[idx - 672]  if idx - 672  >= 0 else np.nan
    lag_28d = target_values[idx - 1344] if idx - 1344 >= 0 else np.nan
    feats += [lag_1d, lag_2d, lag_7d, lag_14d, lag_28d]

    # Rolling stats over last 24h
    w48 = target_values[max(0, idx - 48):idx]
    if len(w48) > 0:
        feats += [float(np.nanmean(w48)), float(np.nanstd(w48)),
                  float(np.nanmax(w48)), float(np.nanmin(w48))]
    else:
        feats += [np.nan, np.nan, np.nan, np.nan]

    # Rolling stats over last 7d
    w336 = target_values[max(0, idx - 336):idx]
    if len(w336) > 0:
        feats += [float(np.nanmean(w336)), float(np.nanstd(w336))]
    else:
        feats += [np.nan, np.nan]

    # Rolling mean over last 14d
    w672 = target_values[max(0, idx - 672):idx]
    feats.append(float(np.nanmean(w672)) if len(w672) > 0 else np.nan)

    # Trend slope: linear regression coefficient over last 48 SPs
    if len(w48) >= 4:
        x_slope = np.arange(len(w48), dtype=np.float32)
        feats.append(float(np.polyfit(x_slope, w48.astype(np.float32), 1)[0]))
    else:
        feats.append(np.nan)

    # is_weekend binary flag
    feats.append(1.0 if day_of_week >= 5 else 0.0)

    # EMA proxy over last 7d (geometrically-weighted average)
    if len(w336) >= 2:
        alpha = 0.02  # decay factor per SP ≈ 1 - exp(-1/(50)) for ~50-SP half-life
        weights = (1 - alpha) ** np.arange(len(w336) - 1, -1, -1, dtype=np.float64)
        ema_val = float(np.average(w336, weights=weights))
    else:
        ema_val = np.nan
    feats.append(ema_val)

    # Lag ratio: lag_1d / lag_7d (relative level — mean-reversion signal)
    if not (np.isnan(lag_1d) or np.isnan(lag_7d) or lag_7d == 0.0):
        feats.append(float(lag_1d / lag_7d))
    else:
        feats.append(np.nan)

    # Price momentum: lag_1d - lag_2d (last 24h directional move)
    if not (np.isnan(lag_1d) or np.isnan(lag_2d)):
        feats.append(float(lag_1d - lag_2d))
    else:
        feats.append(np.nan)

    # MIP features (only meaningful when forecasting SIP)
    if mip_values is not None:
        feats.append(float(mip_values[idx - 1]) if idx > 0 else np.nan)
        mip_w = mip_values[max(0, idx - 48):idx]
        feats.append(float(np.nanmean(mip_w)) if len(mip_w) > 0 else np.nan)
    else:
        feats += [np.nan, np.nan]

    # Auxiliary series features (demand when target=SIP/MIP; SIP when target=demand)
    if aux_values is not None:
        for off in (48, 96):
            feats.append(aux_values[idx - off] if idx - off >= 0 else np.nan)
        aux_w = aux_values[max(0, idx - 48):idx]
        feats.append(float(np.nanmean(aux_w)) if len(aux_w) > 0 else np.nan)
        # Interaction: target × aux (both lagged 1d)
        if idx >= 48:
            feats.append(float(target_values[idx - 48]) * float(aux_values[idx - 48]))
        else:
            feats.append(np.nan)
    else:
        feats += [np.nan, np.nan, np.nan, np.nan]

    # Additional exogenous series — 3 features each: 1d lag, 7d lag, 24h rolling mean
    if exog_dict:
        for _arr in exog_dict.values():
            feats.append(_arr[idx - 48]  if idx >= 48  else np.nan)
            feats.append(_arr[idx - 336] if idx >= 336 else np.nan)
            _w = _arr[max(0, idx - 48):idx]
            feats.append(float(np.nanmean(_w)) if len(_w) > 0 else np.nan)

    return np.array(feats, dtype=np.float32)


# ── Default params (used when no tuned params are available) ──────────────────

_DEFAULT_XGB_PARAMS: Dict[str, object] = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0.1,
    "min_child_weight": 5,
}


# ── Core forecast function ────────────────────────────────────────────────────

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
    Train one XGBoost model per horizon on the lookback window and predict
    the value at origin + horizon. Strict walk-forward: only data before
    origin_idx is used for training and prediction.

    Parameters
    ----------
    sip_values  : Primary target series (may be SIP, MIP, or demand depending on caller).
    mip_values  : MIP series used as a feature when forecasting SIP.
    demand_values : Auxiliary series (demand for SIP/MIP targets; SIP for demand target).
    xgb_params  : Override default XGBRegressor kwargs. Pass tuned params from grid search.
    """
    if not _HAS_XGB:
        logger.warning("xgboost not installed — falling back to TOD mean")
        from src.models.forecaster import _tod_mean_forecast
        return _tod_mean_forecast(sip_values, origin_idx, lookback_sps, horizons)

    params = {**_DEFAULT_XGB_PARAMS, **(xgb_params or {})}

    n = len(sip_values)
    _FEAT_HIST = 48 + 336   # minimum history indices for feature builder
    _MIN_ROWS  = 30

    forecasts: Dict[int, float] = {}

    for h in horizons:
        if origin_idx + h >= n:
            continue

        effective_lb = max(lookback_sps, _FEAT_HIST + h + _MIN_ROWS)
        train_start = max(_FEAT_HIST, origin_idx - effective_lb)

        X_rows, y_rows = [], []
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
        X, col_means = _fill_nans(X)

        model = xgb.XGBRegressor(**params, random_state=42, verbosity=0, n_jobs=1)
        model.fit(X, y)

        x_pred = _build_features(sip_values, origin_idx, h, mip_values, demand_values)
        if x_pred is None:
            continue
        x_pred = np.where(np.isnan(x_pred), col_means, x_pred)
        forecasts[h] = float(model.predict(x_pred.reshape(1, -1))[0])

    return forecasts


# ── Target-specific wrappers ──────────────────────────────────────────────────
# These route the correct series into _xgb_forecast as target vs. auxiliary.
# _xgb_forecast uses its first argument as the forecast target; the naming
# reflects the SIP-centric origin of the function.

def _xgb_demand_forecast(
    demand_values: np.ndarray,
    origin_idx: int,
    lookback_sps: int,
    horizons: List[int],
    sip_values: Optional[np.ndarray] = None,
    xgb_params: Optional[Dict[str, object]] = None,
) -> Dict[int, float]:
    """XGBoost demand forecast: demand is target, SIP is auxiliary feature."""
    if not _HAS_XGB:
        logger.warning("xgboost not installed — falling back to TOD mean for demand")
        from src.models.forecaster import _tod_mean_forecast
        return _tod_mean_forecast(demand_values, origin_idx, lookback_sps, horizons)

    return _xgb_forecast(
        sip_values=demand_values,   # target
        origin_idx=origin_idx,
        lookback_sps=lookback_sps,
        horizons=horizons,
        mip_values=None,
        demand_values=sip_values,   # aux feature
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
    """XGBoost MIP (wholesale) forecast: MIP is target, SIP and/or demand are auxiliary."""
    if not _HAS_XGB:
        logger.warning("xgboost not installed — falling back to TOD mean for MIP")
        from src.models.forecaster import _tod_mean_forecast
        return _tod_mean_forecast(mip_values, origin_idx, lookback_sps, horizons)

    aux = sip_values if sip_values is not None else demand_values
    return _xgb_forecast(
        sip_values=mip_values,      # target
        origin_idx=origin_idx,
        lookback_sps=lookback_sps,
        horizons=horizons,
        mip_values=None,
        demand_values=aux,          # aux feature
        xgb_params=xgb_params,
    )
