"""
Walk-forward forecast engine for SIP backtesting.

Two forecast methods (all strictly walk-forward — no lookahead bias):

1. **Time-of-Day Seasonal Mean (TOD):** For each settlement period, compute
   the rolling mean of that same half-hour over the lookback window.  Captures
   the strong diurnal pattern in SIP.

2. **Exponential Weighted Moving Average (EWMA):** Apply exponential decay
   weighting over the lookback window, giving more weight to recent data.
   Uses the same time-of-day structure.

3. **Market benchmark (MIP):** The Market Index Price at the forecast origin,
   treated as the market's "forward" view (passed through as-is).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Horizon definitions (in settlement periods)
DEFAULT_HORIZONS: List[int] = [1, 2, 6, 12, 48, 336]

HORIZON_LABELS: Dict[int, str] = {
    1: "30 min",
    2: "1 hour",
    6: "3 hours",
    12: "6 hours",
    48: "1 day",
    336: "1 week",
}

# Lookback definitions (in settlement periods)
DEFAULT_LOOKBACKS: List[int] = [48, 336, 672, 1440, 4320]

LOOKBACK_LABELS: Dict[int, str] = {
    48: "1 day",
    336: "7 days",
    672: "14 days",
    1440: "30 days",
    4320: "90 days",
}


@dataclass
class ForecastResult:
    """A single forecast origin with multi-horizon predictions."""
    origin_idx: int
    origin_datetime: pd.Timestamp
    horizon_sps: List[int]
    forecasts: Dict[int, float]
    realised: Dict[int, float]
    market_fwd: Dict[int, float]


def _tod_mean_forecast(
    sip_values: np.ndarray,
    origin_idx: int,
    lookback_sps: int,
    horizons: List[int],
) -> Dict[int, float]:
    """
    Time-of-Day seasonal mean: for each horizon h, find the SP index of
    (origin + h), then average all values at that same half-hour-of-day
    in the lookback window.
    """
    n = len(sip_values)
    forecasts: Dict[int, float] = {}
    for h in horizons:
        target_idx = origin_idx + h
        if target_idx >= n:
            continue
        target_sp_of_day = target_idx % 48
        start = max(0, origin_idx - lookback_sps)
        window = sip_values[start:origin_idx]
        if len(window) == 0:
            continue
        sp_indices = np.arange(start, origin_idx)
        same_sp_mask = (sp_indices % 48) == target_sp_of_day
        if same_sp_mask.any():
            forecasts[h] = float(np.mean(window[same_sp_mask]))
        else:
            forecasts[h] = float(np.mean(window))
    return forecasts


def _ewma_forecast(
    sip_values: np.ndarray,
    origin_idx: int,
    lookback_sps: int,
    horizons: List[int],
    alpha: float = 0.05,
) -> Dict[int, float]:
    """
    EWMA with time-of-day structure: for each horizon, collect same-SP-of-day
    values in the lookback window, then apply exponential weighting
    (most recent gets highest weight).
    """
    n = len(sip_values)
    forecasts: Dict[int, float] = {}
    for h in horizons:
        target_idx = origin_idx + h
        if target_idx >= n:
            continue
        target_sp_of_day = target_idx % 48
        start = max(0, origin_idx - lookback_sps)
        window = sip_values[start:origin_idx]
        if len(window) == 0:
            continue
        sp_indices = np.arange(start, origin_idx)
        same_sp_mask = (sp_indices % 48) == target_sp_of_day
        if same_sp_mask.any():
            vals = window[same_sp_mask]
        else:
            vals = window

        if len(vals) == 0:
            continue

        weights = np.array([(1 - alpha) ** i for i in range(len(vals) - 1, -1, -1)])
        weights /= weights.sum()
        forecasts[h] = float(np.dot(weights, vals))
    return forecasts


def _extract_realised(
    sip_values: np.ndarray,
    origin_idx: int,
    horizons: List[int],
) -> Dict[int, float]:
    n = len(sip_values)
    return {
        h: float(sip_values[origin_idx + h])
        for h in horizons
        if origin_idx + h < n
    }


def _extract_market_forward(
    mip_values: np.ndarray,
    origin_idx: int,
    horizons: List[int],
    lookback_sps: int = 336,
) -> Dict[int, float]:
    """
    Market benchmark: TOD-mean of MIP over the lookback window.

    Rather than using the stale MIP spot-at-origin (which would
    systematically lose to any decent forecast at longer horizons),
    this computes the same-half-hour rolling mean of MIP — i.e. the
    market's best unconditional view at each delivery SP.  This is
    the fair benchmark: if our model can't beat the market's own
    seasonal average, we have no alpha.
    """
    n = len(mip_values)
    forecasts: Dict[int, float] = {}
    for h in horizons:
        target_idx = origin_idx + h
        if target_idx >= n:
            continue
        target_sp_of_day = target_idx % 48
        start = max(0, origin_idx - lookback_sps)
        window = mip_values[start:origin_idx]
        if len(window) == 0:
            forecasts[h] = float(mip_values[min(origin_idx, n - 1)])
            continue
        sp_indices = np.arange(start, origin_idx)
        same_sp_mask = (sp_indices % 48) == target_sp_of_day
        if same_sp_mask.any():
            forecasts[h] = float(np.mean(window[same_sp_mask]))
        else:
            forecasts[h] = float(np.mean(window))
    return forecasts


def run_walk_forward_backtest(
    sip_series: pd.Series,
    mip_series: pd.Series,
    lookback_sps: int,
    horizons: Optional[List[int]] = None,
    method: str = "tod_mean",
    min_history: int = 96,
    step: int = 1,
    ewma_alpha: float = 0.05,
    demand_series: Optional[pd.Series] = None,
) -> List[ForecastResult]:
    """
    Walk-forward backtest: at each origin point, produce forecasts for
    all horizons and record realised values and market forward.

    Parameters
    ----------
    sip_series : Half-hourly SIP indexed by datetime
    mip_series : Half-hourly MIP indexed by datetime (aligned to sip_series)
    lookback_sps : Lookback window in settlement periods
    horizons : Forecast horizons in settlement periods
    method : "tod_mean", "ewma", or "xgb"
    min_history : Minimum data points before first forecast
    step : Step size between origins (1 = every SP, 48 = daily)
    ewma_alpha : EWMA smoothing parameter (higher = more weight on recent obs)
    demand_series : Optional half-hourly demand (used as feature by XGBoost)
    """
    if horizons is None:
        horizons = DEFAULT_HORIZONS

    sip_values = sip_series.values.astype(float)
    mip_values = mip_series.values.astype(float)
    demand_values = demand_series.values.astype(float) if demand_series is not None else None
    n = len(sip_values)

    max_horizon = max(horizons)
    start_idx = max(min_history, lookback_sps)
    end_idx = n - max_horizon

    if start_idx >= end_idx:
        logger.warning(
            "Insufficient data for backtest: n=%d, start=%d, end=%d",
            n, start_idx, end_idx,
        )
        return []

    results: List[ForecastResult] = []

    for idx in range(start_idx, end_idx, step):
        if method == "xgb":
            from src.models.xgb_forecaster import _xgb_forecast

            fc = _xgb_forecast(
                sip_values, idx, lookback_sps, horizons,
                mip_values=mip_values, demand_values=demand_values,
            )
        elif method == "ewma":
            fc = _ewma_forecast(sip_values, idx, lookback_sps, horizons, alpha=ewma_alpha)
        else:
            fc = _tod_mean_forecast(sip_values, idx, lookback_sps, horizons)

        realised = _extract_realised(sip_values, idx, horizons)
        market_fwd = _extract_market_forward(mip_values, idx, horizons,
                                             lookback_sps=lookback_sps)

        valid_horizons = [h for h in horizons if h in fc and h in realised and h in market_fwd]
        if not valid_horizons:
            continue

        results.append(ForecastResult(
            origin_idx=idx,
            origin_datetime=sip_series.index[idx],
            horizon_sps=valid_horizons,
            forecasts={h: fc[h] for h in valid_horizons},
            realised={h: realised[h] for h in valid_horizons},
            market_fwd={h: market_fwd[h] for h in valid_horizons},
        ))

    logger.info(
        "Walk-forward backtest: method=%s, lookback=%d SPs, %d origins, %d horizons",
        method, lookback_sps, len(results), len(horizons),
    )
    return results


def build_aligned_series(
    sip_df: pd.DataFrame,
    mip_df: pd.DataFrame,
    demand_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.Series, pd.Series, Optional[pd.Series]]:
    """
    Align SIP, MIP (and optionally Demand) DataFrames into half-hourly
    Series with a common datetime index.

    Returns (sip_series, mip_series, demand_series).
    demand_series is None when demand_df is not provided or empty.
    """
    sip_col = "systemBuyPrice" if "systemBuyPrice" in sip_df.columns else "systemSellPrice"

    sip = sip_df.copy()
    sip["datetime"] = pd.to_datetime(sip["settlementDate"]) + pd.to_timedelta(
        (sip["settlementPeriod"].astype(int) - 1) * 30, unit="min"
    )
    sip = sip.set_index("datetime")[sip_col].sort_index()
    sip = sip[~sip.index.duplicated(keep="first")]

    mip = mip_df.copy()
    mip["datetime"] = pd.to_datetime(mip["settlementDate"]) + pd.to_timedelta(
        (mip["settlementPeriod"].astype(int) - 1) * 30, unit="min"
    )
    mip_col = "price"
    mip = mip.set_index("datetime")[mip_col].sort_index()
    mip = mip[~mip.index.duplicated(keep="first")]

    common_idx = sip.index.intersection(mip.index)

    demand_series: Optional[pd.Series] = None
    if demand_df is not None and not demand_df.empty:
        dem = demand_df.copy()
        dem["datetime"] = pd.to_datetime(dem["settlementDate"]) + pd.to_timedelta(
            (dem["settlementPeriod"].astype(int) - 1) * 30, unit="min"
        )
        dem_col = "initialDemandOutturn" if "initialDemandOutturn" in dem.columns else dem.columns[-1]
        dem = dem.set_index("datetime")[dem_col].sort_index()
        dem = dem[~dem.index.duplicated(keep="first")]
        common_idx = common_idx.intersection(dem.index)
        if len(common_idx) > 0:
            demand_series = dem.loc[common_idx]

    if len(common_idx) == 0:
        logger.warning("No overlapping timestamps between SIP, MIP and Demand")
        return sip, mip, None

    return sip.loc[common_idx], mip.loc[common_idx], demand_series
