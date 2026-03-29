"""
Rolling forecast backtest engine for market inefficiency detection.

Uses 1-day, 15-day, and 30-day lookbacks to forecast 1 through 14 days
ahead, then compares error against the forward curve (MIP benchmark)
at each horizon.  The crossover point — where our forecast error
exceeds the market's — defines the maximum exploitable forecast horizon.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.models.forecaster import (
    _ewma_forecast,
    _extract_market_forward,
    _extract_realised,
    _tod_mean_forecast,
)
from src.models.stat_tests import diebold_mariano

logger = logging.getLogger(__name__)

# Day-level lookbacks (in settlement periods)
ROLLING_LOOKBACKS = {
    "1 day": 48,
    "15 days": 48 * 15,
    "30 days": 48 * 30,
}

# Day-level horizons: 1 to 14 days ahead (in settlement periods)
ROLLING_HORIZONS = [48 * d for d in range(1, 15)]
ROLLING_HORIZON_LABELS = {48 * d: f"{d}d" for d in range(1, 15)}


@dataclass
class RollingErrorRow:
    """Error metrics for one (lookback, horizon) configuration."""
    lookback_label: str
    lookback_sps: int
    horizon_days: int
    horizon_sps: int
    forecast_mae: float
    forecast_rmse: float
    market_mae: float
    market_rmse: float
    alpha_mae: float          # market_mae - forecast_mae
    dm_pvalue: float
    n_obs: int


@dataclass
class CrossoverResult:
    """The horizon at which our forecast stops beating the market."""
    lookback_label: str
    crossover_day: int        # 0 = we never beat; 15 = we always beat
    last_positive_alpha: float
    first_negative_alpha: float


def run_rolling_backtest(
    sip_series: pd.Series,
    mip_series: pd.Series,
    method: str = "tod_mean",
    ewma_alpha: float = 0.05,
) -> Tuple[List[RollingErrorRow], List[CrossoverResult]]:
    """
    Run the full rolling backtest across all lookback × horizon combos.

    Steps daily (step=48) to keep origins non-overlapping for valid
    statistical inference.

    Returns
    -------
    errors : list of RollingErrorRow for every (lookback, horizon) pair
    crossovers : one CrossoverResult per lookback, indicating where alpha dies
    """
    sip_values = sip_series.values.astype(float)
    mip_values = mip_series.values.astype(float)
    n = len(sip_values)

    forecast_fn = _ewma_forecast if method == "ewma" else _tod_mean_forecast
    step = 48  # daily origins for non-overlapping statistical validity

    errors: List[RollingErrorRow] = []

    for lb_label, lb_sps in ROLLING_LOOKBACKS.items():
        for h_sps in ROLLING_HORIZONS:
            h_days = h_sps // 48
            start_idx = max(96, lb_sps)
            end_idx = n - h_sps

            if start_idx >= end_idx:
                continue

            fc_errors_list = []
            mkt_errors_list = []

            for idx in range(start_idx, end_idx, step):
                if method == "ewma":
                    fc = forecast_fn(sip_values, idx, lb_sps, [h_sps], alpha=ewma_alpha)
                else:
                    fc = forecast_fn(sip_values, idx, lb_sps, [h_sps])

                realised = _extract_realised(sip_values, idx, [h_sps])
                market_fwd = _extract_market_forward(mip_values, idx, [h_sps],
                                                     lookback_sps=lb_sps)

                if h_sps not in fc or h_sps not in realised or h_sps not in market_fwd:
                    continue

                fc_err = abs(fc[h_sps] - realised[h_sps])
                mkt_err = abs(market_fwd[h_sps] - realised[h_sps])
                fc_errors_list.append(fc_err)
                mkt_errors_list.append(mkt_err)

            if len(fc_errors_list) < 5:
                continue

            fc_arr = np.array(fc_errors_list)
            mkt_arr = np.array(mkt_errors_list)

            forecast_mae = float(np.mean(fc_arr))
            forecast_rmse = float(np.sqrt(np.mean(fc_arr ** 2)))
            market_mae = float(np.mean(mkt_arr))
            market_rmse = float(np.sqrt(np.mean(mkt_arr ** 2)))

            _, dm_p = diebold_mariano(fc_arr, mkt_arr, h=max(1, h_days), power=2)

            errors.append(RollingErrorRow(
                lookback_label=lb_label,
                lookback_sps=lb_sps,
                horizon_days=h_days,
                horizon_sps=h_sps,
                forecast_mae=forecast_mae,
                forecast_rmse=forecast_rmse,
                market_mae=market_mae,
                market_rmse=market_rmse,
                alpha_mae=market_mae - forecast_mae,
                dm_pvalue=dm_p,
                n_obs=len(fc_errors_list),
            ))

    # Compute crossover for each lookback
    crossovers: List[CrossoverResult] = []
    for lb_label in ROLLING_LOOKBACKS:
        lb_rows = [e for e in errors if e.lookback_label == lb_label]
        lb_rows.sort(key=lambda e: e.horizon_days)

        crossover_day = 15  # default = we always beat
        last_pos = 0.0
        first_neg = 0.0

        for i, row in enumerate(lb_rows):
            if row.alpha_mae < 0:
                crossover_day = row.horizon_days
                first_neg = row.alpha_mae
                if i > 0:
                    last_pos = lb_rows[i - 1].alpha_mae
                break
            last_pos = row.alpha_mae

        if all(r.alpha_mae >= 0 for r in lb_rows) and lb_rows:
            crossover_day = 15

        crossovers.append(CrossoverResult(
            lookback_label=lb_label,
            crossover_day=crossover_day,
            last_positive_alpha=last_pos,
            first_negative_alpha=first_neg,
        ))

    logger.info(
        "Rolling backtest: %d error rows, %d lookbacks, method=%s",
        len(errors), len(ROLLING_LOOKBACKS), method,
    )
    return errors, crossovers


def build_error_matrix(errors: List[RollingErrorRow]) -> pd.DataFrame:
    """
    Pivot errors into a matrix: rows = lookback, columns = horizon days,
    values = alpha_mae.
    """
    rows = []
    for e in errors:
        rows.append({
            "Lookback": e.lookback_label,
            "Horizon": f"{e.horizon_days}d",
            "Horizon Days": e.horizon_days,
            "Alpha (MAE)": e.alpha_mae,
            "Forecast MAE": e.forecast_mae,
            "Market MAE": e.market_mae,
            "DM p-value": e.dm_pvalue,
            "N": e.n_obs,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    pivot = df.pivot_table(
        index="Lookback",
        columns="Horizon Days",
        values="Alpha (MAE)",
        aggfunc="first",
    )
    pivot.columns = [f"{int(c)}d" for c in pivot.columns]
    return pivot
