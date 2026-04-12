"""
Rolling forecast backtest engine for market inefficiency detection.

Uses 1-day, 3-day, 5-day, and 15-day lookbacks to forecast 1 through 14 days
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

from typing import Optional

logger = logging.getLogger(__name__)

# Day-level lookbacks (in settlement periods)
ROLLING_LOOKBACKS = {
    "1 day":   48,
    "3 days":  48 * 3,
    "5 days":  48 * 5,
    "15 days": 48 * 15,
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
    forecast_mape: float      # mean absolute percentage error (%)
    market_mae: float
    market_rmse: float
    market_mape: float        # mean absolute percentage error (%)
    alpha_mae: float          # market_mae - forecast_mae
    alpha_mape: float         # market_mape - forecast_mape (pp)
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
    target: str = "sip",
    demand_series: Optional[pd.Series] = None,
    xgb_params: Optional[Dict] = None,
    da_series: Optional[pd.Series] = None,
    gen_series: Optional[pd.Series] = None,
) -> Tuple[List[RollingErrorRow], List[CrossoverResult]]:
    """
    Run the full rolling backtest across all lookback × horizon combos.

    Steps daily (step=48) to keep origins non-overlapping for valid
    statistical inference.

    Parameters
    ----------
    target : "sip", "demand", "mip", or "total_generation".
    demand_series : Half-hourly demand (feature for SIP/MIP; target for demand).
    gen_series    : Half-hourly total generation (MW). Required for target="total_generation".
    xgb_params    : Optional XGBRegressor kwargs for method="xgb".

    Returns
    -------
    errors : list of RollingErrorRow for every (lookback, horizon) pair
    crossovers : one CrossoverResult per lookback, indicating where alpha dies
    """
    sip_values    = sip_series.values.astype(float)
    mip_values    = mip_series.values.astype(float)
    demand_values = demand_series.values.astype(float) if demand_series  is not None else None
    da_values     = da_series.values.astype(float)    if da_series      is not None else None
    # Reindex gen onto the SIP index so integer offsets align correctly
    gen_values = (
        gen_series.reindex(sip_series.index).ffill().bfill().values.astype(float)
        if gen_series is not None else None
    )

    if target == "demand":
        if demand_values is None:
            logger.warning("Demand target requested but no demand data provided.")
            return [], []
        target_values    = demand_values
        benchmark_values = demand_values
    elif target == "mip":
        target_values    = mip_values
        benchmark_values = mip_values
    elif target == "total_generation":
        if gen_values is None:
            logger.warning("total_generation target requested but no gen data provided.")
            return [], []
        target_values    = gen_values
        benchmark_values = None   # persistence benchmark computed per-SP below
    else:
        target_values    = sip_values
        benchmark_values = mip_values

    n = len(target_values)
    step = 48

    errors: List[RollingErrorRow] = []

    for lb_label, lb_sps in ROLLING_LOOKBACKS.items():
        for h_sps in ROLLING_HORIZONS:
            h_days = h_sps // 48
            start_idx = max(96, lb_sps)
            end_idx = n - h_sps

            if start_idx >= end_idx:
                continue

            fc_errors_list: list[float] = []
            mkt_errors_list: list[float] = []
            realised_list: list[float] = []

            for idx in range(start_idx, end_idx, step):
                if method == "xgb":
                    from src.models.xgb_forecaster import (
                        _xgb_demand_forecast,
                        _xgb_forecast,
                        _xgb_mip_forecast,
                    )

                    if target == "demand":
                        fc = _xgb_demand_forecast(
                            demand_values, idx, lb_sps, [h_sps],  # type: ignore[arg-type]
                            sip_values=sip_values,
                            xgb_params=xgb_params,
                        )
                    elif target == "mip":
                        fc = _xgb_mip_forecast(
                            mip_values, idx, lb_sps, [h_sps],
                            sip_values=sip_values,
                            demand_values=demand_values,
                            xgb_params=xgb_params,
                        )
                    elif target == "total_generation":
                        # Generation: target series is gen; SIP as price feature
                        tgt_v = gen_values
                        fc = _xgb_forecast(
                            tgt_v, idx, lb_sps, [h_sps],  # type: ignore[arg-type]
                            mip_values=sip_values, demand_values=demand_values,
                            xgb_params=xgb_params,
                        )
                    else:
                        fc = _xgb_forecast(
                            sip_values, idx, lb_sps, [h_sps],
                            mip_values=mip_values, demand_values=demand_values,
                            xgb_params=xgb_params,
                        )
                elif method == "ewma":
                    fc = _ewma_forecast(target_values, idx, lb_sps, [h_sps], alpha=ewma_alpha)
                else:
                    fc = _tod_mean_forecast(target_values, idx, lb_sps, [h_sps])

                realised = _extract_realised(target_values, idx, [h_sps])

                # For total_generation: use 1-day-ago same-SP persistence as the naive benchmark
                # (no external forward curve exists for generation).
                if target == "total_generation":
                    persist_idx = idx + h_sps - 48
                    if persist_idx >= 0:
                        market_fwd = {h_sps: float(target_values[persist_idx])}
                    else:
                        market_fwd = {}
                else:
                    market_fwd = _extract_market_forward(benchmark_values, idx, [h_sps],
                                                         lookback_sps=lb_sps,
                                                         da_values=da_values)

                if h_sps not in fc or h_sps not in realised or h_sps not in market_fwd:
                    continue

                r = realised[h_sps]
                fc_err = abs(fc[h_sps] - r)
                mkt_err = abs(market_fwd[h_sps] - r)
                fc_errors_list.append(fc_err)
                mkt_errors_list.append(mkt_err)
                realised_list.append(r)

            if len(fc_errors_list) < 5:
                continue

            fc_arr = np.array(fc_errors_list)
            mkt_arr = np.array(mkt_errors_list)
            real_arr = np.array(realised_list)

            forecast_mae = float(np.mean(fc_arr))
            forecast_rmse = float(np.sqrt(np.mean(fc_arr ** 2)))
            market_mae = float(np.mean(mkt_arr))
            market_rmse = float(np.sqrt(np.mean(mkt_arr ** 2)))

            # MAPE: cap individual values at 500% to avoid blowup from near-zero/negative SIP.
            # SIP can be negative (reverse cash-out), so MAPE is treated as indicative only;
            # MAE is the primary metric for this application.
            safe_real = np.where(np.abs(real_arr) < 1e-6, 1e-6, real_arr)
            fc_pct  = np.minimum(fc_arr  / np.abs(safe_real) * 100, 500.0)
            mkt_pct = np.minimum(mkt_arr / np.abs(safe_real) * 100, 500.0)
            forecast_mape = float(np.mean(fc_pct))
            market_mape   = float(np.mean(mkt_pct))

            _, dm_p = diebold_mariano(fc_arr, mkt_arr, h=max(1, h_days), power=2)

            errors.append(RollingErrorRow(
                lookback_label=lb_label,
                lookback_sps=lb_sps,
                horizon_days=h_days,
                horizon_sps=h_sps,
                forecast_mae=forecast_mae,
                forecast_rmse=forecast_rmse,
                forecast_mape=forecast_mape,
                market_mae=market_mae,
                market_rmse=market_rmse,
                market_mape=market_mape,
                alpha_mae=market_mae - forecast_mae,
                alpha_mape=market_mape - forecast_mape,
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
