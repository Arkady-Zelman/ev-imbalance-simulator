"""
Alpha detection engine for forecast backtesting.

Compares forecast accuracy against the market forward (MIP) across a grid
of [lookback x horizon] configurations.  Identifies the optimal forecast
horizon where the model consistently generates alpha over the market.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.models.forecaster import (
    ForecastResult,
    HORIZON_LABELS,
    LOOKBACK_LABELS,
)
from src.models.stat_tests import (
    benjamini_hochberg,
    binomial_ci,
    bootstrap_ci,
    diebold_mariano,
    effective_sample_size,
)

logger = logging.getLogger(__name__)


@dataclass
class AlphaResult:
    lookback_label: str
    horizon_label: str
    forecast_mae: float
    forecast_rmse: float
    market_mae: float
    market_rmse: float
    alpha_mae: float            # market_mae - forecast_mae (positive = we beat market)
    alpha_rmse: float
    directional_accuracy: float
    hit_rate: float             # % of times our forecast was closer to realised than MIP
    information_ratio: float    # mean(alpha) / std(alpha)
    n_observations: int
    # Statistical significance fields
    hit_rate_ci_lo: float = 0.0
    hit_rate_ci_hi: float = 1.0
    dm_statistic: float = 0.0   # Diebold-Mariano: negative = we beat market
    dm_pvalue: float = 1.0
    ir_ci_lo: float = 0.0
    ir_ci_hi: float = 0.0
    n_effective: float = 0.0    # effective sample size after autocorrelation


def _compute_alpha_for_config(
    results: List[ForecastResult],
    horizon: int,
) -> AlphaResult | None:
    """Compute alpha metrics for a single (lookback, horizon) pair."""
    fc_errors: List[float] = []
    mkt_errors: List[float] = []
    fc_wins: int = 0
    directional_hits: int = 0
    total: int = 0

    for r in results:
        if horizon not in r.forecasts or horizon not in r.realised or horizon not in r.market_fwd:
            continue
        fc = r.forecasts[horizon]
        mkt = r.market_fwd[horizon]
        real = r.realised[horizon]

        fc_err = abs(fc - real)
        mkt_err = abs(mkt - real)
        fc_errors.append(fc_err)
        mkt_errors.append(mkt_err)

        if fc_err < mkt_err:
            fc_wins += 1

        origin_val = r.realised.get(min(r.horizon_sps), real)
        if (fc - origin_val) * (real - origin_val) > 0:
            directional_hits += 1
        total += 1

    if total < 5:
        return None

    fc_errors_arr = np.array(fc_errors)
    mkt_errors_arr = np.array(mkt_errors)
    alpha_arr = mkt_errors_arr - fc_errors_arr  # positive = we beat market

    forecast_mae = float(np.mean(fc_errors_arr))
    forecast_rmse = float(np.sqrt(np.mean(fc_errors_arr ** 2)))
    market_mae = float(np.mean(mkt_errors_arr))
    market_rmse = float(np.sqrt(np.mean(mkt_errors_arr ** 2)))

    alpha_std = float(np.std(alpha_arr)) if len(alpha_arr) > 1 else 1e-9
    ir = float(np.mean(alpha_arr)) / max(alpha_std, 1e-9)

    # Statistical significance
    _, hr_lo, hr_hi = binomial_ci(fc_wins, total)
    dm_stat, dm_p = diebold_mariano(
        fc_errors_arr, mkt_errors_arr, h=max(1, horizon), power=2,
    )
    _, ir_lo, ir_hi = bootstrap_ci(
        alpha_arr,
        statistic_fn=lambda x: float(np.mean(x)) / max(float(np.std(x)), 1e-9),
        block_size=min(max(horizon, 10), len(alpha_arr) // 4),
    )
    n_eff = effective_sample_size(alpha_arr)

    return AlphaResult(
        lookback_label="",
        horizon_label="",
        forecast_mae=forecast_mae,
        forecast_rmse=forecast_rmse,
        market_mae=market_mae,
        market_rmse=market_rmse,
        alpha_mae=market_mae - forecast_mae,
        alpha_rmse=market_rmse - forecast_rmse,
        directional_accuracy=directional_hits / max(total, 1),
        hit_rate=fc_wins / max(total, 1),
        information_ratio=ir,
        n_observations=total,
        hit_rate_ci_lo=hr_lo,
        hit_rate_ci_hi=hr_hi,
        dm_statistic=dm_stat,
        dm_pvalue=dm_p,
        ir_ci_lo=ir_lo,
        ir_ci_hi=ir_hi,
        n_effective=n_eff,
    )


def compute_alpha_matrix(
    backtest_results: Dict[Tuple[int, str], List[ForecastResult]],
    horizons: List[int],
) -> pd.DataFrame:
    """
    Build the [lookback x horizon] alpha matrix.

    Parameters
    ----------
    backtest_results : dict mapping (lookback_sps, method) -> list of ForecastResult
    horizons : list of horizon SPs

    Returns
    -------
    DataFrame with lookback labels as index, horizon labels as columns,
    values = alpha_mae (positive = we beat the market).
    """
    rows = []
    for (lookback_sps, method), results in sorted(backtest_results.items()):
        lb_label = LOOKBACK_LABELS.get(lookback_sps, f"{lookback_sps} SPs")
        row = {"Lookback": lb_label, "Lookback_SPs": lookback_sps}
        for h in horizons:
            h_label = HORIZON_LABELS.get(h, f"{h} SPs")
            ar = _compute_alpha_for_config(results, h)
            if ar is not None:
                row[h_label] = ar.alpha_mae
            else:
                row[h_label] = np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.set_index("Lookback")
        df = df.drop(columns=["Lookback_SPs"], errors="ignore")
    return df


def compute_full_metrics_table(
    backtest_results: Dict[Tuple[int, str], List[ForecastResult]],
    horizons: List[int],
) -> pd.DataFrame:
    """
    Build the full performance metrics table for trader assessment.
    """
    rows = []
    for (lookback_sps, method), results in sorted(backtest_results.items()):
        lb_label = LOOKBACK_LABELS.get(lookback_sps, f"{lookback_sps} SPs")
        for h in horizons:
            h_label = HORIZON_LABELS.get(h, f"{h} SPs")
            ar = _compute_alpha_for_config(results, h)
            if ar is None:
                continue

            # Max drawdown of alpha: compute running cumulative alpha
            alpha_series = []
            for r in results:
                if h in r.forecasts and h in r.realised and h in r.market_fwd:
                    fc_err = abs(r.forecasts[h] - r.realised[h])
                    mkt_err = abs(r.market_fwd[h] - r.realised[h])
                    alpha_series.append(mkt_err - fc_err)

            cum_alpha = np.cumsum(alpha_series) if alpha_series else np.array([0.0])
            running_max = np.maximum.accumulate(cum_alpha)
            drawdowns = running_max - cum_alpha
            max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

            # Calmar ratio: total accumulated alpha / max drawdown.
            # This avoids the misleading "per-SP * 48 * 365" annualisation.
            total_alpha = float(np.sum(alpha_series)) if alpha_series else 0.0
            calmar = total_alpha / max(max_drawdown, 1e-9)

            # Stability: rolling 30-period hit rate std
            if len(alpha_series) > 30:
                hits = np.array([1.0 if a > 0 else 0.0 for a in alpha_series])
                rolling_hr = pd.Series(hits).rolling(30, min_periods=10).mean()
                stability = float(1.0 - rolling_hr.std())
            else:
                stability = np.nan

            # Walk-forward R-squared
            fc_vals = []
            real_vals = []
            for r in results:
                if h in r.forecasts and h in r.realised:
                    fc_vals.append(r.forecasts[h])
                    real_vals.append(r.realised[h])
            if len(fc_vals) > 2:
                ss_res = np.sum((np.array(fc_vals) - np.array(real_vals)) ** 2)
                ss_tot = np.sum((np.array(real_vals) - np.mean(real_vals)) ** 2)
                r_squared = 1 - ss_res / max(ss_tot, 1e-9)
            else:
                r_squared = np.nan

            rows.append({
                "Lookback": lb_label,
                "Horizon": h_label,
                "Forecast MAE": ar.forecast_mae,
                "Market MAE": ar.market_mae,
                "Alpha (MAE)": ar.alpha_mae,
                "Hit Rate": ar.hit_rate,
                "HR 95% CI": f"[{ar.hit_rate_ci_lo:.1%}, {ar.hit_rate_ci_hi:.1%}]",
                "Directional Acc.": ar.directional_accuracy,
                "Info Ratio": ar.information_ratio,
                "IR 95% CI": f"[{ar.ir_ci_lo:.2f}, {ar.ir_ci_hi:.2f}]",
                "DM p-value": ar.dm_pvalue,
                "Max Drawdown": max_drawdown,
                "Calmar Ratio": calmar,
                "Stability": stability,
                "R²": r_squared,
                "N Obs": ar.n_observations,
                "N Effective": ar.n_effective,
            })

    df = pd.DataFrame(rows)

    # Apply Benjamini-Hochberg FDR correction across all configurations
    if not df.empty and "DM p-value" in df.columns:
        p_vals = df["DM p-value"].tolist()
        significant = benjamini_hochberg(p_vals, alpha=0.05)
        df["Significant (FDR 5%)"] = ["Yes" if s else "No" for s in significant]

    return df


def find_optimal_horizon(
    backtest_results: Dict[Tuple[int, str], List[ForecastResult]],
    horizons: List[int],
) -> Dict[str, Dict]:
    """
    For each lookback, find the horizon where alpha is most consistently
    positive (highest information ratio with positive alpha).
    """
    optimal: Dict[str, Dict] = {}

    for (lookback_sps, method), results in sorted(backtest_results.items()):
        lb_label = LOOKBACK_LABELS.get(lookback_sps, f"{lookback_sps} SPs")
        best_h = None
        best_ir = -np.inf

        for h in horizons:
            ar = _compute_alpha_for_config(results, h)
            if ar is None:
                continue
            if ar.alpha_mae > 0 and ar.information_ratio > best_ir:
                best_ir = ar.information_ratio
                best_h = h

        if best_h is not None:
            ar = _compute_alpha_for_config(results, best_h)
            optimal[lb_label] = {
                "best_horizon": HORIZON_LABELS.get(best_h, f"{best_h} SPs"),
                "alpha_mae": ar.alpha_mae,
                "hit_rate": ar.hit_rate,
                "information_ratio": ar.information_ratio,
                "directional_accuracy": ar.directional_accuracy,
            }
        else:
            optimal[lb_label] = {
                "best_horizon": "None (market wins)",
                "alpha_mae": 0.0,
                "hit_rate": 0.0,
                "information_ratio": 0.0,
                "directional_accuracy": 0.0,
            }

    return optimal


def compute_point_in_time_errors(
    results: List[ForecastResult],
    origin_idx: int,
    horizons: List[int],
) -> pd.DataFrame | None:
    """
    For a selected origin point, compute a table of forecast vs MIP vs realised.
    """
    target = None
    for r in results:
        if r.origin_idx == origin_idx:
            target = r
            break

    if target is None:
        closest = min(results, key=lambda r: abs(r.origin_idx - origin_idx), default=None)
        target = closest

    if target is None:
        return None

    rows = []
    for h in horizons:
        if h not in target.forecasts:
            continue
        fc = target.forecasts.get(h, np.nan)
        mkt = target.market_fwd.get(h, np.nan)
        real = target.realised.get(h, np.nan)
        fc_err = abs(fc - real) if not (np.isnan(fc) or np.isnan(real)) else np.nan
        mkt_err = abs(mkt - real) if not (np.isnan(mkt) or np.isnan(real)) else np.nan
        winner = "Forecast" if (fc_err < mkt_err) else "Market" if (mkt_err < fc_err) else "Tie"

        rows.append({
            "Horizon": HORIZON_LABELS.get(h, f"{h} SPs"),
            "Forecast": fc,
            "Market (MIP)": mkt,
            "Realised": real,
            "Forecast Error": fc_err,
            "Market Error": mkt_err,
            "Winner": winner,
        })

    return pd.DataFrame(rows) if rows else None


def compute_cumulative_alpha_series(
    results: List[ForecastResult],
    horizon: int,
) -> Tuple[List[pd.Timestamp], List[float]]:
    """
    Compute running cumulative alpha (forecast advantage) over time.
    """
    timestamps: List[pd.Timestamp] = []
    cum_alpha: List[float] = []
    running = 0.0

    for r in results:
        if horizon not in r.forecasts or horizon not in r.realised or horizon not in r.market_fwd:
            continue
        fc_err = abs(r.forecasts[horizon] - r.realised[horizon])
        mkt_err = abs(r.market_fwd[horizon] - r.realised[horizon])
        running += (mkt_err - fc_err)
        timestamps.append(r.origin_datetime)
        cum_alpha.append(running)

    return timestamps, cum_alpha


# ══════════════════════════════════════════════════════════════════════════
#  Residual Forecasting Validation
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class ResidualValidation:
    """Results from forecasting the model's own residuals."""
    horizon: int
    horizon_label: str
    n_obs: int

    # Raw residuals: forecast - realised (positive = model over-predicts)
    residual_mean: float
    residual_std: float
    residual_autocorr_1: float        # lag-1 autocorrelation of residuals

    # Residual forecast accuracy
    residual_forecast_mae: float      # MAE of the residual-of-residuals
    residual_predictability: float    # R² of predicted residuals vs actual

    # Bias-corrected forecast (original - predicted_residual)
    corrected_mae: float
    original_mae: float
    correction_improvement: float     # (original_mae - corrected_mae) / original_mae

    # Position confirmation
    confirmation_rate: float          # % of times predicted residual sign confirms position
    confirmed_trade_hit_rate: float   # hit rate when residual confirms
    unconfirmed_trade_hit_rate: float # hit rate when residual contradicts


def _ewma_residual_forecast(
    residuals: np.ndarray,
    idx: int,
    lookback: int,
    alpha: float = 0.1,
) -> float:
    """Forecast the next residual using EWMA over the lookback window."""
    start = max(0, idx - lookback)
    window = residuals[start:idx]
    if len(window) == 0:
        return 0.0
    weights = np.array([(1 - alpha) ** i for i in range(len(window) - 1, -1, -1)])
    weights /= weights.sum()
    return float(np.dot(weights, window))


def run_residual_validation(
    results: List[ForecastResult],
    horizons: List[int],
    residual_lookback: int = 48,
) -> List[ResidualValidation]:
    """
    Residual forecasting validation layer.

    For each horizon:
    1. Extract the residual series: forecast[h] - realised[h] at each origin
    2. Walk-forward forecast those residuals (EWMA on the residual series)
    3. Build a "bias-corrected" forecast = original - predicted_residual
    4. Check whether the corrected forecast beats the original
    5. Check whether the sign of the predicted residual confirms the
       position implied by the main forecast vs MIP

    If residuals are predictable, it means the model has systematic biases
    that a trader should be aware of (and could correct).
    """
    from src.models.forecaster import HORIZON_LABELS

    validations: List[ResidualValidation] = []

    for h in horizons:
        # Step 1: extract raw residuals and metadata
        timestamps = []
        raw_residuals = []      # forecast - realised
        fc_values = []
        real_values = []
        mkt_values = []

        for r in results:
            if h not in r.forecasts or h not in r.realised or h not in r.market_fwd:
                continue
            fc = r.forecasts[h]
            real = r.realised[h]
            mkt = r.market_fwd[h]
            timestamps.append(r.origin_datetime)
            raw_residuals.append(fc - real)
            fc_values.append(fc)
            real_values.append(real)
            mkt_values.append(mkt)

        n = len(raw_residuals)
        if n < residual_lookback + 10:
            continue

        resid_arr = np.array(raw_residuals)
        fc_arr = np.array(fc_values)
        real_arr = np.array(real_values)
        mkt_arr = np.array(mkt_values)

        # Lag-1 autocorrelation of residuals
        if n > 2:
            autocorr = float(np.corrcoef(resid_arr[:-1], resid_arr[1:])[0, 1])
        else:
            autocorr = 0.0

        # Step 2: walk-forward forecast the residuals
        predicted_residuals = np.full(n, np.nan)
        for i in range(residual_lookback, n):
            predicted_residuals[i] = _ewma_residual_forecast(
                resid_arr, i, residual_lookback
            )

        valid_mask = ~np.isnan(predicted_residuals)
        if valid_mask.sum() < 10:
            continue

        valid_idx = np.where(valid_mask)[0]
        pred_resid_valid = predicted_residuals[valid_idx]
        actual_resid_valid = resid_arr[valid_idx]
        fc_valid = fc_arr[valid_idx]
        real_valid = real_arr[valid_idx]
        mkt_valid = mkt_arr[valid_idx]

        # Step 3: residual forecast accuracy
        resid_of_resid = pred_resid_valid - actual_resid_valid
        resid_fc_mae = float(np.mean(np.abs(resid_of_resid)))

        ss_res = float(np.sum((pred_resid_valid - actual_resid_valid) ** 2))
        ss_tot = float(np.sum((actual_resid_valid - np.mean(actual_resid_valid)) ** 2))
        resid_r2 = 1.0 - ss_res / max(ss_tot, 1e-9)

        # Step 4: bias-corrected forecast
        corrected_fc = fc_valid - pred_resid_valid
        original_mae = float(np.mean(np.abs(fc_valid - real_valid)))
        corrected_mae = float(np.mean(np.abs(corrected_fc - real_valid)))
        improvement = (original_mae - corrected_mae) / max(original_mae, 1e-9)

        # Step 5: position confirmation analysis
        # The "position" implied by the main forecast: if forecast > MIP,
        # the model suggests price will be higher than market expects
        # (bullish on imbalance cost → conservative position).
        # The predicted residual says: "our model typically over/under-predicts
        # by this much".
        # Confirmation = predicted residual sign is consistent with the
        # forecast's deviation from market (both suggest same direction of bias).
        fc_vs_mkt = fc_valid - mkt_valid  # positive = our fc above market
        confirms = (np.sign(pred_resid_valid) == np.sign(fc_vs_mkt))

        confirmation_rate = float(confirms.mean())

        # Hit rate stratified by confirmation
        fc_abs_err = np.abs(fc_valid - real_valid)
        mkt_abs_err = np.abs(mkt_valid - real_valid)
        fc_wins = fc_abs_err < mkt_abs_err

        confirmed_mask = confirms
        unconfirmed_mask = ~confirms

        if confirmed_mask.sum() > 0:
            confirmed_hr = float(fc_wins[confirmed_mask].mean())
        else:
            confirmed_hr = np.nan

        if unconfirmed_mask.sum() > 0:
            unconfirmed_hr = float(fc_wins[unconfirmed_mask].mean())
        else:
            unconfirmed_hr = np.nan

        validations.append(ResidualValidation(
            horizon=h,
            horizon_label=HORIZON_LABELS.get(h, f"{h} SPs"),
            n_obs=int(valid_mask.sum()),
            residual_mean=float(np.mean(resid_arr)),
            residual_std=float(np.std(resid_arr)),
            residual_autocorr_1=autocorr,
            residual_forecast_mae=resid_fc_mae,
            residual_predictability=resid_r2,
            corrected_mae=corrected_mae,
            original_mae=original_mae,
            correction_improvement=improvement,
            confirmation_rate=confirmation_rate,
            confirmed_trade_hit_rate=confirmed_hr,
            unconfirmed_trade_hit_rate=unconfirmed_hr,
        ))

    return validations


def build_residual_series_for_chart(
    results: List[ForecastResult],
    horizon: int,
    residual_lookback: int = 48,
) -> Tuple[List, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build parallel arrays for charting: timestamps, actual residuals,
    predicted residuals, and the bias-corrected forecast.
    """
    timestamps = []
    raw_residuals = []
    fc_values = []

    for r in results:
        if horizon not in r.forecasts or horizon not in r.realised:
            continue
        timestamps.append(r.origin_datetime)
        raw_residuals.append(r.forecasts[horizon] - r.realised[horizon])
        fc_values.append(r.forecasts[horizon])

    n = len(raw_residuals)
    resid_arr = np.array(raw_residuals)

    predicted_residuals = np.full(n, np.nan)
    for i in range(residual_lookback, n):
        predicted_residuals[i] = _ewma_residual_forecast(
            resid_arr, i, residual_lookback
        )

    return timestamps, resid_arr, predicted_residuals, np.array(fc_values)
