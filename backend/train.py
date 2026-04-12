"""
Offline training pipeline — train all 4 hybrid LSTM/XGBoost models.

Run:
    python -m backend.train

Outputs:
    models/hybrid_sip.joblib
    models/hybrid_mip.joblib
    models/hybrid_demand.joblib
    models/hybrid_gen.joblib

HPO is triggered automatically per target if the trained model shows no alpha
(i.e., MAE does not beat the MIP forward benchmark across any lookback).
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# ── Path bootstrap (run as `python -m backend.train` from project root) ───────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import FORECAST_TARGETS, MODEL_DIR, PREDICTION_DIR
from src.data.elexon_client import (
    fetch_demand_outturn_raw as fetch_demand_outturn,
    fetch_generation_outturn_raw as fetch_generation_outturn,
    fetch_market_index_raw as fetch_market_index,
    fetch_system_prices_raw as fetch_system_prices,
    pivot_generation_wide,
)
from src.data.weather_client import (
    fetch_demand_forecast_raw as fetch_demand_forecast,
    fetch_weather_data_raw as fetch_weather_data,
    fetch_wind_generation_forecast_raw as fetch_wind_generation_forecast,
)
from src.models.forecaster import build_aligned_series
from src.models.hybrid_forecaster import _compute_weights
from src.models.intraday_eval import eval_lookback_intraday, select_best_lookback
from src.models.calendar_features import build_calendar_exog_series
from src.models.explainability import (
    compute_lstm_integrated_gradients_importance,
    compute_xgb_native_shap_importance,
)
from src.models.lstm_trainer import TrainedLSTMModels, train_lstm_models
from src.models.rolling_backtest import run_rolling_backtest
from src.models.xgb_trainer import TrainedXGBModels, train_xgb_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── ELEXON Insights API data availability ─────────────────────────────────────
# The /balancing/settlement/system-prices endpoint has reliable data from ~2019.
_ELEXON_DATA_START = dt.date(2019, 1, 1)


# ── Data fetching ─────────────────────────────────────────────────────────────

def _probe_sip_start(candidate: dt.date, max_probe_days: int = 180) -> dt.date:
    """
    Walk forward from `candidate` until we find the first date that returns
    at least one SIP record (binary-search style, but linear is fine for ≤180 days).
    Returns candidate unchanged if data is found on the first try.
    """
    from src.data.elexon_client import _fetch_system_prices_for_date
    logger.info("Probing SIP data availability from %s…", candidate)
    cursor = candidate
    limit  = candidate + dt.timedelta(days=max_probe_days)
    while cursor <= limit:
        records = _fetch_system_prices_for_date(cursor)
        if records:
            logger.info("First SIP data found on %s", cursor)
            return cursor
        cursor += dt.timedelta(days=7)   # step weekly to probe quickly
    logger.warning("No SIP data found in probe window; using %s anyway", candidate)
    return candidate


def fetch_all_data(
    date_from: dt.date = _ELEXON_DATA_START,
    date_to: Optional[dt.date] = None,
) -> dict:
    """
    Fetch all raw data needed for training.

    Returns a dict with keys:
        sip_df, mip_df, demand_df, gen_df (raw DataFrames)
        sip_series, mip_series, demand_series (aligned pd.Series)
        gen_series  — total_generation pd.Series aligned to SIP index
        weather_df  — 30-min weather DataFrame
        wind_fc_df  — wind generation forecast
        demand_fc_df — national demand forecast
    """
    if date_to is None:
        date_to = dt.date.today()

    # Probe forward to find the first date with actual data
    date_from = _probe_sip_start(date_from)
    logger.info("Fetching ELEXON data: %s → %s", date_from, date_to)

    sip_df    = fetch_system_prices(date_from, date_to)
    mip_df    = fetch_market_index(date_from, date_to)
    demand_df = fetch_demand_outturn(date_from, date_to)
    gen_df    = fetch_generation_outturn(date_from, date_to)

    if sip_df.empty or mip_df.empty:
        raise RuntimeError(
            f"ELEXON returned empty SIP/MIP data for {date_from} → {date_to}. "
            "Check network connectivity and ELEXON API status."
        )

    # Align SIP, MIP, Demand into common half-hourly index
    sip_series, mip_series, demand_series, _ = build_aligned_series(
        sip_df, mip_df, demand_df
    )

    # Build total generation series aligned to the SIP index
    gen_series: Optional[pd.Series] = None
    if not gen_df.empty:
        gen_long = gen_df.copy()
        gen_long["datetime"] = pd.to_datetime(gen_long["settlementDate"]) + pd.to_timedelta(
            (gen_long["settlementPeriod"].astype(int) - 1) * 30, unit="min"
        )
        total_gen = (
            gen_long.groupby("datetime")["generation"]
            .sum()
            .sort_index()
        )
        total_gen = total_gen[~total_gen.index.duplicated(keep="first")]
        # Align to SIP index
        gen_series = total_gen.reindex(sip_series.index).ffill().bfill()

    logger.info(
        "Aligned series: SIP=%d, MIP=%d, demand=%s, gen=%s",
        len(sip_series), len(mip_series),
        len(demand_series) if demand_series is not None else "N/A",
        len(gen_series)   if gen_series   is not None else "N/A",
    )

    # Weather (Open-Meteo, no API key)
    logger.info("Fetching weather data…")
    weather_df     = fetch_weather_data(date_from, date_to)
    wind_fc_df     = fetch_wind_generation_forecast(date_from, date_to)
    demand_fc_df   = fetch_demand_forecast(date_from, date_to)

    return {
        "sip_df":        sip_df,
        "mip_df":        mip_df,
        "demand_df":     demand_df,
        "gen_df":        gen_df,
        "sip_series":    sip_series,
        "mip_series":    mip_series,
        "demand_series": demand_series,
        "gen_series":    gen_series,
        "weather_df":    weather_df,
        "wind_fc_df":    wind_fc_df,
        "demand_fc_df":  demand_fc_df,
    }


def build_exog_series(data: dict) -> Dict[str, pd.Series]:
    """
    Build the dict of exogenous Series used for all 4 model targets.

    Calendar columns:
        calendar_tod_sin, calendar_tod_cos,
        calendar_dow_sin, calendar_dow_cos,
        calendar_month_sin, calendar_month_cos
    Weather columns (from Open-Meteo):
        temperature_c, wind_speed_100m, solar_radiation, cloud_cover_pct
    ELEXON forecasts:
        wind_fc_mw, demand_fc_mw
    """
    exog: Dict[str, pd.Series] = {}
    sip_series = data.get("sip_series")

    if sip_series is not None and not sip_series.empty:
        exog.update(build_calendar_exog_series(sip_series.index))

    weather_df   = data.get("weather_df",   pd.DataFrame())
    wind_fc_df   = data.get("wind_fc_df",   pd.DataFrame())
    demand_fc_df = data.get("demand_fc_df", pd.DataFrame())

    for col in ("temperature_c", "wind_speed_100m", "solar_radiation", "cloud_cover_pct"):
        if not weather_df.empty and col in weather_df.columns:
            exog[col] = weather_df[col].dropna()

    if not wind_fc_df.empty and "wind_fc_mw" in wind_fc_df.columns:
        exog["wind_fc_mw"] = wind_fc_df["wind_fc_mw"].dropna()

    if not demand_fc_df.empty and "demand_fc_mw" in demand_fc_df.columns:
        exog["demand_fc_mw"] = demand_fc_df["demand_fc_mw"].dropna()

    logger.info("Exog series built: %s", list(exog.keys()))
    return exog


# ── Per-target training ───────────────────────────────────────────────────────

def _best_val_mae(trained) -> Optional[float]:
    """Extract the best (lowest) validation MAE from a trained model artifact."""
    scores = getattr(trained, "best_scores", {})
    maes = []
    for lb_scores in scores.values():
        for score in lb_scores.values():
            if score is not None and np.isfinite(score) and score > 0:
                maes.append(score)
    return float(np.min(maes)) if maes else None


def train_single_target(
    target: str,
    data: dict,
    exog: Dict[str, pd.Series],
    param_search_mode: str = "grid",
) -> dict:
    """
    Train XGBoost + LSTM for one target, compute hybrid weights, return artifact dict.

    For price targets (sip, mip): exog = weather only.
    For volume targets (demand, total_generation): exog = weather + cross-series.
    """
    sip_series    = data["sip_series"]
    mip_series    = data["mip_series"]
    demand_series = data["demand_series"]
    gen_series    = data["gen_series"]

    # Group 2 gets additional cross-series exog
    exog_for_target = dict(exog)  # copy
    if target == "demand" and gen_series is not None:
        exog_for_target["gen_mw"] = gen_series
    elif target == "total_generation" and demand_series is not None:
        exog_for_target["demand_mw"] = demand_series

    logger.info("=== Training XGBoost for target=%s (mode=%s) ===", target, param_search_mode)
    t0 = time.time()
    xgb_trained: TrainedXGBModels = train_xgb_models(
        sip_series=sip_series,
        mip_series=mip_series,
        demand_series=demand_series,
        gen_series=gen_series,
        target=target,
        param_search_mode=param_search_mode,
        exog_series=exog_for_target,
    )
    logger.info("XGBoost training done in %.1fs", time.time() - t0)

    logger.info("=== Training LSTM for target=%s (mode=%s) ===", target, param_search_mode)
    t0 = time.time()
    lstm_trained: TrainedLSTMModels = train_lstm_models(
        sip_series=sip_series,
        mip_series=mip_series,
        demand_series=demand_series,
        gen_series=gen_series,
        target=target,
        param_search_mode=param_search_mode,
        exog_series=exog_for_target,
    )
    logger.info("LSTM training done in %.1fs", time.time() - t0)

    xgb_val_mae  = _best_val_mae(xgb_trained)
    lstm_val_mae = _best_val_mae(lstm_trained)
    w_xgb, w_lstm = _compute_weights(xgb_val_mae, lstm_val_mae)

    logger.info(
        "Hybrid weights for %s: XGB=%.3f (MAE=%.4f), LSTM=%.3f (MAE=%.4f)",
        target, w_xgb, xgb_val_mae or 0.0, w_lstm, lstm_val_mae or 0.0,
    )

    artifact = {
        "target":             target,
        "xgb":                xgb_trained,
        "lstm":               lstm_trained,
        "hybrid_weights":     {"xgb": w_xgb, "lstm": w_lstm},
        "xgb_val_mae":        xgb_val_mae,
        "lstm_val_mae":       lstm_val_mae,
        "xgb_feature_importance": getattr(xgb_trained, "feature_importances_mean", {}) or {},
        "xgb_shap_importance": getattr(xgb_trained, "shap_importances_mean", {}) or {},
        "lstm_feature_attribution": getattr(lstm_trained, "feature_attributions_mean", {}) or {},
        "lstm_channel_names": getattr(lstm_trained, "channel_names", []) or [],
        "training_timestamp": time.time(),
        "hpo_summary":        {
            "xgb_search_mode":  param_search_mode,
            "lstm_search_mode": param_search_mode,
            "xgb_best_scores":  {
                lb: dict(h_scores)
                for lb, h_scores in (xgb_trained.best_scores or {}).items()
            },
        },
        "exog_keys": list(exog_for_target.keys()),
    }
    return artifact


# ── Intraday evaluation ───────────────────────────────────────────────────────

def evaluate_intraday(
    artifact: dict,
    data: dict,
    exog: Dict[str, pd.Series],
    eval_days: int = 30,
) -> dict:
    """
    After training, evaluate each lookback's h=48 model on the last ``eval_days``
    of data.  Adds to artifact:
        best_intraday_lookback  — chosen by lowest CRPS (tiebreaker: spike_recall)
        intraday_weights        — {"xgb": float, "lstm": float} tuned for h=48 MAE
        intraday_scores         — {lb_label: {metric: value}}
    """
    target        = artifact["target"]
    xgb_trained   = artifact["xgb"]
    lstm_trained  = artifact["lstm"]

    sip_series    = data["sip_series"]
    mip_series    = data["mip_series"]
    demand_series = data.get("demand_series")
    gen_series    = data.get("gen_series")

    # Build exog_dict aligned to sip_series index (np.ndarray per key)
    from backend.predict import _build_exog_arrays   # reuse predict-side helper
    exog_keys = artifact.get("exog_keys", [])
    exog_data = {
        "sip_series":    sip_series,
        "mip_series":    mip_series,
        "demand_series": demand_series,
        "gen_series":    gen_series,
    }
    # Merge weather/forecast series into exog_data for _build_exog_arrays
    for k, v in data.items():
        if k not in exog_data:
            exog_data[k] = v

    exog_dict_np = _build_exog_arrays(exog_data, artifact)

    logger.info("=== Intraday eval for target=%s (last %d days) ===", target, eval_days)
    scores, xgb_mae_per_lb, lstm_mae_per_lb = eval_lookback_intraday(
        xgb_trained   = xgb_trained,
        lstm_trained  = lstm_trained,
        sip_series    = sip_series,
        mip_series    = mip_series,
        demand_series = demand_series,
        gen_series    = gen_series,
        exog_dict     = exog_dict_np,
        target        = target,
        eval_days     = eval_days,
    )

    best_lb = select_best_lookback(scores)
    logger.info("Best intraday lookback for %s: %s", target, best_lb)

    # Intraday-specific hybrid weights (inverse-MAE on h=48 window)
    xgb_intra_mae  = xgb_mae_per_lb.get(best_lb)
    lstm_intra_mae = lstm_mae_per_lb.get(best_lb)
    w_xgb_intra, w_lstm_intra = _compute_weights(xgb_intra_mae, lstm_intra_mae)
    logger.info(
        "Intraday hybrid weights for %s: XGB=%.3f (MAE=%.4f), LSTM=%.3f (MAE=%.4f)",
        target, w_xgb_intra, xgb_intra_mae or 0.0, w_lstm_intra, lstm_intra_mae or 0.0,
    )

    artifact["best_intraday_lookback"] = best_lb
    artifact["intraday_weights"]       = {"xgb": w_xgb_intra, "lstm": w_lstm_intra}
    artifact["intraday_scores"]        = scores

    # XGB importances for the h=48 SP model at the chosen intraday lookback (most relevant for flat intraday curves)
    from src.models.xgb_forecaster import xgb_feature_name_list
    x_names = list(getattr(xgb_trained, "feature_names", []) or []) or xgb_feature_name_list(
        artifact.get("exog_keys") or []
    )
    m48 = xgb_trained.final_models.get(best_lb, {}).get(48)
    if m48 is not None and x_names:
        fi = getattr(m48, "feature_importances_", None)
        if fi is not None:
            arr = np.asarray(fi, dtype=float).ravel()
            if arr.shape[0] == len(x_names):
                s = float(arr.sum())
                if s > 0:
                    arr = arr / s
                artifact["xgb_feature_importance_intraday"] = {
                    x_names[i]: float(arr[i]) for i in range(len(x_names))
                }

    sip_v = sip_series.values.astype(np.float32)
    mip_v = mip_series.values.astype(np.float32)
    demand_v = demand_series.values.astype(np.float32) if demand_series is not None else None
    gen_v = gen_series.values.astype(np.float32) if gen_series is not None else None

    try:
        from src.models.xgb_trainer import _build_train_data as xgb_build_train_data
        from src.models.xgb_trainer import _route_series as xgb_route_series

        xgb_target, xgb_mip, xgb_aux = xgb_route_series(target, sip_v, mip_v, demand_v, gen_v)
        X_intra, _, _ = xgb_build_train_data(xgb_target, ROLLING_LOOKBACKS[best_lb], 48, xgb_mip, xgb_aux, exog_dict_np)
        if X_intra is not None and m48 is not None:
            artifact["xgb_shap_importance_intraday"] = compute_xgb_native_shap_importance(
                m48,
                X_intra,
                x_names,
            )
    except Exception as exc:
        logger.debug("Could not compute XGB intraday SHAP summary for %s: %s", target, exc)

    try:
        from src.models.lstm_trainer import _route_series as lstm_route_series
        from src.models.lstm_trainer import _rebuild_model as rebuild_lstm_model
        from src.models.lstm_forecaster import build_lstm_sequences

        lstm_target, lstm_aux_channels = lstm_route_series(target, sip_v, mip_v, demand_v, gen_v)
        lstm_exog_arrays = list(exog_dict_np.values()) if exog_dict_np else []
        X_lstm, _ = build_lstm_sequences(
            lstm_target,
            origin_idx=len(lstm_target),
            lookback_sps=ROLLING_LOOKBACKS[best_lb],
            horizon_sps=48,
            seq_len=lstm_trained.seq_len,
            aux_channels=lstm_aux_channels + lstm_exog_arrays,
        )
        if X_lstm is not None:
            state_np = lstm_trained.final_state_dicts.get(best_lb, {}).get(48)
            config = lstm_trained.model_configs.get(best_lb, {}).get(48)
            scaler = lstm_trained.scalers.get(best_lb, {}).get(48)
            if state_np is not None and config is not None:
                from src.models.lstm_trainer import _apply_scaler
                X_lstm_scaled = _apply_scaler(X_lstm, scaler)
                lstm_model = rebuild_lstm_model(config, state_np)
                artifact["lstm_feature_attribution_intraday"] = compute_lstm_integrated_gradients_importance(
                    lstm_model,
                    X_lstm_scaled,
                    getattr(lstm_trained, "channel_names", []),
                )
    except Exception as exc:
        logger.debug("Could not compute LSTM intraday attribution summary for %s: %s", target, exc)

    return artifact


# ── Alpha check ───────────────────────────────────────────────────────────────

def check_alpha(artifact: dict, sip_series: pd.Series, mip_series: pd.Series) -> bool:
    """
    Returns True if the model beats a naive time-of-day mean baseline (alpha_mae > 0)
    for at least one (lookback, horizon) cell.

    Uses method="tod_mean" — no model re-training, runs in ~1-2 minutes.
    The previous method="xgb" was re-training XGBoost at every daily step
    across 6 years of data (~100K model fits) causing 24-48h runtimes.
    """
    target = artifact["target"]

    errors, _ = run_rolling_backtest(
        sip_series=sip_series,
        mip_series=mip_series,
        method="tod_mean",
        target=target,
        demand_series=None,
    )

    if not errors:
        logger.warning("check_alpha: no backtest rows returned for target=%s", target)
        return True  # assume alpha if backtest returns nothing (don't re-train needlessly)

    has_alpha = any(e.alpha_mae > 0 for e in errors)
    positive_count = sum(1 for e in errors if e.alpha_mae > 0)
    logger.info(
        "Alpha check %s: %d/%d cells positive vs tod_mean baseline → alpha=%s",
        target, positive_count, len(errors), has_alpha,
    )
    return has_alpha


# ── Save artifact ─────────────────────────────────────────────────────────────

def save_artifact(artifact: dict) -> Path:
    target = artifact["target"]
    path = MODEL_DIR / f"hybrid_{target}.joblib"
    joblib.dump(artifact, path)
    logger.info("Saved: %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=== EV Flex Trading — Backend Training Pipeline ===")
    logger.info("Targets: %s", FORECAST_TARGETS)

    data = fetch_all_data()
    exog = build_exog_series(data)

    sip_series = data["sip_series"]
    mip_series = data["mip_series"]

    summary_rows = []

    for target in FORECAST_TARGETS:
        logger.info("\n" + "=" * 60)
        logger.info("TARGET: %s", target.upper())
        logger.info("=" * 60)

        # Train once (grid search across lookbacks × horizons)
        artifact = train_single_target(target, data, exog, param_search_mode="grid")

        # Intraday evaluation — selects best lookback + intraday-specific hybrid weights
        artifact = evaluate_intraday(artifact, data, exog)

        # Alpha check vs naive time-of-day baseline (~1-2 min, no re-training)
        has_alpha = check_alpha(artifact, sip_series, mip_series)

        artifact["alpha_status"] = has_alpha
        save_artifact(artifact)

        best_lb        = artifact.get("best_intraday_lookback", "N/A")
        intra_scores   = artifact.get("intraday_scores", {})
        best_crps      = intra_scores.get(best_lb, {}).get("crps", float("nan"))

        summary_rows.append({
            "target":            target,
            "xgb_val_mae":       f"{artifact['xgb_val_mae']:.4f}" if artifact["xgb_val_mae"] else "N/A",
            "lstm_val_mae":      f"{artifact['lstm_val_mae']:.4f}" if artifact["lstm_val_mae"] else "N/A",
            "w_xgb":             f"{artifact['hybrid_weights']['xgb']:.3f}",
            "w_lstm":            f"{artifact['hybrid_weights']['lstm']:.3f}",
            "alpha":             "YES" if has_alpha else "NO",
            "best_intraday_lb":  best_lb,
            "intraday_crps":     f"{best_crps:.4f}" if np.isfinite(best_crps) else "N/A",
        })

    # Print summary table
    logger.info("\n=== Training Summary ===")
    header = (
        f"{'Target':<20} {'XGB MAE':<12} {'LSTM MAE':<12} {'W_XGB':<8} {'W_LSTM':<8} "
        f"{'Alpha':<8} {'Best Intraday LB':<18} {'Intraday CRPS'}"
    )
    logger.info(header)
    logger.info("-" * len(header))
    for row in summary_rows:
        logger.info(
            f"{row['target']:<20} {row['xgb_val_mae']:<12} {row['lstm_val_mae']:<12} "
            f"{row['w_xgb']:<8} {row['w_lstm']:<8} {row['alpha']:<8} "
            f"{row['best_intraday_lb']:<18} {row['intraday_crps']}"
        )

    logger.info("\nAll models saved to: %s", MODEL_DIR)
    logger.info("Run 'python -m backend.predict' to generate prediction files.")


if __name__ == "__main__":
    main()
