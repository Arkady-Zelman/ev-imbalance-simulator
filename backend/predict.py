"""
Prediction generation pipeline — read trained models, write parquet outputs.

Run:
    python -m backend.predict

Reads:
    models/hybrid_{target}.joblib  (4 files)

Writes:
    data/predictions/sip_predictions.parquet
    data/predictions/mip_predictions.parquet
    data/predictions/demand_predictions.parquet
    data/predictions/gen_predictions.parquet
    data/predictions/backtest_fan.parquet
    data/predictions/metadata.json
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import FORECAST_TARGETS, MODEL_DIR, PREDICTION_DIR
from src.data.elexon_client import (
    fetch_demand_outturn_raw as fetch_demand_outturn,
    fetch_generation_outturn_raw as fetch_generation_outturn,
    fetch_market_index_raw as fetch_market_index,
    fetch_system_prices_raw as fetch_system_prices,
)
from src.data.weather_client import (
    fetch_demand_forecast_raw as fetch_demand_forecast,
    fetch_weather_data_raw as fetch_weather_data,
    fetch_wind_generation_forecast_raw as fetch_wind_generation_forecast,
)
from src.models.forecaster import build_aligned_series
from src.models.calendar_features import build_calendar_exog_series
from src.models.lstm_trainer import (
    TrainedLSTMModels,
    forecast_forward as lstm_forecast_forward,
    forecast_intraday_48sp as lstm_intraday,
)
from src.predictions import (
    PREDICTION_TYPE_FORWARD_DAILY,
    PREDICTION_TYPE_INTRADAY_48SP,
    validate_prediction_products,
)
from src.models.rolling_backtest import ROLLING_LOOKBACKS, run_rolling_backtest
from src.models.xgb_trainer import (
    TrainedXGBModels,
    forecast_forward as xgb_forecast_forward,
    forecast_intraday_48sp as xgb_intraday,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Use last 90 days for recent-data forecasting context
_RECENT_DAYS = 90
# Retrospective fan chart: 52 weekly origins × 4 lookbacks
_FAN_N_ORIGINS = 52
_FAN_STEP_DAYS = 7


# ── Data helpers ──────────────────────────────────────────────────────────────

def _fetch_recent_data() -> dict:
    """Fetch the last 90 days of market + weather data for inference."""
    today = dt.date.today()
    date_from = today - dt.timedelta(days=_RECENT_DAYS)

    sip_df    = fetch_system_prices(date_from, today)
    mip_df    = fetch_market_index(date_from, today)
    demand_df = fetch_demand_outturn(date_from, today)
    gen_df    = fetch_generation_outturn(date_from, today)

    sip_series, mip_series, demand_series, _ = build_aligned_series(
        sip_df, mip_df, demand_df
    )

    gen_series: Optional[pd.Series] = None
    if not gen_df.empty:
        gen_long = gen_df.copy()
        gen_long["datetime"] = pd.to_datetime(gen_long["settlementDate"]) + pd.to_timedelta(
            (gen_long["settlementPeriod"].astype(int) - 1) * 30, unit="min"
        )
        total_gen = gen_long.groupby("datetime")["generation"].sum().sort_index()
        total_gen = total_gen[~total_gen.index.duplicated(keep="first")]
        gen_series = total_gen.reindex(sip_series.index).ffill().bfill()

    weather_df   = fetch_weather_data(date_from, today + dt.timedelta(days=16))
    wind_fc_df   = fetch_wind_generation_forecast(date_from, today + dt.timedelta(days=14))
    demand_fc_df = fetch_demand_forecast(date_from, today + dt.timedelta(days=14))

    return {
        "sip_series":    sip_series,
        "mip_series":    mip_series,
        "demand_series": demand_series,
        "gen_series":    gen_series,
        "weather_df":    weather_df,
        "wind_fc_df":    wind_fc_df,
        "demand_fc_df":  demand_fc_df,
    }


def _build_exog_arrays(data: dict, artifact: dict) -> Dict[str, np.ndarray]:
    """
    Build exog_dict aligned to sip_series index, matching training-time exog_keys.
    """
    sip_series   = data["sip_series"]
    exog_keys    = artifact.get("exog_keys", [])

    exog_sources: Dict[str, pd.Series] = {}
    weather_df   = data.get("weather_df",   pd.DataFrame())
    wind_fc_df   = data.get("wind_fc_df",   pd.DataFrame())
    demand_fc_df = data.get("demand_fc_df", pd.DataFrame())

    if sip_series is not None and not sip_series.empty:
        exog_sources.update(build_calendar_exog_series(sip_series.index))

    for col in ("temperature_c", "wind_speed_100m", "solar_radiation", "cloud_cover_pct"):
        if not weather_df.empty and col in weather_df.columns:
            exog_sources[col] = weather_df[col]

    if not wind_fc_df.empty and "wind_fc_mw" in wind_fc_df.columns:
        exog_sources["wind_fc_mw"] = wind_fc_df["wind_fc_mw"]

    if not demand_fc_df.empty and "demand_fc_mw" in demand_fc_df.columns:
        exog_sources["demand_fc_mw"] = demand_fc_df["demand_fc_mw"]

    gen_series    = data.get("gen_series")
    demand_series = data.get("demand_series")
    if gen_series    is not None: exog_sources["gen_mw"]    = gen_series
    if demand_series is not None: exog_sources["demand_mw"] = demand_series

    result: Dict[str, np.ndarray] = {}
    for key in exog_keys:
        if key in exog_sources:
            s = exog_sources[key]
            aligned = s.reindex(sip_series.index).ffill(limit=4)
            fill = float(aligned.mean()) if not pd.isna(aligned.mean()) else 0.0
            result[key] = aligned.fillna(fill).to_numpy(dtype=np.float32)

    return result


# ── Forward predictions ───────────────────────────────────────────────────────

def generate_forward_predictions(
    target: str,
    artifact: dict,
    data: dict,
) -> pd.DataFrame:
    """
    Generate one daily-horizon scalar forecast per future day and lookback.

    These rows are deliberately marked with ``settlement_period = 0`` because
    they are not a true 48-settlement-period intraday curve.
    """
    xgb_trained: TrainedXGBModels = artifact["xgb"]
    lstm_trained: TrainedLSTMModels = artifact["lstm"]
    w_xgb: float = artifact["hybrid_weights"]["xgb"]
    w_lstm: float = artifact["hybrid_weights"]["lstm"]

    sip_v    = data["sip_series"].values.astype(float)
    mip_v    = data["mip_series"].values.astype(float)
    dem_v    = data["demand_series"].values.astype(float) if data["demand_series"] is not None else None
    gen_v    = data["gen_series"].values.astype(float)   if data["gen_series"]    is not None else None
    exog_d   = _build_exog_arrays(data, artifact)

    xgb_fc   = xgb_forecast_forward(xgb_trained,  sip_v, mip_v, dem_v, 14, gen_v, exog_d)
    lstm_fc  = lstm_forecast_forward(lstm_trained, sip_v, mip_v, dem_v, 14, gen_v, exog_d)

    today = dt.date.today()
    rows: list = []
    for lb_label in ROLLING_LOOKBACKS:
        xgb_lb  = xgb_fc.get(lb_label,  {})
        lstm_lb = lstm_fc.get(lb_label, {})
        all_days = sorted(set(xgb_lb) | set(lstm_lb))
        for day in all_days:
            xgb_val  = xgb_lb.get(day)
            lstm_val = lstm_lb.get(day)

            if xgb_val is not None and lstm_val is not None:
                hybrid = w_xgb * xgb_val + w_lstm * lstm_val
            elif xgb_val is not None:
                hybrid = xgb_val
            elif lstm_val is not None:
                hybrid = lstm_val
            else:
                continue

            forecast_date = today + dt.timedelta(days=day)
            rows.append({
                "forecast_date":      pd.Timestamp(forecast_date),
                "settlement_period":  0,      # daily-forward scalar product
                "target":             target,
                "lookback":           lb_label,
                "horizon_days":       day,
                "hybrid_prediction":  hybrid,
                "xgb_prediction":     xgb_val if xgb_val is not None else np.nan,
                "lstm_prediction":    lstm_val if lstm_val is not None else np.nan,
                "prediction_type":    PREDICTION_TYPE_FORWARD_DAILY,
            })

    return validate_prediction_products(
        pd.DataFrame(rows),
        target=target,
        context=f"{target} forward-daily generation",
    )


# ── Intraday 48-SP predictions ────────────────────────────────────────────────

def generate_intraday_48sp(
    target: str,
    artifact: dict,
    data: dict,
) -> pd.DataFrame:
    """
    Generate 48 SP intraday day-ahead hybrid predictions for each lookback.

    Rows for the best lookback (chosen by CRPS during training) use the
    intraday-specific hybrid weights; all other lookbacks use global weights.
    The ``is_best_lookback`` boolean column marks the best-lookback rows.
    """
    xgb_trained: TrainedXGBModels  = artifact["xgb"]
    lstm_trained: TrainedLSTMModels = artifact["lstm"]

    # Global weights (used for non-best lookbacks)
    w_xgb_global:  float = artifact["hybrid_weights"]["xgb"]
    w_lstm_global: float = artifact["hybrid_weights"]["lstm"]

    # Intraday-specific weights for the best lookback (falls back to global)
    intraday_weights = artifact.get("intraday_weights", artifact["hybrid_weights"])
    w_xgb_intra:  float = intraday_weights["xgb"]
    w_lstm_intra: float = intraday_weights["lstm"]

    best_lb: str = artifact.get("best_intraday_lookback", "")

    sip_v  = data["sip_series"].values.astype(float)
    mip_v  = data["mip_series"].values.astype(float)
    dem_v  = data["demand_series"].values.astype(float) if data["demand_series"] is not None else None
    gen_v  = data["gen_series"].values.astype(float)   if data["gen_series"]    is not None else None
    exog_d = _build_exog_arrays(data, artifact)

    xgb_fc  = xgb_intraday(xgb_trained,  sip_v, mip_v, dem_v, gen_v, exog_d)
    lstm_fc = lstm_intraday(lstm_trained, sip_v, mip_v, dem_v, gen_v, exog_d)

    tomorrow = dt.date.today() + dt.timedelta(days=1)
    rows: list = []

    for lb_label in ROLLING_LOOKBACKS:
        xgb_arr  = xgb_fc.get(lb_label)
        lstm_arr = lstm_fc.get(lb_label)

        if xgb_arr is None and lstm_arr is None:
            continue

        is_best = (lb_label == best_lb)
        w_xgb   = w_xgb_intra  if is_best else w_xgb_global
        w_lstm  = w_lstm_intra if is_best else w_lstm_global

        for sp in range(48):
            xgb_val  = float(xgb_arr[sp])  if xgb_arr  is not None else None
            lstm_val = float(lstm_arr[sp]) if lstm_arr is not None else None

            if xgb_val is not None and lstm_val is not None:
                hybrid = w_xgb * xgb_val + w_lstm * lstm_val
            elif xgb_val is not None:
                hybrid = xgb_val
            else:
                hybrid = lstm_val

            rows.append({
                "forecast_date":      pd.Timestamp(tomorrow),
                "settlement_period":  sp + 1,
                "target":             target,
                "lookback":           lb_label,
                "horizon_days":       0,
                "hybrid_prediction":  hybrid,
                "xgb_prediction":     xgb_val if xgb_val is not None else np.nan,
                "lstm_prediction":    lstm_val if lstm_val is not None else np.nan,
                "prediction_type":    PREDICTION_TYPE_INTRADAY_48SP,
                "is_best_lookback":   is_best,
            })

    return validate_prediction_products(
        pd.DataFrame(rows),
        target=target,
        context=f"{target} intraday-48sp generation",
    )


# ── Retrospective fan data ────────────────────────────────────────────────────

def generate_retrospective_fan_data(
    target: str,
    artifact: dict,
    sip_series: pd.Series,
    mip_series: pd.Series,
    demand_series: Optional[pd.Series],
    gen_series: Optional[pd.Series],
    n_origins: int = _FAN_N_ORIGINS,
    step_days: int = _FAN_STEP_DAYS,
    data: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Walk back n_origins weekly steps from the end of data, generate hybrid
    forecast from each origin across all lookbacks and horizons 1–14 days.

    Columns: origin_date, target, horizon_days, lookback, hybrid_prediction,
             realised_value, forward_curve_value
    """
    xgb_trained:  TrainedXGBModels  = artifact["xgb"]
    lstm_trained: TrainedLSTMModels = artifact["lstm"]
    w_xgb:  float = artifact["hybrid_weights"]["xgb"]
    w_lstm: float = artifact["hybrid_weights"]["lstm"]

    sip_v  = sip_series.values.astype(float)
    mip_v  = mip_series.values.astype(float)
    dem_v  = demand_series.values.astype(float) if demand_series is not None else None
    gen_v  = gen_series.values.astype(float)   if gen_series    is not None else None

    # Build exog arrays (full length) — sliced per origin below
    exog_full: Dict[str, np.ndarray] = {}
    if data is not None:
        exog_full = _build_exog_arrays(
            {**data, "sip_series": sip_series},
            artifact,
        )

    # Target values for this forecast target
    target_map = {
        "sip":              sip_v,
        "mip":              mip_v,
        "demand":           dem_v,
        "total_generation": gen_v,
    }
    target_v = target_map.get(target, sip_v)
    if target_v is None:
        logger.warning("Fan data: target_v is None for target=%s; skipping.", target)
        return pd.DataFrame()

    # MIP as benchmark for price targets; persistence for volume targets
    benchmark_v = mip_v if target in ("sip", "mip") else target_v

    n = len(target_v)
    step_sps = step_days * 48
    max_horizon_sps = 14 * 48
    min_lookback_sps = max(ROLLING_LOOKBACKS.values())  # 15 * 48

    rows: list = []

    # Walk back from near the end; leave room for max horizon
    end_origin_idx = n - max_horizon_sps - 1
    start_origin_idx = max(min_lookback_sps, end_origin_idx - n_origins * step_sps)

    for origin_idx in range(end_origin_idx, start_origin_idx, -step_sps):
        if origin_idx < min_lookback_sps:
            break

        origin_ts = sip_series.index[origin_idx] if origin_idx < len(sip_series) else None
        if origin_ts is None:
            continue

        # Slice data up to origin for inference
        sip_slice = sip_v[:origin_idx + 1]
        mip_slice = mip_v[:origin_idx + 1]
        dem_slice = dem_v[:origin_idx + 1] if dem_v is not None else None
        gen_slice = gen_v[:origin_idx + 1] if gen_v is not None else None
        exog_slice = {k: v[:origin_idx + 1] for k, v in exog_full.items()}

        xgb_fc  = xgb_forecast_forward(xgb_trained,  sip_slice, mip_slice, dem_slice, 14, gen_slice, exog_slice)
        lstm_fc = lstm_forecast_forward(lstm_trained, sip_slice, mip_slice, dem_slice, 14, gen_slice, exog_slice)

        for lb_label in ROLLING_LOOKBACKS:
            xgb_lb  = xgb_fc.get(lb_label,  {})
            lstm_lb = lstm_fc.get(lb_label, {})

            for day in range(1, 15):
                h_sps = day * 48
                future_idx = origin_idx + h_sps
                if future_idx >= n:
                    continue

                xgb_val  = xgb_lb.get(day)
                lstm_val = lstm_lb.get(day)

                if xgb_val is not None and lstm_val is not None:
                    hybrid = w_xgb * xgb_val + w_lstm * lstm_val
                elif xgb_val is not None:
                    hybrid = xgb_val
                elif lstm_val is not None:
                    hybrid = lstm_val
                else:
                    continue

                realised = float(target_v[future_idx])

                # Forward curve value: MIP at that future SP (or persistence for volume)
                if target in ("sip", "mip"):
                    fwd_val = float(benchmark_v[future_idx])
                else:
                    # 1-day-lag persistence benchmark
                    persist_idx = future_idx - 48
                    fwd_val = float(benchmark_v[persist_idx]) if persist_idx >= 0 else realised

                rows.append({
                    "origin_date":         origin_ts,
                    "target":              target,
                    "horizon_days":        day,
                    "lookback":            lb_label,
                    "hybrid_prediction":   hybrid,
                    "realised_value":      realised,
                    "forward_curve_value": fwd_val,
                })

    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=== Ohme Fleet Trading — Prediction Pipeline ===")

    # Load artifacts
    artifacts: Dict[str, dict] = {}
    for target in FORECAST_TARGETS:
        path = MODEL_DIR / f"hybrid_{target}.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found: {path}\n"
                "Run 'python -m backend.train' first."
            )
        logger.info("Loading model: %s", path.name)
        artifacts[target] = joblib.load(path)

    # Fetch recent market data
    logger.info("Fetching recent market data (last %d days)…", _RECENT_DAYS)
    data = _fetch_recent_data()

    sip_series    = data["sip_series"]
    mip_series    = data["mip_series"]
    demand_series = data["demand_series"]
    gen_series    = data["gen_series"]

    metadata = {
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "data_range": {
            "from": str(sip_series.index.min().date()) if not sip_series.empty else None,
            "to":   str(sip_series.index.max().date()) if not sip_series.empty else None,
        },
        "targets": {},
    }

    all_prediction_dfs: Dict[str, list] = {t: [] for t in FORECAST_TARGETS}

    for target in FORECAST_TARGETS:
        artifact = artifacts[target]
        logger.info("\n--- Generating predictions for: %s ---", target)

        # Daily forward scalar predictions (one row per future day)
        fwd_df = generate_forward_predictions(target, artifact, data)
        if not fwd_df.empty:
            all_prediction_dfs[target].append(fwd_df)
            logger.info("  Forward daily predictions: %d rows", len(fwd_df))

        # True intraday 48-SP predictions
        intra_df = generate_intraday_48sp(target, artifact, data)
        if not intra_df.empty:
            all_prediction_dfs[target].append(intra_df)
            logger.info("  Intraday 48-SP predictions: %d rows", len(intra_df))

        metadata["targets"][target] = {
            "training_timestamp":      artifact.get("training_timestamp", 0.0),
            "xgb_val_mae":             artifact.get("xgb_val_mae"),
            "lstm_val_mae":            artifact.get("lstm_val_mae"),
            "hybrid_weights":          artifact.get("hybrid_weights", {}),
            "alpha_status":            artifact.get("alpha_status", False),
            "exog_keys":               artifact.get("exog_keys", []),
            "best_intraday_lookback":  artifact.get("best_intraday_lookback"),
            "intraday_weights":        artifact.get("intraday_weights", {}),
            "xgb_feature_importance":  artifact.get("xgb_feature_importance") or {},
            "xgb_feature_importance_intraday": artifact.get("xgb_feature_importance_intraday") or {},
            "xgb_shap_importance": artifact.get("xgb_shap_importance") or {},
            "xgb_shap_importance_intraday": artifact.get("xgb_shap_importance_intraday") or {},
            "lstm_feature_attribution": artifact.get("lstm_feature_attribution") or {},
            "lstm_feature_attribution_intraday": artifact.get("lstm_feature_attribution_intraday") or {},
            "lstm_channel_names": artifact.get("lstm_channel_names") or [],
        }

    # Save per-target prediction parquets
    target_file_map = {
        "sip":              "sip_predictions.parquet",
        "mip":              "mip_predictions.parquet",
        "demand":           "demand_predictions.parquet",
        "total_generation": "gen_predictions.parquet",
    }
    for target, parts in all_prediction_dfs.items():
        if not parts:
            logger.warning("No predictions generated for %s — skipping parquet.", target)
            continue
        combined = validate_prediction_products(
            pd.concat(parts, ignore_index=True),
            target=target,
            context=f"{target} parquet output",
        )
        out_path = PREDICTION_DIR / target_file_map[target]
        combined.to_parquet(out_path, index=False)
        logger.info("Saved %s (%d rows, %.1f KB)", out_path.name, len(combined), out_path.stat().st_size / 1e3)

    # Generate retrospective fan data (used for Tab 1 fan charts)
    logger.info("\n--- Generating retrospective fan data (%d origins × 4 lookbacks) ---",
                _FAN_N_ORIGINS)
    fan_parts: list = []
    for target in FORECAST_TARGETS:
        artifact = artifacts[target]
        fan_df = generate_retrospective_fan_data(
            target, artifact,
            sip_series, mip_series, demand_series, gen_series,
            data=data,
        )
        if not fan_df.empty:
            fan_parts.append(fan_df)
            logger.info("  Fan data %s: %d rows", target, len(fan_df))

    if fan_parts:
        fan_combined = pd.concat(fan_parts, ignore_index=True)
        fan_path = PREDICTION_DIR / "backtest_fan.parquet"
        fan_combined.to_parquet(fan_path, index=False)
        logger.info("Saved %s (%d rows, %.1f KB)", fan_path.name, len(fan_combined), fan_path.stat().st_size / 1e3)
        metadata["fan_origins"] = _FAN_N_ORIGINS
        metadata["fan_step_days"] = _FAN_STEP_DAYS
    else:
        logger.warning("No fan data generated.")

    # Save metadata
    meta_path = PREDICTION_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info("Saved %s", meta_path.name)

    logger.info("\n=== Prediction pipeline complete ===")
    logger.info("Outputs in: %s", PREDICTION_DIR)


if __name__ == "__main__":
    main()
