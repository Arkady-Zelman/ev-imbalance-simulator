"""
XGBoost grid-search trainer with disk persistence and forward forecasting.

Workflow:
  1. train_xgb_models() — grid search per (lookback, representative horizon),
     retrain final models with best params, run rolling backtest, save to disk.
  2. save_trained_models() / load_trained_models() — joblib-based disk I/O.
  3. forecast_forward() — produce 14-day-ahead point forecasts from the
     latest available data using the stored models.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    xgb = None  # type: ignore[assignment]

try:
    import joblib
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False
    joblib = None  # type: ignore[assignment]

from src.models.rolling_backtest import (
    ROLLING_HORIZONS,
    ROLLING_LOOKBACKS,
    CrossoverResult,
    RollingErrorRow,
    run_rolling_backtest,
)
from src.models.xgb_forecaster import _build_features

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".cache" / "xgb_models"
_MODEL_FILE = _CACHE_DIR / "trained_xgb.joblib"

GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1, 0.2],
}

_FIXED_PARAMS = {
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
}

REPRESENTATIVE_HORIZONS = [48, 48 * 3, 48 * 7, 48 * 14]


@dataclass
class TrainedXGBModels:
    best_params: Dict[str, Dict[int, dict]] = field(default_factory=dict)
    best_scores: Dict[str, Dict[int, float]] = field(default_factory=dict)
    final_models: Dict[str, Dict[int, Any]] = field(default_factory=dict)
    col_means: Dict[str, Dict[int, Any]] = field(default_factory=dict)
    training_timestamp: float = 0.0
    backtest_errors: List[RollingErrorRow] = field(default_factory=list)
    backtest_crossovers: List[CrossoverResult] = field(default_factory=list)


def _generate_param_combos() -> List[dict]:
    combos = []
    for ne in GRID["n_estimators"]:
        for md in GRID["max_depth"]:
            for lr in GRID["learning_rate"]:
                p = dict(_FIXED_PARAMS)
                p["n_estimators"] = ne
                p["max_depth"] = md
                p["learning_rate"] = lr
                combos.append(p)
    return combos


def _build_train_data(
    target_values: np.ndarray,
    lookback_sps: int,
    horizon_sps: int,
    mip_values: Optional[np.ndarray],
    demand_values: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Build feature matrix and target vector for grid search.
    Uses the latest available data (end of array) as the training window.
    Returns (X, y, col_means) or (None, None, None) if insufficient data.
    """
    n = len(target_values)
    end_idx = n - horizon_sps
    start_idx = max(0, end_idx - lookback_sps)
    train_start = max(start_idx, 48 + 336)

    X_rows: list[np.ndarray] = []
    y_rows: list[float] = []

    for i in range(train_start, end_idx):
        if i + horizon_sps >= n:
            break
        feat = _build_features(target_values, i, horizon_sps, mip_values, demand_values)
        if feat is None:
            continue
        X_rows.append(feat)
        y_rows.append(float(target_values[i + horizon_sps]))

    if len(X_rows) < 30:
        return None, None, None

    X = np.vstack(X_rows)
    y = np.array(y_rows, dtype=np.float32)

    nan_mask = np.isnan(X)
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    for c in range(X.shape[1]):
        X[nan_mask[:, c], c] = col_means[c]

    return X, y, col_means


def _grid_search_single(
    X: np.ndarray,
    y: np.ndarray,
    param_combos: List[dict],
) -> Tuple[dict, float]:
    """
    Time-series-safe grid search: last 20% as validation.
    Returns (best_params, best_val_mae).
    """
    n = len(y)
    split = int(n * 0.8)
    if split < 20 or n - split < 5:
        return param_combos[len(param_combos) // 2], float("inf")

    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    best_mae = float("inf")
    best_params = param_combos[0]

    for params in param_combos:
        model = xgb.XGBRegressor(
            **params,
            random_state=42,
            verbosity=0,
            n_jobs=1,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        mae = float(np.mean(np.abs(preds - y_val)))
        if mae < best_mae:
            best_mae = mae
            best_params = params

    return best_params, best_mae


def train_xgb_models(
    sip_series: pd.Series,
    mip_series: pd.Series,
    demand_series: Optional[pd.Series] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> TrainedXGBModels:
    """
    Full training pipeline:
      1. Grid search per (lookback, representative horizon)
      2. Retrain final models with best params on full data
      3. Run rolling backtest with tuned params
      4. Package everything into TrainedXGBModels

    progress_callback(fraction, message) is called to update a progress bar.
    """
    if not _HAS_XGB:
        raise RuntimeError("xgboost is not installed")

    sip_values = sip_series.values.astype(float)
    mip_values = mip_series.values.astype(float)
    demand_values = demand_series.values.astype(float) if demand_series is not None else None

    param_combos = _generate_param_combos()
    result = TrainedXGBModels(training_timestamp=time.time())

    lookbacks = list(ROLLING_LOOKBACKS.items())
    total_steps = len(lookbacks) * len(REPRESENTATIVE_HORIZONS) + 1
    step_idx = 0

    for lb_label, lb_sps in lookbacks:
        result.best_params[lb_label] = {}
        result.best_scores[lb_label] = {}
        result.final_models[lb_label] = {}
        result.col_means[lb_label] = {}

        for h_sps in REPRESENTATIVE_HORIZONS:
            step_idx += 1
            h_days = h_sps // 48
            if progress_callback:
                progress_callback(
                    step_idx / total_steps,
                    f"Grid search: {lb_label} lookback, {h_days}d horizon "
                    f"({len(param_combos)} combos)…",
                )

            X, y, cmeans = _build_train_data(
                sip_values, lb_sps, h_sps, mip_values, demand_values,
            )
            if X is None:
                logger.info("Insufficient data for %s / %dd — skipping", lb_label, h_days)
                continue

            best_p, best_score = _grid_search_single(X, y, param_combos)
            result.best_params[lb_label][h_sps] = best_p
            result.best_scores[lb_label][h_sps] = best_score

            final_model = xgb.XGBRegressor(
                **best_p,
                random_state=42,
                verbosity=0,
                n_jobs=1,
            )
            final_model.fit(X, y)
            result.final_models[lb_label][h_sps] = final_model
            result.col_means[lb_label][h_sps] = cmeans

    if progress_callback:
        progress_callback(
            step_idx / total_steps,
            "Running rolling backtest with tuned params…",
        )

    best_global_params = _pick_global_best(result.best_params)

    errors, crossovers = run_rolling_backtest(
        sip_series, mip_series,
        method="xgb",
        target="sip",
        demand_series=demand_series,
        xgb_params=best_global_params,
    )
    result.backtest_errors = errors
    result.backtest_crossovers = crossovers

    if progress_callback:
        progress_callback(1.0, "Training complete.")

    return result


def _pick_global_best(
    best_params: Dict[str, Dict[int, dict]],
) -> dict:
    """Pick the most common best param combo across all lookback/horizon cells."""
    from collections import Counter
    keys = []
    for lb_params in best_params.values():
        for p in lb_params.values():
            key = (p.get("n_estimators"), p.get("max_depth"), p.get("learning_rate"))
            keys.append(key)
    if not keys:
        return dict(_FIXED_PARAMS, n_estimators=100, max_depth=5, learning_rate=0.1)
    most_common = Counter(keys).most_common(1)[0][0]
    return dict(
        _FIXED_PARAMS,
        n_estimators=most_common[0],
        max_depth=most_common[1],
        learning_rate=most_common[2],
    )


def forecast_forward(
    trained: TrainedXGBModels,
    sip_values: np.ndarray,
    mip_values: Optional[np.ndarray] = None,
    demand_values: Optional[np.ndarray] = None,
    n_days: int = 14,
) -> Dict[str, Dict[int, float]]:
    """
    Produce forward forecasts from the end of the available data.

    Returns {lookback_label: {horizon_days: predicted_value}}.
    Uses the closest trained horizon model for each requested day.
    """
    if not _HAS_XGB:
        return {}

    origin_idx = len(sip_values) - 1
    result: Dict[str, Dict[int, float]] = {}

    for lb_label in trained.final_models:
        lb_forecasts: Dict[int, float] = {}
        available_h = sorted(trained.final_models[lb_label].keys())
        if not available_h:
            continue

        for day in range(1, n_days + 1):
            h_sps = day * 48
            closest_h = min(available_h, key=lambda h: abs(h - h_sps))
            model = trained.final_models[lb_label][closest_h]
            cmeans = trained.col_means[lb_label][closest_h]

            feat = _build_features(sip_values, origin_idx, h_sps, mip_values, demand_values)
            if feat is None:
                continue
            feat = np.where(np.isnan(feat), cmeans, feat)
            pred = float(model.predict(feat.reshape(1, -1))[0])
            lb_forecasts[day] = pred

        result[lb_label] = lb_forecasts

    return result


# ── Disk Persistence ─────────────────────────────────────────────────────

def save_trained_models(trained: TrainedXGBModels) -> bool:
    """Save to disk. Returns True on success."""
    if not _HAS_JOBLIB:
        logger.warning("joblib not installed — cannot save models to disk")
        return False
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(trained, _MODEL_FILE)
        logger.info("Saved trained XGB models to %s", _MODEL_FILE)
        return True
    except OSError as exc:
        logger.warning("Failed to save XGB models: %s", exc)
        return False


def load_trained_models() -> Optional[TrainedXGBModels]:
    """Load from disk. Returns None if not found or load fails."""
    if not _HAS_JOBLIB:
        return None
    if not _MODEL_FILE.exists():
        return None
    try:
        trained = joblib.load(_MODEL_FILE)
        if isinstance(trained, TrainedXGBModels):
            logger.info("Loaded trained XGB models from %s", _MODEL_FILE)
            return trained
        return None
    except Exception as exc:
        logger.warning("Failed to load XGB models: %s", exc)
        return None


def has_trained_models() -> bool:
    """Quick check whether a saved model file exists on disk."""
    return _MODEL_FILE.exists()
