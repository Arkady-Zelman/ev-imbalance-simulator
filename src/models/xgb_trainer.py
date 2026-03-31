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

import itertools
import logging
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

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

def _model_file(target: str = "sip") -> Path:
    return _CACHE_DIR / f"trained_xgb_{target}.joblib"

GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1, 0.2],
    "reg_alpha": [0.01, 0.1, 1.0],
    "reg_lambda": [0.01, 0.1, 1.0],
    "subsample": [0.5, 0.8, 1.0],
    "colsample_bytree": [0.5, 0.8, 1.0],
    "gamma": [0.0, 0.1, 0.5],
    "min_child_weight": [1, 5, 10],
    "max_delta_step": [0, 1, 5],
    "scale_pos_weight": [1, 5, 10],
    "base_score": [0.0, 0.5, 1.0],
}

CORE_GRID_KEYS = ("n_estimators", "max_depth", "learning_rate")

REPRESENTATIVE_HORIZONS = [48, 48 * 3, 48 * 7, 48 * 14]


@dataclass
class TrainedXGBModels:
    target: str = "sip"
    best_params: Dict[str, Dict[int, dict]] = field(default_factory=dict)
    best_scores: Dict[str, Dict[int, float]] = field(default_factory=dict)
    final_models: Dict[str, Dict[int, Any]] = field(default_factory=dict)
    col_means: Dict[str, Dict[int, Any]] = field(default_factory=dict)
    training_timestamp: float = 0.0
    backtest_errors: List[RollingErrorRow] = field(default_factory=list)
    backtest_crossovers: List[CrossoverResult] = field(default_factory=list)


def _grid_middle_value(param: str) -> Any:
    """Single representative value per hyperparameter (middle list entry)."""
    values = GRID[param]
    return values[len(values) // 2]


def _default_params_from_grid() -> dict:
    """Fallback XGBRegressor kwargs when no tuned params exist."""
    return {k: _grid_middle_value(k) for k in GRID}


def _generate_grid_combos() -> List[dict]:
    """
    In-depth search: full Cartesian product over core params only; other
    hyperparameters fixed at GRID middle values (27 combos).
    """
    fixed_tail = {k: _grid_middle_value(k) for k in GRID if k not in CORE_GRID_KEYS}
    combos: List[dict] = []
    for ne, md, lr in itertools.product(
        GRID["n_estimators"],
        GRID["max_depth"],
        GRID["learning_rate"],
    ):
        p = dict(fixed_tail)
        p["n_estimators"] = ne
        p["max_depth"] = md
        p["learning_rate"] = lr
        combos.append(p)
    return combos


def _generate_random_combos(n_samples: int = 30, seed: int = 42) -> List[dict]:
    """
    Random search: each combo picks one value per GRID key independently.
    """
    rng = random.Random(seed)
    keys = list(GRID.keys())
    combos: List[dict] = []
    for _ in range(n_samples):
        combos.append({k: rng.choice(GRID[k]) for k in keys})
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
    _FEATURE_HISTORY = 48 + 336
    _MIN_SAMPLES = 50

    n = len(target_values)
    end_idx = n - horizon_sps
    min_window = _FEATURE_HISTORY + _MIN_SAMPLES
    effective_lb = max(lookback_sps, min_window)
    start_idx = max(0, end_idx - effective_lb)
    train_start = max(start_idx, _FEATURE_HISTORY)

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


_WORST_WINDOW_SPS = 48 * 3  # 3-day rolling window for worst-case MAE


def _worst_window_mae(errors: np.ndarray, window: int = _WORST_WINDOW_SPS) -> float:
    """
    Worst-case MAE over any rolling window of `window` samples.
    Selects the model that minimises the worst consecutive stretch,
    not just the average — more robust for forward-curve trading.
    """
    if len(errors) <= window:
        return float(np.mean(errors))
    cumsum = np.cumsum(errors)
    rolling_sum = cumsum[window:] - cumsum[:-window]
    return float(np.max(rolling_sum) / window)


def _grid_search_single(
    X: np.ndarray,
    y: np.ndarray,
    param_combos: List[dict],
) -> Tuple[dict, float]:
    """
    Time-series-safe grid search: last 20% as validation.
    Picks the combo with the lowest worst-window MAE (worst consecutive
    3-day stretch) rather than average MAE, to ensure robustness at
    any point along the forward curve.
    Returns (best_params, best_worst_window_mae).
    """
    n = len(y)
    split = int(n * 0.8)
    if split < 20 or n - split < 5:
        return param_combos[len(param_combos) // 2], float("inf")

    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    best_score = float("inf")
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
        abs_errors = np.abs(preds - y_val)
        score = _worst_window_mae(abs_errors)
        if score < best_score:
            best_score = score
            best_params = params

    return best_params, best_score


def train_xgb_models(
    sip_series: pd.Series,
    mip_series: pd.Series,
    demand_series: Optional[pd.Series] = None,
    target: str = "sip",
    progress_callback: Optional[Callable[[float, str], None]] = None,
    param_search_mode: Literal["grid", "random"] = "grid",
    random_search_samples: int = 30,
) -> TrainedXGBModels:
    """
    Full training pipeline:
      1. Grid search per (lookback, representative horizon)
      2. Retrain final models with best params on full data
      3. Run rolling backtest with tuned params
      4. Package everything into TrainedXGBModels

    Parameters
    ----------
    target : "sip" or "demand" — which series to forecast.
    progress_callback(fraction, message) is called to update a progress bar.
    param_search_mode : "grid" uses a 27-combo sweep on n_estimators, max_depth,
        and learning_rate with other params at GRID medians; "random" draws
        random_search_samples combos from the full GRID.
    """
    if not _HAS_XGB:
        raise RuntimeError("xgboost is not installed")

    if param_search_mode == "grid":
        param_combos = _generate_grid_combos()
    else:
        n = max(1, int(random_search_samples))
        param_combos = _generate_random_combos(n_samples=n)

    sip_values = sip_series.values.astype(float)
    mip_values = mip_series.values.astype(float)
    demand_values = demand_series.values.astype(float) if demand_series is not None else None

    if target == "demand":
        if demand_values is None:
            raise ValueError("demand_series required when target='demand'")
        train_target = demand_values
        train_mip = None
        train_aux = sip_values
    else:
        train_target = sip_values
        train_mip = mip_values
        train_aux = demand_values

    target_desc = "Demand" if target == "demand" else "SIP"

    result = TrainedXGBModels(target=target, training_timestamp=time.time())

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
                    f"Grid search ({target_desc}): {lb_label} lookback, {h_days}d horizon "
                    f"({len(param_combos)} combos)…",
                )

            X, y, cmeans = _build_train_data(
                train_target, lb_sps, h_sps, train_mip, train_aux,
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
            f"Running rolling backtest ({target_desc}) with tuned params…",
        )

    best_global_params = _pick_global_best(result.best_params)

    errors, crossovers = run_rolling_backtest(
        sip_series, mip_series,
        method="xgb",
        target=target,
        demand_series=demand_series,
        xgb_params=best_global_params,
    )
    result.backtest_errors = errors
    result.backtest_crossovers = crossovers

    if progress_callback:
        progress_callback(1.0, f"{target_desc} training complete.")

    return result


def _pick_global_best(
    best_params: Dict[str, Dict[int, dict]],
) -> dict:
    """Pick the most common best param dict across all lookback/horizon cells."""
    serialized: List[Tuple[Tuple[str, Any], ...]] = []
    for lb_params in best_params.values():
        for p in lb_params.values():
            if not p:
                continue
            serialized.append(tuple(sorted(p.items())))
    if not serialized:
        return _default_params_from_grid()
    most_common_key = Counter(serialized).most_common(1)[0][0]
    return dict(most_common_key)


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
    Automatically routes features based on trained.target.
    """
    if not _HAS_XGB:
        return {}

    target = getattr(trained, "target", "sip")
    if target == "demand":
        if demand_values is None:
            return {}
        fc_target = demand_values
        fc_mip = None
        fc_aux = sip_values
    else:
        fc_target = sip_values
        fc_mip = mip_values
        fc_aux = demand_values

    origin_idx = len(fc_target) - 1
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

            feat = _build_features(fc_target, origin_idx, h_sps, fc_mip, fc_aux)
            if feat is None:
                continue
            feat = np.where(np.isnan(feat), cmeans, feat)
            pred = float(model.predict(feat.reshape(1, -1))[0])
            lb_forecasts[day] = pred

        result[lb_label] = lb_forecasts

    return result


# ── Disk Persistence ─────────────────────────────────────────────────────

def save_trained_models(trained: TrainedXGBModels, target: str = "sip") -> bool:
    """Save to disk for the given target. Returns True on success."""
    if not _HAS_JOBLIB:
        logger.warning("joblib not installed — cannot save models to disk")
        return False
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        fpath = _model_file(target)
        joblib.dump(trained, fpath)
        logger.info("Saved trained XGB models (%s) to %s", target, fpath)
        return True
    except OSError as exc:
        logger.warning("Failed to save XGB models: %s", exc)
        return False


def load_trained_models(target: str = "sip") -> Optional[TrainedXGBModels]:
    """Load from disk for the given target. Returns None if not found."""
    if not _HAS_JOBLIB:
        return None
    fpath = _model_file(target)
    if not fpath.exists():
        return None
    try:
        trained = joblib.load(fpath)
        if isinstance(trained, TrainedXGBModels):
            logger.info("Loaded trained XGB models (%s) from %s", target, fpath)
            return trained
        return None
    except Exception as exc:
        logger.warning("Failed to load XGB models: %s", exc)
        return None


def has_trained_models(target: str = "sip") -> bool:
    """Quick check whether a saved model file exists on disk for the target."""
    return _model_file(target).exists()
