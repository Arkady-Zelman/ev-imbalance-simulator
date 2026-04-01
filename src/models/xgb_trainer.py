"""
XGBoost grid-search trainer with disk persistence and forward forecasting.

Workflow
--------
1. train_xgb_models()  — random search over full hyperparameter space,
   retrain final models with best params, run rolling backtest, save to disk.
2. save_trained_models() / load_trained_models() — joblib-based disk I/O.
3. forecast_forward()  — produce 14-day-ahead point forecasts from the
   latest available data using the stored models.

Search modes
------------
- "grid" (in-depth): 150 random samples across all GRID params.
  Intended to be run infrequently; uses GPU if CUDA is available.
- "random" (quick):  30 random samples for fast point-in-time tuning.
"""

from __future__ import annotations

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
from src.models.xgb_forecaster import _build_features, _fill_nans

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".cache" / "xgb_models"


def _model_file(target: str = "sip") -> Path:
    return _CACHE_DIR / f"trained_xgb_{target}.joblib"


# ── Hyperparameter search space ───────────────────────────────────────────────
# All parameters are sampled in every search; "in-depth" simply draws more
# samples (150) than "quick" (30). Full Cartesian over this space is impractical
# (~4^12 combos), so random search is used for both modes.

GRID = {
    "n_estimators":      [100, 200, 400, 800],  # floored at 100; lr=0.01+100 still learns
    "max_depth":         [3, 4, 5, 6, 7],
    "learning_rate":     [0.01, 0.05, 0.1, 0.2],
    "subsample":         [0.6, 0.8, 1.0],
    "colsample_bytree":  [0.6, 0.8, 1.0],
    "colsample_bylevel": [0.6, 0.8, 1.0],
    "colsample_bynode":  [0.6, 0.8, 1.0],
    "reg_alpha":         [0.0, 0.01, 0.1, 1.0],
    "reg_lambda":        [0.1, 0.5, 1.0, 5.0],
    "gamma":             [0.0, 0.05, 0.1, 0.5],
    "min_child_weight":  [1, 3, 5, 10],
    # max_delta_step omitted — only meaningful for class-imbalance classification
}

# Number of random samples per search mode
_N_SAMPLES = {"grid": 150, "random": 30}

# Representative horizons (SPs) used during grid search training
REPRESENTATIVE_HORIZONS = [48, 48 * 3, 48 * 7, 48 * 14]

# Rolling window (SPs) for worst-case MAE scoring
_WORST_WINDOW_SPS = 48 * 3  # 3-day window


# ── GPU / device detection ────────────────────────────────────────────────────

_DEVICE_PROBED = False
_XGB_DEVICE_KWARGS: dict = {}


def _get_device_kwargs() -> dict:
    """
    Returns {"device": "cuda"} if XGBoost CUDA GPU support is available,
    otherwise returns {} (CPU). Result is cached after the first call.
    """
    global _DEVICE_PROBED, _XGB_DEVICE_KWARGS
    if _DEVICE_PROBED:
        return _XGB_DEVICE_KWARGS
    _DEVICE_PROBED = True
    if not _HAS_XGB:
        return {}
    try:
        probe = xgb.XGBRegressor(device="cuda", n_estimators=1, verbosity=0)
        probe.fit(np.array([[1.0]]), np.array([1.0]))
        _XGB_DEVICE_KWARGS = {"device": "cuda"}
        logger.info("XGBoost: CUDA GPU detected — using GPU acceleration.")
    except Exception:
        _XGB_DEVICE_KWARGS = {}
        logger.info("XGBoost: no CUDA GPU found — using CPU.")
    return _XGB_DEVICE_KWARGS


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class TrainedXGBModels:
    target: str = "sip"
    best_params:  Dict[str, Dict[int, dict]]  = field(default_factory=dict)
    best_scores:  Dict[str, Dict[int, float]] = field(default_factory=dict)
    final_models: Dict[str, Dict[int, Any]]   = field(default_factory=dict)
    col_means:    Dict[str, Dict[int, Any]]   = field(default_factory=dict)
    training_timestamp: float = 0.0
    backtest_errors:    List[RollingErrorRow]  = field(default_factory=list)
    backtest_crossovers: List[CrossoverResult] = field(default_factory=list)


# ── Param combo generation ────────────────────────────────────────────────────

def _generate_random_combos(n_samples: int = 30, seed: int = 42) -> List[dict]:
    """Draw n_samples random parameter combinations from GRID."""
    rng = random.Random(seed)
    return [{k: rng.choice(v) for k, v in GRID.items()} for _ in range(n_samples)]


def _generate_grid_combos() -> List[dict]:
    """Alias for in-depth search: 150 random samples (full grid too large for cartesian)."""
    return _generate_random_combos(n_samples=_N_SAMPLES["grid"])


def _build_param_combos(
    mode: Literal["grid", "random"] = "grid",
    n_random: int = 30,
    seed: int = 42,
) -> List[dict]:
    """
    Build parameter combo list for hyperparameter search.
    mode='grid'   → 150 random samples (in-depth, run infrequently).
    mode='random' → n_random samples (quick estimate).
    """
    n = _N_SAMPLES["grid"] if mode == "grid" else max(1, n_random)
    return _generate_random_combos(n_samples=n, seed=seed)


# ── Target routing ────────────────────────────────────────────────────────────

def _route_series(
    target: str,
    sip_v: np.ndarray,
    mip_v: np.ndarray,
    demand_v: Optional[np.ndarray],
    gen_v: Optional[np.ndarray] = None,
    wind_v: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Returns (target_series, mip_feature, aux_feature) for _build_train_data
    and forecast_forward based on which series we are forecasting.

    SIP              → target=SIP,    mip_feature=MIP,  aux=demand
    MIP              → target=MIP,    mip_feature=None, aux=SIP
    demand           → target=demand, mip_feature=None, aux=SIP
    total_generation → target=gen,    mip_feature=None, aux=demand
    wind             → target=wind,   mip_feature=None, aux=demand
    """
    if target == "demand":
        if demand_v is None:
            raise ValueError("demand_series required when target='demand'")
        return demand_v, None, sip_v
    if target == "mip":
        return mip_v, None, sip_v
    if target == "total_generation":
        if gen_v is None:
            raise ValueError("gen_series required when target='total_generation'")
        return gen_v, None, demand_v
    if target == "wind":
        if wind_v is None:
            raise ValueError("wind_series required when target='wind'")
        return wind_v, None, demand_v
    # default: sip
    return sip_v, mip_v, demand_v


# ── Training data builder ─────────────────────────────────────────────────────

def _build_train_data(
    target_values: np.ndarray,
    lookback_sps: int,
    horizon_sps: int,
    mip_values: Optional[np.ndarray],
    demand_values: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Build feature matrix and target vector for a given (lookback, horizon).
    Uses the most recent data slice (end of array) as training window.
    Returns (X, y, col_means) or (None, None, None) if insufficient data.
    """
    _FEAT_HIST = 48 + 336  # minimum SP history for feature builder
    _MIN_SAMPLES = 50

    n = len(target_values)
    end_idx = n - horizon_sps
    effective_lb = max(lookback_sps, _FEAT_HIST + _MIN_SAMPLES)
    train_start = max(max(0, end_idx - effective_lb), _FEAT_HIST)

    X_rows, y_rows = [], []
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
    X, col_means = _fill_nans(X)
    return X, y, col_means


# ── Scoring criterion ─────────────────────────────────────────────────────────

def _worst_window_mae(errors: np.ndarray, window: int = _WORST_WINDOW_SPS) -> float:
    """
    Worst-case MAE over any rolling window of `window` samples.
    Selects the model that minimises the worst consecutive stretch — more
    robust for forward-curve trading than optimising average MAE.
    Uses a zero-prepended cumsum so the window starting at index 0 is included.
    """
    if len(errors) <= window:
        return float(np.mean(errors))
    cumsum = np.concatenate([[0.0], np.cumsum(errors)])
    rolling_sum = cumsum[window:] - cumsum[:-window]
    return float(np.max(rolling_sum) / window)


# ── Grid search (single cell) ─────────────────────────────────────────────────

def _grid_search_single(
    X: np.ndarray,
    y: np.ndarray,
    param_combos: List[dict],
) -> Tuple[dict, float]:
    """
    Time-series-safe grid search: last 20% held out as validation.
    Scores each combo by worst-window MAE on the validation fold.
    Returns (best_params, best_score).
    """
    n = len(y)
    split = int(n * 0.8)
    if split < 20 or n - split < 5:
        return param_combos[len(param_combos) // 2], float("inf")

    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    device_kwargs = _get_device_kwargs()
    best_score = float("inf")
    best_params = param_combos[0]

    for params in param_combos:
        model = xgb.XGBRegressor(
            **params, **device_kwargs,
            random_state=42, verbosity=0, n_jobs=1,
        )
        model.fit(X_tr, y_tr)
        abs_err = np.abs(model.predict(X_val) - y_val)
        score = _worst_window_mae(abs_err)
        if score < best_score:
            best_score = score
            best_params = params

    return best_params, best_score


# ── Full training pipeline ────────────────────────────────────────────────────

def train_xgb_models(
    sip_series: pd.Series,
    mip_series: pd.Series,
    demand_series: Optional[pd.Series] = None,
    target: str = "sip",
    progress_callback: Optional[Callable[[float, str], None]] = None,
    param_search_mode: Literal["grid", "random"] = "grid",
    random_search_samples: int = 30,
    gen_series: Optional[pd.Series] = None,
    wind_series: Optional[pd.Series] = None,
) -> TrainedXGBModels:
    """
    Full training pipeline:
      1. Build parameter combos (150 for in-depth, 30 for quick).
      2. Grid-search per (lookback, representative horizon).
      3. Retrain final models with best params on full data.
      4. Run rolling backtest with globally best params.
      5. Return everything packaged in TrainedXGBModels.

    Parameters
    ----------
    target : "sip", "demand", "mip", "total_generation", or "wind".
    param_search_mode : "grid" (150 samples, in-depth) or "random" (n samples, quick).
    progress_callback : Optional (fraction, message) callback for progress bars.
    gen_series  : Required when target="total_generation".
    wind_series : Required when target="wind".
    """
    if not _HAS_XGB:
        raise RuntimeError("xgboost is not installed")

    param_combos = _build_param_combos(
        mode=param_search_mode,
        n_random=max(1, int(random_search_samples)),
    )

    sip_v    = sip_series.values.astype(float)
    mip_v    = mip_series.values.astype(float)
    demand_v = demand_series.values.astype(float) if demand_series is not None else None
    gen_v    = gen_series.values.astype(float)   if gen_series   is not None else None
    wind_v   = wind_series.values.astype(float)  if wind_series  is not None else None

    train_target, train_mip, train_aux = _route_series(
        target, sip_v, mip_v, demand_v, gen_v, wind_v
    )
    target_desc = {
        "demand": "Demand", "mip": "Wholesale (MIP)",
        "total_generation": "Total Generation", "wind": "Wind Generation",
    }.get(target, "SIP")

    result = TrainedXGBModels(target=target, training_timestamp=time.time())

    lookbacks = list(ROLLING_LOOKBACKS.items())
    total_steps = len(lookbacks) * len(REPRESENTATIVE_HORIZONS) + 1
    step_idx = 0

    for lb_label, lb_sps in lookbacks:
        result.best_params[lb_label]  = {}
        result.best_scores[lb_label]  = {}
        result.final_models[lb_label] = {}
        result.col_means[lb_label]    = {}

        for h_sps in REPRESENTATIVE_HORIZONS:
            step_idx += 1
            h_days = h_sps // 48
            if progress_callback:
                progress_callback(
                    step_idx / total_steps,
                    f"Searching ({target_desc}): {lb_label} lookback, {h_days}d horizon "
                    f"[{len(param_combos)} combos]…",
                )

            X, y, cmeans = _build_train_data(
                train_target, lb_sps, h_sps, train_mip, train_aux,
            )
            if X is None:
                logger.info("Insufficient data for %s / %dd — skipping", lb_label, h_days)
                continue

            best_p, best_score = _grid_search_single(X, y, param_combos)
            result.best_params[lb_label][h_sps]  = best_p
            result.best_scores[lb_label][h_sps]  = best_score

            device_kwargs = _get_device_kwargs()
            final_model = xgb.XGBRegressor(
                **best_p, **device_kwargs,
                random_state=42, verbosity=0, n_jobs=1,
            )
            final_model.fit(X, y)
            result.final_models[lb_label][h_sps] = final_model
            result.col_means[lb_label][h_sps]    = cmeans

    if progress_callback:
        progress_callback(
            step_idx / total_steps,
            f"Running rolling backtest ({target_desc}) with tuned params…",
        )

    best_global = _pick_global_best(result.best_params, result.best_scores)
    errors, crossovers = run_rolling_backtest(
        sip_series, mip_series,
        method="xgb",
        target=target,
        demand_series=demand_series,
        xgb_params=best_global,
        gen_series=gen_series,
        wind_series=wind_series,
    )
    result.backtest_errors     = errors
    result.backtest_crossovers = crossovers

    if progress_callback:
        progress_callback(1.0, f"{target_desc} training complete.")

    return result


def _pick_global_best(
    best_params: Dict[str, Dict[int, dict]],
    best_scores: Optional[Dict[str, Dict[int, float]]] = None,
) -> dict:
    """
    Pick the param dict with the lowest worst-window MAE across all cells.
    Falls back to plurality vote if scores are not provided.
    Score-weighted selection avoids the plurality-vote bias toward large-data cells.
    """
    if best_scores:
        best_score_overall = float("inf")
        best_p_overall: Optional[dict] = None
        for lb_label, h_dict in best_scores.items():
            for h_sps, score in h_dict.items():
                if score < best_score_overall:
                    p = best_params.get(lb_label, {}).get(h_sps)
                    if p:
                        best_score_overall = score
                        best_p_overall = p
        if best_p_overall:
            return best_p_overall

    # Fallback: plurality vote
    serialized = []
    for lb_params in best_params.values():
        for p in lb_params.values():
            if p:
                serialized.append(tuple(sorted(p.items())))
    if not serialized:
        return _generate_random_combos(1)[0]
    return dict(Counter(serialized).most_common(1)[0][0])


# ── Forward forecasting ───────────────────────────────────────────────────────

def forecast_forward(
    trained: TrainedXGBModels,
    sip_values: np.ndarray,
    mip_values: Optional[np.ndarray] = None,
    demand_values: Optional[np.ndarray] = None,
    n_days: int = 14,
    gen_values: Optional[np.ndarray] = None,
    wind_values: Optional[np.ndarray] = None,
) -> Dict[str, Dict[int, float]]:
    """
    Produce forward forecasts from the end of available data.
    Returns {lookback_label: {horizon_days: predicted_value}}.
    Uses the closest trained horizon model for each requested day.
    """
    if not _HAS_XGB:
        return {}

    target = getattr(trained, "target", "sip")
    try:
        fc_target, fc_mip, fc_aux = _route_series(
            target, sip_values, mip_values, demand_values, gen_values, wind_values
        )
    except ValueError:
        return {}

    origin_idx = len(fc_target) - 1
    result: Dict[str, Dict[int, float]] = {}

    for lb_label, lb_models in trained.final_models.items():
        available_h = sorted(lb_models.keys())
        if not available_h:
            continue

        lb_forecasts: Dict[int, float] = {}
        for day in range(1, n_days + 1):
            h_sps = day * 48
            closest_h = min(available_h, key=lambda h: abs(h - h_sps))
            model   = lb_models[closest_h]
            cmeans  = trained.col_means[lb_label][closest_h]

            # Build features using closest_h (the horizon this model was trained on)
            # so that cyclical horizon features and target-SP encoding match training.
            feat = _build_features(fc_target, origin_idx, closest_h, fc_mip, fc_aux)
            if feat is None:
                continue
            feat = np.where(np.isnan(feat), cmeans, feat)
            lb_forecasts[day] = float(model.predict(feat.reshape(1, -1))[0])

        result[lb_label] = lb_forecasts

    return result


# ── Disk persistence ──────────────────────────────────────────────────────────

def save_trained_models(trained: TrainedXGBModels, target: str = "sip") -> bool:
    """Save TrainedXGBModels to disk. Returns True on success."""
    if not _HAS_JOBLIB:
        logger.warning("joblib not installed — cannot save models to disk")
        return False
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        fpath = _model_file(target)
        joblib.dump(trained, fpath)
        logger.info("Saved XGB models (%s) to %s", target, fpath)
        return True
    except OSError as exc:
        logger.warning("Failed to save XGB models: %s", exc)
        return False


def load_trained_models(target: str = "sip") -> Optional[TrainedXGBModels]:
    """Load TrainedXGBModels from disk. Returns None if not found or invalid."""
    if not _HAS_JOBLIB:
        return None
    fpath = _model_file(target)
    if not fpath.exists():
        return None
    try:
        trained = joblib.load(fpath)
        if isinstance(trained, TrainedXGBModels):
            logger.info("Loaded XGB models (%s) from %s", target, fpath)
            return trained
        return None
    except Exception as exc:
        logger.warning("Failed to load XGB models: %s", exc)
        return None


def has_trained_models(target: str = "sip") -> bool:
    """Return True if a saved model file exists for the given target."""
    return _model_file(target).exists()
