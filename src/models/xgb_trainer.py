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
import threading
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
    "n_estimators":      [100, 200, 400],              # removed 800 — early stopping selects optimal depth
    "max_depth":         [3, 4, 5, 6],                 # removed 7 — too prone to overfit
    "learning_rate":     [0.005, 0.01, 0.05, 0.1],    # added 0.005 for slow-learn + high-reg combos
    "subsample":         [0.5, 0.6, 0.8, 1.0],        # added 0.5 for stronger stochasticity
    "colsample_bytree":  [0.5, 0.6, 0.8, 1.0],        # added 0.5
    "colsample_bylevel": [0.6, 0.8, 1.0],
    "colsample_bynode":  [0.6, 0.8, 1.0],
    "reg_alpha":         [0.0, 0.1, 1.0, 5.0, 10.0],  # extended ceiling for L1 regularisation
    "reg_lambda":        [1.0, 5.0, 10.0, 20.0],      # raised floor + extended ceiling for L2
    "gamma":             [0.0, 0.1, 0.5, 1.0, 2.0],   # extended for stronger split penalty
    "min_child_weight":  [3, 5, 10, 20, 50],           # shifted up — penalise small-leaf overfitting
    "objective":         ["reg:squarederror", "reg:pseudohubererror"],  # huber robust to price spikes
    # max_delta_step omitted — only meaningful for class-imbalance classification
}

# Number of random samples per search mode
_N_SAMPLES = {"grid": 150, "random": 10}

# Representative horizons (SPs) used during grid search training.
# One model per day 1–14 so forecast_forward has an exact-horizon model for
# every day ahead — eliminates the step-function artifact caused by reusing
# the closest model across multiple forecast days.
REPRESENTATIVE_HORIZONS = [48 * d for d in range(1, 15)]      # 14 models
# Reduced set for quick random search (4 pivot horizons)
REPRESENTATIVE_HORIZONS_QUICK = [48, 48 * 3, 48 * 7, 48 * 14]

# Rolling window (SPs) for worst-case MAE scoring
_WORST_WINDOW_SPS = 48 * 3  # 3-day window


# ── GPU / device detection ────────────────────────────────────────────────────

_DEVICE_PROBED = False
_XGB_DEVICE_KWARGS: dict = {}
_DEVICE_LOCK = threading.Lock()


def _get_device_kwargs() -> dict:
    """
    Returns {} (CPU). GPU training is disabled to prevent VRAM exhaustion
    when 4 models train in parallel. Re-enable by restoring the CUDA probe.
    """
    return {}


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
    # Diagnostic fields — populated during training, drive the Model Diagnostics UI
    # {lb_label: {h_sps: [(params_dict, val_score), ...]}} — all combos tried
    grid_search_history: Dict[str, Dict[int, List[tuple]]] = field(default_factory=dict)
    # {lb_label: {h_sps: train_mae}} — in-sample MAE of best combo (for overfitting gap)
    train_scores: Dict[str, Dict[int, float]] = field(default_factory=dict)
    # Ordered list of exogenous series keys used during training (for inference alignment)
    exog_keys: List[str] = field(default_factory=list)


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


# ── Exogenous series alignment ────────────────────────────────────────────────

def _align_to_target(target: pd.Series, exog: pd.Series) -> np.ndarray:
    """
    Reindex exog Series to target's datetime index.
    Forward-fills up to 4 consecutive missing periods (2 hours), then fills
    any remaining NaN with the series mean (or 0 if entirely missing).
    Returns a float32 numpy array of the same length as target.
    """
    aligned = exog.reindex(target.index).ffill(limit=4)
    fill_val = float(aligned.mean()) if not pd.isna(aligned.mean()) else 0.0
    return aligned.fillna(fill_val).to_numpy(dtype=np.float32)


# ── Target routing ────────────────────────────────────────────────────────────

def _route_series(
    target: str,
    sip_v: np.ndarray,
    mip_v: np.ndarray,
    demand_v: Optional[np.ndarray],
    gen_v: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Returns (target_series, mip_feature, aux_feature) for _build_train_data
    and forecast_forward based on which series we are forecasting.

    SIP              → target=SIP,    mip_feature=MIP,  aux=demand
    MIP              → target=MIP,    mip_feature=None, aux=SIP
    demand           → target=demand, mip_feature=None, aux=SIP
    total_generation → target=gen,    mip_feature=None, aux=demand
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
    # default: sip
    return sip_v, mip_v, demand_v


# ── Training data builder ─────────────────────────────────────────────────────

def _build_train_data(
    target_values: np.ndarray,
    lookback_sps: int,
    horizon_sps: int,
    mip_values: Optional[np.ndarray],
    demand_values: Optional[np.ndarray],
    exog_dict: Optional[Dict[str, np.ndarray]] = None,
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
        feat = _build_features(target_values, i, horizon_sps, mip_values, demand_values, exog_dict)
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
) -> Tuple[dict, float, float, List[tuple]]:
    """
    Time-series-safe grid search: last 20% held out as validation.
    Scores each combo by worst-window MAE on the validation fold.
    Uses early stopping (20 rounds) to prevent overfitting per combo and
    auto-select the optimal number of trees.

    Returns (best_params, best_val_score, best_train_score, history).
    history = [(params, val_score), ...] for every combo tried — used for
    the parameter sensitivity diagnostics UI.
    best_params has n_estimators overridden to the early-stopped optimum.
    """
    n = len(y)
    split = int(n * 0.8)
    if split < 20 or n - split < 5:
        return param_combos[len(param_combos) // 2], float("inf"), float("inf"), []

    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    device_kwargs = _get_device_kwargs()
    best_score = float("inf")
    best_params = param_combos[0]
    best_ntrees = best_params.get("n_estimators", 200)
    history: List[tuple] = []

    for params in param_combos:
        try:
            model = xgb.XGBRegressor(
                **params, **device_kwargs,
                early_stopping_rounds=20,   # must be in constructor for XGBoost ≥3.x
                random_state=42, verbosity=0, n_jobs=1,
            )
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            abs_err = np.abs(model.predict(X_val) - y_val)
            score = _worst_window_mae(abs_err)
            history.append((params, score))
            if score < best_score:
                best_score = score
                best_params = params
                best_ntrees = getattr(model, "best_iteration", None) or params.get("n_estimators", 200)
        except Exception:
            history.append((params, float("inf")))
            continue

    # Override n_estimators in best_params with the early-stopped optimum
    best_params = {**best_params, "n_estimators": max(1, best_ntrees)}

    # Compute in-sample (train-set) MAE for the winning params to expose overfitting gap
    try:
        train_model = xgb.XGBRegressor(
            **best_params, **device_kwargs,
            random_state=42, verbosity=0, n_jobs=1,
        )
        train_model.fit(X_tr, y_tr)
        train_score = float(np.mean(np.abs(train_model.predict(X_tr) - y_tr)))
    except Exception:
        train_score = float("inf")

    return best_params, best_score, train_score, history


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
    selected_lookback: Optional[str] = None,
    exog_series: Optional[Dict[str, pd.Series]] = None,
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
    target : "sip", "demand", "mip", or "total_generation".
    param_search_mode : "grid" (150 samples, in-depth) or "random" (n samples, quick).
    progress_callback : Optional (fraction, message) callback for progress bars.
    gen_series  : Required when target="total_generation".
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

    train_target, train_mip, train_aux = _route_series(
        target, sip_v, mip_v, demand_v, gen_v
    )
    target_desc = {
        "demand": "Demand", "mip": "Wholesale (MIP)",
        "total_generation": "Total Generation",
    }.get(target, "SIP")

    # Build exog_dict: align each exogenous Series to the target series index
    target_series_for_align = {
        "demand": demand_series, "mip": mip_series,
        "total_generation": gen_series,
    }.get(target, sip_series)
    if target_series_for_align is None:
        target_series_for_align = sip_series

    exog_dict: Optional[Dict[str, np.ndarray]] = None
    if exog_series:
        exog_dict = {
            k: _align_to_target(target_series_for_align, s)
            for k, s in exog_series.items()
        }

    result = TrainedXGBModels(target=target, training_timestamp=time.time())
    result.exog_keys = list(exog_series.keys()) if exog_series else []

    if selected_lookback and selected_lookback in ROLLING_LOOKBACKS:
        lookbacks = [(selected_lookback, ROLLING_LOOKBACKS[selected_lookback])]
    else:
        lookbacks = list(ROLLING_LOOKBACKS.items())
    rep_horizons = REPRESENTATIVE_HORIZONS_QUICK if param_search_mode == "random" else REPRESENTATIVE_HORIZONS
    total_steps = len(lookbacks) * len(rep_horizons) + 1
    step_idx = 0

    for lb_label, lb_sps in lookbacks:
        result.best_params[lb_label]  = {}
        result.best_scores[lb_label]  = {}
        result.final_models[lb_label] = {}
        result.col_means[lb_label]    = {}

        for h_sps in rep_horizons:
            step_idx += 1
            h_days = h_sps // 48
            if progress_callback:
                progress_callback(
                    step_idx / total_steps,
                    f"Searching ({target_desc}): {lb_label} lookback, {h_days}d horizon "
                    f"[{len(param_combos)} combos]…",
                )

            X, y, cmeans = _build_train_data(
                train_target, lb_sps, h_sps, train_mip, train_aux, exog_dict,
            )
            if X is None:
                logger.info("Insufficient data for %s / %dd — skipping", lb_label, h_days)
                continue

            best_p, best_score, train_score, history = _grid_search_single(X, y, param_combos)
            result.best_params[lb_label][h_sps]  = best_p
            result.best_scores[lb_label][h_sps]  = best_score
            result.train_scores.setdefault(lb_label, {})[h_sps] = train_score
            result.grid_search_history.setdefault(lb_label, {})[h_sps] = history

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
            f"Running backtest ({target_desc}) with trained models…",
        )

    errors, crossovers = _backtest_with_final_models(
        result, sip_series, mip_series, demand_series, gen_series, target, exog_series
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


# ── Fast backtest using pre-trained final models ──────────────────────────────

def _backtest_with_final_models(
    trained: TrainedXGBModels,
    sip_series: pd.Series,
    mip_series: pd.Series,
    demand_series: Optional[pd.Series],
    gen_series: Optional[pd.Series],
    target: str,
    exog_series: Optional[Dict[str, pd.Series]] = None,
) -> Tuple[List[RollingErrorRow], List[CrossoverResult]]:
    """
    Rolling backtest using pre-trained final models — predict-only, no re-training.

    Uses the closest trained horizon model at each daily origin. Produces the same
    RollingErrorRow / CrossoverResult output as run_rolling_backtest(), but completes
    in seconds rather than hours because it calls model.predict() instead of model.fit().

    Note: models were trained on the full dataset, so backtest errors reflect in-sample
    fit rather than true walk-forward generalisation. This is acceptable for the purpose
    of identifying the crossover horizon.
    """
    from src.models.forecaster import _extract_market_forward, _extract_realised
    from src.models.stat_tests import diebold_mariano

    sip_v    = sip_series.values.astype(float)
    mip_v    = mip_series.values.astype(float)
    demand_v = demand_series.values.astype(float) if demand_series is not None else None
    gen_v    = (gen_series.reindex(sip_series.index).ffill().bfill().values.astype(float)
                if gen_series is not None else None)

    fc_target, fc_mip_feat, fc_aux = _route_series(
        target, sip_v, mip_v, demand_v, gen_v
    )

    # Align exog series to the target series index for backtest predictions
    bt_exog_dict: Optional[Dict[str, np.ndarray]] = None
    if exog_series:
        target_series_for_align = {
            "demand": demand_series, "mip": mip_series,
            "total_generation": gen_series,
        }.get(target, sip_series)
        if target_series_for_align is None:
            target_series_for_align = sip_series
        bt_exog_dict = {
            k: _align_to_target(target_series_for_align, s)
            for k, s in exog_series.items()
        }

    if target == "total_generation":
        benchmark_v: Optional[np.ndarray] = None
    elif target == "demand":
        benchmark_v = demand_v
    else:
        benchmark_v = mip_v  # SIP and MIP both benchmark against MIP forward curve

    n = len(fc_target)
    step = 48
    errors: List[RollingErrorRow] = []

    for lb_label, lb_sps in ROLLING_LOOKBACKS.items():
        if lb_label not in trained.final_models or not trained.final_models[lb_label]:
            continue
        lb_models  = trained.final_models[lb_label]
        available_h = sorted(lb_models.keys())

        for h_sps in ROLLING_HORIZONS:
            h_days    = h_sps // 48
            closest_h = min(available_h, key=lambda h: abs(h - h_sps))
            model     = lb_models[closest_h]
            cmeans    = trained.col_means[lb_label][closest_h]

            start_idx = max(96, lb_sps)
            end_idx   = n - h_sps
            if start_idx >= end_idx:
                continue

            fc_errs: list = []
            mkt_errs: list = []
            realised_vals: list = []

            for idx in range(start_idx, end_idx, step):
                feat = _build_features(fc_target, idx, closest_h, fc_mip_feat, fc_aux, bt_exog_dict)
                if feat is None:
                    continue
                if len(feat) != getattr(model, "n_features_in_", len(feat)):
                    continue  # stale model trained on different feature set — skip
                feat = np.where(np.isnan(feat), cmeans, feat) if cmeans is not None and len(cmeans) == len(feat) else np.nan_to_num(feat, nan=0.0)
                pred = float(model.predict(feat.reshape(1, -1))[0])

                realised_d = _extract_realised(fc_target, idx, [h_sps])
                if h_sps not in realised_d:
                    continue
                r = realised_d[h_sps]
                fc_errs.append(abs(pred - r))
                realised_vals.append(r)

                if target == "total_generation":
                    persist_idx = idx + h_sps - 48
                    if persist_idx >= 0:
                        mkt_errs.append(abs(float(fc_target[persist_idx]) - r))
                elif benchmark_v is not None:
                    mkt_d = _extract_market_forward(benchmark_v, idx, [h_sps],
                                                    lookback_sps=lb_sps)
                    if h_sps in mkt_d:
                        mkt_errs.append(abs(mkt_d[h_sps] - r))

            if len(fc_errs) < 5:
                continue
            if not mkt_errs:
                mkt_errs = list(fc_errs)  # fallback: no market benchmark, alpha = 0

            fc_arr   = np.array(fc_errs,      dtype=float)
            mkt_arr  = np.array(mkt_errs,     dtype=float)
            real_arr = np.array(realised_vals, dtype=float)
            safe_real = np.where(np.abs(real_arr) < 1e-6, 1e-6, real_arr)

            forecast_mae  = float(np.mean(fc_arr))
            forecast_rmse = float(np.sqrt(np.mean(fc_arr ** 2)))
            market_mae    = float(np.mean(mkt_arr))
            market_rmse   = float(np.sqrt(np.mean(mkt_arr ** 2)))
            forecast_mape = float(np.mean(np.minimum(fc_arr  / np.abs(safe_real) * 100, 500.0)))
            market_mape   = float(np.mean(np.minimum(mkt_arr / np.abs(safe_real) * 100, 500.0)))
            _, dm_p = diebold_mariano(fc_arr, mkt_arr, h=max(1, h_days), power=2)

            errors.append(RollingErrorRow(
                lookback_label=lb_label, lookback_sps=lb_sps,
                horizon_days=h_days, horizon_sps=h_sps,
                forecast_mae=forecast_mae, forecast_rmse=forecast_rmse,
                forecast_mape=forecast_mape, market_mae=market_mae,
                market_rmse=market_rmse, market_mape=market_mape,
                alpha_mae=market_mae - forecast_mae,
                alpha_mape=market_mape - forecast_mape,
                dm_pvalue=dm_p, n_obs=len(fc_errs),
            ))

    # Crossover computation — only over lookbacks that were actually trained
    crossovers: List[CrossoverResult] = []
    present_lookbacks = {e.lookback_label for e in errors}
    for lb_label in present_lookbacks:
        lb_rows = sorted(
            [e for e in errors if e.lookback_label == lb_label],
            key=lambda e: e.horizon_days,
        )
        crossover_day, last_pos, first_neg = 15, 0.0, 0.0
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
            lookback_label=lb_label, crossover_day=crossover_day,
            last_positive_alpha=last_pos, first_negative_alpha=first_neg,
        ))

    logger.info(
        "Fast XGB backtest: %d error rows, %d lookbacks",
        len(errors), len(ROLLING_LOOKBACKS),
    )
    return errors, crossovers


# ── Forward forecasting ───────────────────────────────────────────────────────

def forecast_forward(
    trained: TrainedXGBModels,
    sip_values: np.ndarray,
    mip_values: Optional[np.ndarray] = None,
    demand_values: Optional[np.ndarray] = None,
    n_days: int = 14,
    gen_values: Optional[np.ndarray] = None,
    exog_dict: Optional[Dict[str, np.ndarray]] = None,
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
            target, sip_values, mip_values, demand_values, gen_values
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
            feat = _build_features(fc_target, origin_idx, closest_h, fc_mip, fc_aux, exog_dict)
            if feat is None:
                continue
            if len(feat) != getattr(model, "n_features_in_", len(feat)):
                continue  # stale model trained on different feature set — skip
            feat = np.where(np.isnan(feat), cmeans, feat) if cmeans is not None and len(cmeans) == len(feat) else np.nan_to_num(feat, nan=0.0)
            lb_forecasts[day] = float(model.predict(feat.reshape(1, -1))[0])

        result[lb_label] = lb_forecasts

    return result


def forecast_intraday_48sp(
    trained: "TrainedXGBModels",
    sip_values: np.ndarray,
    mip_values: Optional[np.ndarray] = None,
    demand_values: Optional[np.ndarray] = None,
    gen_values: Optional[np.ndarray] = None,
    exog_dict: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """
    Generate a true 48-SP intraday day-ahead forecast for each lookback horizon.

    Uses the offset trick on the existing h=48 model — no retraining required.

    For each SP s (0-indexed, 0..47):
        virtual_idx = end_idx - 48 + s
        horizon     = 48
        → predicts target[virtual_idx + 48] = target[end_idx + s]  ← tomorrow's SP s

    SP-of-day cyclical features vary naturally with virtual_idx, so the model
    predicts each settlement period distinctly (not just a flat scaled value).

    Returns {lookback_label: np.ndarray(48,)}.
    """
    if not _HAS_XGB:
        return {}

    target = getattr(trained, "target", "sip")
    try:
        fc_target, fc_mip, fc_aux = _route_series(
            target, sip_values, mip_values, demand_values, gen_values
        )
    except ValueError:
        return {}

    end_idx = len(fc_target)
    result_48: Dict[str, np.ndarray] = {}

    for lb_label, lb_models in trained.final_models.items():
        if not lb_models:
            continue
        # Use 1-day model (h=48) — closest match to "tomorrow"
        h_key = 48 if 48 in lb_models else min(lb_models.keys())
        model  = lb_models[h_key]
        cmeans = trained.col_means.get(lb_label, {}).get(h_key)

        sp_fc = np.full(48, np.nan)
        for sp in range(48):
            virtual_idx = end_idx - 48 + sp
            if virtual_idx < 336:   # need ≥7 days of history for 7-day lag features
                continue
            feat = _build_features(fc_target, virtual_idx, 48, fc_mip, fc_aux, exog_dict)
            if feat is None:
                continue
            if len(feat) != getattr(model, "n_features_in_", len(feat)):
                continue  # stale model trained on different feature set — skip
            feat = np.where(np.isnan(feat), cmeans, feat) if cmeans is not None and len(cmeans) == len(feat) else np.nan_to_num(feat, nan=0.0)
            sp_fc[sp] = float(model.predict(feat.reshape(1, -1))[0])

        result_48[lb_label] = sp_fc

    return result_48


# ── Parallel multi-target training ───────────────────────────────────────────

from concurrent.futures import ThreadPoolExecutor as _TPE, as_completed as _as_completed


@dataclass
class ParallelTrainingResult:
    """Results from simultaneously training multiple XGB targets."""
    models: Dict[str, "TrainedXGBModels"]   # target -> trained model
    errors: Dict[str, str]                   # target -> error message if failed
    elapsed_seconds: float


def train_all_xgb_parallel(
    sip_series: pd.Series,
    mip_series: pd.Series,
    demand_series: Optional[pd.Series] = None,
    gen_series: Optional[pd.Series] = None,
    targets: Optional[List[str]] = None,
    param_search_mode: Literal["grid", "random"] = "grid",
    selected_lookback: Optional[str] = None,
    progress_callbacks: Optional[Dict[str, Callable[[float, str], None]]] = None,
    exog_series: Optional[Dict[str, pd.Series]] = None,
) -> ParallelTrainingResult:
    """
    Train up to 4 XGB models simultaneously using a ThreadPoolExecutor.

    XGBoost's C extension releases the GIL during fit/predict, so threads
    achieve near-true parallelism on multi-core hardware even in CPython.

    Parameters
    ----------
    targets : Targets to train. Defaults to available targets given the data.
    selected_lookback : If set, each model trains only this lookback window.
    progress_callbacks : Dict mapping target name to a (fraction, msg) callback.
    """
    if targets is None:
        targets = ["sip", "mip"]
        if demand_series is not None:
            targets.append("demand")
        if gen_series is not None:
            targets.append("total_generation")

    cbs: Dict[str, Callable] = progress_callbacks or {}
    t0 = time.time()
    results: Dict[str, TrainedXGBModels] = {}
    errs: Dict[str, str] = {}

    def _train_one(tgt: str) -> TrainedXGBModels:
        return train_xgb_models(
            sip_series=sip_series,
            mip_series=mip_series,
            demand_series=demand_series,
            gen_series=gen_series,
            target=tgt,
            progress_callback=cbs.get(tgt),
            param_search_mode=param_search_mode,
            selected_lookback=selected_lookback,
            exog_series=exog_series,
        )

    with _TPE(max_workers=len(targets)) as ex:
        futures = {ex.submit(_train_one, t): t for t in targets}
        for fut in _as_completed(futures):
            tgt = futures[fut]
            try:
                trained = fut.result()
                save_trained_models(trained, target=tgt)
                results[tgt] = trained
            except Exception as exc:
                errs[tgt] = str(exc)
                logger.error("Parallel XGB training failed for %s: %s", tgt, exc)

    return ParallelTrainingResult(
        models=results,
        errors=errs,
        elapsed_seconds=time.time() - t0,
    )


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
            # Force CPU inference: models may have been saved with device="cuda".
            # GPU prediction is unnecessary for the small forward-forecast workload
            # and can crash when 4 parallel training threads exhausted VRAM.
            for lb_models in trained.final_models.values():
                for model in lb_models.values():
                    try:
                        model.set_params(device="cpu")
                    except Exception:
                        pass
            logger.info("Loaded XGB models (%s) from %s", target, fpath)
            return trained
        return None
    except Exception as exc:
        logger.warning("Failed to load XGB models: %s", exc)
        return None


def has_trained_models(target: str = "sip") -> bool:
    """Return True if a saved model file exists for the given target."""
    return _model_file(target).exists()
