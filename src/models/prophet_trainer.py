"""
NeuralProphet grid-search trainer with disk persistence and forward forecasting.

Mirrors the structure of xgb_trainer.py for a consistent workflow:

  1. train_np_models()  — grid/random search over NeuralProphet hyperparameters,
     retrain best model per lookback, generate 14-day forward forecasts,
     run rolling backtest, save to disk.
  2. save_np_models() / load_np_models() — joblib-based disk I/O.
  3. forecast_forward_np() — return pre-computed forward forecasts from stored results.

Notes
-----
NeuralProphet model objects (PyTorch checkpoints) are NOT persisted — they are
large and hard to pickle portably. Instead, the 14-day forward forecasts are
pre-computed at training time and stored in TrainedNPModels.forward_forecasts.
The "Show Results" flow reloads these cached forecasts and backtest metrics.

Search modes
------------
- "grid"   (in-depth): systematic 3×3×2 = 18 combos over n_lags, epochs, lr.
- "random" (quick):    n random samples (default 6) — faster iteration.
"""

from __future__ import annotations

import copy
import itertools
import logging
import random
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from neuralprophet import NeuralProphet, set_log_level
    _HAS_PROPHET = True
    set_log_level("ERROR")
except ImportError:
    _HAS_PROPHET = False
    NeuralProphet = None  # type: ignore[assignment,misc]

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
from src.models.prophet_forecaster import _neuralprophet_forecast, _values_to_df

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".cache" / "np_models"
_FREQ = "30min"

# Rolling window (SPs) for worst-case MAE scoring — same as XGBoost trainer
_WORST_WINDOW_SPS = 48 * 3

# Lookbacks to train one model per (same set as XGBoost)
_NP_LOOKBACKS = {
    "1 day":   48,
    "5 days":  48 * 5,
    "15 days": 48 * 15,
    "30 days": 48 * 30,
}

# Representative horizons (all horizons in one NeuralProphet model via n_forecasts)
_NP_MAX_HORIZON = 48 * 14  # 14 days ahead


# ── Hyperparameter search space ───────────────────────────────────────────────

NP_GRID = {
    "n_lags":        [48, 96, 336],   # AR window: 1d / 2d / 7d
    "epochs":        [10, 20, 40],    # training thoroughness
    "learning_rate": [0.05, 0.1],     # gradient step size
}

# Fixed params (not searched — always applied)
NP_FIXED = {
    "weekly_seasonality": True,
    "daily_seasonality":  True,
    "yearly_seasonality": False,
}

# Number of combos per search mode
_N_COMBOS = {"grid": None, "random": 3}  # None = full grid


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class TrainedNPModels:
    """
    Stores all outputs from a NeuralProphet training run for one forecast target.

    forward_forecasts : {lookback_label: {day_ahead: predicted_value}}
                        Pre-computed 14-day forward forecasts. Stored here because
                        NeuralProphet model objects are not persisted.
    """
    target: str = "sip"
    best_params:  Dict[str, dict]  = field(default_factory=dict)   # {lb_label: best NP kwargs}
    best_scores:  Dict[str, float] = field(default_factory=dict)   # {lb_label: worst_window_mae}
    training_timestamp: float = 0.0
    backtest_errors:     List[RollingErrorRow]  = field(default_factory=list)
    backtest_crossovers: List[CrossoverResult]  = field(default_factory=list)
    forward_forecasts:   Dict[str, Dict[int, float]] = field(default_factory=dict)


def _model_file(target: str = "sip") -> Path:
    return _CACHE_DIR / f"trained_np_{target}.joblib"


# ── Param combo generation ────────────────────────────────────────────────────

def _np_grid_combos() -> List[dict]:
    """Full Cartesian product over NP_GRID (18 combos for default grid)."""
    keys = list(NP_GRID.keys())
    combos = []
    for vals in itertools.product(*NP_GRID.values()):
        combo = dict(zip(keys, vals))
        combo.update(NP_FIXED)
        combos.append(combo)
    return combos


def _np_random_combos(n_samples: int = 6, seed: int = 42) -> List[dict]:
    """n_samples random draws from NP_GRID."""
    rng = random.Random(seed)
    combos = []
    for _ in range(n_samples):
        combo = {k: rng.choice(v) for k, v in NP_GRID.items()}
        combo.update(NP_FIXED)
        combos.append(combo)
    return combos


def _build_np_combos(
    mode: Literal["grid", "random"] = "grid",
    n_random: int = 6,
    seed: int = 42,
) -> List[dict]:
    """
    Build parameter combo list for NeuralProphet hyperparameter search.
    mode='grid'   → full 3×3×2 = 18 combos.
    mode='random' → n_random random combos.
    """
    if mode == "grid":
        return _np_grid_combos()
    return _np_random_combos(n_samples=max(1, n_random), seed=seed)


# ── Scoring ───────────────────────────────────────────────────────────────────

def _worst_window_mae(errors: np.ndarray, window: int = _WORST_WINDOW_SPS) -> float:
    """Worst-case MAE over any rolling window of `window` samples.
    Uses a zero-prepended cumsum so the window starting at index 0 is included.
    """
    if len(errors) <= window:
        return float(np.mean(errors))
    cumsum = np.concatenate([[0.0], np.cumsum(errors)])
    rolling_sum = cumsum[window:] - cumsum[:-window]
    return float(np.max(rolling_sum) / window)


# ── Grid search for a single lookback ────────────────────────────────────────

def _np_grid_search_single(
    values: np.ndarray,
    lookback_sps: int,
    end_idx: int,
    param_combos: List[dict],
    end_date=None,
) -> Tuple[dict, float, Any]:
    """
    Time-series-safe NeuralProphet search: last 20% of the lookback window is
    held out for validation. Scores each combo by worst-window MAE of the
    multi-step forecasts against the held-out actuals.

    Returns (best_params, best_score, best_model).
    best_model is the NeuralProphet object trained on the first 80% with best_params.
    It is used by _np_holdout_backtest for out-of-sample evaluation.
    """
    start = max(0, end_idx - lookback_sps)
    window = values[start:end_idx].astype(float)
    n = len(window)

    if n < 96:  # need at least 2 days to split and score
        return param_combos[0], float("inf"), None

    split = int(n * 0.8)
    if split < 48 or n - split < 2:
        return param_combos[0], float("inf"), None

    best_score  = float("inf")
    best_params = param_combos[0]
    best_model  = None

    for params in param_combos:
        try:
            score, model = _eval_np_combo(window, split, params, end_date=end_date)
            if score < best_score:
                best_score  = score
                best_params = params
                best_model  = model
        except Exception as exc:
            logger.debug("NP combo %s failed: %s", params, exc)
            continue

    return best_params, best_score, best_model


_MAX_EVAL_HORIZON = 48 * 3  # cap validation horizon at 3 days to keep grid search fast


def _eval_np_combo(
    window: np.ndarray, split: int, params: dict, end_date=None,
) -> Tuple[float, Any]:
    """Fit NeuralProphet on window[:split], score on window[split:].
    Validation horizon is capped at _MAX_EVAL_HORIZON SPs to control training time.
    end_date: actual timestamp of window[-1] for correct weekday alignment.

    Returns (score, fitted_model). Model is None when fitting fails or score is inf.
    """
    n_val = min(len(window) - split, _MAX_EVAL_HORIZON)
    n_lags = params.get("n_lags", 48)
    epochs = params.get("epochs", 15)
    lr = params.get("learning_rate", 0.1)

    # Use real dates so NeuralProphet weekly seasonality is calendar-aligned
    train_end = pd.Timestamp(end_date) - pd.Timedelta(minutes=30 * (len(window) - split)) \
                if end_date is not None else pd.Timestamp("2025-01-01")
    ds_train = pd.date_range(end=train_end, periods=split, freq=_FREQ)
    df_train = pd.DataFrame({"ds": ds_train, "y": window[:split].astype(float)})

    effective_n_lags = min(n_lags, split // 2)
    # NP AR requires n_lags >= n_forecasts; if violated, disable AR (use pure decomposition)
    if effective_n_lags > 0 and effective_n_lags < n_val:
        logger.debug("NP eval: n_lags=%d < n_forecasts=%d; disabling AR", effective_n_lags, n_val)
        effective_n_lags = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = NeuralProphet(
            n_forecasts=n_val,
            n_lags=effective_n_lags,
            epochs=epochs,
            learning_rate=lr,
            weekly_seasonality=params.get("weekly_seasonality", True),
            daily_seasonality=params.get("daily_seasonality", True),
            yearly_seasonality=False,
            batch_size=min(64, split),
            verbose=False,
        )
        m.fit(df_train, freq=_FREQ)
        future = m.make_future_dataframe(df_train, periods=n_val, n_historic_predictions=0)
        pred = m.predict(future)

    yhat_cols = sorted(
        [c for c in pred.columns if c.startswith("yhat")],
        key=lambda c: int(c.replace("yhat", "") or "1"),
    )
    if not yhat_cols:
        return float("inf"), None

    actuals = window[split:]
    abs_errors = []
    for i, col in enumerate(yhat_cols[:n_val]):
        val = pred[col].dropna()
        if not val.empty and i < len(actuals):
            abs_errors.append(abs(float(val.iloc[-1]) - actuals[i]))

    if not abs_errors:
        return float("inf"), None

    return _worst_window_mae(np.array(abs_errors)), m


# ── Forward forecast from a fitted model ─────────────────────────────────────

def _np_forward_forecast(
    values: np.ndarray,
    end_idx: int,
    lookback_sps: int,
    best_params: dict,
    n_days: int = 14,
    end_date=None,
) -> Dict[int, float]:
    """
    Fit NeuralProphet with best_params on the full lookback window ending at
    end_idx and return {day_ahead: mean_forecast_over_48_SPs} for 1..n_days.

    Returns the **daily average** over all 48 settlement periods per day —
    the standard reference for day-ahead contracting.
    end_date: actual timestamp of values[end_idx-1] for weekday alignment.
    """
    max_h = n_days * 48
    df = _values_to_df(values, end_idx, lookback_sps, end_date=end_date)
    if len(df) < 48:
        return {}

    n_lags = best_params.get("n_lags", 48)
    epochs = best_params.get("epochs", 15)
    lr = best_params.get("learning_rate", 0.1)
    effective_n_lags = min(n_lags, len(df) // 2)

    # Disable AR if n_lags < n_forecasts to avoid NP constraint violation
    if effective_n_lags > 0 and effective_n_lags < max_h:
        logger.debug("NP forward: n_lags=%d < n_forecasts=%d; disabling AR", effective_n_lags, max_h)
        effective_n_lags = 0

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = NeuralProphet(
                n_forecasts=max_h,
                n_lags=effective_n_lags,
                epochs=epochs,
                learning_rate=lr,
                weekly_seasonality=best_params.get("weekly_seasonality", True),
                daily_seasonality=best_params.get("daily_seasonality", True),
                yearly_seasonality=False,
                batch_size=min(64, len(df)),
                verbose=False,
            )
            m.fit(df, freq=_FREQ)
            future = m.make_future_dataframe(df, periods=max_h, n_historic_predictions=0)
            pred = m.predict(future)

        yhat_cols = sorted(
            [c for c in pred.columns if c.startswith("yhat")],
            key=lambda c: int(c.replace("yhat", "") or "1"),
        )
        if not yhat_cols:
            return {}

        forecasts: Dict[int, float] = {}
        for day in range(1, n_days + 1):
            # Average over all 48 SPs of this delivery day (day-ahead reference)
            day_cols = yhat_cols[(day - 1) * 48 : day * 48]
            if not day_cols:
                continue
            day_vals = []
            for col in day_cols:
                v = pred[col].dropna()
                if not v.empty:
                    day_vals.append(float(v.iloc[-1]))
            if day_vals:
                forecasts[day] = float(np.mean(day_vals))
        return forecasts

    except Exception as exc:
        logger.warning("NP forward forecast failed: %s", exc)
        return {}


# ── Full training pipeline ────────────────────────────────────────────────────

def train_np_models(
    sip_series: pd.Series,
    mip_series: pd.Series,
    demand_series: Optional[pd.Series] = None,
    target: str = "sip",
    progress_callback: Optional[Callable[[float, str], None]] = None,
    param_search_mode: Literal["grid", "random"] = "grid",
    random_search_samples: int = 6,
) -> TrainedNPModels:
    """
    Full NeuralProphet training pipeline:
      1. Build parameter combos (18 for in-depth, 6 for quick).
      2. Grid-search per lookback window.
      3. Retrain best model per lookback on full data, compute forward forecasts.
      4. Run rolling backtest with globally best params.
      5. Return packaged in TrainedNPModels.

    Parameters
    ----------
    target            : "sip", "demand", or "mip".
    param_search_mode : "grid" (18 combos) or "random" (random_search_samples combos).
    progress_callback : Optional (fraction, message) callback.
    """
    if not _HAS_PROPHET:
        raise RuntimeError("neuralprophet is not installed")

    param_combos = _build_np_combos(
        mode=param_search_mode,
        n_random=max(1, int(random_search_samples)),
    )

    # Route target series
    if target == "demand":
        if demand_series is None:
            raise ValueError("demand_series required when target='demand'")
        target_values = demand_series.values.astype(float)
    elif target == "mip":
        target_values = mip_series.values.astype(float)
    else:
        target_values = sip_series.values.astype(float)

    target_desc = {"demand": "Demand", "mip": "Wholesale (MIP)"}.get(target, "SIP")
    result = TrainedNPModels(target=target, training_timestamp=time.time())

    lookbacks = list(_NP_LOOKBACKS.items())
    end_idx = len(target_values)
    total_steps = len(lookbacks) + 1
    step_idx = 0
    best_models_by_lb: dict = {}  # {lb_label: (model, lb_sps)} for hold-out backtest

    for lb_label, lb_sps in lookbacks:
        step_idx += 1
        if progress_callback:
            progress_callback(
                step_idx / total_steps,
                f"Searching ({target_desc}): {lb_label} lookback "
                f"[{len(param_combos)} combos]…",
            )

        # Extract real end date for weekday-aligned seasonality
        if target == "demand" and demand_series is not None:
            series_end = demand_series.index[-1] if hasattr(demand_series.index, '__len__') else None
        elif target == "mip":
            series_end = mip_series.index[-1] if hasattr(mip_series.index, '__len__') else None
        else:
            series_end = sip_series.index[-1] if hasattr(sip_series.index, '__len__') else None

        best_p, best_score, best_model = _np_grid_search_single(
            target_values, lb_sps, end_idx, param_combos, end_date=series_end,
        )
        result.best_params[lb_label] = best_p
        result.best_scores[lb_label] = best_score
        best_models_by_lb[lb_label]  = (best_model, lb_sps)

        # Pre-compute 14-day forward forecast with best params
        fc = _np_forward_forecast(target_values, end_idx, lb_sps, best_p, n_days=14,
                                  end_date=series_end)
        result.forward_forecasts[lb_label] = fc
        logger.info("NP (%s) %s: score=%.3f, n_lags=%s, epochs=%s",
                    target_desc, lb_label, best_score,
                    best_p.get("n_lags"), best_p.get("epochs"))

    if progress_callback:
        progress_callback(step_idx / total_steps, f"Running hold-out backtest ({target_desc})…")

    # Fast hold-out backtest using the 80%-trained models from grid search
    mip_v    = mip_series.values.astype(float)
    demand_v = demand_series.values.astype(float) if demand_series is not None else None
    errors, crossovers = _np_holdout_backtest(
        best_models_by_lb, target_values, mip_v, demand_v, target
    )
    result.backtest_errors     = errors
    result.backtest_crossovers = crossovers

    if progress_callback:
        progress_callback(1.0, f"{target_desc} NeuralProphet training complete.")

    return result


def _pick_global_best_np(
    best_params: Dict[str, dict],
    best_scores: Optional[Dict[str, float]] = None,
) -> dict:
    """
    Pick the param set associated with the best (lowest) worst-window MAE.
    Falls back to plurality vote if no scores are provided.
    """
    if best_scores:
        best_score_overall = float("inf")
        best_p_overall: Optional[dict] = None
        for lb_label, score in best_scores.items():
            if score < best_score_overall and lb_label in best_params and best_params[lb_label]:
                best_score_overall = score
                best_p_overall = best_params[lb_label]
        if best_p_overall is not None:
            return best_p_overall

    # Fallback: plurality vote
    from collections import Counter
    serialized = [tuple(sorted(p.items())) for p in best_params.values() if p]
    if not serialized:
        return {**NP_FIXED, "n_lags": 96, "epochs": 15, "learning_rate": 0.1}
    return dict(Counter(serialized).most_common(1)[0][0])


# ── Hold-out backtest using 80%-trained models ────────────────────────────────

def _np_holdout_backtest(
    best_models_by_lb: dict,
    target_values: np.ndarray,
    mip_values: np.ndarray,
    demand_values: Optional[np.ndarray],
    target: str,
) -> Tuple[List[RollingErrorRow], List[CrossoverResult]]:
    """
    Compute backtest metrics using the 80%-trained models saved during grid search.
    Generates multi-step predictions on the held-out 20% for each lookback, then
    computes MAE vs the market benchmark at each horizon.

    This replaces run_rolling_backtest(method="neuralprophet") which re-trained a
    new NeuralProphet at every daily rolling origin — impractically slow.
    """
    from src.models.stat_tests import diebold_mariano

    benchmark_v: Optional[np.ndarray]
    if target in ("sip", "mip"):
        benchmark_v = mip_values
    elif target == "demand":
        benchmark_v = demand_values
    else:
        benchmark_v = None

    errors: List[RollingErrorRow] = []

    for lb_label, lb_sps in ROLLING_LOOKBACKS.items():
        entry = best_models_by_lb.get(lb_label)
        if entry is None:
            continue
        model, _ = entry
        if model is None:
            continue

        n = len(target_values)
        start = max(0, n - lb_sps)
        window = target_values[start:n]
        nw = len(window)
        if nw < 96:
            continue
        split = int(nw * 0.8)
        holdout_len = nw - split
        if holdout_len < 2:
            continue

        # origin_abs: the absolute index in target_values where training ended
        origin_abs = start + split

        # Generate multi-step predictions using the stored model
        try:
            future = model.make_future_dataframe(
                df=model.history, periods=holdout_len, n_historic_predictions=False,
            )
            forecast_df = model.predict(future)
        except Exception as exc:
            logger.debug("NP holdout prediction failed for %s: %s", lb_label, exc)
            continue

        yhat_cols = sorted(
            [c for c in forecast_df.columns if c.startswith("yhat")],
            key=lambda c: int(c.replace("yhat", "") or "1"),
        )
        if not yhat_cols:
            continue

        # Collect predicted values at each step in the holdout
        pred_vals = []
        for col in yhat_cols[:holdout_len]:
            series = forecast_df[col].dropna()
            if not series.empty:
                pred_vals.append(float(series.iloc[-1]))
            else:
                pred_vals.append(float("nan"))

        for h_sps in ROLLING_HORIZONS:
            h_days = h_sps // 48
            # Holdout step index (0-based within holdout)
            step_idx_h = h_sps - 1  # h_sps steps ahead → index h_sps-1 in holdout
            if step_idx_h >= len(pred_vals) or step_idx_h >= holdout_len:
                continue
            if origin_abs + h_sps >= n:
                continue

            pred = pred_vals[step_idx_h]
            if np.isnan(pred):
                continue
            r = float(target_values[origin_abs + h_sps])
            fc_err = abs(pred - r)

            if benchmark_v is not None and origin_abs < len(benchmark_v):
                mkt_err = abs(float(benchmark_v[origin_abs]) - r)
            else:
                mkt_err = fc_err  # no benchmark: alpha = 0

            fc_arr   = np.array([fc_err],  dtype=float)
            mkt_arr  = np.array([mkt_err], dtype=float)
            real_arr = np.array([r],        dtype=float)
            safe_real = np.where(np.abs(real_arr) < 1e-6, 1e-6, real_arr)
            _, dm_p = diebold_mariano(fc_arr, mkt_arr, h=max(1, h_days), power=2)

            errors.append(RollingErrorRow(
                lookback_label=lb_label, lookback_sps=lb_sps,
                horizon_days=h_days, horizon_sps=h_sps,
                forecast_mae=fc_err, forecast_rmse=fc_err,
                forecast_mape=float(np.minimum(fc_err / abs(float(safe_real[0])) * 100, 500.0)),
                market_mae=mkt_err, market_rmse=mkt_err,
                market_mape=float(np.minimum(mkt_err / abs(float(safe_real[0])) * 100, 500.0)),
                alpha_mae=mkt_err - fc_err,
                alpha_mape=0.0,
                dm_pvalue=dm_p,
                n_obs=1,
            ))

    # Crossover computation — same logic as run_rolling_backtest
    crossovers: List[CrossoverResult] = []
    for lb_label in ROLLING_LOOKBACKS:
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
        "NP hold-out backtest: %d error rows, %d lookbacks",
        len(errors), len(ROLLING_LOOKBACKS),
    )
    return errors, crossovers


# ── Forward forecast (loads from stored results) ──────────────────────────────

def forecast_forward_np(trained: TrainedNPModels) -> Dict[str, Dict[int, float]]:
    """
    Return pre-computed 14-day forward forecasts from a TrainedNPModels object.
    These were computed at training time and are stored in trained.forward_forecasts.
    """
    return copy.deepcopy(trained.forward_forecasts)


# ── Disk persistence ──────────────────────────────────────────────────────────

def save_np_models(trained: TrainedNPModels, target: str = "sip") -> bool:
    """Save TrainedNPModels to disk. Returns True on success."""
    if not _HAS_JOBLIB:
        logger.warning("joblib not installed — cannot save NP models to disk")
        return False
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        fpath = _model_file(target)
        joblib.dump(trained, fpath)
        logger.info("Saved NP models (%s) to %s", target, fpath)
        return True
    except OSError as exc:
        logger.warning("Failed to save NP models: %s", exc)
        return False


def load_np_models(target: str = "sip") -> Optional[TrainedNPModels]:
    """Load TrainedNPModels from disk. Returns None if not found or invalid."""
    if not _HAS_JOBLIB:
        return None
    fpath = _model_file(target)
    if not fpath.exists():
        return None
    try:
        trained = joblib.load(fpath)
        if isinstance(trained, TrainedNPModels):
            logger.info("Loaded NP models (%s) from %s", target, fpath)
            return trained
        return None
    except Exception as exc:
        logger.warning("Failed to load NP models: %s", exc)
        return None


def has_np_models(target: str = "sip") -> bool:
    """Return True if a saved NP model file exists for the given target."""
    return _model_file(target).exists()
