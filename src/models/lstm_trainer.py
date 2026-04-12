"""
LSTM training pipeline — mirrors xgb_trainer.py in public interface.

Workflow
--------
1. train_lstm_models()     — random hyperparameter search, retrain best model,
                             run rolling backtest, return TrainedLSTMModels.
2. save/load_trained_lstm_models() — joblib-based disk I/O.
3. forecast_forward()      — 14-day-ahead point forecasts.
4. forecast_intraday_48sp() — true per-SP intraday using the offset trick.

The TrainedLSTMModels dataclass stores PyTorch state_dicts (plain dicts of
tensors) rather than live nn.Module objects, which makes joblib serialisation
reliable.  On inference the model is reconstructed from `model_configs` and
the state_dict is loaded.
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
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    torch = None   # type: ignore[assignment]
    nn = None      # type: ignore[assignment]

try:
    import joblib
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False
    joblib = None  # type: ignore[assignment]

try:
    from sklearn.preprocessing import MinMaxScaler
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    MinMaxScaler = None  # type: ignore[assignment]

from src.models.lstm_forecaster import (
    LSTMForecaster,
    build_lstm_inference_input,
    build_lstm_sequences,
)
from src.models.rolling_backtest import (
    ROLLING_HORIZONS,
    ROLLING_LOOKBACKS,
    CrossoverResult,
    RollingErrorRow,
)

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".cache" / "lstm_models"


def _model_file(target: str = "sip") -> Path:
    return _CACHE_DIR / f"trained_lstm_{target}.joblib"


# ── Hyperparameter search space ───────────────────────────────────────────────

LSTM_GRID = {
    "hidden_size": [32, 64, 128],
    "num_layers":  [1, 2, 3, 4],
    "dropout":     [0.1, 0.2, 0.3],
    "lr":          [0.001, 0.005, 0.01],
    "batch_size":  [16, 32, 64],
    "seq_len":     [96, 192, 336],      # 2d / 4d / 7d in 30-min SPs
}

_LSTM_N_SAMPLES       = {"grid": 15, "random": 5}
_LSTM_SEARCH_EPOCHS   = 20    # fast per-combo training during search
_LSTM_FINAL_EPOCHS    = 50    # full training for winning params
_LSTM_PATIENCE        = 5     # early stopping patience (validation loss)

# Representative horizons — 7 pivot days so the step-function is at most 2 days
# wide instead of 7. LSTM is slower, so we skip every other day but keep all
# endpoints (1, 2, 4, 7, 10, 12, 14).  forecast_forward still maps each
# requested day to the closest trained horizon.
REPRESENTATIVE_HORIZONS       = [48 * d for d in (1, 2, 4, 7, 10, 12, 14)]
REPRESENTATIVE_HORIZONS_QUICK = [48, 48 * 7]     # 1d, 7d (quick mode)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class TrainedLSTMModels:
    target:             str = "sip"
    best_params:        Dict[str, Dict[int, dict]]          = field(default_factory=dict)
    best_scores:        Dict[str, Dict[int, float]]         = field(default_factory=dict)
    # PyTorch state_dicts stored as plain dicts of numpy arrays (joblib-safe)
    final_state_dicts:  Dict[str, Dict[int, dict]]          = field(default_factory=dict)
    # Architecture params needed to reconstruct LSTMForecaster at inference
    model_configs:      Dict[str, Dict[int, dict]]          = field(default_factory=dict)
    # Fitted MinMaxScaler per (lookback, horizon) cell
    scalers:            Dict[str, Dict[int, Any]]           = field(default_factory=dict)
    training_timestamp: float                               = 0.0
    backtest_errors:    List[RollingErrorRow]               = field(default_factory=list)
    backtest_crossovers: List[CrossoverResult]              = field(default_factory=list)
    # Diagnostic: [(params, val_score), ...] per cell
    grid_search_history: Dict[str, Dict[int, List[tuple]]] = field(default_factory=dict)
    # In-sample Huber loss of best combo (overfitting gap)
    train_scores:       Dict[str, Dict[int, float]]        = field(default_factory=dict)
    seq_len:            int                                = 336
    exog_keys:          List[str]                          = field(default_factory=list)


@dataclass
class ParallelTrainingResult:
    models:          Dict[str, TrainedLSTMModels]
    errors:          Dict[str, str]
    elapsed_seconds: float


# ── Device detection ──────────────────────────────────────────────────────────

def _get_device() -> "torch.device":
    if _HAS_TORCH and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _state_dict_to_numpy(sd: dict) -> dict:
    """Convert a PyTorch state_dict to a plain dict of numpy arrays (joblib-safe)."""
    return {k: v.cpu().numpy() for k, v in sd.items()}


def _numpy_to_state_dict(nd: dict) -> dict:
    """Convert a numpy state_dict back to float32 tensors."""
    return {k: torch.from_numpy(v.astype(np.float32)) for k, v in nd.items()}


def _rebuild_model(config: dict, state_numpy: dict) -> "LSTMForecaster":
    """Reconstruct an LSTMForecaster from its config and numpy state_dict."""
    model = LSTMForecaster(**config)
    model.load_state_dict(_numpy_to_state_dict(state_numpy))
    model.eval()
    return model


def _fit_scaler(X_train: np.ndarray):
    """Fit and return a MinMaxScaler on the (N, seq_len, C) training tensor."""
    if not _HAS_SKLEARN:
        return None
    N, S, C = X_train.shape
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(X_train.reshape(-1, C))
    return scaler


def _apply_scaler(X: np.ndarray, scaler) -> np.ndarray:
    """Apply a fitted scaler to a (N, seq_len, C) or (1, seq_len, C) array."""
    if scaler is None:
        return X
    N, S, C = X.shape
    return scaler.transform(X.reshape(-1, C)).reshape(N, S, C).astype(np.float32)


def _generate_combos(n: int, seed: int = 42) -> List[dict]:
    rng = random.Random(seed)
    return [{k: rng.choice(v) for k, v in LSTM_GRID.items()} for _ in range(n)]


# ── Target routing (mirrors xgb_trainer._route_series) ───────────────────────

def _route_series(
    target: str,
    sip_v: np.ndarray,
    mip_v: np.ndarray,
    demand_v: Optional[np.ndarray],
    gen_v: Optional[np.ndarray],
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Returns (target_array, aux_channels_list).
    aux_channels are additional LSTM input channels (no feature engineering).
    """
    if target == "demand":
        if demand_v is None:
            raise ValueError("demand_series required when target='demand'")
        return demand_v, [sip_v, mip_v]
    if target == "mip":
        return mip_v, [sip_v, demand_v] if demand_v is not None else [sip_v]
    if target == "total_generation":
        if gen_v is None:
            raise ValueError("gen_series required when target='total_generation'")
        return gen_v, [demand_v] if demand_v is not None else []
    # default: sip
    aux = [mip_v]
    if demand_v is not None:
        aux.append(demand_v)
    return sip_v, aux


def _align_to_target(target: pd.Series, exog: pd.Series) -> np.ndarray:
    aligned = exog.reindex(target.index).ffill(limit=4)
    fill_val = float(aligned.mean()) if not pd.isna(aligned.mean()) else 0.0
    return aligned.fillna(fill_val).to_numpy(dtype=np.float32)


# ── Single-cell training (grid search + final model) ─────────────────────────

def _train_single_epoch(
    model: "LSTMForecaster",
    loader: "DataLoader",
    optimizer: "torch.optim.Optimizer",
    criterion: "nn.HuberLoss",
    device: "torch.device",
) -> float:
    model.train()
    total = 0.0
    n = 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * len(y_batch)
        n += len(y_batch)
    return total / max(n, 1)


def _eval_loss(
    model: "LSTMForecaster",
    X: np.ndarray,
    y: np.ndarray,
    criterion: "nn.HuberLoss",
    device: "torch.device",
) -> float:
    model.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(X).to(device)
        yt = torch.from_numpy(y).to(device)
        return float(criterion(model(Xt), yt).item())


def _eval_mae(
    model: "LSTMForecaster",
    X: np.ndarray,
    y: np.ndarray,
    device: "torch.device",
) -> float:
    """MAE in original units (£/MWh or MW) — used for hybrid weight computation."""
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X).to(device)).cpu().numpy()
    return float(np.mean(np.abs(preds - y)))


def _grid_search_single_lstm(
    X: np.ndarray,
    y: np.ndarray,
    param_combos: List[dict],
    n_epochs: int = _LSTM_SEARCH_EPOCHS,
) -> Tuple[dict, float, float, List[tuple]]:
    """
    Time-series 80/20 split → train each combo → select by val Huber loss,
    but return val MAE in original units as best_score so it is on the same
    scale as XGBoost's worst-window MAE for hybrid weight computation.

    Returns (best_params, best_val_mae, best_train_mae, history).
    history entries store Huber loss for diagnostics (relative ranking only).
    """
    device = _get_device()
    n = len(y)
    split = int(n * 0.8)
    if split < 10 or n - split < 5:
        return param_combos[0], float("inf"), float("inf"), []

    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    criterion  = nn.HuberLoss()
    best_huber = float("inf")   # used only for model selection
    best_params = param_combos[0]
    best_model  = None
    history: List[tuple] = []

    input_size = X.shape[2]

    for params in param_combos:
        try:
            model = LSTMForecaster(
                input_size=input_size,
                hidden_size=params["hidden_size"],
                num_layers=params["num_layers"],
                dropout=params["dropout"],
            ).to(device)
            opt = torch.optim.Adam(
                model.parameters(),
                lr=params["lr"],
                weight_decay=1e-4,
            )
            ds = TensorDataset(
                torch.from_numpy(X_tr),
                torch.from_numpy(y_tr),
            )
            loader = DataLoader(ds, batch_size=params["batch_size"], shuffle=False)

            best_val_huber = float("inf")
            patience_ctr = 0
            for _ in range(n_epochs):
                _train_single_epoch(model, loader, opt, criterion, device)
                val_loss = _eval_loss(model, X_val, y_val, criterion, device)
                if val_loss < best_val_huber - 1e-6:
                    best_val_huber = val_loss
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                    if patience_ctr >= _LSTM_PATIENCE:
                        break

            history.append((params, best_val_huber))
            if best_val_huber < best_huber:
                best_huber  = best_val_huber
                best_params = params
                best_model  = model
        except Exception as exc:
            logger.debug("LSTM combo failed: %s", exc)
            history.append((params, float("inf")))

    # Convert winning model's score to MAE in original units for hybrid weighting
    if best_model is not None:
        best_val_mae   = _eval_mae(best_model, X_val, y_val, device)
        best_train_mae = _eval_mae(best_model, X_tr,  y_tr,  device)
    else:
        best_val_mae   = float("inf")
        best_train_mae = float("inf")

    return best_params, best_val_mae, best_train_mae, history


def _train_final_model(
    X: np.ndarray,
    y: np.ndarray,
    params: dict,
    n_epochs: int = _LSTM_FINAL_EPOCHS,
) -> Tuple["LSTMForecaster", float]:
    """Train the winning params on all data for _LSTM_FINAL_EPOCHS epochs."""
    device    = _get_device()
    criterion = nn.HuberLoss()
    model = LSTMForecaster(
        input_size=X.shape[2],
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=1e-4)
    ds  = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(ds, batch_size=params["batch_size"], shuffle=False)

    best_loss = float("inf")
    patience_ctr = 0
    for _ in range(n_epochs):
        loss = _train_single_epoch(model, loader, opt, criterion, device)
        if loss < best_loss - 1e-6:
            best_loss = loss
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= _LSTM_PATIENCE * 2:
                break

    model.eval()
    return model, best_loss


# ── Full training pipeline ────────────────────────────────────────────────────

def train_lstm_models(
    sip_series: pd.Series,
    mip_series: pd.Series,
    demand_series: Optional[pd.Series] = None,
    target: str = "sip",
    progress_callback: Optional[Callable[[float, str], None]] = None,
    param_search_mode: Literal["grid", "random"] = "grid",
    gen_series: Optional[pd.Series] = None,
    selected_lookback: Optional[str] = None,
    exog_series: Optional[Dict[str, pd.Series]] = None,
) -> TrainedLSTMModels:
    """
    Full LSTM training pipeline:
      1. Build hyperparameter combos.
      2. Grid-search per (lookback, representative_horizon).
      3. Retrain final model with best params on full window.
      4. Run rolling backtest (predict-only).
      5. Return TrainedLSTMModels.
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch (torch) is not installed. Run: pip install torch")

    n_samples   = _LSTM_N_SAMPLES[param_search_mode]
    param_combos = _generate_combos(n_samples)

    sip_v    = sip_series.values.astype(np.float32)
    mip_v    = mip_series.values.astype(np.float32)
    demand_v = demand_series.values.astype(np.float32) if demand_series is not None else None
    gen_v    = gen_series.values.astype(np.float32)   if gen_series   is not None else None

    fc_target, aux_channels = _route_series(target, sip_v, mip_v, demand_v, gen_v)

    # Align exog series and append as extra channels
    target_series_for_align = {
        "demand": demand_series, "mip": mip_series, "total_generation": gen_series,
    }.get(target, sip_series)
    if target_series_for_align is None:
        target_series_for_align = sip_series
    exog_arrays: List[np.ndarray] = []
    if exog_series:
        for s in exog_series.values():
            exog_arrays.append(_align_to_target(target_series_for_align, s))
    all_aux = aux_channels + exog_arrays

    target_desc = {
        "demand": "Demand", "mip": "Wholesale (MIP)", "total_generation": "Total Generation",
    }.get(target, "SIP")

    result = TrainedLSTMModels(target=target, training_timestamp=time.time())
    result.exog_keys = list(exog_series.keys()) if exog_series else []

    if selected_lookback and selected_lookback in ROLLING_LOOKBACKS:
        lookbacks = [(selected_lookback, ROLLING_LOOKBACKS[selected_lookback])]
    else:
        lookbacks = list(ROLLING_LOOKBACKS.items())

    rep_horizons = REPRESENTATIVE_HORIZONS_QUICK if param_search_mode == "random" else REPRESENTATIVE_HORIZONS
    total_steps  = len(lookbacks) * len(rep_horizons) + 1
    step_idx     = 0

    for lb_label, lb_sps in lookbacks:
        result.best_params[lb_label]   = {}
        result.best_scores[lb_label]   = {}
        result.final_state_dicts[lb_label] = {}
        result.model_configs[lb_label] = {}
        result.scalers[lb_label]       = {}

        for h_sps in rep_horizons:
            step_idx += 1
            h_days = h_sps // 48
            if progress_callback:
                progress_callback(
                    step_idx / total_steps,
                    f"LSTM search ({target_desc}): {lb_label} lookback, {h_days}d horizon "
                    f"[{n_samples} combos × {_LSTM_SEARCH_EPOCHS} epochs]…",
                )

            origin_idx = len(fc_target)

            # Try each seq_len in the grid — must fit within lookback
            # We'll use a default of 336 for data building; scaler fitted per cell
            X, y = build_lstm_sequences(
                fc_target, origin_idx, lb_sps, h_sps,
                seq_len=max(p["seq_len"] for p in param_combos),
                aux_channels=all_aux,
            )
            if X is None:
                logger.info("Insufficient data for LSTM %s / %dd — skipping", lb_label, h_days)
                continue

            scaler = _fit_scaler(X)
            X_scaled = _apply_scaler(X, scaler)
            result.scalers[lb_label][h_sps] = scaler

            best_p, best_score, train_score, history = _grid_search_single_lstm(
                X_scaled, y, param_combos
            )
            result.best_params[lb_label][h_sps]  = best_p
            result.best_scores[lb_label][h_sps]  = best_score
            result.train_scores.setdefault(lb_label, {})[h_sps] = train_score
            result.grid_search_history.setdefault(lb_label, {})[h_sps] = history

            # Re-build data with best seq_len and retrain final model
            X_final, y_final = build_lstm_sequences(
                fc_target, origin_idx, lb_sps, h_sps,
                seq_len=best_p["seq_len"],
                aux_channels=all_aux,
            )
            if X_final is None:
                continue
            scaler_final = _fit_scaler(X_final)
            X_final_scaled = _apply_scaler(X_final, scaler_final)
            result.scalers[lb_label][h_sps] = scaler_final

            final_model, _ = _train_final_model(X_final_scaled, y_final, best_p)

            config = {
                "input_size":   X_final.shape[2],
                "hidden_size":  best_p["hidden_size"],
                "num_layers":   best_p["num_layers"],
                "dropout":      best_p["dropout"],
            }
            result.final_state_dicts[lb_label][h_sps] = _state_dict_to_numpy(
                final_model.state_dict()
            )
            result.model_configs[lb_label][h_sps] = config
            result.seq_len = best_p["seq_len"]

    if progress_callback:
        progress_callback(step_idx / total_steps, f"Running LSTM backtest ({target_desc})…")

    errors, crossovers = _backtest_with_final_models(
        result, sip_series, mip_series, demand_series, gen_series, target, exog_series
    )
    result.backtest_errors     = errors
    result.backtest_crossovers = crossovers

    if progress_callback:
        progress_callback(1.0, f"LSTM {target_desc} training complete.")

    return result


# ── Fast backtest (predict-only) ─────────────────────────────────────────────

def _backtest_with_final_models(
    trained: TrainedLSTMModels,
    sip_series: pd.Series,
    mip_series: pd.Series,
    demand_series: Optional[pd.Series],
    gen_series: Optional[pd.Series],
    target: str,
    exog_series: Optional[Dict[str, pd.Series]] = None,
) -> Tuple[List[RollingErrorRow], List[CrossoverResult]]:
    from src.models.forecaster import _extract_market_forward, _extract_realised
    from src.models.stat_tests import diebold_mariano

    sip_v    = sip_series.values.astype(np.float32)
    mip_v    = mip_series.values.astype(np.float32)
    demand_v = demand_series.values.astype(np.float32) if demand_series is not None else None
    gen_v    = (gen_series.reindex(sip_series.index).ffill().bfill().values.astype(np.float32)
                if gen_series is not None else None)

    fc_target, aux_channels = _route_series(target, sip_v, mip_v, demand_v, gen_v)

    target_series_for_align = {
        "demand": demand_series, "mip": mip_series, "total_generation": gen_series,
    }.get(target, sip_series)
    if target_series_for_align is None:
        target_series_for_align = sip_series
    exog_arrays: List[np.ndarray] = []
    if exog_series:
        for s in exog_series.values():
            exog_arrays.append(_align_to_target(target_series_for_align, s))
    all_aux = aux_channels + exog_arrays

    if target == "total_generation":
        benchmark_v = None
    elif target == "demand":
        benchmark_v = demand_v
    else:
        benchmark_v = mip_v

    device = _get_device()
    n = len(fc_target)
    step = 48
    errors: List[RollingErrorRow] = []

    for lb_label, lb_sps in ROLLING_LOOKBACKS.items():
        if lb_label not in trained.final_state_dicts or not trained.final_state_dicts[lb_label]:
            continue
        lb_models  = trained.final_state_dicts[lb_label]
        lb_configs = trained.model_configs[lb_label]
        lb_scalers = trained.scalers.get(lb_label, {})
        available_h = sorted(lb_models.keys())

        for h_sps in ROLLING_HORIZONS:
            h_days    = h_sps // 48
            closest_h = min(available_h, key=lambda h: abs(h - h_sps))
            state_np  = lb_models[closest_h]
            config    = lb_configs[closest_h]
            scaler    = lb_scalers.get(closest_h)
            seq_len   = trained.seq_len

            model = _rebuild_model(config, state_np).to(device)

            start_idx = max(seq_len + closest_h, lb_sps)
            end_idx   = n - h_sps
            if start_idx >= end_idx:
                continue

            fc_errs, mkt_errs, realised_vals = [], [], []

            for idx in range(start_idx, end_idx, step):
                x_seq = build_lstm_inference_input(fc_target, idx, seq_len, all_aux)
                if x_seq is None:
                    continue
                x_seq = _apply_scaler(x_seq, scaler)
                with torch.no_grad():
                    xt = torch.from_numpy(x_seq).to(device)
                    pred = float(model(xt).cpu().item())

                realised_d = _extract_realised(fc_target, idx, [h_sps])
                if h_sps not in realised_d:
                    continue
                r = realised_d[h_sps]
                fc_errs.append(abs(pred - r))
                realised_vals.append(r)

                if target == "total_generation":
                    pi = idx + h_sps - 48
                    if pi >= 0:
                        mkt_errs.append(abs(float(fc_target[pi]) - r))
                elif benchmark_v is not None:
                    mkt_d = _extract_market_forward(benchmark_v, idx, [h_sps], lookback_sps=lb_sps)
                    if h_sps in mkt_d:
                        mkt_errs.append(abs(mkt_d[h_sps] - r))

            if len(fc_errs) < 5:
                continue
            if not mkt_errs:
                mkt_errs = list(fc_errs)

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

    crossovers: List[CrossoverResult] = []
    present_lbs = {e.lookback_label for e in errors}
    for lb_label in present_lbs:
        lb_rows = sorted(
            [e for e in errors if e.lookback_label == lb_label],
            key=lambda e: e.horizon_days,
        )
        crossover_day, last_pos, first_neg = 15, 0.0, 0.0
        for i, row in enumerate(lb_rows):
            if row.alpha_mae < 0:
                crossover_day = row.horizon_days
                first_neg     = row.alpha_mae
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

    logger.info("LSTM fast backtest: %d error rows", len(errors))
    return errors, crossovers


# ── Forward forecasting ───────────────────────────────────────────────────────

def forecast_forward(
    trained: TrainedLSTMModels,
    sip_values: np.ndarray,
    mip_values: Optional[np.ndarray] = None,
    demand_values: Optional[np.ndarray] = None,
    n_days: int = 14,
    gen_values: Optional[np.ndarray] = None,
    exog_dict: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Dict[int, float]]:
    """
    Produce forward forecasts from the end of available data.
    Returns {lookback_label: {day: predicted_value}}.
    """
    if not _HAS_TORCH:
        return {}

    target = getattr(trained, "target", "sip")
    try:
        fc_target, aux_channels = _route_series(
            target, sip_values,
            mip_values if mip_values is not None else np.zeros_like(sip_values),
            demand_values, gen_values,
        )
    except ValueError:
        return {}

    exog_arrays = list(exog_dict.values()) if exog_dict else []
    all_aux = aux_channels + exog_arrays

    device     = _get_device()
    origin_idx = len(fc_target) - 1
    seq_len    = trained.seq_len
    result: Dict[str, Dict[int, float]] = {}

    for lb_label, lb_state_dicts in trained.final_state_dicts.items():
        available_h = sorted(lb_state_dicts.keys())
        if not available_h:
            continue
        lb_forecasts: Dict[int, float] = {}

        for day in range(1, n_days + 1):
            h_sps     = day * 48
            closest_h = min(available_h, key=lambda h: abs(h - h_sps))
            state_np  = lb_state_dicts[closest_h]
            config    = trained.model_configs[lb_label][closest_h]
            scaler    = trained.scalers.get(lb_label, {}).get(closest_h)

            x_seq = build_lstm_inference_input(fc_target, origin_idx, seq_len, all_aux)
            if x_seq is None:
                continue
            x_seq = _apply_scaler(x_seq, scaler)

            model = _rebuild_model(config, state_np).to(device)
            with torch.no_grad():
                xt = torch.from_numpy(x_seq).to(device)
                lb_forecasts[day] = float(model(xt).cpu().item())

        result[lb_label] = lb_forecasts

    return result


def forecast_intraday_48sp(
    trained: TrainedLSTMModels,
    sip_values: np.ndarray,
    mip_values: Optional[np.ndarray] = None,
    demand_values: Optional[np.ndarray] = None,
    gen_values: Optional[np.ndarray] = None,
    exog_dict: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """
    True 48-SP intraday day-ahead forecast using the offset trick.

    For SP s (0-47):
        virtual_idx = end_idx - 48 + s
        Build sequence ending at virtual_idx → predict s SPs into tomorrow.
    """
    if not _HAS_TORCH:
        return {}

    target = getattr(trained, "target", "sip")
    try:
        fc_target, aux_channels = _route_series(
            target, sip_values,
            mip_values if mip_values is not None else np.zeros_like(sip_values),
            demand_values, gen_values,
        )
    except ValueError:
        return {}

    exog_arrays = list(exog_dict.values()) if exog_dict else []
    all_aux = aux_channels + exog_arrays

    device  = _get_device()
    seq_len = trained.seq_len
    end_idx = len(fc_target)
    result_48: Dict[str, np.ndarray] = {}

    for lb_label, lb_state_dicts in trained.final_state_dicts.items():
        if not lb_state_dicts:
            continue
        h_key     = 48 if 48 in lb_state_dicts else min(lb_state_dicts.keys())
        state_np  = lb_state_dicts[h_key]
        config    = trained.model_configs[lb_label][h_key]
        scaler    = trained.scalers.get(lb_label, {}).get(h_key)

        model = _rebuild_model(config, state_np).to(device)
        sp_fc = np.full(48, np.nan, dtype=np.float32)

        for sp in range(48):
            virtual_idx = end_idx - 48 + sp
            if virtual_idx < seq_len:
                continue
            x_seq = build_lstm_inference_input(fc_target, virtual_idx, seq_len, all_aux)
            if x_seq is None:
                continue
            x_seq = _apply_scaler(x_seq, scaler)
            with torch.no_grad():
                xt = torch.from_numpy(x_seq).to(device)
                sp_fc[sp] = float(model(xt).cpu().item())

        result_48[lb_label] = sp_fc

    return result_48


# ── Parallel multi-target training ───────────────────────────────────────────

from concurrent.futures import ThreadPoolExecutor as _TPE, as_completed as _as_completed


def train_all_lstm_parallel(
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
    """Train up to 4 LSTM models simultaneously using ThreadPoolExecutor."""
    if targets is None:
        targets = ["sip", "mip"]
        if demand_series is not None:
            targets.append("demand")
        if gen_series is not None:
            targets.append("total_generation")

    cbs: Dict[str, Callable] = progress_callbacks or {}
    t0 = time.time()
    results: Dict[str, TrainedLSTMModels] = {}
    errs:    Dict[str, str] = {}

    def _train_one(tgt: str) -> TrainedLSTMModels:
        return train_lstm_models(
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
                save_trained_lstm_models(trained, target=tgt)
                results[tgt] = trained
            except Exception as exc:
                errs[tgt] = str(exc)
                logger.error("Parallel LSTM training failed for %s: %s", tgt, exc)

    return ParallelTrainingResult(
        models=results,
        errors=errs,
        elapsed_seconds=time.time() - t0,
    )


# ── Persistence ───────────────────────────────────────────────────────────────

def save_trained_lstm_models(trained: TrainedLSTMModels, target: str = "sip") -> bool:
    if not _HAS_JOBLIB:
        logger.warning("joblib not installed — cannot save LSTM models")
        return False
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(trained, _model_file(target))
        logger.info("LSTM model saved to %s", _model_file(target))
        return True
    except Exception as exc:
        logger.error("Failed to save LSTM model: %s", exc)
        return False


def load_trained_lstm_models(target: str = "sip") -> Optional[TrainedLSTMModels]:
    path = _model_file(target)
    if not path.exists():
        return None
    if not _HAS_JOBLIB:
        return None
    try:
        trained = joblib.load(path)
        if not isinstance(trained, TrainedLSTMModels):
            return None
        return trained
    except Exception as exc:
        logger.warning("Failed to load LSTM model (%s): %s", target, exc)
        return None


def has_trained_lstm_models(target: str = "sip") -> bool:
    return _model_file(target).exists()
