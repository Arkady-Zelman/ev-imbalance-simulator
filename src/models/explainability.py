"""Lightweight explainability helpers for XGBoost and LSTM models."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    torch = None  # type: ignore[assignment]

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    xgb = None  # type: ignore[assignment]


def _sample_rows(X: np.ndarray, max_samples: int = 256) -> np.ndarray:
    if X.ndim == 0 or X.shape[0] == 0:
        return X
    if X.shape[0] <= max_samples:
        return X
    idx = np.linspace(0, X.shape[0] - 1, num=max_samples, dtype=int)
    return X[idx]


def normalise_importance_map(values: Dict[str, float]) -> Dict[str, float]:
    if not values:
        return {}
    total = float(sum(max(v, 0.0) for v in values.values()))
    if total <= 0:
        return {k: 0.0 for k in values}
    return {k: float(max(v, 0.0) / total) for k, v in values.items()}


def average_importance_maps(maps: Iterable[Dict[str, float]]) -> Dict[str, float]:
    acc: Dict[str, list[float]] = {}
    for importance_map in maps:
        for key, value in importance_map.items():
            acc.setdefault(key, []).append(float(value))
    if not acc:
        return {}
    averaged = {key: float(np.mean(vals)) for key, vals in acc.items()}
    return normalise_importance_map(averaged)


def compute_xgb_native_shap_importance(
    model,
    X: np.ndarray,
    feature_names: Sequence[str],
    max_samples: int = 256,
) -> Dict[str, float]:
    """
    Compute a normalised mean-absolute SHAP summary using XGBoost's native
    `pred_contribs=True` path, so no external SHAP dependency is required.
    """
    if not _HAS_XGB or model is None or X is None or len(feature_names) == 0:
        return {}
    if X.ndim != 2 or X.shape[0] == 0:
        return {}

    X_sample = _sample_rows(np.asarray(X, dtype=np.float32), max_samples=max_samples)
    if X_sample.shape[1] != len(feature_names):
        return {}

    try:
        booster = model.get_booster()
        dmat = xgb.DMatrix(X_sample, feature_names=list(feature_names))
        contribs = booster.predict(dmat, pred_contribs=True, validate_features=False)
    except Exception:
        return {}

    contribs = np.asarray(contribs, dtype=np.float64)
    if contribs.ndim != 2 or contribs.shape[1] < len(feature_names):
        return {}

    # Final column is the bias term when pred_contribs=True.
    contribs = contribs[:, :len(feature_names)]
    mean_abs = np.mean(np.abs(contribs), axis=0)
    values = {feature_names[i]: float(mean_abs[i]) for i in range(len(feature_names))}
    return normalise_importance_map(values)


def compute_lstm_integrated_gradients_importance(
    model,
    X: np.ndarray,
    feature_names: Sequence[str],
    max_samples: int = 64,
    steps: int = 16,
) -> Dict[str, float]:
    """
    Approximate Integrated Gradients for a regression LSTM and aggregate absolute
    attribution across batch and time dimensions to produce per-channel scores.
    """
    if not _HAS_TORCH or model is None or X is None or len(feature_names) == 0:
        return {}
    if X.ndim != 3 or X.shape[0] == 0:
        return {}
    if X.shape[2] != len(feature_names):
        return {}

    X_sample = _sample_rows(np.asarray(X, dtype=np.float32), max_samples=max_samples)
    device = next(model.parameters()).device
    inputs = torch.from_numpy(X_sample).to(device)
    baseline = torch.zeros_like(inputs)

    model.eval()
    total_grads = torch.zeros_like(inputs)
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=device)[1:]

    for alpha in alphas:
        scaled = (baseline + alpha * (inputs - baseline)).detach().requires_grad_(True)
        outputs = model(scaled)
        grads = torch.autograd.grad(outputs.sum(), scaled, retain_graph=False)[0]
        total_grads = total_grads + grads.detach()

    avg_grads = total_grads / max(len(alphas), 1)
    integrated = (inputs - baseline) * avg_grads
    mean_abs = integrated.abs().mean(dim=(0, 1)).detach().cpu().numpy()
    values = {feature_names[i]: float(mean_abs[i]) for i in range(len(feature_names))}
    return normalise_importance_map(values)
