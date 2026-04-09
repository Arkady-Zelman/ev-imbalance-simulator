"""
PyTorch LSTM forecaster for SIP, MIP, Demand, and Total Generation.

Each model consumes raw multi-channel time-series windows (no hand-engineered
features) — the LSTM learns temporal patterns directly from the data.  One
model is trained per (lookback, horizon) cell, matching the XGBoost structure
so the two can be ensembled via hybrid_forecaster.py.

Channels
--------
Channel 0 : target series (always present)
Channel 1 : MIP (when target = SIP)
Channel 2 : demand (when available)
Channels 3+: exogenous series (wind gen, temperature, etc.)

Sequence layout
---------------
Input:  (batch, seq_len, n_channels) — last seq_len SPs before the origin
Output: (batch,)                     — predicted value at origin + horizon
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    torch = None  # type: ignore[assignment]
    nn = None     # type: ignore[assignment]


# ── Model ─────────────────────────────────────────────────────────────────────

class LSTMForecaster(nn.Module if _HAS_TORCH else object):  # type: ignore[misc]
    """
    Two-layer LSTM with a single linear output head.

    Parameters
    ----------
    input_size  : Number of input channels per timestep.
    hidden_size : LSTM hidden state dimension (default 64).
    num_layers  : Number of stacked LSTM layers (default 2).
    dropout     : Dropout probability between LSTM layers (0 for num_layers=1).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required for LSTMForecaster")
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        # Dropout between LSTM layers only works when num_layers > 1
        _dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=_dropout,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)          # (batch, seq_len, hidden_size)
        return self.fc(out[:, -1, :]).squeeze(-1)  # (batch,) — last timestep


# ── NaN repair ────────────────────────────────────────────────────────────────

def _repair_nans(arr: np.ndarray) -> np.ndarray:
    """
    Forward-fill NaN along axis 0 (time), then backward-fill, then zero-fill.
    Works on 1-D or 2-D (time, channels) arrays.
    """
    if arr.ndim == 1:
        arr = arr.copy().astype(np.float32)
        mask = np.isnan(arr)
        if mask.any():
            idx = np.where(~mask, np.arange(len(arr)), 0)
            np.maximum.accumulate(idx, out=idx)
            arr = arr[idx]
            arr = np.where(np.isnan(arr), 0.0, arr)
        return arr
    # 2-D: repair each channel independently
    return np.stack([_repair_nans(arr[:, c]) for c in range(arr.shape[1])], axis=1)


# ── Sequence builders ─────────────────────────────────────────────────────────

def _stack_channels(
    target: np.ndarray,
    aux_channels: List[np.ndarray],
) -> np.ndarray:
    """
    Stack target + auxiliary arrays into a 2-D (time, C) channel matrix.
    All arrays must have the same length.  NaNs are repaired in-place.
    """
    arrays = [target] + [a for a in aux_channels if a is not None]
    stacked = np.stack(arrays, axis=1).astype(np.float32)  # (T, C)
    return _repair_nans(stacked)


def build_lstm_sequences(
    target: np.ndarray,
    origin_idx: int,
    lookback_sps: int,
    horizon_sps: int,
    seq_len: int,
    aux_channels: List[np.ndarray],
    scaler=None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Slide a window of *seq_len* SPs over the lookback window, building
    (input_sequence, target_scalar) pairs for supervised training.

    Parameters
    ----------
    target        : 1-D target series (full length).
    origin_idx    : The forecast origin — only data before this index is used.
    lookback_sps  : How far back from origin_idx to start the training window.
    horizon_sps   : Steps ahead to predict.
    seq_len       : LSTM input sequence length (SPs).
    aux_channels  : List of auxiliary series arrays (same length as target).
    scaler        : Fitted MinMaxScaler (or None).  If provided, transform is
                    applied; otherwise the raw values are returned.

    Returns
    -------
    X : float32 (N, seq_len, C) or None if insufficient data.
    y : float32 (N,)            or None.
    """
    _MIN_SAMPLES = 30

    channels = _stack_channels(target, aux_channels)   # (T, C)
    T = channels.shape[0]

    train_start = max(seq_len, origin_idx - lookback_sps)
    # Sequence i: input = channels[i-seq_len:i], label = target[i+horizon_sps]
    X_rows, y_rows = [], []
    for i in range(train_start, origin_idx):
        if i - seq_len < 0 or i + horizon_sps >= origin_idx:
            continue
        seq = channels[i - seq_len : i]          # (seq_len, C)
        y_val = float(target[i + horizon_sps])
        if np.isnan(y_val):
            continue
        X_rows.append(seq)
        y_rows.append(y_val)

    if len(X_rows) < _MIN_SAMPLES:
        return None, None

    X = np.stack(X_rows, axis=0).astype(np.float32)   # (N, seq_len, C)
    y = np.array(y_rows, dtype=np.float32)

    if scaler is not None:
        N, S, C = X.shape
        X_flat = X.reshape(-1, C)
        X_flat = scaler.transform(X_flat)
        X = X_flat.reshape(N, S, C)

    return X, y


def build_lstm_inference_input(
    target: np.ndarray,
    idx: int,
    seq_len: int,
    aux_channels: List[np.ndarray],
    scaler=None,
) -> Optional[np.ndarray]:
    """
    Build a single (1, seq_len, C) inference tensor for timestep *idx*.
    Returns None if there is insufficient history.
    """
    if idx < seq_len:
        return None

    channels = _stack_channels(target, aux_channels)   # (T, C)
    seq = channels[idx - seq_len : idx]                # (seq_len, C)

    if scaler is not None:
        N, C = seq.shape
        seq = scaler.transform(seq)

    return seq[np.newaxis].astype(np.float32)          # (1, seq_len, C)
