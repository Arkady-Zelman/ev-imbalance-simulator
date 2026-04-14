"""
MIP Inference — load trained LSTM and predict the next N settlement periods.

Usage
-----
    from models.lstm_mip.predict import load_artifacts, predict

    model, meta = load_artifacts()
    forecast_series = predict(df_recent, model, meta, horizon=48)

`df_recent` must be a DataFrame with at least `meta['lookback']` rows and
the exact columns listed in `meta['feature_cols']` (SIP already excluded).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent


# ── Model definition (must match what was trained) ───────────────────────────

class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm   = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True, dropout=0.2)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :]).squeeze(-1)


# ── Load ─────────────────────────────────────────────────────────────────────

def load_artifacts(
    weights_path: str | Path = _HERE / "lstm_mip.pth",
    scaler_path:  str | Path = _HERE / "lstm_mip_scaler.pkl",
    device: str = "cpu",
) -> Tuple[LSTM, dict]:
    """
    Load model weights and scaler metadata.

    Returns
    -------
    model : LSTM
        Trained model in eval mode.
    meta : dict
        Keys: scaler, target_col, feature_cols, input_size, lookback
    """
    with open(scaler_path, "rb") as f:
        meta = pickle.load(f)

    model = LSTM(input_size=meta["input_size"]).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    return model, meta


# ── Predict ──────────────────────────────────────────────────────────────────

def predict(
    df_recent: pd.DataFrame,
    model: LSTM,
    meta: dict,
    horizon: int = 48,
    device: str = "cpu",
) -> pd.Series:
    """
    Autoregressively forecast MIP for the next `horizon` settlement periods.

    Parameters
    ----------
    df_recent : pd.DataFrame
        Recent data with at least `meta['lookback']` rows.
        Must contain exactly the columns in `meta['feature_cols']`.
    model : LSTM
        Loaded model (from load_artifacts).
    meta : dict
        Loaded metadata (from load_artifacts).
    horizon : int
        Number of 30-min steps to forecast (default 48 = 24 hours).
    device : str
        'cpu' or 'cuda'.

    Returns
    -------
    pd.Series
        Forecast MIP values (GBP/MWh) with a 30-min DatetimeIndex.
    """
    scaler     = meta["scaler"]
    target_col = meta["target_col"]
    lookback   = meta["lookback"]
    feat_cols  = meta["feature_cols"]

    # Align columns to training order
    df_recent = df_recent[feat_cols].copy()

    if len(df_recent) < lookback:
        raise ValueError(
            f"df_recent has {len(df_recent)} rows but lookback={lookback} rows are required."
        )

    # Scale using the fitted scaler
    scaled = scaler.transform(df_recent.values).astype("float32")
    window = scaled[-lookback:].copy()  # (lookback, features)

    preds_scaled = []
    model.eval()
    with torch.no_grad():
        for step in range(horizon):
            x    = torch.from_numpy(window[np.newaxis]).to(device)  # (1, lookback, features)
            pred = model(x).cpu().item()
            preds_scaled.append(pred)

            # Next row: exogenous features from 24h ago, MIP = prediction
            next_row              = window[-lookback].copy()
            next_row[target_col]  = pred
            window                = np.vstack([window[1:], next_row])

    # Inverse-transform predictions back to GBP/MWh
    dummy                  = np.zeros((horizon, scaled.shape[1]), dtype="float32")
    dummy[:, target_col]   = preds_scaled
    forecast               = scaler.inverse_transform(dummy)[:, target_col]

    last_ts        = df_recent.index[-1]
    forecast_index = pd.date_range(
        last_ts + pd.Timedelta(minutes=30), periods=horizon, freq="30min"
    )

    return pd.Series(forecast, index=forecast_index, name="mip_forecast_gbp_mwh")


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(_HERE.parent))

    model, meta = load_artifacts()
    print("Model loaded. Feature cols:", meta["feature_cols"])
    print("Lookback:", meta["lookback"], "  Input size:", meta["input_size"])

    # Load the snapshot parquet as a stand-in for live data
    df = pd.read_parquet(_HERE.parent / "data" / "training_data_snapshot.parquet")
    drop = [c for c in ["sip_gbp_mwh", "wind_fc_mw", "demand_fc_mw"] if c in df.columns]
    df   = df.drop(columns=drop).dropna()

    forecast = predict(df, model, meta, horizon=48)
    print("\nNext 24h MIP forecast (GBP/MWh):")
    print(forecast.to_string())
