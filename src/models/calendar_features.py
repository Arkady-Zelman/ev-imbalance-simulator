"""Shared calendar feature helpers for model training and inference."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _cyclical_pair(values: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    angle = 2.0 * np.pi * values.astype(float) / float(period)
    return np.sin(angle), np.cos(angle)


def build_calendar_exog_series(index: pd.DatetimeIndex) -> Dict[str, pd.Series]:
    """
    Build deterministic cyclical calendar features aligned to a half-hourly index.

    Keys are intentionally stable because they are persisted into model artifacts
    and reused during inference.
    """
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.DatetimeIndex(index)

    tod = index.hour.to_numpy(dtype=float) + index.minute.to_numpy(dtype=float) / 60.0
    dow = index.dayofweek.to_numpy(dtype=float)
    month_zero_based = (index.month.to_numpy(dtype=float) - 1.0)

    tod_sin, tod_cos = _cyclical_pair(tod, 24.0)
    dow_sin, dow_cos = _cyclical_pair(dow, 7.0)
    month_sin, month_cos = _cyclical_pair(month_zero_based, 12.0)

    return {
        "calendar_tod_sin": pd.Series(tod_sin.astype(np.float32), index=index),
        "calendar_tod_cos": pd.Series(tod_cos.astype(np.float32), index=index),
        "calendar_dow_sin": pd.Series(dow_sin.astype(np.float32), index=index),
        "calendar_dow_cos": pd.Series(dow_cos.astype(np.float32), index=index),
        "calendar_month_sin": pd.Series(month_sin.astype(np.float32), index=index),
        "calendar_month_cos": pd.Series(month_cos.astype(np.float32), index=index),
    }
