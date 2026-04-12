"""Tests for shared calendar exogenous features."""

import pandas as pd

from backend.train import build_exog_series
from src.models.calendar_features import build_calendar_exog_series
from src.models.xgb_forecaster import xgb_feature_name_list


def test_build_calendar_exog_series_returns_expected_keys():
    index = pd.date_range("2025-01-01", periods=96, freq="30min")
    exog = build_calendar_exog_series(index)

    assert set(exog) == {
        "calendar_tod_sin",
        "calendar_tod_cos",
        "calendar_dow_sin",
        "calendar_dow_cos",
        "calendar_month_sin",
        "calendar_month_cos",
    }
    for series in exog.values():
        assert series.index.equals(index)
        assert len(series) == len(index)


def test_build_exog_series_includes_calendar_features():
    index = pd.date_range("2025-01-01", periods=48, freq="30min")
    sip_series = pd.Series(range(len(index)), index=index, dtype=float)

    exog = build_exog_series(
        {
            "sip_series": sip_series,
            "weather_df": pd.DataFrame(index=index),
            "wind_fc_df": pd.DataFrame(index=index),
            "demand_fc_df": pd.DataFrame(index=index),
        }
    )

    assert "calendar_tod_sin" in exog
    assert "calendar_dow_cos" in exog
    assert "calendar_month_sin" in exog


def test_xgb_feature_name_list_accounts_for_current_exog_value():
    names = xgb_feature_name_list(["calendar_tod_sin"])
    assert "calendar_tod_sin_current" in names
    assert "calendar_tod_sin_lag48" in names
    assert "calendar_tod_sin_lag336" in names
    assert "calendar_tod_sin_roll48_mean" in names
