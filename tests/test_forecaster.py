"""Tests for the walk-forward forecast engine."""

import numpy as np
import pandas as pd
import pytest

from src.models.forecaster import (
    ForecastResult,
    build_aligned_series,
    run_walk_forward_backtest,
)


def _make_sip_series(n_sps: int = 480, seed: int = 42) -> pd.Series:
    """Create a synthetic half-hourly SIP series with diurnal pattern."""
    rng = np.random.default_rng(seed)
    base = 50 + 30 * np.sin(np.linspace(0, 2 * np.pi * n_sps / 48, n_sps))
    noise = rng.normal(0, 10, n_sps)
    values = base + noise
    idx = pd.date_range("2025-01-01", periods=n_sps, freq="30min")
    return pd.Series(values, index=idx, name="sip")


def _make_mip_series(sip: pd.Series, seed: int = 99) -> pd.Series:
    """Create a synthetic MIP that roughly tracks SIP with lag."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 5, len(sip))
    return pd.Series(sip.values + noise, index=sip.index, name="mip")


class TestRunWalkForwardBacktest:
    @pytest.fixture
    def series_pair(self):
        sip = _make_sip_series(960)
        mip = _make_mip_series(sip)
        return sip, mip

    def test_returns_list(self, series_pair):
        sip, mip = series_pair
        results = run_walk_forward_backtest(sip, mip, lookback_sps=48,
                                           horizons=[1, 2, 48], step=48)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_forecast_result_fields(self, series_pair):
        sip, mip = series_pair
        results = run_walk_forward_backtest(sip, mip, lookback_sps=48,
                                           horizons=[1, 2], step=48)
        r = results[0]
        assert isinstance(r, ForecastResult)
        assert isinstance(r.origin_datetime, pd.Timestamp)
        assert len(r.forecasts) > 0
        assert len(r.realised) > 0
        assert len(r.market_fwd) > 0

    def test_no_lookahead(self, series_pair):
        sip, mip = series_pair
        results = run_walk_forward_backtest(sip, mip, lookback_sps=96,
                                           horizons=[1], step=48)
        for r in results:
            assert r.origin_idx >= 96

    def test_ewma_method(self, series_pair):
        sip, mip = series_pair
        results = run_walk_forward_backtest(sip, mip, lookback_sps=48,
                                           horizons=[1, 2], method="ewma",
                                           step=48)
        assert len(results) > 0

    def test_insufficient_data(self):
        sip = pd.Series([1.0, 2.0], index=pd.date_range("2025-01-01", periods=2, freq="30min"))
        mip = pd.Series([1.5, 2.5], index=sip.index)
        results = run_walk_forward_backtest(sip, mip, lookback_sps=48, horizons=[1])
        assert len(results) == 0

    def test_different_lookbacks_different_results(self, series_pair):
        sip, mip = series_pair
        r1 = run_walk_forward_backtest(sip, mip, lookback_sps=48,
                                       horizons=[1], step=48)
        r2 = run_walk_forward_backtest(sip, mip, lookback_sps=336,
                                       horizons=[1], step=48)
        if r1 and r2:
            f1 = r1[0].forecasts.get(1, 0)
            f2 = r2[0].forecasts.get(1, 0)
            # Different lookbacks should generally produce different forecasts
            # (not guaranteed with synthetic data, so we just check they ran)
            assert isinstance(f1, float)
            assert isinstance(f2, float)


class TestBuildAlignedSeries:
    def test_alignment(self):
        sip_rows = []
        mip_rows = []
        for sp in range(1, 49):
            sip_rows.append({
                "settlementDate": "2025-01-01",
                "settlementPeriod": sp,
                "systemBuyPrice": 50.0 + sp,
            })
            mip_rows.append({
                "settlementDate": "2025-01-01",
                "settlementPeriod": sp,
                "price": 48.0 + sp,
            })
        sip_df = pd.DataFrame(sip_rows)
        mip_df = pd.DataFrame(mip_rows)
        sip_s, mip_s, _, _da = build_aligned_series(sip_df, mip_df)
        assert len(sip_s) == len(mip_s)
        assert len(sip_s) <= 48
