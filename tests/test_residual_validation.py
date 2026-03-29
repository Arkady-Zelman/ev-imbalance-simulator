"""Tests for the residual forecasting validation layer."""

import numpy as np
import pandas as pd
import pytest

from src.models.forecaster import ForecastResult, run_walk_forward_backtest
from src.models.alpha_detector import (
    ResidualValidation,
    build_residual_series_for_chart,
    run_residual_validation,
)


def _make_series(n_sps: int = 960, seed: int = 42):
    rng = np.random.default_rng(seed)
    base = 50 + 30 * np.sin(np.linspace(0, 2 * np.pi * n_sps / 48, n_sps))
    noise = rng.normal(0, 10, n_sps)
    idx = pd.date_range("2025-01-01", periods=n_sps, freq="30min")
    sip = pd.Series(base + noise, index=idx)
    mip = pd.Series(base + rng.normal(0, 5, n_sps), index=idx)
    return sip, mip


@pytest.fixture
def backtest_results():
    sip, mip = _make_series()
    results = run_walk_forward_backtest(
        sip, mip, lookback_sps=48, horizons=[1, 2, 48], step=6,
    )
    return results


class TestRunResidualValidation:
    def test_returns_list(self, backtest_results):
        rv = run_residual_validation(backtest_results, [1, 2, 48])
        assert isinstance(rv, list)
        assert len(rv) > 0

    def test_result_fields(self, backtest_results):
        rv = run_residual_validation(backtest_results, [1, 2, 48])
        r = rv[0]
        assert isinstance(r, ResidualValidation)
        assert r.n_obs > 0
        assert isinstance(r.residual_mean, float)
        assert isinstance(r.residual_std, float)
        assert isinstance(r.residual_autocorr_1, float)
        assert isinstance(r.original_mae, float)
        assert isinstance(r.corrected_mae, float)
        assert isinstance(r.confirmation_rate, float)
        assert 0 <= r.confirmation_rate <= 1

    def test_improvement_bounded(self, backtest_results):
        rv = run_residual_validation(backtest_results, [1, 2, 48])
        for r in rv:
            assert r.correction_improvement <= 1.0
            assert r.corrected_mae >= 0
            assert r.original_mae >= 0

    def test_hit_rates_bounded(self, backtest_results):
        rv = run_residual_validation(backtest_results, [1, 2, 48])
        for r in rv:
            if not np.isnan(r.confirmed_trade_hit_rate):
                assert 0 <= r.confirmed_trade_hit_rate <= 1
            if not np.isnan(r.unconfirmed_trade_hit_rate):
                assert 0 <= r.unconfirmed_trade_hit_rate <= 1

    def test_insufficient_data(self):
        sip, mip = _make_series(n_sps=60)
        results = run_walk_forward_backtest(
            sip, mip, lookback_sps=48, horizons=[1], step=1,
        )
        rv = run_residual_validation(results, [1], residual_lookback=48)
        assert isinstance(rv, list)


class TestBuildResidualSeriesForChart:
    def test_output_shapes(self, backtest_results):
        ts, actual, predicted, fc = build_residual_series_for_chart(
            backtest_results, horizon=1,
        )
        assert len(ts) == len(actual) == len(predicted) == len(fc)
        assert len(ts) > 0

    def test_predicted_has_nans_at_start(self, backtest_results):
        _, _, predicted, _ = build_residual_series_for_chart(
            backtest_results, horizon=1, residual_lookback=48,
        )
        assert np.isnan(predicted[0])
        assert not np.isnan(predicted[-1])
