"""Tests for the rolling forecast backtest engine."""

import numpy as np
import pandas as pd
import pytest

from src.models.rolling_backtest import (
    CrossoverResult,
    RollingErrorRow,
    build_error_matrix,
    run_rolling_backtest,
)


def _make_long_series(n_days: int = 90, seed: int = 42):
    """Create synthetic SIP and MIP series long enough for rolling backtest."""
    rng = np.random.default_rng(seed)
    n_sps = n_days * 48
    base = 50 + 30 * np.sin(np.linspace(0, 2 * np.pi * n_sps / 48, n_sps))
    sip = pd.Series(
        base + rng.normal(0, 15, n_sps),
        index=pd.date_range("2025-01-01", periods=n_sps, freq="30min"),
    )
    mip = pd.Series(
        base + rng.normal(0, 8, n_sps),
        index=sip.index,
    )
    return sip, mip


class TestRunRollingBacktest:
    @pytest.fixture
    def long_series(self):
        return _make_long_series(90)

    def test_returns_errors_and_crossovers(self, long_series):
        sip, mip = long_series
        errors, crossovers = run_rolling_backtest(sip, mip)
        assert isinstance(errors, list)
        assert isinstance(crossovers, list)
        assert len(errors) > 0
        assert len(crossovers) == 3  # one per lookback

    def test_error_fields(self, long_series):
        sip, mip = long_series
        errors, _ = run_rolling_backtest(sip, mip)
        e = errors[0]
        assert isinstance(e, RollingErrorRow)
        assert e.forecast_mae >= 0
        assert e.market_mae >= 0
        assert e.n_obs > 0

    def test_crossover_fields(self, long_series):
        sip, mip = long_series
        _, crossovers = run_rolling_backtest(sip, mip)
        c = crossovers[0]
        assert isinstance(c, CrossoverResult)
        assert 0 <= c.crossover_day <= 15

    def test_insufficient_data(self):
        """Short series should return empty or few results."""
        sip, mip = _make_long_series(10)
        errors, crossovers = run_rolling_backtest(sip, mip)
        assert isinstance(errors, list)


class TestBuildErrorMatrix:
    def test_pivot_shape(self):
        sip, mip = _make_long_series(90)
        errors, _ = run_rolling_backtest(sip, mip)
        matrix = build_error_matrix(errors)
        assert isinstance(matrix, pd.DataFrame)
        if not matrix.empty:
            assert matrix.shape[0] <= 3  # at most 3 lookbacks
            assert matrix.shape[1] <= 14  # at most 14 horizons
