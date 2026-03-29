"""Tests for the Monte Carlo simulation engine."""

import numpy as np
import pytest

from src.models.monte_carlo import (
    SimulationParams,
    SimulationResult,
    prepare_sip_matrix,
    run_simulation,
)
from src.config import DEFAULT_DA_PRICE
import pandas as pd


class TestRunSimulation:
    @pytest.fixture
    def default_sip(self):
        return np.full((10, 48), 50.0)

    def test_returns_result(self, default_sip):
        params = SimulationParams(n_runs=100, seed=42)
        result = run_simulation(params, default_sip)
        assert isinstance(result, SimulationResult)

    def test_output_shapes(self, default_sip):
        params = SimulationParams(n_runs=200, seed=42)
        result = run_simulation(params, default_sip)
        assert result.delivered_mw.shape == (200, 48)
        assert result.plugin_rates.shape == (200, 48)
        assert result.traded_mw.shape == (48,)
        assert result.imbalance_mw.shape == (200, 48)
        assert result.daily_pnl.shape == (200,)
        assert result.daily_revenue.shape == (200,)
        assert result.daily_imbalance_cost.shape == (200,)
        assert result.sp_imbalance_cost.shape == (200, 48)

    def test_pnl_identity(self, default_sip):
        """P&L = revenue - imbalance cost."""
        params = SimulationParams(n_runs=100, seed=42)
        result = run_simulation(params, default_sip)
        expected = result.daily_revenue - result.daily_imbalance_cost
        np.testing.assert_allclose(result.daily_pnl, expected, atol=1e-6)

    def test_deterministic_with_seed(self, default_sip):
        params = SimulationParams(n_runs=50, seed=99)
        r1 = run_simulation(params, default_sip)
        r2 = run_simulation(params, default_sip)
        np.testing.assert_array_equal(r1.daily_pnl, r2.daily_pnl)

    def test_1d_sip(self):
        sip = np.full(48, 80.0)
        params = SimulationParams(n_runs=50, seed=42)
        result = run_simulation(params, sip)
        assert result.daily_pnl.shape == (50,)


class TestPrepareSIPMatrix:
    def test_empty_df(self):
        df = pd.DataFrame()
        result, is_fallback = prepare_sip_matrix(df)
        assert result.shape == (1, 48)
        assert np.all(result == DEFAULT_DA_PRICE)
        assert is_fallback is True

    def test_valid_df(self):
        rows = []
        for sp in range(1, 49):
            rows.append({
                "settlementDate": "2025-01-01",
                "settlementPeriod": sp,
                "systemBuyPrice": 50.0 + sp,
            })
        df = pd.DataFrame(rows)
        result, is_fallback = prepare_sip_matrix(df)
        assert result.shape[0] >= 1
        assert result.shape[1] == 48
        assert is_fallback is False

    def test_multiple_days(self):
        rows = []
        for day in range(5):
            for sp in range(1, 49):
                rows.append({
                    "settlementDate": f"2025-01-{day+1:02d}",
                    "settlementPeriod": sp,
                    "systemBuyPrice": 50.0 + sp + day * 10,
                })
        df = pd.DataFrame(rows)
        result, is_fallback = prepare_sip_matrix(df)
        assert result.shape == (5, 48)
        assert is_fallback is False
