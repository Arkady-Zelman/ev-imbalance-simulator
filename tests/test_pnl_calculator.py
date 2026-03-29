"""Tests for the shared P&L calculator."""

import numpy as np
import pytest

from src.models.pnl_calculator import compute_pnl_for_position


class TestComputePnlForPosition:
    def test_output_shape(self):
        delivered = np.full((100, 48), 10.0)
        traded = np.full(48, 10.0)
        sip = np.full((5, 48), 50.0)
        pnl = compute_pnl_for_position(delivered, traded, sip, da_price=75.0, n_runs=100)
        assert pnl.shape == (100,)

    def test_no_imbalance(self):
        """When delivered == traded, P&L = pure revenue (no imbalance cost)."""
        n_runs = 200
        delivered = np.full((n_runs, 48), 10.0)
        traded = np.full(48, 10.0)
        sip = np.full((5, 48), 50.0)
        da = 75.0
        pnl = compute_pnl_for_position(delivered, traded, sip, da, n_runs)
        expected_rev = 10.0 * 0.5 * 75.0 * 48
        np.testing.assert_allclose(pnl, expected_rev, atol=1e-6)

    def test_1d_sip(self):
        delivered = np.full((50, 48), 8.0)
        traded = np.full(48, 10.0)
        sip = np.full(48, 60.0)
        pnl = compute_pnl_for_position(delivered, traded, sip, da_price=75.0, n_runs=50)
        assert pnl.shape == (50,)

    def test_deterministic_with_seed(self):
        delivered = np.random.default_rng(0).normal(10, 2, (100, 48))
        traded = np.full(48, 10.0)
        sip = np.random.default_rng(1).normal(50, 20, (30, 48))
        pnl1 = compute_pnl_for_position(delivered, traded, sip, 75.0, 100, seed=42)
        pnl2 = compute_pnl_for_position(delivered, traded, sip, 75.0, 100, seed=42)
        np.testing.assert_array_equal(pnl1, pnl2)

    def test_higher_sip_increases_cost(self):
        """Higher SIP should increase imbalance cost when we're short."""
        n = 200
        delivered = np.full((n, 48), 8.0)  # under-delivering
        traded = np.full(48, 10.0)
        sip_low = np.full((5, 48), 30.0)
        sip_high = np.full((5, 48), 300.0)
        pnl_low = compute_pnl_for_position(delivered, traded, sip_low, 75.0, n)
        pnl_high = compute_pnl_for_position(delivered, traded, sip_high, 75.0, n)
        assert np.mean(pnl_low) > np.mean(pnl_high)
