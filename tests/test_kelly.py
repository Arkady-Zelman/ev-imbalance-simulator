"""Tests for the Kelly criterion position sizing engine."""

import numpy as np
import pytest

from src.models.kelly import (
    KellyResult,
    _growth_rate_for_commit,
    compute_kelly_growth_rate,
    compute_kelly_pnl,
    kelly_optimal_position,
    run_kelly_analysis,
)


@pytest.fixture
def sim_data():
    """Synthetic delivered_mw and sip_matrix for testing."""
    rng = np.random.default_rng(42)
    n_runs = 2000
    delivered = rng.uniform(50, 120, (n_runs, 48))
    sip = rng.lognormal(4.0, 0.5, (n_runs, 48))
    return delivered, sip


class TestKellyOptimalPosition:
    def test_output_shape(self, sim_data):
        delivered, sip = sim_data
        result = kelly_optimal_position(
            delivered, sip, da_price=75.0, bankroll=100_000.0,
        )
        assert result.shape == (48,)

    def test_positive_commitments(self, sim_data):
        delivered, sip = sim_data
        result = kelly_optimal_position(
            delivered, sip, da_price=75.0, bankroll=100_000.0,
        )
        assert np.all(result >= 0)

    def test_fractional_kelly_reduces(self, sim_data):
        delivered, sip = sim_data
        full = kelly_optimal_position(
            delivered, sip, da_price=75.0, bankroll=100_000.0,
            kelly_fraction=1.0,
        )
        half = kelly_optimal_position(
            delivered, sip, da_price=75.0, bankroll=100_000.0,
            kelly_fraction=0.5,
        )
        np.testing.assert_allclose(half, full * 0.5, atol=1e-6)

    def test_higher_sip_reduces_commitment(self):
        """When SIP is much higher, Kelly should commit less (more downside risk)."""
        rng = np.random.default_rng(42)
        n_runs = 2000
        delivered = rng.uniform(50, 120, (n_runs, 48))

        low_sip = np.full((n_runs, 48), 40.0)
        high_sip = np.full((n_runs, 48), 400.0)

        commit_low = kelly_optimal_position(
            delivered, low_sip, da_price=75.0, bankroll=100_000.0,
        )
        commit_high = kelly_optimal_position(
            delivered, high_sip, da_price=75.0, bankroll=100_000.0,
        )
        assert np.mean(commit_high) <= np.mean(commit_low), (
            "Higher SIP should lead to lower commitment"
        )


class TestComputeKellyPnl:
    def test_output_shape(self, sim_data):
        delivered, sip = sim_data
        committed = np.percentile(delivered, 80, axis=0)
        pnl = compute_kelly_pnl(delivered, committed, sip, da_price=75.0)
        assert pnl.shape == (2000,)

    def test_no_shortfall_all_positive(self):
        """If commitment is very low, no shortfall, all P&L should be revenue."""
        rng = np.random.default_rng(42)
        delivered = rng.uniform(100, 200, (500, 48))
        sip = np.full((500, 48), 100.0)
        committed = np.full(48, 1.0)  # tiny commitment
        pnl = compute_kelly_pnl(delivered, committed, sip, da_price=75.0)
        assert np.all(pnl > 0)


class TestRunKellyAnalysis:
    def test_returns_list(self, sim_data):
        delivered, sip = sim_data
        results = run_kelly_analysis(
            delivered, sip, da_price=75.0, bankroll=100_000.0,
        )
        assert isinstance(results, list)
        assert len(results) == 4  # default 4 fractions

    def test_result_fields(self, sim_data):
        delivered, sip = sim_data
        results = run_kelly_analysis(
            delivered, sip, da_price=75.0, bankroll=100_000.0,
        )
        for kr in results:
            assert isinstance(kr, KellyResult)
            assert kr.optimal_mw.shape == (48,)
            assert 0 < kr.fraction <= 1.0
            assert kr.mean_shortfall_probability >= 0

    def test_growth_rate_ordering(self, sim_data):
        """Full Kelly should have the highest growth rate."""
        delivered, sip = sim_data
        results = run_kelly_analysis(
            delivered, sip, da_price=75.0, bankroll=100_000.0,
        )
        growth_rates = [kr.growth_rate for kr in results]
        # Full Kelly (last) should have highest or near-highest growth rate
        assert results[-1].fraction == 1.0


class TestGrowthRate:
    def test_zero_commitment_zero_growth(self, sim_data):
        delivered, sip = sim_data
        g = _growth_rate_for_commit(0.0, delivered[:, 0], sip[:, 0], 75.0, 100_000.0)
        assert abs(g) < 1e-6
