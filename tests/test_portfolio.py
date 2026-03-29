"""Tests for the fleet availability model (portfolio.py)."""

import numpy as np
import pytest

from src.models.portfolio import (
    apply_dispatch_and_override,
    generate_plugin_rates,
    theoretical_max_mw,
)
from src.config import build_sp_beta_params, CHARGER_CAPACITY_KW


class TestGeneratePluginRates:
    def test_output_shape(self):
        rates = generate_plugin_rates(100)
        assert rates.shape == (100, 48)

    def test_bounds(self):
        rates = generate_plugin_rates(500, rng=np.random.default_rng(42))
        assert np.all(rates >= 0)
        assert np.all(rates <= 1)

    def test_correlated_vs_uncorrelated(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        corr_rates = generate_plugin_rates(500, correlated=True, rng=rng1)
        uncorr_rates = generate_plugin_rates(500, correlated=False, rng=rng2)
        # Correlated draws should have higher adjacent-SP correlation
        corr_adj = np.corrcoef(corr_rates[:, 0], corr_rates[:, 1])[0, 1]
        uncorr_adj = np.corrcoef(uncorr_rates[:, 0], uncorr_rates[:, 1])[0, 1]
        assert corr_adj > uncorr_adj

    def test_reproducibility(self):
        r1 = generate_plugin_rates(50, rng=np.random.default_rng(123))
        r2 = generate_plugin_rates(50, rng=np.random.default_rng(123))
        np.testing.assert_array_equal(r1, r2)

    def test_custom_params(self):
        alphas = np.full(48, 5.0)
        betas = np.full(48, 5.0)
        rates = generate_plugin_rates(200, alphas=alphas, betas=betas,
                                      rng=np.random.default_rng(0))
        assert rates.shape == (200, 48)
        assert 0.3 < np.mean(rates) < 0.7


class TestApplyDispatchAndOverride:
    def test_output_shape(self):
        plugin = np.full((100, 48), 0.5)
        mw = apply_dispatch_and_override(plugin, fleet_size=10_000)
        assert mw.shape == (100, 48)

    def test_non_negative(self):
        plugin = np.full((200, 48), 0.6)
        mw = apply_dispatch_and_override(plugin, fleet_size=20_000,
                                         rng=np.random.default_rng(0))
        assert np.all(mw >= 0)

    def test_upper_bound(self):
        plugin = np.ones((50, 48))
        mw = apply_dispatch_and_override(plugin, fleet_size=10_000,
                                         dispatch_rate=1.0, override_rate=0.0,
                                         rng=np.random.default_rng(0))
        max_possible = theoretical_max_mw(10_000)
        assert np.all(mw <= max_possible * 1.01)  # small tolerance for Normal approx

    def test_higher_override_reduces_mw(self):
        plugin = np.full((500, 48), 0.7)
        rng1, rng2 = np.random.default_rng(42), np.random.default_rng(42)
        mw_low = apply_dispatch_and_override(plugin, 20_000, override_rate=0.02, rng=rng1)
        mw_high = apply_dispatch_and_override(plugin, 20_000, override_rate=0.10, rng=rng2)
        assert np.mean(mw_low) > np.mean(mw_high)


class TestTheoreticalMaxMW:
    def test_known_value(self):
        assert theoretical_max_mw(10_000) == pytest.approx(10_000 * CHARGER_CAPACITY_KW / 1000)
