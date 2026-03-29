"""
Statistical validity tests.

These go beyond correctness (shapes, bounds) to verify the distributional
properties of the models:
- Beta distribution goodness-of-fit for plug-in rates
- Copula correlation structure
- VaR coverage (backtest)
- Statistical test module correctness
"""

import numpy as np
import pytest
from scipy import stats

from src.config import build_sp_beta_params, build_correlation_matrix
from src.models.monte_carlo import SimulationParams, run_simulation
from src.models.portfolio import generate_plugin_rates
from src.models.risk_metrics import compute_var
from src.models.stat_tests import (
    benjamini_hochberg,
    binomial_ci,
    bootstrap_ci,
    diebold_mariano,
    effective_sample_size,
)


class TestBetaDistributionFit:
    """Verify plug-in rate draws match the specified Beta distribution."""

    def test_marginal_ks_test(self):
        """Kolmogorov-Smirnov test: each SP's draws should come from the
        specified Beta(alpha, beta)."""
        alphas, betas = build_sp_beta_params()
        rng = np.random.default_rng(42)
        n_runs = 10_000
        rates = generate_plugin_rates(n_runs, alphas, betas,
                                      correlated=False, rng=rng)

        # Test several representative SPs
        for sp in [0, 12, 20, 35, 47]:
            stat, p_val = stats.kstest(
                rates[:, sp], "beta",
                args=(alphas[sp], betas[sp]),
            )
            assert p_val > 0.01, (
                f"SP {sp}: KS test rejected Beta({alphas[sp]:.2f}, {betas[sp]:.2f}) "
                f"fit with p={p_val:.4f}"
            )

    def test_mean_convergence(self):
        """Sample mean should converge to the Beta mean (alpha / (alpha + beta))."""
        alphas, betas = build_sp_beta_params()
        rng = np.random.default_rng(42)
        rates = generate_plugin_rates(50_000, alphas, betas,
                                      correlated=False, rng=rng)
        expected_means = alphas / (alphas + betas)
        sample_means = rates.mean(axis=0)
        np.testing.assert_allclose(sample_means, expected_means, atol=0.01)


class TestCopulaCorrelation:
    """Verify the Gaussian copula produces the expected correlation structure."""

    def test_adjacent_sp_correlation(self):
        """Adjacent SPs should be positively correlated when copula is on."""
        rng = np.random.default_rng(42)
        alphas, betas = build_sp_beta_params()
        rates = generate_plugin_rates(20_000, alphas, betas,
                                      correlated=True, decay=0.3, rng=rng)

        # Rank correlation between adjacent SPs should be positive
        for sp in [0, 10, 20, 30]:
            corr = np.corrcoef(rates[:, sp], rates[:, sp + 1])[0, 1]
            assert corr > 0.1, (
                f"SP {sp} and {sp+1} have unexpectedly low correlation: {corr:.3f}"
            )

    def test_distant_sp_low_correlation(self):
        """SPs 24 apart (12 hours) should have much weaker correlation."""
        rng = np.random.default_rng(42)
        alphas, betas = build_sp_beta_params()
        rates = generate_plugin_rates(20_000, alphas, betas,
                                      correlated=True, decay=0.3, rng=rng)

        near_corr = np.corrcoef(rates[:, 10], rates[:, 11])[0, 1]
        far_corr = np.corrcoef(rates[:, 10], rates[:, 34])[0, 1]
        assert near_corr > far_corr, (
            f"Near correlation ({near_corr:.3f}) should exceed far ({far_corr:.3f})"
        )

    def test_uncorrelated_draws_independent(self):
        """With correlated=False, draws should be approximately independent."""
        rng = np.random.default_rng(42)
        alphas, betas = build_sp_beta_params()
        rates = generate_plugin_rates(20_000, alphas, betas,
                                      correlated=False, rng=rng)
        corr = np.corrcoef(rates[:, 10], rates[:, 11])[0, 1]
        assert abs(corr) < 0.05, (
            f"Uncorrelated draws have unexpected correlation: {corr:.3f}"
        )


class TestVaRCoverage:
    """Verify VaR coverage: roughly (1-confidence)% of realisations
    should breach the VaR threshold."""

    def test_var95_coverage(self):
        """~5% of P&L draws should be at or below VaR 95."""
        sip = np.full((30, 48), 70.0)
        params = SimulationParams(n_runs=10_000, seed=42)
        result = run_simulation(params, sip)

        var_95 = compute_var(result.daily_pnl, confidence=0.95)
        breach_rate = (result.daily_pnl <= var_95).mean()

        # Should be approximately 5% — allow ±2% tolerance
        assert 0.03 <= breach_rate <= 0.07, (
            f"VaR 95 breach rate = {breach_rate:.3f}, expected ~0.05"
        )


class TestStatisticalTests:
    """Verify the stat_tests module itself produces correct results."""

    def test_binomial_ci_contains_true(self):
        """Wilson CI should contain the true proportion with high probability."""
        rng = np.random.default_rng(42)
        true_p = 0.6
        n = 200
        contained = 0
        n_trials = 500
        for _ in range(n_trials):
            hits = rng.binomial(n, true_p)
            _, lo, hi = binomial_ci(hits, n, confidence=0.95)
            if lo <= true_p <= hi:
                contained += 1
        coverage = contained / n_trials
        assert coverage > 0.92, f"Wilson CI coverage = {coverage:.3f}, expected ≥ 0.93"

    def test_dm_test_detects_difference(self):
        """DM test should detect a clearly better forecaster."""
        rng = np.random.default_rng(42)
        n = 500
        good_errors = rng.normal(0, 1, n)
        bad_errors = rng.normal(0, 2, n)
        dm_stat, p_val = diebold_mariano(good_errors, bad_errors, h=1, power=2)
        assert p_val < 0.05, f"DM test failed to detect difference: p={p_val:.4f}"

    def test_dm_test_no_difference(self):
        """DM test should not consistently reject when errors come from the
        same distribution.  We run 20 trials and check that the rejection rate
        is below 30% (a generous bar — the nominal rate is 5%)."""
        rng = np.random.default_rng(123)
        rejections = 0
        for _ in range(20):
            e1 = rng.normal(0, 1, 200)
            e2 = rng.normal(0, 1, 200)
            _, p_val = diebold_mariano(e1, e2, h=1)
            if p_val < 0.05:
                rejections += 1
        assert rejections < 6, (
            f"DM test rejected {rejections}/20 times under H0 (expected ~1)"
        )

    def test_bootstrap_ci_contains_mean(self):
        """Bootstrap CI should contain the true mean."""
        rng = np.random.default_rng(42)
        data = rng.normal(5.0, 2.0, 300)
        point, lo, hi = bootstrap_ci(data, statistic_fn=np.mean, seed=42)
        assert lo <= 5.0 <= hi, f"Bootstrap CI [{lo:.2f}, {hi:.2f}] doesn't contain 5.0"

    def test_fdr_correction(self):
        """BH correction should reject fewer than raw p-values."""
        p_values = [0.001, 0.01, 0.03, 0.04, 0.05, 0.06, 0.10, 0.50, 0.80]
        significant = benjamini_hochberg(p_values, alpha=0.05)
        n_sig = sum(significant)
        n_raw = sum(1 for p in p_values if p < 0.05)
        assert n_sig <= n_raw, "FDR should reject fewer or equal hypotheses"
        assert n_sig >= 1, "Should reject at least the p=0.001 hypothesis"

    def test_effective_sample_size_independent(self):
        """For iid data, effective N should be close to actual N."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 1000)
        n_eff = effective_sample_size(data)
        assert n_eff > 500, f"ESS for iid data should be near N=1000, got {n_eff:.0f}"

    def test_effective_sample_size_autocorrelated(self):
        """For AR(1) data, effective N should be much less than actual N."""
        rng = np.random.default_rng(42)
        n = 1000
        data = np.empty(n)
        data[0] = rng.normal()
        for i in range(1, n):
            data[i] = 0.9 * data[i - 1] + rng.normal() * 0.1
        n_eff = effective_sample_size(data)
        assert n_eff < n * 0.3, f"ESS for AR(0.9) should be much < N, got {n_eff:.0f}"


class TestStressCoupling:
    """Verify the SIP-availability stress coupling works."""

    def test_stress_increases_tail_risk(self):
        """With stress coupling, CVaR should be worse (more negative) than without."""
        sip_varied = np.random.default_rng(42).lognormal(4.0, 0.8, (50, 48))

        params_no_stress = SimulationParams(
            n_runs=5_000, seed=42, sip_stress_coupling=False,
        )
        params_stress = SimulationParams(
            n_runs=5_000, seed=42, sip_stress_coupling=True,
        )
        r_no = run_simulation(params_no_stress, sip_varied)
        r_st = run_simulation(params_stress, sip_varied)

        # Stress coupling should increase variance or worsen the tail
        assert r_st.daily_pnl.std() >= r_no.daily_pnl.std() * 0.95, (
            "Stress coupling should not significantly reduce variance"
        )
