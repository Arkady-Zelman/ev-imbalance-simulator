"""Tests for risk and performance metrics."""

import numpy as np
import pytest

from src.models.risk_metrics import (
    compute_capture_ratios,
    compute_cvar,
    compute_risk_summary,
    compute_var,
)


class TestVaR:
    def test_known_distribution(self):
        pnl = np.arange(-100, 101, dtype=float)
        var_95 = compute_var(pnl, 0.95)
        assert var_95 == pytest.approx(-90.0, abs=2.0)

    def test_all_positive(self):
        pnl = np.ones(1000) * 100
        assert compute_var(pnl) == pytest.approx(100.0)

    def test_all_negative(self):
        pnl = np.ones(1000) * -50
        assert compute_var(pnl) == pytest.approx(-50.0)


class TestCVaR:
    def test_worse_than_var(self):
        rng = np.random.default_rng(42)
        pnl = rng.normal(0, 100, size=10_000)
        var = compute_var(pnl)
        cvar = compute_cvar(pnl)
        assert cvar <= var

    def test_constant_distribution(self):
        pnl = np.full(1000, 42.0)
        assert compute_cvar(pnl) == pytest.approx(42.0)


class TestCaptureRatios:
    def test_perfect_delivery(self):
        """When traded == delivered, capture ratio should be ~1.0."""
        delivered = np.full((100, 48), 10.0)
        traded = np.full(48, 10.0)
        ratios = compute_capture_ratios(delivered, traded, da_price=75.0)
        np.testing.assert_allclose(ratios, 1.0, atol=1e-6)

    def test_over_delivery_with_sip(self):
        """When we deliver more than traded, capture > 1 with SIP pricing."""
        delivered = np.full((100, 48), 15.0)
        traded = np.full(48, 10.0)
        sip = np.full((100, 48), 75.0)
        ratios = compute_capture_ratios(delivered, traded, da_price=75.0, sip_matrix=sip)
        assert np.mean(ratios) > 1.0 - 1e-6

    def test_under_delivery_with_sip(self):
        """When we deliver less than traded with high SIP, capture < 1."""
        delivered = np.full((100, 48), 5.0)
        traded = np.full(48, 10.0)
        sip = np.full((100, 48), 150.0)  # SIP > DA → imbalance is extra costly
        ratios = compute_capture_ratios(delivered, traded, da_price=75.0, sip_matrix=sip)
        assert np.all(ratios < 1.0)


class TestRiskSummary:
    def test_summary_fields(self):
        rng = np.random.default_rng(42)
        pnl = rng.normal(100, 50, size=5000)
        cr = np.ones(5000) * 0.95
        summary = compute_risk_summary(pnl, cr)
        assert summary.mean_pnl == pytest.approx(100, abs=5)
        assert summary.std_pnl == pytest.approx(50, abs=5)
        assert summary.capture_ratio_mean == pytest.approx(0.95)
        assert summary.max_loss <= summary.var_95
        assert summary.max_gain >= summary.mean_pnl
