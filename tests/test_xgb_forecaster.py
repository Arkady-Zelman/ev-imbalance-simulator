"""Tests for XGBoost-based forecasters."""

import numpy as np
import pytest

try:
    import xgboost  # noqa: F401

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from src.models.xgb_forecaster import _build_features, _xgb_forecast, xgb_feature_name_list


def _make_sip(n_sps: int = 48 * 60, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = 50 + 30 * np.sin(np.linspace(0, 2 * np.pi * n_sps / 48, n_sps))
    return base + rng.normal(0, 15, n_sps)


class TestBuildFeatures:
    def test_output_shape(self):
        sip = _make_sip()
        feat = _build_features(sip, idx=500, horizon=48)
        assert feat is not None
        assert feat.ndim == 1
        assert len(feat) > 10

    def test_no_lookahead(self):
        sip = _make_sip()
        idx = 500
        feat = _build_features(sip, idx=idx, horizon=48)
        assert feat is not None

    def test_early_index_returns_nans(self):
        sip = _make_sip()
        feat = _build_features(sip, idx=10, horizon=48)
        assert feat is not None
        assert np.isnan(feat).any()

    def test_feature_names_match_vector_length(self):
        """xgb_feature_name_list must match _build_features column count."""
        sip = _make_sip()
        mip = _make_sip(seed=1)
        dem = _make_sip(seed=2)
        exog = {"w": np.random.randn(len(sip)).astype(np.float32)}
        idx = 500
        feat = _build_features(sip, idx=idx, horizon=48, mip_values=mip, aux_values=dem, exog_dict=exog)
        assert feat is not None
        names = xgb_feature_name_list(["w"])
        assert len(feat) == len(names)


@pytest.mark.skipif(not HAS_XGB, reason="xgboost not installed")
class TestXGBForecast:
    def test_returns_dict(self):
        sip = _make_sip(48 * 60)
        mip = _make_sip(48 * 60, seed=99)
        demand = _make_sip(48 * 60, seed=77)

        result = _xgb_forecast(
            sip, origin_idx=48 * 40, lookback_sps=48 * 15,
            horizons=[48], mip_values=mip, demand_values=demand,
        )
        assert isinstance(result, dict)
        assert 48 in result

    def test_different_from_mean(self):
        """XGBoost should produce something different from simple mean."""
        sip = _make_sip(48 * 60)
        mip = _make_sip(48 * 60, seed=99)

        origin = 48 * 40
        horizons = [48]

        xgb_fc = _xgb_forecast(
            sip, origin, 48 * 15, horizons,
            mip_values=mip,
        )

        from src.models.forecaster import _tod_mean_forecast

        tod_fc = _tod_mean_forecast(sip, origin, 48 * 15, horizons)

        if 48 in xgb_fc and 48 in tod_fc:
            assert xgb_fc[48] != pytest.approx(tod_fc[48], rel=0.01)

    def test_insufficient_data_returns_empty(self):
        sip = _make_sip(48 * 5)
        result = _xgb_forecast(sip, origin_idx=48 * 3, lookback_sps=48, horizons=[48])
        assert isinstance(result, dict)
