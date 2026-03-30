"""Tests for XGBoost grid-search trainer and disk persistence."""

import numpy as np
import pandas as pd
import pytest

try:
    import xgboost  # noqa: F401

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import joblib  # noqa: F401

    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

from src.models.xgb_trainer import (
    TrainedXGBModels,
    _build_train_data,
    _generate_param_combos,
    _grid_search_single,
    forecast_forward,
    has_trained_models,
    load_trained_models,
    save_trained_models,
    train_xgb_models,
)


def _make_series(n_days: int = 90, seed: int = 42):
    rng = np.random.default_rng(seed)
    n_sps = n_days * 48
    base = 50 + 30 * np.sin(np.linspace(0, 2 * np.pi * n_sps / 48, n_sps))
    idx = pd.date_range("2025-01-01", periods=n_sps, freq="30min")
    sip = pd.Series(base + rng.normal(0, 15, n_sps), index=idx)
    mip = pd.Series(base + rng.normal(0, 8, n_sps), index=idx)
    demand = pd.Series(
        30000 + 5000 * np.sin(np.linspace(0, 2 * np.pi * n_sps / 48, n_sps))
        + rng.normal(0, 500, n_sps),
        index=idx,
    )
    return sip, mip, demand


class TestParamCombos:
    def test_count(self):
        combos = _generate_param_combos()
        assert len(combos) == 27

    def test_has_required_keys(self):
        combos = _generate_param_combos()
        for c in combos:
            assert "n_estimators" in c
            assert "max_depth" in c
            assert "learning_rate" in c
            assert "subsample" in c


class TestBuildTrainData:
    def test_returns_arrays(self):
        sip, mip, demand = _make_series(90)
        X, y, cmeans = _build_train_data(
            sip.values.astype(float), 48 * 30, 48 * 7,
            mip.values.astype(float), demand.values.astype(float),
        )
        assert X is not None
        assert y is not None
        assert cmeans is not None
        assert X.shape[0] == len(y)
        assert len(cmeans) == X.shape[1]

    def test_insufficient_data_returns_none(self):
        sip, mip, _ = _make_series(5)
        X, y, cmeans = _build_train_data(
            sip.values.astype(float), 48 * 30, 48 * 14,
            mip.values.astype(float), None,
        )
        assert X is None


@pytest.mark.skipif(not HAS_XGB, reason="xgboost not installed")
class TestGridSearchSingle:
    def test_returns_best_params(self):
        sip, mip, demand = _make_series(90)
        X, y, _ = _build_train_data(
            sip.values.astype(float), 48 * 30, 48,
            mip.values.astype(float), demand.values.astype(float),
        )
        combos = _generate_param_combos()
        best_p, best_score = _grid_search_single(X, y, combos)
        assert isinstance(best_p, dict)
        assert best_score >= 0
        assert "n_estimators" in best_p


@pytest.mark.skipif(not HAS_XGB, reason="xgboost not installed")
class TestTrainXGBModels:
    def test_full_pipeline(self):
        sip, mip, demand = _make_series(90)
        trained = train_xgb_models(sip, mip, demand_series=demand)
        assert isinstance(trained, TrainedXGBModels)
        assert trained.training_timestamp > 0
        assert len(trained.best_params) > 0
        assert len(trained.backtest_errors) >= 0
        assert len(trained.backtest_crossovers) >= 0

    def test_has_final_models(self):
        sip, mip, demand = _make_series(90)
        trained = train_xgb_models(sip, mip, demand_series=demand)
        total_models = sum(len(v) for v in trained.final_models.values())
        assert total_models > 0


@pytest.mark.skipif(not HAS_XGB, reason="xgboost not installed")
class TestForecastForward:
    def test_returns_forecasts(self):
        sip, mip, demand = _make_series(90)
        trained = train_xgb_models(sip, mip, demand_series=demand)
        forecasts = forecast_forward(
            trained,
            sip.values.astype(float),
            mip.values.astype(float),
            demand.values.astype(float),
            n_days=14,
        )
        assert isinstance(forecasts, dict)
        assert len(forecasts) > 0
        for lb, day_fc in forecasts.items():
            assert isinstance(day_fc, dict)
            for day, val in day_fc.items():
                assert 1 <= day <= 14
                assert isinstance(val, float)


@pytest.mark.skipif(not HAS_XGB or not HAS_JOBLIB, reason="xgboost/joblib not installed")
class TestDiskPersistence:
    def test_save_and_load(self, tmp_path, monkeypatch):
        import src.models.xgb_trainer as trainer_mod

        monkeypatch.setattr(trainer_mod, "_CACHE_DIR", tmp_path)
        monkeypatch.setattr(trainer_mod, "_MODEL_FILE", tmp_path / "test_model.joblib")

        sip, mip, demand = _make_series(90)
        trained = train_xgb_models(sip, mip, demand_series=demand)

        assert save_trained_models(trained)
        assert has_trained_models()

        loaded = load_trained_models()
        assert loaded is not None
        assert loaded.training_timestamp == trained.training_timestamp
        assert len(loaded.best_params) == len(trained.best_params)
