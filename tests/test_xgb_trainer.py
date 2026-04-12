"""Tests for XGBoost grid-search trainer and disk persistence."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Stub for run_rolling_backtest so training tests don't run the full backtest
_EMPTY_BACKTEST = ([], [])


def _fast_train(sip, mip, demand=None, target="sip", n_combos=3):
    """Call train_xgb_models with minimal combos and mocked backtest."""
    with patch("src.models.rolling_backtest.run_rolling_backtest", return_value=_EMPTY_BACKTEST):
        return __import__("src.models.xgb_trainer", fromlist=["train_xgb_models"]).train_xgb_models(
            sip, mip,
            demand_series=demand,
            target=target,
            param_search_mode="random",
            random_search_samples=n_combos,
        )

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
    GRID,
    TrainedXGBModels,
    _build_train_data,
    _generate_grid_combos,
    _generate_random_combos,
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
    def test_grid_combo_count(self):
        combos = _generate_grid_combos()
        # In-depth mode now uses 150 random samples over the full GRID
        assert len(combos) == 150

    def test_grid_has_all_grid_keys(self):
        combos = _generate_grid_combos()
        for c in combos:
            for k in GRID:
                assert k in c

    def test_random_combo_count(self):
        assert len(_generate_random_combos(n_samples=30)) == 30
        assert len(_generate_random_combos(n_samples=5, seed=0)) == 5


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
        combos = _generate_grid_combos()
        best_p, best_score, _train_score, _history = _grid_search_single(X, y, combos)
        assert isinstance(best_p, dict)
        assert best_score >= 0
        assert "n_estimators" in best_p


@pytest.mark.skipif(not HAS_XGB, reason="xgboost not installed")
class TestTrainXGBModels:
    def test_full_pipeline(self):
        sip, mip, demand = _make_series(90)
        trained = _fast_train(sip, mip, demand)
        assert isinstance(trained, TrainedXGBModels)
        assert trained.training_timestamp > 0
        assert len(trained.best_params) > 0
        # backtest mocked so errors = []
        assert isinstance(trained.backtest_errors, list)

    def test_has_final_models(self):
        sip, mip, demand = _make_series(90)
        trained = _fast_train(sip, mip, demand)
        total_models = sum(len(v) for v in trained.final_models.values())
        assert total_models > 0


@pytest.mark.skipif(not HAS_XGB, reason="xgboost not installed")
class TestForecastForward:
    def test_returns_forecasts(self):
        sip, mip, demand = _make_series(90)
        trained = _fast_train(sip, mip, demand)
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
    def test_save_and_load_sip(self, tmp_path, monkeypatch):
        import src.models.xgb_trainer as trainer_mod
        monkeypatch.setattr(trainer_mod, "_CACHE_DIR", tmp_path)

        sip, mip, demand = _make_series(90)
        trained = _fast_train(sip, mip, demand, target="sip")

        assert save_trained_models(trained, target="sip")
        assert has_trained_models(target="sip")
        assert not has_trained_models(target="demand")

        loaded = load_trained_models(target="sip")
        assert loaded is not None
        assert loaded.target == "sip"
        assert loaded.training_timestamp == trained.training_timestamp
        assert len(loaded.best_params) == len(trained.best_params)

    def test_save_and_load_demand(self, tmp_path, monkeypatch):
        import src.models.xgb_trainer as trainer_mod
        monkeypatch.setattr(trainer_mod, "_CACHE_DIR", tmp_path)

        sip, mip, demand = _make_series(90)
        trained = _fast_train(sip, mip, demand, target="demand")

        assert save_trained_models(trained, target="demand")
        assert has_trained_models(target="demand")

        loaded = load_trained_models(target="demand")
        assert loaded is not None
        assert loaded.target == "demand"

    def test_separate_files(self, tmp_path, monkeypatch):
        import src.models.xgb_trainer as trainer_mod
        monkeypatch.setattr(trainer_mod, "_CACHE_DIR", tmp_path)

        sip, mip, demand = _make_series(90)
        trained_sip = _fast_train(sip, mip, demand, target="sip")
        trained_dem = _fast_train(sip, mip, demand, target="demand")

        save_trained_models(trained_sip, target="sip")
        save_trained_models(trained_dem, target="demand")

        loaded_sip = load_trained_models(target="sip")
        loaded_dem = load_trained_models(target="demand")

        assert loaded_sip is not None and loaded_sip.target == "sip"
        assert loaded_dem is not None and loaded_dem.target == "demand"
        assert loaded_sip.training_timestamp != loaded_dem.training_timestamp
