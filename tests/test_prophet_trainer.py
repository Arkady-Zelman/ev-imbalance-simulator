"""Tests for NeuralProphet grid-search trainer and disk persistence."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

_EMPTY_BACKTEST = ([], [])

try:
    from neuralprophet import NeuralProphet  # noqa: F401
    HAS_NP = True
except ImportError:
    HAS_NP = False

try:
    import joblib  # noqa: F401
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

from src.models.prophet_trainer import (
    NP_GRID,
    TrainedNPModels,
    _build_np_combos,
    _np_grid_combos,
    _np_random_combos,
    has_np_models,
    load_np_models,
    save_np_models,
    train_np_models,
)


def _make_series(n_days: int = 60, seed: int = 42):
    rng = np.random.default_rng(seed)
    n_sps = n_days * 48
    base = 50 + 30 * np.sin(np.linspace(0, 2 * np.pi * n_sps / 48, n_sps))
    idx  = pd.date_range("2025-01-01", periods=n_sps, freq="30min")
    sip  = pd.Series(base + rng.normal(0, 15, n_sps), index=idx)
    mip  = pd.Series(base + rng.normal(0, 8, n_sps), index=idx)
    dem  = pd.Series(
        30000 + 5000 * np.sin(np.linspace(0, 2 * np.pi * n_sps / 48, n_sps))
        + rng.normal(0, 500, n_sps),
        index=idx,
    )
    return sip, mip, dem


class TestNPParamCombos:
    def test_grid_combo_count(self):
        combos = _np_grid_combos()
        # 3 n_lags × 3 epochs × 2 learning_rates = 18
        assert len(combos) == 18

    def test_grid_has_all_keys(self):
        combos = _np_grid_combos()
        for c in combos:
            for k in NP_GRID:
                assert k in c

    def test_random_combo_count(self):
        assert len(_np_random_combos(n_samples=6)) == 6
        assert len(_np_random_combos(n_samples=3, seed=0)) == 3

    def test_build_combos_grid(self):
        combos = _build_np_combos(mode="grid")
        assert len(combos) == 18

    def test_build_combos_random(self):
        combos = _build_np_combos(mode="random", n_random=5)
        assert len(combos) == 5


@pytest.mark.skipif(not HAS_NP, reason="neuralprophet not installed")
class TestTrainNPModels:
    def test_full_pipeline_sip(self):
        sip, mip, dem = _make_series(60)
        with patch("src.models.prophet_trainer.run_rolling_backtest", return_value=_EMPTY_BACKTEST):
            trained = train_np_models(
                sip, mip, demand_series=dem,
                target="sip",
                param_search_mode="random",
                random_search_samples=1,
            )
        assert isinstance(trained, TrainedNPModels)
        assert trained.target == "sip"
        assert trained.training_timestamp > 0
        assert len(trained.best_params) > 0

    def test_full_pipeline_demand(self):
        sip, mip, dem = _make_series(60)
        with patch("src.models.prophet_trainer.run_rolling_backtest", return_value=_EMPTY_BACKTEST):
            trained = train_np_models(
                sip, mip, demand_series=dem,
                target="demand",
                param_search_mode="random",
                random_search_samples=1,
            )
        assert trained.target == "demand"

    def test_forward_forecasts_populated(self):
        sip, mip, dem = _make_series(60)
        with patch("src.models.prophet_trainer.run_rolling_backtest", return_value=_EMPTY_BACKTEST):
            trained = train_np_models(
                sip, mip, demand_series=dem,
                target="sip",
                param_search_mode="random",
                random_search_samples=1,
            )
        total_days = sum(len(v) for v in trained.forward_forecasts.values())
        assert total_days > 0

    def test_missing_demand_raises(self):
        sip, mip, _ = _make_series(60)
        with pytest.raises(ValueError, match="demand_series required"):
            train_np_models(sip, mip, demand_series=None, target="demand")


@pytest.mark.skipif(not HAS_NP or not HAS_JOBLIB, reason="neuralprophet/joblib not installed")
class TestNPDiskPersistence:
    def test_save_and_load_sip(self, tmp_path, monkeypatch):
        import src.models.prophet_trainer as trainer_mod
        monkeypatch.setattr(trainer_mod, "_CACHE_DIR", tmp_path)

        sip, mip, dem = _make_series(60)
        with patch("src.models.prophet_trainer.run_rolling_backtest", return_value=_EMPTY_BACKTEST):
            trained = train_np_models(
                sip, mip, demand_series=dem, target="sip",
                param_search_mode="random", random_search_samples=1,
            )

        assert save_np_models(trained, target="sip")
        assert has_np_models(target="sip")
        assert not has_np_models(target="demand")

        loaded = load_np_models(target="sip")
        assert loaded is not None
        assert loaded.target == "sip"
        assert loaded.training_timestamp == trained.training_timestamp

    def test_separate_files_per_target(self, tmp_path, monkeypatch):
        import src.models.prophet_trainer as trainer_mod
        monkeypatch.setattr(trainer_mod, "_CACHE_DIR", tmp_path)

        sip, mip, dem = _make_series(60)
        with patch("src.models.prophet_trainer.run_rolling_backtest", return_value=_EMPTY_BACKTEST):
            trained_sip = train_np_models(
                sip, mip, demand_series=dem, target="sip",
                param_search_mode="random", random_search_samples=1,
            )
            trained_dem = train_np_models(
                sip, mip, demand_series=dem, target="demand",
                param_search_mode="random", random_search_samples=1,
            )

        save_np_models(trained_sip, target="sip")
        save_np_models(trained_dem, target="demand")

        loaded_sip = load_np_models(target="sip")
        loaded_dem = load_np_models(target="demand")

        assert loaded_sip is not None and loaded_sip.target == "sip"
        assert loaded_dem is not None and loaded_dem.target == "demand"
        assert loaded_sip.training_timestamp != loaded_dem.training_timestamp
