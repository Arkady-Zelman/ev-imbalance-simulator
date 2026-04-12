import pandas as pd
import pytest

from src.predictions import (
    PREDICTION_TYPE_FORWARD_DAILY,
    PREDICTION_TYPE_INTRADAY_48SP,
    PredictionSchemaError,
    load_forward_daily_predictions,
    load_intraday_predictions,
    validate_prediction_products,
)


def _make_prediction_frame() -> pd.DataFrame:
    rows: list[dict] = []
    for day in range(1, 3):
        rows.append({
            "forecast_date": pd.Timestamp("2026-04-13") + pd.Timedelta(days=day - 1),
            "settlement_period": 0,
            "target": "demand",
            "lookback": "30d",
            "horizon_days": day,
            "hybrid_prediction": 30000.0 + day,
            "prediction_type": "forward",
        })

    for sp in range(1, 49):
        rows.append({
            "forecast_date": pd.Timestamp("2026-04-13"),
            "settlement_period": sp,
            "target": "demand",
            "lookback": "30d",
            "horizon_days": 0,
            "hybrid_prediction": 28000.0 + sp,
            "prediction_type": "intraday",
        })

    return pd.DataFrame(rows)


def test_validate_prediction_products_normalises_legacy_labels():
    predictions = validate_prediction_products(_make_prediction_frame(), target="demand")

    assert set(predictions["prediction_type"].unique()) == {
        PREDICTION_TYPE_FORWARD_DAILY,
        PREDICTION_TYPE_INTRADAY_48SP,
    }


def test_load_forward_daily_predictions_enforces_settlement_period_zero():
    predictions = _make_prediction_frame()
    predictions.loc[predictions["settlement_period"] == 0, "settlement_period"] = 1

    with pytest.raises(PredictionSchemaError, match="settlement_period == 0"):
        load_forward_daily_predictions("demand", predictions)


def test_load_intraday_predictions_rejects_settlement_period_zero():
    predictions = _make_prediction_frame()
    predictions.loc[predictions["settlement_period"] == 48, "settlement_period"] = 0

    with pytest.raises(PredictionSchemaError, match="settlement periods in 1..48"):
        load_intraday_predictions("demand", predictions)


def test_load_intraday_predictions_requires_full_curve():
    predictions = _make_prediction_frame()
    predictions = predictions[predictions["settlement_period"] != 48].copy()

    with pytest.raises(PredictionSchemaError, match="full settlement-period curve"):
        load_intraday_predictions("demand", predictions)


def test_product_specific_loaders_return_expected_rows():
    predictions = _make_prediction_frame()

    forward_daily = load_forward_daily_predictions("demand", predictions)
    intraday_48sp = load_intraday_predictions("demand", predictions)

    assert set(forward_daily["prediction_type"].unique()) == {PREDICTION_TYPE_FORWARD_DAILY}
    assert set(forward_daily["settlement_period"].unique()) == {0}
    assert set(intraday_48sp["prediction_type"].unique()) == {PREDICTION_TYPE_INTRADAY_48SP}
    assert intraday_48sp["settlement_period"].min() == 1
    assert intraday_48sp["settlement_period"].max() == 48
