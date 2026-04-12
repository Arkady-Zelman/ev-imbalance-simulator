"""Helpers for loading and validating forecast products."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import PREDICTION_DIR

logger = logging.getLogger(__name__)

PREDICTION_TYPE_FORWARD_DAILY = "forward_daily"
PREDICTION_TYPE_INTRADAY_48SP = "intraday_48sp"

_LEGACY_PREDICTION_TYPE_ALIASES = {
    "forward": PREDICTION_TYPE_FORWARD_DAILY,
    "forward_daily": PREDICTION_TYPE_FORWARD_DAILY,
    "intraday": PREDICTION_TYPE_INTRADAY_48SP,
    "intraday_48sp": PREDICTION_TYPE_INTRADAY_48SP,
}

_TARGET_FILE_MAP = {
    "sip": "sip_predictions.parquet",
    "mip": "mip_predictions.parquet",
    "demand": "demand_predictions.parquet",
    "total_generation": "gen_predictions.parquet",
}

_REQUIRED_COLUMNS = {
    "forecast_date",
    "settlement_period",
    "target",
    "lookback",
    "horizon_days",
    "hybrid_prediction",
    "prediction_type",
}


class PredictionSchemaError(ValueError):
    """Raised when a prediction product does not match the expected schema."""


def _schema_error(context: str, message: str) -> "PredictionSchemaError":
    full_message = f"{context}: {message}"
    logger.warning(full_message)
    return PredictionSchemaError(full_message)


def _context_label(target: str, context: Optional[str]) -> str:
    return context or f"{target} prediction product"


def normalise_prediction_types(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with legacy prediction_type labels mapped to canonical names."""
    if predictions is None:
        return pd.DataFrame()

    if predictions.empty:
        return predictions.copy()

    if "prediction_type" not in predictions.columns:
        raise _schema_error("prediction product", "missing required column 'prediction_type'.")

    normalised = predictions.copy()
    normalised["prediction_type"] = (
        normalised["prediction_type"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(_LEGACY_PREDICTION_TYPE_ALIASES)
        .fillna(normalised["prediction_type"].astype(str).str.strip().str.lower())
    )
    return normalised


def _validate_base_columns(
    predictions: pd.DataFrame,
    *,
    target: Optional[str],
    context: str,
) -> pd.DataFrame:
    if predictions is None:
        return pd.DataFrame()

    normalised = normalise_prediction_types(predictions)
    if normalised.empty:
        return normalised

    missing = sorted(_REQUIRED_COLUMNS.difference(normalised.columns))
    if missing:
        raise _schema_error(context, f"missing required columns: {', '.join(missing)}.")

    if target is not None:
        target_values = set(normalised["target"].dropna().astype(str).unique())
        if target_values and target_values != {target}:
            found = ", ".join(sorted(target_values))
            raise _schema_error(
                context,
                f"expected only target='{target}', found target values: {found}.",
            )

    valid_types = {PREDICTION_TYPE_FORWARD_DAILY, PREDICTION_TYPE_INTRADAY_48SP}
    actual_types = set(normalised["prediction_type"].dropna().astype(str).unique())
    unknown_types = sorted(actual_types.difference(valid_types))
    if unknown_types:
        raise _schema_error(
            context,
            f"unknown prediction_type values: {', '.join(unknown_types)}.",
        )

    return normalised


def validate_forward_daily_predictions(
    predictions: pd.DataFrame,
    *,
    target: Optional[str] = None,
    context: Optional[str] = None,
) -> pd.DataFrame:
    """Validate a daily forward product with settlement_period == 0 only."""
    ctx = _context_label(target or "unknown", context)
    forward_daily = _validate_base_columns(predictions, target=target, context=ctx)
    if forward_daily.empty:
        return forward_daily

    sp_values = pd.to_numeric(forward_daily["settlement_period"], errors="coerce")
    invalid = forward_daily[sp_values != 0]
    if not invalid.empty:
        bad_values = ", ".join(sorted({str(v) for v in invalid["settlement_period"].tolist()[:5]}))
        raise _schema_error(
            ctx,
            "daily forward predictions must use settlement_period == 0 only; "
            f"found: {bad_values}.",
        )

    return forward_daily


def validate_intraday_predictions(
    predictions: pd.DataFrame,
    *,
    target: Optional[str] = None,
    context: Optional[str] = None,
    require_full_day: bool = True,
) -> pd.DataFrame:
    """Validate an intraday 48-SP product with settlement_period values 1..48."""
    ctx = _context_label(target or "unknown", context)
    intraday = _validate_base_columns(predictions, target=target, context=ctx)
    if intraday.empty:
        return intraday

    sp_values = pd.to_numeric(intraday["settlement_period"], errors="coerce")
    invalid = intraday[(sp_values < 1) | (sp_values > 48) | sp_values.isna()]
    if not invalid.empty:
        bad_values = ", ".join(sorted({str(v) for v in invalid["settlement_period"].tolist()[:5]}))
        raise _schema_error(
            ctx,
            "intraday 48-SP predictions must use settlement periods in 1..48; "
            f"found: {bad_values}.",
        )

    duplicate_mask = intraday.duplicated(
        subset=["forecast_date", "lookback", "settlement_period"],
        keep=False,
    )
    if duplicate_mask.any():
        raise _schema_error(
            ctx,
            "intraday 48-SP predictions contain duplicate rows for the same "
            "forecast_date/lookback/settlement_period.",
        )

    if require_full_day:
        expected_sps = set(range(1, 49))
        bad_groups: list[str] = []
        group_cols = ["forecast_date", "lookback"]
        for (forecast_date, lookback), group in intraday.groupby(group_cols, dropna=False):
            group_sps = set(group["settlement_period"].astype(int))
            if group_sps != expected_sps:
                missing = sorted(expected_sps.difference(group_sps))
                extra = sorted(group_sps.difference(expected_sps))
                details: list[str] = []
                if missing:
                    details.append(f"missing SPs {missing[:5]}")
                if extra:
                    details.append(f"unexpected SPs {extra[:5]}")
                group_label = f"{pd.Timestamp(forecast_date).date()} / {lookback}"
                bad_groups.append(f"{group_label} ({'; '.join(details)})")
                if len(bad_groups) >= 3:
                    break

        if bad_groups:
            raise _schema_error(
                ctx,
                "intraday 48-SP predictions must contain a full settlement-period curve "
                "for each forecast_date/lookback; bad groups: "
                + ", ".join(bad_groups)
                + ".",
            )

    return intraday


def validate_prediction_products(
    predictions: pd.DataFrame,
    *,
    target: Optional[str] = None,
    context: Optional[str] = None,
) -> pd.DataFrame:
    """Validate a mixed prediction frame containing daily-forward and intraday products."""
    ctx = _context_label(target or "unknown", context)
    normalised = _validate_base_columns(predictions, target=target, context=ctx)
    if normalised.empty:
        return normalised

    forward_daily = normalised[
        normalised["prediction_type"] == PREDICTION_TYPE_FORWARD_DAILY
    ].copy()
    intraday = normalised[
        normalised["prediction_type"] == PREDICTION_TYPE_INTRADAY_48SP
    ].copy()

    if not forward_daily.empty:
        validate_forward_daily_predictions(forward_daily, target=target, context=ctx)
    if not intraday.empty:
        validate_intraday_predictions(
            intraday,
            target=target,
            context=ctx,
            require_full_day=True,
        )

    return normalised


def load_target_predictions(
    target: str,
    *,
    base_dir: Path = PREDICTION_DIR,
) -> Optional[pd.DataFrame]:
    """Load and validate the full prediction parquet for a target."""
    path = base_dir / _TARGET_FILE_MAP[target]
    if not path.exists():
        return None
    predictions = pd.read_parquet(path)
    return validate_prediction_products(
        predictions,
        target=target,
        context=f"{path.name}",
    )


def load_forward_daily_predictions(
    target: str,
    predictions: Optional[pd.DataFrame] = None,
    *,
    base_dir: Path = PREDICTION_DIR,
    context: Optional[str] = None,
) -> pd.DataFrame:
    """Load or extract the daily forward product for a target."""
    ctx = _context_label(target, context)
    source = load_target_predictions(target, base_dir=base_dir) if predictions is None else validate_prediction_products(
        predictions,
        target=target,
        context=ctx,
    )
    if source is None or source.empty:
        return pd.DataFrame()

    forward_daily = source[
        source["prediction_type"] == PREDICTION_TYPE_FORWARD_DAILY
    ].copy()
    if forward_daily.empty:
        available = ", ".join(sorted(source["prediction_type"].astype(str).unique()))
        raise _schema_error(
            ctx,
            "daily forward product not found. Available prediction_type values: "
            f"{available}.",
        )

    return validate_forward_daily_predictions(
        forward_daily,
        target=target,
        context=ctx,
    )


def load_intraday_predictions(
    target: str,
    predictions: Optional[pd.DataFrame] = None,
    *,
    base_dir: Path = PREDICTION_DIR,
    context: Optional[str] = None,
) -> pd.DataFrame:
    """Load or extract the intraday 48-SP product for a target."""
    ctx = _context_label(target, context)
    source = load_target_predictions(target, base_dir=base_dir) if predictions is None else validate_prediction_products(
        predictions,
        target=target,
        context=ctx,
    )
    if source is None or source.empty:
        return pd.DataFrame()

    intraday = source[
        source["prediction_type"] == PREDICTION_TYPE_INTRADAY_48SP
    ].copy()
    if intraday.empty:
        available = ", ".join(sorted(source["prediction_type"].astype(str).unique()))
        raise _schema_error(
            ctx,
            "intraday 48-SP product not found. Available prediction_type values: "
            f"{available}.",
        )

    return validate_intraday_predictions(
        intraday,
        target=target,
        context=ctx,
        require_full_day=True,
    )
