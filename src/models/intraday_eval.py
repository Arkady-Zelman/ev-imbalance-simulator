"""
Intraday evaluation metrics for h=48 (settlement-period level) forecasts.

All functions operate on (N_days × 48) arrays.  No retraining is done —
quantile predictions are approximated via residual bootstrapping on the
train/val residuals from the existing 80/20 temporal split.

Metrics
-------
mae_intraday        — mean absolute error across all (day, SP) pairs
crps                — approximate CRPS via 99-quantile pinball grid
pinball_p10         — pinball loss at 10th quantile (cost of missing low end)
pinball_p90         — pinball loss at 90th quantile (cost of over-procuring)
directional_accuracy — fraction of correct up/down moves between consecutive SPs
spike_precision     — precision: predicted spike AND actual spike / all predicted spikes
spike_recall        — recall:  predicted spike AND actual spike / all actual spikes
coverage_80pct      — fraction of actuals inside the [p10, p90] interval

Best-lookback selection
-----------------------
Primary:   lowest crps
Tiebreaker: highest spike_recall (we prefer not to miss SIP spikes)
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Pinball & CRPS helpers ────────────────────────────────────────────────────

def _pinball(actual: np.ndarray, pred: np.ndarray, q: float) -> float:
    """Asymmetric pinball loss at quantile q."""
    err = actual - pred
    return float(np.mean(np.where(err >= 0, q * err, (q - 1.0) * err)))


def _compute_crps(
    actual_flat: np.ndarray,
    pred_flat: np.ndarray,
    residuals: np.ndarray,
    n_quantiles: int = 99,
) -> float:
    """
    Approximate CRPS via mean pinball across a uniform quantile grid.

    CRPS = 2 * mean_q[ pinball(actual, F^{-1}(q), q) ]

    The distribution F is estimated empirically from the residual array by
    shifting the point prediction by the empirical quantile of residuals.
    """
    qs = np.linspace(0.01, 0.99, n_quantiles)
    total = 0.0
    for q in qs:
        q_shift = float(np.quantile(residuals, q))
        q_pred  = pred_flat + q_shift
        total  += _pinball(actual_flat, q_pred, q)
    return 2.0 * total / n_quantiles


def _residual_interval(
    pred_flat: np.ndarray,
    residuals: np.ndarray,
    p_low: float = 0.10,
    p_high: float = 0.90,
) -> Tuple[np.ndarray, np.ndarray]:
    """Shift point predictions by empirical residual quantiles to form intervals."""
    return (
        pred_flat + float(np.quantile(residuals, p_low)),
        pred_flat + float(np.quantile(residuals, p_high)),
    )


# ── Core metric function ──────────────────────────────────────────────────────

def eval_intraday_metrics(
    actual_48: np.ndarray,          # (N, 48) actuals
    pred_48: np.ndarray,            # (N, 48) point predictions
    pred_p10: np.ndarray,           # (N, 48) 10th-pct predictions
    pred_p90: np.ndarray,           # (N, 48) 90th-pct predictions
    residuals: Optional[np.ndarray] = None,  # flat residuals for CRPS; computed if None
) -> dict:
    """
    Compute intraday evaluation metrics for a single lookback's h=48 predictions.

    Parameters
    ----------
    actual_48 : (N, 48) array of actual values
    pred_48   : (N, 48) array of point predictions
    pred_p10  : (N, 48) array of 10th-percentile predictions
    pred_p90  : (N, 48) array of 90th-percentile predictions
    residuals : optional flat array of (actual - pred) residuals for CRPS approximation

    Returns
    -------
    dict with keys: mae_intraday, crps, pinball_p10, pinball_p90,
                    directional_accuracy, spike_precision, spike_recall, coverage_80pct
    """
    actual_flat = actual_48.flatten()
    pred_flat   = pred_48.flatten()

    if residuals is None:
        residuals = actual_flat - pred_flat

    # ── MAE ───────────────────────────────────────────────────────────────────
    mae_intraday = float(np.nanmean(np.abs(pred_flat - actual_flat)))

    # ── CRPS ──────────────────────────────────────────────────────────────────
    valid = np.isfinite(actual_flat) & np.isfinite(pred_flat)
    crps = _compute_crps(actual_flat[valid], pred_flat[valid], residuals[np.isfinite(residuals)])

    # ── Pinball at p10 and p90 ────────────────────────────────────────────────
    pinball_p10 = _pinball(actual_flat, pred_p10.flatten(), 0.10)
    pinball_p90 = _pinball(actual_flat, pred_p90.flatten(), 0.90)

    # ── Directional accuracy (sign of SP-to-SP change) ────────────────────────
    N = actual_48.shape[0]
    if N > 1:
        act_diff  = np.diff(actual_48, axis=1)   # (N, 47) — SP-to-SP moves within each day
        pred_diff = np.diff(pred_48,   axis=1)
        dir_match = np.sign(act_diff) == np.sign(pred_diff)
        directional_accuracy = float(np.nanmean(dir_match))
    else:
        directional_accuracy = float("nan")

    # ── Spike precision / recall (actual > 90th pct) ──────────────────────────
    spike_thresh    = float(np.nanpercentile(actual_flat, 90))
    is_spike_actual = actual_flat > spike_thresh
    is_spike_pred   = pred_flat   > spike_thresh

    tp = int(np.sum(is_spike_actual &  is_spike_pred))
    fp = int(np.sum(~is_spike_actual & is_spike_pred))
    fn = int(np.sum(is_spike_actual & ~is_spike_pred))

    spike_precision = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
    spike_recall    = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")

    # ── Coverage (80 % interval) ──────────────────────────────────────────────
    in_interval    = (pred_p10.flatten() <= actual_flat) & (actual_flat <= pred_p90.flatten())
    coverage_80pct = float(np.nanmean(in_interval))

    return {
        "mae_intraday":         mae_intraday,
        "crps":                 crps,
        "pinball_p10":          pinball_p10,
        "pinball_p90":          pinball_p90,
        "directional_accuracy": directional_accuracy,
        "spike_precision":      spike_precision,
        "spike_recall":         spike_recall,
        "coverage_80pct":       coverage_80pct,
    }


# ── Retrospective lookback evaluation ────────────────────────────────────────

def eval_lookback_intraday(
    xgb_trained,
    lstm_trained,
    sip_series:    "pd.Series",
    mip_series:    "pd.Series",
    demand_series: Optional["pd.Series"],
    gen_series:    Optional["pd.Series"],
    exog_dict:     Optional[Dict[str, np.ndarray]],
    target: str = "sip",
    eval_days: int = 30,
) -> Tuple[Dict[str, dict], Dict[str, float], Dict[str, float]]:
    """
    Evaluate each lookback's h=48 (intraday) performance on the last ``eval_days``
    of historical data using a rolling day-by-day walk-forward approach.

    For each "eval day":
        - Slice data up to that day's end index
        - Run forecast_intraday_48sp → 48-SP prediction for the *next* day
        - Compare to actual target_v[end_idx : end_idx+48]

    Parameters
    ----------
    xgb_trained, lstm_trained : trained model objects
    sip_series … gen_series   : aligned pd.Series from build_aligned_series
    exog_dict                  : {key: np.ndarray(N,)} already aligned to sip_series
    target                     : "sip" | "mip" | "demand" | "total_generation"
    eval_days                  : number of recent days used for retrospective scoring

    Returns
    -------
    scores          : {lb_label: metric_dict}
    xgb_mae_per_lb  : {lb_label: float}  — XGB-only intraday MAE per lookback
    lstm_mae_per_lb : {lb_label: float}  — LSTM-only intraday MAE per lookback
    """
    from src.models.rolling_backtest import ROLLING_LOOKBACKS
    from src.models.xgb_trainer  import forecast_intraday_48sp as xgb_intraday
    from src.models.lstm_trainer import forecast_intraday_48sp as lstm_intraday

    sip_v = sip_series.values.astype(float)
    mip_v = mip_series.values.astype(float)
    dem_v = demand_series.values.astype(float) if demand_series is not None else None
    gen_v = gen_series.values.astype(float)   if gen_series    is not None else None

    target_v_map = {
        "sip":              sip_v,
        "mip":              mip_v,
        "demand":           dem_v,
        "total_generation": gen_v,
    }
    target_v = target_v_map.get(target, sip_v)
    if target_v is None:
        logger.warning("eval_lookback_intraday: target_v is None for target=%s", target)
        return {}, {}, {}

    n         = len(target_v)
    eval_sps  = eval_days * 48
    min_lb_sp = max(ROLLING_LOOKBACKS.values())   # 15 days × 48 = 720 SPs

    # Accumulators: one list of (48,) arrays per lookback
    lb_xgb_preds:  Dict[str, list] = {lb: [] for lb in ROLLING_LOOKBACKS}
    lb_lstm_preds: Dict[str, list] = {lb: [] for lb in ROLLING_LOOKBACKS}
    lb_actuals:    Dict[str, list] = {lb: [] for lb in ROLLING_LOOKBACKS}

    for day_back in range(eval_days):
        # end_idx is the last available SP for this "pretend today"
        end_idx = n - eval_sps + day_back * 48
        if end_idx < min_lb_sp + 48 or end_idx + 48 > n:
            continue

        actual_sp = target_v[end_idx: end_idx + 48]
        if len(actual_sp) < 48:
            continue

        sip_s  = sip_v[:end_idx]
        mip_s  = mip_v[:end_idx]
        dem_s  = dem_v[:end_idx] if dem_v is not None else None
        gen_s  = gen_v[:end_idx] if gen_v is not None else None
        exog_s = {k: v[:end_idx] for k, v in (exog_dict or {}).items()} or None

        try:
            xgb_fc  = xgb_intraday(xgb_trained,  sip_s, mip_s, dem_s, gen_s, exog_s)
        except Exception as exc:
            logger.debug("XGB intraday failed for day_back=%d: %s", day_back, exc)
            xgb_fc = {}

        try:
            lstm_fc = lstm_intraday(lstm_trained, sip_s, mip_s, dem_s, gen_s, exog_s)
        except Exception as exc:
            logger.debug("LSTM intraday failed for day_back=%d: %s", day_back, exc)
            lstm_fc = {}

        for lb_label in ROLLING_LOOKBACKS:
            xgb_arr  = xgb_fc.get(lb_label)
            lstm_arr = lstm_fc.get(lb_label)

            if xgb_arr is not None and not np.all(np.isnan(xgb_arr)):
                lb_xgb_preds[lb_label].append(xgb_arr.copy())
            if lstm_arr is not None and not np.all(np.isnan(lstm_arr)):
                lb_lstm_preds[lb_label].append(lstm_arr.copy())
            lb_actuals[lb_label].append(actual_sp.copy())

    # ── Score each lookback ───────────────────────────────────────────────────
    scores:          Dict[str, dict]  = {}
    xgb_mae_per_lb:  Dict[str, float] = {}
    lstm_mae_per_lb: Dict[str, float] = {}

    for lb_label in ROLLING_LOOKBACKS:
        actuals_list = lb_actuals[lb_label]
        if not actuals_list:
            logger.warning("eval_lookback_intraday: no actuals for lb=%s target=%s", lb_label, target)
            continue

        actual_arr = np.array(actuals_list, dtype=float)  # (N_days, 48)

        # Per-model MAE
        xgb_preds_list  = lb_xgb_preds[lb_label]
        lstm_preds_list = lb_lstm_preds[lb_label]

        xgb_mae = float("nan")
        if xgb_preds_list:
            xgb_arr  = np.array(xgb_preds_list, dtype=float)
            n_days   = min(len(xgb_arr), len(actual_arr))
            xgb_mae  = float(np.nanmean(np.abs(xgb_arr[:n_days] - actual_arr[:n_days])))
        xgb_mae_per_lb[lb_label] = xgb_mae

        lstm_mae = float("nan")
        if lstm_preds_list:
            lstm_arr = np.array(lstm_preds_list, dtype=float)
            n_days   = min(len(lstm_arr), len(actual_arr))
            lstm_mae = float(np.nanmean(np.abs(lstm_arr[:n_days] - actual_arr[:n_days])))
        lstm_mae_per_lb[lb_label] = lstm_mae

        # Build hybrid array for metric computation (equal weight; CRPS-selection is
        # about model *distribution*, not the weight tuning)
        if xgb_preds_list and lstm_preds_list:
            n_days     = min(len(xgb_preds_list), len(lstm_preds_list))
            hybrid_arr = 0.5 * np.array(xgb_preds_list[:n_days]) + \
                         0.5 * np.array(lstm_preds_list[:n_days])
            act_for_score = actual_arr[:n_days]
        elif xgb_preds_list:
            hybrid_arr    = np.array(xgb_preds_list, dtype=float)
            act_for_score = actual_arr[:len(xgb_preds_list)]
        elif lstm_preds_list:
            hybrid_arr    = np.array(lstm_preds_list, dtype=float)
            act_for_score = actual_arr[:len(lstm_preds_list)]
        else:
            logger.warning("eval_lookback_intraday: no predictions for lb=%s", lb_label)
            continue

        # Residuals (actual − prediction) for CRPS bootstrapping
        residuals = (act_for_score - hybrid_arr).flatten()
        pred_flat = hybrid_arr.flatten()

        # Approximate p10 / p90 via residual bootstrapping
        pred_p10_flat, pred_p90_flat = _residual_interval(pred_flat, residuals)
        pred_p10 = pred_p10_flat.reshape(hybrid_arr.shape)
        pred_p90 = pred_p90_flat.reshape(hybrid_arr.shape)

        metrics = eval_intraday_metrics(
            act_for_score, hybrid_arr, pred_p10, pred_p90, residuals=residuals,
        )
        scores[lb_label] = metrics
        logger.info(
            "  Intraday eval %s lb=%-8s  MAE=%.2f  CRPS=%.4f  spike_recall=%.3f",
            target, lb_label,
            metrics["mae_intraday"], metrics["crps"], metrics.get("spike_recall", float("nan")),
        )

    return scores, xgb_mae_per_lb, lstm_mae_per_lb


# ── Best-lookback selector ────────────────────────────────────────────────────

def select_best_lookback(scores: Dict[str, dict]) -> str:
    """
    Select the lookback with the lowest CRPS.
    Tiebreaker (within 1 % of best CRPS): highest spike_recall.

    Returns the label (e.g. "5 days") or the first key if scores is empty.
    """
    if not scores:
        return "1 day"

    best_crps     = min(v["crps"] for v in scores.values() if np.isfinite(v.get("crps", float("nan"))))
    crps_tol      = best_crps * 0.01            # 1 % tolerance for tiebreaker
    candidates    = {
        lb: v for lb, v in scores.items()
        if np.isfinite(v.get("crps", float("nan"))) and v["crps"] <= best_crps + crps_tol
    }

    if len(candidates) == 1:
        return next(iter(candidates))

    # Tiebreaker: highest spike_recall
    best_lb = max(
        candidates,
        key=lambda lb: candidates[lb].get("spike_recall", 0.0) or 0.0,
    )
    return best_lb
