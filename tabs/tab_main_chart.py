"""
Tab 1 — Main Page/Chart

Two display modes toggled by a radio button:

14-day Daily Forward
  Historical time series + daily-forward overlays.
  Today's forward forecast auto-shown on load.
  Clicking a row in the origins table (or a chart point) zooms to that origin.

Intraday 48-SP (Tomorrow)
  Half-hourly profile for tomorrow across all lookbacks.
  Best lookback (lowest CRPS) highlighted; P25–P75 band shown across lookbacks.
  Sub-model breakdown: XGB vs LSTM vs Hybrid.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import src.session_keys as sk
from src.config import SP_LABELS
from src.predictions import (
    PredictionSchemaError,
    load_forward_daily_predictions,
    load_intraday_predictions,
)

logger = logging.getLogger(__name__)

_TARGET_LABELS = {
    "sip":              "SIP (Imbalance Price) £/MWh",
    "mip":              "MIP (Wholesale Price) £/MWh",
    "demand":           "National Demand MW",
    "total_generation": "Total Generation MW",
}

_COLOURS = {
    "historical": "#4A5568",
    "forward":    "#00D4AA",
    "forecast":   "#FF6B6B",
    "realised":   "#FFE66D",
    "xgb":        "#4ECDC4",
    "lstm":       "#A29BFE",
}

PLOTLY_TEMPLATE = "plotly_dark"

_TODAY_ORIGIN = "TODAY"


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _build_historical_trace(series: pd.Series, faded: bool = False) -> go.Scatter:
    colour = _COLOURS["historical"] if faded else "#4ECDC4"
    return go.Scatter(
        x=series.index, y=series.values,
        mode="lines", name="Historical",
        line=dict(color=colour, width=1.5),
        opacity=0.3 if faded else 1.0,
    )


# ── 14-day forward helpers ─────────────────────────────────────────────────────

def _build_fan_traces(
    fan_df: pd.DataFrame,
    target: str,
    origin_date: pd.Timestamp,
) -> list:
    sub = fan_df[
        (fan_df["target"] == target) &
        (fan_df["origin_date"] == origin_date)
    ].copy()

    if sub.empty:
        return []

    grouped = (
        sub.groupby("horizon_days")
        .agg(
            hybrid_mean=("hybrid_prediction",   "mean"),
            hybrid_p25= ("hybrid_prediction",   lambda x: x.quantile(0.25)),
            hybrid_p75= ("hybrid_prediction",   lambda x: x.quantile(0.75)),
            forward_mean=("forward_curve_value", "mean"),
            realised=("realised_value",          "mean"),
        )
        .reset_index()
        .sort_values("horizon_days")
    )

    origin_ts = pd.Timestamp(origin_date)
    x_dates = [origin_ts + pd.Timedelta(days=int(d)) for d in grouped["horizon_days"]]

    traces = []
    traces.append(go.Scatter(
        x=x_dates + x_dates[::-1],
        y=list(grouped["hybrid_p75"]) + list(grouped["hybrid_p25"])[::-1],
        fill="toself", fillcolor="rgba(255,107,107,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Forecast P25–P75", showlegend=True, hoverinfo="skip",
    ))
    traces.append(go.Scatter(
        x=x_dates, y=grouped["forward_mean"],
        mode="lines", name="Forward Curve",
        line=dict(color=_COLOURS["forward"], width=2),
    ))
    traces.append(go.Scatter(
        x=x_dates, y=grouped["hybrid_mean"],
        mode="lines", name="Our Forecast",
        line=dict(color=_COLOURS["forecast"], width=2, dash="dot"),
    ))

    last = grouped.iloc[-1]
    traces.append(go.Scatter(
        x=[origin_ts + pd.Timedelta(days=int(last["horizon_days"]))],
        y=[last["realised"]],
        mode="markers", name="Realised",
        marker=dict(color=_COLOURS["realised"], size=10, symbol="circle"),
    ))
    return traces


def _build_today_forecast_traces(pred_df: pd.DataFrame, target: str) -> list:
    """
    Daily-forward traces with an intraday 48-SP shape overlay.

    Strategy:
      1. Average across lookbacks to get one daily scalar per horizon day.
      2. Use the intraday 48-SP predictions (settlement_period 1–48) to derive
         the intraday shape multiplier: shape[s] = intraday[s] / mean(intraday).
      3. Scale each day's scalar by the shape → full half-hourly waveform.
      4. If no intraday predictions exist, fall back to the daily-point trace.
    """
    forward_daily = load_forward_daily_predictions(
        target,
        pred_df,
        context=f"Market Overview daily-forward chart ({target})",
    )
    intraday_48sp = load_intraday_predictions(
        target,
        pred_df,
        context=f"Market Overview intraday shape overlay ({target})",
    )

    if forward_daily.empty:
        return []

    today = pd.Timestamp.today().normalize()

    # ── Daily scalar predictions (mean across lookbacks per horizon day) ───────
    daily = (
        forward_daily.groupby("horizon_days")["hybrid_prediction"]
        .agg(mean="mean", p25=lambda x: x.quantile(0.25), p75=lambda x: x.quantile(0.75))
        .reset_index()
        .sort_values("horizon_days")
    )

    # ── Intraday shape (48-element multiplier) ─────────────────────────────────
    intraday_shape: Optional[np.ndarray] = None
    if not intraday_48sp.empty:
        sp_mean = (
            intraday_48sp.groupby("settlement_period")["hybrid_prediction"]
            .mean()
            .sort_index()
        )
        if len(sp_mean) == 48:
            mu = float(sp_mean.mean())
            if mu > 1e-6:
                intraday_shape = sp_mean.values / mu  # shape[s] ≈ 1 on average

    traces = []

    if intraday_shape is not None:
        # ── Full half-hourly waveform ──────────────────────────────────────────
        x_wave, y_mean, y_p25, y_p75 = [], [], [], []

        for _, row in daily.iterrows():
            d   = int(row["horizon_days"])
            day_base = today + pd.Timedelta(days=d)
            for sp in range(48):
                dt = day_base + pd.Timedelta(minutes=sp * 30)
                x_wave.append(dt)
                y_mean.append(float(row["mean"]) * float(intraday_shape[sp]))
                y_p25.append(float(row["p25"])  * float(intraday_shape[sp]))
                y_p75.append(float(row["p75"])  * float(intraday_shape[sp]))

        traces.append(go.Scatter(
            x=x_wave + x_wave[::-1],
            y=y_p75 + y_p25[::-1],
            fill="toself", fillcolor="rgba(255,107,107,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Forecast P25–P75", showlegend=True, hoverinfo="skip",
        ))
        traces.append(go.Scatter(
            x=x_wave, y=y_mean,
            mode="lines", name="Daily forward (shaped by intraday 48-SP)",
            line=dict(color=_COLOURS["forecast"], width=1.8, dash="dot"),
        ))

    else:
        # ── Fallback: daily points only ────────────────────────────────────────
        x_dates = [today + pd.Timedelta(days=int(d)) for d in daily["horizon_days"]]
        traces.append(go.Scatter(
            x=x_dates + x_dates[::-1],
            y=list(daily["p75"]) + list(daily["p25"])[::-1],
            fill="toself", fillcolor="rgba(255,107,107,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Forecast P25–P75", showlegend=True, hoverinfo="skip",
        ))
        traces.append(go.Scatter(
            x=x_dates, y=daily["mean"].tolist(),
            mode="lines", name="Daily forward scalar",
            line=dict(color=_COLOURS["forecast"], width=2, dash="dot"),
        ))

    return traces


# ── Past-7-day intraday fan ────────────────────────────────────────────────────

def _build_past_intraday_fans(
    hist_series: pd.Series,
    pred_df: pd.DataFrame,
    target: str,
    n_days: int = 7,
) -> list:
    """
    For each of the past n_days, overlay an intraday waveform fan on the chart.

    Method: take today's intraday shape multiplier (SP[s] / mean) from the
    intraday predictions, then scale it by each past day's actual daily mean
    from hist_series.  This shows "what our model would have predicted for
    that day" as a shaded band alongside the actual data.

    Returns a list of Plotly traces (one band + one centre line per day).
    """
    intra = load_intraday_predictions(
        target,
        pred_df,
        context=f"Market Overview past intraday fans ({target})",
    )
    if intra.empty or hist_series is None or hist_series.empty:
        return []

    # Build intraday shape multiplier
    sp_mean_series = (
        intra.groupby("settlement_period")["hybrid_prediction"].mean().sort_index()
    )
    if len(sp_mean_series) != 48:
        return []
    mu = float(sp_mean_series.mean())
    if mu < 1e-6:
        return []
    shape      = sp_mean_series.values / mu          # (48,) — mean≈1
    shape_p25  = shape * float(
        intra.groupby("settlement_period")["hybrid_prediction"]
        .quantile(0.25).sort_index().values.mean()
    ) / mu if mu > 0 else shape
    shape_p75  = shape * float(
        intra.groupby("settlement_period")["hybrid_prediction"]
        .quantile(0.75).sort_index().values.mean()
    ) / mu if mu > 0 else shape

    # Re-derive per-SP p25/p75 shape (relative to mean)
    sp_p25 = intra.groupby("settlement_period")["hybrid_prediction"].quantile(0.25).sort_index().values
    sp_p75 = intra.groupby("settlement_period")["hybrid_prediction"].quantile(0.75).sort_index().values
    if len(sp_p25) != 48 or len(sp_p75) != 48:
        sp_p25 = sp_mean_series.values * 0.9
        sp_p75 = sp_mean_series.values * 1.1
    shape_p25 = sp_p25 / mu
    shape_p75 = sp_p75 / mu

    today = pd.Timestamp.today().normalize()
    traces = []

    for d in range(n_days, 0, -1):  # d=7 (oldest) .. d=1 (yesterday)
        day_start = today - pd.Timedelta(days=d)
        day_end   = day_start + pd.Timedelta(hours=24)

        day_actual = hist_series[
            (hist_series.index >= day_start) & (hist_series.index < day_end)
        ]
        if day_actual.empty:
            continue

        daily_mean = float(day_actual.mean())
        if daily_mean < 1e-6:
            continue

        # Build 48 half-hourly timestamps for this past day
        x_day = [day_start + pd.Timedelta(minutes=s * 30) for s in range(48)]
        y_mid  = [daily_mean * float(shape[s])    for s in range(48)]
        y_lo   = [daily_mean * float(shape_p25[s]) for s in range(48)]
        y_hi   = [daily_mean * float(shape_p75[s]) for s in range(48)]

        age_label = f"{d}d ago"
        alpha     = max(0.04, 0.10 - d * 0.01)   # fade older days slightly

        traces.append(go.Scatter(
            x=x_day + x_day[::-1],
            y=y_hi + y_lo[::-1],
            fill="toself",
            fillcolor=f"rgba(255,107,107,{alpha:.2f})",
            line=dict(color="rgba(0,0,0,0)"),
            name=f"Predicted band ({age_label})",
            legendgroup="past_fans",
            showlegend=(d == n_days),   # one legend entry for the group
            hoverinfo="skip",
        ))
        traces.append(go.Scatter(
            x=x_day, y=y_mid,
            mode="lines",
            name=f"Predicted ({age_label})",
            legendgroup="past_fans",
            showlegend=False,
            line=dict(color="rgba(255,107,107,0.45)", width=1.0, dash="dot"),
        ))

    return traces


# ── Intraday 48-SP helpers ─────────────────────────────────────────────────────

def _build_intraday_traces(pred_df: pd.DataFrame, target: str) -> tuple[go.Figure, pd.DataFrame]:
    """
    Build intraday figure and a per-lookback summary table.

    Returns (fig, summary_df).
    """
    intra = load_intraday_predictions(
        target,
        pred_df,
        context=f"Market Overview intraday chart ({target})",
    )
    fig = go.Figure()

    if intra.empty:
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            title="No intraday predictions available",
            height=480,
        )
        return fig, pd.DataFrame()

    sp_col = "settlement_period"
    lookbacks = sorted(intra["lookback"].unique())
    best_lb_flag = "is_best_lookback" in intra.columns

    # Identify best lookback (first row where flag is True, or first lookback)
    best_lb: Optional[str] = None
    if best_lb_flag:
        best_rows = intra[intra["is_best_lookback"] == True]
        if not best_rows.empty:
            best_lb = best_rows["lookback"].iloc[0]
    if best_lb is None and lookbacks:
        best_lb = lookbacks[0]

    # ── P25–P75 band across all lookbacks ─────────────────────────────────────
    band = (
        intra.groupby(sp_col)["hybrid_prediction"]
        .agg(p25=lambda x: x.quantile(0.25), p75=lambda x: x.quantile(0.75))
        .reset_index()
        .sort_values(sp_col)
    )
    sp_idxs = band[sp_col].values.astype(int) - 1  # 1-based → 0-based
    sp_xs   = [SP_LABELS[i] for i in sp_idxs if 0 <= i < 48]

    fig.add_trace(go.Scatter(
        x=sp_xs + sp_xs[::-1],
        y=list(band["p75"]) + list(band["p25"])[::-1],
        fill="toself", fillcolor="rgba(255,107,107,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Hybrid P25–P75 (all lookbacks)", hoverinfo="skip",
    ))

    # ── Per-lookback hybrid lines (non-best faded) ─────────────────────────────
    for lb in lookbacks:
        lb_df = (
            intra[intra["lookback"] == lb]
            .groupby(sp_col)["hybrid_prediction"]
            .mean()
            .reset_index()
            .sort_values(sp_col)
        )
        xs = [SP_LABELS[int(r[sp_col]) - 1] for _, r in lb_df.iterrows() if 0 <= int(r[sp_col]) - 1 < 48]
        ys = lb_df["hybrid_prediction"].tolist()

        is_best = lb == best_lb
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            name=f"Hybrid {lb}" + (" ★ best" if is_best else ""),
            line=dict(
                color=_COLOURS["forecast"] if is_best else "rgba(255,107,107,0.35)",
                width=2.5 if is_best else 1.0,
                dash="solid" if is_best else "dot",
            ),
        ))

    # ── XGB and LSTM for best lookback ────────────────────────────────────────
    if best_lb is not None:
        best_intra = intra[intra["lookback"] == best_lb].sort_values(sp_col)

        for col, label, colour, dash in [
            ("xgb_prediction",  f"XGBoost ({best_lb})",  _COLOURS["xgb"],  "dash"),
            ("lstm_prediction", f"LSTM ({best_lb})",      _COLOURS["lstm"], "longdash"),
        ]:
            if col not in best_intra.columns:
                continue
            sub = best_intra.groupby(sp_col)[col].mean().reset_index().sort_values(sp_col)
            xs = [SP_LABELS[int(r[sp_col]) - 1] for _, r in sub.iterrows() if 0 <= int(r[sp_col]) - 1 < 48]
            ys = sub[col].tolist()
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode="lines", name=label,
                line=dict(color=colour, width=1.2, dash=dash),
                visible="legendonly",   # hidden by default; toggle in legend
            ))

    # ── Summary table per lookback ─────────────────────────────────────────────
    rows = []
    for lb in lookbacks:
        lb_df = intra[intra["lookback"] == lb]
        hybrid = lb_df.groupby(sp_col)["hybrid_prediction"].mean()
        rows.append({
            "Lookback":     lb,
            "Best":         "★" if lb == best_lb else "",
            "Min (£/MWh)":  round(float(hybrid.min()),  2),
            "Mean":         round(float(hybrid.mean()), 2),
            "Max":          round(float(hybrid.max()),  2),
            "Peak SP":      int(hybrid.idxmax()),
        })
    summary_df = pd.DataFrame(rows)

    return fig, summary_df


def _xgb_importance_figure(fi_dict: dict, title: str) -> Optional[go.Figure]:
    """Horizontal bar chart of top-N XGB gain importances (normalised)."""
    if not fi_dict:
        return None
    top_n = 20
    items = sorted(fi_dict.items(), key=lambda x: -x[1])[:top_n]
    labels = [k for k, _ in items][::-1]
    vals = [v for _, v in items][::-1]
    fig = go.Figure(
        go.Bar(
            x=vals, y=labels, orientation="h",
            marker_color="#4ECDC4",
        )
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=max(280, 18 * len(labels)),
        title=title,
        margin=dict(l=140, r=16, t=36, b=28),
        xaxis_title="Relative importance (gain, normalised)",
        showlegend=False,
    )
    return fig


def _importance_bar_figure(fi_dict: dict, title: str, colour: str, xaxis_title: str) -> Optional[go.Figure]:
    """Horizontal bar chart for SHAP / attribution summaries."""
    if not fi_dict:
        return None
    top_n = 20
    items = sorted(fi_dict.items(), key=lambda x: -x[1])[:top_n]
    labels = [k for k, _ in items][::-1]
    vals = [v for _, v in items][::-1]
    fig = go.Figure(
        go.Bar(
            x=vals, y=labels, orientation="h",
            marker_color=colour,
        )
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=max(280, 18 * len(labels)),
        title=title,
        margin=dict(l=170, r=16, t=36, b=28),
        xaxis_title=xaxis_title,
        showlegend=False,
    )
    return fig


# ── Main render ────────────────────────────────────────────────────────────────

def render() -> None:
    st.header("Market Overview")

    predictions_loaded: bool = st.session_state.get(sk.PREDICTIONS_LOADED, False)

    # ── Controls row ──────────────────────────────────────────────────────────
    col_target, col_mode, col_info = st.columns([3, 3, 4])
    with col_target:
        target = st.selectbox(
            "Variable",
            options=["sip", "mip", "demand", "total_generation"],
            format_func=lambda t: _TARGET_LABELS.get(t, t),
            key=sk.SELECTED_TARGET,
        )
    with col_mode:
        mode = st.radio(
            "Forecast mode",
            options=["14-day Daily Forward", "Intraday 48-SP (Tomorrow)"],
            horizontal=True,
            key=sk.FORECAST_MODE,
        )

    meta_all = st.session_state.get(sk.METADATA) or {}
    tmeta = meta_all.get("targets", {}).get(target, {})
    fi_mean = tmeta.get("xgb_feature_importance") or {}
    fi_intra = tmeta.get("xgb_feature_importance_intraday") or {}
    if fi_mean or fi_intra:
        with st.expander(
            "XGBoost feature importance (gain) — diagnosing flat or smooth forecasts",
            expanded=False,
        ):
            fc1, fc2 = st.columns(2)
            with fc1:
                fig_m = _xgb_importance_figure(fi_mean, "Mean across horizon models")
                if fig_m:
                    st.plotly_chart(fig_m, width="stretch")
                else:
                    st.caption("No aggregate importance in metadata — re-run training + predict.")
            with fc2:
                fig_i = _xgb_importance_figure(
                    fi_intra,
                    "1-day horizon (48 SPs), best intraday lookback",
                )
                if fig_i:
                    st.plotly_chart(fig_i, width="stretch")
                else:
                    st.caption("Intraday slice not available until you retrain with an updated pipeline.")
            st.caption(
                "If cyclical encodings (sp_sin, h_sin) and short lags dominate, the XGB path "
                "tracks a smooth daily shape; weak exogenous weights suggest little signal from weather or cross-series features."
            )

    xgb_shap_mean = tmeta.get("xgb_shap_importance") or {}
    xgb_shap_intra = tmeta.get("xgb_shap_importance_intraday") or {}
    lstm_attr_mean = tmeta.get("lstm_feature_attribution") or {}
    lstm_attr_intra = tmeta.get("lstm_feature_attribution_intraday") or {}
    if xgb_shap_mean or xgb_shap_intra or lstm_attr_mean or lstm_attr_intra:
        with st.expander(
            "SHAP and LSTM attribution diagnostics",
            expanded=False,
        ):
            st.caption(
                "XGBoost uses mean absolute SHAP-style contributions from the fitted trees. "
                "LSTM uses integrated-gradients channel attribution aggregated across sequence time steps."
            )
            row1_col1, row1_col2 = st.columns(2)
            with row1_col1:
                fig_sm = _importance_bar_figure(
                    xgb_shap_mean,
                    "XGBoost SHAP summary — mean across horizon models",
                    colour="#00D4AA",
                    xaxis_title="Mean |SHAP contribution| (normalised)",
                )
                if fig_sm:
                    st.plotly_chart(fig_sm, width="stretch")
                else:
                    st.caption("No aggregate XGBoost SHAP summary in metadata yet.")
            with row1_col2:
                fig_si = _importance_bar_figure(
                    xgb_shap_intra,
                    "XGBoost SHAP summary — 1-day horizon, best intraday lookback",
                    colour="#4ECDC4",
                    xaxis_title="Mean |SHAP contribution| (normalised)",
                )
                if fig_si:
                    st.plotly_chart(fig_si, width="stretch")
                else:
                    st.caption("No intraday XGBoost SHAP summary in metadata yet.")

            row2_col1, row2_col2 = st.columns(2)
            with row2_col1:
                fig_lm = _importance_bar_figure(
                    lstm_attr_mean,
                    "LSTM attribution — mean across horizon models",
                    colour="#A29BFE",
                    xaxis_title="Integrated-gradients attribution (normalised)",
                )
                if fig_lm:
                    st.plotly_chart(fig_lm, width="stretch")
                else:
                    st.caption("No aggregate LSTM attribution in metadata yet.")
            with row2_col2:
                fig_li = _importance_bar_figure(
                    lstm_attr_intra,
                    "LSTM attribution — 1-day horizon, best intraday lookback",
                    colour="#B8A4FF",
                    xaxis_title="Integrated-gradients attribution (normalised)",
                )
                if fig_li:
                    st.plotly_chart(fig_li, width="stretch")
                else:
                    st.caption("No intraday LSTM attribution in metadata yet.")

            st.caption(
                "Calendar channels are prefixed with `calendar_`. If those rise in the rankings, the model is leaning "
                "more on time structure; if weather or cross-series channels rise, the forecast is reacting more to "
                "exogenous conditions."
            )

    if not predictions_loaded:
        st.info(
            "No prediction files found. "
            "Run `python -m backend.predict` to generate them."
        )

    # ── Load prediction data ───────────────────────────────────────────────────
    pred_key_map = {
        "sip":              sk.PRED_SIP,
        "mip":              sk.PRED_MIP,
        "demand":           sk.PRED_DEMAND,
        "total_generation": sk.PRED_GEN,
    }
    pred_df: Optional[pd.DataFrame] = st.session_state.get(pred_key_map[target])
    fan_df:  Optional[pd.DataFrame] = st.session_state.get(sk.FAN_DATA)

    # ── Historical series ──────────────────────────────────────────────────────
    hist_series_key_map = {
        "sip":              sk.SIP_DF,
        "mip":              sk.MIP_DF,
        "demand":           sk.DEMAND_DF,
        "total_generation": sk.GEN_DF,
    }
    hist_df: Optional[pd.DataFrame] = st.session_state.get(hist_series_key_map[target])

    hist_series: Optional[pd.Series] = None
    if hist_df is not None and not hist_df.empty:
        try:
            hdf = hist_df.copy()
            hdf["datetime"] = pd.to_datetime(hdf["settlementDate"]) + pd.to_timedelta(
                (hdf["settlementPeriod"].astype(int) - 1) * 30, unit="min"
            )
            if target == "sip" and "systemBuyPrice" in hdf.columns:
                hist_series = hdf.set_index("datetime")["systemBuyPrice"].sort_index()
            elif target == "mip" and "price" in hdf.columns:
                hist_series = hdf.set_index("datetime")["price"].sort_index()
            elif target == "demand":
                col = "initialDemandOutturn" if "initialDemandOutturn" in hdf.columns else hdf.columns[-1]
                hist_series = hdf.set_index("datetime")[col].sort_index()
            elif target == "total_generation" and "generation" in hdf.columns:
                hist_series = hdf.groupby("datetime")["generation"].sum().sort_index()
        except Exception as exc:
            logger.warning("Could not build historical series for %s: %s", target, exc)

    # ══════════════════════════════════════════════════════════════════════════
    # INTRADAY MODE
    # ══════════════════════════════════════════════════════════════════════════
    if mode == "Intraday 48-SP (Tomorrow)":
        if pred_df is None or pred_df.empty:
            st.warning("No predictions loaded. Run `python -m backend.predict` first.")
            return

        # ── Fan controls sidebar (right column) ───────────────────────────
        chart_col, ctrl_col = st.columns([5, 1])

        with ctrl_col:
            st.caption("Fan controls")
            show_xgb  = st.checkbox("Show XGB",  value=False, key="fan_ctrl_xgb")
            show_lstm = st.checkbox("Show LSTM", value=False, key="fan_ctrl_lstm")
            n_past    = st.selectbox("Past days", [0, 1, 2, 3, 7], index=2, key="fan_ctrl_past")

        try:
            fig, summary_df = _build_intraday_traces(pred_df, target)
        except PredictionSchemaError as exc:
            st.warning(str(exc))
            logger.warning("%s", exc)
            return

        # ── Overlay past N days of actual half-hourly data ─────────────────
        if n_past > 0 and hist_series is not None and not hist_series.empty:
            today = pd.Timestamp.today().normalize()
            past_colours = ["#FFE66D", "#00D4AA", "#A29BFE", "#FF6B6B", "#4ECDC4", "#FFA07A", "#87CEEB"]
            for d in range(n_past, 0, -1):
                day_start = today - pd.Timedelta(days=d)
                day_end   = day_start + pd.Timedelta(hours=24)
                day_slice = hist_series[(hist_series.index >= day_start) & (hist_series.index < day_end)]
                if day_slice.empty:
                    continue
                # Re-index to 48 SPs (some days may have gaps)
                sp_vals: list = [None] * 48
                for ts, val in day_slice.items():
                    sp_idx = int((ts - day_start).total_seconds() // 1800)
                    if 0 <= sp_idx < 48:
                        sp_vals[sp_idx] = float(val)
                xs = [SP_LABELS[i] for i in range(48) if sp_vals[i] is not None]
                ys = [v for v in sp_vals if v is not None]
                colour = past_colours[(d - 1) % len(past_colours)]
                alpha  = max(0.4, 0.7 - d * 0.05)
                label_text = f"Actual {d}d ago ({day_start.strftime('%a %d %b')})"
                fig.add_trace(go.Scatter(
                    x=xs, y=ys,
                    mode="lines",
                    name=label_text,
                    line=dict(color=colour, width=1.2, dash="dot"),
                    opacity=alpha,
                ))

        # ── Toggle XGB / LSTM trace visibility ────────────────────────────
        if not show_xgb or not show_lstm:
            for trace in fig.data:
                name = trace.name or ""
                if not show_xgb and name.startswith("XGBoost"):
                    trace.visible = "legendonly"
                if not show_lstm and name.startswith("LSTM"):
                    trace.visible = "legendonly"

        tomorrow = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            title=f"Intraday Forecast — {_TARGET_LABELS.get(target, target)} | Tomorrow {tomorrow.strftime('%Y-%m-%d')}",
            xaxis_title="Settlement Period (half-hour)",
            yaxis_title=_TARGET_LABELS.get(target, target),
            height=480,
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
            hovermode="x unified",
            xaxis=dict(
                tickmode="array",
                tickvals=SP_LABELS[::4],
                ticktext=SP_LABELS[::4],
                tickangle=-45,
            ),
        )

        with chart_col:
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "Solid coloured line = best lookback (lowest CRPS). "
                "Faint dotted lines = other lookbacks. "
                "Shaded band = P25–P75 across all lookbacks. "
                "XGB / LSTM sub-models for best lookback are toggled in the legend."
            )

        if not summary_df.empty:
            with st.expander("Per-lookback summary", expanded=False):
                st.dataframe(summary_df, width="stretch", hide_index=True)

        return

    # ══════════════════════════════════════════════════════════════════════════
    # 14-DAY DAILY-FORWARD MODE
    # ══════════════════════════════════════════════════════════════════════════

    # Auto-select today's forecast on first load
    selected_origin = st.session_state.get(sk.SELECTED_ORIGIN)
    if selected_origin is None and pred_df is not None and not pred_df.empty:
        st.session_state[sk.SELECTED_ORIGIN] = _TODAY_ORIGIN
        selected_origin = _TODAY_ORIGIN

    is_today = selected_origin == _TODAY_ORIGIN

    fig = go.Figure()

    faded = selected_origin is not None
    if hist_series is not None and not hist_series.empty:
        fig.add_trace(_build_historical_trace(hist_series, faded=faded))

    origin_ts: Optional[pd.Timestamp] = None
    zoom_start: Optional[pd.Timestamp] = None
    zoom_end:   Optional[pd.Timestamp] = None

    if is_today:
        origin_ts = pd.Timestamp.today().normalize()
        # Show the past 7 days + 14 days forward so the intraday fans are visible
        zoom_start = origin_ts - pd.Timedelta(days=7)
        zoom_end   = origin_ts + pd.Timedelta(days=16)

        if pred_df is not None and not pred_df.empty:
            try:
                for trace in _build_past_intraday_fans(hist_series, pred_df, target, n_days=7):
                    fig.add_trace(trace)
                for trace in _build_today_forecast_traces(pred_df, target):
                    fig.add_trace(trace)
            except PredictionSchemaError as exc:
                st.warning(str(exc))
                logger.warning("%s", exc)

    elif selected_origin is not None:
        origin_ts  = pd.Timestamp(selected_origin)
        zoom_start = origin_ts - pd.Timedelta(days=30)
        zoom_end   = origin_ts + pd.Timedelta(days=16)

        if fan_df is not None and not fan_df.empty:
            for trace in _build_fan_traces(fan_df, target, origin_ts):
                fig.add_trace(trace)

    # Vertical origin line
    if origin_ts is not None:
        label  = "Today — Forward Forecast" if is_today else f"Origin: {origin_ts.strftime('%Y-%m-%d')}"
        x_iso  = origin_ts.isoformat()
        fig.add_shape(
            type="line", xref="x", yref="paper",
            x0=x_iso, x1=x_iso, y0=0, y1=1,
            line=dict(color="#95A5A6", dash="dot", width=1),
        )
        fig.add_annotation(
            x=x_iso, yref="paper", y=1.02,
            text=label, showarrow=False,
            font=dict(color="#95A5A6", size=11),
            xanchor="left",
        )

    xaxis_range = [zoom_start, zoom_end] if (zoom_start and zoom_end) else None

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=_TARGET_LABELS.get(target, target),
        xaxis_title="Date",
        yaxis_title=_TARGET_LABELS.get(target, target),
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        hovermode="x unified",
        xaxis=dict(range=xaxis_range) if xaxis_range else {},
    )

    event = st.plotly_chart(
        fig, width="stretch",
        on_select="rerun",
        key=f"main_chart_{target}",
    )

    if event and hasattr(event, "selection") and event.selection:
        pts = event.selection.get("points", [])
        if pts:
            clicked_x = pts[0].get("x")
            if clicked_x:
                try:
                    st.session_state[sk.SELECTED_ORIGIN] = pd.Timestamp(clicked_x)
                    st.rerun()
                except Exception:
                    pass

    # ── Controls ──────────────────────────────────────────────────────────────
    col_clear, col_info2 = st.columns([2, 8])
    with col_clear:
        if selected_origin is not None:
            if st.button("Show full history", key="clear_origin"):
                st.session_state[sk.SELECTED_ORIGIN] = None
                st.rerun()

    if selected_origin is not None:
        with col_info2:
            if is_today:
                st.info("**Today's forecast** | Dotted = Our hybrid · Shaded = P25–P75 range across lookbacks")
            else:
                st.info(
                    f"**Origin:** {pd.Timestamp(selected_origin).strftime('%Y-%m-%d')}  "
                    "| Solid = Market forward · Dotted = Our hybrid · Yellow dot = Realised"
                )

    # ── Clickable origins table ────────────────────────────────────────────────
    if fan_df is not None and not fan_df.empty:
        with st.expander("Fan chart origins — click a row to zoom", expanded=False):
            origins = sorted(
                fan_df[fan_df["target"] == target]["origin_date"].unique(),
                reverse=True,
            )
            origins_df = pd.DataFrame({
                "Origin Date": [pd.Timestamp(o).strftime("%Y-%m-%d") for o in origins],
                "Horizon":     ["14 days forward"] * len(origins),
            })

            selection = st.dataframe(
                origins_df,
                width="stretch",
                height=250,
                on_select="rerun",
                selection_mode="single-row",
                key=f"origins_table_{target}",
            )

            if selection and selection.selection.get("rows"):
                row_idx = selection.selection["rows"][0]
                chosen  = pd.Timestamp(origins[row_idx])
                if st.session_state.get(sk.SELECTED_ORIGIN) != chosen:
                    st.session_state[sk.SELECTED_ORIGIN] = chosen
                    st.rerun()
