"""
Tab 5 — UK Demand Map Visualisation

Shows:
  - National demand vs generation (surplus/deficit)
  - Intraday 48-SP demand shape vs latest actual day
  - Daily-forward demand view kept separate from half-hourly charts
  - Weather overlay (wind speed, temperature)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import src.session_keys as sk
from src.config import DEFAULT_DA_PRICE, SP_LABELS
from src.predictions import (
    PredictionSchemaError,
    load_forward_daily_predictions,
    load_intraday_predictions,
)

logger = logging.getLogger(__name__)

PLOTLY_TEMPLATE = "plotly_dark"

# GB region bounding boxes (approximate centroids for scatter map)
_GB_REGIONS = pd.DataFrame([
    {"region": "Scotland",      "lat": 56.5, "lon": -4.0},
    {"region": "North England", "lat": 53.8, "lon": -1.7},
    {"region": "Midlands",      "lat": 52.5, "lon": -1.8},
    {"region": "Wales",         "lat": 52.1, "lon": -3.8},
    {"region": "East England",  "lat": 52.3, "lon":  0.5},
    {"region": "South West",    "lat": 51.0, "lon": -3.2},
    {"region": "South East",    "lat": 51.4, "lon":  0.2},
    {"region": "London",        "lat": 51.5, "lon": -0.1},
])

# Approximate regional demand fractions (from ELEXON LDRS distribution)
_REGIONAL_FRACTIONS = {
    "Scotland":      0.09,
    "North England": 0.12,
    "Midlands":      0.15,
    "Wales":         0.06,
    "East England":  0.11,
    "South West":    0.08,
    "South East":    0.16,
    "London":        0.23,
}


def _build_demand_series(demand_df: Optional[pd.DataFrame]) -> Optional[pd.Series]:
    if demand_df is None or demand_df.empty:
        return None
    try:
        df = demand_df.copy()
        df["datetime"] = pd.to_datetime(df["settlementDate"]) + pd.to_timedelta(
            (df["settlementPeriod"].astype(int) - 1) * 30, unit="min"
        )
        col = "initialDemandOutturn" if "initialDemandOutturn" in df.columns else df.columns[-1]
        series = df.groupby("datetime")[col].mean().sort_index()
        duplicate_count = len(df) - len(series)
        if duplicate_count > 0:
            logger.warning(
                "Collapsed %s duplicate demand timestamp rows while building the actual demand series.",
                duplicate_count,
            )
        return series
    except Exception as exc:
        logger.warning("Could not build demand series: %s", exc)
        return None


def _build_gen_series(gen_df: Optional[pd.DataFrame]) -> Optional[pd.Series]:
    if gen_df is None or gen_df.empty:
        return None
    try:
        df = gen_df.copy()
        df["datetime"] = pd.to_datetime(df["settlementDate"]) + pd.to_timedelta(
            (df["settlementPeriod"].astype(int) - 1) * 30, unit="min"
        )
        return df.groupby("datetime")["generation"].sum().sort_index()
    except Exception as exc:
        logger.warning("Could not build gen series: %s", exc)
        return None


def _render_demand_map(demand_mw: float) -> None:
    """Scatter map with regional demand estimates."""
    regions = _GB_REGIONS.copy()
    regions["demand_mw"] = regions["region"].map(
        {r: demand_mw * f for r, f in _REGIONAL_FRACTIONS.items()}
    )
    regions["size"] = (regions["demand_mw"] / regions["demand_mw"].max() * 40 + 5).round(1)

    fig = go.Figure(go.Scattermapbox(
        lat=regions["lat"],
        lon=regions["lon"],
        mode="markers+text",
        marker=dict(
            size=regions["size"],
            color=regions["demand_mw"],
            colorscale="YlOrRd",
            showscale=True,
            colorbar=dict(title="MW"),
        ),
        text=regions["region"],
        textposition="top center",
        hovertemplate="<b>%{text}</b><br>%{customdata:.0f} MW<extra></extra>",
        customdata=regions["demand_mw"],
    ))

    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=54.0, lon=-2.5),
            zoom=4.5,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        title="GB Regional Demand Estimate (MW)",
        height=480,
    )
    st.plotly_chart(fig, width="stretch")


def _latest_complete_day(series: Optional[pd.Series]) -> tuple[Optional[pd.Timestamp], Optional[pd.Series]]:
    """Return the latest day with a full 48-settlement-period history."""
    if series is None or series.empty:
        return None, None

    aligned = series.groupby(level=0).mean().sort_index()
    counts = aligned.groupby(aligned.index.normalize()).size()
    full_days = counts[counts >= 48].index.sort_values()
    if len(full_days) == 0:
        return None, None

    day_start = pd.Timestamp(full_days[-1])
    expected_index = pd.date_range(day_start, periods=48, freq="30min")
    day_slice = aligned.reindex(expected_index)
    if day_slice.isna().any():
        return None, None

    return day_start, day_slice


def render() -> None:
    st.header("UK Demand Map & Generation Overview")

    demand_df = st.session_state.get(sk.DEMAND_DF)
    gen_df    = st.session_state.get(sk.GEN_DF)
    pred_dem  = st.session_state.get(sk.PRED_DEMAND)

    demand_series = _build_demand_series(demand_df)
    gen_series    = _build_gen_series(gen_df)

    # ── Map: latest national demand ───────────────────────────────────────────
    if demand_series is not None and not demand_series.empty:
        latest_demand = float(demand_series.iloc[-1])
    else:
        latest_demand = 32_000.0  # approximate GB average

    st.subheader("Regional Demand Distribution")
    _render_demand_map(latest_demand)

    # ── Demand vs Generation time series ─────────────────────────────────────
    st.subheader("Demand vs Generation — Surplus / Deficit")

    if demand_series is not None and gen_series is not None:
        common_idx = demand_series.index.intersection(gen_series.index)
        if len(common_idx) > 0:
            dem_aligned = demand_series.loc[common_idx]
            gen_aligned = gen_series.loc[common_idx]
            balance     = gen_aligned - dem_aligned  # + = surplus, - = deficit

            # Show last 7 days
            cutoff = common_idx[-1] - pd.Timedelta(days=7)
            mask   = common_idx >= cutoff
            x      = common_idx[mask]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x, y=dem_aligned.loc[x],
                mode="lines", name="Demand",
                line=dict(color="#FF6B6B", width=1.5),
            ))
            fig.add_trace(go.Scatter(
                x=x, y=gen_aligned.loc[x],
                mode="lines", name="Generation",
                line=dict(color="#00D4AA", width=1.5),
            ))
            fig.add_trace(go.Scatter(
                x=list(x) + list(x[::-1]),
                y=list(np.maximum(balance.loc[x], 0)) + [0] * len(x),
                fill="toself",
                fillcolor="rgba(0,212,170,0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Surplus",
                hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=list(x) + list(x[::-1]),
                y=[0] * len(x) + list(np.minimum(balance.loc[x], 0)[::-1]),
                fill="toself",
                fillcolor="rgba(255,107,107,0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Deficit",
                hoverinfo="skip",
            ))

            fig.update_layout(
                template=PLOTLY_TEMPLATE,
                title="Demand vs Generation (last 7 days)",
                xaxis_title="Date",
                yaxis_title="MW",
                height=380,
            )
            st.plotly_chart(fig, width="stretch")
    else:
        st.info("Demand and generation data not yet loaded. The app will fetch it from ELEXON on startup.")

    # ── Demand forecast comparison ────────────────────────────────────────────
    st.subheader("Demand Forecast Views")

    if pred_dem is not None and not pred_dem.empty and demand_series is not None:
        try:
            intraday_48sp = load_intraday_predictions(
                "demand",
                pred_dem,
                context="Demand Map half-hourly demand chart",
            )
            forward_daily = load_forward_daily_predictions(
                "demand",
                pred_dem,
                context="Demand Map daily forward chart",
            )
        except PredictionSchemaError as exc:
            logger.warning("%s", exc)
            st.warning(str(exc))
            intraday_48sp = pd.DataFrame()
            forward_daily = pd.DataFrame()

        latest_day_start, latest_day_actuals = _latest_complete_day(demand_series)

        if not intraday_48sp.empty:
            intraday_summary = (
                intraday_48sp.groupby("settlement_period")["hybrid_prediction"]
                .agg(mean="mean", p25=lambda x: x.quantile(0.25), p75=lambda x: x.quantile(0.75))
                .reset_index()
                .sort_values("settlement_period")
            )

            fig2 = go.Figure()
            if latest_day_start is not None and latest_day_actuals is not None:
                fig2.add_trace(go.Scatter(
                    x=SP_LABELS,
                    y=latest_day_actuals.values,
                    mode="lines",
                    name=f"Actuals ({latest_day_start.strftime('%Y-%m-%d')})",
                    line=dict(color="#FFE66D", width=1.5),
                ))

            fig2.add_trace(go.Scatter(
                x=SP_LABELS + SP_LABELS[::-1],
                y=list(intraday_summary["p75"]) + list(intraday_summary["p25"])[::-1],
                fill="toself",
                fillcolor="rgba(0,212,170,0.12)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Hybrid intraday P25–P75",
                hoverinfo="skip",
            ))
            fig2.add_trace(go.Scatter(
                x=SP_LABELS,
                y=intraday_summary["mean"],
                mode="lines",
                name="Hybrid intraday 48-SP forecast",
                line=dict(color="#00D4AA", dash="dot", width=2),
            ))
            fig2.update_layout(
                template=PLOTLY_TEMPLATE,
                title="Tomorrow Demand Shape — Intraday 48-SP Forecast vs Latest Actual Day",
                xaxis_title="Settlement Period (half-hour)",
                yaxis_title="MW",
                height=360,
                xaxis=dict(
                    tickmode="array",
                    tickvals=SP_LABELS[::4],
                    ticktext=SP_LABELS[::4],
                    tickangle=-45,
                ),
            )
            st.plotly_chart(fig2, width="stretch")
            st.caption(
                "This half-hourly demand comparison uses only the intraday 48-SP product. "
                "Daily-forward rows with settlement_period = 0 are blocked from this chart."
            )

        if not forward_daily.empty:
            forward_daily_avg = (
                forward_daily.groupby("forecast_date")["hybrid_prediction"]
                .mean()
                .reset_index()
                .sort_values("forecast_date")
            )
            actual_daily_mean = demand_series.resample("D").mean().dropna().tail(30)

            fig3 = go.Figure()
            if not actual_daily_mean.empty:
                fig3.add_trace(go.Scatter(
                    x=actual_daily_mean.index,
                    y=actual_daily_mean.values,
                    mode="lines",
                    name="Actual daily mean",
                    line=dict(color="#FFE66D", width=1.5),
                ))
            fig3.add_trace(go.Scatter(
                x=forward_daily_avg["forecast_date"],
                y=forward_daily_avg["hybrid_prediction"],
                mode="lines+markers",
                name="Hybrid daily-forward scalar",
                line=dict(color="#00D4AA", dash="dot", width=2),
            ))
            fig3.update_layout(
                template=PLOTLY_TEMPLATE,
                title="Daily Forward Demand View",
                xaxis_title="Date",
                yaxis_title="MW",
                height=320,
            )
            st.plotly_chart(fig3, width="stretch")
            st.caption(
                "This chart shows the daily-forward product separately. It remains a one-value-per-day "
                "view and is not treated as a native half-hourly demand curve."
            )
    else:
        st.info("No demand predictions available. Run `python -m backend.predict` to generate.")

    # ── Weather overlay ───────────────────────────────────────────────────────
    with st.expander("Weather overlay (wind speed & temperature)", expanded=False):
        st.caption(
            "Weather data is fetched live from Open-Meteo (ERA5 + NWP forecast). "
            "Run `python -m backend.predict` to include weather-informed predictions."
        )

        from src.data.weather_client import fetch_weather_data
        import datetime as dt

        today = dt.date.today()
        date_from = today - dt.timedelta(days=7)

        try:
            wx = fetch_weather_data(date_from, today + dt.timedelta(days=3))
            if not wx.empty:
                fig3 = go.Figure()
                if "temperature_c" in wx.columns:
                    fig3.add_trace(go.Scatter(
                        x=wx.index, y=wx["temperature_c"],
                        mode="lines", name="Temperature (°C)",
                        line=dict(color="#FF6B6B"),
                        yaxis="y",
                    ))
                if "wind_speed_100m" in wx.columns:
                    fig3.add_trace(go.Scatter(
                        x=wx.index, y=wx["wind_speed_100m"],
                        mode="lines", name="Wind 100m (km/h)",
                        line=dict(color="#00D4AA"),
                        yaxis="y2",
                    ))
                fig3.update_layout(
                    template=PLOTLY_TEMPLATE,
                    title="GB Weather (last 7 days + 3 day forecast)",
                    yaxis=dict(title="Temperature (°C)"),
                    yaxis2=dict(title="Wind Speed (km/h)", overlaying="y", side="right"),
                    height=320,
                )
                st.plotly_chart(fig3, width="stretch")
        except Exception as exc:
            st.warning(f"Could not fetch weather data: {exc}")
