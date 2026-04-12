"""
Tab 5 — UK Demand Map Visualisation

Shows:
  - National demand vs generation (surplus/deficit)
  - ELEXON NDFD demand forecast vs our hybrid vs actuals
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
from src.config import DEFAULT_DA_PRICE

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
        return df.set_index("datetime")[col].sort_index()
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
    st.subheader("Demand Forecast: ELEXON vs Our Hybrid vs Actuals")

    if pred_dem is not None and not pred_dem.empty and demand_series is not None:
        # Forward predictions averaged across lookbacks
        fwd = pred_dem[pred_dem["prediction_type"] == "forward"].copy()
        if not fwd.empty:
            fwd_avg = fwd.groupby("forecast_date")["hybrid_prediction"].mean().reset_index()
            fwd_avg = fwd_avg.sort_values("forecast_date")

            # Last 30 days of actuals
            cutoff = demand_series.index.max() - pd.Timedelta(days=30)
            actuals_last = demand_series.loc[demand_series.index >= cutoff] if len(demand_series) > 0 else demand_series

            fig2 = go.Figure()
            if not actuals_last.empty:
                fig2.add_trace(go.Scatter(
                    x=actuals_last.index, y=actuals_last.values,
                    mode="lines", name="Actuals",
                    line=dict(color="#FFE66D", width=1.5),
                ))
            fig2.add_trace(go.Scatter(
                x=fwd_avg["forecast_date"], y=fwd_avg["hybrid_prediction"],
                mode="lines", name="Our Hybrid Forecast",
                line=dict(color="#00D4AA", dash="dot", width=2),
            ))
            fig2.update_layout(
                template=PLOTLY_TEMPLATE,
                title="Demand Forecast (MW)",
                xaxis_title="Date",
                yaxis_title="MW",
                height=350,
            )
            st.plotly_chart(fig2, width="stretch")
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
