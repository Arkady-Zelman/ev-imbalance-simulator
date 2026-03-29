"""Historical SIP Explorer tab -- time series, distribution, extremes, rolling stats."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import COLOUR_PRIMARY, PLOTLY_TEMPLATE, SP_LABELS
from src.session_keys import SIP_DF
from src.visualization.charts import sip_distribution, sip_time_series
from src.visualization.heatmaps import rolling_stats_chart


def render(has_results: bool) -> None:
    st.header("Historical SIP Explorer")

    if SIP_DF not in st.session_state or st.session_state[SIP_DF].empty:
        st.info("Run a simulation (which fetches ELEXON data) to explore historical SIP.")
        return

    sip_df: pd.DataFrame = st.session_state[SIP_DF].copy()
    col = "systemBuyPrice" if "systemBuyPrice" in sip_df.columns else "systemSellPrice"

    if col not in sip_df.columns:
        st.warning("SIP price column not found in data.")
        return

    sip_df["datetime"] = pd.to_datetime(sip_df["settlementDate"])
    sip_df.sort_values(["datetime", "settlementPeriod"], inplace=True)

    prices = sip_df[col].dropna().values
    dates = sip_df["datetime"].values

    # ── Time series ───────────────────────────────────────────────────
    st.subheader("SIP Time Series")
    fig_ts = sip_time_series(dates, prices)
    st.plotly_chart(fig_ts, use_container_width=True)

    # ── Distribution ──────────────────────────────────────────────────
    st.subheader("SIP Distribution")
    col_a, col_b = st.columns(2)
    with col_a:
        fig_dist = sip_distribution(prices)
        st.plotly_chart(fig_dist, use_container_width=True)
    with col_b:
        st.markdown("**Descriptive Statistics**")
        stats = {
            "Count": f"{len(prices):,}",
            "Mean": f"£{np.mean(prices):,.2f}",
            "Median": f"£{np.median(prices):,.2f}",
            "Std Dev": f"£{np.std(prices):,.2f}",
            "Min": f"£{np.min(prices):,.2f}",
            "Max": f"£{np.max(prices):,.2f}",
            "P5": f"£{np.percentile(prices, 5):,.2f}",
            "P95": f"£{np.percentile(prices, 95):,.2f}",
            "> £500/MWh": f"{(prices > 500).sum():,} ({(prices > 500).mean():.2%})",
            "> £1,000/MWh": f"{(prices > 1000).sum():,} ({(prices > 1000).mean():.2%})",
            "Negative": f"{(prices < 0).sum():,} ({(prices < 0).mean():.2%})",
        }
        st.dataframe(pd.DataFrame(stats.items(), columns=["Metric", "Value"]),
                      use_container_width=True, hide_index=True)

    # ── Extreme events table ──────────────────────────────────────────
    st.markdown("---")
    st.subheader("Top 20 SIP Spikes")
    top_spikes = (
        sip_df.nlargest(20, col)
        [["settlementDate", "settlementPeriod", col]]
        .rename(columns={col: "SIP (£/MWh)"})
        .reset_index(drop=True)
    )
    if "netImbalanceVolume" in sip_df.columns:
        niv_map = sip_df.set_index(["settlementDate", "settlementPeriod"])["netImbalanceVolume"]
        top_spikes["NIV (MWh)"] = top_spikes.apply(
            lambda r: niv_map.get((r["settlementDate"], r["settlementPeriod"]), np.nan),
            axis=1,
        )
    st.dataframe(top_spikes, use_container_width=True, hide_index=True)

    # ── Rolling statistics ────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Rolling Statistics (30-day)")
    daily_mean = sip_df.groupby("settlementDate")[col].mean()
    fig_roll = rolling_stats_chart(daily_mean.index, daily_mean.values)
    st.plotly_chart(fig_roll, use_container_width=True)

    # ── SIP by time of day ────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Average SIP by Settlement Period")
    sp_avg = sip_df.groupby("settlementPeriod")[col].mean()
    fig_sp = go.Figure()
    fig_sp.add_trace(go.Bar(
        x=[SP_LABELS[int(sp)-1] if 1 <= sp <= 48 else str(sp) for sp in sp_avg.index],
        y=sp_avg.values,
        marker_color=COLOUR_PRIMARY,
    ))
    fig_sp.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Average SIP by Settlement Period",
        xaxis_title="Settlement Period",
        yaxis_title="Avg SIP (£/MWh)",
        margin=dict(l=50, r=30, t=50, b=50),
    )
    st.plotly_chart(fig_sp, use_container_width=True)

    # ── Export ────────────────────────────────────────────────────────
    with st.expander("Export SIP Data"):
        csv = sip_df.to_csv(index=False)
        st.download_button("Download SIP Data (CSV)", csv,
                           file_name="historical_sip.csv", mime="text/csv")
