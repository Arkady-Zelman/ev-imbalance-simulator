"""Monte Carlo Results tab -- P&L histograms, scatter, box plots, stats."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import COLOUR_PRIMARY, PLOTLY_TEMPLATE, RISK_APPETITES
from src.models.pnl_calculator import compute_pnl_for_position
from src.session_keys import ALL_POSITIONS, DA_PRICE, PARAMS, RESULT, RISK_SUMMARY, SIP_MATRIX
from src.visualization.charts import (
    delivered_vs_traded_scatter,
    pnl_comparison_histograms,
    pnl_histogram,
)
from src.visualization.heatmaps import imbalance_boxplots


def render(has_results: bool) -> None:
    st.header("Monte Carlo Simulation Results")

    if not has_results:
        st.info("Run a simulation first to see results here.")
        return

    result = st.session_state[RESULT]
    risk = st.session_state[RISK_SUMMARY]
    params = st.session_state[PARAMS]

    # ── Main P&L histogram ────────────────────────────────────────────
    st.subheader(f"Daily P&L Distribution (P{params.risk_percentile} Position)")
    fig_pnl = pnl_histogram(result.daily_pnl, risk.var_95, risk.es_95)
    st.plotly_chart(fig_pnl, use_container_width=True)

    # ── Summary statistics table ──────────────────────────────────────
    st.subheader("Summary Statistics")
    stats_data = {
        "Metric": ["Mean", "Median", "Std Dev", "Skewness", "Kurtosis",
                    "VaR (95%)", "ES (95%)", "Max Loss", "Max Gain"],
        "Value": [
            f"£{risk.mean_pnl:,.0f}",
            f"£{risk.median_pnl:,.0f}",
            f"£{risk.std_pnl:,.0f}",
            f"{risk.skew_pnl:.3f}",
            f"{risk.kurtosis_pnl:.3f}",
            f"£{risk.var_95:,.0f}",
            f"£{risk.es_95:,.0f}",
            f"£{risk.max_loss:,.0f}",
            f"£{risk.max_gain:,.0f}",
        ],
    }
    st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

    # ── Side-by-side comparison across risk appetites ─────────────────
    st.markdown("---")
    st.subheader("P&L Comparison Across Risk Appetites")
    st.caption("Each histogram shows the P&L distribution for a different position-sizing percentile.")

    all_positions = st.session_state[ALL_POSITIONS]
    sip_matrix = st.session_state[SIP_MATRIX]
    da_price = st.session_state[DA_PRICE]

    comparison_tiers = ["P50", "P80", "P95"]
    pnl_dict = {}
    for tier in comparison_tiers:
        if tier in all_positions:
            pnl_dict[tier] = compute_pnl_for_position(
                result.delivered_mw, all_positions[tier],
                sip_matrix, da_price, params.n_runs,
            )

    if pnl_dict:
        fig_comp = pnl_comparison_histograms(pnl_dict)
        st.plotly_chart(fig_comp, use_container_width=True)

    # ── Delivered vs Traded ───────────────────────────────────────────
    st.markdown("---")
    st.subheader("Delivered vs Traded Volume")
    delivered_daily_mwh = result.delivered_mw.sum(axis=1) * 0.5
    traded_daily_mwh = float(result.traded_mw.sum() * 0.5)
    fig_scatter = delivered_vs_traded_scatter(delivered_daily_mwh, traded_daily_mwh)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Imbalance box plots ───────────────────────────────────────────
    st.markdown("---")
    st.subheader("Imbalance Volume by Settlement Period")
    st.caption("Positive = short (under-delivered). Evening peak SPs typically show the widest spread.")
    fig_box = imbalance_boxplots(result.imbalance_mw)
    st.plotly_chart(fig_box, use_container_width=True)

    # ── CSV export ────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("Export Simulation Data"):
        export_df = pd.DataFrame({
            "run": np.arange(params.n_runs),
            "daily_pnl": result.daily_pnl,
            "daily_imbalance_cost": result.daily_imbalance_cost,
            "daily_revenue": result.daily_revenue,
        })
        csv = export_df.to_csv(index=False)
        st.download_button("Download P&L Data (CSV)", csv,
                           file_name="mc_pnl_results.csv", mime="text/csv")
