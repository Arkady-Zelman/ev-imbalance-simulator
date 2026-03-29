"""Portfolio Availability tab -- plug-in profiles, Beta overlays, MW curves."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import beta as beta_dist

from src.config import (
    CHARGER_CAPACITY_KW,
    NUM_SETTLEMENT_PERIODS,
    PLOTLY_TEMPLATE,
    PLUGIN_CLUSTERS,
    SP_LABELS,
    build_sp_beta_params,
)
from src.session_keys import PARAMS, RESULT
from src.visualization.charts import (
    available_mw_profile,
    beta_distribution_overlay,
    plugin_rate_profile,
)


def render(has_results: bool) -> None:
    st.header("Portfolio Availability Model")

    alphas, betas = build_sp_beta_params()

    means = np.array([alphas[i] / (alphas[i] + betas[i]) for i in range(NUM_SETTLEMENT_PERIODS)])
    p5 = np.array([beta_dist.ppf(0.05, alphas[i], betas[i]) for i in range(NUM_SETTLEMENT_PERIODS)])
    p95 = np.array([beta_dist.ppf(0.95, alphas[i], betas[i]) for i in range(NUM_SETTLEMENT_PERIODS)])

    st.subheader("Plug-in Rate by Settlement Period")
    st.caption("Based on Beta distribution parameters derived from CrowdFlex trial baselines.")
    fig_plugin = plugin_rate_profile(means, p5, p95)
    st.plotly_chart(fig_plugin, use_container_width=True)

    st.subheader("Beta Distribution Shapes")
    st.caption("Select settlement periods to compare the uncertainty in plug-in rates.")
    default_sps = [3, 15, 25, 37, 45]
    selected_sps = st.multiselect(
        "Settlement periods to display",
        options=list(range(NUM_SETTLEMENT_PERIODS)),
        default=default_sps,
        format_func=lambda i: f"SP {i+1} ({SP_LABELS[i]})",
    )
    if selected_sps:
        fig_beta = beta_distribution_overlay(alphas, betas, selected_sps)
        st.plotly_chart(fig_beta, use_container_width=True)

    st.subheader("Time-of-Day Cluster Parameters")
    cluster_data = []
    for c in PLUGIN_CLUSTERS:
        cluster_data.append({
            "Cluster": c.name,
            "Settlement Periods": f"SP {c.sp_range[0]+1}–{c.sp_range[1]+1} ({SP_LABELS[c.sp_range[0]]}–{SP_LABELS[c.sp_range[1]]})",
            "Mean Plug-in": f"{c.mean:.0%}",
            "Concentration (ν)": c.concentration,
            "α": f"{c.alpha:.1f}",
            "β": f"{c.beta_param:.1f}",
        })
    st.dataframe(pd.DataFrame(cluster_data), use_container_width=True, hide_index=True)

    if has_results:
        result = st.session_state[RESULT]
        st.markdown("---")
        st.subheader("Available MW Across the Day")

        delivered = result.delivered_mw
        mean_mw = delivered.mean(axis=0)
        p5_mw = np.percentile(delivered, 5, axis=0)
        p95_mw = np.percentile(delivered, 95, axis=0)

        fig_mw = available_mw_profile(mean_mw, p5_mw, p95_mw, traded_mw=result.traded_mw)
        st.plotly_chart(fig_mw, use_container_width=True)

        st.subheader("Fleet Composition Breakdown (Mean)")
        params = st.session_state[PARAMS]
        plugin_mean = result.plugin_rates.mean(axis=0)
        plugged_in = plugin_mean * params.fleet_size
        dispatched = plugged_in * params.dispatch_rate
        responding = dispatched * (1 - params.override_rate)

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Scatter(
            x=SP_LABELS, y=plugged_in, mode="lines", name="Plugged In",
            line=dict(width=2),
        ))
        fig_comp.add_trace(go.Scatter(
            x=SP_LABELS, y=dispatched, mode="lines", name="Dispatched",
            line=dict(width=2),
        ))
        fig_comp.add_trace(go.Scatter(
            x=SP_LABELS, y=responding, mode="lines", name="Responding",
            line=dict(width=2),
        ))
        fig_comp.update_layout(
            template=PLOTLY_TEMPLATE,
            margin=dict(l=50, r=30, t=50, b=50),
            xaxis_title="Settlement Period",
            yaxis_title="Number of Chargers",
            title="Mean Charger Status by Settlement Period",
        )
        st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.info("Run a simulation to see fleet-specific MW availability curves.")
