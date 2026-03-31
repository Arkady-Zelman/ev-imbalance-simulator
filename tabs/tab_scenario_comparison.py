"""Scenario Comparison tab -- benign vs stressed, custom builder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import COLOUR_ACCENT, PLOTLY_TEMPLATE
from src.models.monte_carlo import SimulationParams, prepare_sip_matrix, run_simulation
from src.models.risk_metrics import compute_es, compute_var
from src.models.sip_models import generate_custom_scenario_sip
from src.session_keys import DA_PRICE, PARAMS, RESULT, SIP_DF
from src.visualization.heatmaps import scenario_side_by_side


def render(has_results: bool) -> None:
    st.header("Scenario Comparison")

    if not has_results:
        st.info("Run a simulation first.")
        return

    result = st.session_state[RESULT]
    params = st.session_state[PARAMS]
    sip_df = st.session_state[SIP_DF]
    da_price = st.session_state[DA_PRICE]

    # ── Auto-detect benign / stressed periods ─────────────────────────
    st.subheader("Benign vs Stressed Market Comparison")
    st.caption(
        "Benign = lowest SIP volatility period in the dataset. "
        "Stressed = highest volatility period. Each window is 30 days."
    )

    if sip_df.empty or "systemBuyPrice" not in sip_df.columns:
        st.warning("Insufficient SIP data to split into scenarios.")
        return

    sip_df = sip_df.copy()
    sip_df["settlementDate"] = pd.to_datetime(sip_df["settlementDate"])
    daily_vol = (
        sip_df.groupby("settlementDate")["systemBuyPrice"]
        .std()
        .sort_values()
    )

    if len(daily_vol) < 60:
        st.warning("Need at least 60 days of data to split into meaningful scenarios.")
        return

    n_window = min(30, len(daily_vol) // 3)
    benign_dates = daily_vol.head(n_window).index
    stressed_dates = daily_vol.tail(n_window).index

    benign_sip = sip_df[sip_df["settlementDate"].isin(benign_dates)]
    stressed_sip = sip_df[sip_df["settlementDate"].isin(stressed_dates)]

    benign_matrix, _ = prepare_sip_matrix(benign_sip)
    stressed_matrix, _ = prepare_sip_matrix(stressed_sip)

    quick_n = min(2_000, params.n_runs)

    p_b = SimulationParams(
        fleet_size=params.fleet_size,
        dispatch_rate=params.dispatch_rate,
        override_rate=params.override_rate,
        n_runs=quick_n,
        risk_percentile=params.risk_percentile,
        seed=42,
    )

    with st.spinner("Running benign scenario…"):
        res_benign = run_simulation(p_b, benign_matrix, da_price=da_price)

    with st.spinner("Running stressed scenario…"):
        res_stressed = run_simulation(p_b, stressed_matrix, da_price=da_price)

    fig_comp = scenario_side_by_side(
        res_benign.daily_pnl, res_stressed.daily_pnl,
        label_a="Benign (Low Vol)", label_b="Stressed (High Vol)",
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    st.subheader("Scenario Metrics")
    comp_data = {
        "Metric": ["Mean P&L", "Std P&L", "VaR (95%)", "ES (95%)", "Max Loss"],
        "Benign": [
            f"£{np.mean(res_benign.daily_pnl):,.0f}",
            f"£{np.std(res_benign.daily_pnl):,.0f}",
            f"£{compute_var(res_benign.daily_pnl):,.0f}",
            f"£{compute_es(res_benign.daily_pnl):,.0f}",
            f"£{np.min(res_benign.daily_pnl):,.0f}",
        ],
        "Stressed": [
            f"£{np.mean(res_stressed.daily_pnl):,.0f}",
            f"£{np.std(res_stressed.daily_pnl):,.0f}",
            f"£{compute_var(res_stressed.daily_pnl):,.0f}",
            f"£{compute_es(res_stressed.daily_pnl):,.0f}",
            f"£{np.min(res_stressed.daily_pnl):,.0f}",
        ],
    }
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

    # ── Custom scenario builder ───────────────────────────────────────
    st.markdown("---")
    st.subheader("Custom Scenario Builder")
    st.caption("Define a hypothetical SIP regime and see the resulting P&L distribution.")

    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        custom_mean = st.number_input("Normal regime mean SIP (£/MWh)",
                                      value=60.0, step=10.0)
        custom_std = st.number_input("Normal regime std (£/MWh)",
                                     value=30.0, step=5.0)
    with cc2:
        spike_prob = st.slider("Spike probability", 0.0, 0.30, 0.05, 0.01)
        spike_mean = st.number_input("Spike mean (£/MWh)", value=500.0, step=50.0)
    with cc3:
        spike_std = st.number_input("Spike std (£/MWh)", value=300.0, step=50.0)

    if st.button("Run Custom Scenario"):
        custom_sip = generate_custom_scenario_sip(
            n_days=200,
            normal_mean=custom_mean,
            normal_std=custom_std,
            spike_prob=spike_prob,
            spike_mean=spike_mean,
            spike_std=spike_std,
            seed=99,
        )

        with st.spinner("Running custom scenario…"):
            res_custom = run_simulation(p_b, custom_sip, da_price=da_price)

        fig_c = go.Figure()
        fig_c.add_trace(go.Histogram(
            x=res_custom.daily_pnl, nbinsx=60,
            marker_color=COLOUR_ACCENT, opacity=0.7,
        ))
        fig_c.update_layout(
            template=PLOTLY_TEMPLATE,
            title="Custom Scenario P&L Distribution",
            xaxis_title="Daily P&L (£)",
            yaxis_title="Frequency",
            margin=dict(l=50, r=30, t=50, b=50),
        )
        st.plotly_chart(fig_c, use_container_width=True)

        st.metric("Custom ES (95%)", f"£{compute_es(res_custom.daily_pnl):,.0f}")
        st.metric("Custom Mean P&L", f"£{np.mean(res_custom.daily_pnl):,.0f}")
