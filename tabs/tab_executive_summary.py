"""Executive Summary tab -- KPI cards, traffic-light indicator, narrative."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.config import CHARGER_CAPACITY_KW, COLOUR_PRIMARY, PLOTLY_TEMPLATE
from src.session_keys import PARAMS, RESULT, RISK_SUMMARY


def render(has_results: bool) -> None:
    st.header("Executive Summary")

    if not has_results:
        st.info("Configure parameters in the sidebar and click **Run Simulation** to generate results.")
        return

    result = st.session_state[RESULT]
    risk = st.session_state[RISK_SUMMARY]
    params = st.session_state[PARAMS]

    # ── KPI cards ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Expected Daily P&L", f"£{risk.mean_pnl:,.0f}",
                   delta=f"median £{risk.median_pnl:,.0f}")
    with c2:
        st.metric("VaR (95%)", f"£{risk.var_95:,.0f}",
                   help="On 95% of days, losses will not exceed this amount")
    with c3:
        st.metric("ES (95%)", f"£{risk.es_95:,.0f}",
                   help="Expected Shortfall: mean P&L on the worst 5% of days")
    with c4:
        st.metric("Capture Ratio", f"{risk.capture_ratio_mean:.3f}",
                   help="Actual revenue / benchmark revenue (1.0 = perfect)")

    c5, c6, c7, c8 = st.columns(4)
    max_mw = params.fleet_size * CHARGER_CAPACITY_KW / 1_000
    traded_total_mwh = float(np.sum(result.traded_mw) * 0.5)
    with c5:
        st.metric("Fleet Size", f"{params.fleet_size:,} chargers")
    with c6:
        st.metric("Max Theoretical MW", f"{max_mw:,.1f} MW")
    with c7:
        st.metric("Daily Traded Volume", f"{traded_total_mwh:,.1f} MWh")
    with c8:
        st.metric("Reward-to-Risk", f"{risk.reward_to_risk:.2f}",
                   help="Mean(P&L) / Std(P&L) — within-day signal-to-noise ratio, not a true Sharpe")


    # ── Traffic-light risk indicator ──────────────────────────────────
    st.markdown("---")
    st.subheader("Risk Status")

    es_abs = abs(risk.es_95)
    daily_rev = float(risk.mean_pnl + np.mean(result.daily_imbalance_cost))

    if daily_rev > 0:
        risk_ratio = es_abs / daily_rev
    else:
        risk_ratio = 10.0

    if risk_ratio < 0.5:
        colour, label, desc = "🟢", "LOW RISK", "ES is well within daily revenue capacity."
    elif risk_ratio < 1.5:
        colour, label, desc = "🟡", "MODERATE RISK", "Tail losses could materially impact daily revenue."
    else:
        colour, label, desc = "🔴", "HIGH RISK", "Tail losses may exceed typical daily revenue — consider reducing traded position."

    st.markdown(f"### {colour} {label}")
    st.caption(desc)

    # ── P&L sparkline ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("P&L Distribution (mini)")
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=result.daily_pnl, nbinsx=60,
        marker_color=COLOUR_PRIMARY, opacity=0.7,
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=200,
        margin=dict(l=30, r=20, t=10, b=30),
        xaxis_title="Daily P&L (£)",
        yaxis_title="",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Auto-generated narrative ──────────────────────────────────────
    st.markdown("---")
    st.subheader("Risk Narrative")

    risk_appetite = params.risk_percentile
    narrative = (
        f"This simulation ran **{params.n_runs:,}** Monte Carlo scenarios for a fleet of "
        f"**{params.fleet_size:,}** Ohme chargers ({CHARGER_CAPACITY_KW} kW each, "
        f"theoretical max **{max_mw:,.1f} MW**). "
        f"The traded position was sized at the **P{risk_appetite}** percentile of simulated "
        f"availability, yielding a daily traded volume of **{traded_total_mwh:,.1f} MWh**.\n\n"
        f"On an average day, the portfolio generates **£{risk.mean_pnl:,.0f}** in net P&L after "
        f"imbalance costs. However, the distribution is **negatively skewed** (skew = {risk.skew_pnl:.2f}), "
        f"reflecting the asymmetric cost of being short at high System Imbalance Prices. "
        f"The 95% Value-at-Risk is **£{risk.var_95:,.0f}**, meaning on 95% of days losses do not "
        f"exceed this level. On the worst 5% of days, the expected loss (Expected Shortfall) is **£{risk.es_95:,.0f}**.\n\n"
        f"The mean capture ratio of **{risk.capture_ratio_mean:.3f}** indicates that, on average, "
    )
    if risk.capture_ratio_mean >= 0.95:
        narrative += "the portfolio captures nearly all of its theoretical benchmark revenue."
    elif risk.capture_ratio_mean >= 0.85:
        narrative += "imbalance costs erode a modest portion of potential revenue."
    else:
        narrative += "significant value is lost to imbalance costs — consider a more conservative position."

    st.markdown(narrative)
