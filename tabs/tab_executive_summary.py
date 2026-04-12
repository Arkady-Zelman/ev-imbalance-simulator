"""
Tab 3 — Executive Summary

KPIs: Expected P&L, VaR (95%), Expected Shortfall (95%),
      Total Capacity, Traded Capacity by Market.

Reads from session state Monte Carlo result and allocation result.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import src.session_keys as sk
from src.config import (
    CHARGER_CAPACITY_KW,
    DEFAULT_DA_PRICE,
    DEFAULT_DISPATCH_SUCCESS_RATE,
    DEFAULT_FLEET_SIZE,
    DEFAULT_MC_RUNS,
    DEFAULT_OVERRIDE_RATE,
    DEFAULT_RISK_APPETITE,
    RISK_APPETITES,
)

logger = logging.getLogger(__name__)

PLOTLY_TEMPLATE = "plotly_dark"


def _traffic_light(ratio: float) -> str:
    if ratio >= 0.6:
        return "🟢 GREEN"
    if ratio >= 0.3:
        return "🟡 AMBER"
    return "🔴 RED"


def _run_mc_if_needed(params: dict, sip_df=None) -> Optional[object]:
    try:
        from src.models.monte_carlo import SimulationParams, run_simulation, prepare_sip_matrix
        da_price = params.pop("da_price", DEFAULT_DA_PRICE)
        p = SimulationParams(**params)
        sip_matrix, _ = prepare_sip_matrix(sip_df if sip_df is not None else __import__("pandas").DataFrame())
        return run_simulation(p, sip_matrix, da_price=da_price)
    except Exception as exc:
        logger.error("MC simulation failed: %s", exc)
        return None


def render() -> None:
    st.header("Executive Summary")

    predictions_loaded: bool = st.session_state.get(sk.PREDICTIONS_LOADED, False)

    # ── Read shared parameters from unified sidebar ───────────────────────────
    fleet_size    = st.session_state.get(sk.SIM_FLEET_SIZE,    DEFAULT_FLEET_SIZE)
    dispatch_rate = st.session_state.get(sk.SIM_DISPATCH_RATE, DEFAULT_DISPATCH_SUCCESS_RATE)
    override_rate = st.session_state.get(sk.SIM_OVERRIDE_RATE, DEFAULT_OVERRIDE_RATE)
    da_price      = float(st.session_state.get(sk.DA_PRICE,    DEFAULT_DA_PRICE))
    risk_app      = st.session_state.get(sk.SIM_RISK_APPETITE, DEFAULT_RISK_APPETITE)
    mc_runs       = st.session_state.get(sk.SIM_MC_RUNS,       DEFAULT_MC_RUNS)
    run_btn       = st.session_state.get(sk.SIM_RUN_REQUESTED, False)

    # ── Load or run simulation ────────────────────────────────────────────────
    result = st.session_state.get(sk.RESULT)

    if run_btn or result is None:
        mc_params = dict(
            fleet_size=fleet_size,
            dispatch_rate=dispatch_rate,
            override_rate=override_rate,
            da_price=da_price,
            n_runs=mc_runs,
        )
        with st.spinner("Running Monte Carlo simulation…"):
            result = _run_mc_if_needed(mc_params, sip_df=st.session_state.get(sk.SIP_DF))
        if result:
            st.session_state[sk.RESULT] = result

    if result is None:
        st.info("Click **Run Simulation** to generate results.")
        return

    # ── Extract metrics ───────────────────────────────────────────────────────
    try:
        pnl_dist = np.asarray(result.daily_pnl,             dtype=float)
        rev_dist = np.asarray(result.daily_revenue,          dtype=float)
        imb_dist = np.asarray(result.daily_imbalance_cost,   dtype=float)
    except AttributeError:
        st.error("Simulation result format unexpected. Please re-run.")
        return

    expected_pnl = float(np.mean(pnl_dist))
    pct = RISK_APPETITES[risk_app]
    var_95       = float(np.percentile(pnl_dist, 5))   # 5th pct = 95% VaR loss
    es_95        = float(np.mean(pnl_dist[pnl_dist <= var_95]))
    total_cap_mw = fleet_size * CHARGER_CAPACITY_KW / 1000.0

    # Traded capacity from allocation result if available
    allocation = st.session_state.get(sk.ALLOCATION_RESULT)
    if allocation:
        wholesale_mw = float(np.mean(np.asarray(allocation.get("wholesale_mw", [0] * 48))))
        balancing_mw = float(np.mean(np.asarray(allocation.get("balancing_mw", [0] * 48))))
    else:
        # Fall back to percentile from distribution
        wholesale_mw = total_cap_mw * pct / 100.0 * 0.6
        balancing_mw = total_cap_mw * pct / 100.0 * 0.4

    held_mw = max(0.0, total_cap_mw - wholesale_mw - balancing_mw)

    # Traffic light
    es_to_rev = abs(es_95) / max(expected_pnl, 1.0)
    traffic    = _traffic_light(1.0 - es_to_rev)

    # ── KPI rows (2-row layout) ───────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Expected P&L",
        f"£{expected_pnl:,.0f}",
        delta=f"£{expected_pnl:+,.0f}",
        delta_color="normal",
    )
    col2.metric(
        "VaR (95%)",
        f"£{var_95:,.0f}",
        delta=f"£{var_95 - expected_pnl:,.0f}",
        delta_color="inverse",
    )
    col3.metric(
        "Expected Shortfall (95%)",
        f"£{es_95:,.0f}",
        delta=f"£{es_95 - expected_pnl:,.0f}",
        delta_color="inverse",
    )
    col4, col5 = st.columns(2)
    col4.metric("Total Capacity", f"{total_cap_mw:.1f} MW")
    col5.metric("Risk Status", traffic)

    st.divider()

    # ── P&L distribution histogram ────────────────────────────────────────────
    col_hist, col_donut = st.columns([6, 4])

    with col_hist:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=pnl_dist, nbinsx=60,
            marker_color="#00D4AA", opacity=0.8,
            name="P&L",
        ))
        fig_hist.add_vline(x=var_95, line_dash="dash", line_color="#FF6B6B",
                           annotation_text="VaR 95%", annotation_position="top left")
        fig_hist.add_vline(x=es_95,  line_dash="dot",  line_color="#FFE66D",
                           annotation_text="ES 95%",  annotation_position="top left")
        fig_hist.update_layout(
            template=PLOTLY_TEMPLATE,
            title="P&L Distribution (Monte Carlo)",
            xaxis_title="P&L (£)",
            yaxis_title="Count",
            height=380,
        )
        st.plotly_chart(fig_hist, width="stretch")

    with col_donut:
        fig_donut = go.Figure(go.Pie(
            labels=["Wholesale", "Balancing", "Held Back"],
            values=[wholesale_mw, balancing_mw, held_mw],
            hole=0.55,
            marker=dict(colors=["#00D4AA", "#FF6B6B", "#4A5568"]),
        ))
        fig_donut.update_layout(
            template=PLOTLY_TEMPLATE,
            title="Capacity Split (MW)",
            height=380,
        )
        st.plotly_chart(fig_donut, width="stretch")

    # ── Revenue / imbalance breakdown ─────────────────────────────────────────
    st.subheader("Revenue vs Imbalance Cost")
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    col_r1.metric("Avg Revenue",          f"£{float(np.mean(rev_dist)):,.0f}")
    col_r2.metric("Avg Imbalance Cost",   f"£{float(np.mean(imb_dist)):,.0f}")
    col_r3.metric("Wholesale MW (avg)",   f"{wholesale_mw:.1f}")
    col_r4.metric("Balancing MW (avg)",   f"{balancing_mw:.1f}")

    # ── Risk metrics table ────────────────────────────────────────────────────
    with st.expander("Full risk metrics table", expanded=False):
        risk_rows = [
            {"Metric": "Expected P&L (£)",            "Value": f"{expected_pnl:,.2f}"},
            {"Metric": "VaR 95% (£)",                 "Value": f"{var_95:,.2f}"},
            {"Metric": "Expected Shortfall 95% (£)",  "Value": f"{es_95:,.2f}"},
            {"Metric": "P&L Std Dev (£)",             "Value": f"{float(np.std(pnl_dist)):,.2f}"},
            {"Metric": "P&L 5th pct (£)",             "Value": f"{float(np.percentile(pnl_dist,  5)):,.2f}"},
            {"Metric": "P&L 25th pct (£)",            "Value": f"{float(np.percentile(pnl_dist, 25)):,.2f}"},
            {"Metric": "P&L 75th pct (£)",            "Value": f"{float(np.percentile(pnl_dist, 75)):,.2f}"},
            {"Metric": "P&L 95th pct (£)",            "Value": f"{float(np.percentile(pnl_dist, 95)):,.2f}"},
            {"Metric": "Total Capacity (MW)",         "Value": f"{total_cap_mw:.2f}"},
            {"Metric": "Wholesale Traded (MW avg)",   "Value": f"{wholesale_mw:.2f}"},
            {"Metric": "Balancing MW (avg)",          "Value": f"{balancing_mw:.2f}"},
        ]
        st.dataframe(pd.DataFrame(risk_rows), width="stretch")
