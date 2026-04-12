"""
Tab 4 — Sensitivity Analysis / Scenario Analysis

Tornado chart: vary fleet_size, dispatch_rate, override_rate, da_price,
               sip_level ±20%; show P&L delta.

Scenario table: Base / High SIP stress / Low availability / Large fleet / Custom
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import src.session_keys as sk
from src.config import (
    DEFAULT_DA_PRICE,
    DEFAULT_DISPATCH_SUCCESS_RATE,
    DEFAULT_FLEET_SIZE,
    DEFAULT_OVERRIDE_RATE,
)

logger = logging.getLogger(__name__)

PLOTLY_TEMPLATE = "plotly_dark"

_SCENARIOS = {
    "Base":              {"fleet_size": DEFAULT_FLEET_SIZE, "dispatch_rate": DEFAULT_DISPATCH_SUCCESS_RATE, "override_rate": DEFAULT_OVERRIDE_RATE, "da_price": DEFAULT_DA_PRICE, "sip_scale": 1.0},
    "High SIP Stress":   {"fleet_size": DEFAULT_FLEET_SIZE, "dispatch_rate": 0.90, "override_rate": 0.05, "da_price": DEFAULT_DA_PRICE, "sip_scale": 1.5},
    "Low Availability":  {"fleet_size": DEFAULT_FLEET_SIZE, "dispatch_rate": 0.75, "override_rate": 0.08, "da_price": DEFAULT_DA_PRICE, "sip_scale": 1.0},
    "Large Fleet":       {"fleet_size": 50_000,              "dispatch_rate": DEFAULT_DISPATCH_SUCCESS_RATE, "override_rate": DEFAULT_OVERRIDE_RATE, "da_price": DEFAULT_DA_PRICE, "sip_scale": 1.0},
    "Low DA Price":      {"fleet_size": DEFAULT_FLEET_SIZE, "dispatch_rate": DEFAULT_DISPATCH_SUCCESS_RATE, "override_rate": DEFAULT_OVERRIDE_RATE, "da_price": 50.0, "sip_scale": 1.0},
}

_TORNADO_PARAMS = [
    ("Fleet size",     "fleet_size",     DEFAULT_FLEET_SIZE * 0.8, DEFAULT_FLEET_SIZE * 1.2),
    ("Dispatch rate",  "dispatch_rate",  DEFAULT_DISPATCH_SUCCESS_RATE * 0.8, min(DEFAULT_DISPATCH_SUCCESS_RATE * 1.2, 1.0)),
    ("Override rate",  "override_rate",  DEFAULT_OVERRIDE_RATE * 0.8, DEFAULT_OVERRIDE_RATE * 1.2),
    ("DA price",       "da_price",       DEFAULT_DA_PRICE * 0.8, DEFAULT_DA_PRICE * 1.2),
]


def _quick_pnl(fleet_size, dispatch_rate, override_rate, da_price, n_runs=500) -> float:
    """Run a quick 500-run MC and return expected P&L."""
    try:
        from src.models.monte_carlo import SimulationParams, run_simulation, prepare_sip_matrix
        import streamlit as _st
        import pandas as _pd
        sip_df = _st.session_state.get(sk.SIP_DF)
        sip_matrix, _ = prepare_sip_matrix(sip_df if sip_df is not None else _pd.DataFrame())
        params = SimulationParams(
            fleet_size=int(fleet_size),
            dispatch_rate=float(dispatch_rate),
            override_rate=float(override_rate),
            n_runs=n_runs,
        )
        result = run_simulation(params, sip_matrix, da_price=float(da_price))
        return float(np.mean(result.daily_pnl))
    except Exception as exc:
        logger.error("Quick P&L failed: %s", exc)
        return 0.0


def _run_scenarios(base_params: dict, n_runs: int) -> pd.DataFrame:
    rows = []
    for name, overrides in _SCENARIOS.items():
        params = {**base_params, **overrides}
        params.pop("sip_scale", None)
        pnl = _quick_pnl(**params, n_runs=n_runs)
        rows.append({"Scenario": name, "Expected P&L (£)": round(pnl, 2)})
    return pd.DataFrame(rows)


def _run_tornado(base_params: dict, base_pnl: float, n_runs: int) -> pd.DataFrame:
    rows = []
    for label, key, low_val, high_val in _TORNADO_PARAMS:
        # Low end
        p_low = {**base_params, key: low_val}
        pnl_low = _quick_pnl(**p_low, n_runs=n_runs)
        # High end
        p_high = {**base_params, key: high_val}
        pnl_high = _quick_pnl(**p_high, n_runs=n_runs)

        rows.append({
            "Parameter":    label,
            "Low Value":    low_val,
            "High Value":   high_val,
            "P&L (Low)":    pnl_low,
            "P&L (High)":   pnl_high,
            "Delta (Low)":  pnl_low  - base_pnl,
            "Delta (High)": pnl_high - base_pnl,
        })
    return pd.DataFrame(rows)


def render() -> None:
    st.header("Sensitivity & Scenario Analysis")

    # ── Read shared parameters from unified sidebar ───────────────────────────
    fleet_size    = st.session_state.get(sk.SIM_FLEET_SIZE,    DEFAULT_FLEET_SIZE)
    dispatch_rate = st.session_state.get(sk.SIM_DISPATCH_RATE, DEFAULT_DISPATCH_SUCCESS_RATE)
    override_rate = st.session_state.get(sk.SIM_OVERRIDE_RATE, DEFAULT_OVERRIDE_RATE)
    da_price      = float(st.session_state.get(sk.DA_PRICE,    DEFAULT_DA_PRICE))
    n_runs_sens   = st.session_state.get(sk.SIM_MC_RUNS_SENS,  500)
    run_btn       = st.session_state.get(sk.SIM_RUN_REQUESTED, False)

    base_params = dict(
        fleet_size=fleet_size,
        dispatch_rate=dispatch_rate,
        override_rate=override_rate,
        da_price=da_price,
    )

    if not run_btn:
        st.info("Configure parameters in the sidebar and click **▶ Run Simulation**.")
        return

    with st.spinner("Running sensitivity analysis…"):
        base_pnl = _quick_pnl(**base_params, n_runs=n_runs_sens)

    tab_tornado, tab_scenario, tab_custom = st.tabs(["Tornado Chart", "Scenario Table", "Custom Scenario"])

    with tab_tornado:
        st.subheader("Tornado Chart — P&L Sensitivity (±20%)")
        with st.spinner("Computing tornado…"):
            tornado_df = _run_tornado(base_params, base_pnl, n_runs_sens)

        tornado_df_sorted = tornado_df.reindex(
            tornado_df["Delta (High)"].abs().sort_values(ascending=True).index
        )

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=tornado_df_sorted["Parameter"],
            x=tornado_df_sorted["Delta (Low)"],
            name="Low (-20%)",
            orientation="h",
            marker_color="#FF6B6B",
        ))
        fig.add_trace(go.Bar(
            y=tornado_df_sorted["Parameter"],
            x=tornado_df_sorted["Delta (High)"],
            name="High (+20%)",
            orientation="h",
            marker_color="#00D4AA",
        ))
        fig.add_vline(x=0, line_color="#95A5A6", line_dash="dash")
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            barmode="overlay",
            title=f"P&L Delta vs Base (£{base_pnl:,.0f})",
            xaxis_title="P&L Delta (£)",
            height=380,
        )
        st.plotly_chart(fig, width="stretch")
        st.dataframe(tornado_df[["Parameter", "Delta (Low)", "Delta (High)"]].round(2),
                     width="stretch")

    with tab_scenario:
        st.subheader("Predefined Scenario Comparison")
        with st.spinner("Running scenarios…"):
            scenario_df = _run_scenarios(base_params, n_runs_sens)

        base_row = scenario_df[scenario_df["Scenario"] == "Base"]["Expected P&L (£)"]
        base_val = float(base_row.values[0]) if not base_row.empty else base_pnl
        scenario_df["vs Base (£)"] = scenario_df["Expected P&L (£)"] - base_val
        st.dataframe(scenario_df.round(2), width="stretch")

        fig2 = go.Figure(go.Bar(
            x=scenario_df["Scenario"],
            y=scenario_df["Expected P&L (£)"],
            marker_color=[
                "#00D4AA" if v >= base_val else "#FF6B6B"
                for v in scenario_df["Expected P&L (£)"]
            ],
        ))
        fig2.add_hline(y=base_val, line_dash="dot", line_color="#FFE66D",
                       annotation_text="Base P&L")
        fig2.update_layout(
            template=PLOTLY_TEMPLATE,
            title="Scenario P&L Comparison",
            yaxis_title="Expected P&L (£)",
            height=350,
        )
        st.plotly_chart(fig2, width="stretch")

    with tab_custom:
        st.subheader("Custom Scenario")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            c_fleet  = st.number_input("Fleet size", 1_000, 100_000, DEFAULT_FLEET_SIZE, 1_000, key="cust_fleet")
        with c2:
            c_disp   = st.slider("Dispatch rate", 0.5, 1.0, DEFAULT_DISPATCH_SUCCESS_RATE, 0.01, key="cust_disp")
        with c3:
            c_over   = st.slider("Override rate", 0.0, 0.2, DEFAULT_OVERRIDE_RATE, 0.005, key="cust_over")
        with c4:
            c_da     = st.number_input("DA price", 0.0, 500.0, DEFAULT_DA_PRICE, 1.0, key="cust_da")

        if st.button("Run Custom Scenario"):
            with st.spinner("Running…"):
                c_pnl = _quick_pnl(c_fleet, c_disp, c_over, c_da, n_runs=n_runs_sens)
            delta = c_pnl - base_pnl
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Custom P&L", f"£{c_pnl:,.2f}")
            col_b.metric("Base P&L",   f"£{base_pnl:,.2f}")
            col_c.metric("Delta",      f"£{delta:,.2f}", delta=f"£{delta:,.2f}")
