"""Sensitivity Analysis tab -- tornado diagram, parameter sweeps, diversification."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.config import CHARGER_CAPACITY_KW, build_sp_beta_params
from src.models.monte_carlo import SimulationParams, prepare_sip_matrix, run_simulation
from src.models.risk_metrics import compute_cvar
from src.session_keys import DA_PRICE, PARAMS, RESULT, RISK_SUMMARY, SIP_MATRIX
from src.visualization.charts import (
    diversification_curve,
    parameter_sweep_chart,
    tornado_diagram,
)


def _quick_cvar(fleet_size, dispatch_rate, override_rate, risk_pct,
                sip_matrix, da_price, n_runs=1_000, seed=42):
    """Helper: run a reduced simulation and return CVaR."""
    params = SimulationParams(
        fleet_size=fleet_size,
        dispatch_rate=dispatch_rate,
        override_rate=override_rate,
        n_runs=n_runs,
        risk_percentile=risk_pct,
        seed=seed,
    )
    res = run_simulation(params, sip_matrix, da_price=da_price)
    return compute_cvar(res.daily_pnl)


def render(has_results: bool) -> None:
    st.header("Sensitivity Analysis")

    if not has_results:
        st.info("Run a simulation first.")
        return

    result = st.session_state[RESULT]
    params = st.session_state[PARAMS]
    sip_matrix = st.session_state[SIP_MATRIX]
    da_price = st.session_state[DA_PRICE]
    risk = st.session_state[RISK_SUMMARY]

    base_cvar = risk.cvar_95
    quick_n = min(1_000, params.n_runs)

    # ── Tornado diagram ───────────────────────────────────────────────
    st.subheader("CVaR Sensitivity (Tornado Diagram)")
    st.caption("Shows how CVaR changes when each parameter is varied between a low and high scenario, "
               "holding all others at their base value.")

    with st.spinner("Running sensitivity scenarios…"):
        base_kw = dict(
            fleet_size=params.fleet_size,
            dispatch_rate=params.dispatch_rate,
            override_rate=params.override_rate,
            risk_pct=params.risk_percentile,
            sip_matrix=sip_matrix,
            da_price=da_price,
            n_runs=quick_n,
        )

        param_defs = [
            ("Fleet Size", "fleet_size", 5_000, 50_000),
            ("Dispatch Rate", "dispatch_rate", 0.90, 0.99),
            ("Override Rate", "override_rate", 0.01, 0.10),
            ("Risk Percentile", "risk_pct", 50, 95),
        ]

        names, lows, highs = [], [], []
        for label, key, lo, hi in param_defs:
            kw_lo = {**base_kw, key: lo}
            kw_hi = {**base_kw, key: hi}
            cvar_lo = _quick_cvar(**kw_lo)
            cvar_hi = _quick_cvar(**kw_hi)
            names.append(f"{label}\n({lo} → {hi})")
            lows.append(cvar_lo)
            highs.append(cvar_hi)

    fig_tornado = tornado_diagram(names, lows, highs, base_cvar)
    st.plotly_chart(fig_tornado, use_container_width=True)

    # ── Individual parameter sweeps ───────────────────────────────────
    st.markdown("---")
    st.subheader("Parameter Sweep Charts")

    sweep_param = st.selectbox("Parameter to sweep", [
        "Fleet Size", "Dispatch Rate", "Override Rate",
    ])

    with st.spinner("Running sweep…"):
        if sweep_param == "Fleet Size":
            x_vals = [5_000, 10_000, 20_000, 35_000, 50_000, 75_000, 100_000]
            key = "fleet_size"
        elif sweep_param == "Dispatch Rate":
            x_vals = [0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99]
            key = "dispatch_rate"
        else:
            x_vals = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15]
            key = "override_rate"

        y_cvar = []
        y_pnl = []
        for v in x_vals:
            kw = {**base_kw, key: v}
            p = SimulationParams(
                fleet_size=kw["fleet_size"],
                dispatch_rate=kw["dispatch_rate"],
                override_rate=kw["override_rate"],
                n_runs=quick_n,
                risk_percentile=kw["risk_pct"],
                seed=42,
            )
            r = run_simulation(p, sip_matrix, da_price=da_price)
            y_cvar.append(compute_cvar(r.daily_pnl))
            y_pnl.append(float(np.mean(r.daily_pnl)))

    c1, c2 = st.columns(2)
    with c1:
        fig_s1 = parameter_sweep_chart(x_vals, y_pnl, sweep_param, "Expected P&L (£)",
                                       title=f"Expected P&L vs {sweep_param}")
        st.plotly_chart(fig_s1, use_container_width=True)
    with c2:
        fig_s2 = parameter_sweep_chart(x_vals, y_cvar, sweep_param, "CVaR 95% (£)",
                                       title=f"CVaR vs {sweep_param}")
        st.plotly_chart(fig_s2, use_container_width=True)

    # ── Portfolio diversification effect ──────────────────────────────
    st.markdown("---")
    st.subheader("Portfolio Diversification Effect")
    st.caption("As fleet size grows, relative uncertainty (coefficient of variation) "
               "in delivered MW decreases — the law of large numbers at work.")

    with st.spinner("Computing diversification curve…"):
        fleet_sizes = [1_000, 2_500, 5_000, 10_000, 20_000, 50_000, 100_000]
        cvs = []
        for fs in fleet_sizes:
            p = SimulationParams(
                fleet_size=fs,
                dispatch_rate=params.dispatch_rate,
                override_rate=params.override_rate,
                n_runs=quick_n,
                risk_percentile=params.risk_percentile,
                seed=42,
            )
            r = run_simulation(p, sip_matrix, da_price=da_price)
            total_delivered = r.delivered_mw.sum(axis=1)
            cv = float(np.std(total_delivered) / max(np.mean(total_delivered), 1e-9))
            cvs.append(cv)

    fig_div = diversification_curve(fleet_sizes, cvs)
    st.plotly_chart(fig_div, use_container_width=True)

    st.info(
        "**Strategic implication:** As Ohme scales from 20,000 to 100,000+ chargers, "
        "relative imbalance risk drops significantly. However, absolute risk still grows. "
        "The operational challenge shifts from forecasting accuracy to managing correlation "
        "and tail events."
    )
