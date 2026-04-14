"""
Tab 2 — Capacity Allocation Suggestion by Half-Hour Block

Shows one day ahead (48 SPs): how much fleet capacity to allocate
to wholesale (DA) vs balancing (SIP) per settlement period.

Uses:
  - Intraday predictions from sip/mip parquets
  - Monte Carlo simulation (monte_carlo.py)
  - Allocation optimiser (allocation_optimizer.py)
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
    SP_LABELS,
)
from src.predictions import PredictionSchemaError, load_intraday_predictions
from src.ui.dataframes import with_optional_background_gradient

logger = logging.getLogger(__name__)

PLOTLY_TEMPLATE = "plotly_dark"

_RISK_PROFILES = ["Conservative", "Moderate", "Aggressive", "Full Risk"]
_RISK_PERCENTILES = {
    "Conservative": 50,
    "Moderate":     70,
    "Aggressive":   85,
    "Full Risk":    95,
}


def _get_intraday_predictions(
    pred_sip: Optional[pd.DataFrame],
    pred_mip: Optional[pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract the best-available intraday SIP and MIP forecast (48 SPs).
    Uses the first available lookback in intraday predictions.
    Falls back to default DA price if MIP unavailable.
    """
    sip_fc = np.full(48, 80.0)
    mip_fc = np.full(48, DEFAULT_DA_PRICE)

    try:
        if pred_sip is not None and not pred_sip.empty:
            intra = load_intraday_predictions(
                "sip",
                pred_sip,
                context="Capacity Allocation SIP intraday input",
            )
            sip_avg = (
                intra.groupby("settlement_period")["hybrid_prediction"]
                .mean()
                .sort_index()
            )
            if len(sip_avg) == 48:
                sip_fc = sip_avg.values.astype(float)
    except PredictionSchemaError as exc:
        logger.warning("%s", exc)

    try:
        if pred_mip is not None and not pred_mip.empty:
            intra = load_intraday_predictions(
                "mip",
                pred_mip,
                context="Capacity Allocation MIP intraday input",
            )
            mip_avg = (
                intra.groupby("settlement_period")["hybrid_prediction"]
                .mean()
                .sort_index()
            )
            if len(mip_avg) == 48:
                mip_fc = mip_avg.values.astype(float)
    except PredictionSchemaError as exc:
        logger.warning("%s", exc)

    return sip_fc, mip_fc


def _run_allocation(
    fleet_size: int,
    dispatch_rate: float,
    override_rate: float,
    da_price: float,
    sip_fc: np.ndarray,
    mip_fc: np.ndarray,
    mc_runs: int,
    risk_percentile: int,
) -> dict:
    """
    Run MC simulation and allocation optimisation for a single risk profile.
    Returns dict with keys: wholesale_mw, balancing_mw, strategy, expected_rev, es_rev.
    """
    try:
        from src.models.monte_carlo import SimulationParams, run_simulation, prepare_sip_matrix
        from src.models.allocation_optimizer import optimize_allocation
        import streamlit as _st

        sip_df = _st.session_state.get(sk.SIP_DF)
        import pandas as _pd
        sip_matrix, _ = prepare_sip_matrix(sip_df if sip_df is not None else _pd.DataFrame())

        params = SimulationParams(
            fleet_size=fleet_size,
            dispatch_rate=dispatch_rate,
            override_rate=override_rate,
            n_runs=mc_runs,
            risk_percentile=risk_percentile,
        )
        result = run_simulation(params, sip_matrix, da_price=da_price)

        allocation = optimize_allocation(
            sip_forecasts=sip_fc,
            mip_forecasts=mip_fc,
            delivered_mw=result.delivered_mw,
            risk_tolerance=risk_percentile / 100.0,
        )
        # Extract per-SP arrays from sp_allocations list
        wholesale_arr = np.array([s.wholesale_mw  for s in allocation.sp_allocations])
        balancing_arr = np.array([s.balancing_mw  for s in allocation.sp_allocations])
        return {
            "wholesale_mw": wholesale_arr,
            "balancing_mw": balancing_arr,
        }

    except Exception as exc:
        logger.error("Allocation failed: %s", exc)
        return {}


def render() -> None:
    st.header("Capacity Allocation — One Day Ahead")

    predictions_loaded: bool = st.session_state.get(sk.PREDICTIONS_LOADED, False)
    if not predictions_loaded:
        st.info("Prediction files not found — using flat default forecasts. Run `python -m backend.predict` for model-based forecasts.")

    pred_sip: Optional[pd.DataFrame] = st.session_state.get(sk.PRED_SIP)
    pred_mip: Optional[pd.DataFrame] = st.session_state.get(sk.PRED_MIP)

    # ── Read shared parameters from unified sidebar ───────────────────────────
    fleet_size       = st.session_state.get(sk.SIM_FLEET_SIZE,    DEFAULT_FLEET_SIZE)
    dispatch_rate    = st.session_state.get(sk.SIM_DISPATCH_RATE, DEFAULT_DISPATCH_SUCCESS_RATE)
    override_rate    = st.session_state.get(sk.SIM_OVERRIDE_RATE, DEFAULT_OVERRIDE_RATE)
    da_price         = float(st.session_state.get(sk.DA_PRICE,    DEFAULT_DA_PRICE))
    mc_runs          = st.session_state.get(sk.SIM_MC_RUNS,       DEFAULT_MC_RUNS)
    profile_selected = st.session_state.get(sk.SIM_RISK_PROFILE,  "Moderate")

    sip_fc, mip_fc = _get_intraday_predictions(pred_sip, pred_mip)
    risk_pct = _RISK_PERCENTILES[profile_selected]

    # ── Run allocation ────────────────────────────────────────────────────────
    with st.spinner("Running MC simulation and allocation…"):
        allocation = _run_allocation(
            fleet_size=fleet_size,
            dispatch_rate=dispatch_rate,
            override_rate=override_rate,
            da_price=da_price,
            sip_fc=sip_fc,
            mip_fc=mip_fc,
            mc_runs=mc_runs,
            risk_percentile=risk_pct,
        )

    if allocation:
        st.session_state[sk.ALLOCATION_RESULT] = allocation

    # ── Fetch stored or current allocation arrays ─────────────────────────────
    wholesale_mw = np.zeros(48)
    balancing_mw = np.zeros(48)
    fleet_mw_total = fleet_size * CHARGER_CAPACITY_KW / 1000.0

    if allocation:
        wholesale_mw = np.asarray(allocation.get("wholesale_mw", np.zeros(48)), dtype=float)
        balancing_mw = np.asarray(allocation.get("balancing_mw", np.zeros(48)), dtype=float)
    else:
        # Fallback: simple proportional split
        held_pct = 1.0 - (risk_pct / 100.0)
        for sp in range(48):
            total = fleet_mw_total
            balancing_mw[sp] = total * (1 - held_pct)
            wholesale_mw[sp] = total * held_pct * 0.6

    held_mw = fleet_mw_total - wholesale_mw - balancing_mw
    held_mw = np.maximum(held_mw, 0)

    # ── Stacked bar chart ─────────────────────────────────────────────────────
    sp_labels = SP_LABELS

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sp_labels, y=wholesale_mw,
        name="Wholesale (DA)", marker_color="#00D4AA",
    ))
    fig.add_trace(go.Bar(
        x=sp_labels, y=balancing_mw,
        name="Balancing (BM)", marker_color="#FF6B6B",
    ))
    fig.add_trace(go.Bar(
        x=sp_labels, y=held_mw,
        name="Held Back", marker_color="#4A5568",
    ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        barmode="stack",
        title=f"Capacity Allocation — Tomorrow | Profile: {profile_selected} | Fleet: {fleet_size:,} vehicles",
        xaxis_title="Settlement Period",
        yaxis_title="MW",
        height=420,
        xaxis=dict(tickangle=-45, tickmode="array",
                   tickvals=sp_labels[::4], ticktext=sp_labels[::4]),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    )
    st.plotly_chart(fig, width="stretch")

    # ── SIP / MIP forecast overlay ────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=sp_labels, y=sip_fc, mode="lines",
            name="Forecast SIP", line=dict(color="#FF6B6B"),
        ))
        fig2.update_layout(
            template=PLOTLY_TEMPLATE, title="SIP Forecast (£/MWh)",
            height=280, xaxis=dict(tickangle=-45, tickmode="array",
                                   tickvals=sp_labels[::4], ticktext=sp_labels[::4]),
        )
        st.plotly_chart(fig2, width="stretch")
    with col2:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=sp_labels, y=mip_fc, mode="lines",
            name="Forecast MIP", line=dict(color="#00D4AA"),
        ))
        fig3.update_layout(
            template=PLOTLY_TEMPLATE, title="MIP Forecast (£/MWh)",
            height=280, xaxis=dict(tickangle=-45, tickmode="array",
                                   tickvals=sp_labels[::4], ticktext=sp_labels[::4]),
        )
        st.plotly_chart(fig3, width="stretch")

    # ── Detailed table ────────────────────────────────────────────────────────
    st.subheader("Settlement Period Detail")

    table_rows = []
    for sp in range(48):
        w_mw = wholesale_mw[sp]
        b_mw = balancing_mw[sp]
        h_mw = held_mw[sp]
        exp_rev   = w_mw * 0.5 * mip_fc[sp] if mip_fc[sp] > 0 else 0.0
        imb_est   = b_mw * 0.5 * sip_fc[sp] if sip_fc[sp] > 0 else 0.0

        if w_mw > b_mw * 1.5:
            strategy = "Wholesale-heavy"
        elif b_mw > w_mw * 1.5:
            strategy = "Balancing-heavy"
        else:
            strategy = "Balanced"

        table_rows.append({
            "SP": sp + 1,
            "Time": SP_LABELS[sp],
            "Wholesale MW": round(w_mw, 2),
            "Balancing MW": round(b_mw, 2),
            "Held Back MW": round(h_mw, 2),
            "Strategy":     strategy,
            "Est. Rev (£)": round(exp_rev, 2),
            "Est. Imb (£)": round(imb_est, 2),
        })

    df_table = pd.DataFrame(table_rows)
    st.dataframe(
        with_optional_background_gradient(
            df_table,
            subset=["Wholesale MW", "Balancing MW"],
            cmap="Blues",
        ),
        width="stretch",
        height=380,
    )

    # ── 4-profile comparison ──────────────────────────────────────────────────
    with st.expander("Compare all 4 risk profiles", expanded=False):
        _render_profile_comparison(fleet_size, dispatch_rate, override_rate, da_price, sip_fc, mip_fc, mc_runs)


def _render_profile_comparison(
    fleet_size, dispatch_rate, override_rate, da_price, sip_fc, mip_fc, mc_runs
) -> None:
    rows = []
    for profile, pct in _RISK_PERCENTILES.items():
        alloc = _run_allocation(
            fleet_size=fleet_size, dispatch_rate=dispatch_rate,
            override_rate=override_rate, da_price=da_price,
            sip_fc=sip_fc, mip_fc=mip_fc,
            mc_runs=min(mc_runs, 1_000),  # quick for comparison
            risk_percentile=pct,
        )
        if alloc:
            w_arr = np.asarray(alloc.get("wholesale_mw", np.zeros(48)), dtype=float)
            b_arr = np.asarray(alloc.get("balancing_mw", np.zeros(48)), dtype=float)
            total_rev = float(np.sum(w_arr * 0.5 * mip_fc))
            total_imb = float(np.sum(b_arr * 0.5 * sip_fc))
            rows.append({
                "Profile":         profile,
                "Avg Wholesale MW": round(float(np.mean(w_arr)), 2),
                "Avg Balancing MW": round(float(np.mean(b_arr)), 2),
                "Est. Total Rev (£)": round(total_rev, 2),
                "Est. Total Imb (£)": round(total_imb, 2),
                "Net (£)":           round(total_rev - total_imb, 2),
            })
        else:
            rows.append({"Profile": profile, "Avg Wholesale MW": 0, "Avg Balancing MW": 0,
                         "Est. Total Rev (£)": 0, "Est. Total Imb (£)": 0, "Net (£)": 0})

    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch")
