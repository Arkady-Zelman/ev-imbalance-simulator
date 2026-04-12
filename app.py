"""
Ohme Fleet Trading — Main Streamlit Frontend
============================================

Once UI scroll layout (Magic Portfolio design system).
Sections: Market & Allocation | Demand Map | Summary | Sensitivity

Model training runs offline via:
    python -m backend.train
    python -m backend.predict

This app reads saved parquet files from data/predictions/ and
runs MC simulation + allocation in-browser.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import streamlit as st

from src.config import DEFAULT_DA_PRICE, PREDICTION_DIR
from src.models.sip_models import derive_da_price_from_mip
import src.session_keys as sk

logger = logging.getLogger(__name__)

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Ohme Fleet Trading",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Once UI CSS ────────────────────────────────────────────────────────────────

from src.ui.styles import ONCE_UI_CSS
st.markdown(ONCE_UI_CSS, unsafe_allow_html=True)

# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def _load_parquet(path: Path):
    import pandas as pd
    if path.exists():
        return pd.read_parquet(path)
    return None


@st.cache_data(show_spinner=False, ttl=3600)
def _load_metadata(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _init_predictions() -> None:
    if st.session_state.get(sk.PREDICTIONS_LOADED):
        return

    pred_sip    = _load_parquet(PREDICTION_DIR / "sip_predictions.parquet")
    pred_mip    = _load_parquet(PREDICTION_DIR / "mip_predictions.parquet")
    pred_demand = _load_parquet(PREDICTION_DIR / "demand_predictions.parquet")
    pred_gen    = _load_parquet(PREDICTION_DIR / "gen_predictions.parquet")
    fan_data    = _load_parquet(PREDICTION_DIR / "backtest_fan.parquet")
    metadata    = _load_metadata(PREDICTION_DIR / "metadata.json")

    all_present = all(df is not None for df in [pred_sip, pred_mip, pred_demand, pred_gen, fan_data])

    st.session_state[sk.PRED_SIP]    = pred_sip
    st.session_state[sk.PRED_MIP]    = pred_mip
    st.session_state[sk.PRED_DEMAND] = pred_demand
    st.session_state[sk.PRED_GEN]    = pred_gen
    st.session_state[sk.FAN_DATA]    = fan_data
    st.session_state[sk.METADATA]    = metadata
    st.session_state[sk.PREDICTIONS_LOADED] = all_present

    if not all_present:
        logger.warning("One or more prediction parquet files missing from %s", PREDICTION_DIR)


def _init_market_data() -> None:
    if sk.SIP_DF in st.session_state and st.session_state[sk.SIP_DF] is not None:
        return

    today     = dt.date.today()
    date_from = today - dt.timedelta(days=90)

    with st.spinner("Loading market data from ELEXON…"):
        try:
            from src.data.elexon_client import (
                fetch_demand_outturn,
                fetch_generation_outturn,
                fetch_market_index,
                fetch_system_prices,
            )
            sip_df    = fetch_system_prices(date_from, today)
            mip_df    = fetch_market_index(date_from, today)
            demand_df = fetch_demand_outturn(date_from, today)
            gen_df    = fetch_generation_outturn(date_from, today)

            st.session_state[sk.SIP_DF]    = sip_df
            st.session_state[sk.MIP_DF]    = mip_df
            st.session_state[sk.DEMAND_DF] = demand_df
            st.session_state[sk.GEN_DF]    = gen_df

            da = derive_da_price_from_mip(mip_df)
            st.session_state[sk.DA_PRICE] = da if da is not None else DEFAULT_DA_PRICE

        except Exception as exc:
            logger.error("Market data fetch failed: %s", exc)
            st.session_state[sk.DA_PRICE] = DEFAULT_DA_PRICE


# ── Initialise data ────────────────────────────────────────────────────────────

_init_predictions()
_init_market_data()

# ── Header ─────────────────────────────────────────────────────────────────────

col_title, col_status = st.columns([8, 2])
with col_title:
    st.title("⚡ Ohme Fleet Trading")
with col_status:
    predictions_loaded = st.session_state.get(sk.PREDICTIONS_LOADED, False)
    metadata = st.session_state.get(sk.METADATA, {})
    if predictions_loaded:
        gen_at = metadata.get("generated_at", "")
        label  = f"Predictions: {gen_at[:10]}" if gen_at else "Predictions loaded"
        st.success(label)
    else:
        st.warning("No predictions — run `python -m backend.predict`")

# ── Unified sidebar ─────────────────────────────────────────────────────────────

from src.config import (
    DEFAULT_DA_PRICE, DEFAULT_DISPATCH_SUCCESS_RATE, DEFAULT_FLEET_SIZE,
    DEFAULT_MC_RUNS, DEFAULT_OVERRIDE_RATE, DEFAULT_RISK_APPETITE, RISK_APPETITES,
)

with st.sidebar:
    run_sim = st.button("▶  Run Simulation", type="primary", use_container_width=True)
    st.session_state[sk.SIM_RUN_REQUESTED] = run_sim

    st.divider()
    st.subheader("Fleet Parameters")
    _fleet    = st.number_input("Fleet size (vehicles)", 1_000, 100_000,
                                st.session_state.get(sk.SIM_FLEET_SIZE, DEFAULT_FLEET_SIZE),
                                1_000, key=sk.SIM_FLEET_SIZE)
    _dispatch = st.slider("Dispatch success rate", 0.5, 1.0,
                          st.session_state.get(sk.SIM_DISPATCH_RATE, DEFAULT_DISPATCH_SUCCESS_RATE),
                          0.01, key=sk.SIM_DISPATCH_RATE)
    _override = st.slider("Override rate", 0.0, 0.2,
                          st.session_state.get(sk.SIM_OVERRIDE_RATE, DEFAULT_OVERRIDE_RATE),
                          0.005, key=sk.SIM_OVERRIDE_RATE)
    _da       = st.number_input("DA price (£/MWh)", 0.0, 500.0,
                                float(st.session_state.get(sk.DA_PRICE, DEFAULT_DA_PRICE)),
                                1.0, key="sidebar_da_price")
    st.session_state[sk.DA_PRICE] = _da

    st.divider()
    st.subheader("Monte Carlo")
    _mc_runs  = st.selectbox("MC runs", [1_000, 5_000, 10_000],
                             index=[1_000, 5_000, 10_000].index(
                                 st.session_state.get(sk.SIM_MC_RUNS, DEFAULT_MC_RUNS)
                             ) if st.session_state.get(sk.SIM_MC_RUNS, DEFAULT_MC_RUNS) in [1_000, 5_000, 10_000] else 1,
                             key=sk.SIM_MC_RUNS)
    _risk_app = st.selectbox("Risk appetite", list(RISK_APPETITES.keys()),
                             index=list(RISK_APPETITES.keys()).index(
                                 st.session_state.get(sk.SIM_RISK_APPETITE, DEFAULT_RISK_APPETITE)
                             ),
                             key=sk.SIM_RISK_APPETITE)
    _risk_prof = st.selectbox("Risk profile (allocation)",
                              ["Conservative", "Moderate", "Aggressive", "Full Risk"],
                              index=["Conservative", "Moderate", "Aggressive", "Full Risk"].index(
                                  st.session_state.get(sk.SIM_RISK_PROFILE, "Moderate")
                              ),
                              key=sk.SIM_RISK_PROFILE)

    st.divider()
    st.caption("MC runs (sensitivity)")
    _mc_sens  = st.selectbox("Per-scenario runs", [500, 1_000, 2_000],
                             index=0, key=sk.SIM_MC_RUNS_SENS,
                             label_visibility="collapsed")

# ── Scroll-spy nav ─────────────────────────────────────────────────────────────

from src.ui.layout import inject_scrollnav, section_start, section_end

inject_scrollnav()

# ── Tab imports ────────────────────────────────────────────────────────────────

from tabs.tab_main_chart          import render as render_main_chart
from tabs.tab_capacity_allocation import render as render_allocation
from tabs.tab_demand_map          import render as render_demand_map
from tabs.tab_executive_summary   import render as render_executive
from tabs.tab_sensitivity         import render as render_sensitivity

# ══════════════════════════════════════════════════════════════════════════════
# Section 1 — Market Overview + Capacity Allocation
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(section_start("sec-market", "Market & Allocation"), unsafe_allow_html=True)
render_main_chart()
st.markdown("<hr style='border:none;border-top:1px solid var(--border);margin:32px 0 24px;'>",
            unsafe_allow_html=True)
render_allocation()
st.markdown(section_end(), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Section 2 — Demand Map
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(section_start("sec-demand", "Demand Map"), unsafe_allow_html=True)
render_demand_map()
st.markdown(section_end(), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Section 3 — Executive Summary
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(section_start("sec-summary", "Summary"), unsafe_allow_html=True)
render_executive()
st.markdown(section_end(), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Section 4 — Sensitivity & Scenario Analysis
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(section_start("sec-sens", "Sensitivity"), unsafe_allow_html=True)
render_sensitivity()
st.markdown(section_end(), unsafe_allow_html=True)
