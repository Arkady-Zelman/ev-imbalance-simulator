"""
EV Flexibility Portfolio Imbalance Exposure Simulator
=====================================================
Main Streamlit entry point.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import streamlit as st

from src.config import (
    CHARGER_CAPACITY_KW,
    DAYTYPE_MULTIPLIERS,
    DEFAULT_DA_PRICE,
    DEFAULT_DISPATCH_SUCCESS_RATE,
    DEFAULT_FLEET_SIZE,
    DEFAULT_MC_RUNS,
    DEFAULT_OVERRIDE_RATE,
    DEFAULT_RISK_APPETITE,
    MAX_FLEET_SIZE,
    MC_RUNS_OPTIONS,
    MIN_FLEET_SIZE,
    PLUGIN_CLUSTERS,
    RISK_APPETITES,
    SEASONAL_MONTHLY,
    SIP_STRESS_DISPATCH_PENALTY,
    SIP_STRESS_PLUGIN_FACTOR,
)
from src.data.elexon_client import fetch_market_index, fetch_system_prices
from src.models.kelly import kelly_optimal_position, run_kelly_analysis
from src.models.monte_carlo import SimulationParams, prepare_sip_matrix, run_simulation
from src.models.risk_metrics import compute_capture_ratios, compute_risk_summary
from src.models.sip_models import (
    derive_da_price_from_mip,
    fit_regime_params,
    generate_regime_switching_sip,
)
from src.models.trading_position import compute_traded_positions
from src.session_keys import (
    ALL_POSITIONS,
    BACKTEST_RESULTS,
    BANKROLL,
    CAPTURE_RATIOS,
    DA_PRICE,
    DATE_FROM,
    DATE_TO,
    KELLY_RESULTS,
    MIP_DF,
    PARAMS,
    RESULT,
    RISK_SUMMARY,
    SIZING_METHOD,
    SIP_DF,
    SIP_MATRIX,
    SIP_MODE,
)

# ── Page configuration ────────────────────────────────────────────────────

st.set_page_config(
    page_title="EV Imbalance Exposure Simulator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚡ Simulation Parameters")
    st.markdown("---")

    # ── Fleet ─────────────────────────────────────────────────────────
    st.subheader("Fleet Configuration")
    fleet_size = st.slider(
        "Fleet Size (chargers)",
        min_value=MIN_FLEET_SIZE,
        max_value=MAX_FLEET_SIZE,
        value=DEFAULT_FLEET_SIZE,
        step=1_000,
        help="Number of active Ohme chargers in the portfolio",
    )
    st.caption(f"Charger capacity: **{CHARGER_CAPACITY_KW} kW** (Ohme Home Pro)")
    max_mw = fleet_size * CHARGER_CAPACITY_KW / 1_000
    st.caption(f"Theoretical max: **{max_mw:,.1f} MW**")

    # ── Reliability ───────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Reliability Parameters")
    dispatch_pct = st.slider(
        "Dispatch Success Rate (%)",
        min_value=85.0, max_value=99.5, value=DEFAULT_DISPATCH_SUCCESS_RATE * 100,
        step=0.5, format="%.1f%%",
        help="Fraction of plugged-in chargers that successfully receive and execute the dispatch signal",
    )
    dispatch_rate = dispatch_pct / 100.0

    override_pct = st.slider(
        "Customer Override Rate (%)",
        min_value=0.0, max_value=15.0, value=DEFAULT_OVERRIDE_RATE * 100,
        step=0.5, format="%.1f%%",
        help="Fraction of dispatched chargers where the customer overrides smart charging",
    )
    override_rate = override_pct / 100.0

    # ── Trading & Risk ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Trading & Risk")

    sizing_method = st.radio(
        "Position Sizing Method",
        ["Percentile", "Kelly Criterion"],
        index=1,
        horizontal=True,
        help="Percentile = arbitrary quantile of availability. "
             "Kelly = maximises long-run geometric growth given DA/SIP payoff asymmetry.",
    )

    if sizing_method == "Percentile":
        risk_label = st.selectbox(
            "Risk Appetite (Percentile)",
            options=list(RISK_APPETITES.keys()),
            index=list(RISK_APPETITES.keys()).index(DEFAULT_RISK_APPETITE),
            help="Trade at this percentile of simulated availability.",
        )
        risk_percentile = RISK_APPETITES[risk_label]
        kelly_fraction = 0.5
        bankroll = 100_000.0
    else:
        risk_percentile = 80
        kelly_fraction = st.select_slider(
            "Kelly Fraction",
            options=[0.25, 0.50, 0.75, 1.00],
            value=0.50,
            format_func=lambda f: {0.25: "¼ Kelly (conservative)",
                                    0.50: "½ Kelly (standard)",
                                    0.75: "¾ Kelly (aggressive)",
                                    1.00: "Full Kelly (max growth)"}[f],
            help="½ Kelly is industry standard: ~75% of growth rate, ~50% of drawdown variance.",
        )
        bankroll = st.number_input(
            "Daily Risk Budget / Bankroll (£)",
            min_value=10_000.0, max_value=10_000_000.0,
            value=100_000.0, step=10_000.0,
            help="Total capital at risk. Kelly sizes positions as a fraction of this.",
        )

    da_price = st.number_input(
        "Day-Ahead Price Assumption (£/MWh)",
        min_value=0.0, max_value=500.0,
        value=DEFAULT_DA_PRICE, step=5.0,
        help=f"Default {DEFAULT_DA_PRICE} is a rough GB average. "
             "After fetching MIP data you can auto-derive this below.",
    )

    # ── Monte Carlo Settings ──────────────────────────────────────────
    st.markdown("---")
    st.subheader("Monte Carlo Settings")
    n_runs = st.select_slider(
        "Number of Simulations",
        options=MC_RUNS_OPTIONS,
        value=DEFAULT_MC_RUNS,
    )
    seed = st.number_input("Random Seed (0 = random)", min_value=0,
                           max_value=999999, value=42, step=1)

    da_noise_sigma = st.slider(
        "DA Price Noise (lognormal σ)",
        min_value=0.0, max_value=0.20, value=0.05, step=0.01,
        help="Revenue jitter applied per MC run. 0.0 = deterministic DA revenue. "
             "0.05 = ±5% mean-preserving lognormal noise. "
             "Models day-to-day DA auction variance.",
    )

    # ── Day-Type & Seasonal ───────────────────────────────────────────
    st.markdown("---")
    st.subheader("Day-Type & Seasonal")
    day_type = st.selectbox(
        "Day Type",
        options=["weekday", "weekend", "holiday"],
        index=0,
        help="Weekend/holiday profiles have higher evening plug-in rates.",
    )
    sim_month = st.select_slider(
        "Month (seasonal profile)",
        options=list(range(1, 13)),
        value=1,
        format_func=lambda m: ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][m - 1],
        help="Winter months have higher plug-in rates; summer has lower.",
    )

    with st.expander("Day-Type Multipliers", expanded=False):
        st.caption(
            "Multipliers applied to base plug-in rates. "
            ">1 means higher availability; <1 means lower. "
            "Source: estimated from CrowdFlex usage patterns."
        )
        dt_weekday = st.number_input(
            "Weekday", min_value=0.50, max_value=1.50,
            value=DAYTYPE_MULTIPLIERS["weekday"], step=0.01, format="%.2f",
            key="dt_weekday",
        )
        dt_weekend = st.number_input(
            "Weekend", min_value=0.50, max_value=1.50,
            value=DAYTYPE_MULTIPLIERS["weekend"], step=0.01, format="%.2f",
            key="dt_weekend",
        )
        dt_holiday = st.number_input(
            "Holiday", min_value=0.50, max_value=1.50,
            value=DAYTYPE_MULTIPLIERS["holiday"], step=0.01, format="%.2f",
            key="dt_holiday",
        )
    custom_daytype = {"weekday": dt_weekday, "weekend": dt_weekend, "holiday": dt_holiday}

    with st.expander("Seasonal Monthly Factors", expanded=False):
        st.caption(
            "Monthly multiplier on plug-in rates (1.0 = baseline). "
            "Source: estimated — no public dataset available. "
            "Winter higher (vehicles at home more), summer lower (travel)."
        )
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        custom_seasonal: dict[int, float] = {}
        cols_s = st.columns(4)
        for i in range(12):
            with cols_s[i % 4]:
                custom_seasonal[i + 1] = st.number_input(
                    month_names[i],
                    min_value=0.50, max_value=1.50,
                    value=SEASONAL_MONTHLY[i + 1],
                    step=0.01, format="%.2f",
                    key=f"seasonal_{i+1}",
                )

    # ── SIP-Stress Coupling ───────────────────────────────────────────
    sip_stress = st.checkbox(
        "SIP-Availability Stress Coupling",
        value=True,
        help="When SIP is high (system stress), degrade plug-in rates "
             "and dispatch success — models the adverse real-world correlation.",
    )

    if sip_stress:
        with st.expander("Stress Coupling Factors", expanded=False):
            st.caption(
                "When SIP is in the top quintile (grid stress), these factors "
                "degrade fleet reliability. Source: engineering estimate — no "
                "public data. <1.0 means degradation."
            )
            stress_plugin = st.slider(
                "Plug-in rate factor under stress",
                min_value=0.70, max_value=1.00,
                value=SIP_STRESS_PLUGIN_FACTOR, step=0.01,
                help="E.g. 0.92 = plug-in rates drop 8% when SIP is high.",
            )
            stress_dispatch = st.slider(
                "Dispatch success factor under stress",
                min_value=0.70, max_value=1.00,
                value=SIP_STRESS_DISPATCH_PENALTY, step=0.01,
                help="E.g. 0.97 = dispatch success drops 3% when SIP is high.",
            )
    else:
        stress_plugin = SIP_STRESS_PLUGIN_FACTOR
        stress_dispatch = SIP_STRESS_DISPATCH_PENALTY

    # ── Plug-in Rate Profiles ─────────────────────────────────────────
    with st.expander("Plug-in Rate Profiles (Beta distribution)", expanded=False):
        st.caption(
            "The entire fleet availability model is driven by these 5 time-of-day "
            "clusters. **Source:** educated guesses inspired by CrowdFlex trial data "
            "(pre-engagement ~50% overnight, ~18-28% daytime, post-engagement ~70% "
            "overnight). No live telemetry feed is available. Adjust here to model "
            "different fleet characteristics."
        )
        plugin_overrides: list[tuple[float, float]] = []
        for cluster in PLUGIN_CLUSTERS:
            st.markdown(f"**{cluster.name}** (SP {cluster.sp_range[0]+1}–{cluster.sp_range[1]+1})")
            c1, c2 = st.columns(2)
            with c1:
                m = st.slider(
                    f"Mean ({cluster.name})",
                    min_value=0.05, max_value=0.95,
                    value=cluster.mean, step=0.01,
                    key=f"plugin_mean_{cluster.name}",
                )
            with c2:
                c = st.slider(
                    f"Concentration ν ({cluster.name})",
                    min_value=5, max_value=200,
                    value=int(cluster.concentration), step=5,
                    key=f"plugin_conc_{cluster.name}",
                )
            plugin_overrides.append((m, float(c)))

    # ── Advanced: Copula & Decay ──────────────────────────────────────
    with st.expander("Advanced: Copula & Decay"):
        corr_decay = st.slider(
            "Inter-SP correlation decay",
            min_value=0.05, max_value=1.0, value=0.3, step=0.05,
            help="Exponential decay rate for the Gaussian copula. "
                 "Lower = stronger correlation between adjacent SPs.",
        )

    # ── Market Data ───────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Market Data (ELEXON)")
    today = dt.date.today()
    default_from = today - dt.timedelta(days=90)
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        date_from = st.date_input("From", value=default_from)
    with col_d2:
        date_to = st.date_input("To", value=today)
    date_span = (date_to - date_from).days
    if date_span > 90:
        st.caption(f"⏳ Fetching {date_span} days of SIP data (~1 API call/day). This may take a minute.")

    sip_mode = st.radio(
        "SIP Modelling",
        ["Empirical Bootstrap", "Regime-Switching (fitted from data)"],
        index=0,
        help="Bootstrap samples full historical days. "
             "Regime-switching fits Normal/LogNormal parameters from your fetched SIP data.",
    )

    st.markdown("---")
    run_clicked = st.button("🚀 Run Simulation", use_container_width=True, type="primary")

# ── Run simulation on button click ────────────────────────────────────────

if run_clicked:
    with st.spinner("Fetching ELEXON market data…"):
        sip_df = fetch_system_prices(date_from, date_to)
        mip_df = fetch_market_index(date_from, date_to)

    if sip_df.empty:
        st.error("No SIP data returned from ELEXON. Check your date range or network connection.")
        st.stop()

    # Derive DA price from MIP if user left the default
    mip_derived_da = derive_da_price_from_mip(mip_df)
    if mip_derived_da is not None and abs(da_price - DEFAULT_DA_PRICE) < 0.01:
        st.info(
            f"**Auto-derived DA price:** Mean MIP over your date range is "
            f"**£{mip_derived_da}/MWh** (your setting: £{da_price}/MWh). "
            f"Consider updating the DA price input to match."
        )

    with st.spinner(f"Running {n_runs:,} Monte Carlo simulations…"):
        sip_matrix, sip_is_fallback = prepare_sip_matrix(sip_df)

        if sip_is_fallback:
            st.warning(
                "**SIP data fallback active:** The ELEXON data could not be parsed into "
                "a valid SIP matrix. The simulation is running on **flat dummy data "
                f"(£{DEFAULT_DA_PRICE}/MWh with zero volatility)**. Results will be "
                "unrealistically benign. Check your date range or API connectivity."
            )

        if sip_mode.startswith("Regime-Switching"):
            fitted_params = fit_regime_params(sip_df)
            st.info(
                f"**Regime params fitted from {len(sip_df)} ELEXON observations:** "
                f"Normal μ=£{fitted_params.normal_mean}/MWh, σ=£{fitted_params.normal_std}/MWh | "
                f"Spike P={fitted_params.spike_probability:.1%}, "
                f"μ_log={fitted_params.spike_mean_log:.2f}, σ_log={fitted_params.spike_std_log:.2f}"
            )
            n_synthetic = max(200, n_runs // 10)
            sip_matrix = generate_regime_switching_sip(
                n_days=n_synthetic,
                params=fitted_params,
                seed=seed if seed else None,
            )

        params = SimulationParams(
            fleet_size=fleet_size,
            dispatch_rate=dispatch_rate,
            override_rate=override_rate,
            n_runs=n_runs,
            risk_percentile=risk_percentile,
            correlation_decay=corr_decay,
            day_type=day_type,
            month=sim_month,
            sip_stress_coupling=sip_stress,
            seed=seed if seed else None,
            plugin_overrides=plugin_overrides,
            da_noise_sigma=da_noise_sigma,
            sip_stress_plugin_factor=stress_plugin,
            sip_stress_dispatch_penalty=stress_dispatch,
            daytype_multipliers=custom_daytype,
            seasonal_monthly=custom_seasonal,
        )

        result = run_simulation(params, sip_matrix, da_price=da_price)

        kelly_results = None
        if sizing_method == "Kelly Criterion":
            with st.spinner("Computing Kelly-optimal positions…"):
                kelly_results = run_kelly_analysis(
                    result.delivered_mw, result.sip_matrix,
                    da_price=da_price, bankroll=bankroll,
                )
                match = [kr for kr in kelly_results if abs(kr.fraction - kelly_fraction) < 0.01]
                if match:
                    result.traded_mw[:] = match[0].optimal_mw

        all_positions = compute_traded_positions(result.delivered_mw)

        capture_ratios = compute_capture_ratios(
            result.delivered_mw, result.traded_mw,
            da_price=da_price,
            sip_matrix=result.sip_matrix,
        )
        risk_summary = compute_risk_summary(result.daily_pnl, capture_ratios)

    st.session_state[RESULT] = result
    st.session_state[ALL_POSITIONS] = all_positions
    st.session_state[CAPTURE_RATIOS] = capture_ratios
    st.session_state[RISK_SUMMARY] = risk_summary
    st.session_state[SIP_DF] = sip_df
    st.session_state[MIP_DF] = mip_df
    st.session_state[SIP_MATRIX] = sip_matrix
    st.session_state[PARAMS] = params
    st.session_state[DA_PRICE] = da_price
    st.session_state[DATE_FROM] = date_from
    st.session_state[DATE_TO] = date_to
    st.session_state[SIP_MODE] = sip_mode
    st.session_state[SIZING_METHOD] = sizing_method
    st.session_state[BANKROLL] = bankroll
    st.session_state[KELLY_RESULTS] = kelly_results

# ── Tab routing ───────────────────────────────────────────────────────────

from tabs.tab_executive_summary import render as render_executive
from tabs.tab_portfolio_availability import render as render_portfolio
from tabs.tab_monte_carlo import render as render_mc
from tabs.tab_risk_analysis import render as render_risk
from tabs.tab_sensitivity import render as render_sensitivity
from tabs.tab_scenario_comparison import render as render_scenario
from tabs.tab_historical_sip import render as render_sip_explorer
from tabs.tab_data_sources import render as render_data_sources
from tabs.tab_backtesting import render as render_backtesting
from tabs.tab_rolling_backtest import render as render_rolling_bt

tabs = st.tabs([
    "📊 Executive Summary",
    "🔌 Portfolio Availability",
    "🎲 Monte Carlo Results",
    "⚖️ Risk Analysis",
    "🔬 Sensitivity Analysis",
    "🌡️ Scenario Comparison",
    "📈 Historical SIP Explorer",
    "🔮 Forecast Backtesting",
    "📉 Rolling Backtest",
    "📚 Data Sources & Methodology",
])

has_results = RESULT in st.session_state

with tabs[0]:
    render_executive(has_results)
with tabs[1]:
    render_portfolio(has_results)
with tabs[2]:
    render_mc(has_results)
with tabs[3]:
    render_risk(has_results)
with tabs[4]:
    render_sensitivity(has_results)
with tabs[5]:
    render_scenario(has_results)
with tabs[6]:
    render_sip_explorer(has_results)
with tabs[7]:
    render_backtesting(has_results)
with tabs[8]:
    render_rolling_bt(has_results)
with tabs[9]:
    render_data_sources()
