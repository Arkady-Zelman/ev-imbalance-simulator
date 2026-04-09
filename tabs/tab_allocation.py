"""
Capacity Allocation Optimizer tab — translates three forecast models
(Price/SIP, Wholesale/MIP, Demand) plus Monte Carlo fleet availability
into actionable trading recommendations: how much to commit to wholesale
vs. balancing markets, and whether to overbook or underbook.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    COLOUR_DANGER,
    COLOUR_MUTED,
    COLOUR_PRIMARY,
    COLOUR_SUCCESS,
    COLOUR_WARNING,
    NUM_SETTLEMENT_PERIODS,
    PLOTLY_TEMPLATE,
    SP_LABELS,
)
from src.models.allocation_optimizer import (
    AllocationResult,
    MultiProfileResult,
    RISK_PROFILES,
    optimize_allocation,
    optimize_all_risk_profiles,
)
from src.models.forecaster import build_aligned_series
from src.models.prophet_trainer import (
    TrainedNPModels,
    forecast_forward_np,
    load_np_models,
)
from src.models.xgb_trainer import (
    forecast_forward,
    forecast_intraday_48sp,
    has_trained_models,
    load_trained_models,
)
from src.session_keys import DEMAND_DF, EXOG_SERIES, MIP_DF, MULTI_PROFILE_RESULTS, RESULT, SIP_DF
from src.models.lstm_trainer import (
    TrainedLSTMModels,
    forecast_intraday_48sp as lstm_forecast_intraday_48sp,
    has_trained_lstm_models,
    load_trained_lstm_models,
)
from src.models import hybrid_forecaster as _hybrid


def _build_exog_dict_for_forecast(
    trained,
    target_series: "pd.Series",
) -> "Optional[Dict[str, np.ndarray]]":
    """
    Reconstruct the exog_dict for inference from session-state exog_series.
    Works for both TrainedXGBModels and TrainedLSTMModels.
    """
    from src.models.xgb_trainer import TrainedXGBModels, _align_to_target
    from src.models.lstm_trainer import _align_to_target as _lstm_align
    exog_keys = getattr(trained, "exog_keys", [])
    if not exog_keys:
        return None
    stored: dict = st.session_state.get(EXOG_SERIES) or {}
    if not stored:
        return None
    align_fn = _align_to_target if isinstance(trained, TrainedXGBModels) else _lstm_align
    result_dict: dict = {}
    for k in exog_keys:
        if k in stored:
            result_dict[k] = align_fn(target_series, stored[k])
    return result_dict if result_dict else None


def render(has_results: bool) -> None:
    st.header("Capacity Allocation Optimizer")
    st.caption(
        "Combines SIP, wholesale (MIP), and demand forecasts with Monte Carlo "
        "fleet availability to recommend the optimal wholesale vs. balancing "
        "market split per settlement period — including overbook/underbook guidance."
    )

    if not has_results:
        st.info("Run a simulation first to use this tool.")
        return

    result = st.session_state.get(RESULT)
    if result is None:
        st.warning("No simulation results available.")
        return

    sip_df = st.session_state.get(SIP_DF)
    mip_df = st.session_state.get(MIP_DF)
    demand_df = st.session_state.get(DEMAND_DF)

    if sip_df is None or sip_df.empty or mip_df is None or mip_df.empty:
        st.warning("SIP and MIP data are both required. Run a simulation first.")
        return

    _render_allocation_body(result)


@st.fragment
def _render_allocation_body(result) -> None:
    sip_df = st.session_state[SIP_DF]
    mip_df = st.session_state[MIP_DF]
    demand_df = st.session_state.get(DEMAND_DF)
    delivered_mw = result.delivered_mw  # (n_runs, 48)

    # ── Forecast model selection ───────────────────────────────────────
    st.markdown("---")
    st.subheader("0. Forecast Model Selection")
    col_s, col_m = st.columns(2)
    _engine_options = ["XGBoost", "LSTM", "Hybrid LSTM+XGBoost", "NeuralProphet"]
    with col_s:
        sip_engine = st.selectbox(
            "SIP forecast engine", _engine_options,
            key="_alloc_sip_engine",
            help="Which trained model to use for the SIP (balancing price) forecast.",
        )
    with col_m:
        mip_engine = st.selectbox(
            "MIP forecast engine", _engine_options,
            key="_alloc_mip_engine",
            help="Which trained model to use for the MIP (wholesale price) forecast.",
        )

    # ── Load or prompt for trained models ─────────────────────────────
    trained_sip = _ensure_model("sip", "Price (SIP)", sip_engine)
    trained_mip = _ensure_model("mip", "Wholesale (MIP)", mip_engine)

    def _model_available(m) -> bool:
        if m is None:
            return False
        if isinstance(m, tuple):
            return any(x is not None for x in m)
        return True

    if not _model_available(trained_sip) and not _model_available(trained_mip):
        st.error(
            "At least one trained model (Price or Wholesale) is required. "
            "Go to the **Rolling Backtest** tab and train an XGBoost, LSTM, or NeuralProphet model first."
        )
        return

    # ── Build aligned series ───────────────────────────────────────────
    sip_series, mip_series, demand_series, _ = build_aligned_series(
        sip_df, mip_df, demand_df=demand_df,
    )
    sip_values = sip_series.values.astype(float)
    mip_values = mip_series.values.astype(float)
    demand_values = demand_series.values.astype(float) if demand_series is not None else None

    # ── Intraday 48-SP day-ahead forecast (lead content) ──────────────
    st.markdown("---")
    st.subheader("1. Intraday Day-Ahead Forecast")
    st.caption(
        "True settlement-period-level forecast for the next 48 SPs using the trained model. "
        "Each point is one 30-minute period. Select the lookback window to use "
        "as the price input to the allocation optimiser below."
    )

    sip_intraday: dict = {}
    mip_intraday: dict = {}

    if isinstance(trained_sip, TrainedNPModels):
        _sip_scalar = _build_48sp_forecast_legacy(
            trained_sip, sip_values, mip_values, demand_values, "SIP"
        )
        if _sip_scalar is not None:
            sip_intraday = {"NP": _sip_scalar}
    elif isinstance(trained_sip, tuple):
        # Hybrid LSTM+XGBoost
        xgb_m, lstm_m = trained_sip
        xgb_fc_sip  = forecast_intraday_48sp(xgb_m,  sip_values, mip_values, demand_values,
                                              exog_dict=_build_exog_dict_for_forecast(xgb_m, sip_series)) if xgb_m else {}
        lstm_fc_sip = lstm_forecast_intraday_48sp(lstm_m, sip_values, mip_values, demand_values,
                                                   exog_dict=_build_exog_dict_for_forecast(lstm_m, sip_series)) if lstm_m else {}
        hybrid_sip  = _hybrid.combine_intraday(xgb_fc_sip, lstm_fc_sip, xgb_trained=xgb_m, lstm_trained=lstm_m)
        sip_intraday = hybrid_sip.combined
        st.caption(f"SIP hybrid weights — LSTM: {hybrid_sip.lstm_weight:.0%} · XGBoost: {hybrid_sip.xgb_weight:.0%}")
    elif isinstance(trained_sip, TrainedLSTMModels):
        _sip_exog = _build_exog_dict_for_forecast(trained_sip, sip_series)
        sip_intraday = lstm_forecast_intraday_48sp(trained_sip, sip_values, mip_values, demand_values, exog_dict=_sip_exog)
    elif trained_sip is not None:
        _sip_exog = _build_exog_dict_for_forecast(trained_sip, sip_series)
        sip_intraday = forecast_intraday_48sp(trained_sip, sip_values, mip_values, demand_values, exog_dict=_sip_exog)

    if isinstance(trained_mip, TrainedNPModels):
        _mip_scalar = _build_48sp_forecast_legacy(
            trained_mip, sip_values, mip_values, demand_values, "MIP"
        )
        if _mip_scalar is not None:
            mip_intraday = {"NP": _mip_scalar}
    elif isinstance(trained_mip, tuple):
        # Hybrid LSTM+XGBoost
        xgb_m, lstm_m = trained_mip
        xgb_fc_mip  = forecast_intraday_48sp(xgb_m,  sip_values, mip_values, demand_values,
                                              exog_dict=_build_exog_dict_for_forecast(xgb_m, mip_series)) if xgb_m else {}
        lstm_fc_mip = lstm_forecast_intraday_48sp(lstm_m, sip_values, mip_values, demand_values,
                                                   exog_dict=_build_exog_dict_for_forecast(lstm_m, mip_series)) if lstm_m else {}
        hybrid_mip  = _hybrid.combine_intraday(xgb_fc_mip, lstm_fc_mip, xgb_trained=xgb_m, lstm_trained=lstm_m)
        mip_intraday = hybrid_mip.combined
        st.caption(f"MIP hybrid weights — LSTM: {hybrid_mip.lstm_weight:.0%} · XGBoost: {hybrid_mip.xgb_weight:.0%}")
    elif isinstance(trained_mip, TrainedLSTMModels):
        _mip_exog = _build_exog_dict_for_forecast(trained_mip, mip_series)
        mip_intraday = lstm_forecast_intraday_48sp(trained_mip, sip_values, mip_values, demand_values, exog_dict=_mip_exog)
    elif trained_mip is not None:
        _mip_exog = _build_exog_dict_for_forecast(trained_mip, mip_series)
        mip_intraday = forecast_intraday_48sp(trained_mip, sip_values, mip_values, demand_values, exog_dict=_mip_exog)

    available_lbs = sorted(set(list(sip_intraday.keys()) + list(mip_intraday.keys())))
    if not available_lbs:
        st.error("Could not generate any intraday forecasts. Train models first.")
        return

    selected_lb = st.radio(
        "Lookback window for optimiser",
        available_lbs,
        horizontal=True,
        key="_alloc_intraday_lb",
    )

    sp_labels_full = SP_LABELS[:NUM_SETTLEMENT_PERIODS]
    hist_sip = sip_values[-48:] if len(sip_values) >= 48 else sip_values
    hist_mip = mip_values[-48:] if len(mip_values) >= 48 else mip_values
    hist_labels = [f"T-{len(hist_sip)-i}" for i in range(len(hist_sip))]

    fig_id = go.Figure()
    fig_id.add_trace(go.Scatter(
        x=hist_labels, y=hist_sip.tolist(),
        mode="lines", name="SIP (last 48 actuals)",
        line=dict(color=COLOUR_MUTED, width=1, dash="dot"),
    ))
    fig_id.add_trace(go.Scatter(
        x=hist_labels, y=hist_mip.tolist(),
        mode="lines", name="MIP (last 48 actuals)",
        line=dict(color=COLOUR_MUTED, width=1, dash="dash"),
    ))
    if selected_lb in sip_intraday:
        fig_id.add_trace(go.Bar(
            x=sp_labels_full, y=sip_intraday[selected_lb].tolist(),
            name=f"SIP Forecast ({selected_lb})",
            marker_color=COLOUR_PRIMARY, opacity=0.8,
        ))
    if selected_lb in mip_intraday:
        fig_id.add_trace(go.Scatter(
            x=sp_labels_full, y=mip_intraday[selected_lb].tolist(),
            mode="lines+markers", name=f"MIP Forecast ({selected_lb})",
            line=dict(color=COLOUR_WARNING, width=2),
            marker=dict(size=4),
        ))
    fig_id.update_layout(
        template=PLOTLY_TEMPLATE,
        title=f"Intraday Day-Ahead Forecast — {selected_lb} lookback",
        xaxis_title="Settlement Period",
        yaxis_title="£/MWh",
        height=460,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_id, use_container_width=True)

    if st.toggle("Compare all lookbacks", key="_alloc_show_all_lbs", value=False):
        fig_all = go.Figure()
        for lb, arr in sip_intraday.items():
            fig_all.add_trace(go.Scatter(
                x=sp_labels_full, y=arr.tolist(), mode="lines", name=f"SIP {lb}",
            ))
        fig_all.update_layout(
            template=PLOTLY_TEMPLATE, title="SIP Forecast — All Lookbacks",
            xaxis_title="Settlement Period", yaxis_title="£/MWh", height=360,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_all, use_container_width=True)

    # Resolve final 48-SP arrays for the optimiser
    sip_fc_48 = sip_intraday.get(selected_lb)
    mip_fc_48 = mip_intraday.get(selected_lb)

    if sip_fc_48 is None or np.all(np.isnan(sip_fc_48)):
        sip_fc_48 = sip_values[-48:].copy()
        st.caption("No SIP intraday forecast — using last observed day as fallback.")
    else:
        sip_fc_48 = np.nan_to_num(sip_fc_48, nan=float(np.nanmean(sip_fc_48)))

    if mip_fc_48 is None or np.all(np.isnan(mip_fc_48)):
        mip_fc_48 = mip_values[-48:].copy()
        st.caption("No MIP intraday forecast — using last observed day as fallback.")
    else:
        mip_fc_48 = np.nan_to_num(mip_fc_48, nan=float(np.nanmean(mip_fc_48)))

    # Cache for multi-profile section
    st.session_state["_alloc_sip_fc_48"] = sip_fc_48
    st.session_state["_alloc_mip_fc_48"] = mip_fc_48

    # Secondary: spread chart
    _render_forecast_summary(sip_fc_48, mip_fc_48, trained_sip, trained_mip)

    # ── Risk tolerance slider ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("2. Risk Tolerance")
    risk_tolerance = st.slider(
        "Risk Tolerance",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        key="alloc_risk",
        help="0 = pure expected-revenue maximisation (risk-neutral). "
             "1 = maximise risk-adjusted return (penalise tail losses via ES).",
    )

    # ── Run optimizer ─────────────────────────────────────────────────
    run_btn = st.button(
        "⚡ Optimise Allocation",
        use_container_width=True,
        type="primary",
        key="alloc_run",
    )

    alloc_key = "_allocation_result"

    if run_btn:
        with st.spinner("Optimising wholesale vs. balancing allocation across 48 SPs…"):
            alloc = optimize_allocation(
                sip_forecasts=sip_fc_48,
                mip_forecasts=mip_fc_48,
                delivered_mw=delivered_mw,
                risk_tolerance=risk_tolerance,
            )
        st.session_state[alloc_key] = alloc
        st.success("Allocation optimisation complete.")

    alloc: AllocationResult | None = st.session_state.get(alloc_key)
    if alloc is None:
        st.info("Click **Optimise Allocation** above to generate recommendations.")
        return

    # ══════════════════════════════════════════════════════════════════
    # Section 3: Optimal Allocation Profile
    # ══════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("3. Optimal Allocation Profile")
    st.caption(
        "Stacked area chart showing the recommended MW to commit to wholesale "
        "vs. balancing markets per settlement period."
    )

    sp_labels = SP_LABELS[:NUM_SETTLEMENT_PERIODS]
    wholesale_mw = [a.wholesale_mw for a in alloc.sp_allocations]
    balancing_mw = [a.balancing_mw for a in alloc.sp_allocations]
    expected_del = [a.expected_delivered for a in alloc.sp_allocations]

    fig_alloc = go.Figure()
    fig_alloc.add_trace(go.Scatter(
        x=sp_labels, y=wholesale_mw,
        mode="lines", name="Wholesale (MIP)",
        line=dict(width=0), fillcolor="rgba(52, 152, 219, 0.5)",
        stackgroup="alloc",
    ))
    fig_alloc.add_trace(go.Scatter(
        x=sp_labels, y=balancing_mw,
        mode="lines", name="Balancing (SIP)",
        line=dict(width=0), fillcolor="rgba(46, 204, 113, 0.5)",
        stackgroup="alloc",
    ))
    fig_alloc.add_trace(go.Scatter(
        x=sp_labels, y=expected_del,
        mode="lines", name="Expected Delivery (P50)",
        line=dict(color="white", width=2, dash="dash"),
    ))
    fig_alloc.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Wholesale vs. Balancing Allocation by Settlement Period",
        xaxis_title="Settlement Period",
        yaxis_title="MW",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_alloc, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════
    # Section 4: Overbook / Underbook Recommendation
    # ══════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("4. Overbook / Underbook Recommendation")

    strategies = [a.strategy for a in alloc.sp_allocations]
    n_over = strategies.count("overbook")
    n_under = strategies.count("underbook")
    n_match = strategies.count("match")

    col1, col2, col3, col4 = st.columns(4)
    total_committed = sum(a.total_committed for a in alloc.sp_allocations)
    total_expected = sum(a.expected_delivered for a in alloc.sp_allocations)
    portfolio_ob = total_committed / total_expected if total_expected > 0 else 1.0

    col1.metric("Portfolio Overbook Ratio", f"{portfolio_ob:.2%}")
    col2.metric("Overbook SPs", f"{n_over} / {NUM_SETTLEMENT_PERIODS}")
    col3.metric("Match SPs", f"{n_match} / {NUM_SETTLEMENT_PERIODS}")
    col4.metric("Underbook SPs", f"{n_under} / {NUM_SETTLEMENT_PERIODS}")

    ob_colour_map = {
        "overbook": COLOUR_DANGER,
        "match": COLOUR_WARNING,
        "underbook": COLOUR_SUCCESS,
    }

    fig_ob = go.Figure()
    fig_ob.add_trace(go.Bar(
        x=sp_labels,
        y=[a.overbook_ratio for a in alloc.sp_allocations],
        marker_color=[ob_colour_map.get(a.strategy, COLOUR_MUTED) for a in alloc.sp_allocations],
        name="Overbook Ratio",
        hovertext=[
            f"SP {a.sp+1}: {a.strategy}<br>"
            f"Wholesale: {a.wholesale_mw:.1f} MW<br>"
            f"Balancing: {a.balancing_mw:.1f} MW<br>"
            f"Committed: {a.total_committed:.1f} MW<br>"
            f"Expected: {a.expected_delivered:.1f} MW"
            for a in alloc.sp_allocations
        ],
    ))
    fig_ob.add_hline(y=1.0, line_color="white", line_width=1, line_dash="dash",
                     annotation_text="Match")
    fig_ob.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Overbook Ratio per Settlement Period",
        xaxis_title="Settlement Period",
        yaxis_title="Overbook Ratio (>1 = overbook)",
        height=400,
    )
    st.plotly_chart(fig_ob, use_container_width=True)

    with st.expander("Detailed SP Allocation Table"):
        detail_rows = []
        for a in alloc.sp_allocations:
            detail_rows.append({
                "SP": sp_labels[a.sp],
                "Wholesale MW": f"{a.wholesale_mw:.2f}",
                "Balancing MW": f"{a.balancing_mw:.2f}",
                "Total Committed": f"{a.total_committed:.2f}",
                "Expected Delivered": f"{a.expected_delivered:.2f}",
                "Overbook Ratio": f"{a.overbook_ratio:.2%}",
                "Strategy": a.strategy,
                "E[Revenue] (£)": f"£{a.expected_revenue:.0f}",
                "ES Revenue (£)": f"£{a.es_revenue:.0f}",
            })
        st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════
    # Section 5: Strategy Comparison
    # ══════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("5. Strategy Comparison")
    st.caption(
        "Compares three strategies across the Monte Carlo delivery distribution: "
        "commit everything wholesale, commit everything to balancing, or use the "
        "optimised split."
    )

    strats = [alloc.pure_wholesale_strategy, alloc.pure_balancing_strategy, alloc.optimal_strategy]
    strat_colours = [COLOUR_PRIMARY, COLOUR_SUCCESS, COLOUR_WARNING]

    comp_rows = []
    for s in strats:
        comp_rows.append({
            "Strategy": s.name,
            "E[Daily P&L]": f"£{s.mean_pnl:,.0f}",
            "Median P&L": f"£{s.median_pnl:,.0f}",
            "Std P&L": f"£{s.std_pnl:,.0f}",
            "ES (5%)": f"£{s.es_5:,.0f}",
            "Max Loss": f"£{s.max_loss:,.0f}",
            "Max Gain": f"£{s.max_gain:,.0f}",
            "Reward/Risk": f"{s.reward_to_risk:.3f}",
        })
    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    fig_pnl = go.Figure()
    for s, colour in zip(strats, strat_colours):
        fig_pnl.add_trace(go.Histogram(
            x=s.daily_pnl,
            name=s.name,
            marker_color=colour,
            opacity=0.6,
            nbinsx=50,
        ))
    fig_pnl.update_layout(
        template=PLOTLY_TEMPLATE,
        barmode="overlay",
        title="Daily P&L Distribution by Strategy",
        xaxis_title="Daily P&L (£)",
        yaxis_title="Count",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_pnl, use_container_width=True)

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=[s.name for s in strats],
        y=[s.mean_pnl for s in strats],
        marker_color=strat_colours,
        text=[f"£{s.mean_pnl:,.0f}" for s in strats],
        textposition="auto",
    ))
    fig_bar.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Expected Daily Revenue by Strategy",
        yaxis_title="E[Daily P&L] (£)",
        height=400,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════
    # Section 6: Sensitivity — risk tolerance
    # ══════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("6. Risk Tolerance Sensitivity")
    st.caption(
        "Shows how the portfolio overbook ratio and expected daily revenue "
        "change as risk tolerance varies from 0 (risk-neutral) to 1 (risk-averse)."
    )

    sens_key = "_alloc_sensitivity"
    if st.button("📈 Run Sensitivity Sweep", key="alloc_sens_btn"):
        sip_fc = st.session_state.get("_alloc_sip_fc_48")
        mip_fc = st.session_state.get("_alloc_mip_fc_48")
        delivered = result.delivered_mw
        if sip_fc is not None and mip_fc is not None:
            with st.spinner("Running sensitivity sweep across risk tolerances…"):
                sweep_results = []
                for rt in np.arange(0.0, 1.05, 0.1):
                    a = optimize_allocation(sip_fc, mip_fc, delivered, risk_tolerance=float(rt))
                    total_c = sum(s.total_committed for s in a.sp_allocations)
                    total_e = sum(s.expected_delivered for s in a.sp_allocations)
                    sweep_results.append({
                        "risk_tolerance": float(rt),
                        "ob_ratio": total_c / total_e if total_e > 0 else 1.0,
                        "expected_rev": a.optimal_strategy.mean_pnl,
                        "es_5": a.optimal_strategy.es_5,
                    })
                st.session_state[sens_key] = sweep_results

    sweep = st.session_state.get(sens_key)
    if sweep:
        sweep_df = pd.DataFrame(sweep)
        from plotly.subplots import make_subplots
        fig_sens = make_subplots(specs=[[{"secondary_y": True}]])
        fig_sens.add_trace(go.Scatter(
            x=sweep_df["risk_tolerance"], y=sweep_df["ob_ratio"],
            mode="lines+markers", name="Overbook Ratio",
            line=dict(color=COLOUR_PRIMARY, width=2),
        ), secondary_y=False)
        fig_sens.add_trace(go.Scatter(
            x=sweep_df["risk_tolerance"], y=sweep_df["expected_rev"],
            mode="lines+markers", name="E[Daily Revenue]",
            line=dict(color=COLOUR_SUCCESS, width=2),
        ), secondary_y=True)
        fig_sens.add_trace(go.Scatter(
            x=sweep_df["risk_tolerance"], y=sweep_df["es_5"],
            mode="lines+markers", name="ES (5%)",
            line=dict(color=COLOUR_DANGER, width=2, dash="dash"),
        ), secondary_y=True)
        fig_sens.update_layout(
            template=PLOTLY_TEMPLATE,
            title="Allocation Sensitivity to Risk Tolerance",
            xaxis_title="Risk Tolerance",
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig_sens.update_yaxes(title_text="Overbook Ratio", secondary_y=False)
        fig_sens.update_yaxes(title_text="Revenue (£)", secondary_y=True)
        st.plotly_chart(fig_sens, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════
    # Section: Multi-Profile Risk Comparison
    # ══════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("Multi-Profile Risk Comparison")
    st.caption(
        "Runs Conservative · Moderate · Aggressive · Full Risk profiles in parallel. "
        "Each profile combines a different MC availability percentile (position size) "
        "with a different risk tolerance (wholesale vs balancing split)."
    )

    _PROFILE_COLOURS = {
        "Conservative": COLOUR_SUCCESS,
        "Moderate":     COLOUR_PRIMARY,
        "Aggressive":   COLOUR_WARNING,
        "Full Risk":    COLOUR_DANGER,
    }

    cols_prof = st.columns(4)
    for i, (pname, p) in enumerate(RISK_PROFILES.items()):
        with cols_prof[i]:
            st.markdown(f"**{pname}**")
            st.caption(f"P{p.percentile} · RT={p.risk_tolerance:.2f}")
            st.caption(p.description)

    mp_btn = st.button(
        "⚡ Compute All Risk Profiles",
        use_container_width=True, type="primary", key="alloc_multi_run",
        help="Optimises allocation for all 4 risk profiles simultaneously.",
    )

    if mp_btn:
        _sip_fc = st.session_state.get("_alloc_sip_fc_48")
        _mip_fc = st.session_state.get("_alloc_mip_fc_48")
        if _sip_fc is None or _mip_fc is None:
            st.error(
                "Forecasts not yet available. "
                "Scroll up and run **Optimise Allocation** once to generate them."
            )
        else:
            with st.spinner("Optimising 4 risk profiles in parallel…"):
                mp = optimize_all_risk_profiles(
                    sip_forecasts=_sip_fc,
                    mip_forecasts=_mip_fc,
                    delivered_mw=delivered_mw,
                )
            st.session_state[MULTI_PROFILE_RESULTS] = mp
            st.success("Multi-profile optimisation complete.")

    mp: MultiProfileResult | None = st.session_state.get(MULTI_PROFILE_RESULTS)
    if mp is not None:
        st.dataframe(mp.comparison_df, use_container_width=True)

        col_left, col_right = st.columns(2)

        with col_left:
            fig_pnl = go.Figure()
            for pname, palloc in mp.profile_results.items():
                fig_pnl.add_trace(go.Histogram(
                    x=palloc.optimal_strategy.daily_pnl,
                    name=pname,
                    marker_color=_PROFILE_COLOURS.get(pname, COLOUR_MUTED),
                    opacity=0.55, nbinsx=50,
                ))
            fig_pnl.update_layout(
                template=PLOTLY_TEMPLATE, barmode="overlay",
                title="P&L Distribution by Risk Profile",
                xaxis_title="Daily P&L (£)", yaxis_title="Count", height=400,
            )
            st.plotly_chart(fig_pnl, use_container_width=True)

        with col_right:
            fig_rr = go.Figure()
            for pname, palloc in mp.profile_results.items():
                opt = palloc.optimal_strategy
                fig_rr.add_trace(go.Scatter(
                    x=[opt.es_5], y=[opt.mean_pnl],
                    mode="markers+text", name=pname,
                    text=[pname], textposition="top center",
                    marker=dict(size=14,
                                color=_PROFILE_COLOURS.get(pname, COLOUR_MUTED)),
                ))
            fig_rr.update_layout(
                template=PLOTLY_TEMPLATE,
                title="Risk-Return Frontier",
                xaxis_title="ES 5% (£)", yaxis_title="E[Daily P&L] (£)", height=400,
            )
            st.plotly_chart(fig_rr, use_container_width=True)

        fig_pos = go.Figure()
        for pname, pos_mw in mp.profile_positions.items():
            fig_pos.add_trace(go.Scatter(
                x=SP_LABELS[:NUM_SETTLEMENT_PERIODS], y=pos_mw,
                mode="lines", name=pname,
                line=dict(color=_PROFILE_COLOURS.get(pname, COLOUR_MUTED), width=2),
            ))
        fig_pos.update_layout(
            template=PLOTLY_TEMPLATE,
            title="Traded Position by Risk Profile (MW per Settlement Period)",
            xaxis_title="Settlement Period", yaxis_title="MW", height=380,
        )
        st.plotly_chart(fig_pos, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════
    # Methodology
    # ══════════════════════════════════════════════════════════════════
    with st.expander("Methodology & Assumptions"):
        st.markdown("""
**What this tool does:**
- For each of the 48 settlement periods, the optimizer decides how many MW to
  commit to the **wholesale market** (sold at MIP) vs. the **balancing mechanism**
  (valued at SIP when dispatched).
- The optimizer uses the Monte Carlo fleet delivery distribution to account for
  delivery uncertainty — shortfalls on wholesale commitments incur imbalance costs.

**Revenue model per SP (per MC run):**
- `wholesale_rev = W × 0.5h × MIP_forecast`
- `bm_dispatched = min(B, max(0, delivered − W))`
- `bm_rev = bm_dispatched × 0.5h × SIP_forecast`
- `shortfall = max(0, W − delivered)`
- `shortfall_cost = shortfall × 0.5h × |SIP_forecast|`
- `net = wholesale_rev + bm_rev − shortfall_cost`

**Optimisation:**
- Grid search over wholesale commitment W from 0 to P99 of delivered MW.
- Balancing reservation B = max(0, P50_delivered − W).
- Score = (1−λ) × E[revenue] + λ × ES₅(revenue), where λ is risk tolerance.
- λ=0 maximises expected revenue (risk-neutral); λ=1 maximises tail performance.

**Overbook / Underbook:**
- If total committed > expected delivery → **overbook** (aggressive, higher revenue,
  higher shortfall risk).
- If total committed < expected delivery → **underbook** (conservative, lower revenue,
  lower shortfall risk).
- The optimal ratio depends on the SIP/MIP spread and delivery uncertainty.

**Key assumption:**
- MIP (Market Index Price) is used as the wholesale price proxy. True DA auction
  prices (EPEX/N2EX) are not freely available on the ELEXON no-key API.
""")


def _ensure_model(target: str, label: str, engine: str = "XGBoost"):
    """Load model from session or disk for the given engine."""
    if engine == "NeuralProphet":
        ss_key = f"_np_trained_models_{target}"
        trained = st.session_state.get(ss_key)
        if trained is None:
            trained = load_np_models(target=target)
            if trained is not None:
                st.session_state[ss_key] = trained
        if trained is not None and trained.forward_forecasts:
            ts = dt.datetime.fromtimestamp(trained.training_timestamp)
            st.caption(f"{label} model: NeuralProphet — trained **{ts:%Y-%m-%d %H:%M}**")
        else:
            st.caption(f"{label} model: NeuralProphet — not yet trained")
            trained = None
        return trained
    elif engine == "LSTM":
        ss_key = f"_lstm_trained_models_{target}"
        trained = st.session_state.get(ss_key)
        if trained is None:
            trained = load_trained_lstm_models(target=target)
            if trained is not None:
                st.session_state[ss_key] = trained
        if trained is not None and trained.final_state_dicts:
            ts = dt.datetime.fromtimestamp(trained.training_timestamp)
            st.caption(f"{label} model: LSTM — trained **{ts:%Y-%m-%d %H:%M}**")
        else:
            st.caption(f"{label} model: LSTM — not yet trained")
            trained = None
        return trained
    elif engine == "Hybrid LSTM+XGBoost":
        # Return a tuple (xgb, lstm); caller handles hybrid dispatch
        xgb_ss  = f"_xgb_trained_models_{target}"
        lstm_ss = f"_lstm_trained_models_{target}"
        xgb_m  = st.session_state.get(xgb_ss)  or load_trained_models(target=target)
        lstm_m = st.session_state.get(lstm_ss) or load_trained_lstm_models(target=target)
        if xgb_m is not None:
            st.session_state[xgb_ss] = xgb_m
        if lstm_m is not None:
            st.session_state[lstm_ss] = lstm_m
        if xgb_m is not None and lstm_m is not None:
            ts_x = dt.datetime.fromtimestamp(xgb_m.training_timestamp)
            ts_l = dt.datetime.fromtimestamp(lstm_m.training_timestamp)
            st.caption(f"{label} model: Hybrid (XGBoost {ts_x:%H:%M} · LSTM {ts_l:%H:%M})")
        elif xgb_m is not None:
            st.caption(f"{label} model: Hybrid fallback → XGBoost only (no LSTM found)")
        elif lstm_m is not None:
            st.caption(f"{label} model: Hybrid fallback → LSTM only (no XGBoost found)")
        else:
            st.caption(f"{label} model: Hybrid — neither model found")
        return (xgb_m, lstm_m)  # tuple sentinel for hybrid
    else:  # XGBoost
        ss_key = f"_xgb_trained_models_{target}"
        trained = st.session_state.get(ss_key)
        if trained is None:
            trained = load_trained_models(target=target)
            if trained is not None:
                st.session_state[ss_key] = trained
        if trained is not None and trained.final_models:
            ts = dt.datetime.fromtimestamp(trained.training_timestamp)
            st.caption(f"{label} model: XGBoost — trained **{ts:%Y-%m-%d %H:%M}**")
        else:
            st.caption(f"{label} model: XGBoost — not yet trained")
            trained = None
        return trained


def _build_48sp_forecast_legacy(
    trained, sip_values, mip_values, demand_values, target_label: str,
) -> np.ndarray | None:
    """
    Legacy scalar×shape fallback — used only for NeuralProphet models.
    Takes the day-1 forward forecast scalar and scales the last-day intraday shape.
    XGBoost models use forecast_intraday_48sp() instead for true per-SP predictions.
    """
    if trained is None:
        return None

    if isinstance(trained, TrainedNPModels):
        forecasts = forecast_forward_np(trained)
    else:
        forecasts = forecast_forward(
            trained, sip_values, mip_values, demand_values, n_days=1,
        )
    if not forecasts:
        return None

    best_lb = max(forecasts.keys(), key=lambda k: len(forecasts[k]))
    day_fc = forecasts[best_lb]
    if 1 not in day_fc:
        return None

    fc_value = day_fc[1]
    fc_48 = np.full(NUM_SETTLEMENT_PERIODS, fc_value)

    target_key = "sip" if target_label == "SIP" else ("mip" if target_label == "MIP" else "demand")
    target = getattr(trained, "target", target_key)

    if target == "sip":
        last_day_shape = sip_values[-48:]
    elif target == "mip":
        last_day_shape = mip_values[-48:]
    elif target == "demand":
        last_day_shape = demand_values[-48:] if demand_values is not None else None
    else:
        last_day_shape = None

    if last_day_shape is not None and len(last_day_shape) == 48:
        daily_mean = np.mean(last_day_shape)
        if abs(daily_mean) > 1e-6:
            shape = last_day_shape / daily_mean
            fc_48 = fc_value * shape

    ss_key = f"_alloc_{target_label.lower()}_fc_48"
    st.session_state[ss_key] = fc_48
    return fc_48


def _render_forecast_summary(
    sip_fc: np.ndarray,
    mip_fc: np.ndarray,
    trained_sip,
    trained_mip,
) -> None:
    """Show the 48-SP forecast inputs used for optimisation."""
    sp_labels = SP_LABELS[:NUM_SETTLEMENT_PERIODS]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sp_labels, y=sip_fc,
        mode="lines", name="SIP Forecast",
        line=dict(color=COLOUR_PRIMARY, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=sp_labels, y=mip_fc,
        mode="lines", name="MIP Forecast",
        line=dict(color=COLOUR_WARNING, width=2),
    ))

    spread = sip_fc - mip_fc
    fig.add_trace(go.Bar(
        x=sp_labels, y=spread,
        name="SIP − MIP Spread",
        marker_color=[COLOUR_SUCCESS if s > 0 else COLOUR_DANGER for s in spread],
        opacity=0.3,
    ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Forecast Inputs: SIP vs. MIP per Settlement Period",
        xaxis_title="Settlement Period",
        yaxis_title="£/MWh",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean SIP Forecast", f"£{np.mean(sip_fc):.2f}/MWh")
    col2.metric("Mean MIP Forecast", f"£{np.mean(mip_fc):.2f}/MWh")
    mean_spread = float(np.mean(spread))
    col3.metric(
        "Mean Spread (SIP − MIP)",
        f"£{mean_spread:.2f}/MWh",
        delta=f"{'Favour BM' if mean_spread > 0 else 'Favour Wholesale'}",
        delta_color="normal" if mean_spread > 0 else "inverse",
    )
