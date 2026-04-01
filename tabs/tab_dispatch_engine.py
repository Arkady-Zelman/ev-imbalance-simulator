"""
Dispatch Decision Engine tab.

Combines GB system supply (generation by fuel), demand, SIP, and MIP forecasts
to recommend, for each future settlement period, whether the Ohme fleet should:

  HOLD        — meet gate-closure commitment exactly
  CHARGE_MORE — absorb excess supply (long system, low SSP)
  CHARGE_LESS — provide demand flexibility (short system, high SBP premium)

Lookahead modes:
  1 Day  — 48 SPs, per-SP detail (operational gate-closure decisions)
  10 Day — 480 SPs aggregated to daily stance (strategic planning)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.config import (
    CHARGER_CAPACITY_KW,
    COLOUR_DANGER,
    COLOUR_MUTED,
    COLOUR_PRIMARY,
    COLOUR_SUCCESS,
    COLOUR_WARNING,
    PLOTLY_TEMPLATE,
)
from src.data.elexon_client import fetch_generation_outturn, gen_cache_timestamp
from src.models.dispatch_engine import (
    NIV_THRESHOLD_MW,
    SBP_PREMIUM_MIN,
    DailyDispatchSummary,
    DispatchRecommendation,
    build_ssp_series,
    compute_dispatch_recommendations,
    process_generation_outturn,
)
from src.models.forecaster import build_aligned_series
from src.session_keys import DA_PRICE, DATE_FROM, DATE_TO, DEMAND_DF, GEN_DF, MIP_DF, PARAMS, SIP_DF

_ACTION_COLOURS = {
    "HOLD":         COLOUR_MUTED,
    "CHARGE_MORE":  COLOUR_PRIMARY,
    "CHARGE_LESS":  COLOUR_WARNING,
}
_STANCE_EMOJI = {
    "LONG":     "🟦",
    "SHORT":    "🟧",
    "BALANCED": "⬜",
}
_ACTION_EMOJI = {
    "HOLD":         "⏸ HOLD",
    "CHARGE_MORE":  "⬆ CHARGE MORE",
    "CHARGE_LESS":  "⬇ CHARGE LESS",
}


def render(has_results: bool) -> None:
    st.header("Dispatch Decision Engine")
    st.caption(
        "Forecasts GB system supply (wind / thermal / interconnectors), demand, "
        "SIP, and MIP over a 1-day or 10-day horizon, then recommends whether to "
        "hold gate-closure commitment, increase, or reduce charging."
    )

    if not has_results:
        st.info("Run a simulation first (fetches ELEXON data) to use this tool.")
        return

    for key in (SIP_DF, MIP_DF):
        if key not in st.session_state or st.session_state[key].empty:
            st.warning("SIP or MIP data missing. Run the simulation first.")
            return

    _render_dispatch_body()


@st.fragment
def _render_dispatch_body() -> None:
    sip_df    = st.session_state[SIP_DF]
    mip_df    = st.session_state[MIP_DF]
    demand_df = st.session_state.get(DEMAND_DF)
    params    = st.session_state.get(PARAMS)
    date_from = st.session_state.get(DATE_FROM)
    date_to   = st.session_state.get(DATE_TO)
    da_price  = st.session_state.get(DA_PRICE, 90.0)

    # ── Generation data ────────────────────────────────────────────────
    gen_df = st.session_state.get(GEN_DF)
    if gen_df is None or gen_df.empty:
        if date_from is not None and date_to is not None:
            with st.spinner("Fetching generation outturn (FUELHH) from ELEXON…"):
                gen_df = fetch_generation_outturn(date_from, date_to)
                st.session_state[GEN_DF] = gen_df
        if gen_df is None or gen_df.empty:
            st.warning(
                "No generation data available. The FUELHH dataset could not be fetched "
                "for your date range — check your network or widen the date range."
            )
            return

    gen_ts = gen_cache_timestamp(date_from, date_to) if date_from else None
    if gen_ts:
        import datetime as dt
        age = (dt.datetime.now() - dt.datetime.fromtimestamp(gen_ts)).total_seconds() / 3600
        st.caption(
            f"Generation data fetched at **{dt.datetime.fromtimestamp(gen_ts):%Y-%m-%d %H:%M}** "
            f"({'fresh' if age < 2 else f'{age:.0f}h old'}). "
            f"{len(gen_df):,} fuel-type records."
        )

    # ── Align series ───────────────────────────────────────────────────
    sip_series, mip_series, demand_series, _ = build_aligned_series(
        sip_df, mip_df, demand_df=demand_df,
    )
    if demand_series is None:
        st.warning("Demand data is required for NIV calculation. Fetch demand in the simulation first.")
        return

    ssp_series = build_ssp_series(sip_df, sip_series.index)
    gen_breakdown = process_generation_outturn(gen_df)

    if len(gen_breakdown.index) == 0:
        st.warning("Could not parse generation data into half-hourly series.")
        return

    # ── Configuration row ──────────────────────────────────────────────
    st.markdown("---")
    col_a, col_b, col_c, col_d = st.columns([2, 2, 2, 2])

    with col_a:
        lookahead_days = st.radio(
            "Lookahead Horizon",
            options=[1, 10],
            format_func=lambda d: f"1 Day (48 SP, per-period detail)" if d == 1
                                  else f"10 Days (daily rollup, strategic)",
            index=0,
            horizontal=False,
        )

    with col_b:
        max_mw = (params.fleet_size * CHARGER_CAPACITY_KW / 1_000) if params else 100.0
        committed_mw = st.number_input(
            "Committed MW (gate-closure)",
            min_value=0.0, max_value=float(max_mw),
            value=min(float(max_mw * 0.8), max_mw),
            step=1.0,
            help="MW committed at gate closure. Used to size headroom for CHARGE_MORE / CHARGE_LESS.",
        )

    with col_c:
        fleet_max_mw = st.number_input(
            "Fleet Max Capacity (MW)",
            min_value=1.0, max_value=float(max_mw * 1.5),
            value=float(max_mw),
            step=1.0,
            help="Maximum MW the fleet can physically charge at. Caps CHARGE_MORE recommendations.",
        )

    with col_d:
        lookback_days = st.selectbox(
            "Lookback Window (TOD-mean)",
            options=[7, 14, 30],
            index=0,
            format_func=lambda d: f"{d} days",
            help="Historical window used for Time-of-Day seasonal mean forecasting.",
        )

    lookback_sps = lookback_days * 48

    # ── Run engine ─────────────────────────────────────────────────────
    with st.spinner(f"Computing {lookahead_days}-day dispatch recommendations…"):
        sp_recs, daily_summaries = compute_dispatch_recommendations(
            sip_series=sip_series,
            ssp_series=ssp_series,
            mip_series=mip_series,
            gen_breakdown=gen_breakdown,
            demand_series=demand_series,
            committed_mw=committed_mw,
            fleet_max_mw=fleet_max_mw,
            lookahead_days=lookahead_days,
            lookback_sps=lookback_sps,
        )

    if not sp_recs:
        st.error("No recommendations produced. Check that there is sufficient historical data.")
        return

    # ── KPI summary row ────────────────────────────────────────────────
    st.markdown("---")
    _render_kpi_row(sp_recs, daily_summaries, committed_mw, lookahead_days)

    # ── Section 1: Generation decomposition ───────────────────────────
    st.markdown("---")
    st.subheader("1. GB Generation Decomposition")
    _render_generation_chart(
        gen_breakdown, sip_series, demand_series, sp_recs, lookahead_days
    )

    # ── Section 2: NIV forecast & system position ──────────────────────
    st.markdown("---")
    st.subheader("2. Net Imbalance Volume (NIV) Forecast")
    _render_niv_chart(sip_series, gen_breakdown, demand_series, sp_recs)

    # ── Section 3: Dispatch decision timeline ─────────────────────────
    st.markdown("---")
    st.subheader("3. Dispatch Recommendation Timeline")
    if lookahead_days == 1:
        _render_sp_decision_chart(sp_recs)
        _render_sp_table(sp_recs)
    else:
        _render_daily_heatmap(daily_summaries)
        _render_daily_table(daily_summaries)

    # ── Section 4: P&L opportunity ────────────────────────────────────
    st.markdown("---")
    st.subheader("4. P&L Opportunity vs HOLD")
    _render_pnl_chart(sp_recs, daily_summaries, committed_mw, lookahead_days)

    # ── Methodology note ──────────────────────────────────────────────
    with st.expander("Methodology & Assumptions"):
        st.markdown(f"""
**NIV forecast** = Total Generation forecast − Demand forecast (TOD-mean of {lookback_days}-day window).

**Decision thresholds (configurable in `dispatch_engine.py`):**
- NIV threshold: **±{NIV_THRESHOLD_MW:,.0f} MW** — magnitude below which the system is considered balanced.
- SBP premium floor: **£{SBP_PREMIUM_MIN:.0f}/MWh** — minimum SBP−MIP spread for CHARGE_LESS to be worthwhile
  (covers DA revenue given up and forecast uncertainty).

**P&L delta per MW:**
- **CHARGE_MORE**: SSP × 0.5h — earn settlement price for surplus consumption above commitment.
- **CHARGE_LESS**: (SBP − MIP) × 0.5h — earn the imbalance premium over DA opportunity cost.

**Limitations:**
- Forecasts use TOD-mean seasonal averages. XGBoost models (train via Rolling Backtest tab) will
  improve accuracy when available.
- SBP and SSP are approximated from historical SIP. In GB single cash-out, SBP ≈ SSP ≈ SIP for most
  settlement periods; they diverge significantly during system stress events.
- Interconnector flows in FUELHH are net imports (positive = importing, negative = exporting).
  Embedded solar and small-scale wind are **not** included in FUELHH.
""")


# ── Chart helpers ──────────────────────────────────────────────────────────────

def _render_kpi_row(
    sp_recs: list[DispatchRecommendation],
    daily_summaries: list[DailyDispatchSummary],
    committed_mw: float,
    lookahead_days: int,
) -> None:
    actions = [r.action for r in sp_recs]
    n_long    = actions.count("CHARGE_MORE")
    n_short   = actions.count("CHARGE_LESS")
    n_hold    = actions.count("HOLD")
    total_pnl = sum(r.pnl_delta_per_mw for r in sp_recs) * committed_mw

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("CHARGE_MORE SPs", f"{n_long}",
                  help="SPs where increasing charging is recommended (long system).")
    with c2:
        st.metric("CHARGE_LESS SPs", f"{n_short}",
                  help="SPs where reducing charging is recommended (short system).")
    with c3:
        st.metric("HOLD SPs", f"{n_hold}")
    with c4:
        st.metric(
            "Total P&L Opportunity",
            f"£{total_pnl:,.0f}",
            help=f"Total £ gain vs HOLD if all recommendations followed "
                 f"at committed position ({committed_mw:.0f} MW).",
        )


def _render_generation_chart(
    gen_breakdown,
    sip_series: pd.Series,
    demand_series: pd.Series,
    sp_recs: list[DispatchRecommendation],
    lookahead_days: int,
) -> None:
    # Show last 7 days historical + forecast window
    lookback_n = min(7 * 48, len(gen_breakdown.index))
    hist_idx   = gen_breakdown.index[-lookback_n:]

    fig = go.Figure()

    # Historical stacked area
    for label, values, colour in [
        ("Wind",           gen_breakdown.wind_mw[-lookback_n:],          "#4CAF50"),
        ("Thermal",        gen_breakdown.thermal_mw[-lookback_n:],       "#FF9800"),
        ("Interconnector", gen_breakdown.interconnector_mw[-lookback_n:], "#9C27B0"),
        ("Storage",        gen_breakdown.storage_mw[-lookback_n:],       "#2196F3"),
    ]:
        fig.add_trace(go.Scatter(
            x=hist_idx, y=values,
            name=label, stackgroup="hist",
            line=dict(width=0), fillcolor=colour,
            mode="lines",
        ))

    # Historical demand line
    dem_hist = demand_series.reindex(hist_idx).ffill()
    fig.add_trace(go.Scatter(
        x=hist_idx, y=dem_hist.values,
        name="Demand (actual)", line=dict(color="white", width=2, dash="dash"),
        mode="lines",
    ))

    # Forecast generation (total line)
    fcast_dts  = [r.forecast_datetime for r in sp_recs]
    fcast_wind = [r.wind_forecast_mw  for r in sp_recs]
    fcast_thm  = [r.thermal_forecast_mw for r in sp_recs]
    fcast_itn  = [r.interconnector_forecast_mw for r in sp_recs]
    fcast_gen  = [r.total_gen_forecast_mw for r in sp_recs]
    fcast_dem  = [r.demand_forecast_mw for r in sp_recs]

    fig.add_trace(go.Scatter(
        x=fcast_dts, y=fcast_gen,
        name="Gen Forecast (total)", line=dict(color="lime", width=2, dash="dot"),
        mode="lines",
    ))
    fig.add_trace(go.Scatter(
        x=fcast_dts, y=fcast_dem,
        name="Demand Forecast", line=dict(color="white", width=2, dash="dot"),
        mode="lines",
    ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=350,
        margin=dict(l=40, r=20, t=10, b=30),
        yaxis_title="MW",
        legend=dict(orientation="h", y=-0.2),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_niv_chart(
    sip_series: pd.Series,
    gen_breakdown,
    demand_series: pd.Series,
    sp_recs: list[DispatchRecommendation],
) -> None:
    lookback_n = min(7 * 48, len(gen_breakdown.index))
    hist_idx   = gen_breakdown.index[-lookback_n:]
    dem_hist   = demand_series.reindex(hist_idx).ffill()
    niv_hist   = gen_breakdown.total_mw[-lookback_n:] - dem_hist.values

    fcast_dts = [r.forecast_datetime for r in sp_recs]
    fcast_niv = [r.niv_forecast_mw   for r in sp_recs]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_idx, y=niv_hist,
        name="NIV (actual)", fill="tozeroy",
        line=dict(color=COLOUR_MUTED, width=1),
        fillcolor="rgba(150,150,150,0.3)",
    ))
    fig.add_trace(go.Scatter(
        x=fcast_dts, y=fcast_niv,
        name="NIV Forecast",
        line=dict(color=COLOUR_PRIMARY, width=2, dash="dot"),
        fill="tozeroy",
        fillcolor="rgba(33,150,243,0.2)",
    ))
    fig.add_hline(y=NIV_THRESHOLD_MW,  line=dict(color="lime",   dash="dash", width=1),
                  annotation_text="Long threshold")
    fig.add_hline(y=-NIV_THRESHOLD_MW, line=dict(color=COLOUR_WARNING, dash="dash", width=1),
                  annotation_text="Short threshold")
    fig.add_hline(y=0, line=dict(color="white", width=1, dash="solid"))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=280,
        margin=dict(l=40, r=20, t=10, b=30),
        yaxis_title="NIV (MW) — +ve = Long",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_sp_decision_chart(sp_recs: list[DispatchRecommendation]) -> None:
    dts    = [r.forecast_datetime for r in sp_recs]
    pnls   = [r.pnl_delta_per_mw  for r in sp_recs]
    cols   = [_ACTION_COLOURS[r.action] for r in sp_recs]
    labels = [_ACTION_EMOJI[r.action]   for r in sp_recs]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dts, y=pnls,
        marker_color=cols,
        text=None,
        hovertext=[f"{l}<br>NIV={r.niv_forecast_mw:+,.0f} MW<br>"
                   f"SBP=£{r.sbp_forecast:.1f} MIP=£{r.mip_forecast:.1f}<br>"
                   f"P&L/MW: £{r.pnl_delta_per_mw:.3f}"
                   for l, r in zip(labels, sp_recs)],
        hoverinfo="text",
    ))
    fig.add_hline(y=0, line=dict(color="white", width=1))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=260,
        margin=dict(l=40, r=20, t=10, b=30),
        yaxis_title="P&L delta (£/MW per SP)",
        bargap=0.05,
        hovermode="x",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_sp_table(sp_recs: list[DispatchRecommendation]) -> None:
    rows = []
    for r in sp_recs:
        rows.append({
            "Time":         r.forecast_datetime.strftime("%H:%M"),
            "System":       _STANCE_EMOJI.get(r.system_position, "") + " " + r.system_position,
            "NIV (MW)":     f"{r.niv_forecast_mw:+,.0f}",
            "SBP (£/MWh)":  f"{r.sbp_forecast:.1f}",
            "SSP (£/MWh)":  f"{r.ssp_forecast:.1f}",
            "MIP (£/MWh)":  f"{r.mip_forecast:.1f}",
            "Wind (MW)":    f"{r.wind_forecast_mw:,.0f}",
            "Action":       _ACTION_EMOJI[r.action],
            "P&L/MW (£)":   f"{r.pnl_delta_per_mw:.3f}",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True, height=400)


def _render_daily_heatmap(daily_summaries: list[DailyDispatchSummary]) -> None:
    dates   = [d.date              for d in daily_summaries]
    nivsann = [d.mean_niv_mw       for d in daily_summaries]
    pnls    = [d.pnl_opportunity_per_mw_day for d in daily_summaries]
    stances = [d.system_stance     for d in daily_summaries]
    cols    = [_ACTION_COLOURS.get(d.dominant_action, COLOUR_MUTED) for d in daily_summaries]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.05)

    fig.add_trace(go.Bar(
        x=dates, y=nivsann, marker_color=cols,
        name="Mean NIV (MW)",
        hovertext=[
            f"{_STANCE_EMOJI[s]} {s} | Wind {d.wind_pct:.0f}%<br>"
            f"NIV={n:+,.0f} MW | SBP=£{d.mean_sbp:.1f} MIP=£{d.mean_mip:.1f}<br>"
            f"P&L opp £{p:,.0f}/MW/day | Strong SPs: {d.n_strong_signal_sps}/48"
            for s, n, p, d in zip(stances, nivsann, pnls, daily_summaries)
        ],
        hoverinfo="text",
    ), row=1, col=1)
    fig.add_hline(y=NIV_THRESHOLD_MW,  row=1, col=1,
                  line=dict(color="lime", dash="dash", width=1))
    fig.add_hline(y=-NIV_THRESHOLD_MW, row=1, col=1,
                  line=dict(color=COLOUR_WARNING, dash="dash", width=1))
    fig.add_hline(y=0, row=1, col=1, line=dict(color="white", width=1))

    fig.add_trace(go.Bar(
        x=dates, y=pnls, marker_color=cols,
        name="P&L opp (£/MW/day)",
    ), row=2, col=1)

    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=420,
        margin=dict(l=40, r=20, t=10, b=30),
        showlegend=False, hovermode="x unified",
    )
    fig.update_yaxes(title_text="Mean NIV (MW)", row=1)
    fig.update_yaxes(title_text="P&L/MW/day (£)", row=2)
    st.plotly_chart(fig, use_container_width=True)


def _render_daily_table(daily_summaries: list[DailyDispatchSummary]) -> None:
    rows = []
    for d in daily_summaries:
        rows.append({
            "Date":              d.date,
            "Stance":            _STANCE_EMOJI.get(d.system_stance, "") + " " + d.system_stance,
            "Mean NIV (MW)":     f"{d.mean_niv_mw:+,.0f}",
            "Wind %":            f"{d.wind_pct:.0f}%",
            "Thermal %":         f"{d.thermal_pct:.0f}%",
            "Interconnector %":  f"{d.interconnector_pct:.0f}%",
            "Mean SBP (£/MWh)":  f"{d.mean_sbp:.1f}",
            "Mean MIP (£/MWh)":  f"{d.mean_mip:.1f}",
            "Action":            _ACTION_EMOJI[d.dominant_action],
            "P&L opp £/MW/day":  f"£{d.pnl_opportunity_per_mw_day:,.1f}",
            "Strong Signal SPs": f"{d.n_strong_signal_sps}/48",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_pnl_chart(
    sp_recs: list[DispatchRecommendation],
    daily_summaries: list[DailyDispatchSummary],
    committed_mw: float,
    lookahead_days: int,
) -> None:
    if lookahead_days == 1:
        dts  = [r.forecast_datetime for r in sp_recs]
        cumulative = np.cumsum([r.pnl_delta_per_mw * committed_mw for r in sp_recs])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dts, y=cumulative, mode="lines+markers",
            line=dict(color=COLOUR_SUCCESS, width=2),
            name="Cumulative P&L vs HOLD",
            fill="tozeroy", fillcolor="rgba(76,175,80,0.2)",
        ))
    else:
        dates = [d.date for d in daily_summaries]
        daily_pnl = [d.pnl_opportunity_per_mw_day * committed_mw for d in daily_summaries]
        cumulative = np.cumsum(daily_pnl)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=dates, y=daily_pnl,
            marker_color=[COLOUR_PRIMARY if v > 0 else COLOUR_DANGER for v in daily_pnl],
            name="Daily P&L opportunity",
        ))
        fig.add_trace(go.Scatter(
            x=dates, y=cumulative, mode="lines+markers",
            line=dict(color=COLOUR_SUCCESS, width=2),
            name="Cumulative", yaxis="y2",
        ))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="Cumulative (£)"))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=280,
        margin=dict(l=40, r=20, t=10, b=30),
        yaxis_title=f"P&L vs HOLD (£) — at {committed_mw:.0f} MW committed",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Totals
    total = sum(r.pnl_delta_per_mw * committed_mw for r in sp_recs)
    n_active = sum(1 for r in sp_recs if r.action != "HOLD")
    st.caption(
        f"Total P&L opportunity vs HOLD: **£{total:,.0f}** over {lookahead_days} day(s) "
        f"at {committed_mw:.0f} MW committed — across **{n_active}** of {len(sp_recs)} SPs "
        f"where a non-HOLD action is recommended."
    )
