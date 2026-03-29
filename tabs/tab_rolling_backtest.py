"""
Rolling Forecast Backtest tab — market inefficiency detection.

Uses 1-day, 15-day, and 30-day lookbacks to forecast 1-14 days ahead,
comparing our forecast error against the forward curve at each horizon.
The crossover point marks the maximum exploitable forecast horizon.
"""

from __future__ import annotations

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
    PLOTLY_TEMPLATE,
)
from src.models.forecaster import build_aligned_series
from src.models.rolling_backtest import (
    ROLLING_HORIZON_LABELS,
    ROLLING_LOOKBACKS,
    build_error_matrix,
    run_rolling_backtest,
)
from src.session_keys import MIP_DF, SIP_DF


def render(has_results: bool) -> None:
    st.header("Rolling Forecast Backtest — Market Inefficiency Detection")
    st.caption(
        "Identifies the maximum forecast horizon where our model consistently "
        "beats the market (MIP forward curve). Uses daily non-overlapping origins "
        "for statistically valid inference."
    )

    if not has_results:
        st.info("Run a simulation first (which fetches ELEXON data) to use this tool.")
        return

    if SIP_DF not in st.session_state or st.session_state[SIP_DF].empty:
        st.warning("No SIP data available.")
        return
    if MIP_DF not in st.session_state or st.session_state[MIP_DF].empty:
        st.warning("No MIP data available.")
        return

    sip_df = st.session_state[SIP_DF]
    mip_df = st.session_state[MIP_DF]

    # ── Configuration ─────────────────────────────────────────────────
    st.subheader("Configuration")
    col1, col2 = st.columns(2)
    with col1:
        method = st.radio(
            "Forecast method",
            ["TOD Mean", "EWMA"],
            horizontal=True,
            key="rb_method",
        )
        method_key = "tod_mean" if method == "TOD Mean" else "ewma"
    with col2:
        ewma_alpha = st.slider(
            "EWMA α", 0.01, 0.30, 0.05, 0.01,
            key="rb_ewma_alpha",
            help="Only used when EWMA is selected.",
        )

    run_btn = st.button(
        "🔬 Run Rolling Backtest",
        use_container_width=True,
        type="primary",
        key="rb_run",
    )

    if run_btn:
        with st.spinner("Aligning SIP and MIP series…"):
            sip_series, mip_series = build_aligned_series(sip_df, mip_df)

        n_days = len(sip_series) // 48
        if n_days < 45:
            st.error(
                f"Need at least 45 days of aligned data for a meaningful rolling backtest. "
                f"Currently have {n_days} days. Increase the date range."
            )
            return

        st.caption(f"Data: {len(sip_series):,} SPs ({n_days} days) — "
                   f"{sip_series.index[0]} → {sip_series.index[-1]}")

        with st.spinner("Running rolling backtest (1/15/30-day lookbacks × 1-14 day horizons)…"):
            errors, crossovers = run_rolling_backtest(
                sip_series, mip_series,
                method=method_key,
                ewma_alpha=ewma_alpha,
            )

        st.session_state["_rb_errors"] = errors
        st.session_state["_rb_crossovers"] = crossovers
        st.success(f"Rolling backtest complete — {len(errors)} configurations evaluated.")

    # ── Results ───────────────────────────────────────────────────────
    if "_rb_errors" not in st.session_state:
        st.info("Configure and run the rolling backtest above.")
        return

    errors = st.session_state["_rb_errors"]
    crossovers = st.session_state["_rb_crossovers"]

    if not errors:
        st.warning("No results — insufficient data for any configuration.")
        return

    # ══════════════════════════════════════════════════════════════════
    # Section 1: Crossover Summary
    # ══════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("1. Maximum Exploitable Forecast Horizon")
    st.caption(
        "The crossover point is where our forecast error first exceeds the "
        "market's. Beyond this horizon, the forward curve is a better predictor."
    )

    cross_rows = []
    for c in crossovers:
        if c.crossover_day >= 15:
            verdict = "We beat the market at all tested horizons (1-14 days)"
        elif c.crossover_day <= 1:
            verdict = "Market wins immediately — no exploitable horizon"
        else:
            verdict = f"Alpha up to **{c.crossover_day - 1} days** ahead"
        cross_rows.append({
            "Lookback": c.lookback_label,
            "Crossover Day": c.crossover_day if c.crossover_day < 15 else ">14",
            "Last +Alpha": f"£{c.last_positive_alpha:+.2f}",
            "First −Alpha": f"£{c.first_negative_alpha:+.2f}" if c.crossover_day < 15 else "N/A",
            "Verdict": verdict,
        })
    st.dataframe(pd.DataFrame(cross_rows), use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════
    # Section 2: Alpha Heatmap (Lookback × Horizon)
    # ══════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("2. Alpha Heatmap — Lookback × Forecast Horizon")
    st.caption("Green = our forecast beats the market. Red = market wins.")

    error_matrix = build_error_matrix(errors)
    if not error_matrix.empty:
        z = error_matrix.values.astype(float)
        text = [[f"{v:+.2f}" if not np.isnan(v) else "N/A" for v in row] for row in z]

        fig_hm = go.Figure()
        fig_hm.add_trace(go.Heatmap(
            z=z,
            x=list(error_matrix.columns),
            y=list(error_matrix.index),
            colorscale=[
                [0.0, COLOUR_DANGER],
                [0.5, "#2C3E50"],
                [1.0, COLOUR_SUCCESS],
            ],
            zmid=0,
            text=text,
            texttemplate="%{text}",
            colorbar=dict(title="Alpha (£/MWh)"),
        ))
        fig_hm.update_layout(
            template=PLOTLY_TEMPLATE,
            margin=dict(l=50, r=30, t=50, b=50),
            title="Alpha: Lookback × Forecast Horizon (days)",
            xaxis_title="Forecast Horizon",
            yaxis_title="Lookback Window",
            height=350,
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════
    # Section 3: Error Curves — Each Lookback
    # ══════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("3. Forecast vs Market Error by Horizon")
    st.caption(
        "The point where the solid line crosses above the dashed line is "
        "where the market becomes a better predictor than our model."
    )

    lb_colours = {
        "1 day": COLOUR_PRIMARY,
        "15 days": COLOUR_WARNING,
        "30 days": COLOUR_SUCCESS,
    }

    fig_err = go.Figure()
    for lb_label in ROLLING_LOOKBACKS:
        lb_rows = sorted(
            [e for e in errors if e.lookback_label == lb_label],
            key=lambda e: e.horizon_days,
        )
        if not lb_rows:
            continue

        days = [e.horizon_days for e in lb_rows]
        fc_maes = [e.forecast_mae for e in lb_rows]
        mkt_maes = [e.market_mae for e in lb_rows]

        colour = lb_colours.get(lb_label, COLOUR_MUTED)

        fig_err.add_trace(go.Scatter(
            x=days, y=fc_maes,
            mode="lines+markers", name=f"Our Forecast ({lb_label})",
            line=dict(color=colour, width=2),
            marker=dict(size=6),
        ))
        fig_err.add_trace(go.Scatter(
            x=days, y=mkt_maes,
            mode="lines+markers", name=f"Market ({lb_label})",
            line=dict(color=colour, width=1, dash="dash"),
            marker=dict(size=4, symbol="x"),
        ))

    fig_err.update_layout(
        template=PLOTLY_TEMPLATE,
        margin=dict(l=50, r=30, t=50, b=50),
        title="MAE by Forecast Horizon (days ahead)",
        xaxis_title="Forecast Horizon (days)",
        yaxis_title="MAE (£/MWh)",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_err, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════
    # Section 4: Alpha Decay Curves
    # ══════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("4. Alpha Decay by Horizon")
    st.caption(
        "How our forecast advantage (alpha) decays with increasing horizon. "
        "Positive = we beat the market. Where it crosses zero = crossover."
    )

    fig_alpha = go.Figure()
    for lb_label in ROLLING_LOOKBACKS:
        lb_rows = sorted(
            [e for e in errors if e.lookback_label == lb_label],
            key=lambda e: e.horizon_days,
        )
        if not lb_rows:
            continue

        days = [e.horizon_days for e in lb_rows]
        alphas = [e.alpha_mae for e in lb_rows]
        colour = lb_colours.get(lb_label, COLOUR_MUTED)

        fig_alpha.add_trace(go.Scatter(
            x=days, y=alphas,
            mode="lines+markers", name=lb_label,
            line=dict(color=colour, width=2),
            marker=dict(size=7),
            fill="tozeroy",
            fillcolor=f"rgba({int(colour[1:3],16)},{int(colour[3:5],16)},{int(colour[5:7],16)},0.08)",
        ))

    fig_alpha.add_hline(y=0, line_color="white", line_width=1, line_dash="dash")
    fig_alpha.update_layout(
        template=PLOTLY_TEMPLATE,
        margin=dict(l=50, r=30, t=50, b=50),
        title="Alpha (Market MAE − Forecast MAE) by Horizon",
        xaxis_title="Forecast Horizon (days)",
        yaxis_title="Alpha (£/MWh)",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_alpha, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════
    # Section 5: Detailed Results Table
    # ══════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("5. Full Results")

    detail_rows = []
    for e in sorted(errors, key=lambda e: (e.lookback_label, e.horizon_days)):
        sig = "Yes" if e.dm_pvalue < 0.05 and e.alpha_mae > 0 else "No"
        detail_rows.append({
            "Lookback": e.lookback_label,
            "Horizon": f"{e.horizon_days}d",
            "Forecast MAE": f"£{e.forecast_mae:.2f}",
            "Market MAE": f"£{e.market_mae:.2f}",
            "Alpha": f"£{e.alpha_mae:+.2f}",
            "DM p-value": f"{e.dm_pvalue:.4f}",
            "Significant": sig,
            "N Obs": e.n_obs,
        })

    st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)

    csv_data = pd.DataFrame(detail_rows).to_csv(index=False)
    st.download_button(
        "📥 Export Rolling Backtest Results (CSV)",
        csv_data,
        file_name="rolling_backtest_results.csv",
        mime="text/csv",
    )

    with st.expander("Methodology & Interpretation"):
        st.markdown("""
**What this tool does:**
- For each of the three lookback windows (1, 15, 30 days), we make forecasts
  at horizons of 1 through 14 days ahead.
- All forecast origins are **daily** (step=48 SPs) so that observations are
  non-overlapping and statistical tests are valid.
- We compare our forecast error (MAE) against the market's forward curve error
  (MIP TOD-mean benchmark) at each horizon.

**The crossover point:**
- Where our forecast MAE first exceeds the market's MAE, our alpha disappears.
- Before the crossover, there is a **market inefficiency** we can exploit.
- After the crossover, we should defer to the forward curve.

**How to use this for trading:**
- If the 15-day lookback shows alpha up to 3 days ahead, commit positions
  no more than 3 days forward.
- If the 1-day lookback loses to the market immediately, it's overfitting
  to recent noise.
- If the 30-day lookback shows alpha at all horizons, the market may have
  a persistent structural inefficiency at these timescales.

**Statistical validity:**
- DM p-values use Newey-West HAC standard errors.
- "Significant" means p < 0.05 AND alpha > 0.
""")
