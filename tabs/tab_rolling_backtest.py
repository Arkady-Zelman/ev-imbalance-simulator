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
from src.models.xgb_trainer import (
    forecast_forward,
    has_trained_models,
    load_trained_models,
    save_trained_models,
    train_xgb_models,
)
from src.session_keys import DEMAND_DF, MIP_DF, SIP_DF


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

    _render_rolling_backtest_body()


@st.fragment
def _render_rolling_backtest_body() -> None:
    """Isolated as a fragment so widget changes don't reset the active tab."""
    sip_df = st.session_state[SIP_DF]
    mip_df = st.session_state[MIP_DF]
    demand_df = st.session_state.get(DEMAND_DF)

    has_demand = demand_df is not None and not demand_df.empty

    # ── Configuration ─────────────────────────────────────────────────
    st.subheader("Configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        target_label = st.radio(
            "Forecast target",
            ["Price", "Demand"] if has_demand else ["Price"],
            horizontal=True,
            key="rb_target",
        )
        target_key = "demand" if target_label == "Demand" else "sip"
    with col2:
        method_options = ["TOD Mean", "EWMA", "XGBoost", "NeuralProphet"]
        method = st.radio(
            "Forecast method",
            method_options,
            horizontal=True,
            key="rb_method",
        )
        method_key = {
            "TOD Mean": "tod_mean",
            "EWMA": "ewma",
            "XGBoost": "xgb",
            "NeuralProphet": "neuralprophet",
        }[method]
    with col3:
        ewma_alpha = st.slider(
            "EWMA α", 0.01, 0.30, 0.05, 0.01,
            key="rb_ewma_alpha",
            help="Only used when EWMA is selected.",
        )

    if target_key == "demand" and not has_demand:
        st.warning("No demand data available. Run a simulation with demand data first.")
        return

    # ── XGBoost: Train / Show Results workflow ────────────────────────
    if method_key == "xgb":
        target_desc = "Demand" if target_key == "demand" else "Price"
        ss_key = f"_xgb_trained_models_{target_key}"

        st.markdown("---")
        st.subheader(f"XGBoost {target_desc} Model Training")
        st.caption(
            f"Train a **{target_desc}** forecast model, then view results anytime. "
            "Price and Demand models are stored separately and persist across sessions."
        )

        col_deep, col_quick, col_show = st.columns(3)
        with col_deep:
            deep_btn = st.button(
                f"🏋️ In-Depth Search ({target_desc})",
                use_container_width=True,
                type="primary",
                key="rb_xgb_deep",
                help="Systematic grid search over core hyperparameters (27 combos). "
                     "Run periodically for best accuracy. Takes several minutes.",
            )
        with col_quick:
            quick_btn = st.button(
                f"⚡ Quick Random Search ({target_desc})",
                use_container_width=True,
                key="rb_xgb_quick",
                help="Random search across all hyperparameters (30 samples). "
                     "Faster estimate for point-in-time tuning.",
            )
        with col_show:
            show_btn = st.button(
                f"📊 Show {target_desc} Results",
                use_container_width=True,
                key="rb_xgb_show",
                help=f"Load and display results from the last trained {target_desc} model.",
            )

        search_mode = None
        if deep_btn:
            search_mode = "grid"
        elif quick_btn:
            search_mode = "random"

        if search_mode is not None:
            mode_label = "In-depth grid" if search_mode == "grid" else "Quick random"
            with st.spinner("Aligning SIP, MIP and Demand series…"):
                sip_series, mip_series, demand_series = build_aligned_series(
                    sip_df, mip_df, demand_df=demand_df,
                )
            n_days = len(sip_series) // 48
            if n_days < 45:
                st.error(
                    f"Need at least 45 days of aligned data. "
                    f"Currently have {n_days} days. Increase the date range."
                )
                return

            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def _progress(frac: float, msg: str) -> None:
                progress_bar.progress(min(frac, 1.0))
                status_text.caption(msg)

            trained = train_xgb_models(
                sip_series, mip_series,
                demand_series=demand_series,
                target=target_key,
                progress_callback=_progress,
                param_search_mode=search_mode,
            )
            progress_bar.empty()
            status_text.empty()

            save_trained_models(trained, target=target_key)
            st.session_state[ss_key] = trained

            _display_training_summary(trained)

            rb_key_prefix = f"_rb_{target_key}_{method_key}"
            st.session_state[f"{rb_key_prefix}_errors"] = trained.backtest_errors
            st.session_state[f"{rb_key_prefix}_crossovers"] = trained.backtest_crossovers
            st.session_state["_rb_last_key"] = rb_key_prefix
            st.success(
                f"{mode_label} {target_desc} training complete — "
                f"{len(trained.backtest_errors)} backtest configurations evaluated. "
                f"Model saved to disk."
            )

        if show_btn:
            trained = st.session_state.get(ss_key)
            if trained is None:
                trained = load_trained_models(target=target_key)
                if trained is not None:
                    st.session_state[ss_key] = trained

            if trained is None:
                st.info(
                    f"No trained {target_desc} model found. "
                    f"Press one of the training buttons first."
                )
                return

            import datetime as dt
            ts = dt.datetime.fromtimestamp(trained.training_timestamp)
            st.caption(f"Loaded {target_desc} model trained at **{ts:%Y-%m-%d %H:%M}**")
            _display_training_summary(trained)

            rb_key_prefix = f"_rb_{target_key}_{method_key}"
            st.session_state[f"{rb_key_prefix}_errors"] = trained.backtest_errors
            st.session_state[f"{rb_key_prefix}_crossovers"] = trained.backtest_crossovers
            st.session_state["_rb_last_key"] = rb_key_prefix

    else:
        # ── TOD Mean / EWMA / NeuralProphet: standard single-button flow
        target_desc = "Demand" if target_key == "demand" else "Price"
        run_btn = st.button(
            f"🔬 Run {target_desc} Rolling Backtest ({method})",
            use_container_width=True,
            type="primary",
            key="rb_run",
        )

        if run_btn:
            with st.spinner("Aligning SIP, MIP and Demand series…"):
                sip_series, mip_series, demand_series = build_aligned_series(
                    sip_df, mip_df, demand_df=demand_df,
                )

            n_days = len(sip_series) // 48
            if n_days < 45:
                st.error(
                    f"Need at least 45 days of aligned data for a meaningful rolling backtest. "
                    f"Currently have {n_days} days. Increase the date range."
                )
                return

            st.caption(f"Data: {len(sip_series):,} SPs ({n_days} days) — "
                       f"{sip_series.index[0]} → {sip_series.index[-1]}  |  "
                       f"Target: **{target_desc}**  |  Method: **{method}**")

            spinner_msg = (
                f"Running rolling backtest ({method}, {target_desc}, "
                f"1/5/15/30-day lookbacks × 1-14 day horizons)…"
            )
            with st.spinner(spinner_msg):
                errors, crossovers = run_rolling_backtest(
                    sip_series, mip_series,
                    method=method_key,
                    ewma_alpha=ewma_alpha,
                    target=target_key,
                    demand_series=demand_series,
                )

            rb_key_prefix = f"_rb_{target_key}_{method_key}"
            st.session_state[f"{rb_key_prefix}_errors"] = errors
            st.session_state[f"{rb_key_prefix}_crossovers"] = crossovers
            st.session_state["_rb_last_key"] = rb_key_prefix
            st.success(f"Rolling backtest complete — {len(errors)} configurations evaluated.")

    # ── Results ───────────────────────────────────────────────────────
    rb_key_prefix = st.session_state.get("_rb_last_key")
    if rb_key_prefix is None or f"{rb_key_prefix}_errors" not in st.session_state:
        st.info("Configure and run the rolling backtest above.")
        return

    errors = st.session_state[f"{rb_key_prefix}_errors"]
    crossovers = st.session_state[f"{rb_key_prefix}_crossovers"]

    is_demand = rb_key_prefix and "_demand_" in rb_key_prefix
    unit = "MW" if is_demand else "£/MWh"

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
        "market's. Beyond this horizon, the benchmark is a better predictor."
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
            "Last +Alpha": f"{c.last_positive_alpha:+.2f} {unit}",
            "First −Alpha": f"{c.first_negative_alpha:+.2f} {unit}" if c.crossover_day < 15 else "N/A",
            "Verdict": verdict,
        })
    st.dataframe(pd.DataFrame(cross_rows), use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════
    # Section 2: Alpha Heatmap (Lookback × Horizon)
    # ══════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("2. Alpha Heatmap — Lookback × Forecast Horizon")
    st.caption("Green = our forecast beats the benchmark. Red = benchmark wins.")

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
            colorbar=dict(title=f"Alpha ({unit})"),
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
        "5 days": COLOUR_MUTED,
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
        yaxis_title=f"MAE ({unit})",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_err, use_container_width=True)

    # MAPE chart
    fig_mape = go.Figure()
    for lb_label in ROLLING_LOOKBACKS:
        lb_rows = sorted(
            [e for e in errors if e.lookback_label == lb_label],
            key=lambda e: e.horizon_days,
        )
        if not lb_rows:
            continue

        days = [e.horizon_days for e in lb_rows]
        fc_mapes = [e.forecast_mape for e in lb_rows]
        mkt_mapes = [e.market_mape for e in lb_rows]

        colour = lb_colours.get(lb_label, COLOUR_MUTED)

        fig_mape.add_trace(go.Scatter(
            x=days, y=fc_mapes,
            mode="lines+markers", name=f"Our Forecast ({lb_label})",
            line=dict(color=colour, width=2),
            marker=dict(size=6),
        ))
        fig_mape.add_trace(go.Scatter(
            x=days, y=mkt_mapes,
            mode="lines+markers", name=f"Market ({lb_label})",
            line=dict(color=colour, width=1, dash="dash"),
            marker=dict(size=4, symbol="x"),
        ))

    fig_mape.update_layout(
        template=PLOTLY_TEMPLATE,
        margin=dict(l=50, r=30, t=50, b=50),
        title="MAPE by Forecast Horizon (days ahead)",
        xaxis_title="Forecast Horizon (days)",
        yaxis_title="MAPE (%)",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_mape, use_container_width=True)

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
        yaxis_title=f"Alpha ({unit})",
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
            f"Forecast MAE ({unit})": f"{e.forecast_mae:.2f}",
            f"Benchmark MAE ({unit})": f"{e.market_mae:.2f}",
            f"Alpha ({unit})": f"{e.alpha_mae:+.2f}",
            "Forecast MAPE": f"{e.forecast_mape:.1f}%",
            "Benchmark MAPE": f"{e.market_mape:.1f}%",
            "Alpha (MAPE)": f"{e.alpha_mape:+.1f}pp",
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

**XGBoost method:**
- Trains a fresh gradient-boosted tree per (lookback, horizon) at each origin.
- Features: time-of-day, day-of-week, lags (1d/2d/7d), rolling stats, MIP, demand.
- **In-depth search:** systematic grid over core hyperparams (27 combos).
- **Quick random search:** samples all hyperparams (30 combos) for fast estimation.
- Assessment criterion: **worst-window MAE** (worst consecutive 3-day stretch) — ensures
  robustness at any point along the forward curve.

**NeuralProphet method:**
- Combines AR-Net (autoregression) with Prophet's trend + seasonality decomposition.
- Captures half-hourly and weekly patterns natively via Fourier terms.
- Heavier than TOD Mean/EWMA but lighter than XGBoost; good for structured patterns.
- Falls back to TOD Mean if neuralprophet is not installed.

**Demand target:**
- When forecasting demand, the benchmark is the TOD-mean of demand itself
  (how well does a simple seasonal average predict demand).
- Demand is smoother than SIP, so baseline models perform better —
  XGBoost/NeuralProphet add value by capturing weather/calendar effects.
""")

    # ══════════════════════════════════════════════════════════════════
    # Section 6: 14-Day Forward Forecast (XGBoost only)
    # ══════════════════════════════════════════════════════════════════
    _render_forward_forecast()

    # ══════════════════════════════════════════════════════════════════
    # Section 7: Combined SIP + Demand Alpha Analysis
    # ══════════════════════════════════════════════════════════════════
    _render_combined_analysis()


def _display_training_summary(trained) -> None:
    """Show a compact summary of best params from grid search."""
    summary_rows = []
    for lb_label in trained.best_params:
        for h_sps, params in trained.best_params[lb_label].items():
            h_days = h_sps // 48
            score = trained.best_scores.get(lb_label, {}).get(h_sps, float("nan"))
            summary_rows.append({
                "Lookback": lb_label,
                "Horizon": f"{h_days}d",
                "n_estimators": params.get("n_estimators"),
                "max_depth": params.get("max_depth"),
                "learning_rate": params.get("learning_rate"),
                "Worst-Window MAE": f"{score:.2f}" if score != float("inf") else "N/A",
            })
    if summary_rows:
        with st.expander("Grid Search Results — Best Params per (Lookback, Horizon)"):
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


def _render_forward_forecast() -> None:
    """Show 14-day forward forecast from trained XGBoost models (SIP and/or Demand)."""
    sip_df = st.session_state.get(SIP_DF)
    mip_df = st.session_state.get(MIP_DF)
    demand_df = st.session_state.get(DEMAND_DF)

    if sip_df is None or sip_df.empty or mip_df is None or mip_df.empty:
        return

    trained_sip = st.session_state.get("_xgb_trained_models_sip")
    trained_demand = st.session_state.get("_xgb_trained_models_demand")

    if (trained_sip is None or not trained_sip.final_models) and \
       (trained_demand is None or not trained_demand.final_models):
        return

    from src.models.forecaster import build_aligned_series

    sip_series, mip_series, demand_series = build_aligned_series(
        sip_df, mip_df, demand_df=demand_df,
    )

    sip_values = sip_series.values.astype(float)
    mip_values = mip_series.values.astype(float)
    demand_values = demand_series.values.astype(float) if demand_series is not None else None

    last_date = sip_series.index[-1]
    lb_colours = {
        "1 day": COLOUR_PRIMARY,
        "5 days": COLOUR_MUTED,
        "15 days": COLOUR_WARNING,
        "30 days": COLOUR_SUCCESS,
    }

    st.markdown("---")
    st.subheader("6. 14-Day Forward Forecast (XGBoost)")
    st.caption(
        "Point forecasts from the latest data point using trained models. "
        "Price and Demand models are shown separately when available."
    )

    for trained, label, unit, y_label in [
        (trained_sip, "Price", "£/MWh", "Predicted Price (£/MWh)"),
        (trained_demand, "Demand", "MW", "Predicted Demand (MW)"),
    ]:
        if trained is None or not trained.final_models:
            continue

        forecasts = forecast_forward(
            trained, sip_values, mip_values, demand_values, n_days=14,
        )
        if not forecasts:
            continue

        fig = go.Figure()
        for lb_label, day_forecasts in forecasts.items():
            if not day_forecasts:
                continue
            days = sorted(day_forecasts.keys())
            dates = [last_date + pd.Timedelta(days=d) for d in days]
            values = [day_forecasts[d] for d in days]
            colour = lb_colours.get(lb_label, COLOUR_MUTED)

            fig.add_trace(go.Scatter(
                x=dates, y=values,
                mode="lines+markers",
                name=f"{lb_label} lookback",
                line=dict(color=colour, width=2),
                marker=dict(size=6),
            ))

        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            title=f"{label} Forward Forecast — Next 14 Days",
            xaxis_title="Date",
            yaxis_title=y_label,
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        fc_rows = []
        for lb_label, day_forecasts in forecasts.items():
            for d in sorted(day_forecasts.keys()):
                fc_rows.append({
                    "Lookback": lb_label,
                    "Day Ahead": d,
                    "Date": (last_date + pd.Timedelta(days=d)).strftime("%Y-%m-%d"),
                    f"Forecast ({unit})": f"{day_forecasts[d]:.2f}",
                })
        if fc_rows:
            with st.expander(f"{label} Forward Forecast Data"):
                st.dataframe(pd.DataFrame(fc_rows), use_container_width=True, hide_index=True)


def _render_combined_analysis() -> None:
    """Show joint SIP + Demand alpha when both backtests have been run."""
    sip_keys = [k for k in st.session_state if k.startswith("_rb_sip_") and k.endswith("_errors")]
    demand_keys = [k for k in st.session_state if k.startswith("_rb_demand_") and k.endswith("_errors")]

    if not sip_keys or not demand_keys:
        return

    st.markdown("---")
    st.subheader("7. Combined Price + Demand Alpha Analysis")
    st.caption(
        "When both a Price and Demand rolling backtest have been run, this section "
        "shows where you have forecast alpha on **both** axes simultaneously — "
        "essential for capacity allocation decisions."
    )

    sip_errors = st.session_state[sip_keys[-1]]
    demand_errors = st.session_state[demand_keys[-1]]

    sip_method = sip_keys[-1].replace("_rb_sip_", "").replace("_errors", "")
    demand_method = demand_keys[-1].replace("_rb_demand_", "").replace("_errors", "")

    best_lb = "30 days"

    sip_by_h = {e.horizon_days: e.alpha_mae for e in sip_errors if e.lookback_label == best_lb}
    dem_by_h = {e.horizon_days: e.alpha_mae for e in demand_errors if e.lookback_label == best_lb}

    if not sip_by_h and not dem_by_h:
        for lb in ["15 days", "1 day"]:
            sip_by_h = {e.horizon_days: e.alpha_mae for e in sip_errors if e.lookback_label == lb}
            dem_by_h = {e.horizon_days: e.alpha_mae for e in demand_errors if e.lookback_label == lb}
            if sip_by_h or dem_by_h:
                best_lb = lb
                break

    common_days = sorted(set(sip_by_h.keys()) & set(dem_by_h.keys()))
    if not common_days:
        st.info("No overlapping horizons between Price and Demand backtests.")
        return

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=[f"{d}d" for d in common_days],
        y=[sip_by_h[d] for d in common_days],
        name=f"Price Alpha ({sip_method})",
        marker_color=COLOUR_PRIMARY,
        opacity=0.8,
    ))
    fig.add_trace(go.Bar(
        x=[f"{d}d" for d in common_days],
        y=[dem_by_h[d] for d in common_days],
        name=f"Demand Alpha ({demand_method})",
        marker_color=COLOUR_SUCCESS,
        opacity=0.8,
    ))

    fig.add_hline(y=0, line_color="white", line_width=1, line_dash="dash")
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        barmode="group",
        title=f"Joint Alpha by Horizon (lookback: {best_lb})",
        xaxis_title="Forecast Horizon",
        yaxis_title="Alpha (lower error than benchmark)",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    both_positive = [d for d in common_days if sip_by_h[d] > 0 and dem_by_h[d] > 0]
    if both_positive:
        st.success(
            f"**Joint alpha detected at horizons:** {', '.join(f'{d}d' for d in both_positive)}. "
            f"At these horizons, you can forecast both price AND demand better than their "
            f"respective benchmarks — these are the strongest signals for capacity allocation."
        )
    else:
        st.warning(
            "No horizons found where both Price and Demand forecasts simultaneously "
            "beat their benchmarks. Consider adjusting methods or lookback windows."
        )

    combo_rows = []
    for d in common_days:
        sip_a = sip_by_h[d]
        dem_a = dem_by_h[d]
        combined = sip_a + dem_a
        both = "✅" if sip_a > 0 and dem_a > 0 else "❌"
        combo_rows.append({
            "Horizon": f"{d}d",
            "Price Alpha (£/MWh)": f"{sip_a:+.2f}",
            "Demand Alpha (MW)": f"{dem_a:+.2f}",
            "Combined Score": f"{combined:+.2f}",
            "Both Positive": both,
        })
    st.dataframe(pd.DataFrame(combo_rows), use_container_width=True, hide_index=True)
