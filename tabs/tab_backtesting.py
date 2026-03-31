"""
Forecast Backtesting tab — walk-forward backtesting, alpha detection,
interactive time-series exploration, and trader assessment metrics.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import COLOUR_PRIMARY, COLOUR_MUTED, PLOTLY_TEMPLATE
from src.models.alpha_detector import (
    build_residual_series_for_chart,
    compute_alpha_matrix,
    compute_cumulative_alpha_series,
    compute_full_metrics_table,
    compute_point_in_time_errors,
    find_optimal_horizon,
    run_residual_validation,
)
from src.models.forecaster import (
    DEFAULT_HORIZONS,
    DEFAULT_LOOKBACKS,
    HORIZON_LABELS,
    LOOKBACK_LABELS,
    ForecastResult,
    build_aligned_series,
    run_walk_forward_backtest,
)
from src.session_keys import BACKTEST_RESULTS, DEMAND_DF, MIP_DF, SIP_DF
from src.visualization.charts import (
    alpha_heatmap,
    confirmation_hit_rate_chart,
    correction_improvement_chart,
    cumulative_alpha_chart,
    error_comparison_chart,
    forecast_fan_chart,
    horizon_error_decay,
    residual_scatter_chart,
    residual_time_series_chart,
)


_TRAFFIC_LIGHT = {
    "green": "background-color: #2ECC71; color: black",
    "amber": "background-color: #F1C40F; color: black",
    "red": "background-color: #E74C3C; color: white",
}


def _traffic_light_style(val, metric: str) -> str:
    """Return CSS for traffic-light colouring."""
    try:
        v = float(val)
    except (ValueError, TypeError):
        return ""

    if metric == "Hit Rate":
        return _TRAFFIC_LIGHT["green"] if v > 0.55 else (_TRAFFIC_LIGHT["amber"] if v > 0.50 else _TRAFFIC_LIGHT["red"])
    if metric == "Alpha (MAE)":
        return _TRAFFIC_LIGHT["green"] if v > 0 else _TRAFFIC_LIGHT["red"]
    if metric == "Info Ratio":
        return _TRAFFIC_LIGHT["green"] if v > 0.5 else (_TRAFFIC_LIGHT["amber"] if v > 0 else _TRAFFIC_LIGHT["red"])
    if metric == "Directional Acc.":
        return _TRAFFIC_LIGHT["green"] if v > 0.55 else (_TRAFFIC_LIGHT["amber"] if v > 0.50 else _TRAFFIC_LIGHT["red"])
    if metric == "Stability":
        return _TRAFFIC_LIGHT["green"] if v > 0.7 else (_TRAFFIC_LIGHT["amber"] if v > 0.5 else _TRAFFIC_LIGHT["red"])
    if metric == "R²":
        return _TRAFFIC_LIGHT["green"] if v > 0.3 else (_TRAFFIC_LIGHT["amber"] if v > 0.1 else _TRAFFIC_LIGHT["red"])
    return ""


def render(has_results: bool) -> None:
    st.header("Forecast Backtesting & Alpha Detection")

    if not has_results:
        st.info("Run a simulation first (which fetches ELEXON data) to use the backtesting engine.")
        return

    if SIP_DF not in st.session_state or st.session_state[SIP_DF].empty:
        st.warning("No SIP data available. Run a simulation first.")
        return

    if MIP_DF not in st.session_state or st.session_state[MIP_DF].empty:
        st.warning("No MIP data available. The backtesting engine needs both SIP and MIP data.")
        return

    _render_backtesting_body()


@st.fragment
def _render_backtesting_body() -> None:
    """Isolated as a fragment so widget changes don't reset the active tab."""
    sip_df = st.session_state[SIP_DF]
    mip_df = st.session_state[MIP_DF]

    # ══════════════════════════════════════════════════════════════════
    # Section 1: Backtest Configuration
    # ══════════════════════════════════════════════════════════════════
    st.subheader("1. Backtest Configuration")

    col1, col2, col3 = st.columns(3)
    with col1:
        available_lookbacks = {LOOKBACK_LABELS[lb]: lb for lb in DEFAULT_LOOKBACKS}
        selected_lb_labels = st.multiselect(
            "Lookback windows",
            options=list(available_lookbacks.keys()),
            default=list(available_lookbacks.keys())[:3],
        )
        selected_lookbacks = [available_lookbacks[lb] for lb in selected_lb_labels]

    with col2:
        method = st.radio(
            "Forecast method",
            ["TOD Mean", "EWMA", "XGBoost", "NeuralProphet"],
            horizontal=True,
        )
        method_key = {
            "TOD Mean": "tod_mean",
            "EWMA": "ewma",
            "XGBoost": "xgb",
            "NeuralProphet": "neuralprophet",
        }[method]

    with col3:
        step_size = st.select_slider(
            "Step size (SPs between origins)",
            options=[1, 2, 6, 12, 48],
            value=6,
            help="Lower = more origins = slower but more precise. 48 = daily origins only.",
        )

    col4, col5 = st.columns(2)
    with col4:
        ewma_alpha = st.slider(
            "EWMA smoothing α",
            min_value=0.01, max_value=0.30, value=0.05, step=0.01,
            help="Higher α = more weight on recent observations. "
                 "Treat as a hyperparameter — optimise via walk-forward CV.",
        )
    with col5:
        st.caption(
            "**Step size note:** For statistical assessments, use step=48 "
            "(daily, non-overlapping) so that confidence intervals and p-values "
            "are not inflated by overlapping forecast origins."
        )

    run_backtest = st.button("🔬 Run Backtest", use_container_width=True, type="primary")

    # ══════════════════════════════════════════════════════════════════
    # Run backtest
    # ══════════════════════════════════════════════════════════════════
    if run_backtest:
        demand_df = st.session_state.get(DEMAND_DF)
        with st.spinner("Aligning SIP, MIP and Demand series…"):
            sip_series, mip_series, demand_series = build_aligned_series(
                sip_df, mip_df, demand_df=demand_df,
            )

        if len(sip_series) < 96:
            st.error(f"Insufficient aligned data: only {len(sip_series)} data points. Need at least 96.")
            return

        st.caption(f"Aligned series: {len(sip_series)} data points "
                   f"({sip_series.index[0]} → {sip_series.index[-1]})")

        backtest_results: Dict[Tuple[int, str], List[ForecastResult]] = {}
        progress = st.progress(0)
        total = len(selected_lookbacks)

        for i, lb in enumerate(selected_lookbacks):
            lb_label = LOOKBACK_LABELS.get(lb, f"{lb} SPs")
            with st.spinner(f"Running backtest: lookback={lb_label}…"):
                results = run_walk_forward_backtest(
                    sip_series, mip_series,
                    lookback_sps=lb,
                    horizons=DEFAULT_HORIZONS,
                    method=method_key,
                    step=step_size,
                    ewma_alpha=ewma_alpha,
                    demand_series=demand_series,
                )
                backtest_results[(lb, method_key)] = results
            progress.progress((i + 1) / total)

        st.session_state[BACKTEST_RESULTS] = backtest_results
        st.session_state["_bt_sip_series"] = sip_series
        st.session_state["_bt_mip_series"] = mip_series
        progress.empty()
        st.success(f"Backtest complete — {sum(len(v) for v in backtest_results.values()):,} total forecast origins.")

    # ══════════════════════════════════════════════════════════════════
    # Display results (if available)
    # ══════════════════════════════════════════════════════════════════
    if BACKTEST_RESULTS not in st.session_state:
        st.info("Configure and run a backtest above to see results.")
        return

    backtest_results = st.session_state[BACKTEST_RESULTS]
    sip_series = st.session_state.get("_bt_sip_series")
    mip_series = st.session_state.get("_bt_mip_series")

    if sip_series is None or mip_series is None:
        st.warning("Backtest series data missing from session. Please re-run the backtest.")
        return

    # ══════════════════════════════════════════════════════════════════
    # Section 2: Alpha Heatmap
    # ══════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("2. Alpha Heatmap (Lookback × Horizon)")
    st.caption("Green = our forecast beats market. Red = market beats us. "
               "Values show alpha in £/MWh (positive = we win).")

    alpha_df = compute_alpha_matrix(backtest_results, DEFAULT_HORIZONS)
    if not alpha_df.empty:
        fig_alpha = alpha_heatmap(alpha_df)
        st.plotly_chart(fig_alpha, use_container_width=True)

        optimal = find_optimal_horizon(backtest_results, DEFAULT_HORIZONS)
        st.subheader("Optimal Forecast Horizons")
        opt_rows = []
        for lb, info in optimal.items():
            opt_rows.append({
                "Lookback": lb,
                "Best Horizon": info["best_horizon"],
                "Alpha (£/MWh)": f"{info['alpha_mae']:+.3f}",
                "Hit Rate": f"{info['hit_rate']:.1%}",
                "Info Ratio": f"{info['information_ratio']:.3f}",
                "Dir. Accuracy": f"{info['directional_accuracy']:.1%}",
            })
        st.dataframe(pd.DataFrame(opt_rows), use_container_width=True, hide_index=True)
    else:
        st.warning("Alpha matrix is empty — insufficient data for the selected lookback windows.")

    # ══════════════════════════════════════════════════════════════════
    # Section 3: Interactive Time Series
    # ══════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("3. Interactive Time Series")
    st.caption("Select a point in time to see forecast fan and error analysis below.")

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=sip_series.index, y=sip_series.values,
        mode="lines", name="Realised SIP",
        line=dict(color=COLOUR_PRIMARY, width=1),
    ))
    fig_ts.add_trace(go.Scatter(
        x=mip_series.index, y=mip_series.values,
        mode="lines", name="MIP (Market Forward)",
        line=dict(color=COLOUR_MUTED, width=1, dash="dash"),
    ))
    fig_ts.update_layout(
        template=PLOTLY_TEMPLATE,
        margin=dict(l=50, r=30, t=50, b=50),
        title="Realised SIP vs Market Index Price",
        xaxis_title="Time",
        yaxis_title="£/MWh",
        height=450,
    )

    selected = st.plotly_chart(fig_ts, use_container_width=True, on_select="rerun",
                                key="backtest_ts_chart")

    selected_idx = None
    if selected and selected.get("selection") and selected["selection"].get("points"):
        pts = selected["selection"]["points"]
        if pts:
            sel_x = pts[0].get("x")
            if sel_x is not None:
                try:
                    sel_dt = pd.Timestamp(sel_x)
                    if sel_dt in sip_series.index:
                        selected_idx = sip_series.index.get_loc(sel_dt)
                except Exception:
                    pass

    if selected_idx is None:
        st.info("Click or select a point on the chart above to see forecast details.")

        # Show a slider as fallback
        st.caption("Or use this slider to pick a time index:")
        selected_idx = st.slider(
            "Time index",
            min_value=max(96, min(lb for lb, _ in backtest_results.keys())),
            max_value=len(sip_series) - max(DEFAULT_HORIZONS) - 1,
            value=min(len(sip_series) // 2, len(sip_series) - max(DEFAULT_HORIZONS) - 1),
            key="bt_time_slider",
        )

    if selected_idx is not None:
        origin_dt = sip_series.index[selected_idx]
        st.markdown(f"**Selected point:** {origin_dt}")

        # ══════════════════════════════════════════════════════════════
        # Section 4: Forecast Fan
        # ══════════════════════════════════════════════════════════════
        st.markdown("---")
        st.subheader("4. Forecast Fan (from selected point)")

        first_key = next(iter(backtest_results))
        first_results = backtest_results[first_key]

        target_result = None
        for r in first_results:
            if r.origin_idx == selected_idx:
                target_result = r
                break
        if target_result is None:
            closest = min(first_results,
                          key=lambda r: abs(r.origin_idx - selected_idx),
                          default=None)
            target_result = closest

        if target_result is not None:
            fig_fan = forecast_fan_chart(
                origin_datetime=target_result.origin_datetime,
                sip_series=sip_series,
                forecasts=target_result.forecasts,
                market_fwd=target_result.market_fwd,
                realised=target_result.realised,
            )
            st.plotly_chart(fig_fan, use_container_width=True)

            # ══════════════════════════════════════════════════════════
            # Section 5: Point-in-Time Error Analysis
            # ══════════════════════════════════════════════════════════
            st.markdown("---")
            st.subheader("5. Point-in-Time Error Analysis")

            pit_df = compute_point_in_time_errors(
                first_results, selected_idx, DEFAULT_HORIZONS
            )
            if pit_df is not None and not pit_df.empty:
                fmt_cols = ["Forecast", "Market (MIP)", "Realised", "Forecast Error", "Market Error"]
                for c in fmt_cols:
                    if c in pit_df.columns:
                        pit_df[c] = pit_df[c].apply(
                            lambda v: f"£{v:,.2f}" if not pd.isna(v) else "N/A"
                        )
                st.dataframe(pit_df, use_container_width=True, hide_index=True)

                # Error comparison bar chart
                raw_pit = compute_point_in_time_errors(
                    first_results, selected_idx, DEFAULT_HORIZONS
                )
                if raw_pit is not None:
                    fig_err = error_comparison_chart(
                        horizons=raw_pit["Horizon"].tolist(),
                        forecast_mae=[float(v) if not pd.isna(v) else 0 for v in raw_pit["Forecast Error"]],
                        market_mae=[float(v) if not pd.isna(v) else 0 for v in raw_pit["Market Error"]],
                    )
                    st.plotly_chart(fig_err, use_container_width=True)
        else:
            st.warning("No forecast data available for this time point.")

        # Cumulative alpha chart
        st.markdown("---")
        st.subheader("Cumulative Alpha (up to selected point)")
        horizon_for_cum = st.selectbox(
            "Horizon for cumulative alpha",
            options=[HORIZON_LABELS.get(h, f"{h} SPs") for h in DEFAULT_HORIZONS],
            index=0,
            key="cum_alpha_horizon",
        )
        h_val = {v: k for k, v in HORIZON_LABELS.items()}.get(horizon_for_cum, DEFAULT_HORIZONS[0])

        timestamps, cum_vals = compute_cumulative_alpha_series(first_results, h_val)
        if timestamps:
            cutoff_idx = next(
                (i for i, t in enumerate(timestamps) if t >= origin_dt),
                len(timestamps)
            )
            if cutoff_idx > 0:
                fig_cum = cumulative_alpha_chart(
                    timestamps[:cutoff_idx],
                    cum_vals[:cutoff_idx],
                )
                st.plotly_chart(fig_cum, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════
    # Section 6: Horizon Error Decay
    # ══════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("6. Error Growth by Horizon")
    st.caption("Where our forecast error crosses the market error line marks the "
               "maximum useful forecast horizon.")

    first_key = next(iter(backtest_results))
    first_results = backtest_results[first_key]

    h_labels_ordered = []
    fc_maes = []
    mkt_maes = []
    for h in DEFAULT_HORIZONS:
        h_label = HORIZON_LABELS.get(h, f"{h} SPs")
        fc_errs = []
        mkt_errs = []
        for r in first_results:
            if h in r.forecasts and h in r.realised and h in r.market_fwd:
                fc_errs.append(abs(r.forecasts[h] - r.realised[h]))
                mkt_errs.append(abs(r.market_fwd[h] - r.realised[h]))
        if fc_errs:
            h_labels_ordered.append(h_label)
            fc_maes.append(float(np.mean(fc_errs)))
            mkt_maes.append(float(np.mean(mkt_errs)))

    if h_labels_ordered:
        fig_decay = horizon_error_decay(h_labels_ordered, fc_maes, mkt_maes)
        st.plotly_chart(fig_decay, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════
    # Section 7: Residual Forecasting Validation
    # ══════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("7. Residual Forecasting Validation")
    st.caption(
        "Validates the forecast by predicting its own residuals (forecast − realised). "
        "If residuals are predictable, the model has systematic biases a trader should know about. "
        "A \"bias-corrected\" forecast subtracts the predicted residual. "
        "Position confirmation checks whether the residual signal agrees with the suggested trade direction."
    )

    first_key = next(iter(backtest_results))
    first_results = backtest_results[first_key]

    residual_validations = run_residual_validation(first_results, DEFAULT_HORIZONS)

    if residual_validations:
        # ── Summary table ─────────────────────────────────────────────
        rv_rows = []
        for rv in residual_validations:
            rv_rows.append({
                "Horizon": rv.horizon_label,
                "Residual Mean": f"£{rv.residual_mean:+.2f}",
                "Residual Std": f"£{rv.residual_std:.2f}",
                "Autocorr(1)": f"{rv.residual_autocorr_1:.3f}",
                "Residual R²": f"{rv.residual_predictability:.3f}",
                "Original MAE": f"£{rv.original_mae:.2f}",
                "Corrected MAE": f"£{rv.corrected_mae:.2f}",
                "Improvement": f"{rv.correction_improvement:+.1%}",
                "Confirm Rate": f"{rv.confirmation_rate:.1%}",
                "Confirmed HR": f"{rv.confirmed_trade_hit_rate:.1%}" if not np.isnan(rv.confirmed_trade_hit_rate) else "N/A",
                "Unconfirmed HR": f"{rv.unconfirmed_trade_hit_rate:.1%}" if not np.isnan(rv.unconfirmed_trade_hit_rate) else "N/A",
            })
        st.dataframe(pd.DataFrame(rv_rows), use_container_width=True, hide_index=True)

        # ── Interpretation callout ────────────────────────────────────
        best_rv = max(residual_validations, key=lambda rv: rv.correction_improvement)
        if best_rv.correction_improvement > 0.01:
            st.success(
                f"**Bias correction adds value at {best_rv.horizon_label}:** "
                f"MAE drops from £{best_rv.original_mae:.2f} to £{best_rv.corrected_mae:.2f} "
                f"({best_rv.correction_improvement:+.1%}). "
                f"Residual autocorrelation = {best_rv.residual_autocorr_1:.3f} — "
                f"the model has systematic, exploitable patterns in its errors."
            )
        else:
            st.info(
                "Bias correction does not materially improve the forecast at any horizon. "
                "This suggests the model's errors are largely random (white noise), "
                "which is a **positive sign** — the model is extracting most available signal."
            )

        # ── Confirmation signal interpretation ────────────────────────
        confirming_rv = [rv for rv in residual_validations
                         if not np.isnan(rv.confirmed_trade_hit_rate)
                         and not np.isnan(rv.unconfirmed_trade_hit_rate)]
        if confirming_rv:
            best_confirm = max(confirming_rv,
                               key=lambda rv: (rv.confirmed_trade_hit_rate - rv.unconfirmed_trade_hit_rate))
            spread = best_confirm.confirmed_trade_hit_rate - best_confirm.unconfirmed_trade_hit_rate
            if spread > 0.03:
                st.success(
                    f"**Residual signal adds value at {best_confirm.horizon_label}:** "
                    f"Hit rate is {best_confirm.confirmed_trade_hit_rate:.1%} when the residual "
                    f"confirms the position vs {best_confirm.unconfirmed_trade_hit_rate:.1%} when it "
                    f"contradicts (spread: {spread:+.1%}). "
                    f"A trader should weight positions more heavily when the residual signal confirms."
                )
            elif spread > 0:
                st.warning(
                    f"Marginal confirmation effect at {best_confirm.horizon_label}: "
                    f"confirmed HR = {best_confirm.confirmed_trade_hit_rate:.1%}, "
                    f"unconfirmed HR = {best_confirm.unconfirmed_trade_hit_rate:.1%}. "
                    f"Signal exists but may not be actionable after transaction costs."
                )
            else:
                st.info(
                    "The residual signal does not meaningfully differentiate trade quality. "
                    "Positions are equally valid regardless of residual direction."
                )

        # ── Charts ────────────────────────────────────────────────────
        chart_horizon = st.selectbox(
            "Horizon for residual detail charts",
            options=[rv.horizon_label for rv in residual_validations],
            index=0,
            key="resid_chart_horizon",
        )
        sel_h = next(rv.horizon for rv in residual_validations if rv.horizon_label == chart_horizon)

        ts, actual_resid, pred_resid, fc_vals = build_residual_series_for_chart(
            first_results, sel_h
        )

        if len(ts) > 0:
            col_a, col_b = st.columns(2)
            with col_a:
                fig_resid_ts = residual_time_series_chart(ts, actual_resid, pred_resid)
                st.plotly_chart(fig_resid_ts, use_container_width=True)
            with col_b:
                fig_resid_sc = residual_scatter_chart(actual_resid, pred_resid)
                st.plotly_chart(fig_resid_sc, use_container_width=True)

        # Bias correction improvement chart
        h_labels = [rv.horizon_label for rv in residual_validations]
        orig_maes = [rv.original_mae for rv in residual_validations]
        corr_maes = [rv.corrected_mae for rv in residual_validations]
        fig_corr = correction_improvement_chart(h_labels, orig_maes, corr_maes)
        st.plotly_chart(fig_corr, use_container_width=True)

        # Confirmation hit rate chart
        conf_hrs = [rv.confirmed_trade_hit_rate for rv in residual_validations]
        unconf_hrs = [rv.unconfirmed_trade_hit_rate for rv in residual_validations]
        if any(not np.isnan(v) for v in conf_hrs):
            fig_conf = confirmation_hit_rate_chart(
                h_labels,
                [v if not np.isnan(v) else 0 for v in conf_hrs],
                [v if not np.isnan(v) else 0 for v in unconf_hrs],
            )
            st.plotly_chart(fig_conf, use_container_width=True)
    else:
        st.warning("Insufficient data for residual analysis. Need longer backtest history.")

    # ══════════════════════════════════════════════════════════════════
    # Section 8: Trader Assessment Criteria
    # ══════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("8. Trader Assessment Criteria")
    st.caption("Full performance metrics with traffic-light scoring. "
               "Green = strong, Amber = marginal, Red = weak.")

    metrics_df = compute_full_metrics_table(backtest_results, DEFAULT_HORIZONS)
    if not metrics_df.empty:
        display_df = metrics_df.copy()
        for col in ["Hit Rate", "Directional Acc.", "Stability"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda v: f"{v:.1%}" if not pd.isna(v) else "N/A"
                )
        for col in ["Alpha (MAE)", "Forecast MAE", "Market MAE", "Max Drawdown"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda v: f"£{v:,.2f}" if not pd.isna(v) else "N/A"
                )
        for col in ["Info Ratio", "Calmar Ratio", "R²"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda v: f"{v:.3f}" if not pd.isna(v) else "N/A"
                )
        if "DM p-value" in display_df.columns:
            display_df["DM p-value"] = display_df["DM p-value"].apply(
                lambda v: f"{v:.4f}" if not pd.isna(v) else "N/A"
            )
        if "N Effective" in display_df.columns:
            display_df["N Effective"] = display_df["N Effective"].apply(
                lambda v: f"{v:,.0f}" if not pd.isna(v) else "N/A"
            )

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Significance summary callout
        if "Significant (FDR 5%)" in metrics_df.columns:
            n_sig = (metrics_df["Significant (FDR 5%)"] == "Yes").sum()
            n_total = len(metrics_df)
            if n_sig > 0:
                st.success(
                    f"**{n_sig} of {n_total} configurations** are statistically significant "
                    f"at FDR 5% (Benjamini-Hochberg corrected Diebold-Mariano test). "
                    f"These survive multiple testing correction."
                )
            else:
                st.warning(
                    f"**None of the {n_total} configurations** are statistically significant "
                    f"at FDR 5% after Benjamini-Hochberg correction. "
                    f"Observed alpha may be due to noise or insufficient sample size."
                )

        # Export button
        csv = metrics_df.to_csv(index=False)
        st.download_button(
            "📥 Export Assessment Metrics (CSV)",
            csv,
            file_name="backtest_assessment_metrics.csv",
            mime="text/csv",
        )
    else:
        st.warning("No metrics available. Ensure the backtest produced sufficient data.")

    # Assessment criteria legend
    with st.expander("Assessment Criteria Definitions"):
        st.markdown("""
| Metric | Definition | Green | Amber | Red |
|--------|-----------|-------|-------|-----|
| **Hit Rate** | % of forecasts closer to realised than MIP | >55% | 50-55% | <50% |
| **HR 95% CI** | Wilson score binomial confidence interval on hit rate | If lower bound > 50%, statistically significant | — | — |
| **Directional Accuracy** | % of times we predicted up/down correctly | >55% | 50-55% | <50% |
| **MAE** | Mean Absolute Error (£/MWh) | Lower is better | — | — |
| **Information Ratio** | Mean(alpha) / Std(alpha) | >0.5 | 0-0.5 | <0 |
| **IR 95% CI** | Block-bootstrapped confidence interval for IR | If lower bound > 0, alpha is robust | — | — |
| **DM p-value** | Diebold-Mariano test p-value (HAC-adjusted) | <0.05 = significant | 0.05-0.10 = marginal | >0.10 |
| **Significant (FDR 5%)** | Survives Benjamini-Hochberg multiple testing correction | Yes = trustworthy alpha | — | No |
| **Max Drawdown** | Largest cumulative alpha deterioration | Lower is better | — | — |
| **Calmar Ratio** | Total alpha / Max drawdown | >1.0 | 0-1.0 | <0 |
| **Stability** | 1 - Std(rolling 30-period hit rate) | >0.7 | 0.5-0.7 | <0.5 |
| **R²** | Walk-forward out-of-sample explanatory power | >0.3 | 0.1-0.3 | <0.1 |
| **N Effective** | Effective sample size after autocorrelation | Higher = more reliable | — | — |
""")

    with st.expander("Residual Validation Definitions (Section 7)"):
        st.markdown("""
| Metric | Definition | Interpretation |
|--------|-----------|----------------|
| **Residual Mean** | Average (forecast − realised) | Non-zero = systematic bias; positive = over-prediction |
| **Residual Std** | Standard deviation of residuals | Lower = more precise model |
| **Autocorr(1)** | Lag-1 autocorrelation of residuals | High = residuals are predictable (model leaves signal on the table) |
| **Residual R²** | Walk-forward R² of predicted vs actual residuals | >0.1 = exploitable pattern; ~0 = white-noise errors (good) |
| **Original MAE** | MAE of the raw forecast | Baseline accuracy |
| **Corrected MAE** | MAE after subtracting predicted residual | If < Original MAE, bias correction adds value |
| **Improvement** | (Original − Corrected) / Original | Positive % = correction helps; negative = overcorrection |
| **Confirm Rate** | % of times predicted-residual sign matches forecast-vs-MIP direction | ~50% = no signal; >>50% = residual confirms position |
| **Confirmed HR** | Hit rate (vs MIP) when residual signal confirms | Higher = the residual is a useful filter |
| **Unconfirmed HR** | Hit rate when residual contradicts | If much lower than Confirmed HR, residual adds real value as a second opinion |
""")
