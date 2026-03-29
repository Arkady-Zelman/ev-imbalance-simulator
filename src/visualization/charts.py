"""
Plotly chart builders for the Imbalance Exposure Simulator.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import (
    COLOUR_ACCENT,
    COLOUR_DANGER,
    COLOUR_MUTED,
    COLOUR_PRIMARY,
    COLOUR_SECONDARY,
    COLOUR_SUCCESS,
    COLOUR_WARNING,
    PLOTLY_TEMPLATE,
    SP_LABELS,
)

_LAYOUT_DEFAULTS = dict(
    template=PLOTLY_TEMPLATE,
    margin=dict(l=50, r=30, t=50, b=50),
    font=dict(size=12),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)


def _apply_defaults(fig: go.Figure, **overrides) -> go.Figure:
    merged = {**_LAYOUT_DEFAULTS, **overrides}
    fig.update_layout(**merged)
    return fig


# ── P&L histogram with VaR / CVaR lines ──────────────────────────────────

def pnl_histogram(
    pnl: np.ndarray,
    var_95: float,
    cvar_95: float,
    title: str = "Daily P&L Distribution",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=pnl, nbinsx=80,
        marker_color=COLOUR_PRIMARY, opacity=0.75,
        name="Daily P&L",
    ))
    fig.add_vline(x=var_95, line_dash="dash", line_color=COLOUR_WARNING,
                  annotation_text=f"VaR(95%) = £{var_95:,.0f}",
                  annotation_position="top left")
    fig.add_vline(x=cvar_95, line_dash="dot", line_color=COLOUR_DANGER,
                  annotation_text=f"CVaR(95%) = £{cvar_95:,.0f}",
                  annotation_position="top left")
    return _apply_defaults(fig, title=title,
                           xaxis_title="Daily P&L (£)",
                           yaxis_title="Frequency")


def pnl_comparison_histograms(
    pnl_dict: Dict[str, np.ndarray],
    title: str = "P&L by Risk Appetite",
) -> go.Figure:
    """Side-by-side overlaid histograms for multiple position-sizing tiers."""
    colours = [COLOUR_PRIMARY, COLOUR_ACCENT, COLOUR_WARNING,
               COLOUR_SECONDARY, COLOUR_MUTED, COLOUR_SUCCESS]
    fig = go.Figure()
    for i, (label, pnl) in enumerate(pnl_dict.items()):
        fig.add_trace(go.Histogram(
            x=pnl, nbinsx=60,
            marker_color=colours[i % len(colours)],
            opacity=0.5, name=label,
        ))
    fig.update_layout(barmode="overlay")
    return _apply_defaults(fig, title=title,
                           xaxis_title="Daily P&L (£)",
                           yaxis_title="Frequency")


# ── Delivered vs Traded scatter ───────────────────────────────────────────

def delivered_vs_traded_scatter(
    delivered_daily: np.ndarray,
    traded_daily: float,
    title: str = "Total Delivered MW vs Traded MW",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=delivered_daily, nbinsx=60,
        marker_color=COLOUR_PRIMARY, opacity=0.7,
        name="Delivered (sum across SPs)",
    ))
    fig.add_vline(x=traded_daily, line_dash="dash", line_color=COLOUR_DANGER,
                  annotation_text=f"Traded = {traded_daily:,.1f} MWh",
                  annotation_position="top right")
    return _apply_defaults(fig, title=title,
                           xaxis_title="Total Daily MWh Delivered",
                           yaxis_title="Frequency")


# ── Risk-return frontier ──────────────────────────────────────────────────

def risk_return_frontier(
    labels: List[str],
    expected_pnl: List[float],
    cvar_values: List[float],
    title: str = "Risk-Return Frontier",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cvar_values, y=expected_pnl,
        mode="lines+markers+text",
        text=labels,
        textposition="top center",
        marker=dict(size=12, color=COLOUR_PRIMARY),
        line=dict(color=COLOUR_ACCENT, width=2),
    ))
    return _apply_defaults(fig, title=title,
                           xaxis_title="CVaR 95% – Expected Tail Loss (£)",
                           yaxis_title="Expected Daily P&L (£)")


# ── Capture ratio histogram ──────────────────────────────────────────────

def capture_ratio_histogram(
    ratios: np.ndarray,
    title: str = "Capture Ratio Distribution",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=ratios, nbinsx=80,
        marker_color=COLOUR_ACCENT, opacity=0.75,
    ))
    fig.add_vline(x=1.0, line_dash="dash", line_color=COLOUR_WARNING,
                  annotation_text="Perfect Capture (1.0)")
    return _apply_defaults(fig, title=title,
                           xaxis_title="Capture Ratio",
                           yaxis_title="Frequency")


# ── Tornado / sensitivity diagram ────────────────────────────────────────

def tornado_diagram(
    param_names: List[str],
    low_values: List[float],
    high_values: List[float],
    base_value: float,
    title: str = "CVaR Sensitivity (Tornado)",
) -> go.Figure:
    sorted_idx = np.argsort([abs(h - l) for l, h in zip(low_values, high_values)])
    param_names = [param_names[i] for i in sorted_idx]
    low_values = [low_values[i] for i in sorted_idx]
    high_values = [high_values[i] for i in sorted_idx]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=param_names,
        x=[l - base_value for l in low_values],
        orientation="h",
        marker_color=COLOUR_SECONDARY,
        name="Low scenario",
    ))
    fig.add_trace(go.Bar(
        y=param_names,
        x=[h - base_value for h in high_values],
        orientation="h",
        marker_color=COLOUR_SUCCESS,
        name="High scenario",
    ))
    fig.add_vline(x=0, line_color="white", line_width=1)
    return _apply_defaults(fig, title=title,
                           xaxis_title=f"Change in CVaR from base (£{base_value:,.0f})",
                           barmode="overlay")


# ── Portfolio diversification curve ───────────────────────────────────────

def diversification_curve(
    fleet_sizes: List[int],
    cv_values: List[float],
    title: str = "Portfolio Diversification Effect",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fleet_sizes, y=cv_values,
        mode="lines+markers",
        marker=dict(size=8, color=COLOUR_PRIMARY),
        line=dict(color=COLOUR_PRIMARY, width=2),
    ))
    return _apply_defaults(fig, title=title,
                           xaxis_title="Fleet Size (chargers)",
                           yaxis_title="Coefficient of Variation (σ/μ) of Delivered MW")


# ── Plug-in rate bar chart with confidence bands ──────────────────────────

def plugin_rate_profile(
    means: np.ndarray,
    p5: np.ndarray,
    p95: np.ndarray,
    title: str = "Plug-in Rate by Settlement Period",
) -> go.Figure:
    labels = SP_LABELS
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=means,
        marker_color=COLOUR_PRIMARY, opacity=0.7,
        name="Mean Plug-in Rate",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=p95,
        mode="lines", line=dict(color=COLOUR_ACCENT, dash="dot"),
        name="P95 Band",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=p5,
        mode="lines", line=dict(color=COLOUR_SECONDARY, dash="dot"),
        fill="tonexty", fillcolor="rgba(78,205,196,0.1)",
        name="P5 Band",
    ))
    return _apply_defaults(fig, title=title,
                           xaxis_title="Settlement Period",
                           yaxis_title="Plug-in Rate")


# ── Available MW curve ────────────────────────────────────────────────────

def available_mw_profile(
    mean_mw: np.ndarray,
    p5_mw: np.ndarray,
    p95_mw: np.ndarray,
    traded_mw: Optional[np.ndarray] = None,
    title: str = "Available MW by Settlement Period",
) -> go.Figure:
    labels = SP_LABELS
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=p95_mw, mode="lines",
        line=dict(color="rgba(78,205,196,0.3)"),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=p5_mw, mode="lines",
        line=dict(color="rgba(78,205,196,0.3)"),
        fill="tonexty", fillcolor="rgba(78,205,196,0.15)",
        name="P5–P95 Band",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=mean_mw, mode="lines",
        line=dict(color=COLOUR_PRIMARY, width=2),
        name="Mean Available MW",
    ))
    if traded_mw is not None:
        fig.add_trace(go.Scatter(
            x=labels, y=traded_mw, mode="lines",
            line=dict(color=COLOUR_DANGER, width=2, dash="dash"),
            name="Traded Position",
        ))
    return _apply_defaults(fig, title=title,
                           xaxis_title="Settlement Period",
                           yaxis_title="MW")


# ── Beta distribution overlay ─────────────────────────────────────────────

def beta_distribution_overlay(
    alphas: np.ndarray,
    betas: np.ndarray,
    sp_indices: List[int],
) -> go.Figure:
    """Show Beta PDF shapes for selected settlement periods."""
    from scipy.stats import beta as beta_dist
    x = np.linspace(0, 1, 200)
    colours = [COLOUR_PRIMARY, COLOUR_ACCENT, COLOUR_WARNING,
               COLOUR_SECONDARY, COLOUR_SUCCESS]
    fig = go.Figure()
    for i, sp in enumerate(sp_indices):
        y = beta_dist.pdf(x, alphas[sp], betas[sp])
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines",
            line=dict(color=colours[i % len(colours)], width=2),
            name=f"SP {sp+1} ({SP_LABELS[sp]})",
        ))
    return _apply_defaults(fig, title="Beta Distribution Shapes by Settlement Period",
                           xaxis_title="Plug-in Rate",
                           yaxis_title="Density")


# ── Sharpe-like bar comparison ────────────────────────────────────────────

def sharpe_comparison(
    labels: List[str],
    values: List[float],
    title: str = "Risk-Adjusted Return by Position Size",
) -> go.Figure:
    colours = [COLOUR_SUCCESS if v > 0 else COLOUR_DANGER for v in values]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=values,
        marker_color=colours,
    ))
    return _apply_defaults(fig, title=title,
                           xaxis_title="Position Sizing Tier",
                           yaxis_title="Sharpe-like Ratio (Mean/Std)")


# ── SIP time series ──────────────────────────────────────────────────────

def sip_time_series(
    dates, prices,
    title: str = "System Imbalance Price – Historical",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=prices,
        mode="lines",
        line=dict(color=COLOUR_PRIMARY, width=1),
    ))
    return _apply_defaults(fig, title=title,
                           xaxis_title="Date",
                           yaxis_title="SIP (£/MWh)")


def sip_distribution(
    prices: np.ndarray,
    title: str = "SIP Distribution",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=prices, nbinsx=120,
        marker_color=COLOUR_ACCENT, opacity=0.7,
    ))
    return _apply_defaults(fig, title=title,
                           xaxis_title="SIP (£/MWh)",
                           yaxis_title="Frequency")


# ── Parameter sweep line chart ────────────────────────────────────────────

def parameter_sweep_chart(
    x_values, y_values,
    x_label: str, y_label: str,
    title: str = "Parameter Sweep",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_values, y=y_values,
        mode="lines+markers",
        marker=dict(size=8, color=COLOUR_PRIMARY),
        line=dict(color=COLOUR_PRIMARY, width=2),
    ))
    return _apply_defaults(fig, title=title,
                           xaxis_title=x_label,
                           yaxis_title=y_label)


# ══════════════════════════════════════════════════════════════════════════
#  Backtest visualisation charts
# ══════════════════════════════════════════════════════════════════════════

def forecast_fan_chart(
    origin_datetime,
    sip_series,
    forecasts: Dict[int, float],
    market_fwd: Dict[int, float],
    realised: Dict[int, float],
    title: str = "Multi-Horizon Forecast Fan",
) -> go.Figure:
    """
    Fan chart showing multiple forecast curves branching from the selected
    point, with MIP forward and realised path overlaid.
    """
    from src.models.forecaster import HORIZON_LABELS

    fig = go.Figure()

    context_start = max(0, sip_series.index.get_loc(origin_datetime) - 48)
    context_end = min(len(sip_series), sip_series.index.get_loc(origin_datetime) + max(forecasts.keys(), default=48) + 1)
    context = sip_series.iloc[context_start:context_end]
    fig.add_trace(go.Scatter(
        x=context.index, y=context.values,
        mode="lines", name="Realised SIP",
        line=dict(color=COLOUR_PRIMARY, width=2),
    ))

    fan_colours = [COLOUR_ACCENT, COLOUR_WARNING, COLOUR_SUCCESS,
                   "#9B59B6", "#3498DB", COLOUR_SECONDARY]

    for i, (h, fc_val) in enumerate(sorted(forecasts.items())):
        if h not in realised:
            continue
        target_dt = origin_datetime + pd.Timedelta(minutes=30 * h)
        fig.add_trace(go.Scatter(
            x=[origin_datetime, target_dt],
            y=[sip_series.loc[origin_datetime], fc_val],
            mode="lines+markers",
            name=f"Forecast {HORIZON_LABELS.get(h, str(h))}",
            line=dict(color=fan_colours[i % len(fan_colours)], width=2, dash="solid"),
            marker=dict(size=8),
        ))

    for i, (h, mkt_val) in enumerate(sorted(market_fwd.items())):
        target_dt = origin_datetime + pd.Timedelta(minutes=30 * h)
        fig.add_trace(go.Scatter(
            x=[origin_datetime, target_dt],
            y=[sip_series.loc[origin_datetime], mkt_val],
            mode="lines+markers",
            name=f"MIP Fwd {HORIZON_LABELS.get(h, str(h))}",
            line=dict(color=COLOUR_MUTED, width=1, dash="dash"),
            marker=dict(size=5, symbol="x"),
            showlegend=(i == 0),
            legendgroup="mip_fwd",
        ))

    origin_str = str(origin_datetime)
    fig.add_shape(
        type="line", x0=origin_str, x1=origin_str,
        y0=0, y1=1, yref="paper",
        line=dict(color="white", width=1, dash="dot"),
    )
    fig.add_annotation(
        x=origin_str, y=1, yref="paper",
        text="Origin", showarrow=False,
        font=dict(color="white", size=10),
        yanchor="bottom",
    )

    return _apply_defaults(fig, title=title,
                           xaxis_title="Time", yaxis_title="SIP (£/MWh)")


def alpha_heatmap(
    alpha_df,
    title: str = "Alpha Matrix: Lookback × Horizon",
) -> go.Figure:
    """
    2D heatmap: lookback on Y, horizon on X, coloured by alpha
    (green = we beat market, red = market beats us).
    """
    z_vals = alpha_df.values.astype(float)
    text_vals = [[f"{v:+.2f}" if not np.isnan(v) else "N/A" for v in row] for row in z_vals]

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=z_vals,
        x=list(alpha_df.columns),
        y=list(alpha_df.index),
        colorscale=[
            [0.0, COLOUR_DANGER],
            [0.5, "#2C3E50"],
            [1.0, COLOUR_SUCCESS],
        ],
        zmid=0,
        text=text_vals,
        texttemplate="%{text}",
        hovertemplate="Lookback: %{y}<br>Horizon: %{x}<br>Alpha: %{z:.3f}<extra></extra>",
        colorbar=dict(title="Alpha (£/MWh)"),
    ))
    return _apply_defaults(fig, title=title,
                           xaxis_title="Forecast Horizon",
                           yaxis_title="Lookback Window",
                           height=400)


def error_comparison_chart(
    horizons: List[str],
    forecast_mae: List[float],
    market_mae: List[float],
    title: str = "Forecast vs Market Error by Horizon",
) -> go.Figure:
    """Grouped bar chart: our MAE vs market MAE at each horizon."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=horizons, y=forecast_mae,
        name="Our Forecast", marker_color=COLOUR_PRIMARY,
    ))
    fig.add_trace(go.Bar(
        x=horizons, y=market_mae,
        name="Market (MIP)", marker_color=COLOUR_MUTED,
    ))
    fig.update_layout(barmode="group")
    return _apply_defaults(fig, title=title,
                           xaxis_title="Forecast Horizon",
                           yaxis_title="MAE (£/MWh)")


def cumulative_alpha_chart(
    timestamps,
    cum_alpha_values,
    title: str = "Cumulative Alpha Over Time",
) -> go.Figure:
    """Line chart of running cumulative alpha."""
    fig = go.Figure()
    colours = [COLOUR_SUCCESS if v >= 0 else COLOUR_DANGER for v in cum_alpha_values]
    fig.add_trace(go.Scatter(
        x=timestamps, y=cum_alpha_values,
        mode="lines",
        line=dict(color=COLOUR_SUCCESS, width=2),
        fill="tozeroy",
        fillcolor="rgba(46,204,113,0.1)",
        name="Cumulative Alpha",
    ))
    fig.add_hline(y=0, line_color="white", line_width=1, line_dash="dash")
    return _apply_defaults(fig, title=title,
                           xaxis_title="Time",
                           yaxis_title="Cumulative Alpha (£/MWh)")


def horizon_error_decay(
    horizon_labels: List[str],
    forecast_errors: List[float],
    market_errors: List[float],
    title: str = "Error Growth by Horizon",
) -> go.Figure:
    """
    Line chart showing how forecast error grows with horizon vs market error.
    The crossing point (if any) marks the maximum useful forecast horizon.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=horizon_labels, y=forecast_errors,
        mode="lines+markers",
        name="Our Forecast Error",
        line=dict(color=COLOUR_PRIMARY, width=2),
        marker=dict(size=8),
    ))
    fig.add_trace(go.Scatter(
        x=horizon_labels, y=market_errors,
        mode="lines+markers",
        name="Market (MIP) Error",
        line=dict(color=COLOUR_MUTED, width=2, dash="dash"),
        marker=dict(size=8),
    ))
    return _apply_defaults(fig, title=title,
                           xaxis_title="Forecast Horizon",
                           yaxis_title="MAE (£/MWh)")


# ══════════════════════════════════════════════════════════════════════════
#  Residual validation charts
# ══════════════════════════════════════════════════════════════════════════

def residual_time_series_chart(
    timestamps,
    actual_residuals: np.ndarray,
    predicted_residuals: np.ndarray,
    title: str = "Actual vs Predicted Residuals",
) -> go.Figure:
    """
    Overlay actual residuals (forecast - realised) with the walk-forward
    predicted residuals.  Where the two lines track each other, the model
    has exploitable systematic bias.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=actual_residuals,
        mode="lines", name="Actual Residual",
        line=dict(color=COLOUR_PRIMARY, width=1),
        opacity=0.6,
    ))
    valid = ~np.isnan(predicted_residuals)
    ts_valid = [t for t, v in zip(timestamps, valid) if v]
    pr_valid = predicted_residuals[valid]
    fig.add_trace(go.Scatter(
        x=ts_valid, y=pr_valid,
        mode="lines", name="Predicted Residual (EWMA)",
        line=dict(color=COLOUR_WARNING, width=2),
    ))
    fig.add_hline(y=0, line_color="white", line_width=1, line_dash="dash")
    return _apply_defaults(fig, title=title,
                           xaxis_title="Time",
                           yaxis_title="Residual (£/MWh) — positive = over-predict")


def residual_scatter_chart(
    actual_residuals: np.ndarray,
    predicted_residuals: np.ndarray,
    title: str = "Predicted vs Actual Residuals",
) -> go.Figure:
    """
    Scatter plot of predicted residual (x) vs actual residual (y).
    Perfect prediction sits on the diagonal.
    """
    valid = ~np.isnan(predicted_residuals)
    actual_v = actual_residuals[valid]
    pred_v = predicted_residuals[valid]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pred_v, y=actual_v,
        mode="markers",
        marker=dict(color=COLOUR_ACCENT, size=4, opacity=0.4),
        name="Observations",
    ))
    all_vals = np.concatenate([actual_v, pred_v])
    lo, hi = float(np.percentile(all_vals, 2)), float(np.percentile(all_vals, 98))
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi],
        mode="lines", name="Perfect prediction",
        line=dict(color="white", width=1, dash="dash"),
    ))
    return _apply_defaults(fig, title=title,
                           xaxis_title="Predicted Residual (£/MWh)",
                           yaxis_title="Actual Residual (£/MWh)")


def correction_improvement_chart(
    horizon_labels: List[str],
    original_mae: List[float],
    corrected_mae: List[float],
    title: str = "Bias-Corrected vs Original Forecast Error",
) -> go.Figure:
    """
    Grouped bar chart comparing original MAE with the bias-corrected MAE
    (original forecast minus predicted residual).
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=horizon_labels, y=original_mae,
        name="Original Forecast", marker_color=COLOUR_PRIMARY,
    ))
    fig.add_trace(go.Bar(
        x=horizon_labels, y=corrected_mae,
        name="Bias-Corrected", marker_color=COLOUR_SUCCESS,
    ))
    fig.update_layout(barmode="group")
    return _apply_defaults(fig, title=title,
                           xaxis_title="Forecast Horizon",
                           yaxis_title="MAE (£/MWh)")


def confirmation_hit_rate_chart(
    horizon_labels: List[str],
    confirmed_hr: List[float],
    unconfirmed_hr: List[float],
    title: str = "Hit Rate: Confirmed vs Unconfirmed Positions",
) -> go.Figure:
    """
    Grouped bar chart showing hit rate when the residual signal
    confirms the position vs when it contradicts.
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=horizon_labels, y=confirmed_hr,
        name="Residual Confirms", marker_color=COLOUR_SUCCESS,
    ))
    fig.add_trace(go.Bar(
        x=horizon_labels, y=unconfirmed_hr,
        name="Residual Contradicts", marker_color=COLOUR_DANGER,
    ))
    fig.add_hline(y=0.5, line_color="white", line_width=1, line_dash="dash",
                  annotation_text="50% baseline")
    fig.update_layout(barmode="group")
    return _apply_defaults(fig, title=title,
                           xaxis_title="Forecast Horizon",
                           yaxis_title="Hit Rate (vs Market)")
