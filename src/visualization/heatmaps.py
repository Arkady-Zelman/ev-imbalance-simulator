"""
Heatmap and scenario-comparison visualisations.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import (
    COLOUR_DANGER,
    COLOUR_PRIMARY,
    COLOUR_SUCCESS,
    PLOTLY_TEMPLATE,
    SP_LABELS,
)

_LAYOUT_DEFAULTS = dict(
    template=PLOTLY_TEMPLATE,
    margin=dict(l=50, r=30, t=50, b=50),
    font=dict(size=12),
)


# ── Time-of-day risk heatmap ─────────────────────────────────────────────

def time_of_day_heatmap(
    values: np.ndarray,
    title: str = "Imbalance Cost by Settlement Period",
    value_label: str = "Avg Imbalance Cost (£)",
) -> go.Figure:
    """
    Single-row heatmap (48 cells) coloured by severity.
    `values` should be shape (48,).
    """
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=[values],
        x=SP_LABELS,
        y=[value_label],
        colorscale=[
            [0.0, "#2ECC71"],
            [0.3, "#F1C40F"],
            [0.6, "#E67E22"],
            [1.0, "#E74C3C"],
        ],
        colorbar=dict(title="£"),
        text=[[f"£{v:,.0f}" for v in values]],
        texttemplate="%{text}",
        hovertemplate="SP %{x}<br>%{z:,.0f} £<extra></extra>",
    ))
    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title=title,
        xaxis_title="Settlement Period",
        height=200,
    )
    return fig


# ── Scenario side-by-side ────────────────────────────────────────────────

def scenario_side_by_side(
    pnl_a: np.ndarray,
    pnl_b: np.ndarray,
    label_a: str = "Benign",
    label_b: str = "Stressed",
    title: str = "Scenario Comparison: P&L Distributions",
) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=(label_a, label_b))
    fig.add_trace(
        go.Histogram(x=pnl_a, nbinsx=60, marker_color=COLOUR_SUCCESS, opacity=0.7),
        row=1, col=1,
    )
    fig.add_trace(
        go.Histogram(x=pnl_b, nbinsx=60, marker_color=COLOUR_DANGER, opacity=0.7),
        row=1, col=2,
    )
    fig.update_layout(**_LAYOUT_DEFAULTS, title=title, showlegend=False)
    fig.update_xaxes(title_text="Daily P&L (£)")
    fig.update_yaxes(title_text="Frequency")
    return fig


# ── Imbalance box plots per SP ───────────────────────────────────────────

def imbalance_boxplots(
    imbalance_mw: np.ndarray,
    title: str = "Imbalance Volume by Settlement Period",
) -> go.Figure:
    """imbalance_mw shape: (n_runs, 48). Positive = short."""
    fig = go.Figure()
    n_runs = imbalance_mw.shape[0]
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(n_runs, min(n_runs, 500), replace=False)
    sampled = imbalance_mw[sample_idx, :]

    for sp in range(48):
        fig.add_trace(go.Box(
            y=sampled[:, sp],
            name=SP_LABELS[sp],
            marker_color=COLOUR_PRIMARY,
            showlegend=False,
            boxpoints=False,
        ))
    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title=title,
        xaxis_title="Settlement Period",
        yaxis_title="Imbalance (MW) — positive = short",
        height=400,
    )
    return fig


# ── Rolling statistics chart ─────────────────────────────────────────────

def rolling_stats_chart(
    dates,
    values: np.ndarray,
    window_label: str = "30-day",
    title: str = "Rolling SIP Statistics",
) -> go.Figure:
    s = pd.Series(values, index=pd.to_datetime(dates))
    rolling_mean = s.rolling(30, min_periods=5).mean()
    rolling_std = s.rolling(30, min_periods=5).std()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_mean.index, y=rolling_mean.values,
        mode="lines", name=f"{window_label} Mean",
        line=dict(color=COLOUR_PRIMARY, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=rolling_std.index, y=rolling_std.values,
        mode="lines", name=f"{window_label} Volatility",
        line=dict(color=COLOUR_DANGER, width=2),
    ))
    fig.update_layout(**_LAYOUT_DEFAULTS, title=title,
                      xaxis_title="Date", yaxis_title="£/MWh")
    return fig
