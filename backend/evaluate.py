"""
Model diagnostics Streamlit app — run separately from the main frontend.

Run:
    streamlit run backend/evaluate.py

Shows:
  - Fan charts: retrospective predictions vs actuals vs market forward
  - HPO heatmap: best validation MAE per (lookback, horizon) cell
  - Train vs validation MAE gap (overfitting diagnostic)
  - Alpha crossover horizon per lookback
  - Hybrid weights: XGB vs LSTM contribution per target
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import FORECAST_TARGETS, MODEL_DIR, PREDICTION_DIR

st.set_page_config(
    page_title="Ohme — Model Diagnostics",
    page_icon="🔬",
    layout="wide",
)

PLOTLY_TEMPLATE = "plotly_dark"

# ── Load helpers ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_parquet(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _load_metadata() -> dict:
    path = PREDICTION_DIR / "metadata.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_resource
def _load_artifact(target: str):
    import joblib
    path = MODEL_DIR / f"hybrid_{target}.joblib"
    if path.exists():
        return joblib.load(path)
    return None


# ── Section renderers ─────────────────────────────────────────────────────────

def _render_fan_chart(fan_df: pd.DataFrame, target: str, lookback: str) -> None:
    sub = fan_df[
        (fan_df["target"]   == target) &
        (fan_df["lookback"] == lookback)
    ].copy()

    if sub.empty:
        st.info("No fan data available for this combination.")
        return

    origin_dates = sorted(sub["origin_date"].unique())

    selected_origin = st.selectbox(
        "Select origin date",
        options=origin_dates,
        format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"),
        key=f"fan_origin_{target}_{lookback}",
    )

    row = sub[sub["origin_date"] == selected_origin].sort_values("horizon_days")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=row["horizon_days"], y=row["hybrid_prediction"],
        mode="lines+markers", name="Our Forecast",
        line=dict(color="#00D4AA", dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=row["horizon_days"], y=row["forward_curve_value"],
        mode="lines+markers", name="Market Forward",
        line=dict(color="#4ECDC4"),
    ))
    fig.add_trace(go.Scatter(
        x=row["horizon_days"], y=row["realised_value"],
        mode="lines+markers", name="Realised",
        line=dict(color="#FFE66D"),
    ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=f"Fan Chart — {target.upper()} | Origin: {pd.Timestamp(selected_origin).strftime('%Y-%m-%d')} | Lookback: {lookback}",
        xaxis_title="Horizon (days)",
        yaxis_title="Value",
        height=400,
    )
    st.plotly_chart(fig, width="stretch")


def _render_hpo_heatmap(artifact: dict, target: str) -> None:
    if artifact is None:
        st.warning(f"No trained model found for {target}.")
        return

    xgb_trained = artifact.get("xgb")
    if xgb_trained is None:
        return

    best_scores: dict = getattr(xgb_trained, "best_scores", {})
    if not best_scores:
        st.info("No grid search scores available.")
        return

    rows = []
    for lb_label, h_scores in best_scores.items():
        for h_sps, score in h_scores.items():
            rows.append({
                "Lookback": lb_label,
                "Horizon": f"{int(h_sps) // 48}d",
                "Horizon SPs": int(h_sps),
                "Val MAE": score,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return

    pivot = df.pivot_table(index="Lookback", columns="Horizon", values="Val MAE", aggfunc="first")

    # Order columns by horizon
    horizon_order = sorted(pivot.columns, key=lambda x: int(x.replace("d", "")))
    pivot = pivot[horizon_order]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="RdYlGn_r",
        text=np.round(pivot.values, 2).astype(str),
        texttemplate="%{text}",
        colorbar=dict(title="Val MAE"),
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=f"XGBoost HPO — Validation MAE per (Lookback × Horizon) — {target.upper()}",
        xaxis_title="Horizon",
        yaxis_title="Lookback",
        height=350,
    )
    st.plotly_chart(fig, width="stretch")


def _render_overfitting_gap(artifact: dict, target: str) -> None:
    if artifact is None:
        return

    xgb_trained  = artifact.get("xgb")
    lstm_trained = artifact.get("lstm")

    rows = []
    for label, trained in [("XGBoost", xgb_trained), ("LSTM", lstm_trained)]:
        if trained is None:
            continue
        train_scores = getattr(trained, "train_scores", {})
        best_scores  = getattr(trained, "best_scores",  {})
        for lb in train_scores:
            for h_sps in train_scores[lb]:
                train_mae = train_scores[lb][h_sps]
                val_mae   = (best_scores.get(lb) or {}).get(h_sps)
                if train_mae is not None and val_mae is not None:
                    rows.append({
                        "Model":    label,
                        "Lookback": lb,
                        "Horizon":  f"{int(h_sps) // 48}d",
                        "Train MAE": train_mae,
                        "Val MAE":   val_mae,
                        "Gap":       val_mae - train_mae,
                    })

    if not rows:
        st.info("No overfitting diagnostics available.")
        return

    df = pd.DataFrame(rows)
    st.dataframe(df.style.background_gradient(subset=["Gap"], cmap="RdYlGn_r"), width="stretch")


def _render_weights(metadata: dict) -> None:
    rows = []
    for target in FORECAST_TARGETS:
        tmeta = metadata.get("targets", {}).get(target, {})
        weights = tmeta.get("hybrid_weights", {})
        rows.append({
            "Target":      target,
            "XGB Weight":  weights.get("xgb", 0.5),
            "LSTM Weight": weights.get("lstm", 0.5),
            "XGB Val MAE": tmeta.get("xgb_val_mae"),
            "LSTM Val MAE": tmeta.get("lstm_val_mae"),
            "Alpha":       "YES" if tmeta.get("alpha_status") else "NO",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["Target"], y=df["XGB Weight"],
        name="XGB", marker_color="#00D4AA",
    ))
    fig.add_trace(go.Bar(
        x=df["Target"], y=df["LSTM Weight"],
        name="LSTM", marker_color="#4ECDC4",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        barmode="stack",
        title="Hybrid Ensemble Weights per Target",
        yaxis_title="Weight",
        height=350,
    )
    st.plotly_chart(fig, width="stretch")


def _render_alpha_crossover(artifact: dict, target: str) -> None:
    if artifact is None:
        return

    xgb_trained = artifact.get("xgb")
    backtest_crossovers = getattr(xgb_trained, "backtest_crossovers", [])

    if not backtest_crossovers:
        st.info("No backtest crossover data available.")
        return

    rows = [
        {
            "Lookback": c.lookback_label,
            "Crossover Day": c.crossover_day,
            "Last +α MAE": round(c.last_positive_alpha, 4),
            "First -α MAE": round(c.first_negative_alpha, 4),
        }
        for c in backtest_crossovers
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch")


# ── App layout ────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("Ohme Fleet Trading — Model Diagnostics")

    metadata = _load_metadata()
    fan_df   = _load_parquet(PREDICTION_DIR / "backtest_fan.parquet")

    if metadata:
        gen_at = metadata.get("generated_at", "unknown")
        data_range = metadata.get("data_range", {})
        st.caption(
            f"Predictions generated: **{gen_at}** | "
            f"Data range: {data_range.get('from', '?')} → {data_range.get('to', '?')}"
        )

    # ── Weights overview ──────────────────────────────────────────────────────
    with st.expander("Hybrid Ensemble Weights & Alpha Status", expanded=True):
        if metadata:
            _render_weights(metadata)
        else:
            st.warning("metadata.json not found — run `python -m backend.predict` first.")

    # ── Per-target diagnostics ────────────────────────────────────────────────
    target = st.selectbox(
        "Select target",
        options=FORECAST_TARGETS,
        format_func=lambda t: {
            "sip": "SIP (Imbalance Price)",
            "mip": "MIP (Wholesale)",
            "demand": "Demand",
            "total_generation": "Total Generation",
        }.get(t, t),
    )

    artifact = _load_artifact(target)

    tab_fan, tab_hpo, tab_overfit, tab_crossover = st.tabs([
        "Fan Charts", "HPO Heatmap", "Overfitting Gap", "Alpha Crossover"
    ])

    with tab_fan:
        st.subheader("Retrospective Fan Chart")
        if fan_df.empty:
            st.warning("backtest_fan.parquet not found — run `python -m backend.predict` first.")
        else:
            lookback = st.selectbox(
                "Lookback window",
                options=["1 day", "3 days", "5 days", "15 days"],
                key=f"fan_lookback_{target}",
            )
            _render_fan_chart(fan_df, target, lookback)

    with tab_hpo:
        st.subheader("XGBoost HPO — Validation MAE Heatmap")
        _render_hpo_heatmap(artifact, target)

    with tab_overfit:
        st.subheader("Train vs Validation MAE (Overfitting Diagnostic)")
        _render_overfitting_gap(artifact, target)

    with tab_crossover:
        st.subheader("Alpha Crossover Horizon")
        st.caption("The horizon day where our forecast stops beating the market benchmark.")
        _render_alpha_crossover(artifact, target)


if __name__ == "__main__":
    main()
