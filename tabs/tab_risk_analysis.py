"""Risk Analysis tab -- frontier, heatmap, capture ratio, Kelly analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import COLOUR_PRIMARY, COLOUR_SUCCESS, COLOUR_WARNING, COLOUR_MUTED, PLOTLY_TEMPLATE, RISK_APPETITES, SP_LABELS
from src.models.kelly import compute_kelly_pnl, run_kelly_analysis
from src.models.pnl_calculator import compute_pnl_for_position
from src.models.risk_metrics import compute_cvar
from src.session_keys import (
    ALL_POSITIONS, BANKROLL, CAPTURE_RATIOS, DA_PRICE,
    KELLY_RESULTS, PARAMS, RESULT, SIZING_METHOD, SIP_MATRIX,
)
from src.visualization.charts import (
    capture_ratio_histogram,
    risk_return_frontier,
    sharpe_comparison,
)
from src.visualization.heatmaps import time_of_day_heatmap


def render(has_results: bool) -> None:
    st.header("Risk Analysis")

    if not has_results:
        st.info("Run a simulation first.")
        return

    result = st.session_state[RESULT]
    sip_matrix = st.session_state[SIP_MATRIX]
    da_price = st.session_state[DA_PRICE]
    params = st.session_state[PARAMS]

    # ── Compute metrics for each risk tier ────────────────────────────
    labels, expected_pnls, cvars, rtr = [], [], [], []

    for tier_label, tier_pct in RISK_APPETITES.items():
        traded = np.percentile(result.delivered_mw, tier_pct, axis=0)
        pnl = compute_pnl_for_position(
            result.delivered_mw, traded,
            sip_matrix, da_price, params.n_runs,
        )
        labels.append(tier_label)
        expected_pnls.append(float(np.mean(pnl)))
        cvars.append(compute_cvar(pnl))
        std = float(np.std(pnl))
        rtr.append(float(np.mean(pnl)) / max(std, 1e-9))

    # ── Risk-Return Frontier ──────────────────────────────────────────
    st.subheader("Risk-Return Frontier")
    st.caption("Each point represents a different position-sizing percentile. "
               "Move right for more tail risk, up for higher expected return.")
    fig_frontier = risk_return_frontier(labels, expected_pnls, cvars)
    st.plotly_chart(fig_frontier, use_container_width=True)

    # ── Metrics table ─────────────────────────────────────────────────
    st.subheader("Metrics by Position Size")
    metrics_df = pd.DataFrame({
        "Tier": labels,
        "Expected P&L (£)": [f"£{v:,.0f}" for v in expected_pnls],
        "CVaR 95% (£)": [f"£{v:,.0f}" for v in cvars],
        "Reward-to-Risk": [f"{v:.3f}" for v in rtr],
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # ── Time-of-day risk heatmap ──────────────────────────────────────
    st.markdown("---")
    st.subheader("Time-of-Day Risk Heatmap")
    st.caption("Average imbalance cost per settlement period across all simulation runs. "
               "Red indicates high risk -- typically the evening peak.")
    avg_imb_cost = result.sp_imbalance_cost.mean(axis=0)
    fig_heat = time_of_day_heatmap(avg_imb_cost)
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Capture ratio ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Capture Ratio Distribution")
    st.caption("Ratio of actual revenue to benchmark revenue. "
               "Values below 1.0 indicate value lost to imbalance costs.")
    capture = st.session_state[CAPTURE_RATIOS]
    fig_cap = capture_ratio_histogram(capture)
    st.plotly_chart(fig_cap, use_container_width=True)

    cap_stats = pd.DataFrame({
        "Metric": ["Mean", "Median", "P5 (worst 5%)", "P95", "< 0.9 frequency"],
        "Value": [
            f"{np.mean(capture):.4f}",
            f"{np.median(capture):.4f}",
            f"{np.percentile(capture, 5):.4f}",
            f"{np.percentile(capture, 95):.4f}",
            f"{(capture < 0.9).mean():.1%}",
        ],
    })
    st.dataframe(cap_stats, use_container_width=True, hide_index=True)

    # ── Reward-to-Risk comparison bar chart ───────────────────────────
    st.markdown("---")
    st.subheader("Reward-to-Risk Comparison")
    st.caption("Mean(P&L) / Std(P&L) — within-simulation signal-to-noise, not an annualised Sharpe ratio.")
    fig_sharpe = sharpe_comparison(labels, rtr)
    st.plotly_chart(fig_sharpe, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════
    # Kelly Criterion Analysis
    # ══════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("Kelly Criterion Position Sizing")
    st.caption(
        "Kelly maximises E[log(1+fR)] — the long-run geometric growth rate — "
        "accounting for the DA revenue upside vs SIP shortfall downside asymmetry. "
        "Fractional Kelly trades growth for drawdown reduction."
    )

    b = st.session_state.get(BANKROLL, 100_000.0)
    kelly_results = st.session_state.get(KELLY_RESULTS)

    # Always compute Kelly for comparison, even if percentile was selected
    if kelly_results is None:
        with st.spinner("Computing Kelly analysis…"):
            kelly_results = run_kelly_analysis(
                result.delivered_mw, st.session_state[SIP_MATRIX],
                da_price=da_price, bankroll=b,
            )

    if kelly_results:
        # ── Summary table ─────────────────────────────────────────────
        kr_rows = []
        for kr in kelly_results:
            kr_rows.append({
                "Fraction": kr.label,
                "E[P&L]": f"£{kr.expected_daily_pnl:,.0f}",
                "Std P&L": f"£{kr.std_daily_pnl:,.0f}",
                "Growth Rate": f"{kr.growth_rate:.6f}",
                "Reward/Risk": f"{kr.expected_daily_pnl / max(kr.std_daily_pnl, 1):,.3f}",
                "Max Commit (MW)": f"{kr.max_commitment_mw:.1f}",
                "P(Shortfall)": f"{kr.mean_shortfall_probability:.1%}",
            })
        st.dataframe(pd.DataFrame(kr_rows), use_container_width=True, hide_index=True)

        # ── Percentile vs Kelly comparison ────────────────────────────
        st.markdown("##### Percentile vs Kelly: Per-SP Commitment")
        fig_cmp = go.Figure()

        # Plot current percentile position
        fig_cmp.add_trace(go.Scatter(
            x=SP_LABELS, y=result.traded_mw,
            mode="lines", name=f"Current (P{params.risk_percentile})",
            line=dict(color=COLOUR_MUTED, width=2, dash="dash"),
        ))

        kelly_colours = [COLOUR_SUCCESS, COLOUR_PRIMARY, COLOUR_WARNING, "#E74C3C"]
        for i, kr in enumerate(kelly_results):
            fig_cmp.add_trace(go.Scatter(
                x=SP_LABELS, y=kr.optimal_mw,
                mode="lines", name=kr.label,
                line=dict(color=kelly_colours[i % len(kelly_colours)], width=2),
            ))

        fig_cmp.update_layout(
            template=PLOTLY_TEMPLATE,
            margin=dict(l=50, r=30, t=50, b=50),
            title="Committed MW per Settlement Period: Kelly vs Percentile",
            xaxis_title="Settlement Period",
            yaxis_title="Committed MW",
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        # ── Growth rate vs shortfall probability ──────────────────────
        st.markdown("##### Growth Rate vs Shortfall Risk")
        fig_gr = go.Figure()
        fracs = [kr.fraction for kr in kelly_results]
        growth_rates = [kr.growth_rate for kr in kelly_results]
        shortfall_probs = [kr.mean_shortfall_probability for kr in kelly_results]

        fig_gr.add_trace(go.Bar(
            x=[kr.label for kr in kelly_results],
            y=growth_rates,
            name="Growth Rate",
            marker_color=COLOUR_SUCCESS,
            yaxis="y",
        ))
        fig_gr.add_trace(go.Scatter(
            x=[kr.label for kr in kelly_results],
            y=shortfall_probs,
            name="P(Shortfall)",
            mode="lines+markers",
            line=dict(color="#E74C3C", width=2),
            marker=dict(size=10),
            yaxis="y2",
        ))
        fig_gr.update_layout(
            template=PLOTLY_TEMPLATE,
            margin=dict(l=50, r=60, t=50, b=50),
            title="Kelly Fraction Trade-off: Growth vs Shortfall Risk",
            yaxis=dict(title="E[log(1+R)]", side="left"),
            yaxis2=dict(title="P(Shortfall)", side="right", overlaying="y"),
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_gr, use_container_width=True)

        with st.expander("Kelly Criterion — How It Works"):
            st.markdown("""
**The core idea:** Instead of committing an arbitrary percentile of expected availability,
Kelly finds the commitment level that maximises the **long-run geometric growth rate** of
your trading P&L: `f* = argmax E[log(1 + f·R)]`.

**Why it matters for EV flexibility:**
- The **upside is bounded** — you earn DA price per committed MW
- The **downside is unbounded** — SIP can spike to thousands of £/MWh on shortfall
- Kelly naturally **commits less** when SIP tail risk is severe (evening peak) and
  **more** when the risk-reward is favorable (overnight, benign SIP)

**Fractional Kelly:**
| Fraction | Growth Rate | Drawdown Variance | Use Case |
|----------|------------|-------------------|----------|
| ¼ Kelly | ~44% of full | ~6% of full | Very conservative desk |
| ½ Kelly | ~75% of full | ~25% of full | **Industry standard** |
| ¾ Kelly | ~94% of full | ~56% of full | Aggressive but disciplined |
| Full Kelly | 100% | 100% | Theoretically optimal, high variance |
""")
