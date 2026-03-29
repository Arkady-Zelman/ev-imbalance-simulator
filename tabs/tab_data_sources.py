"""Data Sources & Methodology tab -- full provenance, formulas, references."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from src.config import (
    CHARGER_CAPACITY_KW,
    CORRELATION_DECAY,
    DAYTYPE_MULTIPLIERS,
    DEFAULT_DA_PRICE,
    DEFAULT_DISPATCH_SUCCESS_RATE,
    DEFAULT_FLEET_SIZE,
    DEFAULT_OVERRIDE_RATE,
    ELEXON_BASE_URL,
    NUM_SETTLEMENT_PERIODS,
    PLUGIN_CLUSTERS,
    SEASONAL_MONTHLY,
    SIP_REGIME_DEFAULTS,
    SIP_STRESS_DISPATCH_PENALTY,
    SIP_STRESS_PLUGIN_FACTOR,
)
from src.session_keys import DATE_FROM, DATE_TO


def render() -> None:
    st.header("Data Sources & Methodology")

    # ── 1. Live Data Sources ──────────────────────────────────────────
    st.subheader("1. Live Data Sources")

    sources = pd.DataFrame([
        {
            "Data": "System Imbalance Price (SIP)",
            "Provider": "ELEXON Insights API",
            "Endpoint": f"`GET {ELEXON_BASE_URL}/balancing/settlement/system-prices/{{date}}`",
            "API Key Required": "No (public)",
            "Update Frequency": "Half-hourly (48 per day)",
            "Format": "JSON",
            "Fields Used": "settlementDate, settlementPeriod, systemBuyPrice, systemSellPrice, netImbalanceVolume",
        },
        {
            "Data": "Market Index Price (MIP)",
            "Provider": "ELEXON Insights API",
            "Endpoint": f"`GET {ELEXON_BASE_URL}/balancing/pricing/market-index?from=...&to=...`",
            "API Key Required": "No (public)",
            "Update Frequency": "Half-hourly",
            "Format": "JSON",
            "Fields Used": "settlementDate, settlementPeriod, price, volume, dataProvider",
        },
        {
            "Data": "Net Imbalance Volume (NIV)",
            "Provider": "ELEXON Insights API",
            "Endpoint": "Via system-prices endpoint (`netImbalanceVolume` field)",
            "API Key Required": "No (public)",
            "Update Frequency": "Half-hourly",
            "Format": "JSON (nested)",
            "Fields Used": "netImbalanceVolume",
        },
    ])
    st.dataframe(sources, use_container_width=True, hide_index=True)

    st.markdown(f"**Base URL:** `{ELEXON_BASE_URL}`")

    with st.expander("Example API Calls (Python)"):
        st.code("""
import requests

# System Imbalance Prices — path-based, one date at a time
url = "https://data.elexon.co.uk/bmrs/api/v1/balancing/settlement/system-prices/2025-12-01"
response = requests.get(url)
data = response.json()["data"]

# Market Index Price — query-based datetime range
url = "https://data.elexon.co.uk/bmrs/api/v1/balancing/pricing/market-index"
params = {"from": "2025-12-01T00:00Z", "to": "2025-12-02T00:00Z", "format": "json"}
response = requests.get(url, params=params)
data = response.json()["data"]
""", language="python")

    # ── Data freshness ────────────────────────────────────────────────
    if DATE_FROM in st.session_state and DATE_TO in st.session_state:
        st.markdown("---")
        st.subheader("Data Freshness")
        from src.data.elexon_client import sip_cache_timestamp, mip_cache_timestamp
        df = st.session_state[DATE_FROM]
        dt_to = st.session_state[DATE_TO]
        sip_ts = sip_cache_timestamp(df, dt_to)
        mip_ts = mip_cache_timestamp(df, dt_to)

        freshness = []
        for name, ts in [("SIP", sip_ts), ("Market Index", mip_ts)]:
            if ts:
                fetched = dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
            else:
                fetched = "Not cached (fetched live)"
            freshness.append({"Dataset": name, "Last Fetched": fetched,
                              "Date Range": f"{df} → {dt_to}"})
        st.dataframe(pd.DataFrame(freshness), use_container_width=True, hide_index=True)

    # ── 2. Fallback / Dummy Data ──────────────────────────────────────
    st.markdown("---")
    st.subheader("2. Fallback Behaviour When API Fails")
    st.warning(
        f"If the ELEXON API returns no parseable SIP data, the simulation falls back "
        f"to a **flat £{DEFAULT_DA_PRICE}/MWh dummy matrix** with zero volatility. "
        f"A prominent warning banner is displayed when this happens. Results under "
        f"fallback are unrealistically benign and should not be used for trading decisions."
    )

    # ── 3. Hardcoded Assumptions ──────────────────────────────────────
    st.markdown("---")
    st.subheader("3. Hardcoded Model Assumptions — Complete Audit")

    st.markdown(
        "The table below lists **every hardcoded assumption** in the model, "
        "its default value, data source (or lack thereof), and whether it can "
        "be changed via the sidebar UI."
    )

    assumptions = pd.DataFrame([
        {
            "Parameter": "Charger capacity",
            "Default": f"{CHARGER_CAPACITY_KW} kW",
            "Source": "Ohme Home Pro hardware specification",
            "Configurable?": "No (fixed hardware)",
            "Location": "src/config.py",
        },
        {
            "Parameter": "Settlement periods per day",
            "Default": str(NUM_SETTLEMENT_PERIODS),
            "Source": "GB electricity market structure (half-hourly)",
            "Configurable?": "No (market rule)",
            "Location": "src/config.py",
        },
        {
            "Parameter": "Plug-in rate: Overnight mean",
            "Default": f"{PLUGIN_CLUSTERS[0].mean:.0%}",
            "Source": "Estimated from CrowdFlex trial baselines (~50% pre-engagement, ~70% post-engagement overnight)",
            "Configurable?": "Yes — sidebar expander",
            "Location": "src/config.py → PLUGIN_CLUSTERS",
        },
        {
            "Parameter": "Plug-in rate: Morning departure mean",
            "Default": f"{PLUGIN_CLUSTERS[1].mean:.0%}",
            "Source": "Estimated from CrowdFlex (vehicles leave for commute)",
            "Configurable?": "Yes — sidebar expander",
            "Location": "src/config.py → PLUGIN_CLUSTERS",
        },
        {
            "Parameter": "Plug-in rate: Daytime mean",
            "Default": f"{PLUGIN_CLUSTERS[2].mean:.0%}",
            "Source": "Estimated from CrowdFlex (~18-28% daytime availability)",
            "Configurable?": "Yes — sidebar expander",
            "Location": "src/config.py → PLUGIN_CLUSTERS",
        },
        {
            "Parameter": "Plug-in rate: Evening peak mean",
            "Default": f"{PLUGIN_CLUSTERS[3].mean:.0%}",
            "Source": "Estimated from CrowdFlex (vehicles returning home)",
            "Configurable?": "Yes — sidebar expander",
            "Location": "src/config.py → PLUGIN_CLUSTERS",
        },
        {
            "Parameter": "Plug-in rate: Late evening mean",
            "Default": f"{PLUGIN_CLUSTERS[4].mean:.0%}",
            "Source": "Estimated from CrowdFlex",
            "Configurable?": "Yes — sidebar expander",
            "Location": "src/config.py → PLUGIN_CLUSTERS",
        },
        {
            "Parameter": "Plug-in rate concentrations (ν)",
            "Default": ", ".join(str(int(c.concentration)) for c in PLUGIN_CLUSTERS),
            "Source": "Educated guess — higher ν = more confidence in the mean",
            "Configurable?": "Yes — sidebar expander",
            "Location": "src/config.py → PLUGIN_CLUSTERS",
        },
        {
            "Parameter": "Day-type multipliers",
            "Default": f"weekday={DAYTYPE_MULTIPLIERS['weekday']}, "
                       f"weekend={DAYTYPE_MULTIPLIERS['weekend']}, "
                       f"holiday={DAYTYPE_MULTIPLIERS['holiday']}",
            "Source": "Estimated — no public dataset. Assumption: ~12% more vehicles "
                      "plugged in on weekends/holidays (no commute)",
            "Configurable?": "Yes — sidebar expander",
            "Location": "src/config.py → DAYTYPE_MULTIPLIERS",
        },
        {
            "Parameter": "Seasonal monthly factors",
            "Default": ", ".join(f"{SEASONAL_MONTHLY[m]}" for m in range(1, 13)),
            "Source": "Estimated — no public dataset. Assumption: winter higher "
                      "(vehicles at home more, longer evenings), summer lower (travel)",
            "Configurable?": "Yes — sidebar expander",
            "Location": "src/config.py → SEASONAL_MONTHLY",
        },
        {
            "Parameter": "SIP-stress plug-in factor",
            "Default": str(SIP_STRESS_PLUGIN_FACTOR),
            "Source": "Engineering estimate — no public data. Concept: system stress "
                      "(cold snaps) also affects EV usage patterns",
            "Configurable?": "Yes — sidebar expander",
            "Location": "src/config.py",
        },
        {
            "Parameter": "SIP-stress dispatch penalty",
            "Default": str(SIP_STRESS_DISPATCH_PENALTY),
            "Source": "Engineering estimate — dispatch failures may rise under grid stress",
            "Configurable?": "Yes — sidebar expander",
            "Location": "src/config.py",
        },
        {
            "Parameter": "SIP regime-switching params",
            "Default": f"Normal μ=£{SIP_REGIME_DEFAULTS.normal_mean}, "
                       f"σ=£{SIP_REGIME_DEFAULTS.normal_std} | "
                       f"Spike P={SIP_REGIME_DEFAULTS.spike_probability:.0%}, "
                       f"μ_log={SIP_REGIME_DEFAULTS.spike_mean_log}, "
                       f"σ_log={SIP_REGIME_DEFAULTS.spike_std_log}",
            "Source": "Now **fitted from fetched ELEXON data** when Regime-Switching "
                      "mode is selected. Defaults used only as fallback if data is insufficient.",
            "Configurable?": "Auto-fitted from data",
            "Location": "src/config.py → SIPRegimeParams (fallback)",
        },
        {
            "Parameter": "DA price default",
            "Default": f"£{DEFAULT_DA_PRICE}/MWh",
            "Source": "Rough GB annual average. When MIP data is available, the "
                      "system suggests a data-derived value.",
            "Configurable?": "Yes — sidebar input",
            "Location": "src/config.py",
        },
        {
            "Parameter": "DA price noise (lognormal σ)",
            "Default": "0.05 (±5%)",
            "Source": "Assumption — models day-to-day DA auction variance. "
                      "No public source; calibrated to approximate GB day-ahead "
                      "price variability over short windows.",
            "Configurable?": "Yes — sidebar slider",
            "Location": "SimulationParams.da_noise_sigma",
        },
        {
            "Parameter": "Dispatch success rate default",
            "Default": f"{DEFAULT_DISPATCH_SUCCESS_RATE:.0%}",
            "Source": "Ohme engineering target for commercial operations",
            "Configurable?": "Yes — sidebar slider",
            "Location": "src/config.py",
        },
        {
            "Parameter": "Customer override rate default",
            "Default": f"{DEFAULT_OVERRIDE_RATE:.0%}",
            "Source": "CrowdFlex trial: opt-out rates were ~2-5% depending on incentive level",
            "Configurable?": "Yes — sidebar slider",
            "Location": "src/config.py",
        },
        {
            "Parameter": "Fleet size default",
            "Default": f"{DEFAULT_FLEET_SIZE:,}",
            "Source": "Representative mid-scale portfolio assumption",
            "Configurable?": "Yes — sidebar slider",
            "Location": "src/config.py",
        },
        {
            "Parameter": "Copula correlation decay",
            "Default": str(CORRELATION_DECAY),
            "Source": "Assumption — not fitted to data. Controls how quickly "
                      "inter-SP correlation falls off. 0.3 gives moderate correlation.",
            "Configurable?": "Yes — sidebar expander",
            "Location": "src/config.py",
        },
    ])
    st.dataframe(assumptions, use_container_width=True, hide_index=True)

    # ── 4. Methodology ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("4. Methodology")

    st.markdown("#### 4.1 Monte Carlo Simulation Framework")
    st.markdown("""
The simulator runs N independent Monte Carlo scenarios. For each scenario and each of the 48
half-hourly settlement periods, the model:

1. **Draws a plug-in rate** from a Beta distribution parameterised by time-of-day cluster
   *(configurable via sidebar)*
2. **Applies day-type and seasonal multipliers** to the Beta mean *(configurable via sidebar)*
3. **Applies SIP-stress coupling** — if SIP is in the top quintile, degrades plug-in rates
   and dispatch success *(configurable factors via sidebar)*
4. **Applies dispatch success** using a Normal approximation to the Binomial distribution
5. **Applies customer override** using a Normal approximation to the Binomial distribution
6. **Calculates delivered MW** = responding chargers × charger capacity
7. **Applies DA price noise** — lognormal jitter per run *(configurable σ via sidebar)*
8. **Computes imbalance** = traded MW − delivered MW
9. **Prices the imbalance** at the System Imbalance Price for that settlement period

The daily P&L is:

```
Daily P&L = Σ(Traded_MW × 0.5h × DA_Price_noisy) − Σ(Imbalance_MW × 0.5h × SIP)
```
""")

    st.markdown("#### 4.2 Plug-in Rate Model (Beta Distribution)")
    st.markdown("""
Plug-in rates are bounded [0, 1] and vary by time of day. The **Beta distribution** is the
natural choice for modelling proportions with varying uncertainty.

**Parameterisation:**
- α = mean × ν (concentration)
- β = (1 − mean) × ν
- Higher ν → tighter distribution (more confidence in the mean)

The default parameters below are inspired by Ohme's CrowdFlex trial baselines, where
pre-engagement overnight plug-in rates were around 50%, rising to ~70% with incentive
programmes. Daytime rates ranged from ~18% to ~28%.

**All 5 cluster means and concentrations are now configurable through the sidebar.**
""")

    cluster_data = []
    for c in PLUGIN_CLUSTERS:
        cluster_data.append({
            "Cluster": c.name,
            "SP Range": f"{c.sp_range[0]+1}–{c.sp_range[1]+1}",
            "Default Mean": f"{c.mean:.0%}",
            "Default ν": c.concentration,
            "α": f"{c.alpha:.1f}",
            "β": f"{c.beta_param:.1f}",
            "Configurable": "Yes",
        })
    st.dataframe(pd.DataFrame(cluster_data), use_container_width=True, hide_index=True)

    st.markdown("#### 4.3 Gaussian Copula (Correlation Structure)")
    st.markdown(f"""
Adjacent settlement periods are correlated — if 17:00 has low plug-in, 17:30 likely will too.
Independent draws would **understate tail risk**.

We use a **Gaussian copula** with an exponential-decay correlation matrix:

```
Corr(SP_i, SP_j) = exp(−decay × |i − j|)
```

Default decay = {CORRELATION_DECAY} *(configurable via sidebar)*. This is **not fitted to data** —
it is an assumption. Lower decay = stronger correlation between adjacent SPs.

Implementation:
1. Compute Cholesky decomposition L of the 48×48 correlation matrix (done once)
2. Draw 48 independent standard normals z for each run
3. Correlate: z_corr = L × z
4. Transform to uniform marginals: u = Φ(z_corr)
5. Transform to Beta marginals: plugin_rate = Beta⁻¹(u; α, β)
""")

    st.markdown("#### 4.4 SIP Modelling")
    st.markdown("""
**Empirical Bootstrap (default):**
- Samples full historical days (preserving intra-day autocorrelation)
- Each simulation run is assigned a randomly selected day's 48 SIP values
- Preserves the real distribution and fat tails
- Limitation: cannot generate scenarios worse than historical record

**Regime-Switching (fitted from data):**
When this mode is selected, the model now **fits** Normal/LogNormal parameters from your
fetched ELEXON SIP data using a percentile-based regime split (P95 threshold). This replaces
the previous behaviour of using hardcoded defaults and discarding live data.

Fallback defaults (used only if data is insufficient):
""")
    rd = SIP_REGIME_DEFAULTS
    regime_data = pd.DataFrame([
        {"Regime": "Normal (~95% of SPs)", "Distribution": "Normal",
         "Mean": f"£{rd.normal_mean}/MWh", "Std": f"£{rd.normal_std}/MWh"},
        {"Regime": "Spike (~5% of SPs)", "Distribution": "LogNormal",
         "Mean (log)": f"{rd.spike_mean_log}", "Std (log)": f"{rd.spike_std_log}"},
    ])
    st.dataframe(regime_data, use_container_width=True, hide_index=True)

    st.markdown("#### 4.5 Risk Metrics")
    st.markdown("""
| Metric | Definition | Formula |
|--------|-----------|---------|
| **VaR (95%)** | On 95% of days, P&L will not be worse than this | 5th percentile of P&L distribution |
| **CVaR (95%)** | Expected loss on the worst 5% of days | Mean of P&L values ≤ VaR |
| **Capture Ratio** | Actual revenue / benchmark revenue | (DA revenue − SIP-settled imbalance cost) / (delivered × DA price) |
| **Reward-to-Risk** | Risk-adjusted return (within-day signal-to-noise) | Mean(P&L) / Std(P&L) |

**Why CVaR over VaR:** VaR tells you the boundary of "normal" losses. CVaR tells you what to
expect *when things go wrong*. Given the fat-tailed SIP distribution, the gap between VaR and
CVaR is large — this is exactly the risk that matters for an EV aggregator.
""")

    st.markdown("#### 4.6 Position Sizing")
    st.markdown("""
**Percentile method:** The traded position for each settlement period is set at the Xth
percentile of the simulated availability distribution (P50, P80, P95 etc.). This is a
simple heuristic that ignores the payoff structure.

**Kelly Criterion:** Maximises E[log(1 + f·R)] — the long-run geometric growth rate.
Naturally penalises over-commitment when SIP tail risk is severe. Half-Kelly (f=0.5)
is industry standard: ~75% of growth rate with ~50% of drawdown variance.
""")

    # ── 5. Forecast Backtesting Methodology ───────────────────────────
    st.markdown("---")
    st.subheader("5. Forecast Backtesting & Alpha Detection")
    st.markdown("""
The backtesting engine evaluates whether our SIP forecasting models can generate
**alpha** (systematic advantage) over the market forward price (MIP) at various
forecast horizons.

**Walk-Forward Protocol:**
- Strictly out-of-sample: at each origin point *t*, the model only sees data up to *t-1*
- No parameter tuning on future data (prevents lookahead bias)
- Origins are spaced by a configurable step size (default: every 6 settlement periods)

**Forecast Models:**

| Model | Description |
|-------|------------|
| **TOD Mean** | For each target SP, average the same half-hour-of-day values over the lookback window |
| **EWMA** | Exponentially weighted average of same-SP-of-day values, recent data weighted more |
| **Market (MIP)** | TOD-mean of MIP over the lookback window (not a stale spot price) |

**Alpha Calculation:**
```
Alpha = Market_MAE − Forecast_MAE
```
Positive alpha means our forecast is closer to realised than the market benchmark.

**Statistical Significance:** All metrics include confidence intervals (Wilson score for
hit rates, block bootstrap for IR), Diebold-Mariano tests (HAC-adjusted) for pairwise
forecast comparison, and Benjamini-Hochberg FDR correction for multiple testing across
the lookback × horizon grid.
""")

    # ── 6. Summary: What's Real vs Assumed ────────────────────────────
    st.markdown("---")
    st.subheader("6. Summary: What's Live Data vs What's Assumed")

    summary = pd.DataFrame([
        {"Component": "SIP prices (empirical bootstrap)", "Source": "ELEXON API (live)", "Hardcoded?": "No"},
        {"Component": "MIP prices", "Source": "ELEXON API (live)", "Hardcoded?": "No"},
        {"Component": "SIP regime-switching params", "Source": "Fitted from ELEXON data", "Hardcoded?": "No (fallback defaults exist)"},
        {"Component": "DA price suggestion", "Source": "Derived from MIP mean", "Hardcoded?": "No (user-overridable)"},
        {"Component": "Fleet plug-in rate profiles", "Source": "CrowdFlex-inspired estimates", "Hardcoded?": "Defaults — configurable via sidebar"},
        {"Component": "Day-type multipliers", "Source": "Estimated (no public data)", "Hardcoded?": "Defaults — configurable via sidebar"},
        {"Component": "Seasonal monthly factors", "Source": "Estimated (no public data)", "Hardcoded?": "Defaults — configurable via sidebar"},
        {"Component": "SIP-stress coupling factors", "Source": "Engineering estimate", "Hardcoded?": "Defaults — configurable via sidebar"},
        {"Component": "DA price noise σ", "Source": "Assumption (DA auction variance)", "Hardcoded?": "Default 0.05 — configurable via sidebar"},
        {"Component": "Dispatch rate default", "Source": "Ohme engineering target", "Hardcoded?": "Default — configurable via sidebar"},
        {"Component": "Override rate default", "Source": "CrowdFlex trial (~2-5%)", "Hardcoded?": "Default — configurable via sidebar"},
        {"Component": "Copula correlation decay", "Source": "Assumption (not fitted)", "Hardcoded?": "Default 0.3 — configurable via sidebar"},
        {"Component": "Charger capacity (7.4 kW)", "Source": "Ohme hardware spec", "Hardcoded?": "Correctly fixed"},
        {"Component": "Settlement periods (48)", "Source": "GB market structure", "Hardcoded?": "Correctly fixed"},
    ])
    st.dataframe(summary, use_container_width=True, hide_index=True)

    # ── 7. Assumptions & Limitations ──────────────────────────────────
    st.markdown("---")
    st.subheader("7. Assumptions & Limitations")
    st.markdown("""
- **Charger homogeneity:** All chargers are assumed to be Ohme Home Pro (7.4 kW). In reality,
  the fleet may include different charger models with varying capacities.
- **Independence of dispatch/override:** Dispatch failures and customer overrides are modelled
  as independent events. Correlated failures (e.g., regional 4G outage) are not captured.
- **Plug-in profiles are estimates:** While now configurable, the defaults are educated guesses
  from CrowdFlex data. Real-world plug-in patterns from Ohme's telemetry would improve accuracy.
- **Day-type and seasonal multipliers are unsourced:** No public dataset exists for EV plug-in
  seasonality. The default factors are plausible assumptions that users should adjust.
- **SIP-stress coupling is approximate:** The adverse correlation between grid stress and fleet
  availability is modelled with fixed degradation factors, not a continuous function.
- **Single DA price per day:** A flat day-ahead price is assumed across all SPs. In reality, DA
  prices vary by settlement period (higher in peak, lower overnight). The lognormal noise
  adds run-to-run variation but not intra-day shape.
- **No rebalancing:** The model assumes the desk commits to a position and does not adjust
  intraday. In practice, the desk would buy back volume if plug-in rates disappoint.
- **Charge cost not modelled:** The cost of overnight catch-up charging is not subtracted from
  P&L. This would reduce net P&L but not affect relative risk comparisons.
- **MIP as forward proxy:** The Market Index Price is used as the market's forward view.
  The true forward curve would be derived from DA auction prices or intraday continuous market.
  MIP is the best publicly available proxy.
- **Copula decay is not fitted:** The correlation decay parameter (default 0.3) is an assumption,
  not calibrated from historical plug-in data.
- **Forecasters are intentionally simple:** TOD Mean and EWMA are baselines. More sophisticated
  models (regime-switching, ML, fundamental features) could improve forecasting but may overfit.
  The backtesting framework is designed to be model-agnostic.
""")

    # ── 8. References ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("8. References")
    st.markdown("""
- **CrowdFlex Trial (Ohme / National Grid ESO):** Demonstrated 42% overnight and 53% daytime
  increases in smart charging engagement with appropriate incentive design. Used as basis for
  plug-in rate parameterisation.
- **P415 (Virtual Trading Parties):** BSC modification allowing aggregators like Ohme to trade
  deviation volumes directly on the wholesale market, taking on supplier-like imbalance obligations.
- **PAR-1 Methodology:** The System Imbalance Price is set by the marginal 1 MWh of balancing
  actions, making it extremely "peaky" and reflective of genuine scarcity.
- **Reserve Scarcity Pricing:** RSP = LoLP × VoLL (£6,000/MWh). When the system is stressed,
  SIP can approach or exceed £6,000/MWh.
- **ELEXON Insights API Documentation:** https://bmrs.elexon.co.uk/api-documentation
- **GB Single Cash-Out (Nov 2015):** SBP = SSP — symmetric imbalance pricing.
- **Kelly Criterion (Kelly, 1956):** "A New Interpretation of Information Rate" — optimal
  position sizing for repeated bets with known payoff distributions.
""")
