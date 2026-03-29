# EV Flexibility Portfolio Imbalance Exposure Simulator - WIP Version

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ev-imbalance-simulator-g4hnyfdnfrafcfffhswnbb.streamlit.app/)

**[Live Dashboard](https://ev-imbalance-simulator-g4hnyfdnfrafcfffhswnbb.streamlit.app/)**

A production-grade Streamlit dashboard for modelling, simulating, and
backtesting the imbalance exposure of an EV smart-charging flexibility
portfolio in the GB electricity market.

This project was inspired by [Ohme](https://ohme.io)'s work aggregating
residential EV chargers into virtual power plants and trading their combined
flexibility on wholesale and balancing markets. I wanted to build a hands-on
tool that captures the core economics of this problem — the asymmetric risk
between day-ahead revenue and system imbalance price exposure — and make it
interactive enough to explore different market conditions and risk appetites.

## Quick Start

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

## Features

- **Monte Carlo simulation** of fleet availability using Beta distributions
  with a Gaussian copula for inter-SP correlation
- **Position sizing** via configurable risk-appetite percentiles (P50–P95)
  or **Kelly Criterion** for optimal geometric growth
- **Empirical bootstrap** or **regime-switching** (data-fitted) SIP modelling
- **Forecast backtesting** with walk-forward TOD-mean and EWMA models
- **Alpha detection** comparing forecast accuracy against market forward
  (MIP) at multiple horizons and lookback windows
- **Rolling backtest** for market inefficiency detection across 1–14 day horizons
- **Statistical significance testing** — Diebold-Mariano tests, FDR correction,
  block bootstrap confidence intervals
- **Full configurability** — every hardcoded assumption (plug-in rates, seasonal
  factors, stress coupling, DA noise) is exposed in the sidebar with source documentation

## Architecture

```
app.py                      # Streamlit entry point + sidebar controls
src/
  config.py                 # All tuneable parameters & constants
  session_keys.py           # Centralised st.session_state key constants
  data/
    cache_manager.py        # Disk-based JSON cache with TTL
    elexon_client.py        # ELEXON Insights API client (SIP + MIP)
  models/
    portfolio.py            # Beta-copula plug-in rate model
    monte_carlo.py          # Vectorised MC simulation engine
    trading_position.py     # Position-sizing logic
    risk_metrics.py         # VaR, CVaR, capture ratio, reward-to-risk
    pnl_calculator.py       # Shared P&L computation
    sip_models.py           # Regime-switching SIP generation + fitting
    kelly.py                # Kelly Criterion optimal position sizing
    forecaster.py           # Walk-forward forecast engine (TOD, EWMA)
    alpha_detector.py       # Alpha matrix & optimal horizon detection
    stat_tests.py           # Diebold-Mariano, bootstrap CI, FDR correction
    rolling_backtest.py     # Rolling forecast backtest engine
  visualization/
    charts.py               # Core Plotly chart builders
    heatmaps.py             # Heatmaps, scenario comparisons, box plots
tabs/
  tab_executive_summary.py
  tab_portfolio_availability.py
  tab_monte_carlo.py
  tab_risk_analysis.py
  tab_sensitivity.py
  tab_scenario_comparison.py
  tab_historical_sip.py
  tab_backtesting.py        # Forecast backtesting & alpha detection
  tab_rolling_backtest.py   # Rolling forecast backtest
  tab_data_sources.py       # Full data provenance & methodology audit
tests/                      # 75 pytest unit tests
```

## Data Sources

All market data is fetched live from the public ELEXON BMRS API. No API key required.

| Dataset | Provider | Endpoint |
|---------|----------|----------|
| System Imbalance Price (SIP) | ELEXON Insights API | `/balancing/settlement/system-prices/{date}` |
| Market Index Price (MIP) | ELEXON Insights API | `/balancing/pricing/market-index?from=&to=` |

Fleet availability parameters (plug-in rates, dispatch success, override rates) are
configurable assumptions inspired by publicly available CrowdFlex trial data. See the
**Data Sources & Methodology** tab in the dashboard for a full audit of every assumption.

## Running Tests

```bash
pytest tests/ -v
```

## Tabs

| # | Tab | What It Shows |
|---|-----|---------------|
| 1 | Executive Summary | Traffic-light risk assessment, key metrics at a glance |
| 2 | Portfolio Availability | Beta distribution plug-in rates by SP, delivered MW |
| 3 | Monte Carlo Results | P&L distribution, imbalance cost breakdown |
| 4 | Risk Analysis | VaR/CVaR, capture ratios, Kelly Criterion analysis |
| 5 | Sensitivity Analysis | Tornado charts, diversification curves |
| 6 | Scenario Comparison | Benign vs stressed SIP regimes, custom scenarios |
| 7 | Historical SIP Explorer | Interactive SIP/MIP time series and distributions |
| 8 | Forecast Backtesting | Walk-forward alpha detection with statistical testing |
| 9 | Rolling Backtest | Market inefficiency detection across forecast horizons |
| 10 | Data Sources | Full provenance, methodology, and assumption audit |

## License

MIT
