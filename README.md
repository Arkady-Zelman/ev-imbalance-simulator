# EV Flexibility Portfolio Imbalance Exposure Simulator - WIP Version

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ev-imbalance-simulator-oc8ybuvakhkkggxjh5amr8.streamlit.app/)

**[Live Dashboard](https://ev-imbalance-simulator-oc8ybuvakhkkggxjh5amr8.streamlit.app/)**

A Streamlit dashboard for modelling, simulating, and
backtesting the imbalance exposure of an EV smart-charging flexibility
portfolio in the GB electricity market.

I wanted to explore aggregating
residential EV chargers into virtual power plants and trading their combined
flexibility on wholesale and balancing markets. I wanted to build a hands-on
tool that captures the core economics of this problem — the asymmetric risk
between day-ahead revenue and system imbalance price exposure — and make it
interactive enough to explore different market conditions and risk appetites.

## Quick Start

```bash
pip install -r requirements.txt
python -m streamlit run app_v2.py
```

## Features

- **Monte Carlo simulation** of fleet availability using Beta distributions
  with a Gaussian copula for inter-SP correlation, with various additional features
  to play around with pertaining to daily availability.
- **XGBoost-LSTM** combined prediction engine for system imbalance pricing, market index pricing, generation and local/global UK demand.
- **Summary** of risk & P&L effects on our position on wholesale markets. WIP to be updated with a mixed position.

## Data Sources

All market data is fetched live from the public ELEXON BMRS API. No API key required.

| Dataset | Provider | Endpoint |
|---------|----------|----------|
| System Imbalance Price (SIP) | ELEXON Insights API | `/balancing/settlement/system-prices/{date}` |
| Market Index Price (MIP) | ELEXON Insights API | `/balancing/pricing/market-index?from=&to=` |

Fleet availability parameters (plug-in rates, dispatch success, override rates) are
configurable assumptions inspired by publicly available CrowdFlex trial data.

## License

MIT
