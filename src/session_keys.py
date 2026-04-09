"""
Centralised session-state key constants.

Every key used with st.session_state should be defined here to prevent
typo-driven bugs and enable IDE auto-complete.
"""

RESULT = "result"
ALL_POSITIONS = "all_positions"
CAPTURE_RATIOS = "capture_ratios"
RISK_SUMMARY = "risk_summary"
SIP_DF = "sip_df"
MIP_DF = "mip_df"
SIP_MATRIX = "sip_matrix"
PARAMS = "params"
DA_PRICE = "da_price"
DATE_FROM = "date_from"
DATE_TO = "date_to"
SIP_MODE = "sip_mode"
BACKTEST_RESULTS = "backtest_results"
KELLY_RESULTS = "kelly_results"
SIZING_METHOD = "sizing_method"
BANKROLL = "bankroll"
DEMAND_DF = "demand_df"
GEN_DF = "gen_df"
SELECTED_LOOKBACK = "selected_lookback"        # Optional[str] — e.g. "5 days"
MULTI_PROFILE_RESULTS = "multi_profile_results"  # MultiProfileResult
EXOG_SERIES = "exog_series"                      # Optional[Dict[str, pd.Series]]
LSTM_SERIES = "lstm_series"                      # Optional[Dict[str, pd.Series]] — exog for LSTM inference
