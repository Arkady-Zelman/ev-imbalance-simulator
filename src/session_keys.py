"""
Centralised session-state key constants.

Every key used with st.session_state should be defined here to prevent
typo-driven bugs and enable IDE auto-complete.
"""

# ── Monte Carlo simulation result ──────────────────────────────────────────
RESULT             = "result"           # SimulationResult
ALL_POSITIONS      = "all_positions"
CAPTURE_RATIOS     = "capture_ratios"
RISK_SUMMARY       = "risk_summary"     # RiskSummary
PARAMS             = "params"           # SimulationParams

# ── Market data (fetched live for MC / allocation) ──────────────────────────
SIP_DF             = "sip_df"           # pd.DataFrame — historical SIP
MIP_DF             = "mip_df"           # pd.DataFrame — historical MIP
SIP_MATRIX         = "sip_matrix"       # np.ndarray (n_days × 48) bootstrap pool
DEMAND_DF          = "demand_df"
GEN_DF             = "gen_df"
DA_PRICE           = "da_price"         # float £/MWh

# ── Sizing method ───────────────────────────────────────────────────────────
SIZING_METHOD      = "sizing_method"    # "Percentile" | "Kelly"
BANKROLL           = "bankroll"
KELLY_RESULTS      = "kelly_results"
MULTI_PROFILE_RESULTS = "multi_profile_results"

# ── Allocation result ───────────────────────────────────────────────────────
ALLOCATION_RESULT  = "allocation_result"   # AllocationResult

# ── Frontend prediction state (loaded from data/predictions/ on startup) ────
PREDICTIONS_LOADED = "predictions_loaded"  # bool
METADATA           = "metadata"            # parsed metadata.json dict
PRED_SIP           = "pred_sip"            # pd.DataFrame from sip_predictions.parquet
PRED_MIP           = "pred_mip"
PRED_DEMAND        = "pred_demand"
PRED_GEN           = "pred_gen"
FAN_DATA           = "fan_data"            # pd.DataFrame from backtest_fan.parquet

# ── Chart interaction ───────────────────────────────────────────────────────
SELECTED_TARGET    = "selected_target"     # "sip" | "mip" | "demand" | "total_generation"
SELECTED_ORIGIN    = "selected_origin"     # datetime — clicked fan chart point
FORECAST_MODE      = "forecast_mode"       # "14day" | "intraday"

# ── Unified sidebar simulation parameters ───────────────────────────────────
SIM_FLEET_SIZE     = "sim_fleet_size"
SIM_DISPATCH_RATE  = "sim_dispatch_rate"
SIM_OVERRIDE_RATE  = "sim_override_rate"
SIM_MC_RUNS        = "sim_mc_runs"
SIM_RISK_APPETITE  = "sim_risk_appetite"
SIM_RISK_PROFILE   = "sim_risk_profile"
SIM_MC_RUNS_SENS   = "sim_mc_runs_sens"
SIM_RUN_REQUESTED  = "sim_run_requested"  # True when Run button clicked this cycle
