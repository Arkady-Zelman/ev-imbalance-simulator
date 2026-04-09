"""
Rolling Forecast Backtest tab — market inefficiency detection.

Uses 1-day, 15-day, and 30-day lookbacks to forecast 1-14 days ahead,
comparing our forecast error against the forward curve at each horizon.
The crossover point marks the maximum exploitable forecast horizon.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

logger = logging.getLogger(__name__)

from src.config import (
    COLOUR_DANGER,
    COLOUR_MUTED,
    COLOUR_PRIMARY,
    COLOUR_SUCCESS,
    COLOUR_WARNING,
    PLOTLY_TEMPLATE,
)
from src.models.forecaster import build_aligned_series
from src.models.rolling_backtest import (
    ROLLING_HORIZON_LABELS,
    ROLLING_LOOKBACKS,
    build_error_matrix,
    run_rolling_backtest,
)
from src.models.prophet_trainer import (
    forecast_forward_np,
    has_np_models,
    load_np_models,
    save_np_models,
    train_all_np_parallel,
    train_np_models,
)
from src.models.xgb_trainer import (
    forecast_forward,
    has_trained_models,
    load_trained_models,
    save_trained_models,
    train_all_xgb_parallel,
    train_xgb_models,
)
from src.models.lstm_trainer import (
    TrainedLSTMModels,
    forecast_forward as lstm_forecast_forward,
    has_trained_lstm_models,
    load_trained_lstm_models,
    save_trained_lstm_models,
    train_all_lstm_parallel,
    train_lstm_models,
)
from src.models.dispatch_engine import process_generation_outturn
from src.session_keys import DEMAND_DF, EXOG_SERIES, GEN_DF, MIP_DF, SELECTED_LOOKBACK, SIP_DF


# ── Exogenous series builder ──────────────────────────────────────────────────

def _build_exog_series(
    sip_df: pd.DataFrame,
    gen_df_raw,
    gen_breakdown,
    use_wind_gen: bool,
    use_nuclear: bool,
    use_ccgt: bool,
    use_net_imp: bool,
    use_niv: bool,
    use_temp: bool,
    use_wind_spd: bool,
    use_solar: bool,
    use_cloud: bool,
    use_wind_fc: bool,
    use_dem_fc: bool,
) -> dict:
    """
    Build a dict of {name: pd.Series} for each enabled exogenous variable.
    All series use a timezone-naive datetime index (matching build_aligned_series output).
    """
    import datetime as dt
    from src.data.elexon_client import extract_niv_series, pivot_generation_wide

    exog: dict = {}

    # ── From already-fetched generation data ──
    if gen_df_raw is not None and not gen_df_raw.empty:
        try:
            gen_wide = pivot_generation_wide(gen_df_raw)
            if use_wind_gen and "wind_gen_mw" in gen_wide.columns:
                exog["wind_gen_mw"] = gen_wide["wind_gen_mw"].dropna()
            if use_nuclear and "nuclear_gen_mw" in gen_wide.columns:
                exog["nuclear_gen_mw"] = gen_wide["nuclear_gen_mw"].dropna()
            if use_ccgt and "ccgt_gen_mw" in gen_wide.columns:
                exog["ccgt_gen_mw"] = gen_wide["ccgt_gen_mw"].dropna()
            if use_net_imp and "net_imports_mw" in gen_wide.columns:
                exog["net_imports_mw"] = gen_wide["net_imports_mw"].dropna()
        except Exception as exc:
            logger.warning("Could not pivot generation data for exog: %s", exc)

    # ── NIV from SIP DataFrame ──
    if use_niv and sip_df is not None and not sip_df.empty:
        try:
            niv = extract_niv_series(sip_df)
            if niv is not None and not niv.empty:
                exog["niv"] = niv
        except Exception as exc:
            logger.warning("Could not extract NIV series: %s", exc)

    # ── Weather + ELEXON forecasts (new fetches) ──
    needs_weather = use_temp or use_wind_spd or use_solar or use_cloud
    needs_wind_fc = use_wind_fc
    needs_dem_fc  = use_dem_fc

    if needs_weather or needs_wind_fc or needs_dem_fc:
        # Determine date range from SIP data
        try:
            if hasattr(sip_df.index, "get_level_values"):
                dates = sip_df.index.get_level_values("settlementDate")
            else:
                dates = sip_df.index
            date_from = pd.Timestamp(dates.min()).date()
            date_to   = pd.Timestamp(dates.max()).date()
            # Extend forward by a few days for forecast coverage
            date_to_ext = date_to + dt.timedelta(days=2)
        except Exception:
            date_from = dt.date.today() - dt.timedelta(days=90)
            date_to_ext = dt.date.today() + dt.timedelta(days=2)

        if needs_weather:
            try:
                from src.data.weather_client import fetch_weather_data
                with st.spinner("Fetching weather data (Open-Meteo)…"):
                    wx = fetch_weather_data(date_from, date_to_ext)
                if not wx.empty:
                    if use_temp and "temperature_c" in wx.columns:
                        exog["temperature_c"] = wx["temperature_c"].dropna()
                    if use_wind_spd and "wind_speed_100m" in wx.columns:
                        exog["wind_speed_100m"] = wx["wind_speed_100m"].dropna()
                    if use_solar and "solar_radiation" in wx.columns:
                        exog["solar_radiation"] = wx["solar_radiation"].dropna()
                    if use_cloud and "cloud_cover_pct" in wx.columns:
                        exog["cloud_cover_pct"] = wx["cloud_cover_pct"].dropna()
            except Exception as exc:
                logger.warning("Weather fetch failed: %s", exc)

        if needs_wind_fc:
            try:
                from src.data.weather_client import fetch_wind_generation_forecast
                with st.spinner("Fetching ELEXON wind generation forecast…"):
                    wfc = fetch_wind_generation_forecast(date_from, date_to_ext)
                if not wfc.empty and "wind_fc_mw" in wfc.columns:
                    exog["wind_fc_mw"] = wfc["wind_fc_mw"].dropna()
            except Exception as exc:
                logger.warning("WINDFOR fetch failed: %s", exc)

        if needs_dem_fc:
            try:
                from src.data.weather_client import fetch_demand_forecast
                with st.spinner("Fetching ELEXON demand forecast…"):
                    dfc = fetch_demand_forecast(date_from, date_to_ext)
                if not dfc.empty and "demand_fc_mw" in dfc.columns:
                    exog["demand_fc_mw"] = dfc["demand_fc_mw"].dropna()
            except Exception as exc:
                logger.warning("NDFD fetch failed: %s", exc)

    return exog


def render(has_results: bool) -> None:
    st.header("Rolling Forecast Backtest — Market Inefficiency Detection")
    st.caption(
        "Identifies the maximum forecast horizon where our model consistently "
        "beats the market (MIP forward curve). Uses daily non-overlapping origins "
        "for statistically valid inference."
    )

    if not has_results:
        st.info("Run a simulation first (which fetches ELEXON data) to use this tool.")
        return

    if SIP_DF not in st.session_state or st.session_state[SIP_DF].empty:
        st.warning("No SIP data available.")
        return
    if MIP_DF not in st.session_state or st.session_state[MIP_DF].empty:
        st.warning("No MIP data available.")
        return

    _render_rolling_backtest_body()


@st.fragment
def _render_rolling_backtest_body() -> None:
    """Isolated as a fragment so widget changes don't reset the active tab."""
    sip_df = st.session_state[SIP_DF]
    mip_df = st.session_state[MIP_DF]
    demand_df = st.session_state.get(DEMAND_DF)
    gen_df_raw = st.session_state.get(GEN_DF)

    has_demand = demand_df is not None and not demand_df.empty
    has_gen = gen_df_raw is not None and not gen_df_raw.empty

    # Pre-process generation breakdown once (lightweight pivot)
    gen_breakdown = process_generation_outturn(gen_df_raw) if has_gen else None

    # ── Configuration ─────────────────────────────────────────────────
    st.subheader("Configuration")

    _TARGET_DESC = {
        "sip": "Price (SIP)",
        "mip": "Wholesale (MIP)",
        "demand": "Demand",
        "total_generation": "Total Generation",
    }

    # Forecast targets — checkboxes
    st.caption("**Forecast targets** — select one or more")
    _target_opts = [("Price (SIP)", "sip"), ("Wholesale (MIP)", "mip")]
    if has_demand:
        _target_opts.append(("Demand", "demand"))
    if has_gen:
        _target_opts.append(("Total Generation", "total_generation"))
    _tc = st.columns(max(len(_target_opts), 2))
    selected_targets = [
        key for (lbl, key), col in zip(_target_opts, _tc)
        if col.checkbox(lbl, value=(key == "sip"), key=f"rb_tgt_{key}")
    ]
    if not selected_targets:
        st.info("Select at least one forecast target above.")
        return
    # Filter targets whose data is unavailable
    if "demand" in selected_targets and not has_demand:
        st.warning("Demand data not available — removing Demand from selection.")
        selected_targets = [t for t in selected_targets if t != "demand"]
    if "total_generation" in selected_targets and not has_gen:
        st.warning("Generation data not available — removing Total Generation from selection.")
        selected_targets = [t for t in selected_targets if t != "total_generation"]
    if not selected_targets:
        st.warning("No valid targets available with current data.")
        return
    target_key = selected_targets[0]  # primary target for single-target operations

    # Forecast method — dropdown
    method_options = ["TOD Mean", "EWMA", "XGBoost", "LSTM", "Hybrid LSTM+XGBoost", "NeuralProphet"]
    method_label = st.selectbox("Forecast method", method_options, index=2, key="rb_method")
    method_key = {
        "TOD Mean": "tod_mean",
        "EWMA": "ewma",
        "XGBoost": "xgb",
        "LSTM": "lstm",
        "Hybrid LSTM+XGBoost": "hybrid",
        "NeuralProphet": "neuralprophet",
    }[method_label]
    selected_methods = [method_key]  # list wrapper for downstream compatibility

    # EWMA alpha — shown only when EWMA is selected
    if method_key == "ewma":
        ewma_alpha = st.slider(
            "EWMA α", 0.01, 0.30, 0.05, 0.01,
            key="rb_ewma_alpha",
            help="Exponential smoothing decay factor.",
        )
    else:
        ewma_alpha = st.session_state.get("rb_ewma_alpha", 0.05)

    # Lookback windows — checkboxes
    st.caption("**Lookback windows** — select one or more")
    _lb_keys = list(ROLLING_LOOKBACKS.keys())
    _lbc = st.columns(len(_lb_keys))
    selected_lookback_list = [
        lb for lb, col in zip(_lb_keys, _lbc)
        if col.checkbox(lb, value=True, key=f"rb_lb_{lb.replace(' ', '_')}")
    ]
    if not selected_lookback_list:
        st.info("Select at least one lookback window above.")
        return
    selected_lookback: Optional[str] = (
        selected_lookback_list[0] if len(selected_lookback_list) == 1 else None
    )
    st.session_state[SELECTED_LOOKBACK] = selected_lookback

    target_desc = " · ".join(_TARGET_DESC[t] for t in selected_targets)

    # Build gen pd.Series aligned to breakdown index (if available)
    gen_series_rb: pd.Series | None = None
    if gen_breakdown is not None:
        gen_series_rb = pd.Series(gen_breakdown.total_mw, index=gen_breakdown.index)

    # ── Exogenous Variables (XGBoost / LSTM / Hybrid) ────────────────
    exog_series_for_training: dict | None = None
    if any(m in ("xgb", "lstm", "hybrid") for m in selected_methods):
        with st.expander("Exogenous Variables", expanded=False):
            st.caption(
                "Additional signals passed to the forecast model as features/channels. "
                "All sources are free and require no API key."
            )
            st.markdown("**Already available (no extra fetch)**")
            c1, c2 = st.columns(2)
            with c1:
                use_wind_gen  = st.checkbox("Wind generation (ELEXON FUELHH)", value=True, key="exog_wind_gen")
                use_nuclear   = st.checkbox("Nuclear generation (ELEXON FUELHH)", value=True, key="exog_nuclear")
                use_ccgt      = st.checkbox("CCGT/gas generation — gas proxy (ELEXON FUELHH)", value=True, key="exog_ccgt")
            with c2:
                use_net_imp   = st.checkbox("Net interconnector imports (ELEXON FUELHH)", value=True, key="exog_net_imp")
                use_niv       = st.checkbox("Net Imbalance Volume (ELEXON SIP)", value=True, key="exog_niv")
            st.markdown("**Live fetches (Open-Meteo · ELEXON)**")
            c3, c4 = st.columns(2)
            with c3:
                use_temp      = st.checkbox("Temperature °C (Open-Meteo ERA5/forecast)", value=True, key="exog_temp")
                use_wind_spd  = st.checkbox("Wind speed 100 m (Open-Meteo)", value=True, key="exog_wind_spd")
                use_solar     = st.checkbox("Solar radiation W/m² (Open-Meteo)", value=True, key="exog_solar")
                use_cloud     = st.checkbox("Cloud cover % (Open-Meteo)", value=False, key="exog_cloud")
            with c4:
                use_wind_fc   = st.checkbox("Day-ahead wind forecast MW (ELEXON WINDFOR)", value=True, key="exog_wind_fc")
                use_dem_fc    = st.checkbox("Day-ahead demand forecast MW (ELEXON NDFD)", value=True, key="exog_dem_fc")

        # Build exog_series dict from checked boxes
        exog_series_for_training = _build_exog_series(
            sip_df=sip_df,
            gen_df_raw=gen_df_raw,
            gen_breakdown=gen_breakdown,
            use_wind_gen=use_wind_gen,
            use_nuclear=use_nuclear,
            use_ccgt=use_ccgt,
            use_net_imp=use_net_imp,
            use_niv=use_niv,
            use_temp=use_temp,
            use_wind_spd=use_wind_spd,
            use_solar=use_solar,
            use_cloud=use_cloud,
            use_wind_fc=use_wind_fc,
            use_dem_fc=use_dem_fc,
        )
        st.session_state[EXOG_SERIES] = exog_series_for_training

    # ── XGBoost: Train / Show Results workflow ────────────────────────
    if "xgb" in selected_methods:
        st.markdown("---")
        st.subheader(f"XGBoost Model Training — {target_desc}")
        st.caption(
            "Trains one model per selected target. Multiple targets run in parallel. "
            "Models persist to disk across sessions."
        )

        col_deep, col_quick, col_show = st.columns(3)
        with col_deep:
            deep_btn = st.button(
                "🏋️ In-Depth Search",
                use_container_width=True, type="primary", key="rb_xgb_deep",
                help="Systematic grid search over core hyperparameters (27 combos).",
            )
        with col_quick:
            quick_btn = st.button(
                "⚡ Quick Random Search",
                use_container_width=True, key="rb_xgb_quick",
                help="Random search across all hyperparameters (30 samples). Faster.",
            )
        with col_show:
            show_btn = st.button(
                "📊 Show Results",
                use_container_width=True, key="rb_xgb_show",
                help="Load and display results from the last trained models.",
            )

        xgb_search_mode = "grid" if deep_btn else ("random" if quick_btn else None)

        if xgb_search_mode is not None:
            mode_label = "In-depth grid" if xgb_search_mode == "grid" else "Quick random"
            with st.spinner("Aligning series…"):
                sip_series, mip_series, demand_series, _ = build_aligned_series(
                    sip_df, mip_df, demand_df=demand_df,
                )
            if len(sip_series) // 48 < 45:
                st.error(f"Need at least 45 days of data. Currently {len(sip_series)//48} days.")
                return

            _st_ctx = get_script_run_ctx()
            para_bars = {t: st.progress(0.0, text=_TARGET_DESC[t]) for t in selected_targets}

            def _make_xgb_cb(tgt):
                def _cb(frac, msg):
                    add_script_run_ctx(threading.current_thread(), _st_ctx)
                    para_bars[tgt].progress(min(float(frac), 1.0), text=msg)
                return _cb

            try:
                with st.spinner(f"Training XGBoost — {target_desc}…"):
                    xgb_para = train_all_xgb_parallel(
                        sip_series=sip_series, mip_series=mip_series,
                        demand_series=demand_series, gen_series=gen_series_rb,
                        targets=selected_targets, param_search_mode=xgb_search_mode,
                        selected_lookback=selected_lookback,
                        progress_callbacks={t: _make_xgb_cb(t) for t in selected_targets},
                        exog_series=exog_series_for_training or None,
                    )
            except Exception as exc:
                st.error(f"XGBoost training failed: {exc}")
                logger.exception("XGBoost training error")
                return

            for t, trained in xgb_para.models.items():
                save_trained_models(trained, target=t)
                st.session_state[f"_xgb_trained_models_{t}"] = trained
                rb_key_prefix = f"_rb_{t}_xgb"
                st.session_state[f"{rb_key_prefix}_errors"] = trained.backtest_errors
                st.session_state[f"{rb_key_prefix}_crossovers"] = trained.backtest_crossovers
                st.session_state["_rb_last_key"] = rb_key_prefix
            for t, err in xgb_para.errors.items():
                st.error(f"{_TARGET_DESC[t]} XGBoost failed: {err}")
            if xgb_para.models:
                st.success(
                    f"{mode_label} XGBoost training complete in {xgb_para.elapsed_seconds:.0f}s — "
                    f"trained: {', '.join(_TARGET_DESC[t] for t in xgb_para.models)}"
                )
            _first_xgb = next(iter(xgb_para.models.values()), None)
            if _first_xgb:
                _display_training_summary(_first_xgb)

        if show_btn:
            import datetime as dt
            for t in selected_targets:
                trained = st.session_state.get(f"_xgb_trained_models_{t}") or load_trained_models(target=t)
                if trained is None:
                    st.info(f"No trained {_TARGET_DESC[t]} XGBoost model found.")
                    continue
                st.session_state[f"_xgb_trained_models_{t}"] = trained
                ts = dt.datetime.fromtimestamp(trained.training_timestamp)
                st.caption(f"{_TARGET_DESC[t]} XGBoost — trained **{ts:%Y-%m-%d %H:%M}**")
                rb_key_prefix = f"_rb_{t}_xgb"
                st.session_state[f"{rb_key_prefix}_errors"] = trained.backtest_errors
                st.session_state[f"{rb_key_prefix}_crossovers"] = trained.backtest_crossovers
                st.session_state["_rb_last_key"] = rb_key_prefix
            _first_xgb = st.session_state.get(f"_xgb_trained_models_{target_key}")
            if _first_xgb:
                _display_training_summary(_first_xgb)

    if "lstm" in selected_methods or "hybrid" in selected_methods:
        # ── LSTM (and Hybrid uses same training UI) ───────────────────────────
        _lstm_method_key = "hybrid" if "hybrid" in selected_methods else "lstm"

        st.markdown("---")
        st.subheader(f"LSTM Model Training — {target_desc}")
        st.caption(
            "Trains one LSTM model per selected target (PyTorch, Huber loss, early stopping). "
            "Multiple targets run in parallel."
        )

        col_ldp, col_lqk, col_lsh = st.columns(3)
        with col_ldp:
            lstm_deep_btn = st.button(
                "🏋️ In-Depth Search",
                use_container_width=True, type="primary", key="rb_lstm_deep",
                help="15 combos × 20 search epochs + 50 final epochs. Best accuracy.",
            )
        with col_lqk:
            lstm_quick_btn = st.button(
                "⚡ Quick Search",
                use_container_width=True, key="rb_lstm_quick",
                help="5 combos × 20 epochs. Fast point-in-time tuning.",
            )
        with col_lsh:
            lstm_show_btn = st.button(
                "📊 Show Results",
                use_container_width=True, key="rb_lstm_show",
                help="Load results from the last trained LSTM models.",
            )

        lstm_search_mode = "grid" if lstm_deep_btn else ("random" if lstm_quick_btn else None)

        if lstm_search_mode is not None:
            mode_label = "In-depth" if lstm_search_mode == "grid" else "Quick"
            with st.spinner("Aligning series…"):
                sip_series, mip_series, demand_series, _ = build_aligned_series(
                    sip_df, mip_df, demand_df=demand_df,
                )
            if len(sip_series) // 48 < 45:
                st.error(f"Need at least 45 days of data. Currently {len(sip_series)//48} days.")
            else:
                _st_ctx_l = get_script_run_ctx()
                lstm_para_bars = {t: st.progress(0.0, text=_TARGET_DESC[t]) for t in selected_targets}

                def _make_lstm_cb(tgt):
                    def _cb(frac, msg):
                        add_script_run_ctx(threading.current_thread(), _st_ctx_l)
                        lstm_para_bars[tgt].progress(min(float(frac), 1.0), text=msg)
                    return _cb

                try:
                    with st.spinner(f"Training LSTM — {target_desc}…"):
                        lstm_para = train_all_lstm_parallel(
                            sip_series=sip_series, mip_series=mip_series,
                            demand_series=demand_series, gen_series=gen_series_rb,
                            targets=selected_targets, param_search_mode=lstm_search_mode,
                            selected_lookback=selected_lookback,
                            progress_callbacks={t: _make_lstm_cb(t) for t in selected_targets},
                            exog_series=exog_series_for_training or None,
                        )
                except Exception as exc:
                    st.error(f"LSTM training failed: {exc}")
                    logger.exception("LSTM training error")
                    lstm_para = None

                if lstm_para is not None:
                    for t, trained_lstm in lstm_para.models.items():
                        save_trained_lstm_models(trained_lstm, target=t)
                        st.session_state[f"_lstm_trained_models_{t}"] = trained_lstm
                        rb_key_prefix = f"_rb_{t}_{_lstm_method_key}"
                        st.session_state[f"{rb_key_prefix}_errors"] = trained_lstm.backtest_errors
                        st.session_state[f"{rb_key_prefix}_crossovers"] = trained_lstm.backtest_crossovers
                        st.session_state["_rb_last_key"] = rb_key_prefix
                    for t, err in lstm_para.errors.items():
                        st.error(f"{_TARGET_DESC[t]} LSTM failed: {err}")
                    if lstm_para.models:
                        st.success(
                            f"{mode_label} LSTM training complete in {lstm_para.elapsed_seconds:.0f}s — "
                            f"trained: {', '.join(_TARGET_DESC[t] for t in lstm_para.models)}"
                        )
                    _first_lstm = next(iter(lstm_para.models.values()), None)
                    if _first_lstm:
                        _display_lstm_training_summary(_first_lstm)

        if lstm_show_btn:
            import datetime as dt
            for t in selected_targets:
                trained_lstm = st.session_state.get(f"_lstm_trained_models_{t}") or load_trained_lstm_models(target=t)
                if trained_lstm is None:
                    st.info(f"No trained {_TARGET_DESC[t]} LSTM model found.")
                    continue
                st.session_state[f"_lstm_trained_models_{t}"] = trained_lstm
                ts = dt.datetime.fromtimestamp(trained_lstm.training_timestamp)
                st.caption(f"{_TARGET_DESC[t]} LSTM — trained **{ts:%Y-%m-%d %H:%M}**")
                rb_key_prefix = f"_rb_{t}_{_lstm_method_key}"
                st.session_state[f"{rb_key_prefix}_errors"] = trained_lstm.backtest_errors
                st.session_state[f"{rb_key_prefix}_crossovers"] = trained_lstm.backtest_crossovers
                st.session_state["_rb_last_key"] = rb_key_prefix
            _first_lstm = st.session_state.get(f"_lstm_trained_models_{target_key}")
            if _first_lstm:
                _display_lstm_training_summary(_first_lstm)

        # Hybrid — additionally show XGBoost component status for primary target
        if "hybrid" in selected_methods:
            st.markdown("---")
            st.subheader(f"Hybrid Ensemble — {target_desc}")
            st.caption(
                "The Hybrid forecast combines LSTM + XGBoost predictions using inverse-MAE "
                "weighting: the model with lower validation error gets higher weight."
            )
            import datetime as dt
            for t in selected_targets:
                trained_lstm_h = st.session_state.get(f"_lstm_trained_models_{t}") or load_trained_lstm_models(t)
                trained_xgb_h  = st.session_state.get(f"_xgb_trained_models_{t}")  or load_trained_models(target=t)
                c_l, c_x, c_w = st.columns(3)
                with c_l:
                    if trained_lstm_h:
                        st.success(f"{_TARGET_DESC[t]} LSTM: {dt.datetime.fromtimestamp(trained_lstm_h.training_timestamp):%H:%M}")
                    else:
                        st.warning(f"{_TARGET_DESC[t]} LSTM: not trained")
                with c_x:
                    if trained_xgb_h:
                        st.success(f"{_TARGET_DESC[t]} XGBoost: {dt.datetime.fromtimestamp(trained_xgb_h.training_timestamp):%H:%M}")
                    else:
                        st.warning(f"{_TARGET_DESC[t]} XGBoost: not trained")
                with c_w:
                    if trained_lstm_h and trained_xgb_h:
                        from src.models.hybrid_forecaster import _best_val_mae
                        w_l = _best_val_mae(trained_lstm_h)
                        w_x = _best_val_mae(trained_xgb_h)
                        if w_l and w_x:
                            total = (1 / w_l) + (1 / w_x)
                            pct_l = round(100 * (1 / w_l) / total)
                            st.caption(f"Weights: LSTM {pct_l}% · XGBoost {100 - pct_l}%")

    if "neuralprophet" in selected_methods:
        # ── NeuralProphet: full training workflow ──────────────────────────
        st.markdown("---")
        st.subheader(f"NeuralProphet Model Training — {target_desc}")
        st.caption(
            "Trains one NeuralProphet model per selected target. Multiple targets run in parallel."
        )

        col_deep, col_quick, col_show = st.columns(3)
        with col_deep:
            np_deep_btn = st.button(
                "🏋️ In-Depth Search",
                use_container_width=True, type="primary", key="rb_np_deep",
                help="18 combos over n_lags, epochs, learning_rate. Takes several minutes.",
            )
        with col_quick:
            np_quick_btn = st.button(
                "⚡ Quick Random Search",
                use_container_width=True, key="rb_np_quick",
                help="6 random combos — faster estimate.",
            )
        with col_show:
            np_show_btn = st.button(
                "📊 Show Results",
                use_container_width=True, key="rb_np_show",
                help="Load and display results from the last trained NP models.",
            )

        np_search_mode = "grid" if np_deep_btn else ("random" if np_quick_btn else None)

        # NeuralProphet only supports SIP/MIP/Demand (not generation)
        _np_targets = [t for t in selected_targets if t != "total_generation"]

        if np_search_mode is not None:
            mode_label = "In-depth grid" if np_search_mode == "grid" else "Quick random"
            with st.spinner("Aligning series…"):
                sip_series, mip_series, demand_series, _ = build_aligned_series(
                    sip_df, mip_df, demand_df=demand_df,
                )
            if len(sip_series) // 48 < 45:
                st.error(f"Need at least 45 days of data. Currently {len(sip_series)//48} days.")
                return

            _st_ctx_np = get_script_run_ctx()
            np_para_bars = {t: st.progress(0.0, text=_TARGET_DESC[t]) for t in _np_targets}

            def _make_np_cb(tgt):
                def _cb(frac, msg):
                    add_script_run_ctx(threading.current_thread(), _st_ctx_np)
                    np_para_bars[tgt].progress(min(float(frac), 1.0), text=msg)
                return _cb

            try:
                with st.spinner(f"Training NeuralProphet — {target_desc}…"):
                    np_para = train_all_np_parallel(
                        sip_series=sip_series, mip_series=mip_series,
                        demand_series=demand_series,
                        targets=_np_targets, param_search_mode=np_search_mode,
                        progress_callbacks={t: _make_np_cb(t) for t in _np_targets},
                    )
            except Exception as exc:
                st.error(f"NeuralProphet training failed: {exc}")
                logger.exception("NeuralProphet training error")
                return

            for t, trained_np in np_para.models.items():
                save_np_models(trained_np, target=t)
                st.session_state[f"_np_trained_models_{t}"] = trained_np
                rb_key_prefix = f"_rb_{t}_neuralprophet"
                st.session_state[f"{rb_key_prefix}_errors"] = trained_np.backtest_errors
                st.session_state[f"{rb_key_prefix}_crossovers"] = trained_np.backtest_crossovers
                st.session_state["_rb_last_key"] = rb_key_prefix
            for t, err in np_para.errors.items():
                st.error(f"{_TARGET_DESC[t]} NeuralProphet failed: {err}")
            if np_para.models:
                st.success(
                    f"{mode_label} NeuralProphet training complete in {np_para.elapsed_seconds:.0f}s — "
                    f"trained: {', '.join(_TARGET_DESC[t] for t in np_para.models)}"
                )
            _first_np = next(iter(np_para.models.values()), None)
            if _first_np:
                _display_np_training_summary(_first_np)

        if np_show_btn:
            import datetime as dt
            for t in _np_targets:
                trained_np = st.session_state.get(f"_np_trained_models_{t}") or load_np_models(target=t)
                if trained_np is None:
                    st.info(f"No trained {_TARGET_DESC[t]} NeuralProphet model found.")
                    continue
                st.session_state[f"_np_trained_models_{t}"] = trained_np
                ts = dt.datetime.fromtimestamp(trained_np.training_timestamp)
                age_hours = (dt.datetime.now() - ts).total_seconds() / 3600
                st.caption(f"{_TARGET_DESC[t]} NP — trained **{ts:%Y-%m-%d %H:%M}**")
                if age_hours > 24:
                    st.warning(f"{_TARGET_DESC[t]} NP model is {age_hours:.0f}h old — consider re-training.")
                rb_key_prefix = f"_rb_{t}_neuralprophet"
                st.session_state[f"{rb_key_prefix}_errors"] = trained_np.backtest_errors
                st.session_state[f"{rb_key_prefix}_crossovers"] = trained_np.backtest_crossovers
                st.session_state["_rb_last_key"] = rb_key_prefix
            _first_np = st.session_state.get(f"_np_trained_models_{target_key}")
            if _first_np:
                _display_np_training_summary(_first_np)

    if method_key in ("tod_mean", "ewma"):
        # ── TOD Mean / EWMA: run per selected target ──────────────────────────
        _m_disp = {"tod_mean": "TOD Mean", "ewma": "EWMA"}[method_key]

        st.markdown("---")
        st.subheader(f"{_m_disp} Backtest — {target_desc}")

        run_btn = st.button(
            f"🔬 Run {_m_disp} Rolling Backtest",
            use_container_width=True, type="primary", key=f"rb_run_{method_key}",
        )

        if run_btn:
            with st.spinner("Aligning SIP, MIP and Demand series…"):
                sip_series, mip_series, demand_series, _ = build_aligned_series(
                    sip_df, mip_df, demand_df=demand_df,
                )
            n_days = len(sip_series) // 48
            if n_days < 45:
                st.error(f"Need at least 45 days. Currently {n_days} days.")
                return

            for _t in selected_targets:
                with st.spinner(f"Running {_m_disp} backtest — {_TARGET_DESC[_t]}…"):
                    errors, crossovers = run_rolling_backtest(
                        sip_series, mip_series,
                        method=method_key,
                        ewma_alpha=ewma_alpha,
                        target=_t,
                        demand_series=demand_series,
                        gen_series=gen_series_rb,
                    )
                rb_key_prefix = f"_rb_{_t}_{method_key}"
                st.session_state[f"{rb_key_prefix}_errors"] = errors
                st.session_state[f"{rb_key_prefix}_crossovers"] = crossovers
                st.session_state["_rb_last_key"] = rb_key_prefix
                st.success(f"{_TARGET_DESC[_t]} — {len(errors)} configurations evaluated.")

    # ── Train All Models Simultaneously ───────────────────────────────
    st.markdown("---")
    st.subheader("Train All Models Simultaneously")

    _target_labels_all = {
        "sip": "Price (SIP)", "mip": "Wholesale (MIP)",
        "demand": "Demand", "total_generation": "Generation",
    }

    # XGBoost — all 4 targets
    st.caption(
        "**XGBoost** — Trains SIP · MIP · Demand · Generation in parallel. "
        "Select a single **Lookback Window** above to keep runtime manageable."
    )
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        para_deep = st.button(
            "🏋️ All 4 XGB — In-Depth",
            use_container_width=True, type="primary", key="rb_para_deep",
            help="150-sample grid search per model. Best accuracy; takes several minutes.",
        )
    with col_p2:
        para_quick = st.button(
            "⚡ All 4 XGB — Quick",
            use_container_width=True, key="rb_para_quick",
            help="30-sample random search per model. Fast point-in-time tuning.",
        )

    para_mode: Optional[str] = "grid" if para_deep else ("random" if para_quick else None)

    if para_mode is not None:
        with st.spinner("Aligning series for XGBoost parallel training…"):
            sip_series_p, mip_series_p, demand_series_p, _ = build_aligned_series(
                sip_df, mip_df, demand_df=demand_df,
            )
        n_days_p = len(sip_series_p) // 48
        if n_days_p < 45:
            st.error(
                f"Need at least 45 days of aligned data. "
                f"Currently have {n_days_p} days. Increase the date range."
            )
        else:
            para_targets = ["sip", "mip"]
            if has_demand:
                para_targets.append("demand")
            if has_gen:
                para_targets.append("total_generation")

            para_bars = {t: st.progress(0.0, text=_target_labels_all.get(t, t))
                         for t in para_targets}

            _st_ctx = get_script_run_ctx()

            def _make_para_cb(tgt: str):
                def _cb(frac: float, msg: str) -> None:
                    add_script_run_ctx(threading.current_thread(), _st_ctx)
                    para_bars[tgt].progress(min(float(frac), 1.0), text=msg)
                return _cb

            para_cbs = {t: _make_para_cb(t) for t in para_targets}

            with st.spinner(f"Training {len(para_targets)} XGBoost models in parallel…"):
                para_result = train_all_xgb_parallel(
                    sip_series=sip_series_p,
                    mip_series=mip_series_p,
                    demand_series=demand_series_p if has_demand else None,
                    gen_series=gen_series_rb,
                    targets=para_targets,
                    param_search_mode=para_mode,
                    selected_lookback=st.session_state.get(SELECTED_LOOKBACK),
                    progress_callbacks=para_cbs,
                    exog_series=st.session_state.get(EXOG_SERIES) or None,
                )

            for t, trained_p in para_result.models.items():
                st.session_state[f"_xgb_trained_models_{t}"] = trained_p
            for t, err in para_result.errors.items():
                st.error(f"{_target_labels_all.get(t, t)} XGB training failed: {err}")

            ok_targets = list(para_result.models.keys())
            if ok_targets:
                st.success(
                    f"XGBoost parallel training complete in {para_result.elapsed_seconds:.0f}s — "
                    f"trained: {', '.join(_target_labels_all.get(t, t) for t in ok_targets)}"
                )

    # NeuralProphet — all 3 targets
    st.caption(
        "**NeuralProphet** — Trains SIP · MIP · Demand in parallel. "
        "Slower than XGBoost — 'Quick' is recommended for first runs."
    )
    col_np1, col_np2 = st.columns(2)
    with col_np1:
        np_para_deep = st.button(
            "🏋️ All 3 NP — In-Depth",
            use_container_width=True, type="primary", key="rb_np_para_deep",
            help="18-combo grid search per model. Best accuracy; takes many minutes.",
        )
    with col_np2:
        np_para_quick = st.button(
            "⚡ All 3 NP — Quick",
            use_container_width=True, key="rb_np_para_quick",
            help="3 random combos per model. Fast estimate for point-in-time tuning.",
        )

    np_para_mode: Optional[str] = "grid" if np_para_deep else ("random" if np_para_quick else None)

    if np_para_mode is not None:
        with st.spinner("Aligning series for NeuralProphet parallel training…"):
            sip_series_np, mip_series_np, demand_series_np, _ = build_aligned_series(
                sip_df, mip_df, demand_df=demand_df,
            )
        n_days_np = len(sip_series_np) // 48
        if n_days_np < 45:
            st.error(
                f"Need at least 45 days of aligned data. "
                f"Currently have {n_days_np} days. Increase the date range."
            )
        else:
            np_para_targets = ["sip", "mip"]
            if has_demand:
                np_para_targets.append("demand")

            np_para_bars = {t: st.progress(0.0, text=_target_labels_all.get(t, t))
                            for t in np_para_targets}

            _st_ctx_np = get_script_run_ctx()

            def _make_np_para_cb(tgt: str):
                def _cb(frac: float, msg: str) -> None:
                    add_script_run_ctx(threading.current_thread(), _st_ctx_np)
                    np_para_bars[tgt].progress(min(float(frac), 1.0), text=msg)
                return _cb

            np_para_cbs = {t: _make_np_para_cb(t) for t in np_para_targets}

            with st.spinner(f"Training {len(np_para_targets)} NeuralProphet models in parallel…"):
                np_para_result = train_all_np_parallel(
                    sip_series=sip_series_np,
                    mip_series=mip_series_np,
                    demand_series=demand_series_np if has_demand else None,
                    targets=np_para_targets,
                    param_search_mode=np_para_mode,
                    progress_callbacks=np_para_cbs,
                )

            for t, trained_np_p in np_para_result.models.items():
                st.session_state[f"_np_trained_models_{t}"] = trained_np_p
            for t, err in np_para_result.errors.items():
                st.error(f"{_target_labels_all.get(t, t)} NP training failed: {err}")

            ok_np = list(np_para_result.models.keys())
            if ok_np:
                st.success(
                    f"NeuralProphet parallel training complete in {np_para_result.elapsed_seconds:.0f}s — "
                    f"trained: {', '.join(_target_labels_all.get(t, t) for t in ok_np)}"
                )

    # LSTM — all 4 targets
    st.caption(
        "**LSTM** — Trains SIP · MIP · Demand · Generation in parallel. "
        "PyTorch Huber-loss LSTM; CUDA GPU used when available."
    )
    col_ll1, col_ll2 = st.columns(2)
    with col_ll1:
        lstm_para_deep = st.button(
            "🏋️ All 4 LSTM — In-Depth",
            use_container_width=True, type="primary", key="rb_lstm_para_deep",
            help="15-combo search per model. Best accuracy; takes several minutes.",
        )
    with col_ll2:
        lstm_para_quick = st.button(
            "⚡ All 4 LSTM — Quick",
            use_container_width=True, key="rb_lstm_para_quick",
            help="5-combo search per model. Fast point-in-time tuning.",
        )

    lstm_para_mode: Optional[str] = "grid" if lstm_para_deep else ("random" if lstm_para_quick else None)

    if lstm_para_mode is not None:
        with st.spinner("Aligning series for LSTM parallel training…"):
            sip_series_ll, mip_series_ll, demand_series_ll, _ = build_aligned_series(
                sip_df, mip_df, demand_df=demand_df,
            )
        n_days_ll = len(sip_series_ll) // 48
        if n_days_ll < 45:
            st.error(
                f"Need at least 45 days of aligned data. "
                f"Currently have {n_days_ll} days. Increase the date range."
            )
        else:
            lstm_para_targets = ["sip", "mip"]
            if has_demand:
                lstm_para_targets.append("demand")
            if has_gen:
                lstm_para_targets.append("total_generation")

            lstm_para_bars = {t: st.progress(0.0, text=_target_labels_all.get(t, t))
                              for t in lstm_para_targets}

            _st_ctx_ll = get_script_run_ctx()

            def _make_lstm_para_cb(tgt: str):
                def _cb(frac: float, msg: str) -> None:
                    add_script_run_ctx(threading.current_thread(), _st_ctx_ll)
                    lstm_para_bars[tgt].progress(min(float(frac), 1.0), text=msg)
                return _cb

            lstm_para_cbs = {t: _make_lstm_para_cb(t) for t in lstm_para_targets}

            with st.spinner(f"Training {len(lstm_para_targets)} LSTM models in parallel…"):
                lstm_para_result = train_all_lstm_parallel(
                    sip_series=sip_series_ll,
                    mip_series=mip_series_ll,
                    demand_series=demand_series_ll if has_demand else None,
                    gen_series=gen_series_rb,
                    targets=lstm_para_targets,
                    param_search_mode=lstm_para_mode,
                    selected_lookback=st.session_state.get(SELECTED_LOOKBACK),
                    progress_callbacks=lstm_para_cbs,
                    exog_series=st.session_state.get(EXOG_SERIES) or None,
                )

            for t, trained_ll in lstm_para_result.models.items():
                st.session_state[f"_lstm_trained_models_{t}"] = trained_ll
            for t, err in lstm_para_result.errors.items():
                st.error(f"{_target_labels_all.get(t, t)} LSTM training failed: {err}")

            ok_ll = list(lstm_para_result.models.keys())
            if ok_ll:
                st.success(
                    f"LSTM parallel training complete in {lstm_para_result.elapsed_seconds:.0f}s — "
                    f"trained: {', '.join(_target_labels_all.get(t, t) for t in ok_ll)}"
                )

    # ── Model Diagnostics (XGBoost / LSTM) ───────────────────────────
    if "xgb" in selected_methods:
        _trained_diag = st.session_state.get(f"_xgb_trained_models_{target_key}")
        if _trained_diag is not None and getattr(_trained_diag, "grid_search_history", None):
            st.markdown("---")
            st.subheader("🔍 Model Diagnostics")
            st.caption(
                "Diagnostics are generated from the last training run. "
                "Re-train to refresh after changing data or parameters."
            )

            diag_tab1, diag_tab2, diag_tab3 = st.tabs([
                "Overfitting Gauge", "Parameter Sensitivity", "Residual Diagnostics"
            ])

            with diag_tab1:
                st.caption(
                    "Train MAE (in-sample) vs Validation MAE (held-out 20%) for each "
                    "lookback × horizon cell. A large gap indicates overfitting."
                )
                ov_rows = []
                for lb, h_dict in _trained_diag.best_scores.items():
                    for h_sps, val_mae in h_dict.items():
                        train_mae = (_trained_diag.train_scores or {}).get(lb, {}).get(h_sps, float("nan"))
                        ratio = val_mae / train_mae if (train_mae and train_mae > 0) else float("nan")
                        ov_rows.append({
                            "Lookback": lb, "Horizon (d)": h_sps // 48,
                            "Label": f"{lb} / {h_sps // 48}d",
                            "Train MAE": train_mae, "Val MAE": val_mae,
                            "Overfit ratio": ratio,
                        })
                if ov_rows:
                    df_ov = pd.DataFrame(ov_rows).dropna(subset=["Train MAE", "Val MAE"])
                    labels = df_ov["Label"].tolist()
                    fig_ov = go.Figure()
                    fig_ov.add_bar(name="Train MAE", x=labels, y=df_ov["Train MAE"].tolist(),
                                   marker_color=COLOUR_PRIMARY)
                    fig_ov.add_bar(name="Val MAE", x=labels, y=df_ov["Val MAE"].tolist(),
                                   marker_color=COLOUR_WARNING)
                    fig_ov.update_layout(
                        template=PLOTLY_TEMPLATE, barmode="group",
                        title="Train vs Validation MAE per (Lookback, Horizon)",
                        xaxis_title="Lookback / Horizon", yaxis_title="MAE (£/MWh or MW)",
                        height=400,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )
                    st.plotly_chart(fig_ov, use_container_width=True)
                    worst_ratio = df_ov["Overfit ratio"].max()
                    if worst_ratio > 2.0:
                        st.error(f"Overfit ratio up to **{worst_ratio:.1f}×** — model is heavily overfitting. Try higher regularisation (reg_alpha, reg_lambda, max_depth) or a shorter lookback.")
                    elif worst_ratio > 1.3:
                        st.warning(f"Overfit ratio up to **{worst_ratio:.1f}×** — mild overfitting. Consider regularisation tuning.")
                    else:
                        st.success(f"Overfit ratio ≤ **{worst_ratio:.2f}×** — train/val gap looks healthy.")

            with diag_tab2:
                st.caption(
                    "Each point is one hyperparameter combo tried during grid search. "
                    "Lower validation score = better. Select a parameter to inspect its effect."
                )
                hist_rows = []
                for lb, h_dict in _trained_diag.grid_search_history.items():
                    for h_sps, combos in h_dict.items():
                        for (params, val_score) in combos:
                            if val_score < 1e9:
                                hist_rows.append({
                                    **params,
                                    "val_score": val_score,
                                    "Lookback": lb,
                                    "Horizon (d)": h_sps // 48,
                                })
                if not hist_rows:
                    st.info(
                        "No valid parameter combinations recorded. "
                        "This usually means early stopping failed during training. "
                        "Retrain the model to populate this chart."
                    )
                else:
                    df_hist = pd.DataFrame(hist_rows)
                    param_cols = [c for c in df_hist.columns
                                  if c not in ("val_score", "Lookback", "Horizon (d)")]
                    chosen_param = st.selectbox(
                        "Hyperparameter to inspect", param_cols,
                        key="rb_diag_param_select",
                    )
                    if chosen_param:
                        fig_ps = go.Figure()
                        for lb_val in df_hist["Lookback"].unique():
                            sub = df_hist[df_hist["Lookback"] == lb_val]
                            fig_ps.add_trace(go.Scatter(
                                x=sub[chosen_param], y=sub["val_score"],
                                mode="markers", name=lb_val, opacity=0.7,
                                marker=dict(size=6),
                            ))
                        fig_ps.update_layout(
                            template=PLOTLY_TEMPLATE,
                            title=f"Validation Score vs {chosen_param}",
                            xaxis_title=chosen_param,
                            yaxis_title="Validation MAE (worst-window)",
                            height=420,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        )
                        st.plotly_chart(fig_ps, use_container_width=True)
                        st.caption(f"Total combos shown: {len(hist_rows)}")

            with diag_tab3:
                st.caption(
                    "Computed from backtest errors. "
                    "**Bias** = systematic over/under-prediction. "
                    "**Lag-1 ACF** = residual autocorrelation (high → model missed trends)."
                )
                _diag_errors = getattr(_trained_diag, "backtest_errors", [])
                if _diag_errors:
                    from collections import defaultdict as _dd
                    lb_errs: dict = _dd(list)
                    for row in _diag_errors:
                        lb_errs[row.lookback_label].append(row.forecast_mae - row.market_mae)
                    res_rows = []
                    for lb_l, diffs in lb_errs.items():
                        arr = np.array(diffs, dtype=float)
                        bias = float(np.mean(arr))
                        acf1 = (float(np.corrcoef(arr[:-1], arr[1:])[0, 1])
                                if len(arr) > 2 else float("nan"))
                        mean_mae = float(np.mean([r.forecast_mae for r in _diag_errors
                                                   if r.lookback_label == lb_l]))
                        res_rows.append({
                            "Lookback": lb_l, "Bias (fc−mkt)": round(bias, 3),
                            "Lag-1 ACF": round(acf1, 3), "Mean Forecast MAE": round(mean_mae, 3),
                        })
                    if res_rows:
                        st.dataframe(pd.DataFrame(res_rows), use_container_width=True, hide_index=True)
                        for r in res_rows:
                            lb_l = r["Lookback"]
                            acf = r["Lag-1 ACF"]
                            bias = r["Bias (fc−mkt)"]
                            if abs(acf) > 0.3:
                                st.warning(f"**{lb_l}**: Lag-1 ACF = {acf:.2f} — errors are serially correlated. Model may be missing a trend or seasonal pattern.")
                            if abs(bias) > 1.0:
                                st.info(f"**{lb_l}**: Bias = {bias:.2f} — forecast is systematically {'above' if bias > 0 else 'below'} market benchmark.")
                else:
                    st.info("No backtest errors available. Show Results first.")

    # ── Results ───────────────────────────────────────────────────────
    rb_key_prefix = st.session_state.get("_rb_last_key")
    if rb_key_prefix is None or f"{rb_key_prefix}_errors" not in st.session_state:
        st.info("Configure and run the rolling backtest above.")
        return

    errors = st.session_state[f"{rb_key_prefix}_errors"]
    crossovers = st.session_state[f"{rb_key_prefix}_crossovers"]

    is_demand = rb_key_prefix and "_demand_" in rb_key_prefix
    is_gen = rb_key_prefix and "_total_generation_" in rb_key_prefix
    unit = "MW" if (is_demand or is_gen) else "£/MWh"
    is_mip = rb_key_prefix and "_mip_" in rb_key_prefix

    if not errors:
        st.warning("No results — insufficient data for any configuration.")
        return

    # ══════════════════════════════════════════════════════════════════
    # Section 1: Crossover Summary
    # ══════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("1. Maximum Exploitable Forecast Horizon")
    st.caption(
        "The crossover point is where our forecast error first exceeds the "
        "market's. Beyond this horizon, the benchmark is a better predictor."
    )

    cross_rows = []
    for c in crossovers:
        if c.crossover_day >= 15:
            verdict = "We beat the market at all tested horizons (1-14 days)"
        elif c.crossover_day <= 1:
            verdict = "Market wins immediately — no exploitable horizon"
        else:
            verdict = f"Alpha up to **{c.crossover_day - 1} days** ahead"
        cross_rows.append({
            "Lookback": c.lookback_label,
            "Crossover Day": c.crossover_day if c.crossover_day < 15 else ">14",
            "Last +Alpha": f"{c.last_positive_alpha:+.2f} {unit}",
            "First −Alpha": f"{c.first_negative_alpha:+.2f} {unit}" if c.crossover_day < 15 else "N/A",
            "Verdict": verdict,
        })
    st.dataframe(pd.DataFrame(cross_rows), use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════
    # Section 2: Alpha Heatmap (Lookback × Horizon)
    # ══════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("2. Alpha Heatmap — Lookback × Forecast Horizon")
    st.caption("Green = our forecast beats the benchmark. Red = benchmark wins.")

    error_matrix = build_error_matrix(errors)
    if not error_matrix.empty:
        z = error_matrix.values.astype(float)
        text = [[f"{v:+.2f}" if not np.isnan(v) else "N/A" for v in row] for row in z]

        fig_hm = go.Figure()
        fig_hm.add_trace(go.Heatmap(
            z=z,
            x=list(error_matrix.columns),
            y=list(error_matrix.index),
            colorscale=[
                [0.0, COLOUR_DANGER],
                [0.5, "#2C3E50"],
                [1.0, COLOUR_SUCCESS],
            ],
            zmid=0,
            text=text,
            texttemplate="%{text}",
            colorbar=dict(title=f"Alpha ({unit})"),
        ))
        fig_hm.update_layout(
            template=PLOTLY_TEMPLATE,
            margin=dict(l=50, r=30, t=50, b=50),
            title="Alpha: Lookback × Forecast Horizon (days)",
            xaxis_title="Forecast Horizon",
            yaxis_title="Lookback Window",
            height=350,
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════
    # Section 3: Error Curves — Each Lookback
    # ══════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("3. Forecast vs Market Error by Horizon")
    st.caption(
        "The point where the solid line crosses above the dashed line is "
        "where the market becomes a better predictor than our model."
    )

    lb_colours = {
        "1 day": COLOUR_PRIMARY,
        "5 days": COLOUR_MUTED,
        "15 days": COLOUR_WARNING,
        "30 days": COLOUR_SUCCESS,
    }

    fig_err = go.Figure()
    for lb_label in ROLLING_LOOKBACKS:
        lb_rows = sorted(
            [e for e in errors if e.lookback_label == lb_label],
            key=lambda e: e.horizon_days,
        )
        if not lb_rows:
            continue

        days = [e.horizon_days for e in lb_rows]
        fc_maes = [e.forecast_mae for e in lb_rows]
        mkt_maes = [e.market_mae for e in lb_rows]

        colour = lb_colours.get(lb_label, COLOUR_MUTED)

        fig_err.add_trace(go.Scatter(
            x=days, y=fc_maes,
            mode="lines+markers", name=f"Our Forecast ({lb_label})",
            line=dict(color=colour, width=2),
            marker=dict(size=6),
        ))
        fig_err.add_trace(go.Scatter(
            x=days, y=mkt_maes,
            mode="lines+markers", name=f"Market ({lb_label})",
            line=dict(color=colour, width=1, dash="dash"),
            marker=dict(size=4, symbol="x"),
        ))

    fig_err.update_layout(
        template=PLOTLY_TEMPLATE,
        margin=dict(l=50, r=30, t=50, b=50),
        title="MAE by Forecast Horizon (days ahead)",
        xaxis_title="Forecast Horizon (days)",
        yaxis_title=f"MAE ({unit})",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_err, use_container_width=True)

    # MAPE chart
    fig_mape = go.Figure()
    for lb_label in ROLLING_LOOKBACKS:
        lb_rows = sorted(
            [e for e in errors if e.lookback_label == lb_label],
            key=lambda e: e.horizon_days,
        )
        if not lb_rows:
            continue

        days = [e.horizon_days for e in lb_rows]
        fc_mapes = [e.forecast_mape for e in lb_rows]
        mkt_mapes = [e.market_mape for e in lb_rows]

        colour = lb_colours.get(lb_label, COLOUR_MUTED)

        fig_mape.add_trace(go.Scatter(
            x=days, y=fc_mapes,
            mode="lines+markers", name=f"Our Forecast ({lb_label})",
            line=dict(color=colour, width=2),
            marker=dict(size=6),
        ))
        fig_mape.add_trace(go.Scatter(
            x=days, y=mkt_mapes,
            mode="lines+markers", name=f"Market ({lb_label})",
            line=dict(color=colour, width=1, dash="dash"),
            marker=dict(size=4, symbol="x"),
        ))

    fig_mape.update_layout(
        template=PLOTLY_TEMPLATE,
        margin=dict(l=50, r=30, t=50, b=50),
        title="MAPE by Forecast Horizon (days ahead)",
        xaxis_title="Forecast Horizon (days)",
        yaxis_title="MAPE (%)",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_mape, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════
    # Section 4: Alpha Decay Curves
    # ══════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("4. Alpha Decay by Horizon")
    st.caption(
        "How our forecast advantage (alpha) decays with increasing horizon. "
        "Positive = we beat the market. Where it crosses zero = crossover."
    )

    fig_alpha = go.Figure()
    for lb_label in ROLLING_LOOKBACKS:
        lb_rows = sorted(
            [e for e in errors if e.lookback_label == lb_label],
            key=lambda e: e.horizon_days,
        )
        if not lb_rows:
            continue

        days = [e.horizon_days for e in lb_rows]
        alphas = [e.alpha_mae for e in lb_rows]
        colour = lb_colours.get(lb_label, COLOUR_MUTED)

        fig_alpha.add_trace(go.Scatter(
            x=days, y=alphas,
            mode="lines+markers", name=lb_label,
            line=dict(color=colour, width=2),
            marker=dict(size=7),
            fill="tozeroy",
            fillcolor=f"rgba({int(colour[1:3],16)},{int(colour[3:5],16)},{int(colour[5:7],16)},0.08)",
        ))

    fig_alpha.add_hline(y=0, line_color="white", line_width=1, line_dash="dash")
    fig_alpha.update_layout(
        template=PLOTLY_TEMPLATE,
        margin=dict(l=50, r=30, t=50, b=50),
        title="Alpha (Market MAE − Forecast MAE) by Horizon",
        xaxis_title="Forecast Horizon (days)",
        yaxis_title=f"Alpha ({unit})",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_alpha, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════
    # Section 5: Detailed Results Table
    # ══════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("5. Full Results")

    detail_rows = []
    for e in sorted(errors, key=lambda e: (e.lookback_label, e.horizon_days)):
        sig = "Yes" if e.dm_pvalue < 0.05 and e.alpha_mae > 0 else "No"
        detail_rows.append({
            "Lookback": e.lookback_label,
            "Horizon": f"{e.horizon_days}d",
            f"Forecast MAE ({unit})": f"{e.forecast_mae:.2f}",
            f"Benchmark MAE ({unit})": f"{e.market_mae:.2f}",
            f"Alpha ({unit})": f"{e.alpha_mae:+.2f}",
            "Forecast MAPE": f"{e.forecast_mape:.1f}%",
            "Benchmark MAPE": f"{e.market_mape:.1f}%",
            "Alpha (MAPE)": f"{e.alpha_mape:+.1f}pp",
            "DM p-value": f"{e.dm_pvalue:.4f}",
            "Significant": sig,
            "N Obs": e.n_obs,
        })

    st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)

    csv_data = pd.DataFrame(detail_rows).to_csv(index=False)
    st.download_button(
        "📥 Export Rolling Backtest Results (CSV)",
        csv_data,
        file_name="rolling_backtest_results.csv",
        mime="text/csv",
    )

    with st.expander("Methodology & Interpretation"):
        st.markdown("""
**What this tool does:**
- For each of the three lookback windows (1, 15, 30 days), we make forecasts
  at horizons of 1 through 14 days ahead.
- All forecast origins are **daily** (step=48 SPs) so that observations are
  non-overlapping and statistical tests are valid.
- We compare our forecast error (MAE) against the market's forward curve error
  (MIP TOD-mean benchmark) at each horizon.

**The crossover point:**
- Where our forecast MAE first exceeds the market's MAE, our alpha disappears.
- Before the crossover, there is a **market inefficiency** we can exploit.
- After the crossover, we should defer to the forward curve.

**How to use this for trading:**
- If the 15-day lookback shows alpha up to 3 days ahead, commit positions
  no more than 3 days forward.
- If the 1-day lookback loses to the market immediately, it's overfitting
  to recent noise.
- If the 30-day lookback shows alpha at all horizons, the market may have
  a persistent structural inefficiency at these timescales.

**Statistical validity:**
- DM p-values use Newey-West HAC standard errors.
- "Significant" means p < 0.05 AND alpha > 0.

**XGBoost method:**
- Trains a fresh gradient-boosted tree per (lookback, horizon) at each origin.
- Features: time-of-day, day-of-week, lags (1d/2d/7d), rolling stats, MIP, demand.
- **In-depth search:** systematic grid over core hyperparams (27 combos).
- **Quick random search:** samples all hyperparams (30 combos) for fast estimation.
- Assessment criterion: **worst-window MAE** (worst consecutive 3-day stretch) — ensures
  robustness at any point along the forward curve.

**NeuralProphet method:**
- Combines AR-Net (autoregression) with Prophet's trend + seasonality decomposition.
- Captures half-hourly and weekly patterns natively via Fourier terms.
- Heavier than TOD Mean/EWMA but lighter than XGBoost; good for structured patterns.
- Falls back to TOD Mean if neuralprophet is not installed.

**Demand target:**
- When forecasting demand, the benchmark is the TOD-mean of demand itself
  (how well does a simple seasonal average predict demand).
- Demand is smoother than SIP, so baseline models perform better —
  XGBoost/NeuralProphet add value by capturing weather/calendar effects.

**Wholesale (MIP) target:**
- MIP (Market Index Price) is the GB wholesale reference price derived from
  power exchange trades — the best available proxy for the day-ahead wholesale curve.
- When forecasting MIP, SIP and demand serve as auxiliary features.
- The benchmark is the TOD-mean of MIP itself.
- Useful for identifying wholesale vs. balancing market price spreads.
""")

    # ══════════════════════════════════════════════════════════════════
    # Section 6: 14-Day Forward Forecast (XGBoost only)
    # ══════════════════════════════════════════════════════════════════
    _render_forward_forecast()

    # ══════════════════════════════════════════════════════════════════
    # Section 7: Combined SIP + Demand Alpha Analysis
    # ══════════════════════════════════════════════════════════════════
    _render_combined_analysis()


def _display_training_summary(trained) -> None:
    """Show a compact summary of XGBoost best params from grid search."""
    summary_rows = []
    for lb_label in trained.best_params:
        for h_sps, params in trained.best_params[lb_label].items():
            h_days = h_sps // 48
            score = trained.best_scores.get(lb_label, {}).get(h_sps, float("nan"))
            summary_rows.append({
                "Lookback": lb_label,
                "Horizon": f"{h_days}d",
                "n_estimators": params.get("n_estimators"),
                "max_depth": params.get("max_depth"),
                "learning_rate": params.get("learning_rate"),
                "Worst-Window MAE": f"{score:.2f}" if score != float("inf") else "N/A",
            })
    if summary_rows:
        with st.expander("Grid Search Results — Best XGBoost Params per (Lookback, Horizon)"):
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


def _display_np_training_summary(trained) -> None:
    """Show a compact summary of NeuralProphet best params from grid search."""
    summary_rows = []
    for lb_label, params in trained.best_params.items():
        score = trained.best_scores.get(lb_label, float("nan"))
        n_fwd = len(trained.forward_forecasts.get(lb_label, {}))
        summary_rows.append({
            "Lookback": lb_label,
            "n_lags": params.get("n_lags"),
            "epochs": params.get("epochs"),
            "learning_rate": params.get("learning_rate"),
            "Worst-Window MAE": f"{score:.2f}" if score not in (float("inf"), float("nan")) else "N/A",
            "Forward Forecast Days": n_fwd,
        })
    if summary_rows:
        with st.expander("NeuralProphet Grid Search — Best Params per Lookback"):
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


def _display_lstm_training_summary(trained) -> None:
    """Show a compact summary of LSTM best hyperparams from grid search."""
    summary_rows = []
    for lb_label in trained.best_params:
        for h_sps, params in trained.best_params[lb_label].items():
            h_days = h_sps // 48
            score = trained.best_scores.get(lb_label, {}).get(h_sps, float("nan"))
            summary_rows.append({
                "Lookback":    lb_label,
                "Horizon":     f"{h_days}d",
                "hidden_size": params.get("hidden_size"),
                "num_layers":  params.get("num_layers"),
                "seq_len":     params.get("seq_len"),
                "dropout":     params.get("dropout"),
                "lr":          params.get("lr"),
                "Val Huber Loss": f"{score:.4f}" if score not in (float("inf"), float("nan")) else "N/A",
            })
    if summary_rows:
        with st.expander("Grid Search Results — Best LSTM Params per (Lookback, Horizon)"):
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


def _render_forward_forecast() -> None:
    """Show 14-day forward forecasts from trained XGBoost, LSTM and/or NeuralProphet models."""
    sip_df = st.session_state.get(SIP_DF)
    mip_df = st.session_state.get(MIP_DF)
    demand_df = st.session_state.get(DEMAND_DF)

    if sip_df is None or sip_df.empty or mip_df is None or mip_df.empty:
        return

    # Collect all available trained models
    xgb_models = {
        "Price (SIP)":      (st.session_state.get("_xgb_trained_models_sip"),    "£/MWh", "Predicted SIP (£/MWh)"),
        "Wholesale (MIP)":  (st.session_state.get("_xgb_trained_models_mip"),    "£/MWh", "Predicted MIP (£/MWh)"),
        "Demand":           (st.session_state.get("_xgb_trained_models_demand"), "MW",    "Predicted Demand (MW)"),
    }
    lstm_models = {
        "Price (SIP)":      (st.session_state.get("_lstm_trained_models_sip"),    "£/MWh", "Predicted SIP (£/MWh)"),
        "Wholesale (MIP)":  (st.session_state.get("_lstm_trained_models_mip"),    "£/MWh", "Predicted MIP (£/MWh)"),
        "Demand":           (st.session_state.get("_lstm_trained_models_demand"), "MW",    "Predicted Demand (MW)"),
    }
    np_models = {
        "Price (SIP)":      (st.session_state.get("_np_trained_models_sip"),     "£/MWh", "Predicted SIP (£/MWh)"),
        "Wholesale (MIP)":  (st.session_state.get("_np_trained_models_mip"),     "£/MWh", "Predicted MIP (£/MWh)"),
        "Demand":           (st.session_state.get("_np_trained_models_demand"),  "MW",    "Predicted Demand (MW)"),
    }

    has_xgb  = any(t is not None and t.final_models for t, _, _ in xgb_models.values())
    has_lstm = any(t is not None and t.final_state_dicts for t, _, _ in lstm_models.values())
    has_np   = any(t is not None and t.forward_forecasts for t, _, _ in np_models.values())

    if not has_xgb and not has_lstm and not has_np:
        return

    from src.models.forecaster import build_aligned_series
    from src.models.xgb_trainer import _align_to_target as _xgb_align
    from src.models.lstm_trainer import _align_to_target as _lstm_align

    sip_series, mip_series, demand_series, _ = build_aligned_series(
        sip_df, mip_df, demand_df=demand_df,
    )
    sip_values    = sip_series.values.astype(float)
    mip_values    = mip_series.values.astype(float)
    demand_values = demand_series.values.astype(float) if demand_series is not None else None
    last_date     = sip_series.index[-1]

    _stored_exog: dict = st.session_state.get(EXOG_SERIES) or {}

    from src.models.xgb_trainer import TrainedXGBModels as _TrainedXGBModels

    def _build_exog(trained, target_series):
        """Reconstruct exog_dict for a trained model using stored exog series."""
        exog_keys = getattr(trained, "exog_keys", [])
        if not exog_keys or not _stored_exog:
            return None
        align_fn = _xgb_align if isinstance(trained, _TrainedXGBModels) else _lstm_align
        result = {k: align_fn(target_series, _stored_exog[k]) for k in exog_keys if k in _stored_exog}
        return result if result else None

    lb_colours = {
        "1 day":  COLOUR_PRIMARY,
        "5 days": COLOUR_MUTED,
        "15 days": COLOUR_WARNING,
        "30 days": COLOUR_SUCCESS,
    }

    st.markdown("---")
    st.subheader("6. 14-Day Forward Forecast")
    st.caption(
        "Point forecasts from the latest data point using trained models. "
        "XGBoost, LSTM and NeuralProphet forecasts are shown side-by-side when available."
    )

    _label_to_series = {
        "Price (SIP)":     sip_series,
        "Wholesale (MIP)": mip_series,
        "Demand":          demand_series if demand_series is not None else sip_series,
    }

    labels_shown = set()
    for label in ["Price (SIP)", "Wholesale (MIP)", "Demand"]:
        xgb_trained,  unit, y_label = xgb_models[label]
        lstm_trained, _,    _       = lstm_models[label]
        np_trained,   _,    _       = np_models[label]

        _ref_series = _label_to_series[label]

        xgb_fc = forecast_forward(
            xgb_trained, sip_values, mip_values, demand_values, n_days=14,
            exog_dict=_build_exog(xgb_trained, _ref_series),
        ) if xgb_trained is not None and xgb_trained.final_models else {}
        lstm_fc = lstm_forecast_forward(
            lstm_trained, sip_values, mip_values, demand_values, n_days=14,
            exog_dict=_build_exog(lstm_trained, _ref_series),
        ) if lstm_trained is not None and lstm_trained.final_state_dicts else {}
        np_fc  = forecast_forward_np(np_trained) \
                 if np_trained is not None and np_trained.forward_forecasts else {}

        if not xgb_fc and not lstm_fc and not np_fc:
            continue

        labels_shown.add(label)
        fig = go.Figure()

        for lb_label, day_forecasts in xgb_fc.items():
            if not day_forecasts:
                continue
            days   = sorted(day_forecasts.keys())
            dates  = [last_date + pd.Timedelta(days=d) for d in days]
            vals   = [day_forecasts[d] for d in days]
            colour = lb_colours.get(lb_label, COLOUR_MUTED)
            fig.add_trace(go.Scatter(
                x=dates, y=vals,
                mode="lines+markers",
                name=f"XGB {lb_label}",
                line=dict(color=colour, width=2),
                marker=dict(size=6),
            ))

        for lb_label, day_forecasts in lstm_fc.items():
            if not day_forecasts:
                continue
            days   = sorted(day_forecasts.keys())
            dates  = [last_date + pd.Timedelta(days=d) for d in days]
            vals   = [day_forecasts[d] for d in days]
            colour = lb_colours.get(lb_label, COLOUR_MUTED)
            fig.add_trace(go.Scatter(
                x=dates, y=vals,
                mode="lines+markers",
                name=f"LSTM {lb_label}",
                line=dict(color=colour, width=2, dash="dash"),
                marker=dict(size=6, symbol="square"),
            ))

        for lb_label, day_forecasts in np_fc.items():
            if not day_forecasts:
                continue
            days   = sorted(day_forecasts.keys())
            dates  = [last_date + pd.Timedelta(days=d) for d in days]
            vals   = [day_forecasts[d] for d in days]
            colour = lb_colours.get(lb_label, COLOUR_MUTED)
            fig.add_trace(go.Scatter(
                x=dates, y=vals,
                mode="lines+markers",
                name=f"NP {lb_label}",
                line=dict(color=colour, width=2, dash="dot"),
                marker=dict(size=5, symbol="diamond"),
            ))

        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            title=f"{label} Forward Forecast — Next 14 Days",
            xaxis_title="Date",
            yaxis_title=y_label,
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Combined data table
        fc_rows = []
        for lb_label, day_forecasts in xgb_fc.items():
            for d in sorted(day_forecasts.keys()):
                fc_rows.append({
                    "Model": f"XGBoost ({lb_label})",
                    "Day Ahead": d,
                    "Date": (last_date + pd.Timedelta(days=d)).strftime("%Y-%m-%d"),
                    f"Forecast ({unit})": f"{day_forecasts[d]:.2f}",
                })
        for lb_label, day_forecasts in lstm_fc.items():
            for d in sorted(day_forecasts.keys()):
                fc_rows.append({
                    "Model": f"LSTM ({lb_label})",
                    "Day Ahead": d,
                    "Date": (last_date + pd.Timedelta(days=d)).strftime("%Y-%m-%d"),
                    f"Forecast ({unit})": f"{day_forecasts[d]:.2f}",
                })
        for lb_label, day_forecasts in np_fc.items():
            for d in sorted(day_forecasts.keys()):
                fc_rows.append({
                    "Model": f"NeuralProphet ({lb_label})",
                    "Day Ahead": d,
                    "Date": (last_date + pd.Timedelta(days=d)).strftime("%Y-%m-%d"),
                    f"Forecast ({unit})": f"{day_forecasts[d]:.2f}",
                })
        if fc_rows:
            with st.expander(f"{label} Forward Forecast Data"):
                st.dataframe(pd.DataFrame(fc_rows), use_container_width=True, hide_index=True)


def _render_combined_analysis() -> None:
    """Show joint SIP + MIP + Demand alpha when multiple backtests have been run."""
    sip_keys = [k for k in st.session_state if k.startswith("_rb_sip_") and k.endswith("_errors")]
    mip_keys = [k for k in st.session_state if k.startswith("_rb_mip_") and k.endswith("_errors")]
    demand_keys = [k for k in st.session_state if k.startswith("_rb_demand_") and k.endswith("_errors")]

    available = {}
    if sip_keys:
        available["Price (SIP)"] = (sip_keys[-1], COLOUR_PRIMARY)
    if mip_keys:
        available["Wholesale (MIP)"] = (mip_keys[-1], COLOUR_WARNING)
    if demand_keys:
        available["Demand"] = (demand_keys[-1], COLOUR_SUCCESS)

    if len(available) < 2:
        return

    st.markdown("---")
    st.subheader("7. Combined Alpha Analysis")
    st.caption(
        "When multiple forecast targets have been backtested, this section "
        "shows where you have forecast alpha on **multiple** axes simultaneously — "
        "essential for capacity allocation decisions."
    )

    best_lb = "30 days"

    alpha_by_target: dict[str, dict[int, float]] = {}
    methods: dict[str, str] = {}
    for label, (key, _colour) in available.items():
        errors = st.session_state[key]
        short_key = key.split("_rb_")[1].replace("_errors", "")
        parts = short_key.split("_", 1)
        methods[label] = parts[1] if len(parts) > 1 else parts[0]

        by_h = {e.horizon_days: e.alpha_mae for e in errors if e.lookback_label == best_lb}
        if not by_h:
            for lb in ["15 days", "5 days", "1 day"]:
                by_h = {e.horizon_days: e.alpha_mae for e in errors if e.lookback_label == lb}
                if by_h:
                    break
        alpha_by_target[label] = by_h

    all_days: set[int] = set()
    for by_h in alpha_by_target.values():
        all_days.update(by_h.keys())
    common_days = sorted(all_days)
    if not common_days:
        st.info("No overlapping horizons between backtests.")
        return

    fig = go.Figure()
    for label, (_key, colour) in available.items():
        by_h = alpha_by_target.get(label, {})
        fig.add_trace(go.Bar(
            x=[f"{d}d" for d in common_days],
            y=[by_h.get(d, 0) for d in common_days],
            name=f"{label} ({methods.get(label, '')})",
            marker_color=colour,
            opacity=0.8,
        ))

    fig.add_hline(y=0, line_color="white", line_width=1, line_dash="dash")
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        barmode="group",
        title=f"Joint Alpha by Horizon (lookback: {best_lb})",
        xaxis_title="Forecast Horizon",
        yaxis_title="Alpha (lower error than benchmark)",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    all_labels = list(available.keys())
    all_positive_days = [
        d for d in common_days
        if all(alpha_by_target.get(lbl, {}).get(d, -1) > 0 for lbl in all_labels)
    ]
    if all_positive_days:
        st.success(
            f"**Joint alpha detected at horizons:** {', '.join(f'{d}d' for d in all_positive_days)}. "
            f"At these horizons, all available forecasts beat their benchmarks — "
            f"these are the strongest signals for capacity allocation."
        )
    else:
        st.warning(
            "No horizons found where all available forecasts simultaneously "
            "beat their benchmarks. Consider adjusting methods or lookback windows."
        )

    combo_rows = []
    for d in common_days:
        row: dict[str, str] = {"Horizon": f"{d}d"}
        total = 0.0
        all_pos = True
        for label in all_labels:
            a = alpha_by_target.get(label, {}).get(d, 0.0)
            unit_str = "MW" if label == "Demand" else "£/MWh"
            row[f"{label} Alpha ({unit_str})"] = f"{a:+.2f}"
            total += a
            if a <= 0:
                all_pos = False
        row["Combined Score"] = f"{total:+.2f}"
        row["All Positive"] = "✅" if all_pos else "❌"
        combo_rows.append(row)
    st.dataframe(pd.DataFrame(combo_rows), use_container_width=True, hide_index=True)
