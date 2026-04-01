"""
Dispatch Decision Engine for Ohme EV Fleet.

Combines forecasts of:
  - GB system supply (generation by fuel type: wind, thermal, interconnectors, storage)
  - GB system demand (INDO)
  - System Imbalance Price (SIP / SBP / SSP)
  - Market Index Price (MIP / APXMIDP)

to recommend, for each future settlement period, whether the fleet should:
  - HOLD     : meet gate-closure commitment exactly (no imbalance)
  - CHARGE_MORE : increase charging to absorb excess supply (long system)
  - CHARGE_LESS : reduce charging to provide demand flexibility (short system)

NIV (Net Imbalance Volume) forecast = total_gen_forecast − demand_forecast.
  NIV > 0  → system long  (surplus generation; SSP depressed; absorbing pays little)
  NIV < 0  → system short (tight supply; SBP elevated; reducing demand earns premium)

P&L delta per MW of deployed flexibility (per 30-min SP):
  CHARGE_MORE : SSP_forecast × 0.5 h   (earn settlement for surplus consumption)
  CHARGE_LESS : (SBP_forecast − MIP_forecast) × 0.5 h  (earn imbalance premium, give up DA revenue)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Fuel type categories ───────────────────────────────────────────────────────

THERMAL_FUELS = frozenset({"CCGT", "COAL", "OCGT", "OIL", "NUCLEAR", "BIOMASS", "OTHER"})
RENEWABLE_FUELS = frozenset({"WIND"})          # embedded solar not in FUELHH
STORAGE_FUELS = frozenset({"PS", "NPSHYD"})
INTERCONNECTOR_FUELS = frozenset({
    "INTELEC", "INTEW", "INTFR", "INTGRNL", "INTIFA2",
    "INTIRL", "INTNED", "INTNEM", "INTNSL", "INTVKL",
})

# ── Thresholds ─────────────────────────────────────────────────────────────────

# MW — NIV magnitude needed to trigger a non-HOLD recommendation.
# Below this we consider the system balanced and the signal too noisy.
NIV_THRESHOLD_MW = 300.0

# £/MWh — minimum spread SBP must exceed MIP for CHARGE_LESS to be worthwhile.
# Covers DA revenue given up plus a small buffer for forecast uncertainty.
SBP_PREMIUM_MIN = 5.0


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class GenerationBreakdown:
    """Processed half-hourly generation split by fuel category."""
    index: pd.DatetimeIndex
    wind_mw: np.ndarray
    thermal_mw: np.ndarray
    interconnector_mw: np.ndarray
    storage_mw: np.ndarray
    total_mw: np.ndarray

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "wind_mw":            self.wind_mw,
            "thermal_mw":         self.thermal_mw,
            "interconnector_mw":  self.interconnector_mw,
            "storage_mw":         self.storage_mw,
            "total_mw":           self.total_mw,
        }, index=self.index)


@dataclass
class DispatchRecommendation:
    """Recommendation for a single future settlement period."""
    horizon_sp: int                   # SPs ahead from forecast origin
    forecast_datetime: pd.Timestamp
    # ── Forecast values ──
    wind_forecast_mw: float
    thermal_forecast_mw: float
    interconnector_forecast_mw: float
    total_gen_forecast_mw: float
    demand_forecast_mw: float
    niv_forecast_mw: float            # total_gen − demand (+ve = long)
    sbp_forecast: float               # £/MWh (System Buy Price — short parties pay)
    ssp_forecast: float               # £/MWh (System Sell Price — long parties receive)
    mip_forecast: float               # £/MWh (near-delivery market price)
    # ── Decision ──
    system_position: str              # "LONG" | "SHORT" | "BALANCED"
    action: str                       # "HOLD" | "CHARGE_MORE" | "CHARGE_LESS"
    pnl_delta_per_mw: float          # £/MW deployed for this 30-min SP
    rationale: str


@dataclass
class DailyDispatchSummary:
    """Per-day rollup for the 10-day lookahead view."""
    date: str
    mean_niv_mw: float
    wind_pct: float                   # wind as % of total generation forecast
    thermal_pct: float
    interconnector_pct: float
    mean_sbp: float
    mean_mip: float
    system_stance: str                # "LONG" | "SHORT" | "BALANCED"
    dominant_action: str
    pnl_opportunity_per_mw_day: float # £/MW/day if action deployed every SP
    n_strong_signal_sps: int          # SPs where |NIV| > NIV_THRESHOLD_MW


# ── Generation processing ──────────────────────────────────────────────────────

def process_generation_outturn(fuelhh_df: pd.DataFrame) -> GenerationBreakdown:
    """
    Pivot FUELHH long-format DataFrame into half-hourly GenerationBreakdown.

    Aggregates fuel types into four categories:
      wind, thermal, interconnector (net imports +ve), storage.
    Total = sum of all four.
    """
    if fuelhh_df.empty or "fuelType" not in fuelhh_df.columns:
        empty = np.array([], dtype=float)
        return GenerationBreakdown(pd.DatetimeIndex([]), empty, empty, empty, empty, empty)

    df = fuelhh_df.copy()
    df["datetime"] = pd.to_datetime(df["settlementDate"]) + pd.to_timedelta(
        (df["settlementPeriod"].astype(int) - 1) * 30, unit="min"
    )

    pivot = df.pivot_table(
        index="datetime", columns="fuelType", values="generation", aggfunc="sum"
    ).fillna(0.0)

    def _sum_fuels(fuel_set: frozenset) -> pd.Series:
        cols = [c for c in fuel_set if c in pivot.columns]
        return pivot[cols].sum(axis=1) if cols else pd.Series(0.0, index=pivot.index)

    wind_s   = _sum_fuels(RENEWABLE_FUELS)
    thermal  = _sum_fuels(THERMAL_FUELS)
    intercon = _sum_fuels(INTERCONNECTOR_FUELS)
    storage  = _sum_fuels(STORAGE_FUELS)
    total    = wind_s + thermal + intercon + storage

    idx = pivot.index.sort_values()
    return GenerationBreakdown(
        index=idx,
        wind_mw=wind_s.reindex(idx).values,
        thermal_mw=thermal.reindex(idx).values,
        interconnector_mw=intercon.reindex(idx).values,
        storage_mw=storage.reindex(idx).values,
        total_mw=total.reindex(idx).values,
    )


def align_generation_to_series(
    gen_breakdown: GenerationBreakdown,
    reference_index: pd.DatetimeIndex,
) -> GenerationBreakdown:
    """Re-index GenerationBreakdown onto a reference datetime index (forward-fill gaps)."""
    df = gen_breakdown.as_dataframe()
    df = df.reindex(reference_index).ffill().bfill().fillna(0.0)
    return GenerationBreakdown(
        index=df.index,
        wind_mw=df["wind_mw"].values,
        thermal_mw=df["thermal_mw"].values,
        interconnector_mw=df["interconnector_mw"].values,
        storage_mw=df["storage_mw"].values,
        total_mw=df["total_mw"].values,
    )


# ── TOD-mean forecasting ───────────────────────────────────────────────────────

def _tod_mean(values: np.ndarray, origin_idx: int, horizons: List[int],
              lookback_sps: int = 336) -> Dict[int, float]:
    """
    Time-of-Day rolling mean: for each horizon h, average historical values
    at the same half-hour-of-day within the lookback window.
    Identical logic to forecaster._tod_mean_forecast but takes any array.
    """
    n = len(values)
    out: Dict[int, float] = {}
    for h in horizons:
        target_idx = origin_idx + h
        if target_idx >= n:
            continue
        sp_of_day = target_idx % 48
        start = max(0, origin_idx - lookback_sps)
        window = values[start:origin_idx]
        if len(window) == 0:
            continue
        sp_indices = np.arange(start, origin_idx)
        mask = (sp_indices % 48) == sp_of_day
        out[h] = float(np.mean(window[mask]) if mask.any() else np.mean(window))
    return out


def _forward_tod_forecasts(
    values: np.ndarray,
    origin_idx: int,
    n_sps: int,
    lookback_sps: int = 336,
) -> np.ndarray:
    """
    Produce a forward array of n_sps TOD-mean forecasts starting from origin_idx+1.
    Returns shape (n_sps,).
    """
    horizons = list(range(1, n_sps + 1))
    fc = _tod_mean(values, origin_idx, horizons, lookback_sps)
    result = np.array([fc.get(h, np.nan) for h in horizons])
    # Fill any NaN with the global rolling mean as fallback
    if np.isnan(result).any():
        fallback = float(np.nanmean(values[max(0, origin_idx - lookback_sps):origin_idx]))
        result = np.where(np.isnan(result), fallback, result)
    return result


# ── SSP extraction ─────────────────────────────────────────────────────────────

def build_ssp_series(sip_df: pd.DataFrame, reference_index: pd.DatetimeIndex) -> pd.Series:
    """
    Extract System Sell Price (SSP) from the SIP DataFrame and align to reference_index.
    In GB single cash-out, SSP = SBP = SIP most of the time, but they can differ.
    Falls back to systemBuyPrice if systemSellPrice is absent.
    """
    df = sip_df.copy()
    df["datetime"] = pd.to_datetime(df["settlementDate"]) + pd.to_timedelta(
        (df["settlementPeriod"].astype(int) - 1) * 30, unit="min"
    )
    col = "systemSellPrice" if "systemSellPrice" in df.columns else "systemBuyPrice"
    s = df.set_index("datetime")[col].sort_index()
    s = s[~s.index.duplicated(keep="first")]
    s = pd.to_numeric(s, errors="coerce")
    return s.reindex(reference_index).ffill().bfill()


# ── Core dispatch engine ───────────────────────────────────────────────────────

def compute_dispatch_recommendations(
    sip_series: pd.Series,           # SBP (System Buy Price) aligned
    ssp_series: pd.Series,           # SSP (System Sell Price) aligned
    mip_series: pd.Series,           # Near-delivery market price
    gen_breakdown: GenerationBreakdown,  # processed generation split
    demand_series: pd.Series,        # GB demand (INDO)
    committed_mw: float,             # current gate-closure commitment
    fleet_max_mw: float,             # maximum fleet charging capacity
    fleet_min_mw: float = 0.0,
    lookahead_days: int = 1,         # 1 or 10
    lookback_sps: int = 336,         # 7 days for TOD-mean
    trained_models: Optional[Dict] = None,  # {target: TrainedXGBModels} — future use
) -> Tuple[List[DispatchRecommendation], List[DailyDispatchSummary]]:
    """
    For each future settlement period (lookahead_days × 48 SPs), compute:
      - Decomposed generation forecast (wind, thermal, interconnector)
      - Demand forecast
      - NIV = gen - demand
      - SBP/SSP/MIP forecasts
      - Recommended dispatch action + P&L delta per MW

    Returns (sp_recommendations, daily_summaries).
    """
    n_sps = lookahead_days * 48

    # ── Align all series to a common reference ─────────────────────────────
    ref_idx = sip_series.index
    gen_aligned = align_generation_to_series(gen_breakdown, ref_idx)
    demand_v   = demand_series.reindex(ref_idx).ffill().bfill().values.astype(float)
    sip_v      = sip_series.values.astype(float)
    ssp_v      = ssp_series.reindex(ref_idx).ffill().bfill().values.astype(float)
    mip_v      = mip_series.reindex(ref_idx).ffill().bfill().values.astype(float)

    wind_v     = gen_aligned.wind_mw
    thermal_v  = gen_aligned.thermal_mw
    intercon_v = gen_aligned.interconnector_mw
    storage_v  = gen_aligned.storage_mw
    total_v    = gen_aligned.total_mw

    # NIV = total generation − demand
    niv_v = total_v - demand_v

    origin_idx = len(sip_v) - 1
    origin_dt  = sip_series.index[-1]

    # ── Forward TOD-mean forecasts ─────────────────────────────────────────
    fc_wind     = _forward_tod_forecasts(wind_v,     origin_idx, n_sps, lookback_sps)
    fc_thermal  = _forward_tod_forecasts(thermal_v,  origin_idx, n_sps, lookback_sps)
    fc_intercon = _forward_tod_forecasts(intercon_v, origin_idx, n_sps, lookback_sps)
    fc_storage  = _forward_tod_forecasts(storage_v,  origin_idx, n_sps, lookback_sps)
    fc_demand   = _forward_tod_forecasts(demand_v,   origin_idx, n_sps, lookback_sps)
    fc_sbp      = _forward_tod_forecasts(sip_v,      origin_idx, n_sps, lookback_sps)
    fc_ssp      = _forward_tod_forecasts(ssp_v,      origin_idx, n_sps, lookback_sps)
    fc_mip      = _forward_tod_forecasts(mip_v,      origin_idx, n_sps, lookback_sps)

    # Derived: total gen includes storage (consistent with historical niv_v = total_v - demand_v)
    fc_total_gen = fc_wind + fc_thermal + fc_intercon + fc_storage
    fc_niv       = fc_total_gen - fc_demand

    headroom_up   = max(0.0, fleet_max_mw - committed_mw)
    headroom_down = max(0.0, committed_mw - fleet_min_mw)

    # ── Per-SP recommendations ─────────────────────────────────────────────
    sp_recs: List[DispatchRecommendation] = []
    for i in range(n_sps):
        h = i + 1
        dt_sp = origin_dt + pd.Timedelta(minutes=30 * h)

        niv  = float(fc_niv[i])
        sbp  = float(fc_sbp[i])
        ssp  = float(fc_ssp[i])
        mip  = float(fc_mip[i])
        wind = float(fc_wind[i])
        thm  = float(fc_thermal[i])
        itn  = float(fc_intercon[i])
        gen  = float(fc_total_gen[i])
        dem  = float(fc_demand[i])

        if niv > NIV_THRESHOLD_MW and headroom_up > 0:
            position = "LONG"
            action   = "CHARGE_MORE"
            # Net P&L = SSP earned for imbalance volume minus MIP cost of incremental energy.
            # Gross SSP would overstate the gain by the wholesale cost of the extra kWh consumed.
            pnl_per_mw = (ssp - mip) * 0.5
            rationale = (
                f"System long by {niv:,.0f} MW. Charging more absorbs excess generation, "
                f"earning SSP=£{ssp:.1f}/MWh net of MIP=£{mip:.1f}/MWh "
                f"(net £{pnl_per_mw:.2f}/MW this SP)."
            )
        elif niv < -NIV_THRESHOLD_MW and (sbp - mip) >= SBP_PREMIUM_MIN and headroom_down > 0:
            position = "SHORT"
            action   = "CHARGE_LESS"
            pnl_per_mw = (sbp - mip) * 0.5   # earn imbalance premium, give up DA cost
            rationale = (
                f"System short by {abs(niv):,.0f} MW. Reducing charging earns "
                f"SBP=£{sbp:.1f}/MWh vs MIP=£{mip:.1f}/MWh "
                f"(net £{pnl_per_mw:.2f}/MW this SP)."
            )
        else:
            position   = "BALANCED"
            action     = "HOLD"
            pnl_per_mw = 0.0
            if abs(niv) <= NIV_THRESHOLD_MW:
                rationale = f"System balanced (NIV={niv:+,.0f} MW). Hold commitment."
            elif niv > 0:
                rationale = (
                    f"System long (NIV={niv:+,.0f} MW) but no headroom to increase charging."
                )
            else:
                rationale = (
                    f"System short (NIV={niv:+,.0f} MW) but SBP premium "
                    f"(£{sbp-mip:.1f}/MWh) below threshold. Hold."
                )

        sp_recs.append(DispatchRecommendation(
            horizon_sp=h,
            forecast_datetime=dt_sp,
            wind_forecast_mw=wind,
            thermal_forecast_mw=thm,
            interconnector_forecast_mw=itn,
            total_gen_forecast_mw=gen,
            demand_forecast_mw=dem,
            niv_forecast_mw=niv,
            sbp_forecast=sbp,
            ssp_forecast=ssp,
            mip_forecast=mip,
            system_position=position,
            action=action,
            pnl_delta_per_mw=pnl_per_mw,
            rationale=rationale,
        ))

    # ── Daily rollups ──────────────────────────────────────────────────────
    daily_summaries: List[DailyDispatchSummary] = []
    for day in range(lookahead_days):
        start_i = day * 48
        end_i   = start_i + 48
        day_recs = sp_recs[start_i:end_i]
        if not day_recs:
            continue

        date_str = day_recs[0].forecast_datetime.strftime("%Y-%m-%d")
        niv_arr  = np.array([r.niv_forecast_mw for r in day_recs])
        gen_arr  = np.array([r.total_gen_forecast_mw for r in day_recs])
        wind_arr = np.array([r.wind_forecast_mw for r in day_recs])
        thm_arr  = np.array([r.thermal_forecast_mw for r in day_recs])
        itn_arr  = np.array([r.interconnector_forecast_mw for r in day_recs])
        sbp_arr  = np.array([r.sbp_forecast for r in day_recs])
        mip_arr  = np.array([r.mip_forecast for r in day_recs])
        pnl_arr  = np.array([r.pnl_delta_per_mw for r in day_recs])
        actions  = [r.action for r in day_recs]

        safe_gen = np.where(gen_arr < 1.0, 1.0, gen_arr)
        wind_pct = float(np.mean(wind_arr / safe_gen * 100))
        thm_pct  = float(np.mean(thm_arr  / safe_gen * 100))
        itn_pct  = float(np.mean(itn_arr  / safe_gen * 100))

        # Dominant stance: majority of SPs (>50% of a 48-SP day = 24 SPs)
        _MAJORITY_SPS = len(day_recs) // 2
        n_long    = sum(1 for r in day_recs if r.system_position == "LONG")
        n_short   = sum(1 for r in day_recs if r.system_position == "SHORT")
        if n_long > n_short and n_long > _MAJORITY_SPS:
            stance = "LONG"
        elif n_short > n_long and n_short > _MAJORITY_SPS:
            stance = "SHORT"
        else:
            stance = "BALANCED"

        action_counts = {a: actions.count(a) for a in set(actions)}
        dominant_action = max(action_counts, key=action_counts.get)

        n_strong = int(np.sum(np.abs(niv_arr) > NIV_THRESHOLD_MW))
        pnl_day  = float(np.sum(pnl_arr))  # £/MW if deployed all 48 SPs

        daily_summaries.append(DailyDispatchSummary(
            date=date_str,
            mean_niv_mw=float(np.mean(niv_arr)),
            wind_pct=wind_pct,
            thermal_pct=thm_pct,
            interconnector_pct=itn_pct,
            mean_sbp=float(np.mean(sbp_arr)),
            mean_mip=float(np.mean(mip_arr)),
            system_stance=stance,
            dominant_action=dominant_action,
            pnl_opportunity_per_mw_day=pnl_day,
            n_strong_signal_sps=n_strong,
        ))

    logger.info(
        "Dispatch engine: %d SP recommendations, %d daily summaries, "
        "lookahead=%dd, committed=%.1f MW",
        len(sp_recs), len(daily_summaries), lookahead_days, committed_mw,
    )
    return sp_recs, daily_summaries
