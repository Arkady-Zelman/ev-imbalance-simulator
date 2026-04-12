"""
Free weather and GB grid forecast fetchers — no API key required.

Sources
-------
- Open-Meteo ERA5 reanalysis (historical, ~5-day lag):
    https://archive-api.open-meteo.com/v1/era5
- Open-Meteo forecast (recent + 16-day ahead):
    https://api.open-meteo.com/v1/forecast
- ELEXON WINDFOR (day-ahead wind generation forecast):
    https://data.elexon.co.uk/bmrs/api/v1/datasets/WINDFOR
- ELEXON NDFD (national demand forecast, daily resolution):
    https://data.elexon.co.uk/bmrs/api/v1/datasets/NDFD

All responses are cached for 1 hour (TTL=3600s) via Streamlit's cache layer.
Returned DataFrames use a timezone-naive datetime index to match the output
of build_aligned_series() in src/models/forecaster.py.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Optional

import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

_GB_LAT = 54.0   # GB population centroid (rough)
_GB_LON = -2.0
_WEATHER_VARS = "temperature_2m,wind_speed_100m,shortwave_radiation,cloud_cover"
_WEATHER_TTL  = 3600   # 1-hour cache
_ELEXON_BASE  = "https://data.elexon.co.uk/bmrs/api/v1"

_session = requests.Session()
_session.mount(
    "https://",
    HTTPAdapter(max_retries=Retry(
        total=3, backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )),
)
_session.headers.update({"Accept": "application/json"})


# ── Internal helpers ───────────────────────────────────────────────────────────

def _hourly_to_30min(df: pd.DataFrame) -> pd.DataFrame:
    """Resample a timezone-naive hourly DataFrame to 30-minute intervals by forward-fill."""
    if df.empty:
        return df
    start = df.index.min()
    end   = df.index.max() + pd.Timedelta(minutes=30)
    full_idx = pd.date_range(start, end, freq="30min", inclusive="left")
    return df.reindex(full_idx).ffill()


def _fetch_open_meteo(url: str, params: dict) -> pd.DataFrame:
    """Fetch hourly weather from an Open-Meteo endpoint; return timezone-naive DataFrame."""
    try:
        resp = _session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        hourly = resp.json().get("hourly", {})
        if not hourly or "time" not in hourly:
            return pd.DataFrame()
        df = pd.DataFrame(hourly)
        df["time"] = pd.to_datetime(df["time"])   # naive local time (tz=Europe/London requested)
        df = df.set_index("time")
        df = df.rename(columns={
            "temperature_2m":    "temperature_c",
            "shortwave_radiation": "solar_radiation",
            "cloud_cover":       "cloud_cover_pct",
        })
        return df
    except Exception as exc:
        logger.warning("Open-Meteo fetch failed (%s): %s", url, exc)
        return pd.DataFrame()


# ── Public API ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=_WEATHER_TTL, show_spinner=False)
def fetch_weather_data(date_from: dt.date, date_to: dt.date) -> pd.DataFrame:
    """
    Fetch GB weather from Open-Meteo at GB centroid (lat=54, lon=-2).

    Uses ERA5 reanalysis for historical dates (available ~5 days ago) and the
    forecast API for recent / future dates.  Both are free with no API key.
    The two sources are combined with the forecast API taking precedence for
    overlapping dates (it is more up-to-date than ERA5 for recent days).

    Returns a 30-minute DataFrame (forward-filled from hourly) with columns:
        temperature_c      — 2-metre temperature (°C)
        wind_speed_100m    — wind speed at 100 m (km/h; relevant for offshore turbines)
        solar_radiation    — shortwave surface radiation (W/m²; embedded solar proxy)
        cloud_cover_pct    — cloud cover (%)

    Index: timezone-naive datetime matching build_aligned_series() output.
    Cached 1 hour.
    """
    today    = dt.date.today()
    base_params = {
        "latitude":  _GB_LAT, "longitude": _GB_LON,
        "hourly":    _WEATHER_VARS,
        "timezone":  "Europe/London",
    }
    all_dfs = []

    # ── ERA5 historical (available up to ~5 days ago) ──
    era5_end = min(date_to, today - dt.timedelta(days=5))
    if date_from <= era5_end:
        params = {**base_params,
                  "start_date": date_from.isoformat(),
                  "end_date":   era5_end.isoformat()}
        df = _fetch_open_meteo("https://archive-api.open-meteo.com/v1/era5", params)
        if not df.empty:
            all_dfs.append(df)

    # ── Forecast API (recent ~3 days + future up to 16 days) ──
    fc_start = max(date_from, today - dt.timedelta(days=3))
    if fc_start <= date_to:
        fc_days  = min(16, (date_to - today).days + 1) if date_to >= today else 1
        params = {**base_params,
                  "start_date":   fc_start.isoformat(),
                  "end_date":     date_to.isoformat(),
                  "forecast_days": max(1, fc_days)}
        df = _fetch_open_meteo("https://api.open-meteo.com/v1/forecast", params)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        logger.warning("Weather fetch: no data for %s → %s", date_from, date_to)
        return pd.DataFrame()

    combined = pd.concat(all_dfs)
    # Forecast API takes precedence for recent dates (keep='last')
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    return _hourly_to_30min(combined)


@st.cache_data(ttl=_WEATHER_TTL, show_spinner=False)
def fetch_wind_generation_forecast(date_from: dt.date, date_to: dt.date) -> pd.DataFrame:
    """
    Fetch ELEXON WINDFOR day-ahead wind generation forecast.

    Endpoint: GET data.elexon.co.uk/bmrs/api/v1/datasets/WINDFOR
    Free, no API key — same ELEXON Insights API already used by the rest of the app.

    Returns a 30-minute DataFrame with column: wind_fc_mw
    Index: timezone-naive datetime matching build_aligned_series() output.
    Cached 1 hour.
    """
    url = f"{_ELEXON_BASE}/datasets/WINDFOR"
    params = {
        "settlementDateFrom": date_from.isoformat(),
        "settlementDateTo":   date_to.isoformat(),
        "format": "json",
    }
    try:
        resp = _session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        records = resp.json().get("data", [])
    except Exception as exc:
        logger.warning("WINDFOR fetch failed: %s", exc)
        return pd.DataFrame()

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    if "startTime" not in df.columns or "generation" not in df.columns:
        return pd.DataFrame()

    # startTime is UTC — convert to Europe/London naive to match SIP index
    df["datetime"] = (
        pd.to_datetime(df["startTime"], utc=True)
        .dt.tz_convert("Europe/London")
        .dt.tz_localize(None)
    )
    df["wind_fc_mw"] = pd.to_numeric(df["generation"], errors="coerce")
    df = (df[["datetime", "wind_fc_mw"]]
          .dropna()
          .set_index("datetime")
          .sort_index())
    df = df[~df.index.duplicated(keep="first")]
    return _hourly_to_30min(df)


@st.cache_data(ttl=_WEATHER_TTL, show_spinner=False)
def fetch_demand_forecast(date_from: dt.date, date_to: dt.date) -> pd.DataFrame:
    """
    Fetch ELEXON NDFD national demand forecast (daily resolution).
    Each day's value is broadcast to all 48 settlement periods of that day.

    Endpoint: GET data.elexon.co.uk/bmrs/api/v1/datasets/NDFD
    Free, no API key.

    Returns a 30-minute DataFrame with column: demand_fc_mw
    Index: timezone-naive datetime matching build_aligned_series() output.
    Cached 1 hour.
    """
    url = f"{_ELEXON_BASE}/datasets/NDFD"
    params = {
        "settlementDateFrom": date_from.isoformat(),
        "settlementDateTo":   date_to.isoformat(),
        "format": "json",
    }
    try:
        resp = _session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        records = resp.json().get("data", [])
    except Exception as exc:
        logger.warning("NDFD fetch failed: %s", exc)
        return pd.DataFrame()

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    if "forecastDate" not in df.columns or "demand" not in df.columns:
        return pd.DataFrame()

    df["demand_fc_mw"] = pd.to_numeric(df["demand"], errors="coerce")
    df = df[["forecastDate", "demand_fc_mw"]].dropna()

    # Broadcast each daily value to 48 half-hourly SPs
    rows = []
    for _, row in df.iterrows():
        day_start = pd.Timestamp(row["forecastDate"])
        for sp in range(48):
            rows.append({
                "datetime":     day_start + pd.Timedelta(minutes=30 * sp),
                "demand_fc_mw": row["demand_fc_mw"],
            })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).set_index("datetime").sort_index()
    return result[~result.index.duplicated(keep="first")]



# ── Raw (no Streamlit cache) variants for backend scripts ────────────────────

def fetch_weather_data_raw(date_from, date_to):
    return fetch_weather_data.__wrapped__(date_from, date_to)


def fetch_wind_generation_forecast_raw(date_from, date_to):
    return fetch_wind_generation_forecast.__wrapped__(date_from, date_to)


def fetch_demand_forecast_raw(date_from, date_to):
    return fetch_demand_forecast.__wrapped__(date_from, date_to)
