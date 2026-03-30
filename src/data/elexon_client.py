"""
Client for the ELEXON Insights API (public, no key required).

Endpoints used:
  - System Imbalance Prices (SIP): /system-prices/{date}
  - Market Index Price: /market-index?from=...&to=...
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

from src.config import CACHE_TTL_SECONDS, ELEXON_BASE_URL
from src.data import cache_manager

logger = logging.getLogger(__name__)

_SESSION = requests.Session()
_SESSION.headers.update({"Accept": "application/json"})

# ── helpers ────────────────────────────────────────────────────────────────

def _fetch_system_prices_for_date(date: dt.date) -> List[Dict]:
    """Fetch SIP for a single settlement date (path-based endpoint)."""
    url = f"{ELEXON_BASE_URL}/balancing/settlement/system-prices/{date.isoformat()}"
    try:
        logger.info("GET %s", url)
        resp = _SESSION.get(url, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        records = payload.get("data", []) if isinstance(payload, dict) else payload
        logger.info("  → %d records for %s (HTTP %d)", len(records), date, resp.status_code)
        return records
    except requests.RequestException as exc:
        logger.warning("SIP fetch failed for %s: %s", date, exc)
        return []
    except ValueError as exc:
        logger.warning("SIP JSON parse failed for %s: %s", date, exc)
        return []


def _fetch_market_index_chunk(
    dt_from: dt.datetime, dt_to: dt.datetime
) -> List[Dict]:
    """Fetch Market Index Price for a datetime range (query-based endpoint)."""
    url = f"{ELEXON_BASE_URL}/balancing/pricing/market-index"
    params = {
        "from": dt_from.strftime("%Y-%m-%dT%H:%MZ"),
        "to": dt_to.strftime("%Y-%m-%dT%H:%MZ"),
        "format": "json",
    }
    try:
        logger.info("GET %s  from=%s  to=%s", url, params["from"], params["to"])
        resp = _SESSION.get(url, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        records = payload.get("data", []) if isinstance(payload, dict) else payload
        logger.info("  → %d MIP records (HTTP %d)", len(records), resp.status_code)
        return records
    except requests.RequestException as exc:
        logger.warning("MIP fetch failed (%s → %s): %s", dt_from, dt_to, exc)
        return []
    except ValueError as exc:
        logger.warning("MIP JSON parse failed (%s → %s): %s", dt_from, dt_to, exc)
        return []


# ── public API ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_system_prices(date_from: dt.date, date_to: dt.date) -> pd.DataFrame:
    """
    Fetch System Buy Price / System Sell Price per settlement period.
    Iterates day-by-day using the path-based endpoint.
    """
    cache_key = f"sip_{date_from}_{date_to}"
    cached = cache_manager.get(cache_key, CACHE_TTL_SECONDS)
    if cached is not None:
        logger.info("SIP cache hit for %s → %s", date_from, date_to)
        return pd.DataFrame(cached)

    all_records: List[Dict] = []
    cursor = date_from
    while cursor <= date_to:
        records = _fetch_system_prices_for_date(cursor)
        all_records.extend(records)
        cursor += dt.timedelta(days=1)

    logger.info("Fetched %d total SIP records for %s → %s", len(all_records), date_from, date_to)

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    keep = [c for c in [
        "settlementDate", "settlementPeriod",
        "systemSellPrice", "systemBuyPrice",
        "netImbalanceVolume",
    ] if c in df.columns]
    df = df[keep].copy()

    for col in ("systemSellPrice", "systemBuyPrice", "netImbalanceVolume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "settlementPeriod" in df.columns:
        df["settlementPeriod"] = pd.to_numeric(df["settlementPeriod"], errors="coerce")
    df.dropna(subset=["settlementPeriod"], inplace=True)
    df.sort_values(["settlementDate", "settlementPeriod"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    cache_manager.put(cache_key, df.to_dict(orient="records"))
    return df


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_market_index(date_from: dt.date, date_to: dt.date) -> pd.DataFrame:
    """
    Fetch Market Index Price (near-delivery wholesale proxy).
    Uses the query-based from/to datetime endpoint, chunking by 7-day windows.
    """
    cache_key = f"mip_{date_from}_{date_to}"
    cached = cache_manager.get(cache_key, CACHE_TTL_SECONDS)
    if cached is not None:
        logger.info("MIP cache hit for %s → %s", date_from, date_to)
        return pd.DataFrame(cached)

    all_records: List[Dict] = []
    cursor = dt.datetime.combine(date_from, dt.time.min)
    end = dt.datetime.combine(date_to, dt.time.max)
    chunk_days = 7

    while cursor < end:
        chunk_end = min(cursor + dt.timedelta(days=chunk_days), end)
        records = _fetch_market_index_chunk(cursor, chunk_end)
        all_records.extend(records)
        cursor = chunk_end

    logger.info("Fetched %d total MIP records for %s → %s", len(all_records), date_from, date_to)

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    keep = [c for c in [
        "settlementDate", "settlementPeriod", "price", "volume",
        "dataProvider",
    ] if c in df.columns]
    df = df[keep].copy()
    for col in ("price", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "settlementPeriod" in df.columns:
        df["settlementPeriod"] = pd.to_numeric(df["settlementPeriod"], errors="coerce")
    df.dropna(subset=["settlementPeriod"], inplace=True)
    if "dataProvider" in df.columns:
        apx = df[df["dataProvider"] == "APXMIDP"]
        if not apx.empty:
            df = apx
    df.sort_values(["settlementDate", "settlementPeriod"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    cache_manager.put(cache_key, df.to_dict(orient="records"))
    return df


def _fetch_demand_outturn_chunk(
    dt_from: dt.datetime, dt_to: dt.datetime
) -> List[Dict]:
    """Fetch demand outturn (INDO/ITSDO) for a datetime range."""
    url = f"{ELEXON_BASE_URL}/demand/outturn"
    params = {
        "settlementDateFrom": dt_from.strftime("%Y-%m-%d"),
        "settlementDateTo": dt_to.strftime("%Y-%m-%d"),
        "format": "json",
    }
    try:
        logger.info("GET %s  from=%s  to=%s", url, params["settlementDateFrom"], params["settlementDateTo"])
        resp = _SESSION.get(url, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        records = payload.get("data", []) if isinstance(payload, dict) else payload
        logger.info("  → %d demand records (HTTP %d)", len(records), resp.status_code)
        return records
    except requests.RequestException as exc:
        logger.warning("Demand fetch failed (%s → %s): %s", dt_from, dt_to, exc)
        return []
    except ValueError as exc:
        logger.warning("Demand JSON parse failed (%s → %s): %s", dt_from, dt_to, exc)
        return []


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_demand_outturn(date_from: dt.date, date_to: dt.date) -> pd.DataFrame:
    """
    Fetch Initial National Demand Outturn (INDO) and Transmission System
    Demand Outturn (ITSDO) per settlement period.
    Uses query-parameter-based /demand/outturn endpoint, chunking by 7 days.
    """
    cache_key = f"demand_{date_from}_{date_to}"
    cached = cache_manager.get(cache_key, CACHE_TTL_SECONDS)
    if cached is not None:
        logger.info("Demand cache hit for %s → %s", date_from, date_to)
        return pd.DataFrame(cached)

    all_records: List[Dict] = []
    cursor = dt.datetime.combine(date_from, dt.time.min)
    end = dt.datetime.combine(date_to, dt.time.max)
    chunk_days = 7

    while cursor < end:
        chunk_end = min(cursor + dt.timedelta(days=chunk_days), end)
        records = _fetch_demand_outturn_chunk(cursor, chunk_end)
        all_records.extend(records)
        cursor = chunk_end

    logger.info("Fetched %d total demand records for %s → %s", len(all_records), date_from, date_to)

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    keep = [c for c in [
        "settlementDate", "settlementPeriod",
        "initialDemandOutturn", "initialTransmissionSystemDemandOutturn",
    ] if c in df.columns]
    df = df[keep].copy()
    for col in ("initialDemandOutturn", "initialTransmissionSystemDemandOutturn"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "settlementPeriod" in df.columns:
        df["settlementPeriod"] = pd.to_numeric(df["settlementPeriod"], errors="coerce")
    df.dropna(subset=["settlementPeriod"], inplace=True)
    df.sort_values(["settlementDate", "settlementPeriod"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    cache_manager.put(cache_key, df.to_dict(orient="records"))
    return df


def demand_cache_timestamp(date_from: dt.date, date_to: dt.date) -> Optional[float]:
    return cache_manager.cache_timestamp(f"demand_{date_from}_{date_to}")


def sip_cache_timestamp(date_from: dt.date, date_to: dt.date) -> Optional[float]:
    """Return the unix timestamp of when SIP data was last fetched, or None."""
    return cache_manager.cache_timestamp(f"sip_{date_from}_{date_to}")


def mip_cache_timestamp(date_from: dt.date, date_to: dt.date) -> Optional[float]:
    return cache_manager.cache_timestamp(f"mip_{date_from}_{date_to}")
