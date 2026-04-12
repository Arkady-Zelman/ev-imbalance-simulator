"""
Default parameters and constants for the EV Flexibility Portfolio
Imbalance Exposure Simulator.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# ---------------------------------------------------------------------------
# Backend paths
# ---------------------------------------------------------------------------
# Root of the project (two levels up from this file: src/config.py → src/ → root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = _PROJECT_ROOT / "models"
PREDICTION_DIR = _PROJECT_ROOT / "data" / "predictions"

# Ensure directories exist when config is imported
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

# Forecast targets
FORECAST_TARGETS = ["sip", "mip", "demand", "total_generation"]

# ---------------------------------------------------------------------------
# Fleet & hardware
# ---------------------------------------------------------------------------
DEFAULT_FLEET_SIZE = 20_000
CHARGER_CAPACITY_KW = 7.4  # typical single-phase home charger
MIN_FLEET_SIZE = 5_000
MAX_FLEET_SIZE = 100_000

# ---------------------------------------------------------------------------
# Dispatch / override
# ---------------------------------------------------------------------------
DEFAULT_DISPATCH_SUCCESS_RATE = 0.95
DEFAULT_OVERRIDE_RATE = 0.03

# ---------------------------------------------------------------------------
# Settlement period helpers
# ---------------------------------------------------------------------------
NUM_SETTLEMENT_PERIODS = 48  # half-hourly across 24 h

SP_LABELS: List[str] = [
    f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)
]

# ---------------------------------------------------------------------------
# Plug-in rate profiles (Beta distribution)
#
# Each time-of-day cluster is defined by (mean, concentration ν).
# α = mean * ν,  β = (1 - mean) * ν
# Higher ν → tighter distribution (more confidence).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PluginCluster:
    name: str
    sp_range: Tuple[int, int]   # inclusive start/end SP indices (0-based)
    mean: float
    concentration: float

    @property
    def alpha(self) -> float:
        return self.mean * self.concentration

    @property
    def beta_param(self) -> float:
        return (1.0 - self.mean) * self.concentration


PLUGIN_CLUSTERS: List[PluginCluster] = [
    PluginCluster("Overnight",          (0, 11),   0.72, 50),   # 00:00-05:30
    PluginCluster("Morning departure",  (12, 17),  0.35, 20),   # 06:00-08:30
    PluginCluster("Daytime",            (18, 33),  0.28, 15),   # 09:00-16:30
    PluginCluster("Evening peak",       (34, 41),  0.55, 25),   # 17:00-20:30
    PluginCluster("Late evening",       (42, 47),  0.65, 35),   # 21:00-23:30
]


def build_sp_beta_params() -> Tuple[np.ndarray, np.ndarray]:
    """Return (alpha, beta) arrays of shape (48,) for each settlement period."""
    alphas = np.zeros(NUM_SETTLEMENT_PERIODS)
    betas = np.zeros(NUM_SETTLEMENT_PERIODS)
    for cluster in PLUGIN_CLUSTERS:
        start, end = cluster.sp_range
        alphas[start : end + 1] = cluster.alpha
        betas[start : end + 1] = cluster.beta_param
    return alphas, betas


def build_sp_means() -> np.ndarray:
    """Return mean plug-in rate array of shape (48,)."""
    means = np.zeros(NUM_SETTLEMENT_PERIODS)
    for cluster in PLUGIN_CLUSTERS:
        start, end = cluster.sp_range
        means[start : end + 1] = cluster.mean
    return means


# ---------------------------------------------------------------------------
# Day-type and seasonal multipliers on plug-in means
# ---------------------------------------------------------------------------
# Applied multiplicatively to the base Beta mean for each cluster.
# Values > 1 = higher plug-in rates; < 1 = lower.
DAYTYPE_MULTIPLIERS = {
    "weekday": 1.0,
    "weekend": 1.12,   # ~12% more vehicles plugged in on weekends
    "holiday": 1.15,   # bank holidays similar to weekends
}

# Monthly seasonal factors (index 1-12).  Winter evenings see higher
# plug-in; summer daytime sees lower because of travel.
SEASONAL_MONTHLY = {
    1: 1.08, 2: 1.06, 3: 1.02, 4: 0.98, 5: 0.95, 6: 0.92,
    7: 0.90, 8: 0.91, 9: 0.96, 10: 1.00, 11: 1.04, 12: 1.07,
}

# ---------------------------------------------------------------------------
# SIP-availability stress coupling
# ---------------------------------------------------------------------------
# When SIP is in the top quintile (system stress), plug-in rates degrade
# by this factor (fewer drivers plug in on cold, high-demand evenings —
# or more plugged in at home but dispatch failures rise).
SIP_STRESS_PLUGIN_FACTOR = 0.92
SIP_STRESS_DISPATCH_PENALTY = 0.97  # dispatch success drops under grid stress

# ---------------------------------------------------------------------------
# Copula – correlation structure
# ---------------------------------------------------------------------------
CORRELATION_DECAY = 0.3  # exponential decay rate for adjacent SPs

def build_correlation_matrix(n: int = NUM_SETTLEMENT_PERIODS,
                             decay: float = CORRELATION_DECAY) -> np.ndarray:
    idx = np.arange(n)
    return np.exp(-decay * np.abs(np.subtract.outer(idx, idx)))

# ---------------------------------------------------------------------------
# Risk appetite tiers (position-sizing percentiles)
# ---------------------------------------------------------------------------
RISK_APPETITES = {
    "P50": 50,
    "P60": 60,
    "P70": 70,
    "P80": 80,
    "P90": 90,
    "P95": 95,
}

DEFAULT_RISK_APPETITE = "P80"

# ---------------------------------------------------------------------------
# Monte Carlo defaults
# ---------------------------------------------------------------------------
MC_RUNS_OPTIONS = [1_000, 5_000, 10_000]
DEFAULT_MC_RUNS = 5_000

# ---------------------------------------------------------------------------
# SIP regime-switching model defaults
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SIPRegimeParams:
    normal_mean: float = 60.0       # £/MWh
    normal_std: float = 30.0
    spike_mean_log: float = 5.8     # ln(£/MWh) → ~£330 median
    spike_std_log: float = 0.9
    spike_probability: float = 0.05

SIP_REGIME_DEFAULTS = SIPRegimeParams()

# ---------------------------------------------------------------------------
# ELEXON API
# ---------------------------------------------------------------------------
ELEXON_BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"
CACHE_TTL_SECONDS = 86_400  # 24 h

# ---------------------------------------------------------------------------
# Day-ahead price assumption (used as benchmark when DA data unavailable)
# ---------------------------------------------------------------------------
DEFAULT_DA_PRICE = 75.0  # £/MWh – reasonable GB average

# ---------------------------------------------------------------------------
# Display / colour palette
# ---------------------------------------------------------------------------
COLOUR_PRIMARY = "#00D4AA"
COLOUR_SECONDARY = "#FF6B6B"
COLOUR_ACCENT = "#4ECDC4"
COLOUR_WARNING = "#FFE66D"
COLOUR_DANGER = "#FF6B6B"
COLOUR_SUCCESS = "#2ECC71"
COLOUR_MUTED = "#95A5A6"

PLOTLY_TEMPLATE = "plotly_dark"
