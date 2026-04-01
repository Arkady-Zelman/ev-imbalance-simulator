"""
Kelly criterion position sizing for EV flexibility trading.

Instead of committing an arbitrary percentile of simulated availability,
Kelly finds the commitment level that maximises the long-run geometric
growth rate of cumulative P&L:

    f* = argmax_f  E[ log(1 + f * R) ]

where R is the per-unit return on committed capacity.  Because SIP has
heavy tails, full Kelly is dangerously aggressive; in practice, trade at
a *fraction* of Kelly (0.25 = quarter, 0.5 = half, 1.0 = full).

The module provides:
- Per-SP Kelly-optimal commitment (vectorised over MC runs)
- Full-curve Kelly position (shape 48)
- Fractional Kelly scaling
- Growth-rate and drawdown analytics for comparison with percentile sizing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.config import NUM_SETTLEMENT_PERIODS

logger = logging.getLogger(__name__)

KELLY_FRACTIONS: List[float] = [0.25, 0.50, 0.75, 1.00]
KELLY_LABELS = {0.25: "¼ Kelly", 0.50: "½ Kelly", 0.75: "¾ Kelly", 1.00: "Full Kelly"}


@dataclass
class KellyResult:
    """Kelly analysis output for a single fractional Kelly level."""
    fraction: float
    label: str
    optimal_mw: np.ndarray            # (48,) — commitment per SP
    expected_daily_pnl: float
    std_daily_pnl: float
    growth_rate: float                 # E[log(1 + R)] at this fraction
    max_commitment_mw: float
    min_commitment_mw: float
    mean_shortfall_probability: float  # avg P(delivered < committed) across SPs


def _growth_rate_for_commit(
    commit: float,
    delivered_sp: np.ndarray,
    sip_sp: np.ndarray,
    da_price: float,
    bankroll: float,
) -> float:
    """
    Compute E[log(1 + R)] for a given commitment level at one SP.

    For each MC run:
      revenue   = commit * 0.5 * da_price
      shortfall = max(commit - delivered, 0)
      imb_cost  = shortfall * 0.5 * sip
      net       = revenue - imb_cost
      R         = net / bankroll
    """
    shortfall = np.maximum(commit - delivered_sp, 0.0)
    revenue = commit * 0.5 * da_price
    imb_cost = shortfall * 0.5 * sip_sp
    net = revenue - imb_cost
    returns = net / bankroll

    # Clip to avoid log(0) — floor at -99% of bankroll
    returns = np.maximum(returns, -0.99)
    return float(np.mean(np.log1p(returns)))


def kelly_optimal_position(
    delivered_mw: np.ndarray,
    sip_matrix: np.ndarray,
    da_price: float,
    bankroll: float,
    kelly_fraction: float = 0.5,
    n_candidates: int = 80,
) -> np.ndarray:
    """
    For each settlement period, find the commitment level that maximises
    E[log(1 + f*R)] over the MC draws, then scale by kelly_fraction.

    Parameters
    ----------
    delivered_mw : (n_runs, 48) — simulated delivered MW per run
    sip_matrix : (n_runs, 48) — SIP draws matched to each run
    da_price : scalar DA price assumption (£/MWh)
    bankroll : total capital / risk budget for Kelly scaling
    kelly_fraction : 0.25 to 1.0
    n_candidates : grid resolution for the optimisation

    Returns
    -------
    optimal_mw : (48,) — Kelly-optimal committed MW per SP
    """
    n_runs, n_sp = delivered_mw.shape
    optimal_mw = np.zeros(n_sp)

    # Allocate bankroll equally across SPs so each SP's return is sized against
    # its share of the daily risk budget, not the full bankroll.
    sp_bankroll = bankroll / n_sp

    for sp in range(n_sp):
        del_sp = delivered_mw[:, sp]
        sip_sp = sip_matrix[:, sp]

        lo = float(np.percentile(del_sp, 1))
        hi = float(np.percentile(del_sp, 99))
        if hi - lo < 0.01:
            optimal_mw[sp] = float(np.median(del_sp))
            continue

        candidates = np.linspace(lo, hi, n_candidates)
        best_growth = -np.inf
        best_commit = lo

        for commit in candidates:
            g = _growth_rate_for_commit(commit, del_sp, sip_sp, da_price, sp_bankroll)
            if g > best_growth:
                best_growth = g
                best_commit = commit

        optimal_mw[sp] = best_commit

    return optimal_mw * kelly_fraction


def compute_kelly_pnl(
    delivered_mw: np.ndarray,
    committed_mw: np.ndarray,
    sip_matrix: np.ndarray,
    da_price: float,
) -> np.ndarray:
    """
    Given a commitment curve (48,), compute the P&L for each MC run.

    Returns shape (n_runs,).
    """
    da_vec = np.full(NUM_SETTLEMENT_PERIODS, float(da_price))
    revenue = np.sum(committed_mw * 0.5 * da_vec)
    shortfall = np.maximum(committed_mw[np.newaxis, :] - delivered_mw, 0.0)
    imb_cost = np.sum(shortfall * 0.5 * sip_matrix, axis=1)
    return np.full(delivered_mw.shape[0], revenue) - imb_cost


def compute_kelly_growth_rate(
    delivered_mw: np.ndarray,
    committed_mw: np.ndarray,
    sip_matrix: np.ndarray,
    da_price: float,
    bankroll: float,
) -> float:
    """Aggregate growth rate for a full commitment curve."""
    pnl = compute_kelly_pnl(delivered_mw, committed_mw, sip_matrix, da_price)
    returns = pnl / bankroll
    returns = np.maximum(returns, -0.99)
    return float(np.mean(np.log1p(returns)))


def run_kelly_analysis(
    delivered_mw: np.ndarray,
    sip_matrix: np.ndarray,
    da_price: float,
    bankroll: float,
    fractions: Optional[List[float]] = None,
) -> List[KellyResult]:
    """
    Run Kelly analysis across multiple fractional levels.

    Returns a KellyResult for each fraction, including the optimal
    commitment curve, expected P&L, growth rate, and shortfall probability.
    """
    if fractions is None:
        fractions = KELLY_FRACTIONS

    results: List[KellyResult] = []

    # First compute the full-Kelly optimal curve
    full_kelly_mw = kelly_optimal_position(
        delivered_mw, sip_matrix, da_price, bankroll,
        kelly_fraction=1.0,
    )

    for frac in fractions:
        committed = full_kelly_mw * frac
        pnl = compute_kelly_pnl(delivered_mw, committed, sip_matrix, da_price)
        growth = compute_kelly_growth_rate(
            delivered_mw, committed, sip_matrix, da_price, bankroll,
        )

        # Shortfall probability: P(delivered < committed) per SP, averaged
        shortfall_prob = np.mean(
            delivered_mw < committed[np.newaxis, :], axis=0
        ).mean()

        results.append(KellyResult(
            fraction=frac,
            label=KELLY_LABELS.get(frac, f"{frac:.0%} Kelly"),
            optimal_mw=committed,
            expected_daily_pnl=float(np.mean(pnl)),
            std_daily_pnl=float(np.std(pnl)),
            growth_rate=growth,
            max_commitment_mw=float(np.max(committed)),
            min_commitment_mw=float(np.min(committed)),
            mean_shortfall_probability=float(shortfall_prob),
        ))

    logger.info(
        "Kelly analysis: bankroll=£%.0f, %d fractions, full-Kelly growth=%.6f",
        bankroll, len(fractions),
        results[-1].growth_rate if results else 0.0,
    )
    return results
