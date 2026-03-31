"""
Capacity Allocation Optimizer — decides how to split available MW
between wholesale (MIP) and balancing (SIP) markets per settlement period,
and whether to overbook, underbook, or match expected delivery.

Uses SIP + MIP forward forecasts and Monte Carlo fleet delivery distributions
to produce risk-adjusted per-SP allocation recommendations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.config import NUM_SETTLEMENT_PERIODS

logger = logging.getLogger(__name__)


@dataclass
class SPAllocation:
    """Optimal allocation for a single settlement period."""
    sp: int
    wholesale_mw: float
    balancing_mw: float
    total_committed: float
    expected_delivered: float
    overbook_ratio: float          # total / expected_delivered
    strategy: str                  # "overbook" / "match" / "underbook"
    expected_revenue: float        # mean net rev across MC runs
    es_revenue: float              # 5th-percentile tail mean


@dataclass
class StrategyResult:
    """P&L distribution for one strategy across all MC runs."""
    name: str
    daily_pnl: np.ndarray          # (n_runs,)
    mean_pnl: float
    median_pnl: float
    es_5: float                    # Expected Shortfall at 5%
    std_pnl: float
    max_loss: float
    max_gain: float
    reward_to_risk: float          # mean / std


@dataclass
class AllocationResult:
    """Full optimizer output."""
    sp_allocations: List[SPAllocation]
    optimal_strategy: StrategyResult
    pure_wholesale_strategy: StrategyResult
    pure_balancing_strategy: StrategyResult
    risk_tolerance: float


def _compute_strategy_result(
    name: str,
    daily_pnl: np.ndarray,
) -> StrategyResult:
    if len(daily_pnl) == 0:
        return StrategyResult(
            name=name, daily_pnl=daily_pnl,
            mean_pnl=0, median_pnl=0, es_5=0, std_pnl=0,
            max_loss=0, max_gain=0, reward_to_risk=0,
        )
    mean_pnl = float(np.mean(daily_pnl))
    std_pnl = float(np.std(daily_pnl))
    var_5 = float(np.percentile(daily_pnl, 5))
    tail = daily_pnl[daily_pnl <= var_5]
    es_5 = float(tail.mean()) if len(tail) > 0 else var_5
    return StrategyResult(
        name=name,
        daily_pnl=daily_pnl,
        mean_pnl=mean_pnl,
        median_pnl=float(np.median(daily_pnl)),
        es_5=es_5,
        std_pnl=std_pnl,
        max_loss=float(np.min(daily_pnl)),
        max_gain=float(np.max(daily_pnl)),
        reward_to_risk=mean_pnl / std_pnl if std_pnl > 1e-9 else 0.0,
    )


def _sp_net_revenue(
    wholesale_mw: float,
    balancing_mw: float,
    delivered_per_run: np.ndarray,
    sip_forecast: float,
    mip_forecast: float,
) -> np.ndarray:
    """
    Compute net revenue per MC run for a single SP given allocation.

    Revenue model (per half-hour = 0.5 h):
      wholesale_rev  = W * 0.5 * MIP
      bm_dispatched  = min(B, max(0, delivered - W))
      bm_rev         = bm_dispatched * 0.5 * SIP
      shortfall      = max(0, W - delivered)
      shortfall_cost = shortfall * 0.5 * abs(SIP)   (pay imbalance)
      net            = wholesale_rev + bm_rev - shortfall_cost
    """
    half_hour = 0.5
    delivered = delivered_per_run

    wholesale_rev = wholesale_mw * half_hour * mip_forecast

    surplus_after_wholesale = np.maximum(0.0, delivered - wholesale_mw)
    bm_dispatched = np.minimum(balancing_mw, surplus_after_wholesale)
    bm_rev = bm_dispatched * half_hour * sip_forecast

    shortfall = np.maximum(0.0, wholesale_mw - delivered)
    shortfall_cost = shortfall * half_hour * abs(sip_forecast)

    return wholesale_rev + bm_rev - shortfall_cost


def optimize_allocation(
    sip_forecasts: np.ndarray,
    mip_forecasts: np.ndarray,
    delivered_mw: np.ndarray,
    risk_tolerance: float = 0.5,
    grid_steps: int = 21,
) -> AllocationResult:
    """
    Find the optimal wholesale vs. balancing split per settlement period.

    Parameters
    ----------
    sip_forecasts : shape (48,) — predicted SIP per SP
    mip_forecasts : shape (48,) — predicted MIP per SP
    delivered_mw  : shape (n_runs, 48) — MC delivery distribution
    risk_tolerance : 0.0 = maximize expected revenue (risk-neutral),
                     1.0 = maximize risk-adjusted (penalise tail losses heavily)
    grid_steps : number of W values to search per SP
    """
    n_runs, n_sps = delivered_mw.shape
    assert n_sps == NUM_SETTLEMENT_PERIODS, f"Expected {NUM_SETTLEMENT_PERIODS} SPs, got {n_sps}"

    sp_allocations: List[SPAllocation] = []

    optimal_pnl_per_run = np.zeros(n_runs)
    wholesale_pnl_per_run = np.zeros(n_runs)
    balancing_pnl_per_run = np.zeros(n_runs)

    for sp in range(n_sps):
        delivered_sp = delivered_mw[:, sp]
        sip_fc = float(sip_forecasts[sp])
        mip_fc = float(mip_forecasts[sp])

        p50 = float(np.median(delivered_sp))
        p99 = float(np.percentile(delivered_sp, 99))

        if p50 < 1e-6:
            alloc = SPAllocation(
                sp=sp, wholesale_mw=0, balancing_mw=0, total_committed=0,
                expected_delivered=p50, overbook_ratio=0, strategy="underbook",
                expected_revenue=0, es_revenue=0,
            )
            sp_allocations.append(alloc)
            continue

        w_candidates = np.linspace(0, p99, grid_steps)

        best_score = -np.inf
        best_w = 0.0
        best_b = p50
        best_revenues = np.zeros(n_runs)

        for w in w_candidates:
            b = max(0.0, p50 - w)
            rev = _sp_net_revenue(w, b, delivered_sp, sip_fc, mip_fc)
            mean_rev = float(np.mean(rev))
            var_5 = float(np.percentile(rev, 5))
            tail = rev[rev <= var_5]
            es = float(tail.mean()) if len(tail) > 0 else var_5

            score = (1 - risk_tolerance) * mean_rev + risk_tolerance * es
            if score > best_score:
                best_score = score
                best_w = w
                best_b = b
                best_revenues = rev

        total = best_w + best_b
        ob_ratio = total / p50 if p50 > 1e-6 else 1.0

        if ob_ratio > 1.02:
            strategy = "overbook"
        elif ob_ratio < 0.98:
            strategy = "underbook"
        else:
            strategy = "match"

        mean_rev = float(np.mean(best_revenues))
        var_5 = float(np.percentile(best_revenues, 5))
        tail = best_revenues[best_revenues <= var_5]
        es_rev = float(tail.mean()) if len(tail) > 0 else var_5

        sp_allocations.append(SPAllocation(
            sp=sp,
            wholesale_mw=best_w,
            balancing_mw=best_b,
            total_committed=total,
            expected_delivered=p50,
            overbook_ratio=ob_ratio,
            strategy=strategy,
            expected_revenue=mean_rev,
            es_revenue=es_rev,
        ))

        optimal_pnl_per_run += best_revenues

        wholesale_rev_all = _sp_net_revenue(p50, 0.0, delivered_sp, sip_fc, mip_fc)
        wholesale_pnl_per_run += wholesale_rev_all

        balancing_rev_all = _sp_net_revenue(0.0, p50, delivered_sp, sip_fc, mip_fc)
        balancing_pnl_per_run += balancing_rev_all

    optimal_strat = _compute_strategy_result("Optimal Split", optimal_pnl_per_run)
    wholesale_strat = _compute_strategy_result("Pure Wholesale", wholesale_pnl_per_run)
    balancing_strat = _compute_strategy_result("Pure Balancing", balancing_pnl_per_run)

    return AllocationResult(
        sp_allocations=sp_allocations,
        optimal_strategy=optimal_strat,
        pure_wholesale_strategy=wholesale_strat,
        pure_balancing_strategy=balancing_strat,
        risk_tolerance=risk_tolerance,
    )
