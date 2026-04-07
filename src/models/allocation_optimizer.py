"""
Capacity Allocation Optimizer — decides how to split available MW
between wholesale (MIP) and balancing (SIP) markets per settlement period,
and whether to overbook, underbook, or match expected delivery.

Uses SIP + MIP forward forecasts and Monte Carlo fleet delivery distributions
to produce risk-adjusted per-SP allocation recommendations.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor as _TPE, as_completed as _ac
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

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


@dataclass(frozen=True)
class RiskProfile:
    """Defines a position-sizing + market-split configuration."""
    name: str
    percentile: int        # MC availability percentile for position anchor
    risk_tolerance: float  # 0.0 = max E[revenue]; 1.0 = max tail protection
    description: str


RISK_PROFILES: Dict[str, RiskProfile] = {
    "Conservative": RiskProfile("Conservative", 50, 0.8,
                                "P50 positions · strongly tail-risk averse"),
    "Moderate":     RiskProfile("Moderate",     70, 0.5,
                                "P70 positions · balanced risk/return"),
    "Aggressive":   RiskProfile("Aggressive",   80, 0.25,
                                "P80 positions · return-seeking"),
    "Full Risk":    RiskProfile("Full Risk",    95, 0.0,
                                "P95 positions · maximise expected revenue"),
}


@dataclass
class MultiProfileResult:
    """Allocation results across all risk profiles."""
    profile_results:   Dict[str, AllocationResult]  # profile name -> result
    profile_positions: Dict[str, np.ndarray]          # profile name -> (48,) MW
    comparison_df:     Any                            # pd.DataFrame summary


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
    base_percentile: int = 50,
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

        p50 = float(np.percentile(delivered_sp, base_percentile))
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


def _portfolio_ob_ratio(alloc: AllocationResult) -> float:
    total_c = sum(a.total_committed    for a in alloc.sp_allocations)
    total_e = sum(a.expected_delivered for a in alloc.sp_allocations)
    return total_c / total_e if total_e > 1e-9 else 1.0


def optimize_all_risk_profiles(
    sip_forecasts: np.ndarray,
    mip_forecasts: np.ndarray,
    delivered_mw: np.ndarray,
    profiles: Optional[Dict[str, RiskProfile]] = None,
    grid_steps: int = 21,
) -> MultiProfileResult:
    """
    Run optimize_allocation() for each risk profile in parallel.

    Each profile specifies its own base_percentile (position-size anchor from
    the MC availability distribution) and risk_tolerance (wholesale vs balancing
    split aggressiveness). Results are returned together with a comparison DataFrame.
    """
    if profiles is None:
        profiles = RISK_PROFILES

    profile_results:   Dict[str, AllocationResult] = {}
    profile_positions: Dict[str, np.ndarray]        = {}

    def _run(name: str, p: RiskProfile):
        alloc = optimize_allocation(
            sip_forecasts=sip_forecasts,
            mip_forecasts=mip_forecasts,
            delivered_mw=delivered_mw,
            risk_tolerance=p.risk_tolerance,
            grid_steps=grid_steps,
            base_percentile=p.percentile,
        )
        position_mw = np.percentile(delivered_mw, p.percentile, axis=0)
        return name, alloc, position_mw

    with _TPE(max_workers=len(profiles)) as ex:
        futures = {ex.submit(_run, n, p): n for n, p in profiles.items()}
        for fut in _ac(futures):
            name, alloc, pos = fut.result()
            profile_results[name]   = alloc
            profile_positions[name] = pos

    rows = []
    for name, p in profiles.items():
        opt = profile_results[name].optimal_strategy
        pos = profile_positions[name]
        rows.append({
            "Profile":          name,
            "Percentile":       f"P{p.percentile}",
            "Risk Tolerance":   p.risk_tolerance,
            "Mean Position MW": round(float(np.mean(pos)), 2),
            "Peak Position MW": round(float(np.max(pos)), 2),
            "E[Daily P&L] £":   round(opt.mean_pnl, 0),
            "Median P&L £":     round(opt.median_pnl, 0),
            "ES 5% £":          round(opt.es_5, 0),
            "Std P&L £":        round(opt.std_pnl, 0),
            "Max Loss £":       round(opt.max_loss, 0),
            "Reward/Risk":      round(opt.reward_to_risk, 3),
            "OB Ratio":         round(_portfolio_ob_ratio(profile_results[name]), 3),
        })

    return MultiProfileResult(
        profile_results=profile_results,
        profile_positions=profile_positions,
        comparison_df=pd.DataFrame(rows),
    )
