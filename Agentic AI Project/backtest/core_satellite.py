"""Core-Satellite Portfolio Analysis.

Answers the Part 2 question:
Does allocating a fraction of capital (w in [0, 50%]) to a satellite strategy
improve the overall portfolio holding the core index (NIFTYBEES), via positive
marginal Sharpe contribution?

A satellite does not need to beat the index head-to-head. If its correlation
to the index is low enough, its diversification benefit can raise overall
portfolio Sharpe and reduce drawdown.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from trading_agent.backtest.metrics import (
    DEFAULT_RISK_FREE_RATE,
    TRADING_DAYS_PER_YEAR,
    max_drawdown,
)

__all__ = ["AllocationPoint", "CoreSatelliteAnalysis", "analyze_core_satellite"]

_MIN_ALIGNED_SESSIONS = 10
_SHARPE_HURDLE = 0.05


@dataclass(frozen=True, slots=True)
class AllocationPoint:
    """Performance metrics for one core-satellite allocation weight."""

    weight_satellite: float
    weight_core: float
    total_return: float
    cagr: float
    sharpe: float
    max_drawdown: float
    annual_volatility: float


@dataclass(frozen=True, slots=True)
class CoreSatelliteAnalysis:
    """Complete core-satellite allocation sweep."""

    correlation: float
    allocations: list[AllocationPoint]
    optimal_satellite_weight: float
    optimal_sharpe: float
    core_only_sharpe: float
    marginal_sharpe_gain: float
    satellite_adds_value: bool  # True if optimal_satellite_weight > 0


def analyze_core_satellite(
    core_equity: pd.Series,
    satellite_equity: pd.Series,
    *,
    weights: Sequence[float] = (
        0.0,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
    ),
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> CoreSatelliteAnalysis:
    """Sweep satellite allocation weights and report marginal Sharpe contribution."""
    # Align curves
    df = pd.DataFrame({"core": core_equity, "sat": satellite_equity}).dropna()
    if len(df) < _MIN_ALIGNED_SESSIONS:
        raise ValueError(f"need at least {_MIN_ALIGNED_SESSIONS} aligned sessions; got {len(df)}")

    core_ret = df["core"].pct_change().dropna()
    sat_ret = df["sat"].pct_change().dropna()

    aligned_ret = pd.DataFrame({"core": core_ret, "sat": sat_ret}).dropna()
    corr = float(aligned_ret["core"].corr(aligned_ret["sat"]))
    if math.isnan(corr):
        corr = 0.0

    days = max((df.index[-1] - df.index[0]).days, 1)
    years = days / 365.25
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR

    points: list[AllocationPoint] = []

    for w_sat in weights:
        w_core = 1.0 - w_sat
        p_ret = w_core * aligned_ret["core"] + w_sat * aligned_ret["sat"]

        comb_curve = (1.0 + p_ret).cumprod()
        tot_ret = float(comb_curve.iloc[-1] - 1.0)
        cagr = (1.0 + tot_ret) ** (1.0 / years) - 1.0 if years > 0 and tot_ret > -1.0 else 0.0

        ann_vol = float(p_ret.std() * math.sqrt(TRADING_DAYS_PER_YEAR)) if len(p_ret) > 1 else 0.0
        excess = p_ret - daily_rf
        sharpe = (
            float(excess.mean() / excess.std() * math.sqrt(TRADING_DAYS_PER_YEAR))
            if len(excess) > 1 and float(excess.std()) > 0
            else 0.0
        )

        max_dd, _ = max_drawdown(comb_curve)

        points.append(
            AllocationPoint(
                weight_satellite=round(w_sat, 4),
                weight_core=round(w_core, 4),
                total_return=tot_ret,
                cagr=cagr,
                sharpe=sharpe,
                max_drawdown=max_dd,
                annual_volatility=ann_vol,
            )
        )

    # Core only benchmark point (w_sat == 0.0)
    core_point = next((p for p in points if p.weight_satellite == 0.0), points[0])
    core_sharpe = core_point.sharpe

    # Optimal allocation by Sharpe
    best_point = max(points, key=lambda p: p.sharpe)
    optimal_w = best_point.weight_satellite
    marginal_gain = best_point.sharpe - core_sharpe

    return CoreSatelliteAnalysis(
        correlation=corr,
        allocations=points,
        optimal_satellite_weight=optimal_w,
        optimal_sharpe=best_point.sharpe,
        core_only_sharpe=core_sharpe,
        marginal_sharpe_gain=marginal_gain,
        satellite_adds_value=optimal_w > 0.0 and marginal_gain > _SHARPE_HURDLE,
    )
