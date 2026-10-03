"""NIFTY200 Momentum 30 replication strategy.

Replicates the methodology published by NSE Indices for factor indices
(NIFTY200 Momentum 30, NIFTY Alpha 50):

1. **Dual Horizons:** Blends 6-month (126 trading days) and 12-month (252 trading days)
   normalized momentum scores.
2. **Volatility Normalization:** Each horizon's price return is divided by the
   annualized daily return volatility over that same horizon.
3. **Z-Score Standardization:** Scores for each horizon are standardized into
   cross-sectional z-scores before averaging, preserving magnitude rather than
   discarding it into ordinal ranks.
4. **Semi-Annual Rebalancing:** Rebalances in June and December (or every 6 months),
   substantially reducing turnover and tax drag (allowing positions to cross the
   12-month LTCG boundary at 12.5% vs 20% STCG).
5. **Inverse-Vol or Score Weighting:** Allocates weights proportionally to
   inverse-volatility (risk parity proxy) or normalized score, capped at
   single-position ceilings with iterative redistribution.
6. **Dual Momentum Option:** ``absolute_momentum=False`` (default) faithfully
   replicates the index by remaining fully invested across the top relative
   performers. Setting ``absolute_momentum=True`` adds an absolute trend gate,
   holding cash when the underlying 12-month return is negative.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import StrEnum

import numpy as np
import pandas as pd

from trading_agent.market_data import IST

__all__ = ["NiftyMomentumStrategy", "WeightingScheme"]

_MIN_HORIZON_SESSIONS = 126
_LONG_HORIZON_SESSIONS = 252
_MIN_TRACKING_UNIVERSE = 5
_MONTHS_IN_YEAR = 12
_ANNUAL_SCALE = math.sqrt(250)


class WeightingScheme(StrEnum):
    EQUAL = "equal"
    INVERSE_VOL = "inverse_vol"
    SCORE_PROPORTIONAL = "score_proportional"


def _cap_and_redistribute(
    raw_weights: dict[str, Decimal], max_weight: Decimal
) -> dict[str, Decimal]:
    """Iteratively cap position weights and redistribute excess proportionally."""
    if not raw_weights or max_weight <= Decimal(0):
        return {}

    weights = dict(raw_weights)
    target_sum = sum(weights.values(), Decimal(0))
    if target_sum <= Decimal(0):
        return {}

    # Normalize initial sum
    for s, w in weights.items():
        weights[s] = w / target_sum

    # Iterative capping and redistribution
    for _ in range(10):
        excess = Decimal(0)
        uncapped: list[str] = []
        for s, w in weights.items():
            if w > max_weight:
                excess += w - max_weight
                weights[s] = max_weight
            elif w < max_weight:
                uncapped.append(s)

        if excess <= Decimal("0.0001") or not uncapped:
            break

        uncapped_sum = sum((weights[s] for s in uncapped), Decimal(0))
        if uncapped_sum <= Decimal(0):
            break

        for s in uncapped:
            weights[s] += excess * (weights[s] / uncapped_sum)

    # Round cleanly
    return {s: Decimal(str(round(w, 4))) for s, w in weights.items()}


@dataclass(frozen=True, slots=True)
class NiftyMomentumStrategy:
    """Semi-annual blended 6m/12m risk-adjusted momentum portfolio.

    ``absolute_momentum``:
    - When False (default, replicating NSE NIFTY200 Momentum 30 index methodology):
      Fully invested across the top candidates by cross-sectional z-score,
      regardless of whether absolute returns are positive or negative.
    - When True (dual momentum): Requires candidate's trailing 12m return to be
      positive (> 0), parking unallocated capital in cash during broad market downturns.
    """

    portfolio_size: int = 30
    weighting: WeightingScheme = WeightingScheme.INVERSE_VOL
    rebalance_months: tuple[int, ...] = (6, 12)  # June and December
    max_position_weight: Decimal = Decimal("0.05")  # 5% index cap
    absolute_momentum: bool = False

    def __post_init__(self) -> None:
        if self.portfolio_size < 1:
            raise ValueError(f"portfolio_size must be at least 1, got {self.portfolio_size}")
        if not Decimal(0) < self.max_position_weight <= Decimal(1):
            raise ValueError(
                f"max_position_weight must lie in (0, 1], got {self.max_position_weight}"
            )
        if not self.rebalance_months:
            raise ValueError("rebalance_months must name at least one month")
        if any(m < 1 or m > _MONTHS_IN_YEAR for m in self.rebalance_months):
            raise ValueError(f"rebalance_months must be 1-12, got {self.rebalance_months}")
        # `size * cap < 1` cannot be satisfied by any allocation, so
        # `_cap_and_redistribute` would run out of uncapped names and silently
        # leave the remainder in cash. A top-12 book at the 5% index cap is 60%
        # invested and 40% idle, which looks like a defensive strategy and is
        # actually a configuration error.
        reachable = Decimal(self.portfolio_size) * self.max_position_weight
        if reachable < Decimal(1):
            raise ValueError(
                f"portfolio_size {self.portfolio_size} at max_position_weight "
                f"{self.max_position_weight} can only ever invest {reachable:.0%} of capital; "
                f"raise the cap to at least {Decimal(1) / Decimal(self.portfolio_size):.4f} "
                f"or hold at least {-(-1 // self.max_position_weight):.0f} names"
            )

    @property
    def name(self) -> str:
        suffix = "_dual" if self.absolute_momentum else ""
        return f"nifty_momentum_{self.portfolio_size}_{self.weighting.value}{suffix}"

    def is_rebalance_date(self, on: date, *, previous: date | None) -> bool:
        """Semi-annual rebalancing triggered on the first session of rebalance months."""
        if previous is None:
            return True
        return on.month in self.rebalance_months and on.month != previous.month

    def target_weights(self, panel: dict[str, pd.DataFrame], *, on: date) -> dict[str, Decimal]:
        """Compute blended 6m/12m z-score momentum targets."""
        target_ts = pd.Timestamp(on, tz=IST)
        metrics = self._calculate_universe_metrics(panel, target_ts)
        if not metrics:
            return {}

        scores_6m, scores_12m, vols, returns_12m = metrics
        common = sorted(set(scores_6m) & set(scores_12m))
        if len(common) < min(self.portfolio_size, _MIN_TRACKING_UNIVERSE):
            return {}

        combined_scores = self._standardize_and_blend(common, scores_6m, scores_12m)
        selected = self._select_candidates(combined_scores, returns_12m)
        if not selected:
            return {}

        raw_weights = self._compute_weights(selected, combined_scores, vols)
        return _cap_and_redistribute(raw_weights, self.max_position_weight)

    def _calculate_universe_metrics(
        self, panel: dict[str, pd.DataFrame], target_ts: pd.Timestamp
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]] | None:
        scores_6m: dict[str, float] = {}
        scores_12m: dict[str, float] = {}
        vols: dict[str, float] = {}
        returns_12m: dict[str, float] = {}

        for symbol, frame in panel.items():
            hist = frame[frame.index <= target_ts]["close"].dropna()
            if len(hist) < _LONG_HORIZON_SESSIONS:
                continue

            p_now = float(hist.iloc[-1])
            p_12m = float(hist.iloc[-_LONG_HORIZON_SESSIONS])
            p_6m = float(hist.iloc[-_MIN_HORIZON_SESSIONS])
            if p_12m <= 0 or p_6m <= 0:
                continue

            ret_12m = (p_now / p_12m) - 1.0
            ret_6m = (p_now / p_6m) - 1.0

            d12 = hist.iloc[-_LONG_HORIZON_SESSIONS:].pct_change().dropna()
            vol_12m = float(d12.std()) * _ANNUAL_SCALE

            d6 = hist.iloc[-_MIN_HORIZON_SESSIONS:].pct_change().dropna()
            vol_6m = float(d6.std()) * _ANNUAL_SCALE

            if vol_12m <= 0 or vol_6m <= 0:
                continue

            scores_12m[symbol] = ret_12m / vol_12m
            scores_6m[symbol] = ret_6m / vol_6m
            vols[symbol] = vol_12m
            returns_12m[symbol] = ret_12m

        return scores_6m, scores_12m, vols, returns_12m

    def _standardize_and_blend(
        self,
        symbols: list[str],
        scores_6m: dict[str, float],
        scores_12m: dict[str, float],
    ) -> dict[str, float]:
        vals_6m = np.array([scores_6m[s] for s in symbols])
        vals_12m = np.array([scores_12m[s] for s in symbols])

        std_6 = float(np.std(vals_6m))
        std_12 = float(np.std(vals_12m))

        z_6 = (vals_6m - float(np.mean(vals_6m))) / std_6 if std_6 > 0 else np.zeros_like(vals_6m)
        z_12 = (
            (vals_12m - float(np.mean(vals_12m))) / std_12
            if std_12 > 0
            else np.zeros_like(vals_12m)
        )

        blended = 0.5 * z_6 + 0.5 * z_12
        return dict(zip(symbols, blended, strict=True))

    def _select_candidates(
        self,
        combined_scores: dict[str, float],
        returns_12m: dict[str, float],
    ) -> list[str]:
        top = sorted(combined_scores.items(), key=lambda pair: -pair[1])[: self.portfolio_size]
        if self.absolute_momentum:
            # Absolute trend gate: requires positive 12-month return
            return [s for s, _ in top if returns_12m.get(s, 0.0) > 0.0]
        # Pure cross-sectional relative momentum: remain fully invested across top z-scores
        return [s for s, _ in top]

    def _compute_weights(
        self,
        selected: list[str],
        combined_scores: dict[str, float],
        vols: dict[str, float],
    ) -> dict[str, Decimal]:
        n = len(selected)
        if self.weighting == WeightingScheme.EQUAL:
            w = Decimal(1) / Decimal(n)
            return dict.fromkeys(selected, w)

        if self.weighting == WeightingScheme.INVERSE_VOL:
            inv_vols = {s: 1.0 / vols[s] for s in selected if vols.get(s, 0) > 0}
            tot_inv = sum(inv_vols.values())
            if tot_inv <= 0:
                w = Decimal(1) / Decimal(n)
                return dict.fromkeys(selected, w)
            return {s: Decimal(str(round(iv / tot_inv, 4))) for s, iv in inv_vols.items()}

        # SCORE_PROPORTIONAL: shift positive so minimum score > 0
        min_sc = min(combined_scores[s] for s in selected)
        shift = abs(min_sc) + 0.1 if min_sc <= 0 else 0.0
        pos_scores = {s: combined_scores[s] + shift for s in selected}
        tot_sc = sum(pos_scores.values())
        if tot_sc <= 0:
            w = Decimal(1) / Decimal(n)
            return dict.fromkeys(selected, w)
        return {s: Decimal(str(round(sc / tot_sc, 4))) for s, sc in pos_scores.items()}
