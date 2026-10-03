"""Jev Kelly Position Sizing Engine.

Computes trade size as a capped fraction (1/4 Kelly maximum) of calibrated probability.
"""

from __future__ import annotations

from decimal import Decimal


class JevKellySizer:
    """Sizes positions strictly as a capped fraction of Kelly criterion."""

    def __init__(
        self,
        max_kelly_fraction: float = 0.25,
        min_probability_cutoff: float = 0.50,
        max_position_fraction: float = 0.25,
    ) -> None:
        self.max_kelly_fraction = max_kelly_fraction
        self.min_probability_cutoff = min_probability_cutoff
        self.max_position_fraction = max_position_fraction

    def compute_size(
        self,
        capital: Decimal,
        calibrated_prob: float,
        reward_risk_ratio: float = 2.0,
    ) -> Decimal:
        """Calculate capital allocation in Decimal currency units.

        f* = (b*p - q) / b
        where b = reward_risk_ratio, p = calibrated_prob, q = 1 - p.
        """
        if capital <= Decimal("0"):
            return Decimal("0")

        if calibrated_prob <= self.min_probability_cutoff or reward_risk_ratio <= 0:
            return Decimal("0")

        q = 1.0 - calibrated_prob
        full_kelly = (reward_risk_ratio * calibrated_prob - q) / reward_risk_ratio

        if full_kelly <= 0.0:
            return Decimal("0")

        # Cap at 1/4 Kelly fraction
        applied_fraction = min(full_kelly * self.max_kelly_fraction, self.max_position_fraction)
        allocated = capital * Decimal(str(round(applied_fraction, 4)))
        return allocated.quantize(Decimal("0.01"))
