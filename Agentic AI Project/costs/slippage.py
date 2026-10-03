"""Slippage model.

Charges are knowable; slippage is estimated. Keeping the two apart matters:
when a backtest overstates returns, you need to know whether the cost schedule
is wrong (fixable, verifiable) or the fill assumption was optimistic (a
modelling judgement).

The default model is spread plus square-root impact:

    slippage_bps = half_spread_bps + coeff * sqrt(participation) * 10_000

where ``participation`` is order quantity over average daily volume. The
square root is the standard empirical shape -- impact grows sublinearly with
size, so a naive linear model badly overcharges small orders.

**Where the coefficient comes from.** The square-root law is usually written
``impact = Y * sigma_daily * sqrt(Q / ADV)`` with ``Y`` close to 1, so the
coefficient above is daily volatility expressed as a fraction. NSE large caps
run near 1.5% daily, which is the 0.015 default. That makes a 1%-of-ADV order
cost about 15 bps of impact -- in line with the published estimates.

The number matters more than it looks. This defaulted to 0.1 for a while, which
is a 10% daily volatility and prices the same 1%-of-ADV order at 100 bps. That
is not a conservative assumption; it is a wrong one, and it makes any strategy
that trades in size look unprofitable for a reason that is not real.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from enum import StrEnum

from trading_agent.domain.enums import Side

__all__ = ["SlippageModel", "SlippageModelType"]


class SlippageModelType(StrEnum):
    NONE = "none"
    FIXED_BPS = "fixed_bps"
    SPREAD_PLUS_IMPACT = "spread_plus_impact"


#: Standard NSE equity tick. Prices off the tick grid cannot trade.
DEFAULT_TICK_SIZE = Decimal("0.05")


@dataclass(frozen=True, slots=True)
class SlippageModel:
    """Estimates the price actually achieved versus the reference price."""

    model: SlippageModelType = SlippageModelType.SPREAD_PLUS_IMPACT
    half_spread_bps: Decimal = Decimal("2.0")
    impact_coefficient: Decimal = Decimal("0.015")
    fixed_bps: Decimal = Decimal("5.0")
    #: Fills are snapped to this grid. A price like 1271.937216 is not a price
    #: any exchange could have printed, and carrying it through to P&L makes
    #: results look more precise than the model actually is.
    tick_size: Decimal = DEFAULT_TICK_SIZE

    def __post_init__(self) -> None:
        if self.half_spread_bps < 0:
            raise ValueError(f"half_spread_bps must be non-negative, got {self.half_spread_bps}")
        if self.impact_coefficient < 0:
            raise ValueError(
                f"impact_coefficient must be non-negative, got {self.impact_coefficient}"
            )

    def slippage_bps(self, *, quantity: int, average_daily_volume: int | None = None) -> Decimal:
        """Estimated slippage in basis points, always non-negative."""
        if self.model is SlippageModelType.NONE:
            return Decimal(0)
        if self.model is SlippageModelType.FIXED_BPS:
            return self.fixed_bps

        impact_bps = Decimal(0)
        if average_daily_volume and average_daily_volume > 0:
            participation = Decimal(quantity) / Decimal(average_daily_volume)
            # Decimal.sqrt is exact-context; float would reintroduce drift.
            impact_bps = self.impact_coefficient * participation.sqrt() * Decimal(10_000)
        return self.half_spread_bps + impact_bps

    def apply(
        self,
        *,
        reference_price: Decimal,
        side: Side,
        quantity: int,
        average_daily_volume: int | None = None,
    ) -> Decimal:
        """The fill price after slippage.

        Slippage always works against you: buys fill higher, sells fill lower.
        A model that can help you is a modelling error, not a lucky fill.
        """
        if reference_price <= 0:
            raise ValueError(f"reference_price must be positive, got {reference_price}")
        bps = self.slippage_bps(quantity=quantity, average_daily_volume=average_daily_volume)
        adjustment = reference_price * bps / Decimal(10_000)
        raw = reference_price + adjustment if side is Side.BUY else reference_price - adjustment
        return self.snap_to_tick(raw, side)

    def snap_to_tick(self, price: Decimal, side: Side) -> Decimal:
        """Round ``price`` onto the exchange tick grid, against the trader.

        Buys round UP and sells round DOWN, so tick rounding can never
        manufacture a better fill than the model estimated. Rounding to
        nearest would give back a fraction of the slippage at random.
        """
        if self.tick_size <= 0:
            return price
        rounding = ROUND_UP if side is Side.BUY else ROUND_DOWN
        ticks = (price / self.tick_size).quantize(Decimal(1), rounding=rounding)
        return (ticks * self.tick_size).quantize(Decimal("0.01"))
