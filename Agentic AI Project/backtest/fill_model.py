"""How an intended trade becomes a fill.

**Next bar's open, plus slippage.** Filling at the same bar's close is
lookahead -- the strategy used that close to decide -- and is rejected in
settings validation rather than offered as an option.

Two exchange realities are modelled because ignoring them flatters results:

* **Gaps.** A stop at 1450 does not fill at 1450 when the bar opens at 1400.
  It fills at the open. Assuming the stop price is a free 50-point gift on
  every gap down, and gaps cluster exactly where strategies lose money.
* **Circuit limits.** A symbol locked limit-up or limit-down cannot be
  traded at all. Modelling a fill there invents liquidity that did not exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from trading_agent.costs.slippage import SlippageModel
from trading_agent.domain.enums import Side

__all__ = ["Fill", "FillModel"]


@dataclass(frozen=True, slots=True)
class Fill:
    """The outcome of attempting to trade."""

    filled: bool
    price: Decimal = Decimal(0)
    reason: str = ""


@dataclass(frozen=True, slots=True)
class FillModel:
    """Turns an intent plus the next bar into a fill price."""

    slippage: SlippageModel
    #: Fraction beyond which a bar is treated as circuit-locked and untradeable.
    circuit_limit_pct: Decimal = Decimal("0.20")

    def fill_at_open(
        self,
        *,
        next_open: Decimal,
        previous_close: Decimal,
        side: Side,
        quantity: int,
        average_daily_volume: int | None = None,
    ) -> Fill:
        """Fill an entry or a scheduled exit at the next bar's open."""
        if next_open <= 0:
            return Fill(False, reason="no next bar to fill against")

        move = abs(next_open - previous_close) / previous_close if previous_close else Decimal(0)
        if move >= self.circuit_limit_pct:
            return Fill(
                False,
                reason=f"circuit-locked: open moved {move:.1%} from the prior close",
            )

        price = self.slippage.apply(
            reference_price=next_open,
            side=side,
            quantity=quantity,
            average_daily_volume=average_daily_volume,
        )
        return Fill(True, price=price)

    def fill_stop(
        self,
        *,
        stop_price: Decimal,
        bar_open: Decimal,
        bar_low: Decimal,
        quantity: int,
        average_daily_volume: int | None = None,
    ) -> Fill:
        """Fill a protective stop on a long position.

        If the bar OPENS below the stop, the gap already happened and the fill
        is at the open, not the stop. Assuming otherwise hands the strategy
        free money precisely when it is losing.
        """
        if bar_open <= stop_price:
            reference = bar_open
        elif bar_low <= stop_price:
            reference = stop_price
        else:
            return Fill(False, reason="stop not touched")

        price = self.slippage.apply(
            reference_price=reference,
            side=Side.SELL,
            quantity=quantity,
            average_daily_volume=average_daily_volume,
        )
        return Fill(True, price=price)

    def fill_target(
        self,
        *,
        target_price: Decimal,
        bar_open: Decimal,
        bar_high: Decimal,
        quantity: int,
        average_daily_volume: int | None = None,
    ) -> Fill:
        """Fill a profit target on a long position.

        Symmetric to the stop: a gap ABOVE the target fills at the open, which
        is favourable, and modelling it is what keeps the two sides honest.
        """
        if bar_open >= target_price:
            reference = bar_open
        elif bar_high >= target_price:
            reference = target_price
        else:
            return Fill(False, reason="target not touched")

        price = self.slippage.apply(
            reference_price=reference,
            side=Side.SELL,
            quantity=quantity,
            average_daily_volume=average_daily_volume,
        )
        return Fill(True, price=price)
