"""Open positions."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from trading_agent.domain.enums import Exchange, ProductType, Side
from trading_agent.domain.money import Rupees

__all__ = ["Position"]


@dataclass(frozen=True, slots=True)
class Position:
    """A net holding in one symbol.

    ``quantity`` is signed: positive is long, negative is short, zero means
    flat. A signed quantity removes an entire class of bug where side and
    magnitude drift out of agreement.
    """

    symbol: str
    quantity: int
    average_price: Decimal
    product: ProductType
    exchange: Exchange = Exchange.NSE
    last_price: Decimal | None = None
    realised_pnl: Rupees = field(default_factory=Rupees.zero)

    @property
    def side(self) -> Side | None:
        if self.quantity == 0:
            return None
        return Side.BUY if self.quantity > 0 else Side.SELL

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0

    @property
    def cost_basis(self) -> Rupees:
        return Rupees(self.average_price * abs(self.quantity))

    @property
    def market_value(self) -> Rupees | None:
        if self.last_price is None:
            return None
        return Rupees(self.last_price * abs(self.quantity))

    @property
    def unrealised_pnl(self) -> Rupees | None:
        """Mark-to-market P&L, gross of exit costs.

        ``None`` when no mark is available. Exit charges are deliberately not
        deducted here -- that is the cost model's job, and folding it in would
        hide which of the two produced a number.
        """
        if self.last_price is None:
            return None
        return Rupees((self.last_price - self.average_price) * self.quantity)
