"""Cash, positions and equity accounting during a backtest.

Every fill goes through the cost model. There is no path that opens or closes
a position without paying charges, because the whole point of this project is
that a strategy which cannot survive Indian transaction costs is not a
strategy.

Money is :class:`~trading_agent.domain.money.Rupees` throughout. Prices remain
``Decimal``. Nothing here touches ``float``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal

from trading_agent.costs.breakdown import ChargeBreakdown
from trading_agent.costs.calculator import ChargeCalculator
from trading_agent.domain.enums import ProductType, Side
from trading_agent.domain.money import Rupees
from trading_agent.domain.trade import RoundTrip
from trading_agent.observability import get_logger

__all__ = ["Holding", "InsufficientCashError", "Portfolio"]

log = get_logger(__name__)


class InsufficientCashError(RuntimeError):
    """The portfolio cannot fund a buy, including its charges."""


@dataclass(slots=True)
class Holding:
    """An open long position inside the backtest."""

    symbol: str
    quantity: int
    average_price: Decimal
    opened_at: datetime
    product: ProductType
    stop_loss: Decimal
    target: Decimal
    entry_charges: Rupees = field(default_factory=Rupees.zero)

    @property
    def cost_basis(self) -> Rupees:
        return Rupees(self.average_price * self.quantity)


class Portfolio:
    """Tracks cash, holdings and realised results over a backtest run.

    Long-only for the baseline: shorting Indian equity cash requires intraday
    product handling and margin rules that Phase 2 deliberately does not model.
    Adding it later means adding it honestly, not approximating it now.
    """

    __slots__ = (
        "_calculator",
        "_cash",
        "_cash_yield_accrued",
        "_charges",
        "_holdings",
        "_initial",
        "_round_trips",
    )

    def __init__(self, initial_capital: Rupees, calculator: ChargeCalculator) -> None:
        if initial_capital <= Rupees.zero():
            raise ValueError("initial capital must be positive")
        self._initial = initial_capital
        self._cash = initial_capital
        self._calculator = calculator
        self._holdings: dict[str, Holding] = {}
        self._round_trips: list[RoundTrip] = []
        self._charges = Rupees.zero()
        self._cash_yield_accrued = Rupees.zero()

    # ------------------------------------------------------------- accessors

    @property
    def cash(self) -> Rupees:
        return self._cash

    @property
    def cash_yield_accrued(self) -> Rupees:
        return self._cash_yield_accrued

    @property
    def initial_capital(self) -> Rupees:
        return self._initial

    @property
    def holdings(self) -> dict[str, Holding]:
        return dict(self._holdings)

    @property
    def round_trips(self) -> list[RoundTrip]:
        return list(self._round_trips)

    @property
    def total_charges(self) -> Rupees:
        return self._charges

    def holds(self, symbol: str) -> bool:
        return symbol in self._holdings

    def equity(self, marks: dict[str, Decimal]) -> Rupees:
        """Cash plus the marked value of open holdings.

        Exit charges are NOT deducted here. Equity is a mark, not a
        liquidation value; netting exit costs into it would double-count them
        when the position actually closes.
        """
        total = self._cash
        for symbol, holding in self._holdings.items():
            mark = marks.get(symbol, holding.average_price)
            total = total + Rupees(mark * holding.quantity)
        return total

    # ------------------------------------------------------------ operations

    def accrue_interest(self, annual_rate: float, trading_days: int = 250) -> Rupees:
        """Accrue daily interest on idle cash balances.

        In India, idle cash in broker accounts or overnight sweep liquid funds
        earns approximately the repo rate (~6.5%). Accruing this prevents penalizing
        cash-holding or risk-off regimes with an unearned negative Sharpe drag.
        """
        if self._cash <= Rupees.zero() or annual_rate <= 0:
            return Rupees.zero()
        daily_rate = Decimal(str(annual_rate)) / Decimal(str(trading_days))
        interest = Rupees(self._cash.value * daily_rate)
        if interest > Rupees.zero():
            self._cash = self._cash + interest
            self._cash_yield_accrued = self._cash_yield_accrued + interest
        return interest

    def affordable_quantity(
        self, *, price: Decimal, budget: Rupees, product: ProductType, on: date
    ) -> int:
        """Largest quantity ``budget`` can fund, charges included.

        Solved by trying the naive quantity and stepping down until the total
        outlay fits. Charges are non-linear in quantity (per-order brokerage
        caps, flat DP debits), so there is no closed form worth trusting.
        """
        if price <= 0:
            raise ValueError(f"price must be positive, got {price}")
        quantity = int(budget.value // price)

        while quantity > 0:
            charges = self._calculator.for_notional(
                price=price, quantity=quantity, side=Side.BUY, product=product, on=on
            )
            if Rupees(price * quantity) + charges.total <= budget:
                return quantity
            quantity -= 1
        return 0

    def buy(
        self,
        *,
        symbol: str,
        quantity: int,
        price: Decimal,
        at: datetime,
        product: ProductType,
        stop_loss: Decimal,
        target: Decimal,
    ) -> ChargeBreakdown:
        """Open a long position, paying entry charges from cash."""
        if quantity <= 0:
            raise ValueError(f"quantity must be positive, got {quantity}")
        if symbol in self._holdings:
            raise ValueError(f"already holding {symbol}; pyramiding is not modelled")

        charges = self._calculator.for_notional(
            price=price, quantity=quantity, side=Side.BUY, product=product, on=at.date()
        )
        outlay = Rupees(price * quantity) + charges.total
        if outlay > self._cash:
            raise InsufficientCashError(
                f"{symbol}: needs {outlay} including charges, cash is {self._cash}"
            )

        self._cash = self._cash - outlay
        self._charges = self._charges + charges.total
        self._holdings[symbol] = Holding(
            symbol=symbol,
            quantity=quantity,
            average_price=price,
            opened_at=at,
            product=product,
            stop_loss=stop_loss,
            target=target,
            entry_charges=charges.total,
        )
        return charges

    def sell(
        self,
        *,
        symbol: str,
        price: Decimal,
        at: datetime,
        reason: str,
        strategy: str,
        quantity: int | None = None,
    ) -> RoundTrip:
        """Close a holding partially or entirely and book the round trip."""
        holding = self._holdings.get(symbol)
        if holding is None:
            raise ValueError(f"not holding {symbol}")

        if quantity is not None and quantity <= 0:
            raise ValueError(f"sell quantity must be positive, got {quantity}")

        sell_qty = holding.quantity if quantity is None else min(quantity, holding.quantity)

        charges = self._calculator.for_notional(
            price=price,
            quantity=sell_qty,
            side=Side.SELL,
            product=holding.product,
            on=at.date(),
        )
        proceeds = Rupees(price * sell_qty) - charges.total
        self._cash = self._cash + proceeds
        self._charges = self._charges + charges.total

        # Pro-rata portion of entry charges attributed to this round trip
        entry_charges_share = (
            Rupees(holding.entry_charges.value * Decimal(sell_qty) / Decimal(holding.quantity))
            if holding.quantity > 0
            else Rupees.zero()
        )

        trip = RoundTrip(
            symbol=symbol,
            side=Side.BUY,
            quantity=sell_qty,
            entry_price=holding.average_price,
            exit_price=price,
            entry_at=holding.opened_at,
            exit_at=at,
            product=holding.product,
            # Both legs: the round trip's cost is what decides whether the
            # edge was real.
            total_charges=entry_charges_share + charges.total,
            strategy=strategy,
            exit_reason=reason,
        )
        self._round_trips.append(trip)

        if sell_qty >= holding.quantity:
            del self._holdings[symbol]
        else:
            holding.quantity -= sell_qty
            holding.entry_charges = holding.entry_charges - entry_charges_share

        return trip
