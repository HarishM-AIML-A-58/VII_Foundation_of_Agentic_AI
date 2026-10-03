"""Executions and completed round trips."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from uuid import UUID, uuid4

from trading_agent.domain.enums import Exchange, ProductType, Side
from trading_agent.domain.money import Rupees

__all__ = ["Execution", "RoundTrip"]


@dataclass(frozen=True, slots=True)
class Execution:
    """One fill. The atomic unit the cost model charges against.

    A partially filled order produces several executions; charges are computed
    per execution because brokerage caps and DP debits apply per order and per
    scrip respectively, not per share.
    """

    symbol: str
    side: Side
    quantity: int
    price: Decimal
    executed_at: datetime
    product: ProductType
    exchange: Exchange = Exchange.NSE
    broker_order_id: str | None = None
    signal_id: UUID | None = None
    execution_id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        if self.executed_at.tzinfo is None:
            raise ValueError("Execution.executed_at must be timezone-aware")
        if self.quantity <= 0:
            raise ValueError(f"quantity must be positive, got {self.quantity}")
        if self.price <= 0:
            raise ValueError(f"price must be positive, got {self.price}")

    @property
    def turnover(self) -> Rupees:
        """Notional value. The base for every percentage-based levy."""
        return Rupees(self.price * self.quantity)


@dataclass(frozen=True, slots=True)
class RoundTrip:
    """A closed position: entry, exit, and what it actually cost.

    ``net_pnl`` is the only number worth reporting. Gross P&L on Indian
    equities routinely flatters a strategy that STT and stamp duty would have
    made unprofitable.
    """

    symbol: str
    side: Side
    quantity: int
    entry_price: Decimal
    exit_price: Decimal
    entry_at: datetime
    exit_at: datetime
    product: ProductType
    total_charges: Rupees
    exchange: Exchange = Exchange.NSE
    strategy: str | None = None
    signal_id: UUID | None = None
    exit_reason: str | None = None

    @property
    def gross_pnl(self) -> Rupees:
        direction = 1 if self.side is Side.BUY else -1
        return Rupees((self.exit_price - self.entry_price) * self.quantity * direction)

    @property
    def net_pnl(self) -> Rupees:
        return self.gross_pnl - self.total_charges

    @property
    def is_winner(self) -> bool:
        return self.net_pnl > Rupees.zero()

    @property
    def holding_period_days(self) -> int:
        return (self.exit_at.date() - self.entry_at.date()).days

    @property
    def return_pct(self) -> Decimal:
        """Net return on the capital committed, as a fraction."""
        basis = self.entry_price * self.quantity
        if basis == 0:
            return Decimal(0)
        return self.net_pnl.value / basis
