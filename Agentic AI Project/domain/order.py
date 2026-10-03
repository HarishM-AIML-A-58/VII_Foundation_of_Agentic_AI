"""Order requests and their broker-acknowledged state."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from uuid import UUID, uuid4

from trading_agent.domain.enums import Exchange, OrderStatus, OrderType, ProductType, Side

__all__ = ["Order", "OrderAck", "OrderRequest"]


@dataclass(frozen=True, slots=True)
class OrderRequest:
    """What we intend to send to the broker.

    ``idempotency_key`` is carried from the originating
    :class:`~trading_agent.domain.signal.TradeSignal` and is what makes
    submission safe to retry.

    ``algo_id`` is the exchange-issued identifier required by SEBI's
    "Safer participation of retail investors in Algorithmic trading" framework
    (circular dated 4 February 2025). Every automated order must be traceable
    back to a registered algorithm, so the identifier travels *with the order*
    rather than living in configuration the order cannot prove it read. It is
    optional at this layer -- a backtest has no exchange registration and must
    not need one -- and made mandatory at the live boundary by
    :func:`trading_agent.execution.compliance.preflight`.
    """

    symbol: str
    side: Side
    quantity: int
    order_type: OrderType
    product: ProductType
    idempotency_key: str
    exchange: Exchange = Exchange.NSE
    #: Required for LIMIT and SL orders, ignored for MARKET.
    limit_price: Decimal | None = None
    #: Trigger for SL and SL-M orders.
    trigger_price: Decimal | None = None
    signal_id: UUID | None = None
    #: Exchange-issued algo identifier. Empty is legitimate in backtest and
    #: paper; the live gate rejects it.
    algo_id: str = ""

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"quantity must be positive, got {self.quantity}")
        if self.order_type in {OrderType.LIMIT, OrderType.STOP_LOSS} and self.limit_price is None:
            raise ValueError(f"{self.order_type} requires a limit_price")
        if (
            self.order_type in {OrderType.STOP_LOSS, OrderType.STOP_LOSS_MARKET}
            and self.trigger_price is None
        ):
            raise ValueError(f"{self.order_type} requires a trigger_price")
        if not self.idempotency_key:
            raise ValueError("idempotency_key is mandatory; unkeyed orders cannot be retried")
        if self.side is Side.SELL and self.product is ProductType.CNC and self.quantity <= 0:
            # Unreachable given the quantity check above; kept so the invariant
            # is stated where the fields are, not only in the risk layer.
            raise ValueError("a CNC sell must reduce an existing holding")


@dataclass(frozen=True, slots=True)
class OrderAck:
    """The broker's immediate response to a submission."""

    broker_order_id: str
    idempotency_key: str
    accepted_at: datetime
    status: OrderStatus = OrderStatus.PENDING
    message: str | None = None
    #: True when the broker returned an existing order for this key rather
    #: than creating a new one -- i.e. the retry guard did its job.
    was_duplicate: bool = False


@dataclass(frozen=True, slots=True)
class Order:
    """An order as the broker currently reports it.

    The broker is the source of truth. This is a local projection that the
    reconciler converges toward it, never the other way around.
    """

    broker_order_id: str
    symbol: str
    side: Side
    quantity: int
    filled_quantity: int
    status: OrderStatus
    order_type: OrderType
    product: ProductType
    placed_at: datetime
    exchange: Exchange = Exchange.NSE
    average_price: Decimal | None = None
    limit_price: Decimal | None = None
    trigger_price: Decimal | None = None
    updated_at: datetime | None = None
    rejection_reason: str | None = None
    idempotency_key: str | None = None
    signal_id: UUID | None = None
    order_id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        if self.filled_quantity < 0:
            raise ValueError(f"filled_quantity must be non-negative, got {self.filled_quantity}")
        if self.filled_quantity > self.quantity:
            raise ValueError(
                f"filled_quantity {self.filled_quantity} exceeds quantity {self.quantity}"
            )

    @property
    def pending_quantity(self) -> int:
        return self.quantity - self.filled_quantity

    @property
    def is_complete(self) -> bool:
        return self.status.is_terminal

    @property
    def is_partial(self) -> bool:
        return 0 < self.filled_quantity < self.quantity
