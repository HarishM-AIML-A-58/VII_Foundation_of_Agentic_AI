"""Closed vocabularies shared by every layer.

String-valued so they serialise cleanly to JSON, Parquet and Postgres without
a mapping table, and stay readable in logs.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "Action",
    "Exchange",
    "OrderStatus",
    "OrderType",
    "ProductType",
    "Side",
    "SignalStatus",
]


class Exchange(StrEnum):
    NSE = "NSE"
    BSE = "BSE"


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"

    @property
    def opposite(self) -> Side:
        return Side.SELL if self is Side.BUY else Side.BUY


class ProductType(StrEnum):
    """Zerodha product codes.

    ``CNC`` settles to the demat account (delivery); ``MIS`` is squared off
    intraday. They carry different STT, stamp duty and DP charges, so this is
    the primary key into the cost model.
    """

    CNC = "CNC"
    MIS = "MIS"


class OrderType(StrEnum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "SL"
    STOP_LOSS_MARKET = "SL-M"


class OrderStatus(StrEnum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

    @property
    def is_terminal(self) -> bool:
        """Terminal orders are never polled again by the reconciler."""
        return self in {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED}


class Action(StrEnum):
    """What a strategy or agent graph concluded."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

    @property
    def is_actionable(self) -> bool:
        return self is not Action.HOLD


class SignalStatus(StrEnum):
    """Lifecycle of a signal from generation to resolution."""

    GENERATED = "GENERATED"
    VETOED = "VETOED"  # risk manager rejected it
    SIZED = "SIZED"
    SUBMITTED = "SUBMITTED"
    EXECUTED = "EXECUTED"
    EXPIRED = "EXPIRED"  # never filled within its validity window
    FAILED = "FAILED"
