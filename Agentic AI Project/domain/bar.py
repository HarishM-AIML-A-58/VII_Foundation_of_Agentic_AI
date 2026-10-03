"""OHLCV bars."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

__all__ = ["Bar"]


@dataclass(frozen=True, slots=True)
class Bar:
    """A single OHLCV bar.

    ``timestamp`` is the bar's OPEN time and must be timezone-aware. Half the
    off-by-one-session bugs in backtesting come from mixing open-stamped and
    close-stamped bars, so the convention is fixed here and asserted.

    Prices are ``Decimal``. Bars feed the cost model, and a float close price
    would reintroduce the error :mod:`trading_agent.domain.money` exists to
    prevent.
    """

    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    #: Close adjusted for splits and bonuses. ``None`` until corporate actions
    #: have been applied; research code must use this, never raw ``close``.
    adjusted_close: Decimal | None = None

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("Bar.timestamp must be timezone-aware")
        if self.high < self.low:
            raise ValueError(f"high {self.high} is below low {self.low}")
        if not (self.low <= self.open <= self.high):
            raise ValueError(f"open {self.open} outside [{self.low}, {self.high}]")
        if not (self.low <= self.close <= self.high):
            raise ValueError(f"close {self.close} outside [{self.low}, {self.high}]")
        if self.volume < 0:
            raise ValueError(f"volume must be non-negative, got {self.volume}")

    @property
    def range(self) -> Decimal:
        return self.high - self.low

    @property
    def is_doji(self) -> bool:
        """True when open and close are within 5% of the bar's range."""
        if self.range == 0:
            return True
        return abs(self.close - self.open) / self.range < Decimal("0.05")

    @property
    def typical_price(self) -> Decimal:
        return (self.high + self.low + self.close) / 3
