"""Tradable instruments."""

from __future__ import annotations

from dataclasses import dataclass

from trading_agent.domain.enums import Exchange

__all__ = ["Instrument"]


@dataclass(frozen=True, slots=True)
class Instrument:
    """An NSE equity.

    ``symbol`` is the exchange trading symbol without any vendor suffix --
    ``RELIANCE``, not ``RELIANCE.NS``. Vendor-specific decoration belongs in
    the adapter that needs it, never in the domain.
    """

    symbol: str
    exchange: Exchange = Exchange.NSE
    name: str | None = None
    sector: str | None = None
    isin: str | None = None
    #: Minimum tradable increment. 1 for cash equity; kept explicit so the
    #: sizing code never assumes it.
    lot_size: int = 1
    #: Minimum price increment on the exchange, in rupees.
    tick_size: str = "0.05"

    def __post_init__(self) -> None:
        if not self.symbol or self.symbol != self.symbol.strip().upper():
            raise ValueError(f"symbol must be non-empty and uppercase, got {self.symbol!r}")
        if self.lot_size < 1:
            raise ValueError(f"lot_size must be >= 1, got {self.lot_size}")

    @property
    def key(self) -> str:
        """Stable identifier used for cache keys and Parquet partitions."""
        return f"{self.exchange.value}:{self.symbol}"

    def __str__(self) -> str:
        return self.key
