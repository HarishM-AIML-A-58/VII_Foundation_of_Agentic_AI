"""Domain models for NSE Equity Derivatives (Futures & Options).

Provides institutional contract master representations, lot-size rounding,
SPAN + Exposure margin accounting, and Section 43(5) business taxation.

Pure domain module: depends only on standard library and trading_agent.domain.money.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import StrEnum

from trading_agent.domain.money import Rupees

__all__ = [
    "DEFAULT_EXPOSURE_MARGIN_PCT",
    "DEFAULT_SPAN_MARGIN_PCT",
    "FoContract",
    "FoContractType",
    "calculate_business_tax",
    "calculate_margin_requirement",
    "calculate_mtm_pnl",
    "round_to_lot",
]

DEFAULT_SPAN_MARGIN_PCT = Decimal("0.175")  # 17.5% baseline SPAN margin
DEFAULT_EXPOSURE_MARGIN_PCT = Decimal("0.035")  # 3.5% baseline exposure margin
_CORPORATE_TAX_RATE = Decimal("0.25")  # 25% base corporate tax rate


class FoContractType(StrEnum):
    """NSE derivatives contract type."""

    FUTSTK = "FUTSTK"  # Stock Futures
    FUTIDX = "FUTIDX"  # Index Futures
    OPTSTK = "OPTSTK"  # Stock Options
    OPTIDX = "OPTIDX"  # Index Options


@dataclass(frozen=True, slots=True)
class FoContract:
    """NSE Equity Derivative Contract Specification."""

    symbol: str
    underlying: str
    contract_type: FoContractType
    expiry_date: date
    lot_size: int
    strike_price: Rupees | None = None

    def __post_init__(self) -> None:
        if self.lot_size <= 0:
            raise ValueError(f"lot_size must be positive, got {self.lot_size}")
        if not self.symbol:
            raise ValueError("symbol cannot be empty")
        if not self.underlying:
            raise ValueError("underlying cannot be empty")


def round_to_lot(target_shares: int, lot_size: int) -> int:
    """Round target share quantity to the nearest integer multiple of lot_size.

    Negative quantities (short positions) are preserved with proper rounding.
    """
    if lot_size <= 0:
        raise ValueError(f"lot_size must be positive, got {lot_size}")
    lots = round(target_shares / lot_size)
    return lots * lot_size


def calculate_margin_requirement(
    price: Rupees,
    quantity: int,
    span_pct: Decimal = DEFAULT_SPAN_MARGIN_PCT,
    exposure_pct: Decimal = DEFAULT_EXPOSURE_MARGIN_PCT,
) -> Rupees:
    """Compute total margin (SPAN + Exposure) required for an F&O position."""
    if quantity == 0:
        return Rupees.zero()
    total_margin_pct = span_pct + exposure_pct
    notional = price * abs(quantity)
    return notional * total_margin_pct


def calculate_mtm_pnl(
    quantity: int,
    entry_price: Rupees,
    current_price: Rupees,
) -> Rupees:
    """Compute Mark-to-Market (MTM) P&L for a derivatives position.

    Positive quantity represents Long; negative quantity represents Short.
    """
    price_delta = current_price - entry_price
    return price_delta * quantity


def calculate_business_tax(
    net_profit: Rupees,
    operational_expenses: Rupees | None = None,
    tax_rate: Decimal = _CORPORATE_TAX_RATE,
) -> tuple[Rupees, Rupees]:
    """Compute business tax under Section 43(5) of the Income Tax Act.

    F&O trading is treated as non-speculative business income. Operational
    expenses (exchange subscriptions, audit, platform fees) are deductible.

    Returns:
        tuple[Rupees, Rupees]: (taxable_income, tax_liability)
    """
    expenses = operational_expenses if operational_expenses is not None else Rupees.zero()
    taxable_income = net_profit - expenses
    if taxable_income <= Rupees.zero():
        return Rupees.zero(), Rupees.zero()
    tax_liability = taxable_income * tax_rate
    return taxable_income, tax_liability
