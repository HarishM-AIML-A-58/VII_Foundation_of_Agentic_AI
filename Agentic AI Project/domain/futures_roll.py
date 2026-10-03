"""Continuous Futures & Calendar Roll Engine.

Provides institutional calendar roll logic for NSE equity and index futures:
- NSE monthly expiry calculation with holiday roll-back rules.
- Basis and annualized cost of carry accounting.
- Calendar spread roll order generation for maintaining continuous exposure.
- Continuous price series back-adjustment ratio calculation.

Pure domain module: depends only on standard library and trading_agent.domain.money.
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta
from decimal import Decimal
from typing import Final

from trading_agent.domain.money import Rupees

__all__ = [
    "DAYS_PER_YEAR",
    "calculate_basis_and_carry",
    "calculate_roll_ratio",
    "generate_calendar_roll_orders",
    "get_nse_monthly_expiry",
]

DAYS_PER_YEAR: Final[float] = 365.0
_MIN_DTE: Final[int] = 1
_DAYS_PER_WEEK: Final[int] = 7
_MIN_MONTH: Final[int] = 1
_MAX_MONTH: Final[int] = 12
_ZERO_DECIMAL: Final[Decimal] = Decimal("0")
_WEEKEND_DAYS: Final[frozenset[int]] = frozenset({calendar.SATURDAY, calendar.SUNDAY})
_ONE_DAY: Final[timedelta] = timedelta(days=1)


def get_nse_monthly_expiry(
    year: int,
    month: int,
    holidays: set[date] | None = None,
) -> date:
    """Compute the NSE monthly derivatives expiry date.

    Finds the last Thursday of the month. If that Thursday falls on an exchange
    trading holiday, steps backward day-by-day until a non-holiday business day
    (Monday through Friday) is found.

    Parameters
    ----------
    year:
        Four-digit calendar year (e.g., 2026).
    month:
        Month number (1 to 12).
    holidays:
        Optional set of dates representing exchange clearing holidays.

    Returns
    -------
    date
        The final settlement/expiry date for the monthly contract.
    """
    if not (_MIN_MONTH <= month <= _MAX_MONTH):
        raise ValueError(f"month must be between {_MIN_MONTH} and {_MAX_MONTH}, got {month}")
    if year < 1:
        raise ValueError(f"year must be positive, got {year}")

    _, num_days = calendar.monthrange(year, month)
    last_day = date(year, month, num_days)
    offset = (last_day.weekday() - calendar.THURSDAY) % _DAYS_PER_WEEK
    candidate = last_day - timedelta(days=offset)

    holiday_set = holidays if holidays is not None else frozenset()
    while candidate in holiday_set or candidate.weekday() in _WEEKEND_DAYS:
        candidate -= _ONE_DAY

    return candidate


def calculate_basis_and_carry(
    futures_price: Rupees,
    spot_price: Rupees,
    dte: int,
) -> tuple[Rupees, float]:
    """Calculate the basis and annualized cost of carry.

    Basis represents the absolute difference between futures and spot prices:
        Basis = futures_price - spot_price

    Annualized Cost of Carry reflects the financing and holding yield:
        Carry = (basis / spot_price) * (365.0 / max(1, dte))

    Parameters
    ----------
    futures_price:
        Current futures contract price.
    spot_price:
        Current spot / cash market underlying price.
    dte:
        Days to expiration. Values less than 1 are clamped to 1 for
        the carry calculation to avoid division by zero.

    Returns
    -------
    tuple[Rupees, float]
        A pair consisting of:
        - basis: Rupees
        - annualized_cost_of_carry: float (e.g., 0.085 for 8.5% p.a.)
    """
    if spot_price.value <= _ZERO_DECIMAL:
        raise ValueError(f"spot_price must be positive, got {spot_price}")
    if dte < 0:
        raise ValueError(f"dte must be non-negative, got {dte}")

    basis = futures_price - spot_price
    effective_dte = max(_MIN_DTE, dte)
    carry = (float(basis.value) / float(spot_price.value)) * (DAYS_PER_YEAR / float(effective_dte))
    return basis, carry


def generate_calendar_roll_orders(
    symbol: str,
    current_expiry: date,
    next_expiry: date,
    quantity: int,
) -> tuple[int, int]:
    """Generate order quantities to execute a calendar spread roll.

    For an existing position of ``quantity`` shares in ``current_expiry``,
    closing requires ``-quantity`` (sell if long, buy to cover if short),
    and opening the far contract requires ``+quantity`` (buy if long,
    sell if short).

    Parameters
    ----------
    symbol:
        Underlying or contract symbol name.
    current_expiry:
        Expiry date of the contract currently held.
    next_expiry:
        Expiry date of the forward contract to roll into.
        Must be strictly after ``current_expiry``.
    quantity:
        Current position quantity (positive for long, negative for short).

    Returns
    -------
    tuple[int, int]
        (close_current_quantity, open_next_quantity)
    """
    if not symbol or not symbol.strip():
        raise ValueError("symbol cannot be empty")
    if next_expiry <= current_expiry:
        raise ValueError(
            f"next_expiry ({next_expiry}) must be strictly after current_expiry ({current_expiry})"
        )

    return -quantity, quantity


def calculate_roll_ratio(
    near_price: Rupees,
    next_price: Rupees,
) -> Decimal:
    """Calculate the ratio for back-adjusting continuous futures price series.

    Ratio = next_price / near_price

    Used in ratio-adjusted (multiplicative) continuous series construction to
    eliminate artificial price jumps across expiration boundaries without
    distorting percentage returns or introducing negative prices.

    Parameters
    ----------
    near_price:
        Settlement or roll-point price of the near (expiring) contract.
    next_price:
        Settlement or roll-point price of the next (deferred) contract.

    Returns
    -------
    Decimal
        The adjustment ratio (next_price / near_price).
    """
    if near_price.value <= _ZERO_DECIMAL:
        raise ValueError(f"near_price must be positive, got {near_price}")
    if next_price.value <= _ZERO_DECIMAL:
        raise ValueError(f"next_price must be positive, got {next_price}")

    return next_price.value / near_price.value
