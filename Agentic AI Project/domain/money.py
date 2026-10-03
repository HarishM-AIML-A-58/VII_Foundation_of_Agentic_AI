"""Monetary quantities in Indian rupees.

Money is ``Decimal``, never ``float``. A float rupee accumulates
representation error across thousands of fills, and the first symptom is a
reconciliation mismatch against the broker that nobody can explain. Every
value that crosses a boundary is quantised to paisa here.

This module is pure: it imports nothing from the rest of the package.
"""

from __future__ import annotations

from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Final, Self

__all__ = ["PAISA", "RUPEE", "Rupees", "rupees", "sum_rupees"]

#: Smallest representable unit: one paisa.
PAISA: Final = Decimal("0.01")

#: One rupee.
RUPEE: Final = Decimal("1")

#: Indian digit grouping: the last three digits, then pairs (12,34,567).
_FIRST_GROUP: Final = 3
_SUBSEQUENT_GROUP: Final = 2


class Rupees:
    """An exact rupee amount, quantised to paisa.

    Immutable. Arithmetic between ``Rupees`` returns ``Rupees``; scaling by an
    ``int``/``Decimal`` is allowed, but multiplying two ``Rupees`` is not --
    rupees squared is not a thing, and forbidding it catches real bugs.

    >>> Rupees("100.005") + Rupees("0.004")
    Rupees('100.01')
    >>> Rupees("1500.50") * 3
    Rupees('4501.50')
    """

    __slots__ = ("_value",)

    _value: Decimal

    def __init__(self, value: Decimal | int | str) -> None:
        if isinstance(value, float):
            raise TypeError(
                "Rupees refuses float: use Decimal, int, or str to avoid "
                "floating-point inaccuracies"
            )
        try:
            raw = Decimal(value)
        except (InvalidOperation, ValueError) as exc:
            raise ValueError(f"not a valid rupee amount: {value!r}") from exc
        if not raw.is_finite():
            raise ValueError(f"rupee amount must be finite, got {value!r}")
        # Bankers' rounding for storage: unbiased across many small amounts.
        object.__setattr__(self, "_value", raw.quantize(PAISA, rounding=ROUND_HALF_EVEN))

    # -- construction ------------------------------------------------------

    @classmethod
    def zero(cls) -> Self:
        return cls(0)

    @classmethod
    def from_paisa(cls, paisa: int) -> Self:
        return cls(Decimal(paisa) / 100)

    # -- accessors ---------------------------------------------------------

    @property
    def value(self) -> Decimal:
        """The exact quantised :class:`~decimal.Decimal`."""
        return self._value

    @property
    def paisa(self) -> int:
        """The amount as a whole number of paisa. Use for storage and hashing."""
        return int(self._value.scaleb(2).to_integral_value(rounding=ROUND_HALF_EVEN))

    def is_zero(self) -> bool:
        return self._value == 0

    def round_to_rupee(self) -> Rupees:
        """Round to the nearest whole rupee, half away from zero.

        Contract notes present some statutory levies this way; see
        ``costs.calculator`` for where it is applied.
        """
        return Rupees(self._value.quantize(RUPEE, rounding=ROUND_HALF_UP))

    # -- arithmetic --------------------------------------------------------

    def __add__(self, other: Rupees) -> Rupees:
        if not isinstance(other, Rupees):
            return NotImplemented
        return Rupees(self._value + other._value)

    def __sub__(self, other: Rupees) -> Rupees:
        if not isinstance(other, Rupees):
            return NotImplemented
        return Rupees(self._value - other._value)

    def __mul__(self, factor: Decimal | int) -> Rupees:
        if isinstance(factor, float):
            raise TypeError(
                "refusing to scale Rupees by float; pass Decimal(str(x)) to keep it exact"
            )
        if not isinstance(factor, Decimal | int):
            return NotImplemented
        return Rupees(self._value * Decimal(factor))

    __rmul__ = __mul__

    def __truediv__(self, divisor: Decimal | int) -> Rupees:
        if isinstance(divisor, float):
            raise TypeError(
                "refusing to divide Rupees by float; pass Decimal(str(x)) to keep it exact"
            )
        if not isinstance(divisor, Decimal | int):
            return NotImplemented
        if Decimal(divisor) == 0:
            raise ZeroDivisionError("cannot divide a rupee amount by zero")
        return Rupees(self._value / Decimal(divisor))

    def __neg__(self) -> Rupees:
        return Rupees(-self._value)

    def __abs__(self) -> Rupees:
        return Rupees(abs(self._value))

    # -- comparison --------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Rupees):
            return NotImplemented
        return self._value == other._value

    def __lt__(self, other: Rupees) -> bool:
        if not isinstance(other, Rupees):
            return NotImplemented
        return self._value < other._value

    def __le__(self, other: Rupees) -> bool:
        if not isinstance(other, Rupees):
            return NotImplemented
        return self._value <= other._value

    def __gt__(self, other: Rupees) -> bool:
        if not isinstance(other, Rupees):
            return NotImplemented
        return self._value > other._value

    def __ge__(self, other: Rupees) -> bool:
        if not isinstance(other, Rupees):
            return NotImplemented
        return self._value >= other._value

    def __hash__(self) -> int:
        return hash(self.paisa)

    # -- rendering ---------------------------------------------------------

    def __repr__(self) -> str:
        return f"Rupees('{self._value}')"

    def __str__(self) -> str:
        """Indian-format with the rupee sign, e.g. ``₹12,34,567.89``."""
        sign = "-" if self._value < 0 else ""
        whole, _, frac = f"{abs(self._value):.2f}".partition(".")
        if len(whole) > _FIRST_GROUP:
            head, tail = whole[:-_FIRST_GROUP], whole[-_FIRST_GROUP:]
            groups: list[str] = []
            while len(head) > _SUBSEQUENT_GROUP:
                groups.insert(0, head[-_SUBSEQUENT_GROUP:])
                head = head[:-_SUBSEQUENT_GROUP]
            if head:
                groups.insert(0, head)
            whole = ",".join([*groups, tail])
        return f"{sign}₹{whole}.{frac}"


def rupees(value: Decimal | int | str) -> Rupees:
    """Shorthand constructor, convenient in tests and research code."""
    return Rupees(value)


def sum_rupees(amounts: object) -> Rupees:
    """Sum an iterable of :class:`Rupees`, returning zero when empty.

    ``sum()`` cannot be used directly because it seeds with ``int`` ``0``.
    """
    if not hasattr(amounts, "__iter__"):
        raise TypeError("sum_rupees expects an iterable of Rupees")
    total = Rupees.zero()
    for amount in amounts:
        if not isinstance(amount, Rupees):
            raise TypeError(f"expected Rupees, got {type(amount).__name__}")
        total = total + amount
    return total
