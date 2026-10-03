"""Capital-gains tax on equity round trips.

**Why this is separate from the charge model, and why it is not per-trade.**
Brokerage, STT and stamp duty are levied on each execution and belong in
``costs/calculator.py``. Capital-gains tax is levied on *net gains over a
financial year*, after short-term losses offset short-term gains and long-term
losses offset long-term gains. It cannot be attached to a single round trip
because the rate a trade pays depends on every other trade in the same year. So
this takes a whole set of round trips and returns the year's liability.

**Why it matters enough to model.** Zerodha Varsity Module 7 (Markets and
Taxation) is explicit that tax is a first-order cost for anyone who trades
rather than holds, and the numbers this repo has been quoting are all
net-of-charges but pre-tax. For a monthly-rebalanced portfolio -- which realises
gains constantly and almost always at the short-term rate -- that gap is large
and always in the flattering direction. A strategy that clears buy-and-hold
pre-tax can lose to it after tax, because buy-and-hold realises nothing until
the end.

**The rates, and the reason they are dated data.** As of the FY2024-25 regime
(post the July 2024 budget), listed-equity delivery with STT paid:

* short-term (held <= 12 months): 20%
* long-term (held > 12 months): 12.5%, with the first Rs 1.25 lakh of long-term
  gains per year exempt.

These change with the budget, so they live in :class:`CapitalGainsRates` with an
``effective_from`` rather than as constants -- the same discipline the charge
schedules follow. A backtest over an older year should construct the rates that
applied then; the default is the current regime and is labelled as such.

**Scope, stated plainly.** This models delivery (CNC) equity gains only. It does
NOT model: intraday/MIS treated as speculative business income, F&O as
non-speculative business income, securities transaction tax (already in the
charge model), the basic exemption limit or slab interactions, surcharge and
cess, advance-tax timing, or carry-forward of losses across years. Those are
real and some are material; modelling them fully is an accountant's job, and
pretending otherwise would be worse than this honest subset. The one
cross-year effect that flatters most -- loss carry-forward -- is deliberately
omitted, which makes this estimate conservative (it may overstate tax), and
that is the safe direction for a go/no-go number.

Not tax advice. Confirm the regime and your own situation with a professional
before filing anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from trading_agent.domain.enums import ProductType
from trading_agent.domain.money import Rupees
from trading_agent.domain.trade import RoundTrip

__all__ = [
    "CapitalGainsRates",
    "CapitalGainsTax",
    "TaxLiability",
    "financial_year",
]

#: Held for more than this many days is long-term for listed equity.
_LONG_TERM_DAYS = 365

#: The Indian financial year starts in April.
_FY_START_MONTH = 4


@dataclass(frozen=True, slots=True)
class CapitalGainsRates:
    """A dated capital-gains regime for listed equity with STT paid.

    Rates are fractions. ``lt_exemption`` is the annual long-term gains
    exemption, applied per financial year before the long-term rate.
    """

    effective_from: date
    short_term_rate: Decimal
    long_term_rate: Decimal
    lt_exemption: Rupees
    label: str = ""

    @classmethod
    def current(cls) -> CapitalGainsRates:
        """The post-July-2024 regime. The default; state the date when quoting."""
        return cls(
            effective_from=date(2024, 7, 23),
            short_term_rate=Decimal("0.20"),
            long_term_rate=Decimal("0.125"),
            lt_exemption=Rupees("125000"),
            label="FY2024-25 (post 23 Jul 2024 budget)",
        )


def financial_year(on: date) -> int:
    """The Indian financial year (Apr-Mar) a date falls in, named by its start.

    FY2024-25 is April 2024 to March 2025, returned as ``2024``. Tax is
    reckoned per financial year, not per calendar year, and getting that wrong
    shifts a March trade into the wrong year's netting.
    """
    return on.year if on.month >= _FY_START_MONTH else on.year - 1


@dataclass(frozen=True, slots=True)
class TaxLiability:
    """One financial year's capital-gains position for delivery equity."""

    financial_year: int
    short_term_gains: Rupees
    long_term_gains: Rupees
    short_term_tax: Rupees
    long_term_tax: Rupees

    @property
    def total_tax(self) -> Rupees:
        return self.short_term_tax + self.long_term_tax

    @property
    def label(self) -> str:
        return f"FY{self.financial_year}-{str(self.financial_year + 1)[-2:]}"


class CapitalGainsTax:
    """Computes capital-gains tax over a set of round trips.

    Delivery only: MIS round trips are speculative business income under a
    different regime and are excluded here rather than taxed at the wrong rate.
    Their exclusion is counted so a caller can see how much P&L was set aside.
    """

    __slots__ = ("_rates",)

    def __init__(self, rates: CapitalGainsRates | None = None) -> None:
        self._rates = rates or CapitalGainsRates.current()

    @property
    def rates(self) -> CapitalGainsRates:
        return self._rates

    def _is_long_term(self, trip: RoundTrip) -> bool:
        return trip.holding_period_days > _LONG_TERM_DAYS

    def liability_by_year(self, trips: list[RoundTrip]) -> dict[int, TaxLiability]:
        """Tax owed per financial year, netting gains against losses within each.

        Short-term and long-term are netted separately, as the law requires:
        a short-term loss reduces short-term gains, not long-term ones. The
        realising date is the EXIT date -- a position is not a realised gain
        until it is sold, whatever year it was opened in.
        """
        st_by_year: dict[int, Rupees] = {}
        lt_by_year: dict[int, Rupees] = {}

        for trip in trips:
            if trip.product is not ProductType.CNC:
                continue  # speculative/business income; a different regime
            year = financial_year(trip.exit_at.date())
            # Net of the charges already paid on the trade -- tax is on the gain
            # after transaction costs, not on the gross move.
            gain = trip.net_pnl
            if self._is_long_term(trip):
                lt_by_year[year] = lt_by_year.get(year, Rupees.zero()) + gain
            else:
                st_by_year[year] = st_by_year.get(year, Rupees.zero()) + gain

        liabilities: dict[int, TaxLiability] = {}
        for year in sorted(set(st_by_year) | set(lt_by_year)):
            st_gain = st_by_year.get(year, Rupees.zero())
            lt_gain = lt_by_year.get(year, Rupees.zero())

            # Tax applies only to a net gain; a net loss in a bucket owes
            # nothing (and, since carry-forward is not modelled, simply lapses).
            st_tax = (
                Rupees(st_gain.value * self._rates.short_term_rate)
                if st_gain > Rupees.zero()
                else Rupees.zero()
            )
            taxable_lt = lt_gain - self._rates.lt_exemption
            lt_tax = (
                Rupees(taxable_lt.value * self._rates.long_term_rate)
                if taxable_lt > Rupees.zero()
                else Rupees.zero()
            )
            liabilities[year] = TaxLiability(
                financial_year=year,
                short_term_gains=st_gain,
                long_term_gains=lt_gain,
                short_term_tax=st_tax,
                long_term_tax=lt_tax,
            )
        return liabilities

    def total_tax(self, trips: list[RoundTrip]) -> Rupees:
        """Total capital-gains tax across every financial year in ``trips``."""
        return Rupees(
            sum(
                (liability.total_tax.value for liability in self.liability_by_year(trips).values()),
                Decimal(0),
            )
        )

    def excluded_pnl(self, trips: list[RoundTrip]) -> Rupees:
        """Net P&L set aside as non-delivery (MIS), not taxed by this model.

        Reported rather than hidden: a strategy that is mostly intraday has most
        of its tax unmodelled here, and the caller should know that before
        trusting the after-tax figure.
        """
        return Rupees(
            sum(
                (t.net_pnl.value for t in trips if t.product is not ProductType.CNC),
                Decimal(0),
            )
        )
