"""Itemised charge breakdown.

Every levy is kept as a separate line rather than collapsed into a total. When
``costs verify`` disagrees with a contract note, an itemised breakdown says
*which* levy is wrong; a single total says only that something is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from trading_agent.domain.money import Rupees

__all__ = ["ChargeBreakdown"]


@dataclass(frozen=True, slots=True)
class ChargeBreakdown:
    """The full set of charges on one execution.

    Line names match the vocabulary Indian contract notes use, so a human can
    read this side by side with a broker statement without translating.
    """

    turnover: Rupees
    brokerage: Rupees = field(default_factory=Rupees.zero)
    securities_transaction_tax: Rupees = field(default_factory=Rupees.zero)
    exchange_transaction_charge: Rupees = field(default_factory=Rupees.zero)
    sebi_turnover_fee: Rupees = field(default_factory=Rupees.zero)
    investor_protection_fund: Rupees = field(default_factory=Rupees.zero)
    stamp_duty: Rupees = field(default_factory=Rupees.zero)
    goods_and_services_tax: Rupees = field(default_factory=Rupees.zero)
    depository_participant_charge: Rupees = field(default_factory=Rupees.zero)

    @property
    def statutory_levies(self) -> Rupees:
        """Charges collected on behalf of the state or the exchange.

        These are unavoidable: no broker choice reduces them.
        """
        return (
            self.securities_transaction_tax
            + self.exchange_transaction_charge
            + self.sebi_turnover_fee
            + self.investor_protection_fund
            + self.stamp_duty
        )

    @property
    def broker_charges(self) -> Rupees:
        """Charges the broker sets, and which a different broker would change."""
        return self.brokerage + self.depository_participant_charge

    @property
    def total(self) -> Rupees:
        return (
            self.brokerage
            + self.securities_transaction_tax
            + self.exchange_transaction_charge
            + self.sebi_turnover_fee
            + self.investor_protection_fund
            + self.stamp_duty
            + self.goods_and_services_tax
            + self.depository_participant_charge
        )

    @property
    def total_as_bps_of_turnover(self) -> float:
        """Total cost in basis points. The number worth carrying in your head."""
        if self.turnover.is_zero():
            return 0.0
        return float(self.total.value / self.turnover.value) * 10_000

    def __add__(self, other: ChargeBreakdown) -> ChargeBreakdown:
        """Combine two breakdowns, e.g. the two legs of a round trip."""
        if not isinstance(other, ChargeBreakdown):
            return NotImplemented
        return ChargeBreakdown(
            turnover=self.turnover + other.turnover,
            brokerage=self.brokerage + other.brokerage,
            securities_transaction_tax=(
                self.securities_transaction_tax + other.securities_transaction_tax
            ),
            exchange_transaction_charge=(
                self.exchange_transaction_charge + other.exchange_transaction_charge
            ),
            sebi_turnover_fee=self.sebi_turnover_fee + other.sebi_turnover_fee,
            investor_protection_fund=(
                self.investor_protection_fund + other.investor_protection_fund
            ),
            stamp_duty=self.stamp_duty + other.stamp_duty,
            goods_and_services_tax=self.goods_and_services_tax + other.goods_and_services_tax,
            depository_participant_charge=(
                self.depository_participant_charge + other.depository_participant_charge
            ),
        )

    def as_lines(self) -> list[tuple[str, Rupees]]:
        """Ordered, human-readable lines for CLI and UI rendering."""
        return [
            ("Brokerage", self.brokerage),
            ("Securities Transaction Tax", self.securities_transaction_tax),
            ("Exchange Transaction Charge", self.exchange_transaction_charge),
            ("SEBI Turnover Fee", self.sebi_turnover_fee),
            ("Investor Protection Fund", self.investor_protection_fund),
            ("Stamp Duty", self.stamp_duty),
            ("GST", self.goods_and_services_tax),
            ("DP Charges", self.depository_participant_charge),
        ]
