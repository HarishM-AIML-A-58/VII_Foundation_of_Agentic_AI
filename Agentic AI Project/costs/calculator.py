"""The Indian equity charge calculator.

This is the module everything else is measured against. A strategy that looks
profitable is only profitable if this file is right, so it is deliberately
explicit: one method per levy, no clever shared arithmetic, and the rounding
applied exactly where a contract note applies it.

Coverage requirement for this module is 100% (see ``pyproject.toml``).
"""

from __future__ import annotations

import warnings
from datetime import date
from decimal import Decimal
from pathlib import Path

from trading_agent.costs.breakdown import ChargeBreakdown
from trading_agent.costs.schedule import (
    BrokerageModel,
    ChargeSchedule,
    RoundingMode,
    ScheduleRegistry,
)
from trading_agent.domain.enums import ProductType, Side
from trading_agent.domain.money import Rupees
from trading_agent.domain.trade import Execution

__all__ = ["ChargeCalculator", "UnverifiedScheduleWarning"]


class UnverifiedScheduleWarning(UserWarning):
    """Raised when charges are computed from a schedule that has never been
    checked against a real contract note."""


def _round(amount: Rupees, mode: RoundingMode) -> Rupees:
    if mode is RoundingMode.NEAREST_RUPEE:
        return amount.round_to_rupee()
    return amount


class ChargeCalculator:
    """Computes charges for executions using date-appropriate schedules."""

    __slots__ = ("_broker", "_registry", "_segment", "_warned")

    def __init__(
        self,
        registry: ScheduleRegistry,
        *,
        broker: str,
        segment: str = "equity",
    ) -> None:
        self._registry = registry
        self._broker = broker
        self._segment = segment
        self._warned: set[date] = set()

    @classmethod
    def from_directory(
        cls, directory: Path | str, *, broker: str, segment: str = "equity"
    ) -> ChargeCalculator:
        registry = ScheduleRegistry.from_directory(Path(directory))
        return cls(registry, broker=broker, segment=segment)

    # ----------------------------------------------------------------- API

    def for_execution(
        self, execution: Execution, *, apply_depository_charge: bool = True
    ) -> ChargeBreakdown:
        """Charges on a single fill.

        ``apply_depository_charge`` exists because the DP debit is levied once
        per scrip per day, not per fill. When a delivery sell is filled in
        three parts, the caller passes ``True`` for the first and ``False``
        for the rest. :meth:`for_executions` does this bookkeeping for you.
        """
        schedule = self._schedule_for(execution.executed_at.date())
        return self._compute(
            schedule=schedule,
            turnover=execution.turnover,
            side=execution.side,
            product=execution.product,
            apply_depository_charge=apply_depository_charge,
        )

    def for_executions(self, executions: list[Execution]) -> ChargeBreakdown:
        """Charges on many fills, handling per-scrip-per-day DP debits.

        Fills are grouped by (symbol, session date) so the DP charge lands
        exactly once per group, matching how the depository actually bills.
        Brokerage is capped per executed order (broker_order_id), not per fill.
        """
        total = ChargeBreakdown(turnover=Rupees.zero())
        seen_dp_groups: set[tuple[str, date]] = set()

        # Group fills by broker_order_id (or individual fill if no broker_order_id)
        order_fills: dict[str, list[Execution]] = {}
        for execution in executions:
            key = execution.broker_order_id or f"fill:{execution.execution_id}"
            order_fills.setdefault(key, []).append(execution)

        for execution in sorted(executions, key=lambda e: e.executed_at):
            group = (execution.symbol, execution.executed_at.date())
            first_in_group = group not in seen_dp_groups
            breakdown = self.for_execution(execution, apply_depository_charge=first_in_group)
            if first_in_group and not breakdown.depository_participant_charge.is_zero():
                seen_dp_groups.add(group)
            total = total + breakdown

        # Adjust brokerage if multiple fills shared a broker_order_id with a cap
        total_brokerage = Rupees.zero()
        for fills in order_fills.values():
            if len(fills) > 1 and fills[0].broker_order_id:
                schedule = self._schedule_for(fills[0].executed_at.date())
                rules = schedule.product(fills[0].product)
                order_turnover = sum((f.turnover for f in fills), Rupees.zero())
                order_brokerage = _round(
                    self._brokerage(order_turnover, rules.brokerage),
                    schedule.rounding_for("brokerage"),
                )
                total_brokerage = total_brokerage + order_brokerage
            else:
                for f in fills:
                    schedule = self._schedule_for(f.executed_at.date())
                    rules = schedule.product(f.product)
                    f_brokerage = _round(
                        self._brokerage(f.turnover, rules.brokerage),
                        schedule.rounding_for("brokerage"),
                    )
                    total_brokerage = total_brokerage + f_brokerage

        if executions and total.brokerage != total_brokerage:
            # Only reachable with fills: an empty list sums to zero brokerage both ways.
            brokerage_diff = total.brokerage - total_brokerage
            first = self._schedule_for(executions[0].executed_at.date())
            gst_adjustment = _round(
                brokerage_diff * first.common.goods_and_services_tax.rate,
                first.rounding_for("goods_and_services_tax"),
            )
            total = ChargeBreakdown(
                turnover=total.turnover,
                brokerage=total_brokerage,
                securities_transaction_tax=total.securities_transaction_tax,
                exchange_transaction_charge=total.exchange_transaction_charge,
                sebi_turnover_fee=total.sebi_turnover_fee,
                investor_protection_fund=total.investor_protection_fund,
                stamp_duty=total.stamp_duty,
                goods_and_services_tax=total.goods_and_services_tax - gst_adjustment,
                depository_participant_charge=total.depository_participant_charge,
            )

        return total

    def for_notional(
        self,
        *,
        price: Decimal,
        quantity: int,
        side: Side,
        product: ProductType,
        on: date,
        apply_depository_charge: bool = True,
    ) -> ChargeBreakdown:
        """Charges on a hypothetical trade. Used by the backtester and sizing,
        which reason about intended trades rather than realised fills."""
        if quantity <= 0:
            raise ValueError(f"quantity must be positive, got {quantity}")
        if price <= 0:
            raise ValueError(f"price must be positive, got {price}")
        return self._compute(
            schedule=self._schedule_for(on),
            turnover=Rupees(price * quantity),
            side=side,
            product=product,
            apply_depository_charge=apply_depository_charge,
        )

    def round_trip(
        self,
        *,
        entry_price: Decimal,
        exit_price: Decimal,
        quantity: int,
        side: Side,
        product: ProductType,
        entry_date: date,
        exit_date: date | None = None,
    ) -> ChargeBreakdown:
        """Total charges to open and close a position.

        This is the number that decides whether a strategy edge survives. For
        delivery round trips it includes STT on both legs plus the DP debit,
        which together dominate the cost of small positions.
        """
        exit_on = exit_date if exit_date is not None else entry_date
        entry = self.for_notional(
            price=entry_price,
            quantity=quantity,
            side=side,
            product=product,
            on=entry_date,
        )
        exit_leg = self.for_notional(
            price=exit_price,
            quantity=quantity,
            side=side.opposite,
            product=product,
            on=exit_on,
        )
        return entry + exit_leg

    # ------------------------------------------------------------ internals

    def _schedule_for(self, on: date) -> ChargeSchedule:
        schedule = self._registry.for_date(on, broker=self._broker, segment=self._segment)
        if not schedule.verified_against_contract_note and on not in self._warned:
            self._warned.add(on)
            warnings.warn(
                f"charge schedule {schedule.broker}/{schedule.segment} effective "
                f"{schedule.effective_from} has NOT been verified against a real contract "
                f"note. Run `trading-agent costs verify` before trusting these numbers.",
                UnverifiedScheduleWarning,
                stacklevel=3,
            )
        return schedule

    def _compute(
        self,
        *,
        schedule: ChargeSchedule,
        turnover: Rupees,
        side: Side,
        product: ProductType,
        apply_depository_charge: bool,
    ) -> ChargeBreakdown:
        rules = schedule.product(product)
        common = schedule.common

        brokerage = _round(
            self._brokerage(turnover, rules.brokerage),
            schedule.rounding_for("brokerage"),
        )

        stt_rate = (
            rules.securities_transaction_tax.buy_rate
            if side is Side.BUY
            else rules.securities_transaction_tax.sell_rate
        )
        securities_transaction_tax = _round(
            turnover * stt_rate, schedule.rounding_for("securities_transaction_tax")
        )

        exchange_transaction_charge = _round(
            turnover * common.exchange_transaction_charge.rate,
            schedule.rounding_for("exchange_transaction_charge"),
        )
        sebi_turnover_fee = _round(
            turnover * common.sebi_turnover_fee.rate,
            schedule.rounding_for("sebi_turnover_fee"),
        )
        investor_protection_fund = _round(
            turnover * common.investor_protection_fund.rate,
            schedule.rounding_for("investor_protection_fund"),
        )

        stamp_duty = Rupees.zero()
        if side is rules.stamp_duty.side:
            stamp_duty = _round(
                turnover * rules.stamp_duty.rate, schedule.rounding_for("stamp_duty")
            )

        depository_participant_charge = Rupees.zero()
        dp_rule = rules.depository_participant_charge
        if apply_depository_charge and dp_rule.enabled and side is dp_rule.side:
            depository_participant_charge = Rupees(dp_rule.amount_per_scrip)

        # GST is levied on the SERVICE components named in the schedule, never
        # on STT or stamp duty. The DP charge is separately flagged because
        # depositories bill it inclusive or exclusive depending on the broker.
        gst_base = Rupees.zero()
        taxable = {
            "brokerage": brokerage,
            "exchange_transaction_charge": exchange_transaction_charge,
            "sebi_turnover_fee": sebi_turnover_fee,
            "investor_protection_fund": investor_protection_fund,
        }
        for component, amount in taxable.items():
            if component in common.goods_and_services_tax.applies_to:
                gst_base = gst_base + amount
        if dp_rule.gst_applicable:
            gst_base = gst_base + depository_participant_charge

        goods_and_services_tax = _round(
            gst_base * common.goods_and_services_tax.rate,
            schedule.rounding_for("goods_and_services_tax"),
        )

        return ChargeBreakdown(
            turnover=turnover,
            brokerage=brokerage,
            securities_transaction_tax=securities_transaction_tax,
            exchange_transaction_charge=exchange_transaction_charge,
            sebi_turnover_fee=sebi_turnover_fee,
            investor_protection_fund=investor_protection_fund,
            stamp_duty=stamp_duty,
            goods_and_services_tax=goods_and_services_tax,
            depository_participant_charge=depository_participant_charge,
        )

    @staticmethod
    def _brokerage(turnover: Rupees, rule: object) -> Rupees:
        from trading_agent.costs.schedule import BrokerageRule  # local: avoid cycle

        assert isinstance(rule, BrokerageRule)  # noqa: S101 -- narrowing for mypy

        if rule.model is BrokerageModel.ZERO:
            return Rupees.zero()
        if rule.model is BrokerageModel.FLAT:
            assert rule.flat_per_order is not None  # noqa: S101 -- validated on load
            return Rupees(rule.flat_per_order)
        # PERCENT_CAPPED: the lower of a percentage of turnover and a flat cap.
        assert rule.cap_per_order is not None  # noqa: S101 -- validated on load
        percentage = turnover * rule.rate
        cap = Rupees(rule.cap_per_order)
        return min(percentage, cap)
