"""Dated charge schedules loaded from YAML.

Rates change. A backtest over 2019 that applies 2026 rates is quietly wrong in
a direction that always flatters the strategy, because charges have generally
fallen. So schedules carry an effective date range and the registry picks the
one covering each trade date.

The YAML is validated on load by Pydantic: a typo in a rate fails at startup
rather than silently costing zero.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import Annotated, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from trading_agent.domain.enums import ProductType, Side

__all__ = [
    "BrokerageModel",
    "ChargeSchedule",
    "RoundingMode",
    "ScheduleNotFoundError",
    "ScheduleRegistry",
]

#: A levy rate expressed as a fraction of turnover (0.001 == 0.1%).
Rate = Annotated[Decimal, Field(ge=0, le=1)]


class ScheduleNotFoundError(LookupError):
    """No charge schedule covers the requested broker and date."""


class RoundingMode(StrEnum):
    PAISA = "paisa"
    NEAREST_RUPEE = "nearest_rupee"


class BrokerageModel(StrEnum):
    ZERO = "zero"
    PERCENT_CAPPED = "percent_capped"
    FLAT = "flat"


class _Strict(BaseModel):
    """Reject unknown keys: a misspelled rate must not be silently ignored."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class BrokerageRule(_Strict):
    model: BrokerageModel
    rate: Rate = Decimal(0)
    cap_per_order: Decimal | None = None
    flat_per_order: Decimal | None = None

    @model_validator(mode="after")
    def _check_required_fields(self) -> Self:
        if self.model is BrokerageModel.PERCENT_CAPPED and self.cap_per_order is None:
            raise ValueError("percent_capped brokerage requires cap_per_order")
        if self.model is BrokerageModel.FLAT and self.flat_per_order is None:
            raise ValueError("flat brokerage requires flat_per_order")
        return self


class TransactionTaxRule(_Strict):
    """STT. Asymmetric by design: delivery is charged both sides, intraday
    only on the sell."""

    buy_rate: Rate
    sell_rate: Rate


class StampDutyRule(_Strict):
    rate: Rate
    side: Side = Side.BUY


class DepositoryChargeRule(_Strict):
    """DP debit. Flat per scrip on delivery sells, independent of quantity."""

    enabled: bool
    side: Side = Side.SELL
    amount_per_scrip: Decimal = Decimal(0)
    gst_applicable: bool = True


class ProductSchedule(_Strict):
    brokerage: BrokerageRule
    securities_transaction_tax: TransactionTaxRule
    stamp_duty: StampDutyRule
    depository_participant_charge: DepositoryChargeRule


class RateRule(_Strict):
    rate: Rate


_KNOWN_ROUNDING_KEYS: frozenset[str] = frozenset(
    {
        "securities_transaction_tax",
        "stamp_duty",
        "exchange_transaction_charge",
        "sebi_turnover_fee",
        "investor_protection_fund",
        "brokerage",
        "goods_and_services_tax",
        "depository_participant_charge",
        "total",
    }
)

_KNOWN_GST_KEYS: frozenset[str] = frozenset(
    {
        "brokerage",
        "exchange_transaction_charge",
        "sebi_turnover_fee",
        "investor_protection_fund",
        "depository_participant_charge",
    }
)


class GstRule(_Strict):
    rate: Rate
    applies_to: frozenset[str]

    @model_validator(mode="after")
    def _reject_invalid_keys(self) -> Self:
        unknown = self.applies_to - _KNOWN_GST_KEYS
        if unknown:
            raise ValueError(
                f"unrecognized GST components: {sorted(unknown)}. "
                "Check applies_to for a misspelled charge name."
            )
        forbidden = {"securities_transaction_tax", "stamp_duty", "goods_and_services_tax"}
        overlap = self.applies_to & forbidden
        if overlap:
            raise ValueError(
                f"GST cannot apply to statutory taxes: {sorted(overlap)}. "
                "GST is levied on brokerage and exchange/regulator fees only."
            )
        return self


class CommonRates(_Strict):
    exchange_transaction_charge: RateRule
    sebi_turnover_fee: RateRule
    investor_protection_fund: RateRule
    goods_and_services_tax: GstRule


class ChargeSchedule(_Strict):
    """One broker's charges for one segment over one date range."""

    schema_version: int
    broker: str
    segment: str
    exchange: str
    currency: str
    effective_from: date
    effective_to: date | None
    source_url: str
    verified_against_contract_note: bool
    products: dict[ProductType, ProductSchedule]
    common: CommonRates
    rounding: dict[str, RoundingMode]

    @model_validator(mode="after")
    def _check_schedule(self) -> Self:
        if self.effective_to is not None and self.effective_to < self.effective_from:
            raise ValueError(
                f"effective_to {self.effective_to} precedes effective_from {self.effective_from}"
            )
        unknown_rounding = set(self.rounding.keys()) - _KNOWN_ROUNDING_KEYS
        if unknown_rounding:
            raise ValueError(
                f"unrecognized rounding keys: {sorted(unknown_rounding)}. "
                "A misspelled key would silently fall back to paisa rounding."
            )
        return self

    def covers(self, on: date) -> bool:
        if on < self.effective_from:
            return False
        return self.effective_to is None or on <= self.effective_to

    def rounding_for(self, component: str) -> RoundingMode:
        return self.rounding.get(component, RoundingMode.PAISA)

    def product(self, product: ProductType) -> ProductSchedule:
        try:
            return self.products[product]
        except KeyError as exc:
            raise ScheduleNotFoundError(
                f"schedule {self.broker}/{self.segment} defines no rates for {product}"
            ) from exc

    @classmethod
    def from_yaml(cls, path: Path) -> ChargeSchedule:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError(f"{path} does not contain a YAML mapping")
        try:
            return cls.model_validate(raw)
        except Exception as exc:
            raise ValueError(f"invalid charge schedule {path.name}: {exc}") from exc


class ScheduleRegistry:
    """All known schedules, queryable by broker and date."""

    __slots__ = ("_schedules",)

    def __init__(self, schedules: list[ChargeSchedule]) -> None:
        if not schedules:
            raise ValueError("ScheduleRegistry requires at least one schedule")
        self._schedules = sorted(schedules, key=lambda s: s.effective_from, reverse=True)
        self._check_no_overlap()

    def _check_no_overlap(self) -> None:
        """Two schedules covering one date would make charges depend on load
        order, which is a bug that only shows up as an unreproducible backtest."""
        by_broker: dict[tuple[str, str], list[ChargeSchedule]] = {}
        for schedule in self._schedules:
            by_broker.setdefault((schedule.broker, schedule.segment), []).append(schedule)

        for (broker, segment), group in by_broker.items():
            ordered = sorted(group, key=lambda s: s.effective_from)
            for earlier, later in pairwise(ordered):
                if earlier.effective_to is None or earlier.effective_to >= later.effective_from:
                    raise ValueError(
                        f"overlapping charge schedules for {broker}/{segment}: "
                        f"{earlier.effective_from}..{earlier.effective_to} overlaps "
                        f"{later.effective_from}..{later.effective_to}"
                    )

    @classmethod
    def from_directory(cls, directory: Path, *, broker: str | None = None) -> ScheduleRegistry:
        paths = sorted(directory.glob("*.yaml"))
        if not paths:
            raise ScheduleNotFoundError(f"no charge schedules found in {directory}")
        schedules = [ChargeSchedule.from_yaml(p) for p in paths]
        if broker is not None:
            schedules = [s for s in schedules if s.broker == broker]
            if not schedules:
                raise ScheduleNotFoundError(f"no charge schedules for broker {broker!r}")
        return cls(schedules)

    def for_date(self, on: date, *, broker: str, segment: str = "equity") -> ChargeSchedule:
        for schedule in self._schedules:
            if schedule.broker == broker and schedule.segment == segment and schedule.covers(on):
                return schedule
        raise ScheduleNotFoundError(
            f"no {broker}/{segment} charge schedule covers {on.isoformat()}. "
            f"Add a dated schedule to config/charges/ rather than widening an existing one."
        )

    def __len__(self) -> int:
        return len(self._schedules)

    def __iter__(self) -> object:
        return iter(self._schedules)
