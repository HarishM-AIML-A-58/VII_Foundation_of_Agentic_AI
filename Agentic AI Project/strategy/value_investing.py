"""Value investing: buy quality businesses below intrinsic value, sell at it.

Four rulesets, each taken from its investor's published method rather than
from their holdings:

- ``graham``: The Intelligent Investor, defensive-investor criteria. Low debt,
  unbroken earnings, P/E < 15 and P/B < 1.5 (or below the Graham number).
- ``buffett``: high return on capital, low debt, earnings backed by cash, no
  promoter pledge, and a price well below earnings power value.
- ``greenblatt``: The Little Book That Beats the Market. Rank by earnings yield
  (EBIT / EV) plus return on capital; skip Altman-distressed names.
- ``lynch``: One Up On Wall Street. Steady 10-30% EPS growth bought at PEG <= 1.

Each ruleset buys and sells on its own investor's terms:

- ``graham`` buys on his multiples test and below intrinsic value, and sells
  at intrinsic value.
- ``buffett`` buys a high-return business at or below its own 10-year median
  P/E -- "a wonderful company at a fair price" -- and sells only when the
  business stops qualifying. Not earnings power value: it credits no growth,
  and the liquid200 names that pass his quality screen trade at a median 8x
  it, so no price ever qualified.
- ``greenblatt`` buys the top of the Magic Formula ranking and turns each name
  over after a year, as the book prescribes. No price target.
- ``lynch`` buys at PEG <= 1 and sells once PEG passes 2, when the multiple
  has run well ahead of the growth.

Shared rules:

- **Sell** when a new filing breaks the quality filter, or when the stock
  leaves the universe.
- **No price stop.** A value position is not sold because the price fell
  further; that is when the margin of safety is widest.
- An optional ``min_drawdown`` (off in every preset) requires the price to be
  that far below its 52-week high. None of the four investors used one, and
  with it the books sat in cash for most of the backtest.

Names are held until a sell rule fires, not swapped out when their rank drops,
so the held set lives on the instance. It resets on the first rebalance of a
run (``previous is None``), so walk-forward windows start clean.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from enum import StrEnum

import pandas as pd

from trading_agent.features.value import (
    ALTMAN_DISTRESS,
    SymbolFundamentals,
    ValueSnapshot,
    value_snapshot,
)

__all__ = [
    "PRESETS",
    "RankBy",
    "ValueBasis",
    "ValueDecision",
    "ValueInvestingStrategy",
    "ValueRules",
]


class RankBy(StrEnum):
    MARGIN = "margin"
    MAGIC_FORMULA = "magic_formula"
    PEG = "peg"


class ValueBasis(StrEnum):
    """Which estimate the buy margin and the take-profit are measured against."""

    INTRINSIC = "intrinsic"
    EARNINGS_POWER = "earnings_power"
    #: Current EPS at the stock's own 10-year median P/E: a fair price by the
    #: company's own history, which credits the growth it has actually had.
    OWN_PE = "own_pe"


@dataclass(frozen=True, slots=True)
class ValueRules:
    """One investor's published method, as checks on a :class:`ValueSnapshot`.

    ``None`` on a threshold means the ruleset does not check it.
    """

    name: str
    label: str
    #: Required discount to the basis value; ``None`` when the ruleset values
    #: a stock some other way (a multiples test, a ranking, PEG).
    min_margin: float | None
    rank_by: RankBy = RankBy.MARGIN
    basis: ValueBasis = ValueBasis.INTRINSIC
    max_debt_to_equity: float | None = None
    min_roce: float | None = None
    min_cfo_to_profit: float | None = None
    max_pledge: float | None = None
    min_positive_eps_years: int | None = None
    min_positive_cfo_years: int | None = None
    min_eps_cagr: float | None = None
    max_eps_cagr: float | None = None
    max_peg: float | None = None
    graham_price_test: bool = False
    reject_distress: bool = False
    min_drawdown: float = 0.0
    #: Sell when price reaches the basis value.
    sell_at_value: bool = True
    #: Sell once PEG rises above this.
    sell_above_peg: float | None = None
    #: Sell after holding this many months, whatever the price.
    hold_months: int | None = None

    def basis_value(self, snap: ValueSnapshot) -> float | None:
        if self.basis is ValueBasis.EARNINGS_POWER:
            return snap.earnings_power_value
        if self.basis is ValueBasis.OWN_PE:
            return snap.own_pe_value
        return snap.intrinsic_value

    def quality_checks(self, snap: ValueSnapshot) -> list[tuple[str, bool]]:
        """Each quality rule this ruleset applies, as (rule, passed).

        Missing data fails a rule, except promoter pledge: the page often has no
        pledge row, and that means none was reported, not that it is unknown.
        """
        checks: list[tuple[str, bool]] = [
            ("not a lender (statements comparable)", not snap.financial)
        ]
        if self.max_debt_to_equity is not None:
            de = snap.debt_to_equity
            checks.append(
                (
                    f"debt/equity below {self.max_debt_to_equity}",
                    de is not None and de < self.max_debt_to_equity,
                )
            )
        if self.min_roce is not None:
            roce = snap.roce_median_5y
            checks.append(
                (
                    f"5y median ROCE at least {self.min_roce:.0f}%",
                    roce is not None and roce >= self.min_roce,
                )
            )
        if self.min_cfo_to_profit is not None:
            ratio = snap.cfo_to_net_profit_5y
            checks.append(
                (
                    f"5y operating cash flow at least {self.min_cfo_to_profit:.0%} of profit",
                    ratio is not None and ratio >= self.min_cfo_to_profit,
                )
            )
        if self.max_pledge is not None:
            pledge = snap.promoter_pledge
            checks.append(
                (
                    f"promoter pledge below {self.max_pledge:.0f}%",
                    pledge is None or pledge < self.max_pledge,
                )
            )
        if self.min_positive_eps_years is not None:
            checks.append(
                (
                    f"positive EPS in at least {self.min_positive_eps_years} of 5 years",
                    snap.positive_eps_years_5y >= self.min_positive_eps_years,
                )
            )
        if self.min_positive_cfo_years is not None:
            checks.append(
                (
                    f"positive operating cash flow in {self.min_positive_cfo_years}+ of 5 years",
                    snap.positive_cfo_years_5y >= self.min_positive_cfo_years,
                )
            )
        if self.min_eps_cagr is not None or self.max_eps_cagr is not None:
            growth = snap.eps_cagr_3y
            low = self.min_eps_cagr if self.min_eps_cagr is not None else float("-inf")
            high = self.max_eps_cagr if self.max_eps_cagr is not None else float("inf")
            checks.append(
                (
                    f"3y EPS growth between {low:.0%} and {high:.0%}",
                    growth is not None and low <= growth <= high,
                )
            )
        if self.reject_distress:
            z = snap.altman_z
            checks.append(
                (f"Altman Z not below {ALTMAN_DISTRESS}", z is None or z >= ALTMAN_DISTRESS)
            )
        return checks

    def price_checks(self, snap: ValueSnapshot) -> list[tuple[str, bool]]:
        """Each entry-price rule, as (rule, passed)."""
        basis = {
            ValueBasis.EARNINGS_POWER: "earnings power value",
            ValueBasis.OWN_PE: "its own 10-year median P/E",
            ValueBasis.INTRINSIC: "intrinsic value",
        }[self.basis]
        checks: list[tuple[str, bool]] = []
        if self.min_margin is not None:
            margin = snap.margin_against(self.basis_value(snap))
            checks.append(
                (
                    f"price at least {self.min_margin:.0%} below {basis}"
                    if self.min_margin > 0
                    else f"price below {basis}",
                    margin is not None
                    and margin >= self.min_margin
                    and (margin > 0 or not self.sell_at_value),
                )
            )
        if self.min_drawdown > 0:
            checks.append(
                (
                    f"price at least {self.min_drawdown:.0%} below 52-week high",
                    snap.drawdown >= self.min_drawdown,
                )
            )
        if self.graham_price_test:
            cheap_multiples = (
                snap.pe is not None and snap.pe < 15 and snap.pb is not None and snap.pb < 1.5  # noqa: PLR2004 -- Graham's own numbers
            )
            below_number = snap.graham_number is not None and snap.price < snap.graham_number
            checks.append(
                (
                    "P/E < 15 and P/B < 1.5, or price below Graham number",
                    cheap_multiples or below_number,
                )
            )
        if self.max_peg is not None:
            checks.append(
                (
                    f"PEG at or below {self.max_peg}",
                    snap.peg is not None and snap.peg <= self.max_peg,
                )
            )
        return checks

    def quality_failures(self, snap: ValueSnapshot) -> list[str]:
        return [rule for rule, passed in self.quality_checks(snap) if not passed]

    def price_failures(self, snap: ValueSnapshot) -> list[str]:
        return [rule for rule, passed in self.price_checks(snap) if not passed]

    def reached_value(self, snap: ValueSnapshot) -> bool:
        value = self.basis_value(snap)
        return value is not None and snap.price >= value

    def exit_reason(self, snap: ValueSnapshot, *, months_held: int | None) -> str | None:
        """The price- or time-based sell rule that fires, if any."""
        if self.sell_at_value and self.reached_value(snap):
            return "reached value"
        peg_cap = self.sell_above_peg
        if peg_cap is not None and snap.peg is not None and snap.peg > peg_cap:
            return f"PEG above {peg_cap:g}"
        hold = self.hold_months
        if hold is not None and months_held is not None and months_held >= hold:
            return f"held {hold} months"
        return None


PRESETS: dict[str, ValueRules] = {
    "graham": ValueRules(
        name="graham",
        label="Benjamin Graham, defensive investor",
        # His margin of safety is the multiples test below; price must also
        # sit under intrinsic value so the sell rule does not fire at once.
        min_margin=0.0,
        max_debt_to_equity=0.5,
        min_positive_eps_years=5,
        min_positive_cfo_years=4,
        graham_price_test=True,
    ),
    "buffett": ValueRules(
        name="buffett",
        label="Warren Buffett, quality at a fair price",
        min_margin=0.0,
        basis=ValueBasis.OWN_PE,
        sell_at_value=False,
        min_roce=18.0,
        max_debt_to_equity=0.5,
        min_cfo_to_profit=0.8,
        max_pledge=5.0,
    ),
    "greenblatt": ValueRules(
        name="greenblatt",
        label="Joel Greenblatt, Magic Formula",
        min_margin=None,
        rank_by=RankBy.MAGIC_FORMULA,
        sell_at_value=False,
        hold_months=12,
        min_roce=15.0,
        reject_distress=True,
    ),
    "lynch": ValueRules(
        name="lynch",
        label="Peter Lynch, growth at a reasonable price",
        min_margin=None,
        rank_by=RankBy.PEG,
        sell_at_value=False,
        sell_above_peg=2.0,
        max_debt_to_equity=1.0,
        min_eps_cagr=0.10,
        max_eps_cagr=0.30,
        max_peg=1.0,
    ),
}


@dataclass(frozen=True, slots=True)
class ValueDecision:
    """BUY / HOLD / SELL / PASS for one stock, and why."""

    symbol: str
    action: str
    reasons: tuple[str, ...]
    snapshot: ValueSnapshot | None


def _rank_scores(rules: ValueRules, snaps: list[ValueSnapshot]) -> dict[str, float]:
    """Higher is better."""
    if rules.rank_by is RankBy.PEG:
        return {s.symbol: -(s.peg if s.peg is not None else 99.0) for s in snaps}
    if rules.rank_by is RankBy.MAGIC_FORMULA:
        by_yield = sorted(snaps, key=lambda s: -(s.ebit_to_ev or float("-inf")))
        by_roce = sorted(snaps, key=lambda s: -(s.roce_median_5y or float("-inf")))
        yield_rank = {s.symbol: i for i, s in enumerate(by_yield)}
        roce_rank = {s.symbol: i for i, s in enumerate(by_roce)}
        return {s.symbol: -float(yield_rank[s.symbol] + roce_rank[s.symbol]) for s in snaps}
    return {s.symbol: s.margin_against(rules.basis_value(s)) or float("-inf") for s in snaps}


@dataclass(slots=True)
class ValueInvestingStrategy:
    """A concentrated value book: up to ``size`` names at ``1/size`` each.

    Satisfies :class:`~trading_agent.strategy.protocols.PortfolioStrategy`.
    """

    rules: ValueRules
    fundamentals: Mapping[str, SymbolFundamentals]
    size: int = 10
    max_sector_exposure: Decimal = Decimal("0.30")
    is_eligible: Callable[[str, date], bool] | None = None
    #: (session, symbol, reason) for every sell decided, so a report can say how
    #: many exits were at fair value and how many were broken theses.
    exits: list[tuple[date, str, str]] = field(default_factory=list)
    #: (session, names held) after every rebalance: how invested the book was.
    holdings_log: list[tuple[date, tuple[str, ...]]] = field(default_factory=list)
    _held: set[str] = field(default_factory=set)
    #: When each held name was bought, for ``hold_months``. Unknown after
    #: :meth:`resume`, and an unknown date never triggers a time-based sale.
    _entered: dict[str, date] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ValueError(f"size must be at least 1, got {self.size}")

    @property
    def name(self) -> str:
        return f"value_{self.rules.name}_{self.size}"

    @property
    def held(self) -> frozenset[str]:
        return frozenset(self._held)

    def is_rebalance_date(self, on: date, *, previous: date | None) -> bool:
        if previous is None:
            self._held = set()
            self._entered = {}
            self.exits = []
            self.holdings_log = []
            return True
        return (on.year, on.month) != (previous.year, previous.month)

    def resume(self, held: set[str] | frozenset[str]) -> None:
        """Continue from a known book, e.g. a backtest's final holdings."""
        self._held = set(held)
        self._entered = {}

    def _eligible(self, symbol: str, on: date) -> bool:
        return self.is_eligible is None or self.is_eligible(symbol, on)

    def _snapshot(
        self, panel: Mapping[str, pd.DataFrame], symbol: str, on: date
    ) -> ValueSnapshot | None:
        funds = self.fundamentals.get(symbol)
        frame = panel.get(symbol)
        if funds is None or frame is None or frame.empty:
            return None
        return value_snapshot(funds, frame["close"], on)

    def decide(self, panel: Mapping[str, pd.DataFrame], *, on: date) -> list[ValueDecision]:
        """A decision for every held or eligible symbol, without changing state."""
        decisions: list[ValueDecision] = []
        for symbol in sorted(set(panel) | self._held):
            held = symbol in self._held
            if not held and not self._eligible(symbol, on):
                continue
            snap = self._snapshot(panel, symbol, on)
            if snap is None:
                action = "HOLD" if held else "PASS"
                decisions.append(ValueDecision(symbol, action, ("no data",), None))
                continue
            quality = self.rules.quality_failures(snap)
            if held:
                if not self._eligible(symbol, on):
                    decisions.append(ValueDecision(symbol, "SELL", ("left universe",), snap))
                elif exit_reason := self.rules.exit_reason(
                    snap, months_held=self._months_held(symbol, on)
                ):
                    decisions.append(ValueDecision(symbol, "SELL", (exit_reason,), snap))
                elif quality:
                    reasons = ("thesis broken", *quality)
                    decisions.append(ValueDecision(symbol, "SELL", reasons, snap))
                else:
                    decisions.append(ValueDecision(symbol, "HOLD", ("below value",), snap))
                continue
            failures = quality + self.rules.price_failures(snap)
            action = "PASS" if failures else "BUY"
            decisions.append(ValueDecision(symbol, action, tuple(failures), snap))
        return decisions

    def _months_held(self, symbol: str, on: date) -> int | None:
        entered = self._entered.get(symbol)
        if entered is None:
            return None
        return (on.year - entered.year) * 12 + on.month - entered.month

    def rank(self, panel: dict[str, pd.DataFrame], *, on: date) -> list[tuple[str, float]]:
        buys = [d.snapshot for d in self.decide(panel, on=on) if d.action == "BUY" and d.snapshot]
        scores = _rank_scores(self.rules, buys)
        return sorted(scores.items(), key=lambda pair: -pair[1])

    def target_weights(self, panel: dict[str, pd.DataFrame], *, on: date) -> dict[str, Decimal]:
        decisions = self.decide(panel, on=on)
        keep = {d.symbol for d in decisions if d.action == "HOLD"}
        self.exits.extend((on, d.symbol, d.reasons[0]) for d in decisions if d.action == "SELL")
        buys = [d.snapshot for d in decisions if d.action == "BUY" and d.snapshot]
        scores = _rank_scores(self.rules, buys)
        snaps = {d.symbol: d.snapshot for d in decisions}

        weight = Decimal(1) / Decimal(self.size)
        per_sector = max(1, int(self.max_sector_exposure / weight))
        sector_count: dict[str, int] = {}
        for symbol in keep:
            snap = snaps.get(symbol)
            if snap is not None and snap.sector:
                sector_count[snap.sector] = sector_count.get(snap.sector, 0) + 1

        chosen = set(keep)
        for symbol, _ in sorted(scores.items(), key=lambda pair: -pair[1]):
            if len(chosen) >= self.size:
                break
            snap = snaps.get(symbol)
            sector = snap.sector if snap is not None else None
            if sector and sector_count.get(sector, 0) >= per_sector:
                continue
            chosen.add(symbol)
            if sector:
                sector_count[sector] = sector_count.get(sector, 0) + 1

        self._entered = {s: self._entered.get(s, on) for s in chosen}
        self._held = chosen
        self.holdings_log.append((on, tuple(sorted(chosen))))
        return dict.fromkeys(sorted(chosen), weight)
