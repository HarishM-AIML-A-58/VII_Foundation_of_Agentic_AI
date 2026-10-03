"""Position sizing.

**Size from risk, not from conviction.**

The quantity is chosen so that being stopped out costs at most a fixed
fraction of equity. Confidence does not enter the calculation. Sizing up on
high confidence feels rational and is how accounts die: conviction is highest
precisely when a position is most crowded, and a run of confident losses
compounds far faster than a run of ordinary ones.

Charges are inside the risk figure, not outside it. A stop-out costs the price
move **plus** the round trip's charges, and a sizing model that ignores them
systematically risks more than it claims to.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from trading_agent.costs.calculator import ChargeCalculator
from trading_agent.domain.enums import ProductType, Side
from trading_agent.domain.money import Rupees
from trading_agent.domain.signal import TradeSignal
from trading_agent.observability import get_logger

__all__ = ["SizingResult", "size_position"]

log = get_logger(__name__)

#: Sizing solves by stepping down from an upper bound. Bounded so a pathological
#: input cannot spin: 200 iterations is far beyond any realistic correction.
_MAX_REFINEMENTS = 200


@dataclass(frozen=True, slots=True)
class SizingResult:
    quantity: int
    #: Worst-case loss if the stop fills exactly, charges included.
    risk_amount: Rupees
    #: Cash required to open, charges included.
    outlay: Rupees
    reason: str = ""

    @property
    def is_tradeable(self) -> bool:
        return self.quantity > 0


def size_position(
    signal: TradeSignal,
    *,
    equity: Rupees,
    available_cash: Rupees,
    max_risk_fraction: Decimal,
    calculator: ChargeCalculator,
    on: date | None = None,
    product: ProductType = ProductType.CNC,
    max_position_weight: Decimal | None = None,
) -> SizingResult:
    """Shares to trade so a stop-out loses at most ``max_risk_fraction`` of equity.

    Three constraints bind, and the smallest wins:

    1. **Risk budget.** ``(entry - stop) * qty + round-trip charges`` must not
       exceed ``equity * max_risk_fraction``.
    2. **Position weight.** When configured, ``entry * qty + entry charges``
       must not exceed ``equity * max_position_weight``.
    3. **Cash.** ``entry * qty + entry charges`` must not exceed available cash.

    Charges are non-linear in quantity — brokerage caps per order, DP debits
    are flat per scrip — so there is no closed form worth trusting. The
    analytic estimate is used as an upper bound and refined downward until it
    actually fits.
    """
    if max_risk_fraction <= 0 or max_risk_fraction > 1:
        raise ValueError(f"max_risk_fraction must lie in (0, 1], got {max_risk_fraction}")
    if max_position_weight is not None and not 0 < max_position_weight <= 1:
        raise ValueError(f"max_position_weight must lie in (0, 1], got {max_position_weight}")
    if not signal.action.is_actionable:
        return SizingResult(0, Rupees.zero(), Rupees.zero(), "signal is not actionable")

    risk_per_share = signal.risk_per_share
    if risk_per_share <= 0:
        return SizingResult(0, Rupees.zero(), Rupees.zero(), "stop gives no risk distance")

    session = on or signal.session_date
    risk_budget = Rupees(equity.value * max_risk_fraction)

    max_allowed_outlay = available_cash
    if max_position_weight is not None:
        max_weight_budget = Rupees(equity.value * max_position_weight)
        max_allowed_outlay = min(available_cash, max_weight_budget)

    # Upper bound ignoring charges. Charges only ever reduce it, so starting
    # here and stepping down converges from the correct side.
    quantity = int(risk_budget.value / risk_per_share)
    if max_allowed_outlay.value > 0 and signal.entry > 0:
        quantity = min(quantity, int(max_allowed_outlay.value / signal.entry))

    if quantity <= 0:
        return SizingResult(
            0,
            Rupees.zero(),
            Rupees.zero(),
            f"risk budget {risk_budget} is smaller than one share's risk "
            f"({risk_per_share} per share)",
        )

    side = Side.BUY if signal.action.value == "BUY" else Side.SELL

    for _ in range(_MAX_REFINEMENTS):
        if quantity <= 0:
            break

        entry_charges = calculator.for_notional(
            price=signal.entry, quantity=quantity, side=side, product=product, on=session
        )
        exit_charges = calculator.for_notional(
            price=signal.stop_loss,
            quantity=quantity,
            side=side.opposite,
            product=product,
            on=session,
        )
        round_trip = entry_charges.total + exit_charges.total

        total_risk = Rupees(risk_per_share * quantity) + round_trip
        outlay = Rupees(signal.entry * quantity) + entry_charges.total

        if total_risk <= risk_budget and outlay <= max_allowed_outlay:
            return SizingResult(quantity, total_risk, outlay)

        # Step down proportionally to the binding overshoot rather than by one:
        # a single-share decrement takes thousands of iterations on a large
        # budget, and each iteration prices two charge calculations.
        over_risk = total_risk.value / risk_budget.value if risk_budget.value else Decimal(2)
        over_outlay = (
            outlay.value / max_allowed_outlay.value if max_allowed_outlay.value else Decimal(2)
        )
        worst = max(over_risk, over_outlay)
        reduced = int(Decimal(quantity) / worst) if worst > 1 else quantity - 1
        quantity = min(reduced, quantity - 1)

    reason = (
        f"cannot fit a position: risk budget {risk_budget}, max outlay {max_allowed_outlay}, "
        f"risk per share {risk_per_share}"
    )
    log.info("sizing_rejected", symbol=signal.symbol, reason=reason)
    return SizingResult(0, Rupees.zero(), Rupees.zero(), reason)
