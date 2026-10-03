"""Portfolio-level limits.

The veto in :mod:`trading_agent.risk.veto` judges a trade on its own merits.
These limits judge it against everything already open — the same trade can be
fine in isolation and reckless as the sixth position in one sector.

All checks are arithmetic over current state. Nothing here consults a model.

Limits are checked **before** an order goes out and never used to liquidate.
A breach discovered after the fact is reported, not traded out of: forced
selling to satisfy a limit turns a paper problem into a realised loss plus
charges.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from trading_agent.domain.money import Rupees
from trading_agent.domain.position import Position
from trading_agent.domain.signal import TradeSignal
from trading_agent.observability import get_logger

__all__ = ["LimitBreach", "LimitCheck", "PortfolioLimits", "check_limits"]

log = get_logger(__name__)


class LimitBreach(StrEnum):
    MAX_POSITIONS = "max_positions"
    POSITION_WEIGHT = "position_weight"
    SECTOR_CONCENTRATION = "sector_concentration"
    DAILY_LOSS = "daily_loss"
    ALREADY_HOLDING = "already_holding"
    INSUFFICIENT_CASH = "insufficient_cash"
    ADV_PARTICIPATION = "adv_participation"


@dataclass(frozen=True, slots=True)
class PortfolioLimits:
    """Configured ceilings. Mirrors the ``risk`` block in settings."""

    max_positions: int = 10
    max_sector_exposure: Decimal = Decimal("0.30")
    max_daily_loss: Decimal = Decimal("0.05")

    #: Cap on a single position's notional as a fraction of equity.
    #:
    #: Risk-based sizing alone does NOT bound notional. Risking 2% of equity
    #: behind a 3.5% stop implies a position worth 57% of the account -- the
    #: arithmetic is correct and the concentration is reckless. A gap through
    #: the stop then loses far more than the 2% the sizing promised, because
    #: the promise assumed the stop would hold.
    max_position_weight: Decimal = Decimal("0.25")

    #: Maximum fraction of 20-day Average Daily Volume (ADV) a single order may consume.
    #: Orders larger than this face severe execution impact or circuit risk.
    max_adv_participation: Decimal = Decimal("0.02")

    def __post_init__(self) -> None:
        if self.max_positions < 1:
            raise ValueError(f"max_positions must be at least 1, got {self.max_positions}")
        if not 0 < self.max_sector_exposure <= 1:
            raise ValueError("max_sector_exposure must lie in (0, 1]")
        if not 0 < self.max_daily_loss <= 1:
            raise ValueError("max_daily_loss must lie in (0, 1]")
        if not 0 < self.max_position_weight <= 1:
            raise ValueError("max_position_weight must lie in (0, 1]")
        if not 0 < self.max_adv_participation <= 1:
            raise ValueError("max_adv_participation must lie in (0, 1]")


@dataclass(frozen=True, slots=True)
class LimitCheck:
    allowed: bool
    breach: LimitBreach | None = None
    detail: str = ""


def check_limits(
    signal: TradeSignal,
    *,
    limits: PortfolioLimits,
    open_positions: Sequence[Position],
    equity: Rupees,
    available_cash: Rupees,
    intended_value: Rupees,
    sectors: Mapping[str, str] | None = None,
    realised_pnl_today: Rupees | None = None,
    intended_quantity: int | None = None,
    average_daily_volume: int | None = None,
) -> LimitCheck:
    """Check a proposed trade against portfolio-level ceilings.

    Ordered cheapest-and-most-decisive first: the daily loss limit is a
    circuit breaker, so it settles the question before anything else is
    considered.
    """
    # -- 1. daily loss circuit breaker ------------------------------------
    # First because a day that has already gone badly is exactly when the
    # temptation to trade is highest and the judgement is worst.
    if realised_pnl_today is not None and equity.value > 0:
        loss_fraction = -realised_pnl_today.value / equity.value
        if loss_fraction >= limits.max_daily_loss:
            return LimitCheck(
                False,
                LimitBreach.DAILY_LOSS,
                f"down {loss_fraction:.1%} today, at or beyond the "
                f"{limits.max_daily_loss:.1%} daily limit; no new positions",
            )

    # -- 2. already holding ------------------------------------------------
    held = {p.symbol for p in open_positions if not p.is_flat}
    if signal.symbol in held:
        return LimitCheck(
            False,
            LimitBreach.ALREADY_HOLDING,
            f"already holding {signal.symbol}; pyramiding is not modelled",
        )

    # -- 3. position count -------------------------------------------------
    if len(held) >= limits.max_positions:
        return LimitCheck(
            False,
            LimitBreach.MAX_POSITIONS,
            f"{len(held)} positions open, limit is {limits.max_positions}",
        )

    # -- 4. cash -----------------------------------------------------------
    if intended_value > available_cash:
        return LimitCheck(
            False,
            LimitBreach.INSUFFICIENT_CASH,
            f"needs {intended_value}, available {available_cash}",
        )

    # -- 5. single-position notional --------------------------------------
    # Risk sizing bounds the loss IF the stop holds. This bounds the damage
    # when it gaps through -- the case the risk budget quietly assumes away.
    if equity.value > 0:
        weight = intended_value.value / equity.value
        if weight > limits.max_position_weight:
            return LimitCheck(
                False,
                LimitBreach.POSITION_WEIGHT,
                f"{signal.symbol} would be {weight:.1%} of equity, above the "
                f"{limits.max_position_weight:.1%} single-position cap "
                f"(risk sizing bounds the stop-out, not the gap)",
            )

    # -- 6. sector concentration ------------------------------------------
    # Evaluated on the portfolio as it WOULD be, not as it is: a trade that
    # only breaches the limit once filled must be stopped before it fills.
    #
    # Measured against EQUITY, not against the invested subset. Dividing by
    # invested value makes the limit tighten as the account holds less, which
    # is backwards -- and it makes the first position in any sector 100% of a
    # portfolio of one, so an empty account could never open a position at all.
    # Cash is part of the portfolio; a single position funded from an all-cash
    # account is the least concentrated state there is, not the most.
    if sectors and equity.value > 0:
        sector = sectors.get(signal.symbol)
        if sector:
            in_sector = Rupees.zero()
            for position in open_positions:
                if position.is_flat:
                    continue
                if sectors.get(position.symbol) == sector:
                    in_sector = in_sector + (position.market_value or position.cost_basis)

            weight = (in_sector + intended_value).value / equity.value
            if weight > limits.max_sector_exposure:
                return LimitCheck(
                    False,
                    LimitBreach.SECTOR_CONCENTRATION,
                    f"{sector} would reach {weight:.1%} of equity, "
                    f"above the {limits.max_sector_exposure:.1%} limit",
                )

    # -- 7. ADV participation cap -----------------------------------------
    if (
        intended_quantity is not None
        and average_daily_volume is not None
        and average_daily_volume > 0
    ):
        participation = Decimal(str(intended_quantity)) / Decimal(str(average_daily_volume))
        if participation > limits.max_adv_participation:
            detail = (
                f"{signal.symbol} order qty {intended_quantity} is {participation:.2%} of "
                f"20-day ADV ({average_daily_volume}), above {limits.max_adv_participation:.1%} cap"
            )
            return LimitCheck(False, LimitBreach.ADV_PARTICIPATION, detail)

    return LimitCheck(True)
