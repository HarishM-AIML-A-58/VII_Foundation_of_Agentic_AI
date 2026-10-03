"""The binding risk gate.

**This is code, not a prompt.** The reference implementation ends its pipeline
with a "risk judge" LLM whose verdict is final. That is not a risk control: a
model can be argued with, can be prompt-injected through the market data it
reads, and can simply have an off day.

Everything here is arithmetic. It cannot be persuaded.

The agent debate is *advice*. This decides.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from trading_agent.domain.enums import Action, ProductType
from trading_agent.domain.signal import TradeSignal
from trading_agent.observability import get_logger

__all__ = ["VetoReason", "VetoResult", "apply_veto"]

log = get_logger(__name__)


class VetoReason(StrEnum):
    REWARD_TO_RISK_TOO_LOW = "reward_to_risk_too_low"
    NO_STOP_LOSS = "no_stop_loss"
    STOP_ON_WRONG_SIDE = "stop_on_wrong_side"
    RISK_EXCEEDS_REWARD_AFTER_COSTS = "risk_exceeds_reward_after_costs"
    CONFIDENCE_TOO_LOW = "confidence_too_low"
    NOT_ACTIONABLE = "not_actionable"
    SHORT_IN_DELIVERY = "short_in_delivery"


@dataclass(frozen=True, slots=True)
class VetoResult:
    approved: bool
    reason: VetoReason | None = None
    detail: str = ""


#: Round-trip cost for NSE delivery, in basis points. A move that does not
#: clear this is a loss regardless of direction. Sourced from the cost model's
#: own output (11-12 bps), rounded up.
ROUND_TRIP_COST_BPS = Decimal("12")

#: Multiple of round-trip cost the reward must clear. 3x is not a deep edge;
#: it merely rules out trades whose entire thesis is smaller than their bill.
MIN_REWARD_COST_MULTIPLE = Decimal("3")


def apply_veto(
    signal: TradeSignal,
    *,
    min_reward_to_risk: Decimal,
    min_confidence: float = 0.0,
) -> TradeSignal:
    """Return ``signal``, marked vetoed if it fails any hard rule.

    Checks, in order of how cheaply they settle the question:

    1. Actionable at all.
    2. The trade is *possible* on the chosen product.
    3. A stop exists and sits on the correct side of entry.
    4. Reward-to-risk clears the floor -- **recomputed here net of costs**,
       never taken from whatever the model claimed.
    5. The reward clears round-trip costs with margin.
    6. Confidence clears the floor, when one is configured.
    """
    result = evaluate(signal, min_reward_to_risk=min_reward_to_risk, min_confidence=min_confidence)
    if result.approved:
        return signal

    log.warning(
        "signal_vetoed",
        symbol=signal.symbol,
        reason=result.reason.value if result.reason else "unknown",
        detail=result.detail,
    )
    return signal.vetoed(f"{result.reason.value if result.reason else 'vetoed'}: {result.detail}")


def evaluate(
    signal: TradeSignal, *, min_reward_to_risk: Decimal, min_confidence: float = 0.0
) -> VetoResult:
    """The decision, separated so it can be tested without constructing state."""
    if not signal.action.is_actionable:
        return VetoResult(False, VetoReason.NOT_ACTIONABLE, f"action is {signal.action.value}")

    # A short cannot be carried in the delivery product on NSE cash equity.
    # Intraday (MIS) shorts are squared off before the close; a CNC short would
    # be an attempt to deliver shares that were never bought, which the broker
    # rejects -- or, worse, sells an unrelated holding of the same symbol.
    #
    # This is checked here rather than at the broker because the backtest
    # engine silently drops non-BUY signals, so nothing upstream exercises the
    # path: the first time a BEARISH verdict would meet reality is with real
    # money. Varsity M10 Ch16 states the constraint plainly -- to be short and
    # hold, you need the futures segment.
    if signal.action is Action.SELL and signal.product is ProductType.CNC:
        return VetoResult(
            False,
            VetoReason.SHORT_IN_DELIVERY,
            "cannot carry a short in CNC delivery on NSE cash equity; "
            "use MIS for an intraday short, or the futures segment to carry one",
        )

    if signal.stop_loss <= 0:
        return VetoResult(False, VetoReason.NO_STOP_LOSS, "no protective stop was set")

    # TradeSignal validates stop side at construction, so reaching here means
    # something bypassed it. Checked anyway: this gate must not assume that
    # every upstream invariant held.
    if signal.risk_per_share <= 0:
        return VetoResult(
            False,
            VetoReason.STOP_ON_WRONG_SIDE,
            f"stop {signal.stop_loss} gives no risk distance from entry {signal.entry}",
        )

    # Reward-to-risk NET OF COSTS, not gross.
    #
    # Costs cut both ways: they shrink the win and deepen the loss. A trade at
    # entry 1325 / stop 1278 / target 1412 shows a gross ratio of 1.85 and
    # passes a 1.8 floor -- but at 12 bps round trip (Rs 1.59 per share) the
    # effective ratio is 1.76, and it should not.
    #
    # Found by the risk-manager agent flagging it on a live run against real
    # RELIANCE data while this gate was still checking the gross figure.
    cost_per_share = signal.entry * ROUND_TRIP_COST_BPS / Decimal(10_000)
    net_reward = signal.reward_per_share - cost_per_share
    net_risk = signal.risk_per_share + cost_per_share
    reward_to_risk = net_reward / net_risk if net_risk > 0 else Decimal(0)

    if reward_to_risk < min_reward_to_risk:
        return VetoResult(
            False,
            VetoReason.REWARD_TO_RISK_TOO_LOW,
            f"reward-to-risk {reward_to_risk:.2f} net of costs is below the "
            f"{min_reward_to_risk} floor (gross {signal.reward_to_risk:.2f}, "
            f"costs {cost_per_share:.2f}/share)",
        )

    # A trade can clear the R:R floor and still be uneconomic: risking 2 to
    # make 4 is pointless when the round trip costs 3. Reward is measured in
    # basis points of entry so it is comparable with the cost figure.
    reward_bps = (signal.reward_per_share / signal.entry) * Decimal(10_000)
    required_bps = ROUND_TRIP_COST_BPS * MIN_REWARD_COST_MULTIPLE
    if reward_bps < required_bps:
        return VetoResult(
            False,
            VetoReason.RISK_EXCEEDS_REWARD_AFTER_COSTS,
            f"reward {reward_bps:.0f} bps does not clear {required_bps:.0f} bps "
            f"({ROUND_TRIP_COST_BPS} bps round trip x{MIN_REWARD_COST_MULTIPLE})",
        )

    if min_confidence > 0 and signal.confidence < min_confidence:
        return VetoResult(
            False,
            VetoReason.CONFIDENCE_TOO_LOW,
            f"confidence {signal.confidence:.2f} is below the {min_confidence:.2f} floor",
        )

    return VetoResult(True)
