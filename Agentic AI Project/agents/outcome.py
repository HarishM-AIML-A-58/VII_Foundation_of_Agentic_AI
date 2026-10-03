"""Turning a finished debate into a domain signal.

The graph returns state, not a decision the rest of the system can act on.
This module is the single conversion between the two, so the mapping from
"what the agents said" to "what we might trade" exists once and is testable
without a model.

Two rules it exists to enforce:

**Absence of a decision is never promoted into a decision.** No verdict, a
HOLD verdict, or missing entry geometry all yield ``None``. A debate that
failed to conclude must not become a trade with defaulted numbers.

**The signal produced here is a candidate, not an approval.** It carries
``SignalStatus.GENERATED`` and nothing else. The binding gates -- the veto,
the limits, the sizing -- live in :mod:`trading_agent.risk`, which sits
*above* this layer and is applied by the caller. Nothing in the agents package
can approve its own output.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation

from trading_agent.agents.state import DebateState
from trading_agent.domain import (
    Action,
    AgentVote,
    Exchange,
    ProductType,
    TradeSignal,
    build_idempotency_key,
)
from trading_agent.observability import get_logger

__all__ = ["DEBATE_STRATEGY", "collect_votes", "to_signal"]

log = get_logger(__name__)

#: Strategy name recorded on every signal the debate produces. Also feeds the
#: idempotency key, so a debate and a quantitative strategy proposing the same
#: symbol on the same session stay distinguishable.
DEBATE_STRATEGY = "agent_debate"

_STANCE_TO_ACTION = {"BULLISH": Action.BUY, "BEARISH": Action.SELL, "NEUTRAL": Action.HOLD}


def _to_decimal(value: float | None) -> Decimal | None:
    """Convert a schema float to Decimal via ``str``.

    ``Decimal(1000.1)`` carries the float's binary error into the domain;
    ``Decimal("1000.1")`` does not. The schemas use ``float`` because that is
    what JSON structured output produces, so this is the boundary where it
    stops being a float.
    """
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def collect_votes(state: DebateState) -> dict[str, AgentVote]:
    """Every voice in the debate, as domain votes.

    Analysts, researchers, the moderator and the risk manager all appear, so
    the persisted transcript is the whole record rather than the conclusion.
    Token counts are carried through from the per-role usage, which is what
    makes "the bear researcher is 60% of the bill" answerable later.
    """
    usage_by_role = {u.role: u for u in state.get("usage", [])}

    def tokens(role: str) -> tuple[int, int]:
        record = usage_by_role.get(role)
        return (0, 0) if record is None else (record.input_tokens, record.output_tokens)

    def deployment(role: str) -> str | None:
        record = usage_by_role.get(role)
        return None if record is None else record.deployment

    votes: dict[str, AgentVote] = {}

    for role, opinion in state.get("opinions", {}).items():
        input_tokens, output_tokens = tokens(role)
        votes[role] = AgentVote(
            agent=role,
            action=_STANCE_TO_ACTION.get(opinion.stance, Action.HOLD),
            confidence=opinion.confidence,
            rationale=opinion.summary,
            deployment=deployment(role),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    # One vote per researcher, from their final turn: the last argument is the
    # position they actually ended on. Earlier turns stay in `turns` for the
    # transcript, but a vote is a conclusion, not a history.
    for side in ("BULL", "BEAR"):
        side_turns = [t for t in state.get("turns", []) if t.side == side]
        if not side_turns:
            continue
        final = side_turns[-1]
        role = "bull_researcher" if side == "BULL" else "bear_researcher"
        input_tokens, output_tokens = tokens(role)
        votes[role] = AgentVote(
            agent=role,
            action=Action.BUY if side == "BULL" else Action.SELL,
            confidence=final.confidence,
            rationale=final.argument,
            deployment=deployment(role),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    verdict = state.get("verdict")
    if verdict is not None:
        input_tokens, output_tokens = tokens("debate_moderator")
        votes["debate_moderator"] = AgentVote(
            agent="debate_moderator",
            action=Action(verdict.action),
            confidence=verdict.confidence,
            rationale=verdict.rationale,
            deployment=deployment("debate_moderator"),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    risk = state.get("risk")
    if risk is not None:
        input_tokens, output_tokens = tokens("risk_manager")
        concerns = "; ".join(risk.concerns) if risk.concerns else "no specific concerns"
        # Advisory: recorded as a vote, binding on nothing. Withholding
        # approval reads as HOLD; approving endorses whatever the moderator
        # proposed. The rejection that actually stops a trade is `risk.veto`,
        # which is arithmetic and cannot be argued with.
        if risk.approve and verdict is not None:
            risk_action = Action(verdict.action)
        else:
            risk_action = Action.HOLD
        votes["risk_manager"] = AgentVote(
            agent="risk_manager",
            action=risk_action,
            confidence=risk.confidence,
            rationale=f"{risk.rationale} (concerns: {concerns})",
            deployment=deployment("risk_manager"),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    return votes


def to_signal(
    state: DebateState,
    *,
    product: ProductType = ProductType.CNC,
    exchange: Exchange = Exchange.NSE,
    strategy: str = DEBATE_STRATEGY,
    generated_at: datetime | None = None,
    sequence: int = 0,
) -> TradeSignal | None:
    """Convert a finished debate into a candidate signal, or ``None``.

    ``None`` means there is nothing to trade, for one of four reasons, all of
    which are logged rather than silently swallowed:

    * the moderator abstained, so there is no verdict at all,
    * the verdict was HOLD, which is a first-class answer and not a failure,
    * the verdict proposed no entry geometry, so there is no stop to size against,
    * the geometry is internally inconsistent (a long stop above entry, say),
      which the domain rejects and we decline to repair by guessing.
    """
    symbol = state["symbol"]
    session_date = state["session_date"]
    verdict = state.get("verdict")

    if verdict is None:
        log.info("no_signal", symbol=symbol, reason="moderator abstained")
        return None
    if verdict.action == "HOLD":
        log.info("no_signal", symbol=symbol, reason="verdict was HOLD")
        return None

    entry = _to_decimal(verdict.entry)
    stop_loss = _to_decimal(verdict.stop_loss)
    target = _to_decimal(verdict.target)
    if entry is None or stop_loss is None or target is None:
        # An actionable verdict with no stop cannot be sized: there is no risk
        # distance. Inventing one would fabricate the number the whole risk
        # layer is built on.
        log.warning("no_signal", symbol=symbol, reason="verdict proposed no entry geometry")
        return None

    try:
        return TradeSignal(
            symbol=symbol,
            action=Action(verdict.action),
            product=product,
            entry=entry,
            stop_loss=stop_loss,
            target=target,
            generated_at=generated_at or state["generated_at"],
            session_date=session_date,
            strategy=strategy,
            idempotency_key=build_idempotency_key(
                strategy=strategy,
                symbol=symbol,
                session_date=session_date,
                sequence=sequence,
            ),
            exchange=exchange,
            confidence=verdict.confidence,
            rationale=verdict.rationale,
            agent_votes=collect_votes(state),
        )
    except ValueError as exc:
        # The domain refused the geometry -- e.g. a long whose stop sits above
        # its entry. The model contradicted itself; we do not repair it.
        log.warning("no_signal", symbol=symbol, reason=f"invalid geometry: {exc}")
        return None
