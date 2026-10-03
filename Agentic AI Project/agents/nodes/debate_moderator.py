"""The moderator: synthesises the exchange into a typed decision."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from trading_agent.agents.llm import FoundryClient
from trading_agent.agents.nodes._common import (
    AbstentionError,
    call_role,
    render_forensic,
    render_opinions,
)
from trading_agent.agents.schemas import ModeratorVerdict
from trading_agent.agents.state import DebateState
from trading_agent.observability import get_logger

__all__ = ["debate_moderator"]

log = get_logger(__name__)


def debate_moderator(client: FoundryClient) -> Callable[[DebateState], Awaitable[DebateState]]:
    async def node(state: DebateState) -> DebateState:
        turns = state.get("turns", [])
        transcript = (
            "\n\n".join(
                f"[round {t.round_number}] {t.side} (confidence {t.confidence:.2f})\n"
                f"{t.argument}\n"
                f"  strongest point: {t.strongest_point}\n"
                f"  conceded: {', '.join(t.conceded) if t.conceded else 'NOTHING'}"
                for t in turns
            )
            or "The debate produced no arguments."
        )

        prompt = (
            f"Symbol: {state['symbol']}\n"
            f"Session: {state['session_date'].isoformat()}\n\n"
            f"## Analyst reports\n\n{render_opinions(state)}\n\n"
            f"## Forensic audit\n\n{render_forensic(state)}\n\n"
            f"## Full debate\n\n{transcript}\n\n"
            f"## Market data\n\n{state['market_context']}\n\n"
            f"Decide. HOLD is a first-class answer. "
            f"A HIGH manipulation risk or DISTRESS solvency zone is a reason to HOLD "
            f"or reject, not something to average away."
        )
        try:
            verdict, usage = await call_role(
                client, role="debate_moderator", user_content=prompt, schema=ModeratorVerdict
            )
        except AbstentionError as exc:
            # No verdict means no trade. Absence of a decision is never
            # promoted into a decision.
            log.warning("moderator_abstained", reason=str(exc))
            return DebateState(abstentions=["debate_moderator"])

        log.info(
            "moderator_verdict",
            symbol=state["symbol"],
            action=verdict.action,
            confidence=verdict.confidence,
            winning_side=verdict.winning_side,
        )
        return DebateState(verdict=verdict, usage=[usage])

    node.__name__ = "debate_moderator"
    return node
