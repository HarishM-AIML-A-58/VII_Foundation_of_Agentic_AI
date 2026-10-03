"""Bull and bear researcher nodes.

They alternate, each seeing the opponent's last argument, for a bounded number
of rounds. Bounded because an unbounded debate is an unbounded bill, and
because two models rarely produce new information after the second exchange.

Deliberately on **different model families** (see `config/settings.base.yaml`):
two instances of one model tend to be wrong in the same direction, which turns
a debate into theatre.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from trading_agent.agents.llm import FoundryClient
from trading_agent.agents.nodes._common import AbstentionError, call_role, render_opinions
from trading_agent.agents.schemas import DebateArgument
from trading_agent.agents.state import DebateState, DebateTurn
from trading_agent.observability import get_logger

__all__ = ["bear_researcher", "bull_researcher", "make_researcher"]

log = get_logger(__name__)

ResearcherNode = Callable[[DebateState], Awaitable[DebateState]]

#: How many previous turns to replay. The full history is NOT re-sent: it
#: grows without bound and the older turns stop informing the argument long
#: before they stop costing tokens.
_HISTORY_WINDOW = 4


def _render_history(state: DebateState, *, side: str) -> str:
    turns = state.get("turns", [])
    if not turns:
        opponent = "bear" if side == "BULL" else "bull"
        return f"You are opening the debate. There is no {opponent} argument yet."

    recent = turns[-_HISTORY_WINDOW:]
    rendered = "\n\n".join(
        f"[round {t.round_number}] {t.side}: {t.argument}\n"
        f"  strongest point: {t.strongest_point}\n"
        f"  conceded: {', '.join(t.conceded) if t.conceded else 'nothing'}"
        for t in recent
    )
    opposing = [t for t in turns if t.side != side]
    last = opposing[-1].argument if opposing else "none yet"
    return f"{rendered}\n\nThe argument you must answer:\n{last}"


def make_researcher(client: FoundryClient, side: str) -> ResearcherNode:
    """Build the bull (``side='BULL'``) or bear (``side='BEAR'``) node."""
    role = "bull_researcher" if side == "BULL" else "bear_researcher"

    async def node(state: DebateState) -> DebateState:
        round_number = state.get("round_number", 0) + (1 if side == "BULL" else 0)
        prompt = (
            f"Symbol: {state['symbol']}\n"
            f"Session: {state['session_date'].isoformat()}\n\n"
            f"## Analyst reports\n\n{render_opinions(state)}\n\n"
            f"## Debate so far\n\n{_render_history(state, side=side)}\n\n"
            f"Make the {side.lower()} case for {state['symbol']}."
        )
        try:
            argument, usage = await call_role(
                client, role=role, user_content=prompt, schema=DebateArgument
            )
        except AbstentionError as exc:
            log.warning("researcher_abstained", role=role, reason=str(exc))
            return DebateState(abstentions=[role], round_number=round_number)

        turn = DebateTurn(
            side=side,
            round_number=round_number,
            argument=argument.argument,
            strongest_point=argument.strongest_point,
            conceded=tuple(argument.conceded),
            confidence=argument.confidence,
        )
        return DebateState(turns=[turn], usage=[usage], round_number=round_number)

    node.__name__ = role
    return node


def bull_researcher(client: FoundryClient) -> ResearcherNode:
    return make_researcher(client, "BULL")


def bear_researcher(client: FoundryClient) -> ResearcherNode:
    return make_researcher(client, "BEAR")
