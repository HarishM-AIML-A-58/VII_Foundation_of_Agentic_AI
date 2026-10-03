"""The three analyst nodes.

They run **in parallel**: each reads the same market context and writes one
key of ``opinions``. That fan-out is why ``DebateState.opinions`` carries a
reducer -- without it LangGraph raises the moment the second analyst returns.

One factory rather than three near-identical modules. The reference
implementation has a separate file per analyst that differ only in their
prompt and output key, plus a separate `should_continue_X` for each; the
duplication is where they drift apart.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from trading_agent.agents.llm import FoundryClient
from trading_agent.agents.nodes._common import AbstentionError, call_role
from trading_agent.agents.schemas import AnalystOpinion
from trading_agent.agents.state import DebateState
from trading_agent.observability import get_logger

__all__ = ["fundamental_analyst", "make_analyst", "sentiment_analyst", "technical_analyst"]

log = get_logger(__name__)

AnalystNode = Callable[[DebateState], Awaitable[DebateState]]


def make_analyst(client: FoundryClient, role: str) -> AnalystNode:
    """Build an analyst node for ``role``."""

    async def node(state: DebateState) -> DebateState:
        prompt = (
            f"Symbol: {state['symbol']}\n"
            f"Session: {state['session_date'].isoformat()}\n\n"
            f"{state['market_context']}\n\n"
            f"Give your read of {state['symbol']} on {state['session_date'].isoformat()}."
        )
        try:
            opinion, usage = await call_role(
                client, role=role, user_content=prompt, schema=AnalystOpinion
            )
        except AbstentionError as exc:
            # One analyst falling over must not abort the session. It becomes
            # a named missing voice that downstream prompts can discount.
            log.warning("analyst_abstained", role=role, reason=str(exc))
            return DebateState(abstentions=[role])

        return DebateState(opinions={role: opinion}, usage=[usage])

    node.__name__ = role
    return node


def technical_analyst(client: FoundryClient) -> AnalystNode:
    return make_analyst(client, "technical_analyst")


def fundamental_analyst(client: FoundryClient) -> AnalystNode:
    return make_analyst(client, "fundamental_analyst")


def sentiment_analyst(client: FoundryClient) -> AnalystNode:
    return make_analyst(client, "sentiment_analyst")
