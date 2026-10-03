"""LangGraph assembly for the debate.

Shape::

        START
          │
    ┌─────┴─────┬─────────────┬────────────────┐   four voices, PARALLEL
    ▼           ▼             ▼                ▼
  technical  fundamental  sentiment    forensic_skeptic
    └─────┬─────┴─────────────┴────────────────┘   fan-in (reducers)
          ▼
        bull ⇄ bear                    alternating, BOUNDED rounds
          ▼
      moderator                        typed verdict, sees forensic_audit
          ▼
     risk manager                      ADVISORY only
          ▼
         END
          │
          └──▶ risk.veto()             BINDING, in code, unpersuadable

The last step is outside the graph on purpose. Everything inside can be
influenced by a prompt; the veto cannot.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from trading_agent.agents.llm import FoundryClient
from trading_agent.agents.nodes import (
    bear_researcher,
    bull_researcher,
    debate_moderator,
    forensic_skeptic,
    fundamental_analyst,
    risk_manager,
    sentiment_analyst,
    technical_analyst,
)
from trading_agent.agents.state import DebateState, new_state
from trading_agent.observability import get_logger, trade_context

__all__ = ["build_debate_graph", "run_debate"]

log = get_logger(__name__)

#: Full bull+bear exchanges. Two is the practical sweet spot: the first
#: establishes both cases, the second tests them. Beyond that the arguments
#: repeat while the bill keeps growing.
DEFAULT_DEBATE_ROUNDS = 2

#: Four voices fan out from START. The forensic skeptic is an auditor, not a
#: BUY/SELL voter; it writes ``forensic_audit`` and the moderator is told to
#: weigh it. Parallel with the analysts because it only needs market_context.
_ANALYSTS = (
    "technical_analyst",
    "fundamental_analyst",
    "sentiment_analyst",
    "forensic_skeptic",
)


def build_debate_graph(client: FoundryClient, *, rounds: int = DEFAULT_DEBATE_ROUNDS) -> Any:
    """Compile the debate graph.

    ``rounds`` bounds the bull/bear exchange. The termination condition is a
    plain counter rather than a model deciding when it is finished -- an agent
    asked whether it has more to say will generally say yes.
    """
    if rounds < 1:
        raise ValueError(f"rounds must be at least 1, got {rounds}")

    from langgraph.graph import END, START, StateGraph

    builder: Any = StateGraph(DebateState)

    builder.add_node("technical_analyst", technical_analyst(client))
    builder.add_node("fundamental_analyst", fundamental_analyst(client))
    builder.add_node("sentiment_analyst", sentiment_analyst(client))
    builder.add_node("forensic_skeptic", forensic_skeptic(client))
    builder.add_node("bull_researcher", bull_researcher(client))
    builder.add_node("bear_researcher", bear_researcher(client))
    builder.add_node("debate_moderator", debate_moderator(client))
    builder.add_node("risk_manager", risk_manager(client))

    # Analysts fan out from START. Three edges from one source is what makes
    # them concurrent, and what makes the reducers on `opinions` mandatory.
    for analyst in _ANALYSTS:
        builder.add_edge(START, analyst)
        builder.add_edge(analyst, "bull_researcher")

    builder.add_edge("bull_researcher", "bear_researcher")

    def should_continue(state: DebateState) -> str:
        """Another exchange, or synthesise?

        A counter, not a judgement. Deliberate: asking a model whether the
        debate is finished reliably produces "one more round".
        """
        completed = state.get("round_number", 0)
        return "bull_researcher" if completed < rounds else "debate_moderator"

    builder.add_conditional_edges(
        "bear_researcher",
        should_continue,
        {"bull_researcher": "bull_researcher", "debate_moderator": "debate_moderator"},
    )
    builder.add_edge("debate_moderator", "risk_manager")
    builder.add_edge("risk_manager", END)

    return builder.compile()


async def run_debate(
    client: FoundryClient,
    *,
    symbol: str,
    session_date: date,
    market_context: str,
    generated_at: datetime,
    rounds: int = DEFAULT_DEBATE_ROUNDS,
) -> DebateState:
    """Run one symbol through the debate and return the final state.

    Returns state rather than a :class:`~trading_agent.domain.signal.TradeSignal`
    on purpose: converting to a signal means applying the risk layer, and that
    belongs to the caller, not to the thing being judged.
    """
    # Scoped: a research run debates many symbols in one task, and an unscoped
    # bind would leave the previous symbol's name on every line until the next
    # debate overwrote it.
    with trade_context(symbol=symbol, session=session_date.isoformat()):
        graph = build_debate_graph(client, rounds=rounds)

        initial = new_state(
            symbol=symbol,
            session_date=session_date,
            market_context=market_context,
            generated_at=generated_at,
        )
        final: DebateState = await graph.ainvoke(initial)

        usage = final.get("usage", [])
        log.info(
            "debate_complete",
            symbol=symbol,
            turns=len(final.get("turns", [])),
            abstentions=final.get("abstentions", []),
            input_tokens=sum(u.input_tokens for u in usage),
            output_tokens=sum(u.output_tokens for u in usage),
        )
        return final
