"""Debate nodes. One module per role, each mapped to a Foundry deployment.

Every node follows the same contract:

* takes :class:`~trading_agent.agents.state.DebateState`,
* returns ONLY the keys it owns (LangGraph merges via the state reducers),
* returns a schema-validated object, never prose to be parsed later,
* records token usage for cost attribution,
* on failure records an abstention rather than raising.

That last rule matters most. One analyst hitting a content filter or a quota
wall must not abort the session; it must be visible as a missing voice, and
the moderator is told which voices are missing so it can discount accordingly.
"""

from trading_agent.agents.nodes.analysts import (
    fundamental_analyst,
    sentiment_analyst,
    technical_analyst,
)
from trading_agent.agents.nodes.debate_moderator import debate_moderator
from trading_agent.agents.nodes.forensic_skeptic import forensic_skeptic
from trading_agent.agents.nodes.ipo_researcher import IPOResearchInput, research_ipo
from trading_agent.agents.nodes.researchers import bear_researcher, bull_researcher
from trading_agent.agents.nodes.risk_manager import risk_manager

__all__ = [
    "IPOResearchInput",
    "bear_researcher",
    "bull_researcher",
    "debate_moderator",
    "forensic_skeptic",
    "fundamental_analyst",
    "research_ipo",
    "risk_manager",
    "sentiment_analyst",
    "technical_analyst",
]
