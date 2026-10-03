"""Multi-agent debate on Microsoft Foundry.

Design decisions that differ from the reference implementations
(TradingAgents / SkopaqTrader), and why:

* **Structured output, not prose parsing.** Every node returns a validated
  Pydantic object. The references produce prose and spend a second LLM call
  extracting "BUY/SELL/HOLD" from it.
* **Prompts in versioned `.md` files**, not f-strings inside node code, so a
  prompt change is a reviewable prose diff.
* **Typed debate turns**, not an accumulating string. The reference
  concatenates history into one growing `str`, which re-sends everything on
  every turn.
* **A hard-coded veto** in `risk.veto`, outside the graph. The reference ends
  with an LLM "risk judge" whose verdict is final; a model that can be argued
  with is not a risk control.
* **Abstention over failure.** A filtered or rate-limited node becomes a named
  missing voice rather than aborting the session.
"""

from trading_agent.agents.cache import ResponseCache, cache_from_settings, cache_key
from trading_agent.agents.graph import DEFAULT_DEBATE_ROUNDS, build_debate_graph, run_debate
from trading_agent.agents.llm import (
    AGENT_ROLES,
    ConnectivityResult,
    ContentFilteredError,
    FoundryClient,
    QuotaExceededError,
)
from trading_agent.agents.outcome import DEBATE_STRATEGY, collect_votes, to_signal
from trading_agent.agents.prompts import available_prompts, load_prompt
from trading_agent.agents.schemas import (
    AnalystOpinion,
    DebateArgument,
    ModeratorVerdict,
    RiskVerdict,
)
from trading_agent.agents.state import DebateState, DebateTurn, TokenUsage, new_state

__all__ = [
    "AGENT_ROLES",
    "DEBATE_STRATEGY",
    "DEFAULT_DEBATE_ROUNDS",
    "AnalystOpinion",
    "ConnectivityResult",
    "ContentFilteredError",
    "DebateArgument",
    "DebateState",
    "DebateTurn",
    "FoundryClient",
    "ModeratorVerdict",
    "QuotaExceededError",
    "ResponseCache",
    "RiskVerdict",
    "TokenUsage",
    "available_prompts",
    "build_debate_graph",
    "cache_from_settings",
    "cache_key",
    "collect_votes",
    "load_prompt",
    "new_state",
    "run_debate",
    "to_signal",
]
