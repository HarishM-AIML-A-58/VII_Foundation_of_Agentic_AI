"""Typed state threaded through the debate graph.

Two design decisions worth the words, both learned from reading the reference
implementations and deciding differently:

**1. Reducers are mandatory, not optional.**
LangGraph's default ``LastValue`` channel raises when two parallel branches
write the same key. The three analysts fan out concurrently, so every key they
touch needs an explicit reducer. This is the single most common way a
LangGraph fan-out breaks, and it breaks at runtime, mid-session.

**2. History is a list of typed turns, not an accumulated string.**
TradingAgents concatenates the debate into one growing ``str``. That is a
token bomb (every turn re-sends the whole history), it cannot be rendered
without re-parsing, and nothing checks its shape. Typed turns cost nothing
extra and the UI renders them directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Annotated, Any, TypedDict

from trading_agent.agents.schemas import (
    AnalystOpinion,
    ForensicSkepticVerdict,
    ModeratorVerdict,
    RiskVerdict,
)

__all__ = ["DebateState", "DebateTurn", "TokenUsage", "new_state"]


@dataclass(frozen=True, slots=True)
class DebateTurn:
    """One researcher's turn. Rendered verbatim in the UI transcript."""

    side: str  # BULL | BEAR
    round_number: int
    argument: str
    strongest_point: str
    conceded: tuple[str, ...] = ()
    confidence: float = 0.0


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Per-role token spend, for cost attribution.

    Kept because "the agents are expensive" is not actionable; "the bear
    researcher is 60% of the bill" is.
    """

    role: str
    deployment: str
    input_tokens: int
    output_tokens: int


# -- reducers ---------------------------------------------------------------
# Each is total and order-independent, because LangGraph gives no guarantee
# about the order parallel branches converge in.


def merge_opinions(
    left: dict[str, AnalystOpinion], right: dict[str, AnalystOpinion]
) -> dict[str, AnalystOpinion]:
    """Merge analyst opinions from the parallel fan-out.

    Each analyst writes exactly one key, so a dict merge is correct and
    order-independent. Without this reducer LangGraph raises
    ``InvalidUpdateError`` the moment the second analyst returns.
    """
    return {**left, **right}


def append_turns(left: list[DebateTurn], right: list[DebateTurn]) -> list[DebateTurn]:
    """Append debate turns in arrival order."""
    return [*left, *right]


def append_usage(left: list[TokenUsage], right: list[TokenUsage]) -> list[TokenUsage]:
    return [*left, *right]


def keep_latest(left: Any, right: Any) -> Any:
    """Keep the newer value, treating None as "no update"."""
    return right if right is not None else left


class DebateState(TypedDict, total=False):
    """State for one symbol's debate.

    ``total=False`` because nodes populate it progressively; every consumer
    uses ``.get`` with a default rather than assuming a key exists.
    """

    # -- inputs, written once before the graph runs --------------------
    symbol: str
    session_date: date
    market_context: str
    generated_at: datetime

    # -- analyst fan-out (parallel: needs a reducer) -------------------
    opinions: Annotated[dict[str, AnalystOpinion], merge_opinions]

    # -- researcher debate (sequential, but appended) ------------------
    turns: Annotated[list[DebateTurn], append_turns]
    round_number: Annotated[int, keep_latest]

    # -- synthesis and risk --------------------------------------------
    verdict: Annotated[ModeratorVerdict | None, keep_latest]
    risk: Annotated[RiskVerdict | None, keep_latest]
    forensic_audit: Annotated[ForensicSkepticVerdict | None, keep_latest]

    # -- bookkeeping ----------------------------------------------------
    usage: Annotated[list[TokenUsage], append_usage]
    #: Roles that abstained -- content filter, quota exhaustion, malformed
    #: output. Never silently dropped: an abstention changes what the debate
    #: means, and the moderator is told which voices are missing.
    abstentions: Annotated[list[str], append_usage]  # same append semantics


def new_state(
    *, symbol: str, session_date: date, market_context: str, generated_at: datetime
) -> DebateState:
    """Seed a debate. Every accumulating channel starts empty, not absent."""
    return DebateState(
        symbol=symbol,
        session_date=session_date,
        market_context=market_context,
        generated_at=generated_at,
        opinions={},
        turns=[],
        round_number=0,
        verdict=None,
        risk=None,
        forensic_audit=None,
        usage=[],
        abstentions=[],
    )
