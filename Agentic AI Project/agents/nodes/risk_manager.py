"""The risk manager node.

**Advisory only.** Its opinion is recorded and shown, but the binding checks
-- the reward-to-risk floor, position and sector limits, the kill switch --
run afterwards in :mod:`trading_agent.risk` as arithmetic in code.

This is the deliberate difference from the reference implementation, where a
"risk judge" LLM has the final say. A model that can be argued with is not a
risk control. A function that rejects reward-to-risk below 1.8 cannot be
persuaded, prompt-injected, or have a bad day.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from trading_agent.agents.llm import FoundryClient
from trading_agent.agents.nodes._common import AbstentionError, call_role, render_opinions
from trading_agent.agents.schemas import RiskVerdict
from trading_agent.agents.state import DebateState
from trading_agent.observability import get_logger

__all__ = ["risk_manager"]

log = get_logger(__name__)


def risk_manager(client: FoundryClient) -> Callable[[DebateState], Awaitable[DebateState]]:
    async def node(state: DebateState) -> DebateState:
        verdict = state.get("verdict")
        if verdict is None or verdict.action == "HOLD":
            # Nothing to assess. Skipping the call also skips its cost.
            return DebateState()

        geometry = (
            f"entry {verdict.entry}, stop {verdict.stop_loss}, target {verdict.target}"
            if verdict.entry and verdict.stop_loss and verdict.target
            else "the moderator proposed no entry geometry"
        )
        prompt = (
            f"Symbol: {state['symbol']}\n"
            f"Session: {state['session_date'].isoformat()}\n\n"
            f"## Proposed trade\n\n"
            f"action: {verdict.action} (confidence {verdict.confidence:.2f})\n"
            f"{geometry}\n"
            f"rationale: {verdict.rationale}\n\n"
            f"## Analyst reports\n\n{render_opinions(state)}\n\n"
            f"## Market data\n\n{state['market_context']}\n\n"
            f"Assess the risk. Recompute the reward-to-risk ratio yourself."
        )
        try:
            risk, usage = await call_role(
                client, role="risk_manager", user_content=prompt, schema=RiskVerdict
            )
        except AbstentionError as exc:
            # No risk opinion means the trade proceeds to the CODE gates with
            # no advisory sign-off. Those gates are what actually protect the
            # account, so this degrades safely.
            log.warning("risk_manager_abstained", reason=str(exc))
            return DebateState(abstentions=["risk_manager"])

        log.info(
            "risk_verdict",
            symbol=state["symbol"],
            approve=risk.approve,
            concerns=len(risk.concerns),
        )
        return DebateState(risk=risk, usage=[usage])

    node.__name__ = "risk_manager"
    return node
