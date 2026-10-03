"""The forensic skeptic node.

An adversarial auditor node evaluating earnings quality, manipulation risk
(Beneish M-Score), balance sheet solvency (Altman Z-Score), and establishing
explicit thesis invalidation conditions.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from trading_agent.agents.llm import FoundryClient
from trading_agent.agents.nodes._common import AbstentionError, call_role
from trading_agent.agents.schemas import ForensicSkepticVerdict
from trading_agent.agents.state import DebateState
from trading_agent.observability import get_logger

__all__ = ["forensic_skeptic"]

log = get_logger(__name__)

ForensicNode = Callable[[DebateState], Awaitable[DebateState]]


def forensic_skeptic(client: FoundryClient) -> ForensicNode:
    """Build a forensic skeptic audit node."""

    async def node(state: DebateState) -> DebateState:
        computed = _computed_block(state["symbol"], state["session_date"])
        prompt = (
            f"Symbol: {state['symbol']}\n"
            f"Session: {state['session_date'].isoformat()}\n\n"
            f"{state['market_context']}\n\n"
            f"{computed}\n\n"
            f"Conduct an adversarial forensic audit on {state['symbol']}. "
            "Audit earnings quality, accruals, solvency, and specify 1-3 concrete "
            "thesis invalidation triggers. If the computed block says a score is "
            "not in the data, you MUST set that classification from the data you "
            "have or abstain — never invent a Beneish M-Score or Altman Z-Score."
        )
        try:
            verdict, usage = await call_role(
                client, role="forensic_skeptic", user_content=prompt, schema=ForensicSkepticVerdict
            )
        except AbstentionError as exc:
            log.warning("forensic_skeptic_abstained", reason=str(exc))
            return DebateState(abstentions=["forensic_skeptic"])

        return DebateState(forensic_audit=verdict, usage=[usage])

    node.__name__ = "forensic_skeptic"
    return node


def _computed_block(symbol: str, session_date: object) -> str:
    """Attach arithmetic forensic inputs when the statement snapshot has them.

    Failure to load the store is not a clean audit: the model is told the
    numbers are absent.
    """
    try:
        from datetime import date as date_cls

        from trading_agent.features.statement_snapshot import forensic_prompt_block

        on = session_date if isinstance(session_date, date_cls) else None
        return forensic_prompt_block(symbol, on)
    except Exception as exc:  # snapshot is optional; the node must still run
        log.info("forensic_snapshot_unavailable", symbol=symbol, error=str(exc)[:200])
        return (
            "## Computed forensic inputs\n"
            "not in the data — the statement snapshot could not be loaded. "
            "Do not invent M-Score, Z-Score, or cash-flow ratios."
        )
