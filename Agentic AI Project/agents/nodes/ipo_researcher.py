"""IPO research note: one LLM read over an issue's scraped disclosures.

Not part of the debate graph -- a single call, not a multi-agent discussion,
and not a scored input to :mod:`trading_agent.features.ipo_gonogo`. NSE and
Groww give real numbers for price, dates, and demand; this fills in the
narrative reading an investor would otherwise have to do themselves over the
financials, issue structure, and risk disclosures those sources already
scraped -- never over data the sources didn't have.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from trading_agent.agents.llm import FoundryClient
from trading_agent.agents.nodes._common import AbstentionError, call_role
from trading_agent.agents.schemas import IPOResearchNote
from trading_agent.agents.state import TokenUsage
from trading_agent.observability import get_logger

__all__ = ["IPOResearchInput", "research_ipo"]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class IPOResearchInput:
    """Exactly the scraped fields the prompt is allowed to reason over."""

    symbol: str
    company_name: str
    sector: str | None
    category: str
    price_band_low: float | None
    price_band_high: float | None
    issue_size_cr: float | None
    issue_structure: str | None
    anchor_shares: int | None
    lead_managers: str | None
    registrar: str | None
    about_company: str | None
    financials: list[dict[str, Any]] = field(default_factory=list)
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)


def _render_prompt(data: IPOResearchInput) -> str:
    lines = [
        f"Symbol: {data.symbol}",
        f"Company: {data.company_name}",
        f"Category: {data.category}",
        f"Sector: {data.sector or 'not disclosed'}",
    ]
    if data.price_band_low is not None and data.price_band_high is not None:
        lines.append(f"Price band: Rs.{data.price_band_low:g} to Rs.{data.price_band_high:g}")
    if data.issue_size_cr is not None:
        lines.append(f"Issue size: Rs.{data.issue_size_cr:.1f} crore")
    lines.append(f"Issue structure (as disclosed): {data.issue_structure or 'not disclosed'}")
    if data.anchor_shares is not None:
        lines.append(f"Anchor portion: {data.anchor_shares:,} equity shares")
    if data.lead_managers:
        lines.append(f"Book running lead manager(s): {data.lead_managers}")
    if data.registrar:
        lines.append(f"Registrar: {data.registrar}")
    if data.about_company:
        lines.append(
            f"\nCompany description and use of proceeds, as disclosed:\n{data.about_company}"
        )

    if data.financials:
        lines.append("\nFinancials (₹ crore, yearly, as disclosed):")
        for series in data.financials:
            yearly = series.get("yearly") or {}
            if yearly:
                figures = ", ".join(f"{y}: {v}" for y, v in sorted(yearly.items()))
                lines.append(f"- {series.get('title', 'Unnamed')}: {figures}")
    else:
        lines.append("\nNo financials were disclosed for this issue.")

    lines.append(
        "\nNo quarterly figures were disclosed for this issue."
        if not any((series.get("quarterly") or {}) for series in data.financials)
        else "\nQuarterly figures were disclosed -- see financials above."
    )

    if data.pros:
        lines.append("\nPros, as disclosed:\n" + "\n".join(f"- {p}" for p in data.pros))
    if data.cons:
        lines.append("\nCons, as disclosed:\n" + "\n".join(f"- {c}" for c in data.cons))

    return "\n".join(lines)


async def research_ipo(
    client: FoundryClient, data: IPOResearchInput
) -> tuple[IPOResearchNote, TokenUsage] | None:
    """Produce one research note, or ``None`` if the role abstained.

    An abstention (content filter, quota, malformed output) is logged and
    skipped -- exactly like a debate node -- rather than raised, so one IPO's
    failure never stops the rest of the sync's issues from getting theirs.
    """
    prompt = _render_prompt(data)
    try:
        note, usage = await call_role(
            client, role="ipo_researcher", user_content=prompt, schema=IPOResearchNote
        )
    except AbstentionError as exc:
        log.warning("ipo_researcher_abstained", symbol=data.symbol, reason=str(exc))
        return None
    return note, usage
