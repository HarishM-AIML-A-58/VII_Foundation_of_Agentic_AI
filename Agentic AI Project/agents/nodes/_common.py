"""Shared node plumbing.

Kept in one place so the failure semantics are identical for every role: a
node that cannot answer records an abstention and returns, and the graph keeps
going with one fewer voice.
"""

from __future__ import annotations

from typing import Any

from trading_agent.agents.llm import ContentFilteredError, FoundryClient, QuotaExceededError
from trading_agent.agents.prompts import load_prompt
from trading_agent.agents.state import DebateState, TokenUsage
from trading_agent.observability import get_logger

__all__ = ["AbstentionError", "call_role", "render_forensic", "render_opinions"]

log = get_logger(__name__)


class AbstentionError(RuntimeError):
    """A role could not produce a usable answer and is sitting this one out."""


async def call_role(
    client: FoundryClient,
    *,
    role: str,
    user_content: str,
    schema: type[Any],
) -> tuple[Any, TokenUsage]:
    """Invoke ``role`` with its versioned prompt and return validated output.

    Raises :class:`AbstentionError` for every failure a debate can survive --
    content filters, exhausted quota, malformed output. Callers convert that
    into a recorded abstention. Programming errors still propagate.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [SystemMessage(content=load_prompt(role)), HumanMessage(content=user_content)]
    try:
        parsed, (input_tokens, output_tokens) = await client.structured(role, messages, schema)
    except ContentFilteredError as exc:
        raise AbstentionError(f"{role}: content filtered") from exc
    except QuotaExceededError as exc:
        raise AbstentionError(f"{role}: quota exhausted") from exc
    except ValueError as exc:
        raise AbstentionError(f"{role}: malformed output ({exc})") from exc

    deployment = client.deployment_name(role)
    log.info(
        "agent_responded",
        role=role,
        deployment=deployment,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    return parsed, TokenUsage(role, deployment, input_tokens, output_tokens)


def render_opinions(state: DebateState) -> str:
    """Render the analyst reports for a downstream prompt.

    Abstentions are named explicitly rather than omitted: "the sentiment
    analyst did not report" is information, and silently dropping it would let
    a downstream agent read a two-voice consensus as a three-voice one.
    """
    opinions = state.get("opinions", {})
    if not opinions:
        return "No analyst reports are available."

    lines = []
    for role, opinion in sorted(opinions.items()):
        evidence = "; ".join(opinion.key_evidence) if opinion.key_evidence else "none cited"
        lines.append(
            f"### {role.replace('_', ' ')}\n"
            f"stance: {opinion.stance} (confidence {opinion.confidence:.2f})\n"
            f"{opinion.summary}\n"
            f"evidence: {evidence}"
        )

    for missing in state.get("abstentions", []):
        lines.append(f"### {missing}\nDID NOT REPORT -- discount the consensus accordingly.")

    return "\n\n".join(lines)


def render_forensic(state: DebateState) -> str:
    """Render the forensic skeptic's audit for the moderator.

    Absence is named. A missing auditor is not a clean bill of health.
    """
    audit = state.get("forensic_audit")
    if audit is None:
        if "forensic_skeptic" in state.get("abstentions", []):
            return (
                "The forensic skeptic abstained. Do not treat the books as clean; "
                "the auditor did not report."
            )
        return "No forensic audit is available."
    flags = "; ".join(audit.red_flags) if audit.red_flags else "none cited"
    triggers = (
        "; ".join(audit.invalidation_triggers) if audit.invalidation_triggers else "none stated"
    )
    return (
        f"manipulation_risk: {audit.manipulation_risk}\n"
        f"solvency_zone: {audit.solvency_zone}\n"
        f"forensic_score: {audit.forensic_score:.1f}/100 (confidence {audit.confidence:.2f})\n"
        f"{audit.summary}\n"
        f"red_flags: {flags}\n"
        f"invalidation_triggers: {triggers}"
    )
