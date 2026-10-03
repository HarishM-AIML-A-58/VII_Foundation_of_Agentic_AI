"""The research chat loop: plan, call specialist tools, synthesise, suggest.

One lead model on Foundry routes to the tools in :mod:`.tools`, exactly the
shape of the superinvesting.ai orchestrator (a single planner with tool calls,
not a LangGraph of sub-models): each "agent" is deterministic arithmetic, and
only the planner and the prose cost tokens.

:meth:`ResearchChat.stream` yields typed events the UI renders as a stepper:

* ``step_start`` / ``step_finish`` -- one per tool call, carrying the model's
  own progress label;
* ``text`` -- answer tokens as they stream;
* ``suggestions`` -- the three follow-up pills;
* ``error`` and ``done``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Final, Literal

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.utils.function_calling import convert_to_openai_tool

from trading_agent.agents.llm import ContentFilteredError, FoundryClient, QuotaExceededError
from trading_agent.agents.prompts import load_prompt
from trading_agent.agents.research_chat.tools import AGENT_LABELS, TOOL_ARGS, ResearchToolkit
from trading_agent.observability import get_logger

__all__ = ["ChatTurn", "ResearchChat", "tool_definitions"]

log = get_logger(__name__)

ROLE: Final = "research_chat"

#: Planner rounds before the model is made to answer with what it has. Five
#: covers profile -> compare -> forensic -> news -> write with room to spare;
#: a loop that needs more is usually a loop that is stuck.
MAX_TOOL_ROUNDS: Final = 6

#: History sent back to the model. Older turns cost tokens on every request
#: and rarely change the answer to the newest question.
MAX_HISTORY_TURNS: Final = 16

#: A tool result larger than this is cut. The model reads a truncated table
#: fine; it does not read a 60k-token one at all well.
MAX_TOOL_RESULT_CHARS: Final = 14_000


@dataclass(frozen=True, slots=True)
class _Call:
    id: str
    name: str
    args: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ChatTurn:
    role: Literal["user", "assistant"]
    content: str


def tool_definitions() -> list[dict[str, Any]]:
    """OpenAI function definitions for every tool, named as the UI expects."""
    definitions = []
    for name, schema in TOOL_ARGS.items():
        tool = convert_to_openai_tool(schema)
        tool["function"]["name"] = name
        definitions.append(tool)
    return definitions


def _text_of(content: Any) -> str:
    """Chunk content as plain text; reasoning models may send content parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return ""


def _summarise(name: str, result: dict[str, Any]) -> str:
    """One line for the stepper: what the tool found, or why it failed."""
    if "error" in result:
        return str(result["error"])[:140]
    if name == "screener":
        if result.get("mode") == "screen":
            return f"{result.get('matches', 0)} matches"
        return f"{len(result.get('profiles', []))} profile(s)"
    if name == "browser":
        return f"{result.get('matched', 0)} news items"
    if name == "comparison":
        return f"{len(result.get('companies', []))} companies benchmarked"
    if name == "persona":
        return f"{len(result.get('cards', []))} persona card(s)"
    if name == "memory":
        return f"{len(result.get('hits') or [])} ledger hits"
    if name == "research":
        return (
            f"{len(result.get('flags', []))} flag(s) across {len(result.get('checks', []))} checks"
        )
    if name == "sector":
        return f"{result.get('companies', 0)} companies"
    if name == "portfolio":
        if "baskets" in result:
            return f"{len(result['baskets'])} baskets"
        if "basket" in result:
            return f"{result.get('constituents', 0)} constituents"
        if result.get("connected") is False:
            return "broker not connected"
        return f"{result.get('positions', 0)} positions"
    return "done"


class ResearchChat:
    """Runs one conversational turn against Foundry and the toolkit."""

    def __init__(
        self,
        client: FoundryClient,
        toolkit: ResearchToolkit,
        *,
        max_rounds: int = MAX_TOOL_ROUNDS,
    ) -> None:
        self._client = client
        self._toolkit = toolkit
        self._max_rounds = max_rounds
        self._tools = tool_definitions()

    def _messages(self, history: list[ChatTurn]) -> list[BaseMessage]:
        messages: list[BaseMessage] = [SystemMessage(content=load_prompt(ROLE))]
        for turn in history[-MAX_HISTORY_TURNS:]:
            if turn.role == "user":
                messages.append(HumanMessage(content=turn.content))
            else:
                messages.append(AIMessage(content=turn.content))
        return messages

    async def stream(self, history: list[ChatTurn]) -> AsyncIterator[dict[str, Any]]:
        if not history or history[-1].role != "user":
            yield {"type": "error", "message": "The conversation must end with a user message."}
            return

        messages = self._messages(history)
        usage = {"input_tokens": 0, "output_tokens": 0}
        steps = 0
        wrote_text = False

        try:
            for round_number in range(self._max_rounds + 1):
                final_round = round_number == self._max_rounds
                if final_round:
                    messages.append(
                        HumanMessage(
                            content="Tool budget reached. Write the answer now from the "
                            "results above, and name anything left unverified."
                        )
                    )

                response: AIMessageChunk | None = None
                separated = False
                async for chunk in self._client.stream(
                    ROLE, messages, tools=None if final_round else self._tools
                ):
                    delta = _text_of(chunk.content)
                    if delta:
                        if wrote_text and not separated:
                            # Text from an earlier round ("Let me pull the
                            # numbers...") must not run into this round's.
                            yield {"type": "text", "delta": "\n\n"}
                        separated = True
                        wrote_text = True
                        yield {"type": "text", "delta": delta}
                    response = chunk if response is None else response + chunk

                if response is None:
                    break
                if response.usage_metadata:
                    usage["input_tokens"] += response.usage_metadata["input_tokens"]
                    usage["output_tokens"] += response.usage_metadata["output_tokens"]

                # The API always assigns tool-call ids; the SDK types them as
                # optional only because hand-built messages may omit them.
                # The API always assigns tool-call ids; the SDK types them as
                # optional only because hand-built messages may omit them.
                calls = [
                    _Call(id=c["id"] or f"call_{i}", name=c["name"], args=c["args"])
                    for i, c in enumerate(response.tool_calls)
                ]
                messages.append(
                    AIMessage(
                        content=response.content,
                        tool_calls=[ToolCall(id=c.id, name=c.name, args=c.args) for c in calls],
                    )
                )
                if not calls:
                    break

                work = [c for c in calls if c.name != "suggestionTool"]
                results: dict[str, dict[str, Any]] = {}
                for call in calls:
                    if call.name == "suggestionTool":
                        suggestions = call.args.get("suggestions") or []
                        yield {"type": "suggestions", "items": [str(s) for s in suggestions][:3]}
                        results[call.id] = {"ok": True}

                for call in work:
                    steps += 1
                    yield {
                        "type": "step_start",
                        "id": call.id,
                        "tool": call.name,
                        "label": AGENT_LABELS.get(call.name, call.name),
                        "message": str(call.args.get("message") or f"Running {call.name}"),
                    }

                outcomes = await asyncio.gather(
                    *(self._toolkit.run(call.name, call.args) for call in work)
                )
                for call, outcome in zip(work, outcomes, strict=True):
                    results[call.id] = outcome
                    yield {
                        "type": "step_finish",
                        "id": call.id,
                        "tool": call.name,
                        "ok": "error" not in outcome,
                        "summary": _summarise(call.name, outcome),
                    }

                for call in calls:
                    payload = json.dumps(results[call.id], ensure_ascii=False, default=str)
                    if len(payload) > MAX_TOOL_RESULT_CHARS:
                        payload = (
                            payload[:MAX_TOOL_RESULT_CHARS]
                            + "... [truncated; narrow the request for the rest]"
                        )
                    messages.append(ToolMessage(content=payload, tool_call_id=call.id))

                if not work and wrote_text:
                    # Only the follow-up pills were requested, after the
                    # answer: the turn is complete.
                    break

        except ContentFilteredError:
            yield {"type": "error", "message": "The response was blocked by the content filter."}
        except QuotaExceededError:
            yield {
                "type": "error",
                "message": "The model deployment is rate-limited. Try again shortly.",
            }
        except (KeyError, ValueError) as exc:
            # Unconfigured deployment or missing key: say which, it is fixable.
            log.warning("research_chat_unconfigured", error=str(exc)[:300])
            yield {"type": "error", "message": str(exc)[:300]}
        except Exception as exc:  # a stream must end with an event, not a hang
            log.exception("research_chat_failed")
            yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"[:300]}

        log.info("research_chat_turn", steps=steps, **usage)
        yield {"type": "done", "steps": steps, "usage": usage}
