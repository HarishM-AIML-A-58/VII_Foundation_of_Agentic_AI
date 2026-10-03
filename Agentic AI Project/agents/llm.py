"""Microsoft Foundry client (formerly Azure AI Foundry).

Foundry is reached over its OpenAI-compatible surface via ``langchain-openai``
rather than ``langchain-azure-ai``; see ADR-0003 for why.

Auth is an API key by default. Entra ID is supported but opt-in, because it
needs a *data-plane* role ("Cognitive Services User") that subscription Owner
does not confer -- so a key is the only credential that works on a fresh
subscription without a separate role assignment.

Three things Foundry forces on any caller, all handled here:

1. **Tiering is by deployment, not model.** Roles resolve to a deployment
   name in config -- one shared default, optionally overridden per role -- so
   the code never hardcodes a model.
2. **Quota is per deployment.** A seven-agent debate over a ten-symbol
   shortlist will hit 429s, so concurrency is bounded per deployment and
   retries honour ``Retry-After`` instead of guessing a backoff. Note that
   with one shared deployment this bound applies to the whole debate.
3. **Content filters return a finish reason, not an exception.** Callers must
   degrade to an abstention rather than crash mid-session.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from trading_agent.agents.cache import ResponseCache, cache_key
from trading_agent.observability import get_logger
from trading_agent.settings import FoundrySettings

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

__all__ = [
    "AGENT_ROLES",
    "ConnectivityResult",
    "ContentFilteredError",
    "FoundryClient",
    "QuotaExceededError",
]

log = get_logger(__name__)

#: Every role the debate graph calls a model for, in debate order.
#:
#: Declared here rather than derived from `foundry.deployments`, which is now
#: an optional override map: with one shared deployment it is empty, so it can
#: no longer serve as the list of roles to check.
AGENT_ROLES: Final[tuple[str, ...]] = (
    "technical_analyst",
    "fundamental_analyst",
    "sentiment_analyst",
    "forensic_skeptic",
    "bull_researcher",
    "bear_researcher",
    "debate_moderator",
    "risk_manager",
)

#: Entra ID scope for Azure AI / Cognitive Services data-plane calls.
_TOKEN_SCOPE = "https://cognitiveservices.azure.com/.default"  # noqa: S105 -- OAuth scope, not a secret


class QuotaExceededError(RuntimeError):
    """Foundry returned 429 and the retry budget was exhausted."""


class ContentFilteredError(RuntimeError):
    """The response was blocked by Foundry's content filter.

    Callers treat this as an abstention (HOLD), never as a trading signal.
    """


def _token_counts(raw: Any) -> tuple[int, int]:
    """Extract (input, output) token counts, tolerating provider differences.

    Cost attribution should never be the reason a trading decision fails, so
    an unreadable usage block degrades to (0, 0) rather than raising.
    """
    if raw is None:
        return (0, 0)
    usage = getattr(raw, "usage_metadata", None) or {}
    if isinstance(usage, dict) and usage:
        return (int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0)))
    metadata = getattr(raw, "response_metadata", {}) or {}
    token_usage = metadata.get("token_usage") or {}
    return (
        int(token_usage.get("prompt_tokens", 0)),
        int(token_usage.get("completion_tokens", 0)),
    )


@dataclass(frozen=True, slots=True)
class ConnectivityResult:
    role: str
    deployment: str
    ok: bool
    latency_seconds: float = 0.0
    detail: str = ""


class FoundryClient:
    """Builds and calls per-role chat models against Foundry.

    Pass a :class:`~trading_agent.agents.cache.ResponseCache` to make repeated
    research free and deterministic. Without one every call bills, which makes
    an evaluation sweep over months of history unaffordable and makes agent
    behaviour irreproducible -- the GPT-5 reasoning family will not accept
    ``temperature=0``, so replay is the only source of determinism there is.
    """

    __slots__ = ("_cache", "_models", "_semaphores", "_settings", "_token_provider")

    def __init__(self, settings: FoundrySettings, *, cache: ResponseCache | None = None) -> None:
        self._settings = settings
        self._cache = cache
        self._models: dict[str, BaseChatModel] = {}
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._token_provider: Any | None = None

    @property
    def cache_stats(self) -> dict[str, int]:
        """Hits and misses, or zeroes when no cache is attached."""
        return self._cache.stats if self._cache is not None else {"hits": 0, "misses": 0}

    # ------------------------------------------------------------ credentials

    def _credential(self) -> tuple[Any | None, str | None]:
        """Return ``(token_provider, api_key)``; exactly one is populated.

        API key is the default path. Entra ID is opt-in via
        ``foundry.use_entra_id`` and needs a data-plane role on the account --
        subscription Owner is not sufficient, which is why it is not the
        default.
        """
        if not self._settings.use_entra_id:
            key = self._settings.api_key.get_secret_value()
            if not key:
                raise ValueError(
                    "No Foundry API key is set. Put your key in "
                    "TRADING_AGENT__FOUNDRY__API_KEY, or set "
                    "TRADING_AGENT__FOUNDRY__USE_ENTRA_ID=true to authenticate with "
                    "`az login` / managed identity instead."
                )
            return None, key

        if self._token_provider is None:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider

            # Resolves managed identity in production and `az login` locally.
            # Requires "Cognitive Services User" (or equivalent) on the
            # account: the data actions are not implied by Owner.
            self._token_provider = get_bearer_token_provider(DefaultAzureCredential(), _TOKEN_SCOPE)
        return self._token_provider, None

    # ---------------------------------------------------------------- models

    def model_for(self, role: str) -> BaseChatModel:
        """The chat model for ``role``, built once and reused."""
        if role in self._models:
            return self._models[role]

        from langchain_openai import AzureChatOpenAI

        deployment = self._settings.deployment_for(role)
        token_provider, api_key = self._credential()

        kwargs: dict[str, Any] = {
            "azure_endpoint": self._settings.project_endpoint,
            "azure_deployment": deployment,
            # Azure OpenAI models take the deployment from the URL path and
            # ignore this field. Foundry CATALOG models (Grok, DeepSeek, Kimi)
            # are served by a different backend that requires `model` in the
            # request body and rejects the null that AzureChatOpenAI would
            # otherwise send, failing with an invalid-type deserialisation
            # error. Setting it to the deployment name satisfies both.
            "model": deployment,
            "api_version": self._settings.api_version,
            "max_tokens": self._settings.max_tokens,
            "timeout": self._settings.timeout_seconds,
            # Retries are handled here, with Retry-After honoured; the SDK's
            # own blind backoff would double-count the wait.
            "max_retries": 0,
            # Streaming responses omit token usage unless asked; without it a
            # chat turn would bill Foundry and record nothing.
            "stream_usage": True,
        }
        # Sent only when configured: the GPT-5 reasoning family returns 400
        # for any explicit temperature, including the value it would use.
        temperature = self._settings.temperature_for(role)
        if temperature is not None:
            kwargs["temperature"] = temperature

        if token_provider is not None:
            kwargs["azure_ad_token_provider"] = token_provider
        else:
            kwargs["api_key"] = api_key

        model: BaseChatModel = AzureChatOpenAI(**kwargs)
        self._models[role] = model
        return model

    def deployment_name(self, role: str) -> str:
        """Deployment backing ``role``. Used for logging and cost attribution."""
        return self._settings.deployment_for(role)

    def _semaphore_for(self, deployment: str) -> asyncio.Semaphore:
        """One semaphore per deployment: quota is per deployment, so two roles
        sharing a deployment must share the limit."""
        if deployment not in self._semaphores:
            self._semaphores[deployment] = asyncio.Semaphore(
                self._settings.max_concurrent_per_deployment
            )
        return self._semaphores[deployment]

    # ----------------------------------------------------------------- calls

    async def structured(
        self, role: str, messages: list[Any], schema: type[Any]
    ) -> tuple[Any, tuple[int, int]]:
        """Call ``role`` and return ``(validated_object, (input, output) tokens)``.

        Structured output is the whole point: the model is constrained to the
        schema, so the decision arrives typed. The reference implementations
        return prose and spend a second LLM call extracting "BUY/SELL/HOLD"
        from it -- fragile, billable, and untyped.

        A cache hit reports ``(0, 0)`` tokens, because that is what it cost.
        Attributing the tokens the call *would* have spent would make the
        Foundry bill and the recorded usage disagree.
        """
        deployment = self._settings.deployment_for(role)

        # Checked before the semaphore: a cache hit consumes no quota, so
        # queueing it behind in-flight calls would serialise free work.
        key = cache_key(deployment=deployment, schema_name=schema.__name__, messages=messages)
        if self._cache is not None:
            cached = await self._cache.get(key, schema)
            if cached is not None:
                log.debug("foundry_cache_hit", role=role, deployment=deployment)
                return cached, (0, 0)

        model = self.model_for(role).with_structured_output(schema, include_raw=True)
        retry = self._settings.retry

        async with self._semaphore_for(deployment):
            backoff = retry.initial_backoff_seconds
            last_error: Exception | None = None

            for attempt in range(1, retry.max_attempts + 1):
                try:
                    response = await model.ainvoke(messages)
                except Exception as exc:
                    last_error = exc
                    delay = self._retry_delay(exc, backoff, retry.respect_retry_after)
                    if delay is None:
                        raise
                    log.warning(
                        "foundry_retry",
                        role=role,
                        deployment=deployment,
                        attempt=attempt,
                        error=type(exc).__name__,
                    )
                    await asyncio.sleep(delay)
                    backoff = min(backoff * 2, retry.max_backoff_seconds)
                    continue

                raw = response.get("raw") if isinstance(response, dict) else None
                parsed = response.get("parsed") if isinstance(response, dict) else response

                if raw is not None:
                    self._raise_if_filtered(raw, role=role)
                if parsed is None:
                    # The model answered but not in the schema. Treated as an
                    # abstention by callers, never guessed at.
                    error = response.get("parsing_error") if isinstance(response, dict) else None
                    raise ValueError(
                        f"role {role!r} returned no schema-valid output; error: {error}"
                    )
                if self._cache is not None:
                    # Only validated objects are cached. A malformed response
                    # must not be replayable -- it would fail identically on
                    # every future run, for free, and look like a model fault.
                    await self._cache.set(key, parsed)
                return parsed, _token_counts(raw)

            raise QuotaExceededError(
                f"deployment {deployment!r} (role {role!r}) still rate-limited after "
                f"{retry.max_attempts} attempts"
            ) from last_error

    async def invoke(self, role: str, messages: list[Any]) -> Any:
        """Call ``role``'s deployment with bounded concurrency and retries.

        Raises :class:`QuotaExceededError` when the retry budget is spent and
        :class:`ContentFilteredError` when the filter blocked the response.
        """
        deployment = self._settings.deployment_for(role)
        model = self.model_for(role)
        retry = self._settings.retry

        async with self._semaphore_for(deployment):
            backoff = retry.initial_backoff_seconds
            last_error: Exception | None = None

            for attempt in range(1, retry.max_attempts + 1):
                try:
                    response = await model.ainvoke(messages)
                except Exception as exc:
                    last_error = exc
                    delay = self._retry_delay(exc, backoff, retry.respect_retry_after)
                    if delay is None:
                        raise
                    log.warning(
                        "foundry_retry",
                        role=role,
                        deployment=deployment,
                        attempt=attempt,
                        max_attempts=retry.max_attempts,
                        sleeping=round(delay, 2),
                        error=type(exc).__name__,
                    )
                    await asyncio.sleep(delay)
                    backoff = min(backoff * 2, retry.max_backoff_seconds)
                    continue

                self._raise_if_filtered(response, role=role)
                return response

            raise QuotaExceededError(
                f"deployment {deployment!r} (role {role!r}) still rate-limited after "
                f"{retry.max_attempts} attempts"
            ) from last_error

    async def stream(
        self, role: str, messages: list[Any], *, tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[Any]:
        """Stream ``role``'s response chunk by chunk, optionally with tools bound.

        Retries only until the first chunk arrives. After that, text has
        already reached the caller, and replaying the request would stream the
        same answer twice -- so a mid-stream failure propagates instead.

        Not cached: a conversational turn depends on everything said before
        it, so a cache keyed on the transcript would almost never hit and
        would hold every user's conversation in Redis for no return.
        """
        deployment = self._settings.deployment_for(role)
        model = self.model_for(role)
        runnable = model.bind_tools(tools) if tools else model
        retry = self._settings.retry

        async with self._semaphore_for(deployment):
            backoff = retry.initial_backoff_seconds
            last_error: Exception | None = None

            for attempt in range(1, retry.max_attempts + 1):
                started = False
                try:
                    async for chunk in runnable.astream(messages):
                        started = True
                        self._raise_if_filtered(chunk, role=role)
                        yield chunk
                except ContentFilteredError:
                    raise
                except Exception as exc:
                    if started:
                        raise
                    last_error = exc
                    delay = self._retry_delay(exc, backoff, retry.respect_retry_after)
                    if delay is None:
                        raise
                    log.warning(
                        "foundry_retry",
                        role=role,
                        deployment=deployment,
                        attempt=attempt,
                        streaming=True,
                        error=type(exc).__name__,
                    )
                    await asyncio.sleep(delay)
                    backoff = min(backoff * 2, retry.max_backoff_seconds)
                else:
                    return

            raise QuotaExceededError(
                f"deployment {deployment!r} (role {role!r}) still rate-limited after "
                f"{retry.max_attempts} attempts"
            ) from last_error

    @staticmethod
    def _retry_delay(exc: Exception, backoff: float, respect_retry_after: bool) -> float | None:
        """Seconds to wait, or ``None`` when the error is not retryable.

        Only throttling and transient server errors are retried. Retrying an
        auth or bad-request failure just burns quota to fail identically.
        """
        status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
        retryable = status in {408, 429, 500, 502, 503, 504}
        if not retryable and "rate limit" not in str(exc).lower():
            return None

        if respect_retry_after:
            headers = getattr(exc, "response_headers", None) or getattr(exc, "headers", None) or {}
            raw = headers.get("retry-after") or headers.get("Retry-After")
            if raw is not None:
                try:
                    # Server-supplied wait beats a guess; it reflects the real
                    # quota window rather than our exponential curve.
                    return float(raw)
                except (TypeError, ValueError):
                    pass

        # Jitter so parallel agents do not retry in lockstep and re-throttle.
        return backoff * (0.5 + random.random())  # noqa: S311 -- jitter, not crypto

    @staticmethod
    def _raise_if_filtered(response: Any, *, role: str) -> None:
        metadata = getattr(response, "response_metadata", {}) or {}
        if metadata.get("finish_reason") == "content_filter":
            raise ContentFilteredError(
                f"Foundry content filter blocked the response for role {role!r}"
            )

    async def aclose(self) -> None:
        """Close the HTTP clients behind every model built so far.

        A short-lived process (the CLI) must call this before its event loop
        ends. Otherwise the last streamed response is closed during garbage
        collection, after the loop is gone, and Python prints a traceback for
        a request that had in fact finished cleanly.
        """
        for model in self._models.values():
            client = getattr(model, "root_async_client", None)
            close = getattr(client, "close", None)
            if close is not None:
                await close()
        self._models.clear()

    # ---------------------------------------------------------- diagnostics

    async def check(self, role: str) -> ConnectivityResult:
        """Send one trivial prompt and report whether it worked."""
        from langchain_core.messages import HumanMessage

        try:
            deployment = self._settings.deployment_for(role)
        except (KeyError, ValueError) as exc:
            return ConnectivityResult(role, "-", ok=False, detail=str(exc))

        started = time.perf_counter()
        try:
            await self.invoke(role, [HumanMessage(content="Reply with the single word: ok")])
        except Exception as exc:  # noqa: BLE001 -- diagnostics report, never raise
            return ConnectivityResult(
                role, deployment, ok=False, detail=f"{type(exc).__name__}: {exc}"[:200]
            )
        return ConnectivityResult(
            role, deployment, ok=True, latency_seconds=time.perf_counter() - started, detail="ok"
        )

    async def check_all(self, roles: list[str]) -> list[ConnectivityResult]:
        """Check every role concurrently, respecting per-deployment limits."""
        return list(await asyncio.gather(*(self.check(role) for role in roles)))
