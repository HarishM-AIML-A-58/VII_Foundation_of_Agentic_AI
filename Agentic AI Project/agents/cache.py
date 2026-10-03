"""Content-addressed cache for Foundry responses.

Two jobs, and the second matters more than the cost saving:

1. **Spend.** Re-running research over the same session re-bills every call.
   One symbol, one round is ~17k tokens; an evaluation sweep over months of
   history without a cache is unaffordable.
2. **Determinism.** Agent behaviour is not reproducible by default -- the
   GPT-5 reasoning family will not even accept `temperature=0`. Replaying
   recorded responses is what makes an agent test deterministic, and it is
   why we did not need temperature control for reproducibility.

Keys are content-addressed over (deployment, schema, messages). Any change to
the prompt, the schema or the model produces a different key, so a stale entry
cannot silently answer a question it was not asked.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from trading_agent.keyvalue import KeyValueStore
from trading_agent.observability import get_logger
from trading_agent.settings import Settings

__all__ = ["ResponseCache", "cache_from_settings", "cache_key"]

log = get_logger(__name__)


def cache_key(*, deployment: str, schema_name: str, messages: list[Any]) -> str:
    """Stable key over everything that can change the answer.

    Deliberately includes the schema name: the same prompt under a changed
    output schema is a different question, and reusing the old answer would
    return an object that no longer validates.
    """
    payload = json.dumps(
        {
            "deployment": deployment,
            "schema": schema_name,
            "messages": [
                {"type": type(m).__name__, "content": str(getattr(m, "content", m))}
                for m in messages
            ],
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


class ResponseCache:
    """Caches validated agent outputs.

    Stores the *validated* object's JSON, not the raw completion: a cache hit
    then costs one `model_validate` rather than a re-parse, and a schema
    change invalidates the entry by construction.
    """

    __slots__ = ("_enabled", "_hits", "_misses", "_namespace", "_store", "_ttl")

    def __init__(
        self,
        store: KeyValueStore,
        *,
        ttl_seconds: int = 86_400,
        namespace: str = "trading_agent:llm",
        enabled: bool = True,
    ) -> None:
        self._store = store
        self._ttl = ttl_seconds
        self._namespace = namespace
        self._enabled = enabled
        self._hits = 0
        self._misses = 0

    @property
    def stats(self) -> dict[str, int]:
        return {"hits": self._hits, "misses": self._misses}

    async def get(self, key: str, schema: type[Any]) -> Any | None:
        """Return the cached object, or None.

        A corrupt entry, a schema mismatch, or an unreachable backend are all
        treated as a miss rather than an error: a bad cache must never be able
        to stop research. Losing the cache costs money; letting it raise
        cancels the session.
        """
        if not self._enabled:
            return None
        try:
            raw = await self._store.get(f"{self._namespace}:{key}")
        except Exception:  # noqa: BLE001 -- an unreachable cache is a miss
            log.warning("cache_unavailable", operation="get", key=key)
            self._misses += 1
            return None
        if raw is None:
            self._misses += 1
            return None
        try:
            value = schema.model_validate_json(raw)
        except Exception:  # noqa: BLE001 -- a bad entry is a miss, never fatal
            log.warning("cache_entry_unusable", key=key, schema=schema.__name__)
            self._misses += 1
            return None
        self._hits += 1
        return value

    async def set(self, key: str, value: Any) -> None:
        if not self._enabled:
            return
        try:
            payload = value.model_dump_json()
        except AttributeError:
            return
        try:
            await self._store.set(f"{self._namespace}:{key}", payload, ttl_seconds=self._ttl)
        except Exception:  # noqa: BLE001 -- failing to memoise is not a failure
            log.warning("cache_unavailable", operation="set", key=key)


def cache_from_settings(settings: Settings) -> ResponseCache:
    """Build the Redis-backed cache described by ``settings``.

    Returns a disabled cache rather than ``None`` when caching is switched
    off, so callers have one code path: a disabled cache answers every ``get``
    with a miss and drops every ``set``.
    """
    from trading_agent.keyvalue import RedisKeyValueStore

    return ResponseCache(
        RedisKeyValueStore.from_url(settings.redis.url.get_secret_value()),
        ttl_seconds=settings.foundry.cache.ttl_seconds,
        enabled=settings.foundry.cache.enabled,
    )
