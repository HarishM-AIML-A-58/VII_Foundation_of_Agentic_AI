"""Global kill switch.

A flag checked before **every single order**, not once at startup. Flipping it
must stop new orders within one order's latency, from any process, without a
deploy and without a restart.

It is intentionally fail-safe on the side of *not* trading: if the backing
store is unreachable, :meth:`is_engaged` returns ``True``. A system that
cannot verify it is allowed to trade must not trade. The opposite default --
trading on when Redis is down -- is how a bad afternoon becomes an expensive
one.
"""

from __future__ import annotations

from trading_agent.keyvalue import KeyValueStore
from trading_agent.observability import get_logger

__all__ = ["KillSwitch"]

log = get_logger(__name__)

#: No TTL. A kill switch that quietly expires is not a kill switch.
_NO_EXPIRY = 0


class KillSwitch:
    """Redis-backed stop for all new order flow."""

    __slots__ = ("_fail_closed", "_key", "_store")

    def __init__(
        self,
        store: KeyValueStore,
        *,
        key: str = "trading_agent:kill_switch",
        fail_closed: bool = True,
    ) -> None:
        self._store = store
        self._key = key
        self._fail_closed = fail_closed

    async def is_engaged(self) -> bool:
        try:
            return await self._store.get(self._key) is not None
        except Exception:
            if self._fail_closed:
                log.exception("kill_switch_unreadable", action="treating as ENGAGED")
                return True
            log.exception("kill_switch_unreadable", action="treating as released")
            return False

    async def reason(self) -> str | None:
        try:
            return await self._store.get(self._key)
        except Exception:  # noqa: BLE001
            return "kill switch state could not be read"

    async def engage(self, reason: str) -> None:
        if not reason.strip():
            raise ValueError("engaging the kill switch requires a reason")
        await self._store.set(self._key, reason, ttl_seconds=_NO_EXPIRY)
        log.error("kill_switch_engaged", reason=reason)

    async def release(self) -> None:
        """Release the switch. Deliberately a separate, explicit action."""
        await self._store.delete(self._key)
        log.warning("kill_switch_released")
