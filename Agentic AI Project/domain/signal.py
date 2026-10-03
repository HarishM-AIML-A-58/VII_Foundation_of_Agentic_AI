"""Trade signals -- the contract every layer speaks.

A signal is produced by a strategy or the agent graph, sized by the risk
layer, and consumed by execution. It is the single structure that crosses all
three, so it is defined once here and never redeclared.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from types import MappingProxyType
from uuid import UUID, uuid4

from trading_agent.domain.enums import Action, Exchange, ProductType, SignalStatus
from trading_agent.domain.money import Rupees

__all__ = ["AgentVote", "TradeSignal", "build_idempotency_key"]


def build_idempotency_key(
    *, strategy: str, symbol: str, session_date: date, sequence: int = 0
) -> str:
    """Derive the key that makes order submission safe to retry.

    Deterministic in its inputs: the same strategy proposing the same symbol
    on the same session produces the same key, so a retry after a timeout --
    or a process restart mid-submit -- cannot create a second order.

    ``sequence`` distinguishes deliberate repeat entries on one session; it is
    not a retry counter and must never be incremented on failure.
    """
    if sequence < 0:
        raise ValueError(f"sequence must be non-negative, got {sequence}")
    raw = f"{strategy}:{symbol}:{session_date.isoformat()}:{sequence}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"{strategy}-{symbol}-{session_date.isoformat()}-{digest}"


@dataclass(frozen=True, slots=True)
class AgentVote:
    """One agent's contribution to a debate.

    ``rationale`` is written to the journal and rendered verbatim in the UI's
    debate transcript, so it must be human-readable prose rather than a blob.
    """

    agent: str
    action: Action
    confidence: float
    rationale: str
    #: Foundry deployment that produced this vote, for cost attribution.
    deployment: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if not self.rationale.strip():
            raise ValueError(f"agent {self.agent!r} produced an empty rationale")


@dataclass(frozen=True, slots=True)
class TradeSignal:
    """A proposed trade, fully specified and ready to size or reject."""

    symbol: str
    action: Action
    product: ProductType
    entry: Decimal
    stop_loss: Decimal
    target: Decimal
    generated_at: datetime
    session_date: date
    strategy: str
    idempotency_key: str

    exchange: Exchange = Exchange.NSE
    confidence: float = 0.0
    quantity: int = 0
    risk_amount: Rupees = field(default_factory=Rupees.zero)
    rationale: str = ""
    agent_votes: Mapping[str, AgentVote] = field(default_factory=lambda: MappingProxyType({}))
    status: SignalStatus = SignalStatus.GENERATED
    veto_reason: str | None = None
    signal_id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        if self.generated_at.tzinfo is None:
            raise ValueError("TradeSignal.generated_at must be timezone-aware")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.quantity < 0:
            raise ValueError(f"quantity must be non-negative, got {self.quantity}")

        if self.action.is_actionable:
            if self.entry <= 0:
                raise ValueError(f"entry must be positive, got {self.entry}")
            if self.stop_loss <= 0:
                raise ValueError(f"stop_loss must be positive, got {self.stop_loss}")
            # A stop on the wrong side of entry is not a stop. Catching it here
            # means the risk layer can trust the geometry unconditionally.
            if self.action is Action.BUY and self.stop_loss >= self.entry:
                raise ValueError(
                    f"long stop_loss {self.stop_loss} must sit below entry {self.entry}"
                )
            if self.action is Action.SELL and self.stop_loss <= self.entry:
                raise ValueError(
                    f"short stop_loss {self.stop_loss} must sit above entry {self.entry}"
                )

        # Freeze the votes mapping so a downstream layer cannot mutate the
        # record of what the agents actually said.
        if not isinstance(self.agent_votes, MappingProxyType):
            object.__setattr__(self, "agent_votes", MappingProxyType(dict(self.agent_votes)))

    # -- derived geometry --------------------------------------------------

    @property
    def risk_per_share(self) -> Decimal:
        return abs(self.entry - self.stop_loss)

    @property
    def reward_per_share(self) -> Decimal:
        return abs(self.target - self.entry)

    @property
    def reward_to_risk(self) -> Decimal:
        """Reward-to-risk ratio; zero when the stop sits at the entry."""
        risk = self.risk_per_share
        if risk == 0:
            return Decimal(0)
        return self.reward_per_share / risk

    @property
    def is_vetoed(self) -> bool:
        return self.status is SignalStatus.VETOED

    # -- transitions (always return a new instance) ------------------------

    def vetoed(self, reason: str) -> TradeSignal:
        return self.replace(status=SignalStatus.VETOED, veto_reason=reason)

    def sized(self, *, quantity: int, risk_amount: Rupees) -> TradeSignal:
        return self.replace(quantity=quantity, risk_amount=risk_amount, status=SignalStatus.SIZED)

    def replace(self, **changes: object) -> TradeSignal:
        """Return a copy with ``changes`` applied, re-running validation."""
        current = {f: getattr(self, f) for f in self.__slots__}
        current.update(changes)
        return TradeSignal(**current)
