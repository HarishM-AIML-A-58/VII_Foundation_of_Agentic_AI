"""Replaying agent verdicts through a backtest, so the agent layer is measurable.

**The gap this closes.** ARCHITECTURE.md states the Phase 3 exit gate as "agent
filter improves out-of-sample Sharpe". Until now nothing in ``backtest/`` or
``strategy/`` imported ``agents`` at all, so that gate could not be evaluated
even in principle: the debate existed only on the live path, where the first
measurement of its value would have been taken with real money.

**Why a replay and not a live call.** Running seven agents over a decade of
sessions costs more than the strategy could plausibly earn, and it would not be
reproducible -- the same session would score differently on two runs. So this
consumes *recorded* verdicts. A verdict is produced once, cached, and replayed
deterministically as many times as the research needs.

**What it measures, precisely.** The agent is used as a **filter**, never as a
generator: it may veto a signal the quantitative strategy already produced, and
it may not invent one. Two reasons, and the second is the binding one.

The measurement reason: a filter's contribution is a clean difference. Run the
same strategy over the same panel twice, once with the filter and once without,
and the delta is the agent's entire effect. A generator shares no baseline with
anything and its output cannot be attributed.

The regulatory reason: under SEBI's retail algo framework an algorithm whose
logic is not disclosed and not replicable is black-box, carrying registration and
documentation duties, with re-registration required whenever the logic materially
changes -- and a prompt edit is a logic change. Keeping the model out of the
order path leaves the *deployed* algorithm white-box. A veto is still part of the
order decision, so a filter proven useful here is evidence for a research tool,
not a licence to arm it.

**Absence is not approval.** A session with no recorded verdict for a symbol is
reported as unknown, and the caller chooses the policy: ``AbsentVerdict.BLOCK``
treats silence as a veto, ``ALLOW`` lets the signal through. Neither is a
default worth hiding, because the choice changes what the measurement means --
``ALLOW`` measures the agent on the subset it happened to cover, ``BLOCK``
measures a system that trades only what the agent looked at.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum

from trading_agent.domain.enums import Action
from trading_agent.observability import get_logger

__all__ = [
    "AbsentVerdict",
    "AgentFilter",
    "FilterOutcome",
    "RecordedVerdict",
]

log = get_logger(__name__)


class AbsentVerdict(StrEnum):
    """What to do when no verdict was recorded for a symbol on a session."""

    #: Silence blocks the trade. Measures a system that trades only what the
    #: agent examined.
    BLOCK = "block"
    #: Silence lets the trade through. Measures the agent on the subset of
    #: sessions it happened to cover, which is the more flattering reading and
    #: the one to state out loud when quoting a result.
    ALLOW = "allow"


@dataclass(frozen=True, slots=True)
class RecordedVerdict:
    """One agent debate's conclusion, as replayed.

    ``action`` is the moderator's verdict. ``confidence`` is carried so a filter
    can require more than a bare direction -- an agent that is right but
    uncertain should not be given the same weight as one that is right and sure.
    """

    symbol: str
    session_date: date
    action: Action
    confidence: float = 0.0
    #: Roles that abstained. A verdict reached with three of seven voices
    #: missing is a different object from a full debate, and a filter that
    #: cannot see the difference cannot discount it.
    abstentions: tuple[str, ...] = ()

    @property
    def is_supportive(self) -> bool:
        """True when the debate concluded in favour of a long."""
        return self.action is Action.BUY


@dataclass(slots=True)
class FilterOutcome:
    """What the filter did across a run, so its effect is attributable."""

    allowed: int = 0
    blocked_by_verdict: int = 0
    blocked_by_confidence: int = 0
    blocked_as_absent: int = 0
    allowed_as_absent: int = 0
    #: symbol -> times blocked. Named so a systematic disagreement about one
    #: name is visible rather than averaged away.
    blocked_symbols: dict[str, int] = field(default_factory=dict)

    @property
    def considered(self) -> int:
        return (
            self.allowed
            + self.blocked_by_verdict
            + self.blocked_by_confidence
            + self.blocked_as_absent
            + self.allowed_as_absent
        )

    @property
    def blocked(self) -> int:
        return self.blocked_by_verdict + self.blocked_by_confidence + self.blocked_as_absent

    @property
    def coverage(self) -> float:
        """Fraction of decisions the agent actually had an opinion on.

        The number that decides whether a result means anything. A filter with
        5% coverage that improves Sharpe has improved nothing measurable.
        """
        if self.considered == 0:
            return 0.0
        absent = self.blocked_as_absent + self.allowed_as_absent
        return (self.considered - absent) / self.considered

    def summary(self) -> str:
        return (
            f"{self.allowed + self.allowed_as_absent} allowed, {self.blocked} blocked "
            f"of {self.considered} considered; agent covered {self.coverage:.0%}"
        )


class AgentFilter:
    """Replays recorded verdicts as an entry gate for the backtest engine.

    Built to be passed straight to ``run_backtest(is_eligible=...)``, which
    already applies a ``(symbol, session) -> bool`` predicate to every entry. The
    engine needs no knowledge of agents, and the layering contract holds:
    ``backtest`` never imports ``agents``, it just takes a predicate.
    """

    __slots__ = ("_absent", "_min_confidence", "_outcome", "_verdicts")

    def __init__(
        self,
        verdicts: list[RecordedVerdict],
        *,
        absent: AbsentVerdict = AbsentVerdict.ALLOW,
        min_confidence: float = 0.0,
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(f"min_confidence must be in [0, 1], got {min_confidence}")
        self._verdicts = {(v.symbol.upper(), v.session_date): v for v in verdicts}
        self._absent = absent
        self._min_confidence = min_confidence
        self._outcome = FilterOutcome()

    @property
    def outcome(self) -> FilterOutcome:
        return self._outcome

    def verdict_for(self, symbol: str, on: date) -> RecordedVerdict | None:
        return self._verdicts.get((symbol.upper(), on))

    def allows(self, symbol: str, on: date) -> bool:
        """The predicate. Records why, so the effect can be attributed later."""
        verdict = self.verdict_for(symbol, on)

        if verdict is None:
            if self._absent is AbsentVerdict.BLOCK:
                self._outcome.blocked_as_absent += 1
                self._note_block(symbol)
                return False
            self._outcome.allowed_as_absent += 1
            return True

        if not verdict.is_supportive:
            self._outcome.blocked_by_verdict += 1
            self._note_block(symbol)
            return False

        if verdict.confidence < self._min_confidence:
            self._outcome.blocked_by_confidence += 1
            self._note_block(symbol)
            return False

        self._outcome.allowed += 1
        return True

    def _note_block(self, symbol: str) -> None:
        key = symbol.upper()
        self._outcome.blocked_symbols[key] = self._outcome.blocked_symbols.get(key, 0) + 1

    def combined_with(self, other: Callable[[str, date], bool]) -> Callable[[str, date], bool]:
        """Compose with another predicate, e.g. index membership.

        Order matters for the bookkeeping: ``other`` is checked first, so a
        symbol that was not an index member does not count as an agent decision.
        Counting it would dilute ``coverage`` with sessions the agent was never
        asked about.
        """

        def predicate(symbol: str, on: date) -> bool:
            return other(symbol, on) and self.allows(symbol, on)

        return predicate
