"""Post-close review.

Writes the day's journal and runs the reflection loop that compares what each
agent predicted against what happened. This is what turns a static prompt set
into something that can be calibrated.

The measurement is deliberately harsh in three ways, because a review that
flatters the system is worse than no review:

**Vetoed signals are scored too.** They are the only counterfactual available.
A veto rule that consistently blocks winners is costing money silently, and
the only way to see it is to grade the trades that never happened.

**Confidence is scored, not just direction.** An agent that is right 55% of
the time and says 0.9 every time is badly calibrated even though it is
profitable, and the calibration gap is the thing a prompt change can fix.
Being right for the wrong reason is not a repeatable edge.

**The benchmark is the same-day index move, not zero.** A long that gained 1%
on a day the index gained 2% was a bad call, and grading it against zero would
record it as a win.

Nothing here changes a prompt automatically. It produces the evidence; a human
decides. An agent system that rewrites its own instructions based on one
session's P&L will chase noise faster than any human could.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal

import pandas as pd

from trading_agent.domain import Action, SignalStatus, TradeSignal
from trading_agent.market_data import BarStore
from trading_agent.observability import get_logger
from trading_agent.persistence import Database, JournalEntry, JournalRepository, SignalRepository
from trading_agent.settings import Settings

__all__ = [
    "AgentScorecard",
    "PostCloseReview",
    "SessionReview",
    "SignalOutcome",
    "run_post_close_review",
]

log = get_logger(__name__)

#: Move, in percent, below which a session is called flat rather than
#: directional. Grading a 0.02% drift as a "correct bullish call" is noise
#: dressed as skill.
FLAT_THRESHOLD_PCT = Decimal("0.15")


@dataclass(frozen=True, slots=True)
class SignalOutcome:
    """What one signal predicted, and what the market did next."""

    symbol: str
    action: Action
    confidence: float
    status: SignalStatus
    entry: Decimal
    #: Close of the session after the signal. None when it has not happened
    #: yet -- a review run the same evening cannot grade itself.
    realised_close: Decimal | None
    move_pct: Decimal | None
    benchmark_move_pct: Decimal | None
    signal_id: str
    agents: dict[str, Action] = field(default_factory=dict)

    @property
    def is_gradeable(self) -> bool:
        return self.move_pct is not None

    @property
    def excess_pct(self) -> Decimal | None:
        """Move net of the index. The number that says whether it was skill."""
        if self.move_pct is None:
            return None
        if self.benchmark_move_pct is None:
            return self.move_pct
        return self.move_pct - self.benchmark_move_pct

    @property
    def correct(self) -> bool | None:
        """Whether the direction was right, measured against the index.

        ``None`` when the session was flat: neither side was wrong, and
        counting a flat day as a win for whoever guessed inflates every
        hit rate in the report.
        """
        excess = self.excess_pct
        if excess is None:
            return None
        if abs(excess) < FLAT_THRESHOLD_PCT:
            return None
        if self.action is Action.BUY:
            return excess > 0
        if self.action is Action.SELL:
            return excess < 0
        return None

    @property
    def was_taken(self) -> bool:
        return self.status is not SignalStatus.VETOED


@dataclass(slots=True)
class AgentScorecard:
    """One agent's record over the session.

    Kept per agent rather than per debate because "the debate was 60% right"
    is not actionable, and "the sentiment analyst is 35% right and says 0.8"
    names a prompt to change.
    """

    agent: str
    graded: int = 0
    correct: int = 0
    confidence_sum: float = 0.0

    @property
    def hit_rate(self) -> float:
        return self.correct / self.graded if self.graded else 0.0

    @property
    def mean_confidence(self) -> float:
        return self.confidence_sum / self.graded if self.graded else 0.0

    @property
    def calibration_gap(self) -> float:
        """Stated confidence minus realised hit rate.

        Positive is overconfidence, which is the dangerous direction: the risk
        layer weights by confidence, so an agent that overstates it is sizing
        up exactly the trades it is worst at.
        """
        return self.mean_confidence - self.hit_rate


@dataclass(slots=True)
class SessionReview:
    """The day's report."""

    session_date: date
    outcomes: list[SignalOutcome] = field(default_factory=list)
    scorecards: dict[str, AgentScorecard] = field(default_factory=dict)
    pending: int = 0

    @property
    def graded(self) -> list[SignalOutcome]:
        return [o for o in self.outcomes if o.correct is not None]

    @property
    def taken(self) -> list[SignalOutcome]:
        return [o for o in self.graded if o.was_taken]

    @property
    def vetoed(self) -> list[SignalOutcome]:
        return [o for o in self.graded if not o.was_taken]

    @property
    def hit_rate(self) -> float:
        graded = self.graded
        return sum(1 for o in graded if o.correct) / len(graded) if graded else 0.0

    @property
    def veto_cost_pct(self) -> Decimal:
        """Excess return the vetoes gave up, summed.

        Positive means the veto rules blocked trades that would have worked.
        Persistently positive is a reason to revisit the thresholds -- not to
        remove them, since the veto also blocks the tail that would end the
        account.
        """
        total = Decimal(0)
        for outcome in self.vetoed:
            excess = outcome.excess_pct
            if excess is None:
                continue
            total += excess if outcome.action is Action.BUY else -excess
        return total

    @property
    def worst_calibrated(self) -> AgentScorecard | None:
        graded = [s for s in self.scorecards.values() if s.graded]
        return max(graded, key=lambda s: s.calibration_gap) if graded else None

    def summary_rows(self) -> list[tuple[str, str]]:
        return [
            ("Session", self.session_date.isoformat()),
            ("Signals reviewed", str(len(self.outcomes))),
            ("Gradeable", str(len(self.graded))),
            ("Awaiting the next session", str(self.pending)),
            ("Hit rate (excess of index)", f"{self.hit_rate:.1%}"),
            ("Taken / vetoed", f"{len(self.taken)} / {len(self.vetoed)}"),
            ("Veto cost", f"{self.veto_cost_pct:+.2f}%"),
        ]


class PostCloseReview:
    """Grades a session's signals against what the market then did."""

    __slots__ = ("_benchmark", "_database", "_settings", "_store")

    def __init__(
        self,
        *,
        store: BarStore,
        database: Database,
        settings: Settings,
        benchmark_symbol: str | None = None,
    ) -> None:
        self._store = store
        self._database = database
        self._settings = settings
        self._benchmark = benchmark_symbol or settings.backtest.benchmark_symbol

    async def run(self, *, on: date) -> SessionReview:
        """Review the signals generated for ``on``."""
        async with self._database.session() as session:
            signals = list(await SignalRepository(session).list_for_session(on))

        review = SessionReview(session_date=on)
        if not signals:
            log.info("review_empty", session=on.isoformat())
            return review

        benchmark_move = self._move_after(self._benchmark, on)

        for signal in signals:
            outcome = self._grade(signal, on, benchmark_move)
            review.outcomes.append(outcome)
            if not outcome.is_gradeable:
                review.pending += 1
                continue
            _score(review.scorecards, outcome)

        await self._journal(review)
        log.info(
            "review_complete",
            session=on.isoformat(),
            reviewed=len(review.outcomes),
            graded=len(review.graded),
            pending=review.pending,
            hit_rate=round(review.hit_rate, 3),
            veto_cost=str(review.veto_cost_pct),
        )
        return review

    # ------------------------------------------------------------- internals

    def _grade(
        self, signal: TradeSignal, on: date, benchmark_move: Decimal | None
    ) -> SignalOutcome:
        move = self._move_after(signal.symbol, on)
        realised = self._close_after(signal.symbol, on)
        return SignalOutcome(
            symbol=signal.symbol,
            action=signal.action,
            confidence=signal.confidence,
            status=signal.status,
            entry=signal.entry,
            realised_close=realised,
            move_pct=move,
            benchmark_move_pct=benchmark_move,
            signal_id=str(signal.signal_id),
            agents={name: vote.action for name, vote in signal.agent_votes.items()},
        )

    def _frame(self, symbol: str) -> pd.DataFrame | None:
        if not self._store.has(symbol):
            return None
        frame = self._store.read(symbol)
        return None if frame.empty else frame

    def _close_after(self, symbol: str, on: date) -> Decimal | None:
        """Close of the first session strictly after ``on``."""
        frame = self._frame(symbol)
        if frame is None:
            return None
        index = pd.DatetimeIndex(frame.index)
        later = frame[index > pd.Timestamp(on, tz=index.tz) + pd.Timedelta(hours=23, minutes=59)]
        if later.empty:
            return None
        return Decimal(str(round(float(later["close"].iloc[0]), 2)))

    def _move_after(self, symbol: str, on: date) -> Decimal | None:
        """Percent move from ``on``'s close to the next session's close.

        The signal was generated pre-open using data up to ``on``'s previous
        close, and would have been entered during ``on``. Grading it on the
        move from ``on``'s close forward measures the decision, not the gap it
        was already positioned for.
        """
        frame = self._frame(symbol)
        if frame is None:
            return None
        index = pd.DatetimeIndex(frame.index)
        cutoff = pd.Timestamp(on, tz=index.tz) + pd.Timedelta(hours=23, minutes=59)

        upto = frame[index <= cutoff]
        later = frame[index > cutoff]
        if upto.empty or later.empty:
            return None

        base = Decimal(str(round(float(upto["close"].iloc[-1]), 4)))
        nxt = Decimal(str(round(float(later["close"].iloc[0]), 4)))
        if base == 0:
            return None
        return (nxt - base) / base * 100

    async def _journal(self, review: SessionReview) -> None:
        worst = review.worst_calibrated
        async with self._database.session() as session:
            journal = JournalRepository(session)
            await journal.add(
                JournalEntry(
                    event="post_close_review",
                    summary=(
                        f"{len(review.graded)} of {len(review.outcomes)} signals graded; "
                        f"hit rate {review.hit_rate:.0%} against the index"
                    ),
                    occurred_at=datetime.now(UTC),
                    session_date=review.session_date,
                    payload={
                        "graded": len(review.graded),
                        "pending": review.pending,
                        "hit_rate": round(review.hit_rate, 4),
                        "taken": len(review.taken),
                        "vetoed": len(review.vetoed),
                        "veto_cost_pct": str(review.veto_cost_pct),
                        "scorecards": {
                            name: {
                                "graded": card.graded,
                                "hit_rate": round(card.hit_rate, 4),
                                "mean_confidence": round(card.mean_confidence, 4),
                                "calibration_gap": round(card.calibration_gap, 4),
                            }
                            for name, card in sorted(review.scorecards.items())
                        },
                    },
                )
            )
            if worst is not None and worst.calibration_gap > 0:
                # Named explicitly so the finding is greppable, rather than
                # buried in a payload nobody opens.
                await journal.add(
                    JournalEntry(
                        event="calibration_drift",
                        summary=(
                            f"{worst.agent} stated {worst.mean_confidence:.2f} confidence "
                            f"and was right {worst.hit_rate:.0%} of the time"
                        ),
                        occurred_at=datetime.now(UTC),
                        session_date=review.session_date,
                        payload={
                            "agent": worst.agent,
                            "graded": worst.graded,
                            "calibration_gap": round(worst.calibration_gap, 4),
                        },
                    )
                )


def _score(scorecards: dict[str, AgentScorecard], outcome: SignalOutcome) -> None:
    """Credit every agent that took a directional view on this signal."""
    if outcome.correct is None:
        return
    for agent, action in outcome.agents.items():
        if action is Action.HOLD:
            # An agent that abstained gets neither credit nor blame. Counting
            # HOLD as wrong would push every agent toward reckless conviction.
            continue
        card = scorecards.setdefault(agent, AgentScorecard(agent=agent))
        card.graded += 1
        card.confidence_sum += outcome.confidence
        agreed = action is outcome.action
        if agreed == outcome.correct:
            card.correct += 1


async def run_post_close_review(
    *,
    on: date,
    store: BarStore,
    database: Database,
    settings: Settings,
    benchmark_symbol: str | None = None,
) -> SessionReview:
    """Review one session. The scheduler calls this after the close."""
    review = PostCloseReview(
        store=store, database=database, settings=settings, benchmark_symbol=benchmark_symbol
    )
    return await review.run(on=on)


def scorecard_rows(review: SessionReview) -> list[tuple[str, str, str, str]]:
    """Per-agent rows for CLI rendering: agent, graded, hit rate, calibration."""
    rows: list[tuple[str, str, str, str]] = []
    for name, card in sorted(review.scorecards.items(), key=lambda kv: -kv[1].calibration_gap):
        rows.append(
            (
                name,
                str(card.graded),
                f"{card.hit_rate:.0%}",
                f"{card.calibration_gap:+.2f}",
            )
        )
    return rows


def outcomes_for(review: SessionReview, *, taken_only: bool = False) -> Sequence[SignalOutcome]:
    return review.taken if taken_only else review.graded
