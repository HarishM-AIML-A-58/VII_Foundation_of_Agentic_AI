"""Pre-open research pipeline.

Order matters for cost: the cheap quantitative screen runs first and hands the
agents a shortlist. Running a seven-agent debate across the full universe
instead of ten candidates is the difference between a sane Foundry bill and an
absurd one -- at roughly 19k tokens per symbol per round, NIFTY 50 is five
times the bill of a ten-name shortlist for the same ten decisions worth having.

The pipeline is deliberately a sequence of narrow steps, each of which can be
inspected on its own:

    screen  ->  debate  ->  convert  ->  veto  ->  persist

``screen`` is arithmetic over stored bars, so it costs nothing and is
reproducible. ``debate`` is the only step that spends money. ``veto`` is code
in :mod:`trading_agent.risk` and cannot be argued with. ``persist`` writes the
signal *and* its journal trail in one transaction, so a signal that exists is
always explainable.

**Every shortlisted symbol is persisted, including the rejected ones.** A
vetoed signal and a HOLD verdict are the most useful records the post-close
review has: they are the only evidence of what the system declined to do, and
without them the only measurable population is the trades that happened.

Sizing is *not* done here. A position size depends on account equity and
available cash, which come from the broker, and pre-open research runs before
that snapshot is meaningful. Signals land as GENERATED or VETOED; execution
sizes them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

import pandas as pd

from trading_agent.agents import (
    DEFAULT_DEBATE_ROUNDS,
    FoundryClient,
    run_debate,
    to_signal,
)
from trading_agent.domain import SignalStatus, TradeSignal
from trading_agent.features import build_features, feature_set
from trading_agent.market_data import BarStore, Universe
from trading_agent.market_data.superinvesting_lake import SuperinvestingLake
from trading_agent.observability import get_logger
from trading_agent.persistence import Database, JournalEntry, JournalRepository, SignalRepository
from trading_agent.risk import apply_veto
from trading_agent.settings import Settings

__all__ = ["Candidate", "DailyResearch", "ResearchOutcome", "run_daily_research"]

log = get_logger(__name__)

#: Sessions of history handed to the agents. Fifteen is enough to see a trend
#: and a reversal without burning the context window on bars nobody reads.
CONTEXT_BARS = 15

#: Bars a symbol must have before it can be screened. The longest indicator in
#: `agent_context` is a 200-day SMA; ranking on a column that is still NaN
#: silently sorts the least-established names to the top.
MIN_HISTORY_BARS = 200

#: How stale the last bar may be before a symbol is skipped. A symbol whose
#: data stopped updating is not a hold, it is a gap in the data feed, and
#: debating it produces confident nonsense about a stale price.
MAX_STALENESS_DAYS = 7


def before_session(bars: pd.DataFrame, on: date) -> pd.DataFrame:
    """Bars strictly before ``on``.

    The lookahead guard, applied identically by the screen and by the context
    handed to the agents. Deciding what to debate before the session opens,
    using the session's own close, is lookahead of the quietest kind: nothing
    errors, the backtest simply gets better.
    """
    index = pd.DatetimeIndex(bars.index)
    return bars[index < pd.Timestamp(on, tz=index.tz)]


@dataclass(frozen=True, slots=True)
class Candidate:
    """A symbol that survived the quantitative screen.

    ``stretch`` is how far price sits from its 50-day mean, measured in ATRs.
    Signed, so the sort can be by magnitude while the direction stays legible
    in the log: a large negative stretch is a candidate short, not a weak long.
    """

    symbol: str
    stretch: float
    close: float
    session: date

    @property
    def magnitude(self) -> float:
        return abs(self.stretch)


@dataclass(slots=True)
class ResearchOutcome:
    """What one run of the pipeline actually did."""

    session_date: date
    screened: int = 0
    shortlisted: int = 0
    debated: int = 0
    signals: list[TradeSignal] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def actionable(self) -> list[TradeSignal]:
        """Signals that survived the veto. The only ones worth executing."""
        return [s for s in self.signals if not s.is_vetoed]

    @property
    def vetoed(self) -> list[TradeSignal]:
        return [s for s in self.signals if s.is_vetoed]

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class DailyResearch:
    """The pre-open pipeline, with its collaborators injected.

    Injected rather than constructed internally so the whole pipeline runs in
    a test against a fake client and an in-memory store -- the alternative is
    that the only way to exercise it is to spend money.
    """

    __slots__ = ("_client", "_database", "_lake", "_settings", "_store", "_universe")

    def __init__(
        self,
        *,
        client: FoundryClient,
        store: BarStore,
        universe: Universe,
        database: Database,
        settings: Settings,
        lake: SuperinvestingLake | None = None,
    ) -> None:
        self._client = client
        #: Optional third-party evidence. Absent, the debate runs on bars
        #: alone, exactly as before -- the lake widens what agents see, it is
        #: never a precondition for a session.
        self._lake = lake
        self._store = store
        self._universe = universe
        self._database = database
        self._settings = settings

    # ------------------------------------------------------------- screening

    def screen(self, on: date, *, limit: int) -> list[Candidate]:
        """Rank the universe by how far price has stretched from its mean.

        ATR-normalised so the ranking compares a 3000-rupee stock with a
        300-rupee one honestly: two rupees of move mean very different things
        in each, and an un-normalised screen simply surfaces the expensive
        names every day.

        This is a *funnel*, not a strategy. It decides what is worth an
        expensive opinion, and nothing here decides direction -- that is what
        the debate and then the veto are for.
        """
        candidates: list[Candidate] = []
        for symbol in self._universe.symbols_on(on):
            candidate = self._score(symbol, on)
            if candidate is not None:
                candidates.append(candidate)

        candidates.sort(key=lambda c: c.magnitude, reverse=True)
        return candidates[:limit]

    def _score(self, symbol: str, on: date) -> Candidate | None:
        """Score one symbol, or return None with the reason logged."""
        if not self._store.has(symbol):
            return None

        bars = self._store.read(symbol)
        if len(bars) < MIN_HISTORY_BARS:
            log.debug("screen_skip", symbol=symbol, reason="insufficient history", bars=len(bars))
            return None

        usable = before_session(bars, on)
        if len(usable) < MIN_HISTORY_BARS:
            log.debug("screen_skip", symbol=symbol, reason="insufficient history before session")
            return None

        last_session = usable.index[-1].date()
        if (on - last_session).days > MAX_STALENESS_DAYS:
            log.warning(
                "screen_skip", symbol=symbol, reason="stale data", last_session=str(last_session)
            )
            return None

        features = build_features(usable, feature_set("agent_context"), verify=False)
        latest = features.iloc[-1]
        atr = float(latest["atr_14"])
        sma_50 = float(latest["sma_50"])
        close = float(latest["close"])

        if not atr > 0 or pd.isna(sma_50) or pd.isna(close):
            log.debug("screen_skip", symbol=symbol, reason="indicators unavailable")
            return None

        return Candidate(
            symbol=symbol, stretch=(close - sma_50) / atr, close=close, session=last_session
        )

    # ---------------------------------------------------------------- context

    def market_context(self, symbol: str, on: date) -> str:
        """The bars and indicators handed to every agent for ``symbol``."""
        usable = before_session(self._store.read(symbol), on)
        features = build_features(usable, feature_set("agent_context"), verify=False)
        tail = features.tail(CONTEXT_BARS)
        latest = tail[["sma_20", "sma_50", "rsi_14", "atr_14"]].iloc[-1].round(2)
        context = (
            f"## Recent daily bars (last {CONTEXT_BARS} sessions)\n"
            f"{tail[['open', 'high', 'low', 'close', 'volume']].round(2).to_string()}\n\n"
            f"## Indicators (latest session)\n{latest.to_string()}\n\n"
            f"Exchange: NSE. All prices in INR."
        )
        evidence = self._lake.evidence_for(symbol, on) if self._lake is not None else None
        return f"{context}\n\n{evidence}" if evidence else context

    # --------------------------------------------------------------- the run

    async def run(
        self, *, on: date, shortlist_size: int = 10, rounds: int = DEFAULT_DEBATE_ROUNDS
    ) -> ResearchOutcome:
        """Screen, debate the shortlist, veto, and persist everything."""
        outcome = ResearchOutcome(session_date=on)
        outcome.screened = len(self._universe.symbols_on(on))

        shortlist = self.screen(on, limit=shortlist_size)
        outcome.shortlisted = len(shortlist)
        log.info(
            "research_shortlist",
            session=on.isoformat(),
            screened=outcome.screened,
            shortlisted=[c.symbol for c in shortlist],
        )

        for candidate in shortlist:
            try:
                signal = await self._research_one(candidate, on=on, rounds=rounds, outcome=outcome)
            except Exception as exc:
                # A failure on one name is not a reason to abandon the other
                # nine. It is recorded and the run continues.
                log.exception("research_failed", symbol=candidate.symbol, error=str(exc))
                outcome.skipped[candidate.symbol] = f"{type(exc).__name__}: {exc}"
                continue

            if signal is not None:
                outcome.signals.append(signal)

        log.info(
            "research_complete",
            session=on.isoformat(),
            debated=outcome.debated,
            actionable=len(outcome.actionable),
            vetoed=len(outcome.vetoed),
            skipped=len(outcome.skipped),
            total_tokens=outcome.total_tokens,
        )
        return outcome

    async def _research_one(
        self, candidate: Candidate, *, on: date, rounds: int, outcome: ResearchOutcome
    ) -> TradeSignal | None:
        """Debate one candidate and persist whatever it produced."""
        state = await run_debate(
            self._client,
            symbol=candidate.symbol,
            session_date=on,
            market_context=self.market_context(candidate.symbol, on),
            generated_at=datetime.now(UTC),
            rounds=rounds,
        )
        outcome.debated += 1
        for record in state.get("usage", []):
            outcome.input_tokens += record.input_tokens
            outcome.output_tokens += record.output_tokens

        signal = to_signal(state)
        if signal is None:
            # HOLD, an abstaining moderator, or geometry that did not hold
            # together. Recorded so the review can tell a considered HOLD from
            # a symbol that was never looked at.
            await self._journal_only(
                candidate, on=on, state_abstentions=state.get("abstentions", [])
            )
            return None

        judged = apply_veto(
            signal,
            min_reward_to_risk=self._settings.risk.min_reward_to_risk,
        )
        await self._persist(judged, candidate=candidate, abstentions=state.get("abstentions", []))
        return judged

    # ------------------------------------------------------------ persistence

    async def _persist(
        self, signal: TradeSignal, *, candidate: Candidate, abstentions: Sequence[str]
    ) -> None:
        """Write the signal and its trail in ONE transaction.

        Not two: a signal without its journal entry is a decision with no
        recorded reason, which is precisely the state this system exists to
        avoid.
        """
        async with self._database.session() as session:
            await SignalRepository(session).add(signal)
            from trading_agent.persistence.repositories.memory_repository import (
                MemoryEntryRepository,
                new_entry,
            )

            await MemoryEntryRepository(session).upsert(
                new_entry(
                    kind="debate",
                    title=f"{signal.action.value} {signal.symbol}",
                    body=(
                        f"{signal.action.value} {signal.symbol} at {signal.entry} "
                        f"(stop {signal.stop_loss}, target {signal.target}). "
                        f"confidence {signal.confidence:.2f}. "
                        f"{signal.veto_reason or signal.rationale or ''}"
                    )[:2000],
                    session_date=signal.session_date,
                    symbol=signal.symbol,
                    payload={
                        "confidence": signal.confidence,
                        "status": signal.status.value,
                        "signal_id": str(signal.signal_id),
                    },
                )
            )
            journal = JournalRepository(session)
            await journal.add(
                JournalEntry(
                    event="signal_generated",
                    summary=(
                        f"{signal.action.value} {signal.symbol} at {signal.entry} "
                        f"(stop {signal.stop_loss}, target {signal.target})"
                    ),
                    occurred_at=signal.generated_at,
                    session_date=signal.session_date,
                    symbol=signal.symbol,
                    signal_id=signal.signal_id,
                    payload={
                        "confidence": signal.confidence,
                        "reward_to_risk": str(signal.reward_to_risk),
                        "screen_stretch_atr": round(candidate.stretch, 3),
                        "abstentions": list(abstentions),
                        "agents": sorted(signal.agent_votes),
                    },
                )
            )
            if signal.status is SignalStatus.VETOED:
                await journal.add(
                    JournalEntry(
                        event="signal_vetoed",
                        summary=f"{signal.symbol} vetoed: {signal.veto_reason}",
                        occurred_at=signal.generated_at,
                        session_date=signal.session_date,
                        symbol=signal.symbol,
                        signal_id=signal.signal_id,
                        payload={"reason": signal.veto_reason or ""},
                    )
                )

    async def _journal_only(
        self, candidate: Candidate, *, on: date, state_abstentions: Sequence[str]
    ) -> None:
        """Record a debate that produced no tradeable signal."""
        async with self._database.session() as session:
            await JournalRepository(session).add(
                JournalEntry(
                    event="no_signal",
                    summary=f"{candidate.symbol} debated, no actionable verdict",
                    occurred_at=datetime.now(UTC),
                    session_date=on,
                    symbol=candidate.symbol,
                    payload={
                        "screen_stretch_atr": round(candidate.stretch, 3),
                        "abstentions": list(state_abstentions),
                    },
                )
            )


async def run_daily_research(
    *,
    on: date,
    settings: Settings,
    universe_name: str = "nifty50",
    shortlist_size: int = 10,
    rounds: int = DEFAULT_DEBATE_ROUNDS,
) -> ResearchOutcome:
    """Build the pipeline from settings and run it for ``on``.

    The convenience entry point for the scheduler and the CLI. Tests build
    :class:`DailyResearch` directly with fakes.
    """
    from trading_agent.agents import cache_from_settings

    client = FoundryClient(settings.foundry, cache=cache_from_settings(settings))
    store = BarStore(settings.curated_dir, interval=settings.data.bar_interval)
    universe_dir = settings.path(settings.data.universe_dir)
    universe = Universe.from_csv(
        universe_name,
        current_path=universe_dir / f"{universe_name}_current.csv",
        history_path=universe_dir / f"{universe_name}_constituent_history.csv",
    )
    database = Database.from_settings(settings)
    lake_root = settings.curated_dir / "superinvesting"
    lake = SuperinvestingLake(lake_root) if SuperinvestingLake.available(lake_root) else None

    pipeline = DailyResearch(
        client=client,
        store=store,
        universe=universe,
        database=database,
        settings=settings,
        lake=lake,
    )
    try:
        return await pipeline.run(on=on, shortlist_size=shortlist_size, rounds=rounds)
    finally:
        await database.dispose()
