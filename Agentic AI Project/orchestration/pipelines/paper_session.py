"""The entry path: signals become orders.

This is the step that did not exist. ``daily_research`` generated signals and
persisted them; ``intraday_monitor`` watched positions and could close them.
Nothing in between sized a signal and sent it, so :class:`OrderRouter` -- with
every gate the project cares about built into it -- was reachable only from
its own unit tests. Paper trading could not start because the loop was open.

The order of operations, and why it differs from the one
:mod:`trading_agent.risk` describes:

1. **Size.** :func:`check_limits` needs the notional a trade would take on,
   and the notional is what sizing produces. Sizing is a pure function with no
   side effect, so computing it before the limit check costs nothing and
   cannot arm anything.
2. **Check limits.** Against the book *as this session has already changed
   it* -- each submitted order updates the running position list and cash, so
   the fifth signal of a session is judged against the four that preceded it
   rather than against the state at open. Checking every signal against the
   opening snapshot is how five signals each pass a "max four positions"
   limit.
3. **Route.** :class:`OrderRouter` re-checks the kill switch and the
   idempotency claim per order. Those are not duplicated here: a gate
   enforced in two places is a gate that will eventually disagree with
   itself.

Nothing in this module knows whether the broker is paper or live. That is the
point -- the path under test in paper is the path that runs in live, and a
paper run that exercised a different code path would prove nothing about it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal

from trading_agent.costs.calculator import ChargeCalculator
from trading_agent.domain import Order, Position, Rupees, SignalStatus, TradeSignal
from trading_agent.execution import Broker, OrderRouter, RoutingRejectedError, fetch_book
from trading_agent.execution.account import AccountSnapshot, snapshot_account
from trading_agent.market_data import Universe
from trading_agent.observability import get_logger
from trading_agent.orchestration.pipelines.ledger import stamp, sync_order_ledger
from trading_agent.persistence import (
    Database,
    JournalEntry,
    JournalRepository,
    OrderRepository,
    SignalRepository,
)
from trading_agent.risk import PortfolioLimits, check_limits, size_position
from trading_agent.settings import Settings

__all__ = ["PaperSession", "SessionOutcome", "SubmittedOrder", "run_paper_session"]

log = get_logger(__name__)

#: Sessions averaged for the ADV participation cap. Matches the window the
#: slippage impact model uses, so the gate and the cost model agree.
ADV_WINDOW_SESSIONS = 20


@dataclass(frozen=True, slots=True)
class SubmittedOrder:
    """One signal that made it all the way to the broker."""

    symbol: str
    quantity: int
    entry: Decimal
    broker_order_id: str
    was_duplicate: bool

    @property
    def notional(self) -> Rupees:
        return Rupees(self.entry * self.quantity)


@dataclass(slots=True)
class SessionOutcome:
    """What one session's entry pass actually did."""

    session_date: date
    considered: int = 0
    submitted: list[SubmittedOrder] = field(default_factory=list)
    #: symbol -> why it did not trade. Every declined signal appears here;
    #: a signal that vanishes without a reason is the bug this prevents.
    declined: dict[str, str] = field(default_factory=dict)
    opening: AccountSnapshot | None = None
    closing: AccountSnapshot | None = None

    @property
    def submitted_count(self) -> int:
        return len(self.submitted)

    @property
    def deployed(self) -> Rupees:
        total = Rupees.zero()
        for order in self.submitted:
            total = total + order.notional
        return total

    def summary_rows(self) -> list[tuple[str, str]]:
        rows = [
            ("Session", self.session_date.isoformat()),
            ("Signals considered", str(self.considered)),
            ("Orders submitted", str(self.submitted_count)),
            ("Capital deployed", str(self.deployed)),
            ("Declined", str(len(self.declined))),
        ]
        if self.closing is not None:
            rows.extend(self.closing.summary_rows())
        return rows


class PaperSession:
    """Sizes, vets and routes one session's signals.

    Collaborators are injected so the whole path runs in a test against a
    ``PaperBroker`` and an in-memory key-value store. The alternative is that
    the only way to exercise the entry path is to place an order.
    """

    __slots__ = (
        "_adv_lookup",
        "_broker",
        "_calculator",
        "_database",
        "_initial_capital",
        "_router",
        "_settings",
        "_universe",
    )

    def __init__(
        self,
        *,
        broker: Broker,
        router: OrderRouter,
        database: Database,
        calculator: ChargeCalculator,
        settings: Settings,
        universe: Universe | None = None,
        initial_capital: Rupees | None = None,
        adv_lookup: Callable[[str], int | None] | None = None,
    ) -> None:
        self._broker = broker
        self._router = router
        self._database = database
        self._calculator = calculator
        self._settings = settings
        self._universe = universe
        self._initial_capital = initial_capital or Rupees(settings.backtest.initial_capital)
        # How a symbol's average daily volume is found when the caller does not
        # hand one over. Without it the ADV participation cap in `check_limits`
        # is unreachable in production: the limit exists, is tested, and can
        # never fire, which is worse than not having it.
        self._adv_lookup = adv_lookup

    @property
    def limits(self) -> PortfolioLimits:
        risk = self._settings.risk
        return PortfolioLimits(
            max_positions=risk.max_positions,
            max_sector_exposure=risk.max_sector_exposure,
            max_daily_loss=risk.max_daily_loss,
            max_position_weight=risk.max_position_weight,
            max_adv_participation=risk.max_adv_participation,
        )

    # ------------------------------------------------------------- the run

    async def run(
        self,
        *,
        on: date,
        signals: Sequence[TradeSignal] | None = None,
        marks: dict[str, Decimal] | None = None,
        live_acknowledged: bool = False,
        advs: dict[str, int] | None = None,
    ) -> SessionOutcome:
        """Route every actionable signal for ``on`` that survives the gates.

        ``signals`` defaults to the session's persisted, un-vetoed signals, so
        the scheduler can run this straight after research. Passing them
        explicitly is what lets a strategy drive the same path.
        """
        outcome = SessionOutcome(session_date=on)

        candidates = list(signals) if signals is not None else await self._persisted(on)
        # Deterministic order by conviction, breaking ties by symbol: which signals get
        # funded when cash runs out prioritises highest confidence rather than alphabetical order.
        candidates = sorted(candidates, key=lambda s: (-s.confidence, s.symbol))
        outcome.considered = len(candidates)

        # Fills since the last run reach the ledger before cash is read from
        # it; otherwise yesterday's buys never left the cash balance.
        await sync_order_ledger(self._broker, self._database)
        ledger = await self._ledger()
        # Holdings included: yesterday's CNC buys are in holdings, not
        # positions, and limits blind to them buy the same name twice.
        positions = [p for p in await fetch_book(self._broker) if not p.is_flat]
        account = snapshot_account(
            initial_capital=self._initial_capital,
            orders=ledger,
            positions=positions,
            calculator=self._calculator,
            marks=marks,
        )
        outcome.opening = account

        log.info(
            "paper_session_open",
            session=on.isoformat(),
            signals=len(candidates),
            equity=str(account.equity),
            cash=str(account.cash),
            positions=len(positions),
        )

        # The running book: mutated as orders go out so each signal is judged
        # against what this session has already committed.
        book = list(positions)
        cash = account.available_cash

        for signal in candidates:
            decision = await self._consider(
                signal,
                on=on,
                book=book,
                equity=account.equity,
                cash=cash,
                live_acknowledged=live_acknowledged,
                outcome=outcome,
                advs=advs,
            )
            if decision is None:
                continue

            submitted, sized = decision
            outcome.submitted.append(submitted)
            cash = cash - submitted.notional
            book.append(
                Position(
                    symbol=sized.symbol,
                    quantity=sized.quantity,
                    average_price=sized.entry,
                    product=sized.product,
                    exchange=sized.exchange,
                    last_price=sized.entry,
                )
            )

        outcome.declined = dict(sorted(outcome.declined.items()))

        await sync_order_ledger(self._broker, self._database)
        closing_ledger = await self._ledger()
        outcome.closing = snapshot_account(
            initial_capital=self._initial_capital,
            orders=closing_ledger,
            positions=[p for p in await fetch_book(self._broker) if not p.is_flat],
            calculator=self._calculator,
            marks=marks,
        )

        log.info(
            "paper_session_complete",
            session=on.isoformat(),
            submitted=outcome.submitted_count,
            declined=len(outcome.declined),
            equity=str(outcome.closing.equity),
        )
        return outcome

    def _average_daily_volume(self, symbol: str, advs: dict[str, int] | None) -> int | None:
        """ADV for ``symbol``: the caller's table first, then the store lookup.

        An explicit table wins so a replay can price the participation cap off
        the very panel it is replaying, rather than off whatever the curated
        store happens to hold today.
        """
        if advs is not None and symbol in advs:
            return advs[symbol]
        if self._adv_lookup is None:
            return None
        try:
            return self._adv_lookup(symbol)
        except Exception:  # noqa: BLE001 -- a missing bar file must not stop the session
            log.warning("adv_lookup_failed", symbol=symbol)
            return None

    async def _consider(
        self,
        signal: TradeSignal,
        *,
        on: date,
        book: list[Position],
        equity: Rupees,
        cash: Rupees,
        live_acknowledged: bool,
        outcome: SessionOutcome,
        advs: dict[str, int] | None = None,
    ) -> tuple[SubmittedOrder, TradeSignal] | None:
        """Size, vet and route one signal. ``None`` when it did not trade."""
        if signal.is_vetoed:
            await self._decline(
                signal, on=on, reason=f"vetoed: {signal.veto_reason}", outcome=outcome
            )
            return None
        if not signal.action.is_actionable:
            await self._decline(
                signal, on=on, reason=f"action is {signal.action.value}", outcome=outcome
            )
            return None

        sizing = size_position(
            signal,
            equity=equity,
            available_cash=cash,
            max_risk_fraction=self._settings.risk.max_portfolio_risk_per_trade,
            max_position_weight=self.limits.max_position_weight,
            calculator=self._calculator,
            on=on,
            product=signal.product,
        )
        if not sizing.is_tradeable:
            await self._decline(
                signal, on=on, reason=sizing.reason or "sized to zero shares", outcome=outcome
            )
            return None

        verdict = check_limits(
            signal,
            limits=self.limits,
            open_positions=book,
            equity=equity,
            available_cash=cash,
            intended_value=sizing.outlay,
            sectors=self._sectors(),
            intended_quantity=sizing.quantity,
            average_daily_volume=self._average_daily_volume(signal.symbol, advs),
        )
        if not verdict.allowed:
            reason = f"{verdict.breach}: {verdict.detail}" if verdict.breach else "limit breach"
            await self._decline(signal, on=on, reason=reason, outcome=outcome)
            return None

        sized = signal.sized(quantity=sizing.quantity, risk_amount=sizing.risk_amount)
        await self._save_signal(sized)

        try:
            result = await self._router.submit(sized, live_acknowledged=live_acknowledged)
        except RoutingRejectedError as exc:
            # The router refused at a gate this pipeline does not duplicate
            # (kill switch, live acknowledgement, an in-flight claim).
            await self._decline(sized, on=on, reason=f"router: {exc.reason.value}", outcome=outcome)
            return None

        await self._record_submission(sized, ack_id=result.ack.broker_order_id)
        return (
            SubmittedOrder(
                symbol=sized.symbol,
                quantity=sized.quantity,
                entry=sized.entry,
                broker_order_id=result.ack.broker_order_id,
                was_duplicate=result.was_duplicate,
            ),
            sized,
        )

    # -------------------------------------------------------------- helpers

    def _sectors(self) -> dict[str, str]:
        """Symbol -> sector, for the concentration limit.

        Empty when no universe is loaded: the sector limit then cannot bind,
        which :func:`check_limits` treats as "unknown", not as "zero exposure".
        """
        if self._universe is None:
            return {}
        return {
            symbol: sector
            for symbol in self._universe.current_symbols()
            if (sector := self._universe.sector_of(symbol)) is not None
        }

    async def _persisted(self, on: date) -> list[TradeSignal]:
        async with self._database.session() as session:
            rows = await SignalRepository(session).list_for_session(
                on, status=SignalStatus.GENERATED
            )
            return list(rows)

    async def _ledger(self) -> list[Order]:
        """Every order ever placed, from the database rather than the broker.

        The database is durable across restarts and the in-process paper
        broker is not, so cash reconstructed from the broker's memory would
        silently reset to the starting capital after any restart.
        """
        async with self._database.session() as session:
            return list(await OrderRepository(session).list_placed_between(_EPOCH, _now()))

    async def _save_signal(self, signal: TradeSignal) -> None:
        async with self._database.session() as session:
            await SignalRepository(session).save(signal)

    async def _record_submission(self, signal: TradeSignal, *, ack_id: str) -> None:
        """Persist the order the broker acknowledged, and why it went out."""
        async with self._database.session() as session:
            # Stamped with the signal: OpenAlgo's order status carries no
            # signal id, and without it the monitor cannot find the stop.
            order = stamp(
                await self._broker.get_order(ack_id),
                signal_id=signal.signal_id,
                idempotency_key=signal.idempotency_key,
            )
            await OrderRepository(session).save(order)
            await SignalRepository(session).save(signal.replace(status=SignalStatus.SUBMITTED))
            await JournalRepository(session).add(
                JournalEntry(
                    event="order_submitted",
                    summary=(
                        f"{signal.action.value} {signal.quantity} {signal.symbol} "
                        f"at {signal.entry} (stop {signal.stop_loss})"
                    ),
                    occurred_at=_now(),
                    session_date=signal.session_date,
                    symbol=signal.symbol,
                    signal_id=signal.signal_id,
                    payload={
                        "broker_order_id": ack_id,
                        "quantity": signal.quantity,
                        "risk_amount": str(signal.risk_amount.value),
                        "strategy": signal.strategy,
                    },
                )
            )

    async def _decline(
        self, signal: TradeSignal, *, on: date, reason: str, outcome: SessionOutcome
    ) -> None:
        """Record a signal that did not trade, and why.

        Journalled as well as returned: the post-close review compares what was
        taken against what was refused, and a refusal that exists only in a log
        line is not in the population it measures.
        """
        outcome.declined[signal.symbol] = reason
        log.debug("paper_signal_declined", symbol=signal.symbol, reason=reason)
        async with self._database.session() as session:
            await JournalRepository(session).add(
                JournalEntry(
                    event="entry_declined",
                    summary=f"{signal.symbol} not entered: {reason}",
                    occurred_at=_now(),
                    session_date=on,
                    symbol=signal.symbol,
                    signal_id=signal.signal_id,
                    payload={"reason": reason, "strategy": signal.strategy},
                )
            )


def _now() -> datetime:
    return datetime.now(UTC)


#: Lower bound for the ledger query. Before any Indian equity tick this system
#: could have produced, so "everything" needs no special case.
_EPOCH = datetime(2000, 1, 1, tzinfo=UTC)


async def run_paper_session(
    *,
    on: date,
    broker: Broker,
    router: OrderRouter,
    database: Database,
    settings: Settings,
    universe_name: str | None = None,
    live_acknowledged: bool = False,
) -> SessionOutcome:
    """Build the pipeline from settings and route ``on``'s signals.

    The entry point the scheduler fires each session. Tests construct
    :class:`PaperSession` directly with fakes.
    """
    from trading_agent.costs import ChargeCalculator
    from trading_agent.market_data import BarStore

    name = universe_name or settings.data.default_universe
    directory = settings.path(settings.data.universe_dir)
    universe = Universe.from_csv(
        name,
        current_path=directory / f"{name}_current.csv",
        history_path=directory / f"{name}_constituent_history.csv",
    )

    # Marks come from the curated store rather than the broker: holdings must
    # be valued to decide how much cash a new position may consume, and the
    # broker's last-traded price is stale for anything that has not printed
    # today.
    store = BarStore(settings.curated_dir, interval=settings.data.bar_interval)
    marks: dict[str, Decimal] = {}
    for position in await fetch_book(broker):
        if position.is_flat or not store.has(position.symbol):
            continue
        bars = store.read(position.symbol)
        if not bars.empty:
            marks[position.symbol] = Decimal(str(round(float(bars["close"].iloc[-1]), 2)))

    def adv_lookup(symbol: str) -> int | None:
        """Twenty-session average volume from the curated store.

        Twenty sessions matches the window the slippage impact model prices
        against, so the cap that refuses an order and the model that charges
        for it are reading the same denominator.
        """
        if not store.has(symbol):
            return None
        bars = store.read(symbol)
        if bars.empty or "volume" not in bars.columns:
            return None
        window = bars["volume"].tail(ADV_WINDOW_SESSIONS).dropna()
        if window.empty:
            return None
        return int(window.mean())

    pipeline = PaperSession(
        broker=broker,
        router=router,
        database=database,
        calculator=ChargeCalculator.from_directory(
            settings.charges_dir, broker=settings.costs.broker
        ),
        settings=settings,
        universe=universe,
        adv_lookup=adv_lookup,
    )
    return await pipeline.run(on=on, marks=marks, live_acknowledged=live_acknowledged)
