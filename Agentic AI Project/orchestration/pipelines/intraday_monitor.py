"""Intraday monitoring: stops, exits and periodic reconciliation.

Runs on a timer through the session. Three jobs, in this order, and the order
is the design:

1. **Reconcile.** Before acting on any belief about the book, check that
   belief against the broker. Acting first and reconciling later is how a
   position gets exited twice.
2. **Check the daily loss limit.** A day that has already gone badly is
   exactly when the temptation to trade is highest and the judgement is
   worst, so the circuit breaker is evaluated before any exit logic. The
   limit is a fraction of the account's equity at the session's open.
3. **Evaluate stops and targets.** Compare each open position's mark against
   the stop and target recorded on the signal that opened it -- found through
   the entry order in the ledger, not by "latest signal for this symbol",
   which an exit order's own signal would otherwise become.

An engaged kill switch halts new risk, not protection: stops and targets are
still evaluated and, when ``risk.kill_switch_allows_exits`` is on, still sent.
A halt that also froze every stop would leave the book unprotected on exactly
the day the halt fired.

"The book" is holdings plus positions (:func:`fetch_book`). Overnight CNC
stock lives only in holdings, and a monitor that read positions alone would
watch nothing from the second day of every trade.

**This module does not place orders unless it is given a router.** The default
is observe-and-report: it produces :class:`ExitIntent` s and journals them.
That is deliberate for Phase 5 -- paper trading needs the decisions visible
and auditable before anything is allowed to act on them automatically, and an
exit path that has never been read is not one to arm silently.

Stops are evaluated on the *mark the broker reports*, not on a bar. Intraday,
a daily bar does not exist yet, and reconstructing one from ticks to decide an
exit would be inventing the price that triggers a trade.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal

from trading_agent.costs import ChargeCalculator
from trading_agent.domain import (
    Action,
    Position,
    ProductType,
    Rupees,
    Side,
    SignalStatus,
    TradeSignal,
)
from trading_agent.execution import (
    Broker,
    OrderRouter,
    Reconciler,
    ReconciliationReport,
    fetch_book,
    snapshot_account,
)
from trading_agent.market_data import IST, TradingCalendar
from trading_agent.observability import get_logger
from trading_agent.orchestration.pipelines.ledger import (
    LedgerView,
    filled_entries,
    ledger_view,
    stamp,
    sync_order_ledger,
)
from trading_agent.persistence import (
    Database,
    JournalEntry,
    JournalRepository,
    OrderRepository,
    SignalRepository,
)
from trading_agent.risk import KillSwitch
from trading_agent.settings import Settings

__all__ = ["ExitIntent", "ExitTrigger", "IntradayMonitor", "MonitorTick"]

log = get_logger(__name__)


class ExitTrigger:
    """Why a position should be closed. Strings, so they journal legibly."""

    STOP_LOSS = "stop_loss"
    TARGET = "target"
    KILL_SWITCH = "kill_switch"
    SESSION_END = "session_end"


@dataclass(frozen=True, slots=True)
class ExitIntent:
    """A position that should be closed, and why.

    An *intent*, not an order. Whether it becomes one depends on whether a
    router was supplied.
    """

    symbol: str
    product: ProductType
    quantity: int
    side: Side
    trigger: str
    mark: Decimal
    reference: Decimal | None = None
    signal_id: str | None = None
    submitted: bool = False

    @property
    def detail(self) -> str:
        if self.reference is None:
            return f"{self.trigger} at {self.mark}"
        return f"{self.trigger}: mark {self.mark} against {self.reference}"


@dataclass(slots=True)
class MonitorTick:
    """What one pass of the monitor saw and did."""

    at: datetime
    session_date: date
    in_session: bool
    positions: int = 0
    exits: list[ExitIntent] = field(default_factory=list)
    reconciliation: ReconciliationReport | None = None
    kill_switch_engaged: bool = False
    unrealised: Rupees = field(default_factory=Rupees.zero)
    realised: Rupees = field(default_factory=Rupees.zero)
    #: Equity at the session's open: the base the daily loss limit divides by.
    opening_equity: Rupees | None = None
    skipped_reason: str = ""

    @property
    def acted(self) -> bool:
        return any(exit_.submitted for exit_ in self.exits)

    @property
    def needs_attention(self) -> bool:
        """Whether a human should look at this tick before trading continues."""
        if self.kill_switch_engaged:
            return True
        return self.reconciliation is not None and self.reconciliation.unexplained > 0


class IntradayMonitor:
    """Watches open positions through the session.

    Collaborators are injected so the whole thing runs against a paper broker
    and an in-memory store: an exit path that can only be tested with real
    money is one that never gets tested.
    """

    __slots__ = (
        "_broker",
        "_calculator",
        "_calendar",
        "_database",
        "_kill_switch",
        "_live_acknowledged",
        "_opening_pnl",
        "_reconciler",
        "_router",
        "_settings",
    )

    def __init__(
        self,
        *,
        broker: Broker,
        calendar: TradingCalendar,
        kill_switch: KillSwitch,
        database: Database,
        settings: Settings,
        router: OrderRouter | None = None,
        reconciler: Reconciler | None = None,
        opening_pnl: Rupees | None = None,
        live_acknowledged: bool = False,
        calculator: ChargeCalculator | None = None,
    ) -> None:
        self._broker = broker
        self._calculator = calculator
        self._calendar = calendar
        self._kill_switch = kill_switch
        self._database = database
        self._settings = settings
        #: None means observe-and-report. See the module docstring.
        self._router = router
        self._live_acknowledged = live_acknowledged
        self._reconciler = reconciler or Reconciler(broker)
        self._opening_pnl = opening_pnl

    async def tick(self, *, now: datetime | None = None) -> MonitorTick:
        """One pass. Safe to call on a closed market -- it does nothing."""
        moment = now or datetime.now(UTC)
        local = moment.astimezone(IST)
        session_date = local.date()
        tick = MonitorTick(at=moment, session_date=session_date, in_session=False)

        if not self._calendar.is_trading_day(session_date):
            tick.skipped_reason = "not a trading day"
            return tick
        if not self._calendar.is_open_at(moment):
            tick.skipped_reason = "outside session hours"
            return tick
        tick.in_session = True

        # -- 1. reconcile before believing anything ------------------------
        # Fills first: the ledger learns what the broker did since the last
        # tick, or every position replayed from it is stale by one tick.
        await sync_order_ledger(self._broker, self._database)
        ledger = await ledger_view(self._database, session_date=session_date)
        # The reconciler's "orders we created" set lives in memory, and this
        # monitor is rebuilt on every scheduled tick -- so without reloading
        # that set from the order ledger, every order the system placed itself
        # reads as an orphan, reconciliation is never clean, and the exit
        # evaluation below is never reached. Stops would silently never fire.
        for order in ledger.orders:
            self._reconciler.remember(order.broker_order_id)

        positions = await fetch_book(self._broker)
        tick.reconciliation = await self._reconciler.reconcile(
            local_positions=ledger.positions, managed_symbols=ledger.managed_symbols
        )
        if tick.reconciliation.unexplained:
            # An unexplained divergence means the book is not what we think it
            # is. Exiting on a stop computed from a wrong position size is
            # worse than not exiting, so stop here and raise it.
            log.error(
                "monitor_unexplained_divergence",
                count=tick.reconciliation.unexplained,
                session=session_date.isoformat(),
            )
            await self._journal_divergence(tick)
            tick.skipped_reason = "unexplained divergence; not acting"
            return tick

        # Only what this system manages. External holdings are the operator's,
        # and a stop the agent never set is not one it may act on.
        open_positions = [
            p for p in positions if not p.is_flat and p.symbol in ledger.managed_symbols
        ]
        tick.positions = len(open_positions)
        tick.unrealised = _sum_unrealised(open_positions)
        tick.realised = _sum_realised(positions)
        opening_pnl, tick.opening_equity = await self._session_baseline(
            tick, ledger=ledger, positions=open_positions
        )

        # -- 2. the circuit breaker ----------------------------------------
        if await self._kill_switch.is_engaged():
            tick.kill_switch_engaged = True
        elif self._breached_daily_loss(tick, opening_pnl=opening_pnl):
            daily_pnl = tick.realised + tick.unrealised - opening_pnl
            await self._kill_switch.engage(
                f"daily loss limit breached: session P&L {daily_pnl} against opening "
                f"equity {tick.opening_equity} on {session_date.isoformat()}"
            )
            tick.kill_switch_engaged = True

        # -- 3. stops and targets --------------------------------------
        # Evaluated whether or not the switch is engaged: it halts new risk,
        # and a stop only ever removes risk.
        tick.exits = await self._evaluate_exits(open_positions, session_date, ledger=ledger)

        exits_may_go = not tick.kill_switch_engaged or self._settings.risk.kill_switch_allows_exits
        if self._router is not None and exits_may_go:
            tick.exits = await self._submit(tick.exits, session_date=session_date)

        await self._journal(tick)
        log.info(
            "monitor_tick",
            session=session_date.isoformat(),
            positions=tick.positions,
            exits=len(tick.exits),
            submitted=sum(1 for e in tick.exits if e.submitted),
            kill_switch=tick.kill_switch_engaged,
        )
        return tick

    # ------------------------------------------------------------- internals

    async def _session_baseline(
        self, tick: MonitorTick, *, ledger: LedgerView, positions: Sequence[Position]
    ) -> tuple[Rupees, Rupees]:
        """This session's opening marked P&L and opening equity.

        Written to the journal on the session's first tick and read back on
        every later one, so a monitor rebuilt per tick measures the day
        against one fixed baseline rather than against its own last tick.

        An explicit ``opening_pnl`` wins over the journal. A replay passes one
        it knows is right, and the persistent replay store can still hold a
        baseline an earlier run of the same date left behind.
        """
        async with self._database.session() as session:
            journal = JournalRepository(session)
            entries = await journal.list_for_session(tick.session_date, event="session_opening_pnl")
            stored = entries[0].payload if entries else None

            current_equity = self._equity(ledger, positions)
            if stored is not None:
                pnl = (
                    self._opening_pnl
                    if self._opening_pnl is not None
                    else Rupees(str(stored["pnl"]))
                )
                equity = Rupees(str(stored["equity"])) if stored.get("equity") else current_equity
                return pnl, equity

            opening = (
                self._opening_pnl
                if self._opening_pnl is not None
                else tick.realised + tick.unrealised
            )
            await journal.add(
                JournalEntry(
                    event="session_opening_pnl",
                    summary=f"session opened with marked P&L {opening}, equity {current_equity}",
                    occurred_at=datetime.combine(
                        tick.session_date,
                        self._settings.market.session_open,
                        tzinfo=IST,
                    ),
                    session_date=tick.session_date,
                    payload={"pnl": str(opening.value), "equity": str(current_equity.value)},
                )
            )
            return opening, current_equity

    def _equity(self, ledger: LedgerView, positions: Sequence[Position]) -> Rupees:
        """Cash from the ledger plus the book at its marks, charges included.

        The same arithmetic the entry path sizes against, so the loss limit
        and the position sizes are fractions of one number.
        """
        initial = Rupees(self._settings.backtest.initial_capital)
        calculator = self._calculator
        if calculator is None:
            try:
                calculator = ChargeCalculator.from_directory(
                    self._settings.charges_dir, broker=self._settings.costs.broker
                )
            except Exception:  # noqa: BLE001
                log.warning("monitor_equity_uses_initial_capital", reason="no charge schedule")
                return initial
            self._calculator = calculator
        return snapshot_account(
            initial_capital=initial,
            orders=ledger.orders,
            positions=list(positions),
            calculator=calculator,
        ).equity

    def _breached_daily_loss(self, tick: MonitorTick, *, opening_pnl: Rupees) -> bool:
        """Whether this session's marked loss has hit the ceiling.

        Unrealised loss counts. A position that is down 6% and still open has
        lost the money; refusing to count it until it is closed is how a bad
        day becomes a bad week.
        """
        equity = tick.opening_equity
        if equity is None or equity.value <= 0:
            return False
        current_pnl = tick.realised + tick.unrealised
        loss = -(current_pnl.value - opening_pnl.value)
        if loss <= 0:
            return False
        return loss / equity.value >= self._settings.risk.max_daily_loss

    async def _evaluate_exits(
        self, positions: Sequence[Position], session_date: date, *, ledger: LedgerView
    ) -> list[ExitIntent]:
        """Compare each mark against the stop and target that opened it."""
        if not positions:
            return []

        signals = await self._signals_for(positions, ledger)
        intents: list[ExitIntent] = []

        for position in positions:
            if exit_working(position, ledger):
                # An earlier exit is still working at the broker. A second one
                # now would sell shares the first is already selling.
                log.info("monitor_exit_already_working", symbol=position.symbol)
                continue

            mark = _mark(position)
            if mark is None:
                # No mark means no basis for an exit decision. Guessing one
                # from the average price would trigger on a number we made up.
                log.warning("monitor_no_mark", symbol=position.symbol)
                continue

            signal = signals.get(position.symbol)
            if signal is None:
                log.warning(
                    "monitor_no_signal", symbol=position.symbol, session=session_date.isoformat()
                )
                continue

            trigger = breached_level(position, signal, mark)
            if trigger is not None:
                reference = signal.stop_loss if trigger == ExitTrigger.STOP_LOSS else signal.target
                intents.append(
                    _intent(
                        position,
                        trigger,
                        mark=mark,
                        reference=reference,
                        signal_id=str(signal.signal_id),
                    )
                )
        return intents

    async def _signals_for(
        self, positions: Sequence[Position], ledger: LedgerView
    ) -> dict[str, TradeSignal]:
        return await opening_signals(self._database, positions, ledger)

    async def _submit(self, intents: list[ExitIntent], *, session_date: date) -> list[ExitIntent]:
        """Turn intents into exit orders. Only reached when a router exists."""
        assert self._router is not None  # noqa: S101 -- guarded by the caller
        submitted: list[ExitIntent] = []

        for intent in intents:
            exit_signal = TradeSignal(
                symbol=intent.symbol,
                # The exit is the opposite side of the position.
                action=Action.SELL if intent.side is Side.SELL else Action.BUY,
                product=intent.product,
                entry=intent.mark,
                # A market exit has no protective geometry of its own; the
                # domain still wants a coherent stop, so it is placed beyond
                # the mark on the correct side and never used.
                stop_loss=_nominal_stop(intent),
                target=intent.mark,
                generated_at=datetime.now(UTC),
                session_date=session_date,
                strategy="intraday_exit",
                # The quantity is in the key: after a partial fill the rest is
                # a different order, and must not be refused as a duplicate of
                # the first. A rejected order under the same key is re-opened
                # by the router, which asks the broker whether it died.
                idempotency_key=(
                    f"exit-{intent.symbol}-{session_date.isoformat()}-{intent.trigger}"
                    f"-q{abs(intent.quantity)}"
                ),
                quantity=abs(intent.quantity),
                rationale=intent.detail,
                # SIZED, not GENERATED. The router refuses anything still
                # GENERATED as "not through the risk layer yet" -- correct for
                # an entry, wrong for an exit, whose quantity is not a sizing
                # decision at all but the position that is already open.
                # Left as GENERATED, every exit is rejected at the router.
                status=SignalStatus.SIZED,
            )
            try:
                result = await self._router.submit(
                    exit_signal,
                    live_acknowledged=self._live_acknowledged,
                )
            except Exception as exc:
                log.exception("exit_submission_failed", symbol=intent.symbol, error=str(exc))
                submitted.append(intent)
                continue

            # Persist what was placed. Without this the exit exists only at the
            # broker: the cash ledger misses the proceeds, and every later
            # reconciliation flags the order as an orphan -- which stops the
            # monitor acting at all from the next tick onward.
            #
            # Not on a duplicate: the guard recognised an order that already
            # exists, and it was recorded when it was first placed.
            if not result.was_duplicate:
                await self._record(exit_signal, ack_id=result.ack.broker_order_id)

            submitted.append(
                ExitIntent(
                    symbol=intent.symbol,
                    product=intent.product,
                    quantity=intent.quantity,
                    side=intent.side,
                    trigger=intent.trigger,
                    mark=intent.mark,
                    reference=intent.reference,
                    signal_id=intent.signal_id,
                    submitted=True,
                )
            )
        return submitted

    async def _record(self, signal: TradeSignal, *, ack_id: str) -> None:
        """Store the exit order and the signal that produced it."""
        order = stamp(
            await self._broker.get_order(ack_id),
            signal_id=signal.signal_id,
            idempotency_key=signal.idempotency_key,
        )
        async with self._database.session() as session:
            await OrderRepository(session).save(order)
            await SignalRepository(session).save(signal)

    async def _journal(self, tick: MonitorTick) -> None:
        if not tick.exits and not tick.kill_switch_engaged:
            # A quiet tick is not worth a row. Writing one every thirty
            # seconds would bury the ticks that matter.
            return
        async with self._database.session() as session:
            journal = JournalRepository(session)
            if tick.kill_switch_engaged:
                await journal.add(
                    JournalEntry(
                        event="kill_switch_engaged",
                        summary=(
                            f"kill switch active; new entries halted, "
                            f"{sum(1 for e in tick.exits if e.submitted)} protective exit(s) sent"
                        ),
                        occurred_at=tick.at,
                        session_date=tick.session_date,
                        payload={
                            "realised": str(tick.realised.value),
                            "unrealised": str(tick.unrealised.value),
                        },
                    )
                )
            for intent in tick.exits:
                await journal.add(
                    JournalEntry(
                        event="exit_submitted" if intent.submitted else "exit_signalled",
                        summary=f"{intent.symbol}: {intent.detail}",
                        occurred_at=tick.at,
                        session_date=tick.session_date,
                        symbol=intent.symbol,
                        payload={
                            "trigger": intent.trigger,
                            "mark": str(intent.mark),
                            "reference": str(intent.reference) if intent.reference else None,
                            "quantity": intent.quantity,
                            "submitted": intent.submitted,
                        },
                    )
                )

    async def _journal_divergence(self, tick: MonitorTick) -> None:
        report = tick.reconciliation
        if report is None:
            return
        async with self._database.session() as session:
            await JournalRepository(session).add(
                JournalEntry(
                    event="reconciliation_unexplained",
                    summary=(
                        f"{report.unexplained} unexplained divergence(s); "
                        f"the monitor is not acting until a human looks"
                    ),
                    occurred_at=tick.at,
                    session_date=tick.session_date,
                    payload={
                        "orphan_orders": report.orphan_orders,
                        "positions_corrected": report.positions_corrected,
                        "details": [d.detail for d in report.divergences][:20],
                    },
                )
            )


# --------------------------------------------------------------------- helpers


#: Signal states that mean the signal actually went to the broker.
_ROUTED = frozenset({SignalStatus.SIZED, SignalStatus.SUBMITTED, SignalStatus.EXECUTED})


def _mark(position: Position) -> Decimal | None:
    return position.last_price


def _opens(signal: TradeSignal, position: Position) -> bool:
    """Whether ``signal`` is on the side that opens ``position``."""
    if signal.strategy == "intraday_exit":
        return False
    return (signal.action is Action.BUY) == (position.quantity > 0)


def exit_working(position: Position, ledger: LedgerView) -> bool:
    """Whether a non-terminal order on the closing side is still at the broker."""
    closing = Side.SELL if position.quantity > 0 else Side.BUY
    return any(
        o.symbol == position.symbol
        and o.product is position.product
        and o.side is closing
        and not o.status.is_terminal
        for o in ledger.orders
    )


def _sum_unrealised(positions: Sequence[Position]) -> Rupees:
    total = Rupees.zero()
    for position in positions:
        pnl = position.unrealised_pnl
        if pnl is not None:
            total = total + pnl
    return total


def _sum_realised(positions: Sequence[Position]) -> Rupees:
    total = Rupees.zero()
    for position in positions:
        total = total + position.realised_pnl
    return total


def breached_level(position: Position, signal: TradeSignal, mark: Decimal) -> str | None:
    """Which level the mark has crossed, if any.

    The stop is checked first. When a violent move puts the mark beyond both
    levels in one tick, the stop is the honest reading -- assuming the target
    filled first would book a profit the market never offered.
    """
    if position.quantity > 0:  # long
        if mark <= signal.stop_loss:
            return ExitTrigger.STOP_LOSS
        if mark >= signal.target:
            return ExitTrigger.TARGET
        return None

    if mark >= signal.stop_loss:  # short
        return ExitTrigger.STOP_LOSS
    if mark <= signal.target:
        return ExitTrigger.TARGET
    return None


def _intent(
    position: Position,
    trigger: str,
    *,
    mark: Decimal | None,
    reference: Decimal | None = None,
    signal_id: str | None = None,
) -> ExitIntent:
    return ExitIntent(
        symbol=position.symbol,
        product=position.product,
        quantity=position.quantity,
        # Closing a long is a SELL; closing a short is a BUY.
        side=Side.SELL if position.quantity > 0 else Side.BUY,
        trigger=trigger,
        mark=mark if mark is not None else position.average_price,
        reference=reference,
        signal_id=signal_id,
    )


def _nominal_stop(intent: ExitIntent) -> Decimal:
    """A coherent but unused stop for a market exit order.

    ``intent.side`` is the side of the *exit*, so selling to close a long is a
    SELL, and :class:`TradeSignal` requires a SELL's stop to sit **above** its
    entry. Getting this backwards does not misprice anything -- the stop is
    never used -- it raises out of the domain validator and aborts the exit,
    which is how a stop-loss silently stops working.
    """
    offset = intent.mark * Decimal("0.1")
    return intent.mark + offset if intent.side is Side.SELL else intent.mark - offset


async def opening_signals(
    database: Database, positions: Sequence[Position], ledger: LedgerView
) -> dict[str, TradeSignal]:
    """The signal that opened each position, whatever session that was.

    Found through the ledger: the newest filled entry order carries the
    id of the signal it was placed for. "The latest signal for the symbol"
    is not that -- once an exit order is placed, its own signal is the
    latest, and its stop sits on the wrong side of the mark by design.

    Orders recorded before the ledger kept signal ids fall back to the
    newest routed entry signal in the symbol.
    """
    latest: dict[str, TradeSignal] = {}
    async with database.session() as session:
        repository = SignalRepository(session)
        recent: Sequence[TradeSignal] | None = None
        for position in positions:
            for order in filled_entries(ledger.orders, position):
                if order.signal_id is None:
                    continue
                signal = await repository.get(order.signal_id)
                if signal is not None and _opens(signal, position):
                    latest[position.symbol] = signal
                    break
            if position.symbol in latest:
                continue
            if recent is None:
                recent = await repository.list_recent(limit=500)
            for signal in recent:
                if (
                    signal.symbol == position.symbol
                    and _opens(signal, position)
                    and signal.status in _ROUTED
                ):
                    latest[position.symbol] = signal
                    break
    return latest


async def run_intraday_monitor(
    *,
    broker: Broker,
    calendar: TradingCalendar,
    kill_switch: KillSwitch,
    database: Database,
    settings: Settings,
    router: OrderRouter | None = None,
    now: datetime | None = None,
    live_acknowledged: bool = False,
    opening_pnl: Rupees | None = None,
    calculator: ChargeCalculator | None = None,
) -> MonitorTick:
    """One monitoring pass. The scheduler calls this on a timer."""
    monitor = IntradayMonitor(
        broker=broker,
        calendar=calendar,
        kill_switch=kill_switch,
        database=database,
        settings=settings,
        router=router,
        live_acknowledged=live_acknowledged,
        opening_pnl=opening_pnl,
        calculator=calculator,
    )
    return await monitor.tick(now=now)
