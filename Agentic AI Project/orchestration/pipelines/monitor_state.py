"""What the stop-loss monitor sees, for a person to read.

The monitor acts on a timer and journals what it did; this assembles the same
facts on demand -- the book it watches, the stop on each position, whether
the book reconciles, and how today's P&L sits against the loss limit -- from
the same functions the monitor calls. A dashboard that recomputed any of it
separately would, sooner or later, show a stop the monitor is not using.

Nothing here places or cancels an order. Reading the state does converge the
ledger toward the broker (fills the broker reported), which is the direction
the ledger only ever moves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from trading_agent.domain import Position, Rupees
from trading_agent.execution import ReconciliationReport, fetch_book
from trading_agent.orchestration.pipelines.intraday_monitor import (
    breached_level,
    exit_working,
    opening_signals,
)
from trading_agent.orchestration.pipelines.ledger import (
    ledger_view,
    reconcile_account,
)
from trading_agent.persistence import Database, JournalEntry, JournalRepository
from trading_agent.settings import Settings

__all__ = ["DailyLoss", "MonitorSnapshot", "WatchedPosition", "monitor_snapshot"]

#: Journal events that describe what the monitor did, newest first on the page.
MONITOR_EVENTS = frozenset(
    {
        "exit_submitted",
        "exit_signalled",
        "kill_switch_engaged",
        "reconciliation_unexplained",
        "session_opening_pnl",
    }
)


@dataclass(frozen=True, slots=True)
class WatchedPosition:
    """One managed position and the levels that protect it."""

    symbol: str
    product: str
    quantity: int
    average_price: Decimal
    mark: Decimal | None
    stop_loss: Decimal | None
    target: Decimal | None
    strategy: str | None
    signal_id: str | None
    exit_working: bool
    #: ``protected`` / ``stop_breached`` / ``target_reached`` / ``exit_working``
    #: / ``no_stop`` / ``no_mark``. The last two are the ones to worry about:
    #: the monitor skips them.
    status: str

    @property
    def stop_distance(self) -> Decimal | None:
        """How far the mark sits above a long's stop, as a fraction of the mark."""
        if self.mark is None or self.stop_loss is None or self.mark == 0:
            return None
        gap = self.mark - self.stop_loss if self.quantity > 0 else self.stop_loss - self.mark
        return gap / self.mark


@dataclass(frozen=True, slots=True)
class DailyLoss:
    """Today's marked P&L against the limit, on the monitor's own baseline."""

    limit: Decimal
    #: ``None`` until the monitor's first tick of the session records it.
    opening_equity: Rupees | None
    opening_pnl: Rupees | None
    current_pnl: Rupees

    @property
    def loss_fraction(self) -> Decimal | None:
        if self.opening_equity is None or self.opening_pnl is None:
            return None
        if self.opening_equity.value <= 0:
            return None
        loss = -(self.current_pnl.value - self.opening_pnl.value)
        return max(loss, Decimal(0)) / self.opening_equity.value


@dataclass(slots=True)
class MonitorSnapshot:
    session_date: date
    reconciliation: ReconciliationReport
    daily_loss: DailyLoss
    positions: list[WatchedPosition] = field(default_factory=list)
    external: list[Position] = field(default_factory=list)
    recent_events: list[JournalEntry] = field(default_factory=list)


async def monitor_snapshot(
    *, broker: object, database: Database, settings: Settings, session_date: date
) -> MonitorSnapshot:
    """Assemble the monitor's view of the account for ``session_date``."""
    report = await reconcile_account(broker, database, session_date=session_date)
    ledger = await ledger_view(database, session_date=session_date)
    book = [p for p in await fetch_book(broker) if not p.is_flat]

    managed = [p for p in book if p.symbol in ledger.managed_symbols]
    external = [p for p in book if p.symbol not in ledger.managed_symbols]
    signals = await opening_signals(database, managed, ledger)

    watched: list[WatchedPosition] = []
    for position in managed:
        signal = signals.get(position.symbol)
        working = exit_working(position, ledger)
        mark = position.last_price
        if working:
            status = "exit_working"
        elif signal is None:
            status = "no_stop"
        elif mark is None:
            status = "no_mark"
        else:
            status = {
                "stop_loss": "stop_breached",
                "target": "target_reached",
            }.get(breached_level(position, signal, mark) or "", "protected")
        watched.append(
            WatchedPosition(
                symbol=position.symbol,
                product=position.product.value,
                quantity=position.quantity,
                average_price=position.average_price,
                mark=mark,
                stop_loss=signal.stop_loss if signal else None,
                target=signal.target if signal else None,
                strategy=signal.strategy if signal else None,
                signal_id=str(signal.signal_id) if signal else None,
                exit_working=working,
                status=status,
            )
        )

    current = Rupees.zero()
    for position in managed:
        current = current + position.realised_pnl
        if position.unrealised_pnl is not None:
            current = current + position.unrealised_pnl

    async with database.session() as session:
        journal = JournalRepository(session)
        baseline = await journal.list_for_session(session_date, event="session_opening_pnl")
        recent = [e for e in await journal.list_recent(limit=200) if e.event in MONITOR_EVENTS]

    payload = baseline[0].payload if baseline else {}
    daily = DailyLoss(
        limit=settings.risk.max_daily_loss,
        opening_equity=Rupees(str(payload["equity"])) if payload.get("equity") else None,
        opening_pnl=Rupees(str(payload["pnl"])) if payload.get("pnl") is not None else None,
        current_pnl=current,
    )

    return MonitorSnapshot(
        session_date=session_date,
        reconciliation=report,
        daily_loss=daily,
        positions=sorted(watched, key=lambda w: w.symbol),
        external=sorted(external, key=lambda p: p.symbol),
        recent_events=recent[:25],
    )
