"""The durable order ledger, kept converged with the broker.

Orders are written to the ledger the moment the broker acknowledges them. For
a market order through OpenAlgo that moment is before the fill: the row says
PENDING, nothing filled, no price. Cash, positions and the reconciler are all
replayed from these rows, so a ledger that is never told about the fill
reports an account that never spent a rupee -- and sizes the next session's
orders against money that is already in stock.

:func:`sync_order_ledger` is the step that closes that gap. Every pipeline
that reads the ledger calls it first.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from uuid import UUID

from trading_agent.domain import Order, Position, ProductType
from trading_agent.execution import Reconciler, ReconciliationReport, positions_from_ledger
from trading_agent.market_data import IST
from trading_agent.observability import get_logger
from trading_agent.persistence import Database, OrderRepository

__all__ = [
    "LedgerView",
    "filled_entries",
    "ledger_view",
    "reconcile_account",
    "stamp",
    "sync_order_ledger",
]

log = get_logger(__name__)

#: Before any tick this system could have produced; "everything" needs no
#: special case.
_EPOCH = datetime(2000, 1, 1, tzinfo=UTC)


async def sync_order_ledger(broker: object, database: Database) -> int:
    """Converge every ledger row the broker has news about. Returns rows changed.

    Two sources, because the broker's order book is a day book: Kite's
    ``orderbook`` lists today's orders only. A ledger row still open from an
    earlier session is looked up by id instead; if the broker cannot answer,
    the row is left as it is and logged rather than guessed at.
    """
    try:
        reported = {o.broker_order_id: o for o in await broker.get_orders()}  # type: ignore[attr-defined]
    except Exception:
        log.exception("ledger_sync_orderbook_failed")
        return 0

    changed = 0
    async with database.session() as session:
        repository = OrderRepository(session)
        known = await repository.broker_order_ids()
        still_open = {o.broker_order_id: o for o in await repository.list_open()}

        for broker_order_id, order in reported.items():
            if broker_order_id not in known:
                continue  # an orphan: the reconciler's business, not ours
            current = await repository.get(broker_order_id)
            if current is not None and _differs(current, order):
                await repository.save(order)
                changed += 1

        for broker_order_id, stale in still_open.items():
            if broker_order_id in reported:
                continue
            try:
                order = await broker.get_order(broker_order_id)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                log.warning(
                    "ledger_sync_order_unreachable",
                    broker_order_id=broker_order_id,
                    symbol=stale.symbol,
                    status=stale.status.value,
                )
                continue
            if _differs(stale, order):
                await repository.save(order)
                changed += 1

    if changed:
        log.info("ledger_synced", changed=changed)
    return changed


@dataclass(frozen=True, slots=True)
class LedgerView:
    """What our own records say the account holds."""

    orders: list[Order]
    positions: list[Position]
    #: Every symbol this system has ever placed an order in. A broker position
    #: outside it is not the agent's to manage.
    managed_symbols: set[str]


async def ledger_view(database: Database, *, session_date: date) -> LedgerView:
    """Replay the ledger into positions.

    Delivery (CNC) positions are replayed from every order ever placed: they
    persist until sold. Everything else is intraday and squared off by the
    broker at the close -- often by an order the broker places itself and the
    ledger never sees -- so only ``session_date``'s orders count for those.
    """
    async with database.session() as session:
        orders = list(await OrderRepository(session).list_placed_between(_EPOCH, _now()))

    delivery = [o for o in orders if o.product is ProductType.CNC]
    intraday = [
        o
        for o in orders
        if o.product is not ProductType.CNC and o.placed_at.astimezone(IST).date() == session_date
    ]
    positions = [
        p
        for p in positions_from_ledger(delivery) + positions_from_ledger(intraday)
        if not p.is_flat
    ]
    return LedgerView(
        orders=orders,
        positions=positions,
        managed_symbols={o.symbol for o in orders},
    )


def _differs(local: Order, reported: Order) -> bool:
    return (
        local.status is not reported.status
        or local.filled_quantity != reported.filled_quantity
        or local.average_price != reported.average_price
    )


def _now() -> datetime:
    return datetime.now(UTC)


def stamp(order: Order, *, signal_id: UUID | None, idempotency_key: str) -> Order:
    """``order`` carrying the signal and key it was placed for.

    OpenAlgo's order status carries neither, and without them the ledger
    cannot say which signal opened a position -- which is where the stop that
    protects it is recorded.
    """
    return replace(order, signal_id=signal_id, idempotency_key=idempotency_key)


def filled_entries(orders: Sequence[Order], position: Position) -> list[Order]:
    """Orders that built ``position``, newest first.

    Same symbol and product, filled at least in part, on the side that opens
    it: BUY for a long, SELL for a short.
    """
    opening_side = "BUY" if position.quantity > 0 else "SELL"
    matches = [
        o
        for o in orders
        if o.symbol == position.symbol
        and o.product is position.product
        and o.side.value == opening_side
        and o.filled_quantity > 0
    ]
    return sorted(matches, key=lambda o: o.placed_at, reverse=True)


async def reconcile_account(
    broker: object, database: Database, *, session_date: date
) -> ReconciliationReport:
    """Sync fills, replay the ledger, and reconcile it against the broker's book.

    The one entry point the CLI, the API and the monitor share, so "is the
    account what we think it is?" has one answer wherever it is asked.
    """
    await sync_order_ledger(broker, database)
    view = await ledger_view(database, session_date=session_date)
    reconciler = Reconciler(broker, known_order_ids={o.broker_order_id for o in view.orders})
    return await reconciler.reconcile(
        local_positions=view.positions, managed_symbols=view.managed_symbols
    )
