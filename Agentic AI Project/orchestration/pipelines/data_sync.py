"""Keep the curated bar store current, from the NSE archive.

Everything downstream of the bar store -- the screener, the charts, the
backtester, the factor panel, the daily research run -- reads bars and computes
nothing without them. Yet nothing in the schedule ever *wrote* bars: the store
was filled by an operator running ``trading-agent data sync`` by hand.

On a laptop with a populated ``data/`` directory that is invisible. On a fresh
deployment it is fatal and silent: the container mounts an empty volume, every
panel renders its honest "no bars" state, and the system looks broken while
reporting itself healthy. This pipeline closes that gap by making the sync part
of the session, like research and the post-close review.

It runs twice:

* **Pre-open**, to catch up anything missed while the process was down --
  a weekend, a redeploy, a VM that was off.
* **Post-close**, once the archive has published the session that just ended.

The NSE archive is used rather than the broker: it needs no Kite session, so
the sync keeps working on a day nobody has logged in, which is exactly the day
a stale store would otherwise go unnoticed.

The benchmark index is synced alongside the equities, from the separate index
archive. It is not optional decoration: the regime classifier, the factor
panel's market model and every backtest's alpha figure are all measured
against it, so a store with equities and no benchmark leaves those three
reporting "not measured" while looking fully populated.

Failure here is reported, never raised past the job boundary: a missing archive
file must not take the rest of the session's schedule down with it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Final

from trading_agent.market_data import DEFAULT_INDEX, BarStore
from trading_agent.market_data.bhavcopy_source import BhavcopySource
from trading_agent.market_data.calendar import TradingCalendar
from trading_agent.market_data.index_source import IndexSource
from trading_agent.market_data.universe import Universe
from trading_agent.observability import get_logger
from trading_agent.settings import Settings

__all__ = ["DataSyncOutcome", "run_data_sync"]

log = get_logger(__name__)

#: How far back a routine catch-up reaches. Long enough to cover a long weekend
#: plus a redeploy, short enough that the daily job stays a few seconds.
DEFAULT_LOOKBACK_DAYS: Final = 10

#: A store with fewer symbols than this has clearly never been seeded, so the
#: catch-up widens to a full backfill instead of pulling ten days onto nothing.
_EMPTY_STORE_SYMBOLS: Final = 5

#: Where a first-run backfill starts when the store is empty. The same date
#: ``trading-agent data sync`` has always defaulted to, so a seeded deployment
#: ends up with the history a developer's machine has rather than a truncated
#: version of it.
#:
#: It is a lot of sessions -- roughly 2,750, about fifteen minutes and a
#: gigabyte of cached archive over a normal link. That is a one-time cost on a
#: fresh volume, and it buys the thing a shorter window cannot: several
#: distinct regimes to walk a backtest across. Callers wanting less pass
#: ``cold_start_from``.
COLD_START_FROM: Final = date(2015, 1, 1)


@dataclass(frozen=True, slots=True)
class DataSyncOutcome:
    """What the sync managed to write."""

    session: date
    start: date
    end: date
    symbols_requested: int
    symbols_written: int
    bars_written: int
    cold_start: bool = False
    #: Bars written for the benchmark index, and why not when zero.
    benchmark_symbol: str = ""
    benchmark_bars: int = 0
    benchmark_error: str = ""
    failures: list[tuple[str, str]] = field(default_factory=list)


def _universe_symbols(settings: Settings, universe_name: str) -> list[str]:
    """Every symbol the universe has ever held, current members included.

    Former constituents are kept: without their prices a backtest tests only
    the survivors, however carefully it gates membership.
    """
    directory = settings.path(settings.data.universe_dir)
    current = directory / f"{universe_name}_current.csv"
    if not current.exists():
        return []
    universe = Universe.from_csv(
        universe_name,
        current_path=current,
        history_path=directory / f"{universe_name}_constituent_history.csv",
    )
    return sorted(set(universe.current_symbols()) | set(universe.former_members()))


def _symbols_to_sync(settings: Settings, store: BarStore, universe_name: str) -> list[str]:
    """The universe, plus everything the store already holds.

    A store is routinely wider than the universe file that seeded it -- ours
    holds 580 symbols against a 200-name universe. Syncing only the universe
    would leave the rest frozen at whatever session they were last written on,
    and the screener drops any symbol whose last bar predates the newest one in
    the set. Those symbols would not go stale visibly; they would silently
    disappear from every scan and ranking.

    So the store's own contents are part of the sync. A name that has genuinely
    delisted returns no rows and lands in `failures`, which is the honest
    outcome and cheap: the archive is fetched per session, not per symbol.
    """
    universe = set(_universe_symbols(settings, universe_name))
    held = {s for s in store.symbols() if s != settings.backtest.benchmark_symbol}
    return sorted(universe | held)


def run_data_sync(
    *,
    on: date,
    settings: Settings,
    universe_name: str | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    cold_start_from: date | None = None,
) -> DataSyncOutcome:
    """Pull recent sessions from the NSE archive into the curated store.

    Args:
        on: The session being run for. The window ends here.
        settings: Live settings; supplies the store, cache and calendar paths.
        universe_name: Which universe file names the symbols to sync.
            Defaults to ``settings.data.sync_universe``. Symbols already in the
            store are synced too, whatever this names.
        lookback_days: Calendar days of catch-up on a store that already holds
            bars.
        cold_start_from: Where the backfill starts when the store looks
            unseeded. Defaults to :data:`COLD_START_FROM`.

    Returns:
        A :class:`DataSyncOutcome` describing what was written. Per-symbol
        failures are collected rather than raised: one delisted name must not
        abort the sync for the other two hundred.
    """
    universe_name = universe_name or settings.data.sync_universe
    store = BarStore(settings.curated_dir, interval=settings.data.bar_interval)
    symbols = _symbols_to_sync(settings, store, universe_name)
    if not symbols:
        log.warning("data_sync_no_universe", universe=universe_name)
        return DataSyncOutcome(
            session=on,
            start=on,
            end=on,
            symbols_requested=0,
            symbols_written=0,
            bars_written=0,
        )

    cold_start = len(store.symbols()) < _EMPTY_STORE_SYMBOLS
    start = (
        (cold_start_from or COLD_START_FROM) if cold_start else on - timedelta(days=lookback_days)
    )

    calendar = TradingCalendar.from_csv(
        settings.holidays_path,
        session_open=settings.market.session_open,
        session_close=settings.market.session_close,
    )
    source = BhavcopySource(settings.raw_dir, calendar)

    log.info(
        "data_sync_start",
        session=on.isoformat(),
        start=start.isoformat(),
        symbols=len(symbols),
        cold_start=cold_start,
    )

    frames = source.fetch_universe_history(symbols, start, on)

    bars_written = 0
    written = 0
    failures: list[tuple[str, str]] = []
    for symbol in symbols:
        frame = frames.get(symbol)
        if frame is None or frame.empty:
            failures.append((symbol, "no rows in any session (delisted or renamed?)"))
            continue
        try:
            bars_written += store.upsert(symbol, frame, source=source.name)
            written += 1
        except (OSError, ValueError) as exc:  # one bad symbol must not stop the sync
            failures.append((symbol, str(exc)[:120]))

    benchmark_bars, benchmark_error = _sync_benchmark(settings, store, calendar, start, on)

    log.info(
        "data_sync_complete",
        session=on.isoformat(),
        bars=bars_written,
        symbols_written=written,
        benchmark_bars=benchmark_bars,
        failures=len(failures),
    )
    return DataSyncOutcome(
        session=on,
        start=start,
        end=on,
        symbols_requested=len(symbols),
        symbols_written=written,
        bars_written=bars_written,
        cold_start=cold_start,
        benchmark_symbol=settings.backtest.benchmark_symbol,
        benchmark_bars=benchmark_bars,
        benchmark_error=benchmark_error,
        failures=failures,
    )


def _sync_benchmark(
    settings: Settings,
    store: BarStore,
    calendar: TradingCalendar,
    start: date,
    end: date,
) -> tuple[int, str]:
    """Pull the benchmark index from the NSE index archive.

    A separate archive from the equity bhavcopy, hence a separate fetch. Its
    failure is returned rather than raised: the equities are already written by
    this point and losing them to a missing index file would be the worse
    outcome.
    """
    symbol = settings.backtest.benchmark_symbol
    try:
        source = IndexSource(settings.raw_dir, calendar)
        frame = source.fetch_daily(start, end, index_name=DEFAULT_INDEX)
        if frame.empty:
            return 0, f"No rows for index {DEFAULT_INDEX!r} between {start} and {end}."
        return store.upsert(symbol.upper(), frame, source=source.name), ""
    except (OSError, ValueError, KeyError) as exc:
        log.warning("data_sync_benchmark_failed", symbol=symbol, error=str(exc)[:200])
        return 0, str(exc)[:200]
