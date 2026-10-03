"""Bar-by-bar backtest loop.

Event-driven rather than vectorised, because the things that decide whether an
Indian equity strategy is real cannot be expressed as a column operation:
path-dependent stops, gap fills, circuit locks, per-order brokerage caps and
per-scrip DP debits.

The loop's contract, and the reason it is written this way:

1. On bar *t* the strategy sees data **up to and including t** and nothing more.
2. Exits are evaluated **before** entries, on bar *t*'s own high/low -- a stop
   hit this morning frees capital that this morning's signal may use.
3. Entries decided on bar *t* fill at bar *t+1*'s **open**.
4. Every fill pays charges through the cost model. There is no free trade.

Rule 3 is the one that separates an honest backtest from a flattering one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

import pandas as pd

from trading_agent.backtest.fill_model import FillModel
from trading_agent.backtest.metrics import PerformanceMetrics, compute_metrics
from trading_agent.backtest.portfolio import InsufficientCashError, Portfolio
from trading_agent.costs.calculator import ChargeCalculator
from trading_agent.domain.enums import Action, ProductType, Side
from trading_agent.domain.money import Rupees
from trading_agent.domain.signal import TradeSignal
from trading_agent.domain.trade import RoundTrip
from trading_agent.observability import get_logger

__all__ = ["BacktestResult", "run_backtest"]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class BacktestResult:
    equity_curve: pd.Series
    round_trips: list[RoundTrip]
    metrics: PerformanceMetrics
    benchmark_curve: pd.Series | None = None
    benchmark_metrics: PerformanceMetrics | None = None
    skipped: dict[str, int] = field(default_factory=dict)

    @property
    def beat_benchmark(self) -> bool | None:
        """Whether the strategy beat buy-and-hold on total return, net.

        ``None`` when no benchmark was supplied. This is the number the whole
        exercise exists to produce.
        """
        if self.benchmark_metrics is None:
            return None
        return self.metrics.total_return > self.benchmark_metrics.total_return


def _to_decimal(value: object) -> Decimal:
    return Decimal(str(round(float(value), 2)))  # type: ignore[arg-type]


def run_backtest(
    *,
    strategy: object,
    panel: dict[str, pd.DataFrame],
    calculator: ChargeCalculator,
    fill_model: FillModel,
    initial_capital: Rupees,
    max_positions: int = 5,
    product: ProductType = ProductType.CNC,
    benchmark: pd.Series | None = None,
    is_eligible: Callable[[str, date], bool] | None = None,
) -> BacktestResult:
    """Run ``strategy`` over ``panel``.

    ``panel`` maps symbol to a feature frame indexed by tz-aware timestamp.
    Every frame must already carry its indicators; this loop computes none, so
    it cannot accidentally introduce lookahead of its own.

    ``is_eligible(symbol, session)`` gates ENTRIES on point-in-time index
    membership. Without it the run buys names years before they joined the
    index -- the mirror image of survivorship bias, and just as invisible:
    TRENT and BEL joined the NIFTY 50 in September 2024, and an ungated run
    trades them from 2015. Exits are deliberately NOT gated: a position held
    when its symbol leaves the index still has to be sold.
    """
    if not panel:
        raise ValueError("panel is empty; nothing to backtest")

    portfolio = Portfolio(initial_capital, calculator)
    strategy_name = str(getattr(strategy, "name", "strategy"))
    # The union of all symbols' bars: symbols may list or delist mid-run.
    sessions = sorted({ts for frame in panel.values() for ts in frame.index})
    if not sessions:
        raise ValueError("panel contains no bars")

    equity_points: list[float] = []
    skipped: dict[str, int] = {}

    def _skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    pending_signals: list[TradeSignal] = []

    for position, session in enumerate(sessions):
        is_last = position == len(sessions) - 1
        marks: dict[str, Decimal] = {}

        # -- 1. exits on holdings touched during this session ---------------
        _process_exits(
            portfolio=portfolio,
            panel=panel,
            session=session,
            fill_model=fill_model,
            strategy_name=strategy_name,
        )

        # -- 2. entries decided previously, filling at today's open ---------
        if pending_signals:
            _process_pending_entries(
                portfolio=portfolio,
                panel=panel,
                session=session,
                signals=pending_signals,
                fill_model=fill_model,
                product=product,
                max_positions=max_positions,
                on_skip=_skip,
            )
            pending_signals = []

        # -- 3. marks for the equity curve at today's close -----------------
        for symbol, frame in panel.items():
            if session in frame.index:
                marks[symbol] = _to_decimal(frame.loc[session, "close"])

        equity_points.append(float(portfolio.equity(marks).value))

        # -- 4. generate signals on today's close to execute tomorrow -------
        if not is_last:
            pending_signals = _generate(strategy, panel, session, is_eligible)

    equity_curve = pd.Series(equity_points, index=pd.DatetimeIndex(sessions), name="equity")

    metrics = compute_metrics(
        equity_curve, portfolio.round_trips, total_charges=portfolio.total_charges
    )

    benchmark_metrics = None
    if benchmark is not None and not benchmark.empty:
        aligned = benchmark.reindex(equity_curve.index).ffill().dropna()
        if not aligned.empty:
            scaled = aligned / aligned.iloc[0] * float(initial_capital.value)
            benchmark_metrics = compute_metrics(scaled, [], total_charges=Rupees.zero())
            benchmark = scaled

    log.info(
        "backtest_complete",
        sessions=len(sessions),
        trades=len(portfolio.round_trips),
        final_equity=str(metrics.final_equity),
        total_return=f"{metrics.total_return:.2%}",
    )

    return BacktestResult(
        equity_curve=equity_curve,
        round_trips=portfolio.round_trips,
        metrics=metrics,
        benchmark_curve=benchmark if benchmark_metrics else None,
        benchmark_metrics=benchmark_metrics,
        skipped=skipped,
    )


def _process_exits(
    *,
    portfolio: Portfolio,
    panel: dict[str, pd.DataFrame],
    session: pd.Timestamp,
    fill_model: FillModel,
    strategy_name: str,
) -> None:
    """Close holdings whose stop or target was touched by THIS bar.

    Runs before entries: capital freed by a stop-out this morning is available
    to this morning's signals, which is how a real account behaves.
    """
    for symbol in list(portfolio.holdings):
        frame = panel.get(symbol)
        if frame is None or session not in frame.index:
            continue

        bar = frame.loc[session]
        holding = portfolio.holdings[symbol]
        volume = int(bar.get("avg_volume_20", bar["volume"]) or 0)

        # Stop first. When a single bar spans both the stop and the target,
        # daily data cannot say which came first, so the conservative reading
        # wins -- assuming the favourable one is how backtests lie.
        stop = fill_model.fill_stop(
            stop_price=holding.stop_loss,
            bar_open=_to_decimal(bar["open"]),
            bar_low=_to_decimal(bar["low"]),
            quantity=holding.quantity,
            average_daily_volume=volume,
        )
        if stop.filled:
            portfolio.sell(
                symbol=symbol,
                price=stop.price,
                at=session,
                reason="stop_loss",
                strategy=strategy_name,
            )
            continue

        target = fill_model.fill_target(
            target_price=holding.target,
            bar_open=_to_decimal(bar["open"]),
            bar_high=_to_decimal(bar["high"]),
            quantity=holding.quantity,
            average_daily_volume=volume,
        )
        if target.filled:
            portfolio.sell(
                symbol=symbol,
                price=target.price,
                at=session,
                reason="target",
                strategy=strategy_name,
            )


def _process_pending_entries(
    *,
    portfolio: Portfolio,
    panel: dict[str, pd.DataFrame],
    session: pd.Timestamp,
    signals: list[TradeSignal],
    fill_model: FillModel,
    product: ProductType,
    max_positions: int,
    on_skip: Callable[[str], None],
) -> None:
    """Execute pending signals decided at the prior session, filling at THIS session's open."""
    available = max_positions - len(portfolio.holdings)
    if available <= 0:
        return

    for signal in signals:
        if available <= 0:
            return
        if portfolio.holds(signal.symbol):
            on_skip("already holding")
            continue

        frame = panel.get(signal.symbol)
        if frame is None or session not in frame.index:
            on_skip("no next bar")
            continue

        bar = frame.loc[session]
        volume = int(bar.get("avg_volume_20", bar["volume"]) or 0)

        # Equal-weight the remaining capacity so an early signal cannot
        # consume the whole book.
        budget = Rupees(portfolio.cash.value / max(available, 1))
        reference = _to_decimal(bar["open"])

        quantity = portfolio.affordable_quantity(
            price=reference, budget=budget, product=product, on=session.date()
        )
        if quantity <= 0:
            on_skip("insufficient cash")
            continue

        loc = frame.index.get_loc(session)
        previous_close = (
            _to_decimal(frame.iloc[loc - 1]["close"])
            if isinstance(loc, int) and loc > 0
            else _to_decimal(bar["close"])
        )

        fill = fill_model.fill_at_open(
            next_open=reference,
            previous_close=previous_close,
            side=Side.BUY,
            quantity=quantity,
            average_daily_volume=volume,
        )
        if not fill.filled:
            on_skip(fill.reason)
            continue

        try:
            portfolio.buy(
                symbol=signal.symbol,
                quantity=quantity,
                price=fill.price,
                at=session,
                product=product,
                stop_loss=signal.stop_loss,
                target=signal.target,
            )
            available -= 1
        except InsufficientCashError:
            on_skip("insufficient cash")


def _generate(
    strategy: object,
    panel: dict[str, pd.DataFrame],
    session: date,
    is_eligible: Callable[[str, date], bool] | None = None,
) -> list[TradeSignal]:
    """Ask the strategy for signals, passing only data up to ``session``.

    Long-only by construction, and gated on index membership as of ``session``
    when an eligibility predicate is supplied. Both filters are applied here
    rather than inside each strategy so no strategy can forget one.
    """
    generate = getattr(strategy, "generate", None)
    if generate is None:
        raise TypeError("strategy must expose a generate(panel, on=...) method")
    signals: list[TradeSignal] = generate(panel, on=session)
    longs = [s for s in signals if s.action is Action.BUY]
    if is_eligible is None:
        return longs
    # `session` arrives as a tz-aware pd.Timestamp. Membership windows are
    # plain dates, and comparing the two raises rather than coercing, so the
    # conversion is explicit here instead of hoped for downstream.
    on = session.date() if isinstance(session, pd.Timestamp) else session
    return [s for s in longs if is_eligible(s.symbol, on)]
