"""Scheduled-rebalance backtest loop, for portfolio strategies.

Separate from :mod:`trading_agent.backtest.engine` on purpose. That loop holds a
position until its stop or target is hit; this one holds until the next
rebalance replaces it. Squeezing the second into the first is what turned twelve
decisions a year into 465 round trips.

The same four rules as the signal engine, because they are what make a backtest
honest rather than flattering:

1. On session *t* the strategy sees data **up to and including t**, never more.
2. Sells are processed **before** buys, so the cash a sale frees is available to
   the same rebalance -- which is what actually happens, and assuming otherwise
   silently understates achievable turnover.
3. Decisions on session *t* fill at session *t+1*'s **open**.
4. Every fill pays charges through the cost model.

Two controls live here rather than in the strategy, because both are properties
of the *portfolio* and not of the ranking:

**The regime filter.** Varsity's warning about momentum portfolios is explicit:
they work in an uptrend, perform poorly when choppy, and "bleed heavier than the
markets itself" when markets fall. The filter answers that directly -- when the
benchmark closes below its own long moving average, the portfolio stops opening
new positions and liquidates. It does not try to be clever about it.

**The portfolio stop.** Varsity's answer on stops for this strategy is 2% at
portfolio level, not per stock. Here it is expressed as a drawdown-from-peak
limit: breach it and the book is closed until the next rebalance date.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

import pandas as pd

from trading_agent.backtest.fill_model import FillModel
from trading_agent.backtest.metrics import (
    PerformanceMetrics,
    compute_metrics,
)
from trading_agent.backtest.portfolio import InsufficientCashError, Portfolio
from trading_agent.backtest.volatility_target import VolatilityTargeter
from trading_agent.costs.calculator import ChargeCalculator
from trading_agent.domain.enums import ProductType, Side
from trading_agent.domain.money import Rupees
from trading_agent.domain.trade import RoundTrip
from trading_agent.observability import get_logger

__all__ = [
    "MIN_TRADEABLE_WEIGHT",
    "RebalanceResult",
    "RegimeFilter",
    "run_rebalance_backtest",
]

log = get_logger(__name__)

#: Weight below which a target is treated as zero. A rebalance that buys a
#: 0.0001 position pays a full DP charge for nothing.
#:
#: Public because callers have to plan around it. An equal-weight strategy over
#: more than ``1 / MIN_TRADEABLE_WEIGHT`` names produces targets that all land
#: below this floor, and the book then stays flat for the whole run -- which is
#: only detectable from outside if the number is visible.
MIN_TRADEABLE_WEIGHT = Decimal("0.005")
_MIN_WEIGHT = MIN_TRADEABLE_WEIGHT

#: Shortest usable moving-average window. One point is not an average.
_MIN_WINDOW = 2


@dataclass(frozen=True, slots=True)
class RegimeFilter:
    """Suppress new positions when the benchmark is below its own trend.

    ``series`` is the benchmark close, ``window`` its moving-average length.
    200 sessions is the conventional long trend; Varsity commenters propose a
    20-day variant, which is the same idea with a shorter memory and more
    whipsaw.

    ``liquidate`` decides whether a risk-off regime merely stops buying or also
    sells what is held. **Default False, and that default was set by
    measurement, not by argument.** Selling looks like the braver reading of
    Varsity's "the momentum portfolio bleeds heavier than the market", and it is
    what this filter did first. On the NIFTY 50 over 2015-2026 it was
    comprehensively worse on every axis:

    ==============================  =========  ==========  =======
    12-1 momentum, top 12            OOS ret   OOS Sharpe    trips
    ==============================  =========  ==========  =======
    no filter                        +64.13%       -0.08      165
    stop adding only (default)       +56.58%       -0.13      135
    ==============================  =========  ==========  =======

    Measured on the point-in-time panel of 82 symbols, walk-forward, 20 windows.
    On this panel the filter costs return out-of-sample too, so it is off by
    default at the engine level and only ``liquidate`` is defaulted here.

    On the earlier survivorship-biased panel, liquidating was far worse than
    either: +93.56% against +164.78% for stop-adding-only, with Sharpe turning
    negative and turnover *rising* from 231 to 404 round trips. The mechanism is
    whipsaw -- every dip through the average sells the book and the recovery buys
    it back higher, paying the round trip each time. Refusing to add expresses
    the same caution without paying for it, which is why that is the default.
    """

    series: pd.Series
    window: int = 200
    liquidate: bool = False

    def __post_init__(self) -> None:
        if self.window < _MIN_WINDOW:
            raise ValueError(f"window must be at least {_MIN_WINDOW}, got {self.window}")
        if self.series.empty:
            raise ValueError("benchmark series is empty; regime filter cannot operate")

    def is_risk_on(self, session: pd.Timestamp) -> bool:
        """True when the benchmark close on or before ``session`` is above its average.

        Uses only data up to and including ``session`` -- no lookahead. If the
        series has not reached ``window`` points yet, returns True: an uninitialised
        filter must not stop trading on session 1.
        """
        history = self.series[self.series.index <= session]
        if len(history) < self.window:
            return True
        average = float(history.iloc[-self.window :].mean())
        return float(history.iloc[-1]) >= average


@dataclass(slots=True)
class RebalanceResult:
    """What one rebalance-driven run produced."""

    strategy: str
    equity_curve: pd.Series
    metrics: PerformanceMetrics
    round_trips: list[RoundTrip]
    total_charges: Rupees
    initial_capital: Rupees
    final_equity: Rupees
    rebalances: int = 0
    #: Sessions the regime filter held the book flat or refused to add.
    risk_off_sessions: int = 0
    #: Times the portfolio-level stop closed the book.
    portfolio_stops: int = 0
    benchmark: PerformanceMetrics | None = None
    skipped: dict[str, int] = field(default_factory=dict)
    cash_yield_accrued: Rupees = field(default_factory=Rupees.zero)

    @property
    def beat_benchmark(self) -> bool | None:
        if self.benchmark is None:
            return None
        return self.metrics.total_return > self.benchmark.total_return


def _as_date(value: pd.Timestamp) -> date:
    return value.date()


@dataclass(slots=True)
class _RunState:
    """Mutable counters for one run.

    Extracted so the session loop reads as the sequence of decisions it is,
    rather than as a wall of local variables being incremented.
    """

    peak_equity: Rupees
    rebalances: int = 0
    risk_off_sessions: int = 0
    portfolio_stops: int = 0
    previous_rebalance: date | None = None
    pending: dict[str, Decimal] | None = None
    liquidate_next: bool = False
    skipped: dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def drawdown_from_peak(self, equity: Rupees) -> Decimal:
        if self.peak_equity <= Rupees.zero():
            return Decimal(0)
        return (self.peak_equity.value - equity.value) / self.peak_equity.value


def _marks(panel: dict[str, pd.DataFrame], session: pd.Timestamp) -> dict[str, Decimal]:
    """Closing prices for every symbol that traded on ``session``."""
    return {
        symbol: Decimal(str(frame.loc[session, "close"]))
        for symbol, frame in panel.items()
        if session in frame.index
    }


def _breached_stop(
    state: _RunState,
    *,
    equity: Rupees,
    limit: Decimal | None,
    holding_anything: bool,
    session: date,
) -> bool:
    """True when the portfolio-level drawdown stop should close the book."""
    if limit is None or not holding_anything:
        return False
    drawdown = state.drawdown_from_peak(equity)
    if drawdown < limit:
        return False
    log.warning(
        "portfolio_stop_hit",
        session=session.isoformat(),
        drawdown=f"{drawdown:.2%}",
        limit=f"{limit:.2%}",
    )
    return True


def _validate_run_args(
    panel: dict[str, pd.DataFrame],
    initial_capital: Rupees,
    min_capital: Rupees | None,
    max_drawdown_stop: Decimal | None,
    strategy: object,
) -> None:
    if not panel:
        raise ValueError("panel is empty; nothing to backtest")
    if min_capital is not None and initial_capital < min_capital:
        raise ValueError(
            f"initial capital {initial_capital} is below minimum required capital {min_capital}"
        )
    if max_drawdown_stop is not None and not Decimal(0) < max_drawdown_stop < Decimal(1):
        raise ValueError(f"max_drawdown_stop must be between 0 and 1, got {max_drawdown_stop}")

    for required in ("is_rebalance_date", "target_weights"):
        if not callable(getattr(strategy, required, None)):
            raise TypeError(f"strategy must expose {required}(); got {type(strategy).__name__}")


def run_rebalance_backtest(
    *,
    strategy: object,
    panel: dict[str, pd.DataFrame],
    calculator: ChargeCalculator,
    initial_capital: Rupees,
    product: ProductType = ProductType.CNC,
    benchmark: pd.Series | None = None,
    regime: RegimeFilter | None = None,
    max_drawdown_stop: Decimal | None = None,
    is_eligible: Callable[[str, date], bool] | None = None,
    measure_from: pd.Timestamp | None = None,
    fill_model: FillModel | None = None,
    cash_yield_rate: float = 0.0,
    rebalance_band: Decimal | None = None,
    max_position_weight: Decimal | None = None,
    sectors: Mapping[str, str] | None = None,
    max_sector_exposure: Decimal | None = None,
    min_capital: Rupees | None = None,
    evaluate_on_rebalance_only: bool = False,
    cost_benchmark: bool = False,
    volatility_targeter: VolatilityTargeter | None = None,
) -> RebalanceResult:
    """Run a :class:`~trading_agent.strategy.protocols.PortfolioStrategy`.

    ``max_drawdown_stop`` is a fraction, e.g. ``Decimal("0.20")`` for 20%. When
    equity falls that far below its peak the book is closed and stays closed
    until the next rebalance date, which is the portfolio-level stop Varsity
    recommends for this strategy in place of per-stock stops.

    ``is_eligible`` gates buys on point-in-time index membership. Sells are
    never gated: a holding whose symbol left the index still has to be sold.

    ``measure_from`` splits the panel into history and measurement. Sessions
    before it are walked so a trailing ranking has bars to read, but nothing is
    traded and no equity point is recorded; from it onward the run is normal.

    ``fill_model`` models slippage and checks circuit price limits.

    ``cash_yield_rate`` accrues daily interest on idle cash balances, aligning
    with the Sharpe risk-free rate hurdle.

    ``rebalance_band`` defines the relative drift tolerance (e.g. 0.25 for +-25%)
    before an existing winner is trimmed back to target, bounding concentration.

    ``evaluate_on_rebalance_only`` evaluates the regime filter on rebalance dates
    only, eliminating whipsaws caused by single-day dips below the moving average.
    """
    _validate_run_args(panel, initial_capital, min_capital, max_drawdown_stop, strategy)

    portfolio = Portfolio(initial_capital, calculator)
    strategy_name = str(getattr(strategy, "name", "portfolio_strategy"))
    sessions = sorted({ts for frame in panel.values() for ts in frame.index})
    if not sessions:
        raise ValueError("panel contains no bars")

    state = _RunState(peak_equity=initial_capital)
    equity_points: list[float] = []

    for position, session in enumerate(sessions):
        # -- 0. history only: read the bars, trade nothing, record nothing ---
        if measure_from is not None and session < measure_from:
            continue

        # -- 1. execute what the previous session decided, at today's open ---
        if state.pending is not None or state.liquidate_next:
            _apply(
                portfolio=portfolio,
                panel=panel,
                session=session,
                targets=state.pending or {},
                strategy_name=strategy_name,
                product=product,
                on_skip=state.skip,
                is_eligible=is_eligible,
                fill_model=fill_model,
                rebalance_band=rebalance_band,
                max_position_weight=max_position_weight,
                sectors=sectors,
                max_sector_exposure=max_sector_exposure,
            )
            state.pending = None
            state.liquidate_next = False

        # -- 2. mark the book and accrue idle cash yield --------------------
        if cash_yield_rate > 0:
            portfolio.accrue_interest(cash_yield_rate)
        equity = portfolio.equity(_marks(panel, session))
        equity_points.append(float(equity.value))
        state.peak_equity = max(state.peak_equity, equity)

        # Nothing decided on the last session can ever fill: there is no next
        # open to fill at.
        if position == len(sessions) - 1:
            continue

        _decide(
            state,
            strategy=strategy,
            panel=panel,
            session=session,
            equity=equity,
            holding_anything=bool(portfolio.holdings),
            regime=regime,
            max_drawdown_stop=max_drawdown_stop,
            evaluate_on_rebalance_only=evaluate_on_rebalance_only,
            volatility_targeter=volatility_targeter,
        )

    measured = [s for s in sessions if s >= measure_from] if measure_from is not None else sessions
    equity_curve = pd.Series(equity_points, index=pd.DatetimeIndex(measured), name="equity")
    metrics = compute_metrics(
        equity_curve, portfolio.round_trips, total_charges=portfolio.total_charges
    )

    benchmark_metrics: PerformanceMetrics | None
    if cost_benchmark and benchmark is not None and not benchmark.empty:
        # The benchmark an investor could actually have bought: ETF entry
        # charges, expense ratio and terminal LTCG, rather than an index number
        # that no one can hold.
        from trading_agent.backtest.benchmark import cost_benchmark_curve

        benchmark_metrics = cost_benchmark_curve(
            benchmark,
            initial_capital=initial_capital,
            calculator=calculator,
            target_index=pd.DatetimeIndex(equity_curve.index),
        ).metrics
    else:
        benchmark_metrics = _benchmark_metrics(benchmark, equity_curve, initial_capital)

    log.info(
        "rebalance_backtest_complete",
        strategy=strategy_name,
        sessions=len(sessions),
        rebalances=state.rebalances,
        round_trips=len(portfolio.round_trips),
        risk_off_sessions=state.risk_off_sessions,
        portfolio_stops=state.portfolio_stops,
        total_return=f"{metrics.total_return:.2%}",
    )

    return RebalanceResult(
        strategy=strategy_name,
        equity_curve=equity_curve,
        metrics=metrics,
        round_trips=portfolio.round_trips,
        total_charges=portfolio.total_charges,
        initial_capital=initial_capital,
        final_equity=Rupees(Decimal(str(equity_points[-1]))) if equity_points else initial_capital,
        rebalances=state.rebalances,
        risk_off_sessions=state.risk_off_sessions,
        portfolio_stops=state.portfolio_stops,
        benchmark=benchmark_metrics,
        skipped=state.skipped,
        cash_yield_accrued=portfolio.cash_yield_accrued,
    )


def _decide(
    state: _RunState,
    *,
    strategy: object,
    panel: dict[str, pd.DataFrame],
    session: pd.Timestamp,
    equity: Rupees,
    holding_anything: bool,
    regime: RegimeFilter | None,
    max_drawdown_stop: Decimal | None,
    evaluate_on_rebalance_only: bool = False,
    volatility_targeter: VolatilityTargeter | None = None,
) -> None:
    """The gates settling whether to rebalance, stop, or hold."""
    session_date = _as_date(session)

    # -- 3. portfolio-level stop --------------------------------------------
    if _breached_stop(
        state,
        equity=equity,
        limit=max_drawdown_stop,
        holding_anything=holding_anything,
        session=session_date,
    ):
        state.portfolio_stops += 1
        state.liquidate_next = True
        return

    if evaluate_on_rebalance_only:
        # Check rebalance schedule first
        if not strategy.is_rebalance_date(  # type: ignore[attr-defined]
            session_date, previous=state.previous_rebalance
        ):
            return
        state.previous_rebalance = session_date
        state.rebalances += 1

        # Evaluate regime filter only on the rebalance date to prevent 1-day whipsaw
        if regime is not None and not regime.is_risk_on(session):
            state.risk_off_sessions += 1
            if regime.liquidate and holding_anything:
                state.liquidate_next = True
            return

        targets = strategy.target_weights(panel, on=session_date)  # type: ignore[attr-defined]
        if volatility_targeter is not None and targets:
            targets = volatility_targeter.apply(panel, targets, session)
        state.pending = targets
    else:
        # -- 4. regime filter ----------------------------------------------------
        if regime is not None and not regime.is_risk_on(session):
            state.risk_off_sessions += 1
            if regime.liquidate and holding_anything:
                state.liquidate_next = True
            return

        # -- 5. rebalance --------------------------------------------------------
        if not strategy.is_rebalance_date(  # type: ignore[attr-defined]
            session_date, previous=state.previous_rebalance
        ):
            return
        state.previous_rebalance = session_date
        state.rebalances += 1
        targets = strategy.target_weights(panel, on=session_date)  # type: ignore[attr-defined]
        if volatility_targeter is not None and targets:
            targets = volatility_targeter.apply(panel, targets, session)
        state.pending = targets


def _benchmark_metrics(
    benchmark: pd.Series | None, equity_curve: pd.Series, initial_capital: Rupees
) -> PerformanceMetrics | None:
    """Buy-and-hold on the benchmark, scaled to the same starting capital."""
    if benchmark is None or benchmark.empty:
        return None
    aligned = benchmark.reindex(equity_curve.index).ffill().dropna()
    if aligned.empty:
        return None
    scaled = aligned / float(aligned.iloc[0]) * float(initial_capital.value)
    return compute_metrics(scaled, [], total_charges=Rupees.zero())


def _trim_drift(
    *,
    portfolio: Portfolio,
    wanted: dict[str, Decimal],
    session: pd.Timestamp,
    equity: Rupees,
    marks: dict[str, Decimal],
    bar_info: Callable[[str], tuple[Decimal | None, Decimal | None, int | None]],
    strategy_name: str,
    fill_model: FillModel | None,
    rebalance_band: Decimal | None,
    max_position_weight: Decimal | None,
) -> None:
    """Trim holdings that drifted past the rebalance band or position weight ceiling."""
    if (rebalance_band is None and max_position_weight is None) or equity.value <= 0:
        return

    for symbol in list(portfolio.holdings):
        if symbol not in wanted:
            continue
        target_w = wanted[symbol]
        cap_w = max_position_weight if max_position_weight is not None else Decimal(1)
        ceiling_w = min(target_w * (Decimal(1) + (rebalance_band or Decimal("0.25"))), cap_w)
        holding = portfolio.holdings[symbol]
        mark = marks.get(symbol, holding.average_price)
        if mark <= Decimal(0):
            continue
        current_w = (Decimal(holding.quantity) * mark) / equity.value
        if current_w > ceiling_w:
            target_notional = equity.value * min(target_w, cap_w)
            target_qty = int(target_notional // mark)
            trim_qty = holding.quantity - target_qty
            if trim_qty > 0:
                price, prev_close, adv = bar_info(symbol)
                fill_price = price or mark
                if fill_model is not None and price is not None and prev_close is not None:
                    fill = fill_model.fill_at_open(
                        next_open=price,
                        previous_close=prev_close,
                        side=Side.SELL,
                        quantity=trim_qty,
                        average_daily_volume=adv,
                    )
                    if fill.filled:
                        fill_price = fill.price
                portfolio.sell(
                    symbol=symbol,
                    price=fill_price,
                    at=session.to_pydatetime(),
                    reason="trim_drift",
                    strategy=strategy_name,
                    quantity=trim_qty,
                )


def _collect_buy_candidates(
    *,
    portfolio: Portfolio,
    wanted: dict[str, Decimal],
    equity: Rupees,
    marks: dict[str, Decimal],
    session_date: date,
    is_eligible: Callable[[str, date], bool] | None,
    bar_info: Callable[[str], tuple[Decimal | None, Decimal | None, int | None]],
    on_skip: Callable[[str], None],
    max_position_weight: Decimal | None,
    sectors: Mapping[str, str] | None,
    max_sector_exposure: Decimal | None,
) -> list[tuple[str, Decimal]]:
    """Filter and size candidates satisfying eligibility and sector caps."""
    candidates: list[tuple[str, Decimal]] = []
    sector_exposure: dict[str, Decimal] = {}
    if sectors:
        for s, h in portfolio.holdings.items():
            sec = sectors.get(s)
            if sec and equity.value > 0:
                cur_w = (Decimal(h.quantity) * marks.get(s, h.average_price)) / equity.value
                sector_exposure[sec] = sector_exposure.get(sec, Decimal(0)) + cur_w

    for symbol, weight in sorted(wanted.items(), key=lambda pair: -pair[1]):
        if portfolio.holds(symbol):
            continue
        if is_eligible is not None and not is_eligible(symbol, session_date):
            on_skip("not an index member yet")
            continue
        price, prev_close, _ = bar_info(symbol)
        if price is None or prev_close is None:
            on_skip("no bar to enter on")
            continue

        eff_weight = min(weight, max_position_weight) if max_position_weight is not None else weight
        if sectors and max_sector_exposure is not None:
            sec = sectors.get(symbol)
            if sec:
                cur_sec_w = sector_exposure.get(sec, Decimal(0))
                avail_sec_w = max_sector_exposure - cur_sec_w
                if avail_sec_w < _MIN_WEIGHT:
                    on_skip(f"sector limit reached for {sec}")
                    continue
                eff_weight = min(eff_weight, avail_sec_w)
                sector_exposure[sec] = cur_sec_w + eff_weight

        candidates.append((symbol, eff_weight))
    return candidates


def _execute_candidate_buys(
    *,
    portfolio: Portfolio,
    candidates: list[tuple[str, Decimal]],
    equity: Rupees,
    session: pd.Timestamp,
    session_date: date,
    product: ProductType,
    bar_info: Callable[[str], tuple[Decimal | None, Decimal | None, int | None]],
    on_skip: Callable[[str], None],
    fill_model: FillModel | None,
) -> None:
    """Pro-rata fund and execute buys for qualified candidates."""
    total_requested = sum((Rupees(equity.value * w) for _, w in candidates), Rupees.zero())
    scale = Decimal(1)
    if total_requested > portfolio.cash and total_requested > Rupees.zero():
        scale = portfolio.cash.value / total_requested.value

    for symbol, eff_weight in candidates:
        price, prev_close, adv = bar_info(symbol)
        if price is None or prev_close is None:
            continue
        budget = Rupees(equity.value * eff_weight * scale)
        budget = min(budget, portfolio.cash)
        quantity = portfolio.affordable_quantity(
            price=price, budget=budget, product=product, on=session_date
        )
        if quantity <= 0:
            on_skip("budget below one share")
            continue

        fill_price = price
        if fill_model is not None:
            fill = fill_model.fill_at_open(
                next_open=price,
                previous_close=prev_close,
                side=Side.BUY,
                quantity=quantity,
                average_daily_volume=adv,
            )
            if not fill.filled:
                on_skip(fill.reason)
                continue
            fill_price = fill.price

        try:
            portfolio.buy(
                symbol=symbol,
                quantity=quantity,
                price=fill_price,
                at=session.to_pydatetime(),
                product=product,
                stop_loss=Decimal(0),
                target=Decimal(0),
            )
        except InsufficientCashError:
            on_skip("insufficient cash")


def _apply(
    *,
    portfolio: Portfolio,
    panel: dict[str, pd.DataFrame],
    session: pd.Timestamp,
    targets: dict[str, Decimal],
    strategy_name: str,
    product: ProductType,
    on_skip: Callable[[str], None],
    is_eligible: Callable[[str, date], bool] | None,
    fill_model: FillModel | None = None,
    rebalance_band: Decimal | None = None,
    max_position_weight: Decimal | None = None,
    sectors: Mapping[str, str] | None = None,
    max_sector_exposure: Decimal | None = None,
) -> None:
    """Move the book toward ``targets``, filling at ``session``'s open."""
    session_date = _as_date(session)
    wanted = {s: w for s, w in targets.items() if w >= _MIN_WEIGHT}

    if targets and not wanted:
        smallest = min(targets.values())
        log.warning(
            "all_targets_below_min_weight",
            session=session_date.isoformat(),
            targets=len(targets),
            smallest_weight=str(smallest),
            min_weight=str(_MIN_WEIGHT),
            detail=(
                "every target weight is below the tradeable floor, so the book goes "
                "flat. A universe of more than ~200 equally weighted names hits this; "
                "cap the holdings or raise capital."
            ),
        )
        on_skip("all targets below the minimum tradeable weight")

    def bar_info(symbol: str) -> tuple[Decimal | None, Decimal | None, int | None]:
        frame = panel.get(symbol)
        if frame is None or session not in frame.index:
            return None, None, None
        price = Decimal(str(frame.loc[session, "open"]))
        if price <= 0:
            return None, None, None
        loc = frame.index.get_loc(session)
        if isinstance(loc, int) and loc > 0:
            prev_close = Decimal(str(frame.iloc[loc - 1]["close"]))
        else:
            prev_close = price
        bar = frame.loc[session]
        adv = int(bar.get("avg_volume_20", bar.get("volume", 0)) or 0)
        return price, prev_close, adv

    # -- 1. sells (liquidate unwanted) ---------------------------------------
    _liquidate_unwanted(
        portfolio=portfolio,
        wanted=set(wanted),
        session=session,
        bar_info=bar_info,
        strategy_name=strategy_name,
        on_skip=on_skip,
        fill_model=fill_model,
    )

    # -- 2. mark the book and trim drifted holdings (F5, R6) -----------------
    marks = {s: h.average_price for s, h in portfolio.holdings.items()}
    for symbol in wanted:
        p, _, _ = bar_info(symbol)
        if p is not None:
            marks[symbol] = p
    equity = portfolio.equity(marks)

    _trim_drift(
        portfolio=portfolio,
        wanted=wanted,
        session=session,
        equity=equity,
        marks=marks,
        bar_info=bar_info,
        strategy_name=strategy_name,
        fill_model=fill_model,
        rebalance_band=rebalance_band,
        max_position_weight=max_position_weight,
    )

    if not wanted:
        return

    # -- 3. buys (screen eligibility, cap sectors, pro-rata cash) ------------
    equity = portfolio.equity(marks)
    candidates = _collect_buy_candidates(
        portfolio=portfolio,
        wanted=wanted,
        equity=equity,
        marks=marks,
        session_date=session_date,
        is_eligible=is_eligible,
        bar_info=bar_info,
        on_skip=on_skip,
        max_position_weight=max_position_weight,
        sectors=sectors,
        max_sector_exposure=max_sector_exposure,
    )

    if not candidates:
        return

    _execute_candidate_buys(
        portfolio=portfolio,
        candidates=candidates,
        equity=equity,
        session=session,
        session_date=session_date,
        product=product,
        bar_info=bar_info,
        on_skip=on_skip,
        fill_model=fill_model,
    )


def _liquidate_unwanted(
    *,
    portfolio: Portfolio,
    wanted: set[str],
    session: pd.Timestamp,
    bar_info: Callable[[str], tuple[Decimal | None, Decimal | None, int | None]],
    strategy_name: str,
    on_skip: Callable[[str], None],
    fill_model: FillModel | None = None,
) -> None:
    """Sell everything not in ``wanted``, at ``session``'s open."""
    for symbol in list(portfolio.holdings):
        if symbol in wanted:
            continue
        price, prev_close, adv = bar_info(symbol)
        if price is None or prev_close is None:
            # No bar to sell into: the position survives to the next session
            # rather than being closed at an invented price.
            on_skip("no bar to exit on")
            continue
        fill_price = price
        if fill_model is not None:
            fill = fill_model.fill_at_open(
                next_open=price,
                previous_close=prev_close,
                side=Side.SELL,
                quantity=portfolio.holdings[symbol].quantity,
                average_daily_volume=adv,
            )
            if not fill.filled:
                on_skip(fill.reason)
                continue
            fill_price = fill.price

        portfolio.sell(
            symbol=symbol,
            price=fill_price,
            at=session.to_pydatetime(),
            reason="rebalanced out",
            strategy=strategy_name,
        )
