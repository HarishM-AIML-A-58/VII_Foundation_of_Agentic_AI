"""Walk-forward for the scheduled-rebalance engine.

Separate from :mod:`trading_agent.backtest.walk_forward` because a portfolio
strategy is rolled differently, and pretending otherwise would produce a number
that looks comparable and is not.

**The difference that matters.** The signal engine's walk-forward calls a factory
with each *training* slice, so a strategy that fits stop/target parameters fits
them there and nowhere else. A ranking portfolio has no parameters to fit in that
sense -- it has a lookback, and the lookback is data it *reads*, not a parameter
it *learns*. So the honest question here is different: not "did the fitted
parameters generalise" but "did the ranking keep working in periods the
researcher had not looked at when choosing the configuration".

That makes the training window's job to supply history rather than to fit. A
twelve-month ranking needs twelve months of bars before its first rebalance, so
each test window is run over a panel that **includes** its training history and
an equity curve measured only across the test period. Slicing the panel to the
test window alone would leave the ranking unscoreable for its first year and
quietly turn a five-year test into four.

**Efficiency is still the number to quote**, and it means the same thing: out-of
sample return over in-sample. Near 1.0, the configuration generalised. Near zero,
the in-sample figure was a property of the window it was chosen on.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

import pandas as pd

from trading_agent.backtest.fill_model import FillModel
from trading_agent.backtest.metrics import PerformanceMetrics, compute_metrics
from trading_agent.backtest.rebalance import (
    RebalanceResult,
    RegimeFilter,
    run_rebalance_backtest,
)
from trading_agent.backtest.volatility_target import VolatilityTargeter
from trading_agent.backtest.walk_forward import _chain, _sessions, _windows
from trading_agent.costs.calculator import ChargeCalculator
from trading_agent.domain.enums import ProductType
from trading_agent.domain.money import Rupees
from trading_agent.domain.trade import RoundTrip
from trading_agent.observability import get_logger

__all__ = [
    "RebalanceWalkForwardResult",
    "RebalanceWalkForwardWindow",
    "walk_forward_rebalance",
]

log = get_logger(__name__)

Panel = dict[str, pd.DataFrame]
#: Called once per window. Takes nothing: a portfolio strategy's configuration
#: is chosen by the researcher, not fitted to the training slice, and a factory
#: that received the slice would invite fitting it there by accident.
PortfolioFactory = Callable[[], object]


@dataclass(frozen=True, slots=True)
class RebalanceWalkForwardWindow:
    """One train/test pair and what the test period produced."""

    train_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    result: RebalanceResult

    @property
    def test_return(self) -> float:
        return self.result.metrics.total_return

    @property
    def label(self) -> str:
        return f"{self.test_start:%Y-%m} to {self.test_end:%Y-%m}"


@dataclass(frozen=True, slots=True)
class RebalanceWalkForwardResult:
    """Every window, and the out-of-sample record they chain into."""

    windows: list[RebalanceWalkForwardWindow]
    combined: PerformanceMetrics
    combined_curve: pd.Series
    in_sample_return: float
    round_trips: list[RoundTrip] = field(default_factory=list)
    #: Buy-and-hold over the same out-of-sample span, so the comparison is
    #: like-for-like. Quoting an out-of-sample return against an in-sample
    #: benchmark is a category error that always favours the strategy.
    benchmark_out_of_sample: PerformanceMetrics | None = None

    @property
    def out_of_sample_return(self) -> float:
        return self.combined.total_return

    @property
    def efficiency(self) -> float:
        """Out-of-sample over in-sample. Zero when in-sample lost money.

        The ratio stops meaning anything once the denominator is negative, and
        reporting a tidy positive number there would be worse than nothing.
        """
        if self.in_sample_return <= 0:
            return 0.0
        return self.out_of_sample_return / self.in_sample_return

    @property
    def profitable_windows(self) -> int:
        return sum(1 for w in self.windows if w.test_return > 0)

    @property
    def consistency(self) -> float:
        """Fraction of test windows that made money.

        A strategy whose entire return came from one window was lucky once,
        whatever its total says.
        """
        return self.profitable_windows / len(self.windows) if self.windows else 0.0

    @property
    def beat_benchmark_out_of_sample(self) -> bool | None:
        if self.benchmark_out_of_sample is None:
            return None
        return self.out_of_sample_return > self.benchmark_out_of_sample.total_return

    def summary_rows(self) -> list[tuple[str, str]]:
        rows = [
            ("Windows", str(len(self.windows))),
            ("In-sample return", f"{self.in_sample_return:+.2%}"),
            ("Out-of-sample return", f"{self.out_of_sample_return:+.2%}"),
            ("Walk-forward efficiency", f"{self.efficiency:.2f}"),
            ("Profitable windows", f"{self.profitable_windows}/{len(self.windows)}"),
            ("Consistency", f"{self.consistency:.0%}"),
            ("OOS Sharpe", f"{self.combined.sharpe:.2f}"),
            ("OOS max drawdown", f"{self.combined.max_drawdown:+.2%}"),
            ("OOS round trips", str(len(self.round_trips))),
        ]
        if self.benchmark_out_of_sample is not None:
            rows.append(
                (
                    "Benchmark over the same span",
                    f"{self.benchmark_out_of_sample.total_return:+.2%}",
                )
            )
        return rows


def walk_forward_rebalance(
    *,
    strategy_factory: PortfolioFactory,
    panel: Panel,
    calculator: ChargeCalculator,
    initial_capital: Rupees,
    train_months: int = 24,
    test_months: int = 6,
    product: ProductType = ProductType.CNC,
    benchmark: pd.Series | None = None,
    regime: RegimeFilter | None = None,
    max_drawdown_stop: Decimal | None = None,
    is_eligible: Callable[[str, date], bool] | None = None,
    fill_model: FillModel | None = None,
    cash_yield_rate: float = 0.0,
    rebalance_band: Decimal | None = None,
    max_position_weight: Decimal | None = None,
    sectors: Mapping[str, str] | None = None,
    max_sector_exposure: Decimal | None = None,
    evaluate_on_rebalance_only: bool = False,
    continuous_capital: bool = False,
    volatility_targeter: VolatilityTargeter | None = None,
) -> RebalanceWalkForwardResult:
    """Roll a train/test window across ``panel`` and report the out-of-sample record.

    Each window's panel spans ``[train_start, test_end)`` so the ranking has its
    lookback available, while the equity curve is measured over
    ``[test_start, test_end)`` only. A fresh strategy instance is built per
    window, so per-instance state -- the thin-universe warning latch, for one --
    cannot leak between windows and make later ones behave differently.

    ``continuous_capital`` allows running windows where each test window inherits
    compounded capital from the previous window, correctly pricing non-linear
    brokerage caps and flat DP fees.
    """
    if train_months < 1:
        raise ValueError(f"train_months must be at least 1, got {train_months}")
    if test_months < 1:
        raise ValueError(f"test_months must be at least 1, got {test_months}")
    if not panel:
        raise ValueError("panel is empty; nothing to walk forward over")

    sessions = _sessions(panel)
    if not sessions:
        raise ValueError("panel contains no bars")

    boundaries = _windows(sessions, train_months=train_months, test_months=test_months)
    if not boundaries:
        raise ValueError(
            f"{len(sessions)} sessions from {sessions[0]:%Y-%m-%d} to {sessions[-1]:%Y-%m-%d} "
            f"cannot fit a {train_months}-month train plus a {test_months}-month test window"
        )

    windows: list[RebalanceWalkForwardWindow] = []
    curves: list[pd.Series] = []
    trips: list[RoundTrip] = []

    running_capital = initial_capital
    for train_start, test_start, test_end in boundaries:
        # The panel carries the training history so the ranking is scoreable
        # from the first test session; the curve is trimmed to the test span so
        # the training period contributes no return.
        window_panel = _slice_with_history(panel, train_start, test_end)
        if not window_panel:
            continue

        window_capital = running_capital if continuous_capital else initial_capital

        result = run_rebalance_backtest(
            strategy=strategy_factory(),
            panel=window_panel,
            calculator=calculator,
            initial_capital=window_capital,
            product=product,
            regime=regime,
            max_drawdown_stop=max_drawdown_stop,
            is_eligible=is_eligible,
            measure_from=test_start,
            fill_model=fill_model,
            cash_yield_rate=cash_yield_rate,
            rebalance_band=rebalance_band,
            max_position_weight=max_position_weight,
            sectors=sectors,
            max_sector_exposure=max_sector_exposure,
            evaluate_on_rebalance_only=evaluate_on_rebalance_only,
            volatility_targeter=volatility_targeter,
        )
        if result.equity_curve.empty:
            continue

        if continuous_capital:
            running_capital = result.final_equity

        windows.append(
            RebalanceWalkForwardWindow(
                train_start=train_start,
                test_start=test_start,
                test_end=test_end,
                result=result,
            )
        )
        curves.append(result.equity_curve)
        trips.extend(result.round_trips)
        log.info(
            "rebalance_walk_forward_window",
            train=f"{train_start:%Y-%m} to {test_start:%Y-%m}",
            test=f"{test_start:%Y-%m} to {test_end:%Y-%m}",
            total_return=round(result.metrics.total_return, 4),
            round_trips=len(result.round_trips),
        )

    if not windows:
        raise ValueError("no window produced a usable equity curve")

    combined_curve = _chain(curves, initial_capital)
    total_charges = Rupees(sum((t.total_charges.value for t in trips), Decimal(0)))
    combined = compute_metrics(combined_curve, trips, total_charges=total_charges)

    # In-sample: the same configuration over the whole panel at once. This is
    # the number a naive backtest quotes, and the ratio between the two is the
    # entire point of the exercise.
    in_sample = run_rebalance_backtest(
        strategy=strategy_factory(),
        panel=dict(panel),
        calculator=calculator,
        initial_capital=initial_capital,
        product=product,
        regime=regime,
        max_drawdown_stop=max_drawdown_stop,
        is_eligible=is_eligible,
        fill_model=fill_model,
        cash_yield_rate=cash_yield_rate,
        rebalance_band=rebalance_band,
        max_position_weight=max_position_weight,
        sectors=sectors,
        max_sector_exposure=max_sector_exposure,
        evaluate_on_rebalance_only=evaluate_on_rebalance_only,
        volatility_targeter=volatility_targeter,
    )

    outcome = RebalanceWalkForwardResult(
        windows=windows,
        combined=combined,
        combined_curve=combined_curve,
        in_sample_return=in_sample.metrics.total_return,
        round_trips=trips,
        benchmark_out_of_sample=_benchmark_over(benchmark, combined_curve, initial_capital),
    )
    log.info(
        "rebalance_walk_forward_complete",
        windows=len(windows),
        in_sample=round(outcome.in_sample_return, 4),
        out_of_sample=round(outcome.out_of_sample_return, 4),
        efficiency=round(outcome.efficiency, 3),
        consistency=round(outcome.consistency, 3),
    )
    return outcome


def _slice_with_history(panel: Panel, start: pd.Timestamp, end: pd.Timestamp) -> Panel:
    """Bars in ``[start, end)`` for every symbol that has any."""
    sliced: Panel = {}
    for symbol, frame in panel.items():
        index = pd.DatetimeIndex(frame.index)
        window = frame[(index >= start) & (index < end)]
        if not window.empty:
            sliced[symbol] = window
    return sliced


def _benchmark_over(
    benchmark: pd.Series | None, curve: pd.Series, initial: Rupees
) -> PerformanceMetrics | None:
    """Buy-and-hold across exactly the out-of-sample span.

    Aligned to the combined curve rather than to the whole panel: comparing an
    out-of-sample return against a benchmark measured over the full history
    credits the strategy with the years it was not tested on.
    """
    if benchmark is None or benchmark.empty or curve.empty:
        return None
    aligned = benchmark.reindex(curve.index).ffill().dropna()
    if aligned.empty:
        return None
    scaled = aligned / float(aligned.iloc[0]) * float(initial.value)
    return compute_metrics(scaled, [], total_charges=Rupees.zero())
