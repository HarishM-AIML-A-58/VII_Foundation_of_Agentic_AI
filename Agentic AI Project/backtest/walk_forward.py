"""Walk-forward evaluation.

A single in-sample backtest measures curve fitting. Walk-forward -- fit on a
window, test on the window after it, roll -- is the cheapest defence, and the
only result worth quoting.

The discipline this module enforces is narrow and absolute: **the strategy is
built from the training window and evaluated only on the window after it.**
``strategy_factory`` receives the training panel and nothing else. A strategy
that wants to fit parameters does it there; one that does not simply ignores
the argument. Either way the test window is untouched at fit time, which is
the whole point.

Two numbers come out, and the second matters more:

``combined`` metrics
    Every test window chained into one out-of-sample equity curve. This is the
    number to quote. It is the only one nothing was fitted on.
``efficiency``
    Out-of-sample return divided by in-sample return. Near 1.0 means the
    strategy generalised. Near zero -- or negative -- means the in-sample
    result was fitting, and the honest reading is that there is no strategy
    here. A high in-sample return with low efficiency is the single most
    common way a backtest lies.

Windows roll forward by the *test* length, not the train length, so every
session after the first training window is tested exactly once and none is
tested twice.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

import pandas as pd

from trading_agent.backtest.engine import BacktestResult, run_backtest
from trading_agent.backtest.fill_model import FillModel
from trading_agent.backtest.metrics import PerformanceMetrics, compute_metrics
from trading_agent.costs import ChargeCalculator
from trading_agent.domain.enums import ProductType
from trading_agent.domain.money import Rupees
from trading_agent.domain.trade import RoundTrip
from trading_agent.observability import get_logger

__all__ = ["WalkForwardResult", "WalkForwardWindow", "walk_forward"]

log = get_logger(__name__)

#: A window with fewer sessions than this cannot produce a meaningful metric:
#: Sharpe over three days is noise wearing a number's clothes.
MIN_TEST_SESSIONS = 20

Panel = Mapping[str, pd.DataFrame]
StrategyFactory = Callable[[Panel], object]


@dataclass(frozen=True, slots=True)
class WalkForwardWindow:
    """One train/test pair and what the test window produced."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    result: BacktestResult

    @property
    def test_return(self) -> float:
        return self.result.metrics.total_return

    @property
    def label(self) -> str:
        return f"{self.test_start:%Y-%m} to {self.test_end:%Y-%m}"


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    """Every window, and the out-of-sample record they chain into."""

    windows: list[WalkForwardWindow]
    combined: PerformanceMetrics
    combined_curve: pd.Series
    in_sample_return: float
    round_trips: list[RoundTrip] = field(default_factory=list)

    @property
    def out_of_sample_return(self) -> float:
        return self.combined.total_return

    @property
    def efficiency(self) -> float:
        """Out-of-sample return over in-sample return.

        Zero when the in-sample run lost money -- the ratio stops meaning
        anything once the denominator is negative, and reporting a tidy
        positive number there would be worse than reporting nothing.
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

        A strategy whose entire return came from one window is a strategy that
        was lucky once, whatever its total says.
        """
        return self.profitable_windows / len(self.windows) if self.windows else 0.0

    def summary_rows(self) -> list[tuple[str, str]]:
        return [
            ("Windows", str(len(self.windows))),
            ("In-sample return", f"{self.in_sample_return:+.2%}"),
            ("Out-of-sample return", f"{self.out_of_sample_return:+.2%}"),
            ("Walk-forward efficiency", f"{self.efficiency:.2f}"),
            ("Profitable windows", f"{self.profitable_windows}/{len(self.windows)}"),
            ("Consistency", f"{self.consistency:.0%}"),
            ("OOS Sharpe", f"{self.combined.sharpe:.2f}"),
            ("OOS max drawdown", f"{self.combined.max_drawdown:.2%}"),
            ("OOS trades", str(self.combined.trades)),
        ]


def _sessions(panel: Panel) -> list[pd.Timestamp]:
    return sorted({ts for frame in panel.values() for ts in frame.index})


def _slice(panel: Panel, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Bars in ``[start, end)``, dropping symbols left with nothing."""
    sliced: dict[str, pd.DataFrame] = {}
    for symbol, frame in panel.items():
        window = frame[(frame.index >= start) & (frame.index < end)]
        if not window.empty:
            sliced[symbol] = window
    return sliced


def _windows(
    sessions: list[pd.Timestamp], *, train_months: int, test_months: int
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """``(train_start, test_start, test_end)`` triples, rolling by test length."""
    first, last = sessions[0], sessions[-1]
    boundaries: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []

    train_start = first
    while True:
        test_start = train_start + pd.DateOffset(months=train_months)
        test_end = test_start + pd.DateOffset(months=test_months)
        if test_start > last:
            break
        boundaries.append((train_start, test_start, min(test_end, last + pd.Timedelta(days=1))))
        if test_end > last:
            break
        train_start = train_start + pd.DateOffset(months=test_months)
    return boundaries


def _chain(curves: list[pd.Series], initial: Rupees) -> pd.Series:
    """Compound each window's returns onto one continuous equity curve.

    Windows are run independently -- each starts from the same capital, so
    each is comparable -- but the out-of-sample record is what an account
    would actually have done, which means compounding them. Concatenating the
    raw curves instead would show equity jumping back to its starting value
    at every window boundary.
    """
    equity = float(initial.value)
    points: list[float] = []
    index: list[pd.Timestamp] = []

    for curve in curves:
        if curve.empty:
            continue
        base = float(curve.iloc[0])
        if base == 0:
            continue
        for timestamp, value in curve.items():
            points.append(equity * float(value) / base)
            index.append(timestamp)  # type: ignore[arg-type]
        equity = points[-1]

    return pd.Series(points, index=pd.DatetimeIndex(index), name="equity")


def walk_forward(
    *,
    strategy_factory: StrategyFactory,
    panel: Panel,
    calculator: ChargeCalculator,
    fill_model: FillModel,
    initial_capital: Rupees,
    train_months: int = 24,
    test_months: int = 6,
    max_positions: int = 5,
    product: ProductType = ProductType.CNC,
    risk_free_rate: float | None = None,
    is_eligible: Callable[[str, date], bool] | None = None,
) -> WalkForwardResult:
    """Roll a train/test window across ``panel`` and report the out-of-sample record.

    ``strategy_factory`` is called once per window with the *training* slice.
    A strategy that fits parameters must do so there and nowhere else.

    ``is_eligible`` is forwarded to every window so the out-of-sample record is
    gated on point-in-time index membership too. An unforwarded predicate would
    make the in-sample and out-of-sample runs answer different questions, which
    is exactly the comparison walk-forward exists to make trustworthy.
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

    windows: list[WalkForwardWindow] = []
    curves: list[pd.Series] = []
    trips: list[RoundTrip] = []

    for train_start, test_start, test_end in boundaries:
        train_panel = _slice(panel, train_start, test_start)
        test_panel = _slice(panel, test_start, test_end)

        test_sessions = len(_sessions(test_panel))
        if test_sessions < MIN_TEST_SESSIONS:
            # Sharpe over a handful of days is noise. Dropped loudly rather
            # than folded in, so the window count reflects what was measured.
            log.warning(
                "walk_forward_window_skipped",
                test_start=f"{test_start:%Y-%m-%d}",
                sessions=test_sessions,
                reason="too few sessions to measure",
            )
            continue
        if not train_panel:
            log.warning(
                "walk_forward_window_skipped",
                test_start=f"{test_start:%Y-%m-%d}",
                reason="no training data",
            )
            continue

        strategy = strategy_factory(train_panel)
        result = run_backtest(
            strategy=strategy,
            panel=test_panel,
            calculator=calculator,
            fill_model=fill_model,
            initial_capital=initial_capital,
            max_positions=max_positions,
            product=product,
            is_eligible=is_eligible,
        )
        windows.append(
            WalkForwardWindow(
                train_start=train_start,
                train_end=test_start,
                test_start=result.metrics.start,
                test_end=result.metrics.end,
                result=result,
            )
        )
        curves.append(result.equity_curve)
        trips.extend(result.round_trips)
        log.info(
            "walk_forward_window",
            train=f"{train_start:%Y-%m} to {test_start:%Y-%m}",
            test=f"{test_start:%Y-%m} to {test_end:%Y-%m}",
            total_return=round(result.metrics.total_return, 4),
            trades=result.metrics.trades,
        )

    if not windows:
        raise ValueError(
            "no window had enough test sessions to measure; shorten train_months "
            "or supply a longer panel"
        )

    combined_curve = _chain(curves, initial_capital)
    total_charges = Rupees(sum((w.result.metrics.total_charges.value for w in windows), Decimal(0)))
    metric_kwargs = {"total_charges": total_charges}
    if risk_free_rate is not None:
        metric_kwargs["risk_free_rate"] = risk_free_rate  # type: ignore[assignment]
    combined = compute_metrics(combined_curve, trips, **metric_kwargs)  # type: ignore[arg-type]

    # The in-sample comparison: the same strategy over the whole panel at once,
    # fitted on everything. This is the number a naive backtest would quote,
    # and the ratio between the two is the point of the exercise.
    in_sample = run_backtest(
        strategy=strategy_factory(panel),
        panel=dict(panel),
        calculator=calculator,
        fill_model=fill_model,
        initial_capital=initial_capital,
        max_positions=max_positions,
        product=product,
        is_eligible=is_eligible,
    )

    # Not `result`: that name belongs to the per-window BacktestResult above,
    # and reusing it here makes the return type quietly ambiguous.
    outcome = WalkForwardResult(
        windows=windows,
        combined=combined,
        combined_curve=combined_curve,
        in_sample_return=in_sample.metrics.total_return,
        round_trips=trips,
    )
    log.info(
        "walk_forward_complete",
        windows=len(windows),
        in_sample=round(outcome.in_sample_return, 4),
        out_of_sample=round(outcome.out_of_sample_return, 4),
        efficiency=round(outcome.efficiency, 3),
        consistency=round(outcome.consistency, 3),
    )
    return outcome
