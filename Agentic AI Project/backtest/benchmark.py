"""Purchasable ETF benchmark model.

Comparing a trading strategy against an uncosted, abstract index series is
structurally biased:
1. An investor cannot buy the index for zero friction. They buy an ETF like
   NIFTYBEES.
2. An ETF charges an ongoing expense ratio (~5 bps to 20 bps/year).
3. The initial ETF purchase pays delivery charges (brokerage, STT, stamp duty).
4. Terminal gains in delivery equity/ETF are subject to Long-Term Capital
   Gains (LTCG) tax (12.5% above the annual Rs 1.25 lakh exemption).

This module models a real, tradeable benchmark investment alongside the raw index.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import pandas as pd

from trading_agent.backtest.metrics import (
    DEFAULT_RISK_FREE_RATE,
    PerformanceMetrics,
    compute_metrics,
)
from trading_agent.costs.calculator import ChargeCalculator
from trading_agent.costs.tax import CapitalGainsRates
from trading_agent.domain.enums import ProductType, Side
from trading_agent.domain.money import Rupees
from trading_agent.domain.trade import RoundTrip

__all__ = ["PurchasableBenchmark", "cost_benchmark_curve", "session_dates"]

_DAYS_IN_YEAR = 365

_EXCHANGE_TZ = "Asia/Kolkata"


def session_dates(index: pd.Index) -> pd.DatetimeIndex:
    """Calendar dates of NSE sessions, timezone-free.

    The bar store stamps sessions at IST midnight (tz-aware), while equity
    curves read back from JSON are plain dates. Pandas treats the two as
    different labels, so a reindex between them matches nothing.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(index))
    if idx.tz is not None:
        idx = idx.tz_convert(_EXCHANGE_TZ).tz_localize(None)
    return idx.normalize()


#: Standard expense ratio for NIFTYBEES / liquid Indian index ETFs (5 bps/year).
DEFAULT_ETF_EXPENSE_RATIO = 0.0005


@dataclass(frozen=True, slots=True)
class PurchasableBenchmark:
    """A tradeable benchmark investment holding an index ETF like NIFTYBEES."""

    equity_curve: pd.Series
    metrics: PerformanceMetrics
    entry_charges: Rupees
    expense_drag: Rupees
    terminal_tax: Rupees


def cost_benchmark_curve(
    benchmark_series: pd.Series,
    *,
    initial_capital: Rupees,
    calculator: ChargeCalculator,
    target_index: pd.DatetimeIndex | None = None,
    expense_ratio: float | None = None,
    expense_ratio_bps: Decimal | None = None,
    rates: CapitalGainsRates | None = None,
) -> PurchasableBenchmark:
    """Simulate purchasing and holding a low-cost ETF with realistic Indian drag.

    1. Entry transaction costs (delivery brokerage, STT, turnover charges, GST, stamp duty)
    2. Ongoing expense ratio deduction (default 5 bps/year, aligned with NIFTYBEES)
    3. Terminal LTCG tax liability if held for more than 1 year (> 365 days)
    """
    if benchmark_series.empty:
        raise ValueError("benchmark series is empty")

    if target_index is not None:
        if target_index.empty:
            raise ValueError("target index is empty")
        bench = benchmark_series.copy()
        bench.index = session_dates(bench.index)
        bench = bench[~bench.index.duplicated(keep="last")].sort_index()
        aligned = bench.reindex(session_dates(target_index)).ffill().dropna()
    else:
        aligned = benchmark_series.dropna()

    if aligned.empty:
        raise ValueError("benchmark has no overlapping sessions with target index")

    start_session = aligned.index[0]
    first_price = Decimal(str(aligned.iloc[0]))
    qty = max(1, int(initial_capital.value // first_price))
    entry_charges = calculator.for_notional(
        price=first_price,
        quantity=qty,
        side=Side.BUY,
        product=ProductType.CNC,
        on=start_session.date(),
    ).total

    invested_capital = initial_capital - entry_charges
    if invested_capital <= Rupees.zero():
        raise ValueError(
            f"initial capital {initial_capital} cannot cover entry charges {entry_charges}"
        )

    # 2. Daily Expense Ratio Drag
    if expense_ratio is not None:
        ann_expense_ratio = expense_ratio
    elif expense_ratio_bps is not None:
        ann_expense_ratio = float(expense_ratio_bps) / 10000.0
    else:
        ann_expense_ratio = DEFAULT_ETF_EXPENSE_RATIO

    daily_expense_factor = (1.0 - ann_expense_ratio) ** (1.0 / 250.0)

    benchmark_returns = aligned.pct_change().fillna(0.0)
    normalized_returns = benchmark_returns.iloc[1:]

    current_equity = float(invested_capital.value)
    equity_points = [current_equity]

    for ret in normalized_returns:
        current_equity = current_equity * (1.0 + float(ret)) * daily_expense_factor
        equity_points.append(current_equity)

    costed_curve = pd.Series(equity_points, index=aligned.index, name="benchmark_equity")
    raw_final = costed_curve.iloc[-1]
    gross_final = float(aligned.iloc[-1] / aligned.iloc[0] * float(invested_capital.value))
    expense_drag = Rupees(Decimal(str(max(0.0, gross_final - raw_final))))

    # 3. Terminal LTCG Tax
    rates = rates or CapitalGainsRates.current()
    end_session = aligned.index[-1]
    holding_days = max((end_session - start_session).days, 1)

    # Synthetic round trip for LTCG tax calculation
    terminal_net_pnl = Rupees(Decimal(str(raw_final)) - initial_capital.value)
    trips: list[RoundTrip] = []
    if holding_days > _DAYS_IN_YEAR and terminal_net_pnl > Rupees.zero():
        # Represent as a long-term round trip
        trips.append(
            RoundTrip(
                symbol="NIFTYBEES",
                side=Side.BUY,
                quantity=1,
                entry_price=initial_capital.value,
                exit_price=Decimal(str(raw_final)),
                entry_at=start_session.to_pydatetime(),
                exit_at=end_session.to_pydatetime(),
                product=ProductType.CNC,
                total_charges=entry_charges,
            )
        )

    metrics = compute_metrics(
        costed_curve,
        trips,
        total_charges=entry_charges,
        risk_free_rate=DEFAULT_RISK_FREE_RATE,
        tax_rates=rates,
    )

    return PurchasableBenchmark(
        equity_curve=costed_curve,
        metrics=metrics,
        entry_charges=entry_charges,
        expense_drag=expense_drag,
        terminal_tax=metrics.total_tax,
    )
