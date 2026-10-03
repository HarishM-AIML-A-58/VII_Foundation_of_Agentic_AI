"""Performance metrics.

All computed on **net** returns, after charges and slippage. Gross metrics on
Indian equities routinely flatter a strategy that STT alone would sink, so
gross numbers are not produced at all -- a figure that cannot be reported
cannot be quoted by accident.

The risk-free rate defaults to Indian short-term government yields. Using a US
rate here would overstate Sharpe substantially, since the two differ by
several percentage points.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal

import pandas as pd

from trading_agent.costs.tax import CapitalGainsRates, CapitalGainsTax, TaxLiability
from trading_agent.domain.money import Rupees
from trading_agent.domain.trade import RoundTrip

__all__ = [
    "DEFAULT_RISK_FREE_RATE",
    "TRADING_DAYS_PER_YEAR",
    "PerformanceMetrics",
    "compute_metrics",
    "max_drawdown",
]

#: NSE sessions in a typical year, after weekends and ~15 exchange holidays.
TRADING_DAYS_PER_YEAR = 250

#: Approximate Indian short-term government yield.
DEFAULT_RISK_FREE_RATE = 0.065


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """What a strategy actually did, net of everything."""

    start: pd.Timestamp
    end: pd.Timestamp
    initial_equity: Rupees
    final_equity: Rupees

    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    max_drawdown_days: int
    volatility: float

    trades: int
    win_rate: float
    profit_factor: float
    average_win: Rupees
    average_loss: Rupees
    total_charges: Rupees

    total_tax: Rupees = field(default_factory=Rupees.zero)
    after_tax_final_equity: Rupees | None = None
    after_tax_total_return: float | None = None
    after_tax_cagr: float | None = None
    tax_liabilities: dict[int, TaxLiability] | None = None

    @property
    def charges_as_pct_of_initial(self) -> float:
        if self.initial_equity.is_zero():
            return 0.0
        return float(self.total_charges.value / self.initial_equity.value)

    def summary_rows(self) -> list[tuple[str, str]]:
        """Ordered, human-readable rows for CLI and report rendering."""
        rows = [
            ("Period", f"{self.start:%Y-%m-%d} to {self.end:%Y-%m-%d}"),
            ("Initial equity", str(self.initial_equity)),
            ("Final equity", str(self.final_equity)),
            ("Total return", f"{self.total_return:+.2%}"),
            ("CAGR", f"{self.cagr:+.2%}"),
            ("Volatility (ann.)", f"{self.volatility:.2%}"),
            ("Sharpe", f"{self.sharpe:.2f}"),
            ("Sortino", f"{self.sortino:.2f}"),
            ("Max drawdown", f"{self.max_drawdown:.2%}"),
            ("Longest drawdown", f"{self.max_drawdown_days} days"),
            ("Trades", str(self.trades)),
            ("Win rate", f"{self.win_rate:.1%}"),
            ("Profit factor", f"{self.profit_factor:.2f}"),
            ("Average win", str(self.average_win)),
            ("Average loss", str(self.average_loss)),
            ("Total charges", str(self.total_charges)),
            ("Charges / initial", f"{self.charges_as_pct_of_initial:.2%}"),
        ]
        if self.total_tax > Rupees.zero() or self.after_tax_total_return is not None:
            rows.extend(
                [
                    ("Total tax (est.)", str(self.total_tax)),
                    (
                        "After-tax return",
                        f"{self.after_tax_total_return:+.2%}"
                        if self.after_tax_total_return is not None
                        else "N/A",
                    ),
                    (
                        "After-tax CAGR",
                        f"{self.after_tax_cagr:+.2%}" if self.after_tax_cagr is not None else "N/A",
                    ),
                ]
            )
        return rows


def _max_drawdown(equity: pd.Series) -> tuple[float, int]:
    """Deepest peak-to-trough fall, and the longest time spent below a peak."""
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    deepest = float(drawdown.min()) if len(drawdown) else 0.0

    longest = 0
    current = 0
    for value in (equity < running_peak).to_numpy():
        current = current + 1 if value else 0
        longest = max(longest, current)
    return deepest, longest


max_drawdown = _max_drawdown


def compute_metrics(
    equity_curve: pd.Series,
    round_trips: list[RoundTrip],
    *,
    total_charges: Rupees,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    tax_rates: CapitalGainsRates | None = None,
) -> PerformanceMetrics:
    """Metrics from a daily equity curve and the trades that produced it."""
    if equity_curve.empty:
        raise ValueError("equity curve is empty; nothing to measure")

    initial = Rupees(Decimal(str(equity_curve.iloc[0])))
    final = Rupees(Decimal(str(equity_curve.iloc[-1])))

    returns = equity_curve.pct_change().dropna()
    total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0)

    days = max((equity_curve.index[-1] - equity_curve.index[0]).days, 1)
    years = days / 365.25
    if total_return <= -1.0:
        cagr = -1.0
    elif years > 0:
        cagr = (1.0 + total_return) ** (1.0 / years) - 1.0
    else:
        cagr = 0.0

    volatility = (
        float(returns.std() * math.sqrt(TRADING_DAYS_PER_YEAR)) if len(returns) > 1 else 0.0
    )

    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = returns - daily_rf
    sharpe = (
        float(excess.mean() / excess.std() * math.sqrt(TRADING_DAYS_PER_YEAR))
        if len(excess) > 1 and float(excess.std()) > 0
        else 0.0
    )

    # Sortino penalises only downside deviation: upside volatility is not risk.
    # Measured as downside semi-deviation over all N periods, not merely the
    # subset of losing days.
    downside_sq = (excess.clip(upper=0.0)) ** 2
    downside_dev = math.sqrt(float(downside_sq.mean())) if len(downside_sq) > 0 else 0.0
    sortino = (
        float(excess.mean() / downside_dev * math.sqrt(TRADING_DAYS_PER_YEAR))
        if downside_dev > 0
        else 0.0
    )

    max_dd, dd_days = _max_drawdown(equity_curve)

    wins = [t for t in round_trips if t.is_winner]
    losses = [t for t in round_trips if not t.is_winner]
    gross_profit = sum((t.net_pnl.value for t in wins), Decimal(0))
    gross_loss = abs(sum((t.net_pnl.value for t in losses), Decimal(0)))

    # Capital gains tax liability (FY2024-25 regime: 20% STCG, 12.5% LTCG)
    tax_calc = CapitalGainsTax(tax_rates)
    liabilities = tax_calc.liability_by_year(round_trips) if round_trips else {}
    total_tax = Rupees(sum((liab.total_tax.value for liab in liabilities.values()), Decimal(0)))
    after_tax_final = final - total_tax
    after_tax_return = (
        float(after_tax_final.value / initial.value) - 1.0 if initial.value > 0 else 0.0
    )
    if after_tax_return <= -1.0:
        after_tax_cagr = -1.0
    elif years > 0:
        after_tax_cagr = (1.0 + after_tax_return) ** (1.0 / years) - 1.0
    else:
        after_tax_cagr = 0.0

    return PerformanceMetrics(
        start=equity_curve.index[0],
        end=equity_curve.index[-1],
        initial_equity=initial,
        final_equity=final,
        total_return=total_return,
        cagr=cagr,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        max_drawdown_days=dd_days,
        volatility=volatility,
        trades=len(round_trips),
        win_rate=len(wins) / len(round_trips) if round_trips else 0.0,
        # A strategy with no losses has infinite profit factor, which is not a
        # useful number; report 0 and let the trade count speak.
        profit_factor=float(gross_profit / gross_loss) if gross_loss > 0 else 0.0,
        average_win=Rupees(gross_profit / len(wins)) if wins else Rupees.zero(),
        average_loss=Rupees(-gross_loss / len(losses)) if losses else Rupees.zero(),
        total_charges=total_charges,
        total_tax=total_tax,
        after_tax_final_equity=after_tax_final,
        after_tax_total_return=after_tax_return,
        after_tax_cagr=after_tax_cagr,
        tax_liabilities=liabilities,
    )
