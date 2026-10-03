"""Indian equity transaction costs.

Charges (`calculator`) are knowable and verifiable against a contract note.
Slippage (`slippage`) is estimated. They are separate modules so a backtest
discrepancy can be attributed to one or the other.
"""

from trading_agent.costs.breakdown import ChargeBreakdown
from trading_agent.costs.calculator import ChargeCalculator, UnverifiedScheduleWarning
from trading_agent.costs.schedule import (
    BrokerageModel,
    ChargeSchedule,
    RoundingMode,
    ScheduleNotFoundError,
    ScheduleRegistry,
)
from trading_agent.costs.slippage import SlippageModel, SlippageModelType
from trading_agent.costs.tax import (
    CapitalGainsRates,
    CapitalGainsTax,
    TaxLiability,
    financial_year,
)

__all__ = [
    "BrokerageModel",
    "CapitalGainsRates",
    "CapitalGainsTax",
    "ChargeBreakdown",
    "ChargeCalculator",
    "ChargeSchedule",
    "RoundingMode",
    "ScheduleNotFoundError",
    "ScheduleRegistry",
    "SlippageModel",
    "SlippageModelType",
    "TaxLiability",
    "UnverifiedScheduleWarning",
    "financial_year",
]
