"""Pure domain types.

This package imports nothing from the rest of ``trading_agent`` and performs
no I/O. The rule is enforced in CI by import-linter (see ``pyproject.toml``),
because a domain that quietly grows a database call stops being testable.
"""

from trading_agent.domain.bar import Bar
from trading_agent.domain.derivatives import (
    DEFAULT_EXPOSURE_MARGIN_PCT,
    DEFAULT_SPAN_MARGIN_PCT,
    FoContract,
    FoContractType,
    calculate_business_tax,
    calculate_margin_requirement,
    calculate_mtm_pnl,
    round_to_lot,
)
from trading_agent.domain.enums import (
    Action,
    Exchange,
    OrderStatus,
    OrderType,
    ProductType,
    Side,
    SignalStatus,
)
from trading_agent.domain.futures_roll import (
    DAYS_PER_YEAR,
    calculate_basis_and_carry,
    calculate_roll_ratio,
    generate_calendar_roll_orders,
    get_nse_monthly_expiry,
)
from trading_agent.domain.instrument import Instrument
from trading_agent.domain.money import PAISA, Rupees, rupees, sum_rupees
from trading_agent.domain.order import Order, OrderAck, OrderRequest
from trading_agent.domain.position import Position
from trading_agent.domain.signal import AgentVote, TradeSignal, build_idempotency_key
from trading_agent.domain.trade import Execution, RoundTrip

__all__ = [
    "DAYS_PER_YEAR",
    "DEFAULT_EXPOSURE_MARGIN_PCT",
    "DEFAULT_SPAN_MARGIN_PCT",
    "PAISA",
    "Action",
    "AgentVote",
    "Bar",
    "Exchange",
    "Execution",
    "FoContract",
    "FoContractType",
    "Instrument",
    "Order",
    "OrderAck",
    "OrderRequest",
    "OrderStatus",
    "OrderType",
    "Position",
    "ProductType",
    "RoundTrip",
    "Rupees",
    "Side",
    "SignalStatus",
    "TradeSignal",
    "build_idempotency_key",
    "calculate_basis_and_carry",
    "calculate_business_tax",
    "calculate_margin_requirement",
    "calculate_mtm_pnl",
    "calculate_roll_ratio",
    "generate_calendar_roll_orders",
    "get_nse_monthly_expiry",
    "round_to_lot",
    "rupees",
    "sum_rupees",
]
