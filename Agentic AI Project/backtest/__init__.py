"""Event-driven backtester built for Indian equity mechanics."""

from trading_agent.backtest.agent_filter import (
    AbsentVerdict,
    AgentFilter,
    FilterOutcome,
    RecordedVerdict,
)
from trading_agent.backtest.benchmark import PurchasableBenchmark, cost_benchmark_curve
from trading_agent.backtest.core_satellite import (
    AllocationPoint,
    CoreSatelliteAnalysis,
    analyze_core_satellite,
)
from trading_agent.backtest.cpcv import (
    CombinatorialPurgedCV,
    CPCVFold,
    CPCVPathResult,
    CPCVResult,
    calculate_pbo,
)
from trading_agent.backtest.engine import BacktestResult, run_backtest
from trading_agent.backtest.fill_model import Fill, FillModel
from trading_agent.backtest.metrics import PerformanceMetrics, compute_metrics
from trading_agent.backtest.portfolio import Holding, InsufficientCashError, Portfolio
from trading_agent.backtest.rebalance import (
    MIN_TRADEABLE_WEIGHT,
    RebalanceResult,
    RegimeFilter,
    run_rebalance_backtest,
)
from trading_agent.backtest.rebalance_walk_forward import (
    RebalanceWalkForwardResult,
    RebalanceWalkForwardWindow,
    walk_forward_rebalance,
)
from trading_agent.backtest.report import render_report, write_report
from trading_agent.backtest.statistics import (
    DeflatedSharpeResult,
    calculate_return_moments,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from trading_agent.backtest.volatility_target import VolatilityTargeter
from trading_agent.backtest.walk_forward import (
    WalkForwardResult,
    WalkForwardWindow,
    walk_forward,
)

__all__ = [
    "MIN_TRADEABLE_WEIGHT",
    "AbsentVerdict",
    "AgentFilter",
    "AllocationPoint",
    "BacktestResult",
    "CPCVFold",
    "CPCVPathResult",
    "CPCVResult",
    "CombinatorialPurgedCV",
    "CoreSatelliteAnalysis",
    "DeflatedSharpeResult",
    "Fill",
    "FillModel",
    "FilterOutcome",
    "Holding",
    "InsufficientCashError",
    "PerformanceMetrics",
    "Portfolio",
    "PurchasableBenchmark",
    "RebalanceResult",
    "RebalanceWalkForwardResult",
    "RebalanceWalkForwardWindow",
    "RecordedVerdict",
    "RegimeFilter",
    "VolatilityTargeter",
    "WalkForwardResult",
    "WalkForwardWindow",
    "analyze_core_satellite",
    "calculate_pbo",
    "calculate_return_moments",
    "compute_metrics",
    "cost_benchmark_curve",
    "deflated_sharpe_ratio",
    "probabilistic_sharpe_ratio",
    "render_report",
    "run_backtest",
    "run_rebalance_backtest",
    "walk_forward",
    "walk_forward_rebalance",
    "write_report",
]
