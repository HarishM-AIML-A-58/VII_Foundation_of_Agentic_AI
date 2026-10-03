"""Position sizing, exposure limits, veto, kill switch, and stress testing.

Every check that can stop a trade is arithmetic or a flag lookup -- never a
model's judgement. The agent debate advises; this layer decides.

Order of application, and why:

1. `veto`           -- is the trade sound on its own terms? (cheapest, most decisive)
2. `limits`         -- is it sound given everything already open?
3. `sizing`         -- how much, such that being wrong is survivable?
4. `stress_testing` -- parametric/historical VaR & CVaR, macro crisis scenarios,
                       and intraday margin monitoring.

Sizing runs last because it is the only step that needs a number from the
other two, and the most expensive to compute.
"""

from trading_agent.risk.jev_kelly_sizer import JevKellySizer
from trading_agent.risk.kill_switch import KillSwitch
from trading_agent.risk.limits import (
    LimitBreach,
    LimitCheck,
    PortfolioLimits,
    check_limits,
)
from trading_agent.risk.position_sizing import SizingResult, size_position
from trading_agent.risk.stress_testing import (
    MacroScenario,
    MarginMonitor,
    MarginStatus,
    ScenarioParameters,
    ScenarioStressTester,
    StressTestResult,
    calculate_expected_shortfall,
    calculate_historical_expected_shortfall,
    calculate_historical_var,
    calculate_parametric_var,
)
from trading_agent.risk.veto import VetoReason, VetoResult, apply_veto, evaluate

__all__ = [
    "JevKellySizer",
    "KillSwitch",
    "LimitBreach",
    "LimitCheck",
    "MacroScenario",
    "MarginMonitor",
    "MarginStatus",
    "PortfolioLimits",
    "ScenarioParameters",
    "ScenarioStressTester",
    "SizingResult",
    "StressTestResult",
    "VetoReason",
    "VetoResult",
    "apply_veto",
    "calculate_expected_shortfall",
    "calculate_historical_expected_shortfall",
    "calculate_historical_var",
    "calculate_parametric_var",
    "check_limits",
    "evaluate",
    "size_position",
]
