"""Real-time intraday risk engine and macro scenario stress testing.

Provides institutional risk measures for Indian equity portfolios:
1. Parametric & Historical Value-at-Risk (VaR at 99%) and Expected Shortfall (CVaR).
2. Macro Historical Scenario Stress Testing (GFC 2008, Demonetization 2016,
   COVID-19 2020, Election 2024).
3. Intraday Margin Call Simulator (SPAN + Exposure margin accounting, maintenance
   buffer compliance, and required liquidation calculation).

Pure risk module: depends only on standard library, numpy, scipy, and
trading_agent.domain.money.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Final

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from trading_agent.domain.money import Rupees
from trading_agent.observability import get_logger

__all__ = [
    "COVID_EQUITY_SHOCK",
    "COVID_VOL_MULTIPLIER",
    "DEFAULT_CONFIDENCE_99",
    "DEFAULT_CONFIDENCE_99_Z",
    "DEFAULT_HORIZON_DAYS",
    "DEFAULT_MACRO_SCENARIOS",
    "DEFAULT_MAINTENANCE_BUFFER",
    "DEMONETIZATION_EQUITY_SHOCK",
    "DEMONETIZATION_VOL_MULTIPLIER",
    "ELECTION_EQUITY_SHOCK",
    "ELECTION_VOL_MULTIPLIER",
    "GFC_EQUITY_SHOCK",
    "GFC_VOL_MULTIPLIER",
    "MacroScenario",
    "MarginMonitor",
    "MarginStatus",
    "ScenarioParameters",
    "ScenarioStressTester",
    "StressTestResult",
    "calculate_expected_shortfall",
    "calculate_historical_expected_shortfall",
    "calculate_historical_var",
    "calculate_parametric_var",
]

log = get_logger(__name__)

# --- Parametric VaR Constants ---
DEFAULT_CONFIDENCE_99: Final[float] = 0.99
DEFAULT_CONFIDENCE_99_Z: Final[float] = 2.3263
DEFAULT_HORIZON_DAYS: Final[int] = 1

_SQRT_2PI: Final[float] = math.sqrt(2.0 * math.pi)
_HALF: Final[float] = 0.5
_PERCENT_SCALE: Final[float] = 100.0
_FLOAT_TOLERANCE: Final[float] = 1e-6

# --- Margin Monitor Constants ---
DEFAULT_MAINTENANCE_BUFFER: Final[float] = 0.10

# --- Historical Macro Crisis Parameters ---
GFC_EQUITY_SHOCK: Final[float] = -0.55
GFC_VOL_MULTIPLIER: Final[float] = 2.5

DEMONETIZATION_EQUITY_SHOCK: Final[float] = -0.065
DEMONETIZATION_VOL_MULTIPLIER: Final[float] = 1.5

COVID_EQUITY_SHOCK: Final[float] = -0.384
COVID_VOL_MULTIPLIER: Final[float] = 4.5

ELECTION_EQUITY_SHOCK: Final[float] = -0.060
ELECTION_VOL_MULTIPLIER: Final[float] = 2.0


def _standard_normal_pdf(z: float) -> float:
    """Evaluate standard normal probability density function phi(z)."""
    return math.exp(-_HALF * z * z) / _SQRT_2PI


def _get_z_quantile(confidence: float) -> float:
    """Retrieve standard normal quantile for a given confidence level.

    Uses the institutional standard 2.3263 for 99% confidence.
    """
    if abs(confidence - DEFAULT_CONFIDENCE_99) < _FLOAT_TOLERANCE:
        return DEFAULT_CONFIDENCE_99_Z
    return float(norm.ppf(confidence))


def _scale_rupees(amount: Rupees, factor: float) -> Rupees:
    """Scale a Rupees amount by a non-negative float factor, quantising to paisa."""
    if factor <= 0.0:
        return Rupees.zero()
    dec_factor = Decimal(str(factor))
    return Rupees(amount.value * dec_factor)


# ==============================================================================
# 1. Parametric & Historical Value-at-Risk (VaR) and Expected Shortfall (CVaR)
# ==============================================================================


def calculate_parametric_var(
    portfolio_value: Rupees,
    daily_volatility: float,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    confidence: float = DEFAULT_CONFIDENCE_99,
) -> Rupees:
    """Calculate parametric Value-at-Risk (VaR) under a normal distribution.

    Formula:
        VaR = portfolio_value * z * daily_volatility * sqrt(horizon_days)

    Uses standard normal quantile (2.3263 for 99%).
    """
    if portfolio_value < Rupees.zero():
        raise ValueError(f"portfolio_value must be non-negative, got {portfolio_value}")
    if daily_volatility < 0.0:
        raise ValueError(f"daily_volatility must be non-negative, got {daily_volatility}")
    if horizon_days <= 0:
        raise ValueError(f"horizon_days must be positive, got {horizon_days}")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be between 0 and 1 exclusive, got {confidence}")

    if portfolio_value.is_zero() or daily_volatility == 0.0:
        return Rupees.zero()

    z = _get_z_quantile(confidence)
    loss_fraction = z * daily_volatility * math.sqrt(horizon_days)
    return _scale_rupees(portfolio_value, loss_fraction)


def calculate_expected_shortfall(
    portfolio_value: Rupees,
    daily_volatility: float,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    confidence: float = DEFAULT_CONFIDENCE_99,
) -> Rupees:
    """Calculate parametric Expected Shortfall (CVaR) under a normal distribution.

    Formula:
        ES = portfolio_value * (phi(z) / (1 - confidence)) * daily_volatility * sqrt(horizon_days)

    Where phi(z) is the standard normal density at quantile z.
    """
    if portfolio_value < Rupees.zero():
        raise ValueError(f"portfolio_value must be non-negative, got {portfolio_value}")
    if daily_volatility < 0.0:
        raise ValueError(f"daily_volatility must be non-negative, got {daily_volatility}")
    if horizon_days <= 0:
        raise ValueError(f"horizon_days must be positive, got {horizon_days}")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be between 0 and 1 exclusive, got {confidence}")

    if portfolio_value.is_zero() or daily_volatility == 0.0:
        return Rupees.zero()

    z = _get_z_quantile(confidence)
    phi_z = _standard_normal_pdf(z)
    tail_multiplier = phi_z / (1.0 - confidence)
    loss_fraction = tail_multiplier * daily_volatility * math.sqrt(horizon_days)
    return _scale_rupees(portfolio_value, loss_fraction)


def calculate_historical_var(
    portfolio_value: Rupees,
    returns: Sequence[float] | Sequence[Decimal] | npt.NDArray[np.float64],
    confidence: float = DEFAULT_CONFIDENCE_99,
) -> Rupees:
    """Calculate non-parametric Historical Value-at-Risk (VaR) from return series.

    Empirical quantile at (1 - confidence) level.
    """
    if portfolio_value < Rupees.zero():
        raise ValueError(f"portfolio_value must be non-negative, got {portfolio_value}")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be between 0 and 1 exclusive, got {confidence}")
    if len(returns) == 0:
        raise ValueError("returns sequence cannot be empty")

    clean_returns = np.asarray(returns, dtype=np.float64)
    alpha_percentile = (1.0 - confidence) * _PERCENT_SCALE
    tail_cutoff = float(np.percentile(clean_returns, alpha_percentile))

    if tail_cutoff >= 0.0:
        return Rupees.zero()

    return _scale_rupees(portfolio_value, abs(tail_cutoff))


def calculate_historical_expected_shortfall(
    portfolio_value: Rupees,
    returns: Sequence[float] | Sequence[Decimal] | npt.NDArray[np.float64],
    confidence: float = DEFAULT_CONFIDENCE_99,
) -> Rupees:
    """Calculate non-parametric Historical Expected Shortfall (CVaR).

    Average loss conditional on return being at or beyond the empirical VaR threshold.
    """
    if portfolio_value < Rupees.zero():
        raise ValueError(f"portfolio_value must be non-negative, got {portfolio_value}")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be between 0 and 1 exclusive, got {confidence}")
    if len(returns) == 0:
        raise ValueError("returns sequence cannot be empty")

    clean_returns = np.asarray(returns, dtype=np.float64)
    alpha_percentile = (1.0 - confidence) * _PERCENT_SCALE
    tail_cutoff = float(np.percentile(clean_returns, alpha_percentile))

    tail_losses = clean_returns[clean_returns <= tail_cutoff]
    mean_tail_loss = tail_cutoff if len(tail_losses) == 0 else float(np.mean(tail_losses))

    if mean_tail_loss >= 0.0:
        return Rupees.zero()

    return _scale_rupees(portfolio_value, abs(mean_tail_loss))


# ==============================================================================
# 2. Macro Historical Scenario Stress Testing
# ==============================================================================


class MacroScenario(StrEnum):
    """Historical macro crises for institutional stress testing."""

    GFC_2008 = "GFC_2008"
    DEMONETIZATION_2016 = "DEMONETIZATION_2016"
    COVID_2020 = "COVID_2020"
    ELECTION_2024 = "ELECTION_2024"


@dataclass(frozen=True, slots=True)
class ScenarioParameters:
    """Shock parameters calibrating a historical macro event."""

    equity_shock: float
    vol_multiplier: float
    description: str = ""


DEFAULT_MACRO_SCENARIOS: Final[dict[MacroScenario, ScenarioParameters]] = {
    MacroScenario.GFC_2008: ScenarioParameters(
        equity_shock=GFC_EQUITY_SHOCK,
        vol_multiplier=GFC_VOL_MULTIPLIER,
        description="2008 GFC: severe equity shock (-55%), 2.5x volatility spike",
    ),
    MacroScenario.DEMONETIZATION_2016: ScenarioParameters(
        equity_shock=DEMONETIZATION_EQUITY_SHOCK,
        vol_multiplier=DEMONETIZATION_VOL_MULTIPLIER,
        description="2016 Demonetization: liquidity shock (-6.5%), 1.5x volatility spike",
    ),
    MacroScenario.COVID_2020: ScenarioParameters(
        equity_shock=COVID_EQUITY_SHOCK,
        vol_multiplier=COVID_VOL_MULTIPLIER,
        description="2020 COVID-19 pandemic shock: rapid crash (-38.4%), 4.5x volatility spike",
    ),
    MacroScenario.ELECTION_2024: ScenarioParameters(
        equity_shock=ELECTION_EQUITY_SHOCK,
        vol_multiplier=ELECTION_VOL_MULTIPLIER,
        description="2024 Election result shock: sharp drop (-6.0%), 2.0x volatility spike",
    ),
}


@dataclass(frozen=True, slots=True)
class StressTestResult:
    """Stress test outcome for a single macro scenario."""

    scenario: MacroScenario
    equity_shock: float
    vol_multiplier: float
    drawdown_pct: float
    stressed_loss: Rupees
    stressed_value: Rupees
    stressed_volatility: float | None = None

    @property
    def loss_pct(self) -> float:
        """Drawdown expressed as a positive loss fraction."""
        return abs(self.drawdown_pct) if self.drawdown_pct < 0.0 else 0.0

    @property
    def is_solvent(self) -> bool:
        """True if portfolio preserves positive equity after the stressed crisis."""
        return self.stressed_value > Rupees.zero()


class ScenarioStressTester:
    """Simulates portfolio drawdown and dollar loss across historical crises."""

    def __init__(
        self,
        scenarios: Mapping[MacroScenario, ScenarioParameters] | None = None,
    ) -> None:
        self._scenarios = dict(scenarios if scenarios is not None else DEFAULT_MACRO_SCENARIOS)

    def run_stress_test(
        self,
        portfolio_value: Rupees,
        beta: float,
        gross_leverage: float,
        baseline_volatility: float | None = None,
    ) -> dict[MacroScenario, StressTestResult]:
        """Run macro crisis simulation across all calibrated scenarios.

        Parameters:
            portfolio_value: Total mark-to-market equity.
            beta: Portfolio market beta sensitivity.
            gross_leverage: Gross leverage ratio (gross notional / equity >= 0.0).
            baseline_volatility: Optional current portfolio volatility.

        Returns:
            dict mapping each MacroScenario to its StressTestResult.
        """
        if portfolio_value < Rupees.zero():
            raise ValueError(f"portfolio_value must be non-negative, got {portfolio_value}")
        if gross_leverage < 0.0:
            raise ValueError(f"gross_leverage must be non-negative, got {gross_leverage}")
        if baseline_volatility is not None and baseline_volatility < 0.0:
            raise ValueError(f"baseline_volatility must be non-negative, got {baseline_volatility}")

        results: dict[MacroScenario, StressTestResult] = {}
        for scenario, params in self._scenarios.items():
            drawdown = params.equity_shock * beta * gross_leverage

            if drawdown < 0.0:
                stressed_loss = _scale_rupees(portfolio_value, abs(drawdown))
            else:
                stressed_loss = Rupees.zero()

            stressed_value = portfolio_value - stressed_loss

            stressed_vol = (
                baseline_volatility * params.vol_multiplier
                if baseline_volatility is not None
                else None
            )

            results[scenario] = StressTestResult(
                scenario=scenario,
                equity_shock=params.equity_shock,
                vol_multiplier=params.vol_multiplier,
                drawdown_pct=drawdown,
                stressed_loss=stressed_loss,
                stressed_value=stressed_value,
                stressed_volatility=stressed_vol,
            )

        return results


# ==============================================================================
# 3. Intraday Margin Call Simulator
# ==============================================================================


@dataclass(frozen=True, slots=True)
class MarginStatus:
    """Point-in-time account margin utilization and call diagnostic."""

    current_equity: Rupees
    required_span_margin: Rupees
    free_margin: Rupees
    utilization_pct: float
    is_margin_call: bool
    required_liquidation: Rupees
    maintenance_threshold: float
    margin_deficit: Rupees = field(default_factory=Rupees.zero)

    @property
    def required_margin(self) -> Rupees:
        return self.required_span_margin

    @property
    def margin_utilization_pct(self) -> float:
        return self.utilization_pct

    @property
    def liquidation_amount(self) -> Rupees:
        return self.required_liquidation

    @property
    def required_liquidation_amount(self) -> Rupees:
        return self.required_liquidation

    @property
    def margin_call_triggered(self) -> bool:
        return self.is_margin_call

    def required_notional_liquidation(
        self,
        margin_rate: Decimal | float = Decimal("0.20"),
    ) -> Rupees:
        """Compute gross position notional to liquidate to release required margin.

        For instance, at 20% margin rate, releasing ₹20,000 margin requires
        liquidating ₹1,00,000 in position notional.
        """
        if self.required_liquidation.is_zero():
            return Rupees.zero()
        rate = Decimal(str(margin_rate))
        if rate <= Decimal(0):
            raise ValueError(f"margin_rate must be strictly positive, got {margin_rate}")
        return Rupees(self.required_liquidation.value / rate)


class MarginMonitor:
    """Monitors intraday margin utilization and simulates exchange margin calls.

    Computes free margin, margin utilization percentage, and determines whether
    a margin call is triggered under a maintenance buffer (e.g. 10% safety cushion).
    When breached, solves for the exact margin liquidation amount required to restore
    compliance.
    """

    def __init__(
        self,
        current_equity: Rupees | None = None,
        required_span_margin: Rupees | None = None,
        maintenance_threshold: float = DEFAULT_MAINTENANCE_BUFFER,
    ) -> None:
        if maintenance_threshold < 0.0:
            raise ValueError(
                f"maintenance_threshold cannot be negative, got {maintenance_threshold}"
            )
        # Interpret values > 1.0 (e.g. 10.0) as percentages (10% -> 0.10)
        self.maintenance_threshold = (
            maintenance_threshold / _PERCENT_SCALE
            if maintenance_threshold > 1.0
            else maintenance_threshold
        )
        self._status: MarginStatus | None = None

        if current_equity is not None and required_span_margin is not None:
            self._status = self.evaluate(
                current_equity=current_equity,
                required_span_margin=required_span_margin,
                maintenance_threshold=self.maintenance_threshold,
            )

    def evaluate(
        self,
        current_equity: Rupees,
        required_span_margin: Rupees,
        maintenance_threshold: float | None = None,
    ) -> MarginStatus:
        """Evaluate margin compliance against SPAN requirements and maintenance buffer."""
        buffer_val = (
            self.maintenance_threshold
            if maintenance_threshold is None
            else (
                maintenance_threshold / _PERCENT_SCALE
                if maintenance_threshold > 1.0
                else maintenance_threshold
            )
        )
        if buffer_val < 0.0:
            raise ValueError(f"maintenance_threshold cannot be negative, got {buffer_val}")

        free_margin = current_equity - required_span_margin

        # Margin utilization percentage
        if current_equity <= Rupees.zero():
            utilization_pct = 0.0 if required_span_margin.is_zero() else float("inf")
        else:
            utilization_pct = float(required_span_margin.value / current_equity.value)

        # Maintenance requirement = required_span_margin * (1 + buffer)
        buffer_factor = Decimal(str(1.0 + buffer_val))
        maintenance_requirement = Rupees(required_span_margin.value * buffer_factor)

        is_call = current_equity < maintenance_requirement

        if not is_call:
            required_liquidation = Rupees.zero()
            margin_deficit = Rupees.zero()
        else:
            margin_deficit = maintenance_requirement - current_equity
            if current_equity <= Rupees.zero():
                # Complete equity exhaustion: full liquidation of open positions required
                required_liquidation = required_span_margin
            else:
                # Solve target margin: current_equity / (1 + buffer)
                target_margin_dec = current_equity.value / buffer_factor
                liquidation_dec = required_span_margin.value - target_margin_dec
                required_liquidation = Rupees(max(Decimal(0), liquidation_dec))

        status = MarginStatus(
            current_equity=current_equity,
            required_span_margin=required_span_margin,
            free_margin=free_margin,
            utilization_pct=utilization_pct,
            is_margin_call=is_call,
            required_liquidation=required_liquidation,
            maintenance_threshold=buffer_val,
            margin_deficit=margin_deficit,
        )
        self._status = status
        return status

    # Aliases for diverse invocation styles
    check_margin = evaluate
    simulate_margin_call = evaluate

    @classmethod
    def simulate(
        cls,
        current_equity: Rupees,
        required_span_margin: Rupees,
        maintenance_threshold: float = DEFAULT_MAINTENANCE_BUFFER,
    ) -> MarginStatus:
        """Classmethod shorthand to evaluate margin state directly."""
        monitor = cls(maintenance_threshold=maintenance_threshold)
        return monitor.evaluate(current_equity, required_span_margin)

    # Property proxies when initialized with equity and margin
    @property
    def free_margin(self) -> Rupees:
        if self._status is None:
            raise RuntimeError("evaluate() must be called first when initialized without amounts")
        return self._status.free_margin

    @property
    def utilization_pct(self) -> float:
        if self._status is None:
            raise RuntimeError("evaluate() must be called first when initialized without amounts")
        return self._status.utilization_pct

    @property
    def margin_utilization_pct(self) -> float:
        return self.utilization_pct

    @property
    def is_margin_call(self) -> bool:
        if self._status is None:
            raise RuntimeError("evaluate() must be called first when initialized without amounts")
        return self._status.is_margin_call

    @property
    def margin_call_triggered(self) -> bool:
        return self.is_margin_call

    @property
    def required_liquidation(self) -> Rupees:
        if self._status is None:
            raise RuntimeError("evaluate() must be called first when initialized without amounts")
        return self._status.required_liquidation

    @property
    def liquidation_amount(self) -> Rupees:
        return self.required_liquidation

    @property
    def required_liquidation_amount(self) -> Rupees:
        return self.required_liquidation

    @property
    def margin_deficit(self) -> Rupees:
        if self._status is None:
            raise RuntimeError("evaluate() must be called first when initialized without amounts")
        return self._status.margin_deficit

    @property
    def status(self) -> MarginStatus:
        if self._status is None:
            raise RuntimeError("evaluate() must be called first when initialized without amounts")
        return self._status
