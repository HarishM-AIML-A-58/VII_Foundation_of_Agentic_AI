"""
Deterministic Risk Guardian & Position Sizing Engine.
Hardcoded rules that can NEVER be overridden by model outputs:
1. Max position size limit.
2. Daily loss limit.
3. Max cumulative drawdown limit.
4. Kill switch (immediate liquidation & halt).
5. Manual approval requirement on orders above defined dollar threshold.
6. Capped 1/4 Kelly position sizing based on calibrated probability.
"""
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class RiskLimits:
    max_position_size_usd: float = 10000.0
    max_daily_loss_usd: float = 2000.0
    max_drawdown_pct: float = 0.15  # 15%
    manual_approval_threshold_usd: float = 5000.0


class RiskGuardian:
    def __init__(self, initial_capital: float, limits: Optional[RiskLimits] = None):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.daily_pnl = 0.0
        self.limits = limits or RiskLimits()
        self.kill_switch_engaged = False
        self.open_position_size_usd = 0.0

    def trigger_kill_switch(self, reason: str = "Manual Trigger"):
        """Immediately halts all trading and enforces complete risk liquidation."""
        self.kill_switch_engaged = True

    def reset_daily_pnl(self):
        """Called at start of each trading session."""
        self.daily_pnl = 0.0

    def update_pnl(self, realized_pnl: float):
        """Tracks capital and drawdown metrics."""
        self.daily_pnl += realized_pnl
        self.current_capital += realized_pnl
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital

        # Auto-trip kill switch if daily loss limit breached
        if self.daily_pnl <= -self.limits.max_daily_loss_usd:
            self.trigger_kill_switch(f"Daily loss limit breached: {self.daily_pnl}")

        # Auto-trip kill switch if max drawdown breached
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        if current_drawdown >= self.limits.max_drawdown_pct:
            self.trigger_kill_switch(f"Max drawdown breached: {current_drawdown:.2%}")

    def compute_kelly_size(
        self,
        calibrated_prob: float,
        reward_risk_ratio: float,
        kelly_fraction: float = 0.25,  # 1/4 Kelly maximum
    ) -> float:
        """
        Calculates capped fraction of Kelly criterion:
        f* = (b*p - q) / b
        where b = reward_risk_ratio, p = calibrated_prob, q = 1 - p.
        Returns size in USD.
        """
        if calibrated_prob <= 0.50:
            return 0.0

        q = 1.0 - calibrated_prob
        full_kelly = (reward_risk_ratio * calibrated_prob - q) / reward_risk_ratio
        if full_kelly <= 0.0:
            return 0.0

        # Apply fraction cap (1/4 Kelly max)
        applied_fraction = min(full_kelly * kelly_fraction, 0.25)
        raw_size = self.current_capital * applied_fraction
        return min(raw_size, self.limits.max_position_size_usd)

    def validate_order(
        self,
        requested_size_usd: float,
        calibrated_prob: float,
    ) -> Tuple[bool, str, bool]:
        """
        Returns (is_approved, reason, requires_manual_approval).
        Models CANNOT override any rejection here.
        """
        if self.kill_switch_engaged:
            return False, "Kill switch is active. Trading halted.", False

        if self.daily_pnl <= -self.limits.max_daily_loss_usd:
            return False, f"Daily loss limit reached (${self.daily_pnl:.2f})", False

        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        if current_drawdown >= self.limits.max_drawdown_pct:
            return False, f"Max drawdown reached ({current_drawdown:.2%})", False

        if requested_size_usd <= 0.0:
            return False, "Order size must be positive", False

        # Enforce max position size cap
        if self.open_position_size_usd + requested_size_usd > self.limits.max_position_size_usd:
            return (
                False,
                f"Exceeds max position size (${self.limits.max_position_size_usd:.2f})",
                False,
            )

        # Check manual approval trigger
        requires_manual = requested_size_usd >= self.limits.manual_approval_threshold_usd

        return True, "Approved by deterministic risk rules", requires_manual
