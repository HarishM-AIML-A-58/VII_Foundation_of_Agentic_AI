"""Portfolio Volatility Targeting.

**Why this lives in `backtest/` and not in `risk/`.** It reads
``TRADING_DAYS_PER_YEAR`` from :mod:`trading_agent.backtest.metrics`, and the
layering contract puts ``risk`` *below* ``backtest`` -- a module in ``risk``
may not import upward. More than a technicality: everything in ``risk/`` is a
gate that answers yes or no about one proposed trade, using only current state.
This answers a different question -- how large the whole book should be, given
what volatility has been doing -- which is portfolio construction, and belongs
beside the engine that applies it.

Implements R5:
Volatilities cluster strongly (volatility is autocorrelated, returns are not).
By scaling exposure inversely to forecast portfolio volatility, the portfolio
takes smaller positions in turbulent markets and fuller positions in quiet ones,
stabilizing the risk curve and preventing momentum drawdowns.

Formula:
    scalar = min(max_leverage, target_vol / realised_vol)
    scaled_weight_i = weight_i * scalar
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal

import numpy as np
import pandas as pd

from trading_agent.backtest.metrics import TRADING_DAYS_PER_YEAR

__all__ = ["VolatilityTargeter"]

_MIN_HISTORY_SESSIONS = 10
_MIN_ANNUAL_VOL = 0.001


@dataclass(frozen=True, slots=True)
class VolatilityTargeter:
    """Calculates scaling factors to target a constant annualized portfolio volatility."""

    target_annual_vol: float = 0.15  # 15% annualized target volatility
    lookback_sessions: int = 60
    max_leverage: float = 1.0  # Long-only unlevered cash cap
    min_scalar: float = 0.20  # Never scale down below 20%

    def calculate_scalar(
        self,
        panel: dict[str, pd.DataFrame],
        weights: dict[str, Decimal],
        session: pd.Timestamp,
    ) -> Decimal:
        """Compute the exposure scalar for proposed target weights."""
        if not weights or self.target_annual_vol <= 0:
            return Decimal(1)

        # Collect return histories
        returns_dict: dict[str, pd.Series] = {}
        for symbol in weights:
            frame = panel.get(symbol)
            if frame is None or frame.empty:
                continue
            hist = frame[frame.index <= session]["close"].dropna()
            if len(hist) > self.lookback_sessions:
                returns_dict[symbol] = hist.iloc[-self.lookback_sessions :].pct_change().dropna()

        if not returns_dict:
            return Decimal(1)

        ret_df = pd.DataFrame(returns_dict).dropna()
        if len(ret_df) < _MIN_HISTORY_SESSIONS:
            return Decimal(1)

        weight_vec = np.array([float(weights.get(s, Decimal(0))) for s in ret_df.columns])
        # Daily portfolio returns
        portfolio_returns = ret_df.to_numpy() @ weight_vec
        realised_daily_vol = float(np.std(portfolio_returns))
        realised_annual_vol = realised_daily_vol * math.sqrt(TRADING_DAYS_PER_YEAR)

        if realised_annual_vol <= _MIN_ANNUAL_VOL:
            return Decimal(str(self.max_leverage))

        raw_scalar = self.target_annual_vol / realised_annual_vol
        bounded_scalar = max(self.min_scalar, min(self.max_leverage, raw_scalar))
        return Decimal(str(round(bounded_scalar, 4)))

    def apply(
        self,
        panel: dict[str, pd.DataFrame],
        weights: dict[str, Decimal],
        session: pd.Timestamp,
    ) -> dict[str, Decimal]:
        """Scale target weights by the volatility targeting scalar."""
        scalar = self.calculate_scalar(panel, weights, session)
        return {symbol: Decimal(str(round(w * scalar, 4))) for symbol, w in weights.items()}
