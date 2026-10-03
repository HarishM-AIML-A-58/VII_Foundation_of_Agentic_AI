"""Institutional Long/Short Equity Market-Neutral (L/S EMN) quantitative strategy.

Implements an institutional hedge fund pipeline:
1. Multi-factor cross-sectional alpha extraction (Skip-Month Momentum,
   Residual Momentum, Betting Against Beta, Idiosyncratic Volatility).
2. OLS factor neutralization against broad market returns.
3. Ledoit-Wolf constant correlation covariance shrinkage.
4. Convex quadratic programming optimization with dollar neutrality,
   beta neutrality, and position concentration caps.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import StrEnum

import pandas as pd

from trading_agent.features.cross_sectional import (
    MultiFactorComposite,
    calculate_betting_against_beta,
    calculate_idiosyncratic_volatility,
    calculate_residual_momentum,
    calculate_skip_month_momentum,
    orthogonalize_factor,
    rank_and_gaussianize,
)
from trading_agent.market_data import IST
from trading_agent.strategy.portfolio_optimizer import (
    LedoitWolfOptimizer,
    PortfolioMandate,
)

__all__ = [
    "InstitutionalMarketNeutralStrategy",
    "StrategyRebalanceFreq",
]

_MIN_HISTORY_SESSIONS = 126
_LONG_LOOKBACK = 252
_DEFAULT_MAX_SINGLE_WEIGHT = 0.05
_DEFAULT_MAX_GROSS_LEVERAGE = 2.0
_MIN_CANDIDATES = 5
_MIN_WEIGHT_THRESHOLD = 0.0005


class StrategyRebalanceFreq(StrEnum):
    """Rebalancing periodicity."""

    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


def _extract_panel_series(
    panel: dict[str, pd.DataFrame],
    cutoff: pd.Timestamp,
) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    """Extract valid returns and close price series up to the cutoff date."""
    returns_dict: dict[str, pd.Series] = {}
    price_dict: dict[str, pd.Series] = {}

    for sym, df in panel.items():
        if df.empty or "close" not in df.columns:
            continue
        idx = pd.DatetimeIndex(df.index)
        usable = df[idx <= cutoff]
        if len(usable) < _MIN_HISTORY_SESSIONS:
            continue
        closes = usable["close"].dropna()
        rets = closes.pct_change().dropna()
        if len(rets) >= _MIN_HISTORY_SESSIONS:
            returns_dict[sym] = rets
            price_dict[sym] = closes

    return returns_dict, price_dict


def _compute_alpha_factors(
    returns_dict: dict[str, pd.Series],
    price_dict: dict[str, pd.Series],
    market_returns: pd.Series,
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute individual factor scores and betas."""
    skip_mom: dict[str, float] = {}
    res_mom: dict[str, float] = {}
    bab_scores: dict[str, float] = {}
    ivol_scores: dict[str, float] = {}
    betas: dict[str, float] = {}

    for sym, closes in price_dict.items():
        s_mom = calculate_skip_month_momentum(closes, lookback=_LONG_LOOKBACK, lag=21)
        if s_mom is not None:
            skip_mom[sym] = s_mom

        r_series = returns_dict[sym]
        r_mom = calculate_residual_momentum(r_series, market_returns, lookback=_LONG_LOOKBACK)
        if r_mom is not None:
            res_mom[sym] = r_mom

        bab = calculate_betting_against_beta(r_series, market_returns, lookback=_LONG_LOOKBACK)
        if bab is not None:
            bab_scores[sym] = bab
            betas[sym] = -bab

        ivol = calculate_idiosyncratic_volatility(r_series, market_returns, lookback=_LONG_LOOKBACK)
        if ivol is not None:
            ivol_scores[sym] = ivol

    if len(res_mom) < _MIN_CANDIDATES:
        return {}, {}

    beta_ref = {s: betas.get(s, 1.0) for s in res_mom}
    ortho_res_mom = orthogonalize_factor(res_mom, beta_ref)

    composite_builder = MultiFactorComposite()
    alphas = composite_builder.combine(
        {
            "residual_momentum": rank_and_gaussianize(ortho_res_mom),
            "skip_month_momentum": rank_and_gaussianize(skip_mom),
            "betting_against_beta": rank_and_gaussianize(bab_scores),
            "idiosyncratic_volatility": rank_and_gaussianize(ivol_scores),
        }
    )

    return alphas, betas


@dataclass(frozen=True, slots=True)
class InstitutionalMarketNeutralStrategy:
    """Institutional Multi-Factor Market-Neutral Strategy."""

    frequency: StrategyRebalanceFreq = StrategyRebalanceFreq.MONTHLY
    mandate: PortfolioMandate = PortfolioMandate.MARKET_NEUTRAL_LS
    max_single_weight: float = _DEFAULT_MAX_SINGLE_WEIGHT
    max_gross_leverage: float = _DEFAULT_MAX_GROSS_LEVERAGE
    risk_aversion: float = 1.0

    def is_rebalance_date(self, on: date, *, previous: date | None) -> bool:
        """Evaluate whether 'on' is the start of a new rebalance cycle."""
        if previous is None:
            return True
        if self.frequency == StrategyRebalanceFreq.MONTHLY:
            return (on.year, on.month) != (previous.year, previous.month)
        return (on.year, (on.month - 1) // 3) != (previous.year, (previous.month - 1) // 3)

    def target_weights(
        self,
        panel: dict[str, pd.DataFrame],
        on: date,
    ) -> dict[str, Decimal]:
        """Compute optimal target weights via multi-factor QP optimization."""
        cutoff = pd.Timestamp(on, tz=IST)
        returns_dict, price_dict = _extract_panel_series(panel, cutoff)

        if len(returns_dict) < _MIN_CANDIDATES:
            return {}

        returns_df = pd.DataFrame(returns_dict).dropna(how="all").fillna(0.0)
        market_returns = returns_df.mean(axis=1)

        alphas, betas = _compute_alpha_factors(returns_dict, price_dict, market_returns)
        if not alphas:
            return {}

        optimizer = LedoitWolfOptimizer(
            mandate=self.mandate,
            risk_aversion=self.risk_aversion,
            max_gross_leverage=self.max_gross_leverage,
            max_single_weight=self.max_single_weight,
        )

        opt_res = optimizer.optimize(alphas, returns_df, betas=betas)
        if not opt_res.converged:
            return {}

        result_weights: dict[str, Decimal] = {}
        for s, w in opt_res.weights.items():
            if abs(w) >= _MIN_WEIGHT_THRESHOLD:
                result_weights[s] = Decimal(str(round(w, 4)))

        return result_weights
