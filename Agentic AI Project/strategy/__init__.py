"""Strategy protocols and implementations."""

from trading_agent.strategy.alpha_composite import (
    SELECTED_ALPHAS,
    AlphaComposite,
    attach_alpha_composite,
)
from trading_agent.strategy.baseline_ma_crossover import BaselineMaCrossover
from trading_agent.strategy.equal_weight_portfolio import EqualWeightPortfolio
from trading_agent.strategy.institutional_market_neutral import (
    InstitutionalMarketNeutralStrategy,
    StrategyRebalanceFreq,
)
from trading_agent.strategy.jev_calibrated_strategy import JevCalibratedStrategy
from trading_agent.strategy.momentum_composite import FACTOR_WEIGHTS, MomentumComposite
from trading_agent.strategy.nifty_momentum_portfolio import (
    NiftyMomentumStrategy,
    WeightingScheme,
)
from trading_agent.strategy.portfolio_optimizer import (
    LedoitWolfOptimizer,
    OptimizationResult,
    PortfolioMandate,
    compute_ledoit_wolf_covariance,
)
from trading_agent.strategy.protocols import PortfolioStrategy, Strategy
from trading_agent.strategy.value_investing import (
    PRESETS as VALUE_PRESETS,
)
from trading_agent.strategy.value_investing import (
    ValueInvestingStrategy,
    ValueRules,
)
from trading_agent.strategy.varsity_momentum_portfolio import (
    DEFAULT_PORTFOLIO_SIZE,
    MIN_TRACKING_UNIVERSE,
    RankingMethod,
    RebalanceFrequency,
    VarsityMomentumPortfolio,
)

__all__ = [
    "DEFAULT_PORTFOLIO_SIZE",
    "FACTOR_WEIGHTS",
    "MIN_TRACKING_UNIVERSE",
    "SELECTED_ALPHAS",
    "VALUE_PRESETS",
    "AlphaComposite",
    "BaselineMaCrossover",
    "EqualWeightPortfolio",
    "InstitutionalMarketNeutralStrategy",
    "JevCalibratedStrategy",
    "LedoitWolfOptimizer",
    "MomentumComposite",
    "NiftyMomentumStrategy",
    "OptimizationResult",
    "PortfolioMandate",
    "PortfolioStrategy",
    "RankingMethod",
    "RebalanceFrequency",
    "Strategy",
    "StrategyRebalanceFreq",
    "ValueInvestingStrategy",
    "ValueRules",
    "VarsityMomentumPortfolio",
    "WeightingScheme",
    "attach_alpha_composite",
    "compute_ledoit_wolf_covariance",
]
