"""Statistical testing for backtests and multi-trial evaluation.

Implements the Deflated Sharpe Ratio (DSR) and Probabilistic Sharpe Ratio (PSR)
from Bailey & López de Prado (2014), "The Deflated Sharpe Ratio: Correcting for
Selection Bias, Backtest Overfitting, and Non-Normality" (Journal of Portfolio
Management).

A backtest Sharpe ratio of +0.24 over ~2,200 sessions has an estimated standard
error of ~0.33. If 20 different parameter or strategy variants were tested to find
it, the expected maximum Sharpe by chance alone is positive. The Deflated Sharpe
Ratio calculates the probability that the observed Sharpe exceeds zero after
penalizing for:
1. Non-normality (skewness and kurtosis of daily returns)
2. Sample length (number of sessions T)
3. Number of independent trials N
4. Variance of Sharpes across the trials V
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

__all__ = [
    "DeflatedSharpeResult",
    "calculate_return_moments",
    "deflated_sharpe_ratio",
    "probabilistic_sharpe_ratio",
]

#: Euler-Mascheroni constant
_EULER_MASCHERONI = 0.57721566490153286


@dataclass(frozen=True, slots=True)
class DeflatedSharpeResult:
    """The outcome of a Deflated Sharpe Ratio test."""

    observed_sharpe: float
    expected_max_sharpe: float
    dsr: float  # Probability in [0, 1]
    is_significant: bool  # True if DSR >= significance_level (default 0.95)
    trials: int
    sessions: int
    skewness: float
    kurtosis: float


_MIN_SAMPLE_SIZE = 3
_MIN_SESSIONS = 2


def calculate_return_moments(
    returns: pd.Series | Sequence[float],
) -> tuple[float, float, float, float]:
    """Calculate mean, standard deviation, sample skewness, and sample kurtosis."""
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n < _MIN_SAMPLE_SIZE:
        return 0.0, 0.0, 0.0, 3.0

    mean = float(np.mean(arr))
    diff = arr - mean
    var = float(np.mean(diff**2))
    std = math.sqrt(var) if var > 0 else 0.0
    if std == 0.0:
        return mean, 0.0, 0.0, 3.0

    skewness = float(np.mean(diff**3) / (std**3))
    kurtosis = float(np.mean(diff**4) / (std**4))  # Pearson kurtosis (normal = 3)
    return mean, std, skewness, kurtosis


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    benchmark_sharpe: float = 0.0,
    *,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    n_sessions: int,
    is_annualized: bool = True,
    trading_days_per_year: int = 250,
) -> float:
    """Probabilistic Sharpe Ratio (PSR) testing if SR > benchmark_sharpe.

    PSR computes Phi((SR - SR*) / sigma_SR), where sigma_SR accounts for
    skewness and fat tails (kurtosis) via Mertens (2002) variance:

        sigma^2 = (1 - gamma_3 * SR + ((gamma_4 - 1) / 4) * SR^2) / (T - 1)

    The Mertens variance operates on the per-observation (daily) Sharpe ratio.
    If ``is_annualized`` is True (default), ``observed_sharpe`` and
    ``benchmark_sharpe`` are de-annualized by dividing by
    sqrt(trading_days_per_year) before computing the standard error.
    """
    if n_sessions <= 1:
        return 0.5

    sr_scale = math.sqrt(trading_days_per_year) if is_annualized else 1.0
    sr_obs = observed_sharpe / sr_scale
    sr_bm = benchmark_sharpe / sr_scale

    # Variance of the Sharpe ratio estimator under non-normality (Mertens 2002)
    sr2 = sr_obs**2
    variance = (1.0 - skewness * sr_obs + ((kurtosis - 1.0) / 4.0) * sr2) / (n_sessions - 1)
    if variance <= 0.0:
        return 1.0 if sr_obs > sr_bm else 0.0

    std_err = math.sqrt(variance)
    z = (sr_obs - sr_bm) / std_err
    return NormalDist().cdf(z)


def deflated_sharpe_ratio(
    observed_sharpe: float,
    *,
    trials: int,
    variance_of_sharpes: float,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    n_sessions: int,
    is_annualized: bool = True,
    trading_days_per_year: int = 250,
    significance_level: float = 0.95,
) -> DeflatedSharpeResult:
    """Compute the Deflated Sharpe Ratio (DSR) discounting for multiple trials.

    Parameters:
    - observed_sharpe: Best observed Sharpe ratio (annualized if is_annualized=True).
    - trials: Number of independent strategy configurations or parameter combinations tested.
    - variance_of_sharpes: Variance of the Sharpe ratios across all tested trials.
    - skewness: Skewness of the daily return distribution.
    - kurtosis: Pearson kurtosis of the daily return distribution (normal = 3.0).
    - n_sessions: Total number of observations (sessions) T.
    - is_annualized: Whether observed_sharpe and variance_of_sharpes are annualized.
    - trading_days_per_year: Trading sessions per year (default 250).
    - significance_level: Threshold to reject the null hypothesis (default 0.95).
    """
    if trials < 1:
        raise ValueError(f"trials must be >= 1, got {trials}")
    if n_sessions < _MIN_SESSIONS:
        raise ValueError(f"n_sessions must be >= {_MIN_SESSIONS}, got {n_sessions}")

    sr_scale = math.sqrt(trading_days_per_year) if is_annualized else 1.0
    daily_var = variance_of_sharpes / (sr_scale**2) if is_annualized else variance_of_sharpes

    if trials == 1 or daily_var <= 0.0:
        # Single trial: benchmark is zero
        expected_max_daily_sr = 0.0
    else:
        norm = NormalDist()
        # Extreme value expectation under N standard Gaussian trials using Euler-Mascheroni constant
        z1 = norm.inv_cdf(1.0 - 1.0 / trials)
        z2 = norm.inv_cdf(1.0 - 1.0 / (trials * math.e))
        expected_max_daily_sr = math.sqrt(daily_var) * (
            (1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2
        )

    expected_max_annualized_sr = expected_max_daily_sr * sr_scale

    dsr = probabilistic_sharpe_ratio(
        observed_sharpe,
        expected_max_annualized_sr,
        skewness=skewness,
        kurtosis=kurtosis,
        n_sessions=n_sessions,
        is_annualized=is_annualized,
        trading_days_per_year=trading_days_per_year,
    )

    return DeflatedSharpeResult(
        observed_sharpe=observed_sharpe,
        expected_max_sharpe=expected_max_annualized_sr,
        dsr=dsr,
        is_significant=dsr >= significance_level,
        trials=trials,
        sessions=n_sessions,
        skewness=skewness,
        kurtosis=kurtosis,
    )
