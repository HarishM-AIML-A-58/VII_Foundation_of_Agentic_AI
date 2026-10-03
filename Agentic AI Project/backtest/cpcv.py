"""Combinatorial Purged Cross-Validation (CPCV) and Probability of Backtest Overfitting.

Implements Marcos López de Prado (2018) "Advances in Financial Machine Learning":
- Chapter 12: Combinatorial Purged Cross-Validation (CPCV)
- Chapter 11: The Probability of Backtest Overfitting (PBO)

Generates C(N, k) combinations of train/test folds, purges overlapping event
horizons, applies post-test embargoes, and measures the empirical distribution
of out-of-sample Sharpe ratios to detect selection bias.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from trading_agent.backtest.metrics import TRADING_DAYS_PER_YEAR

__all__ = [
    "CPCVFold",
    "CPCVPathResult",
    "CPCVResult",
    "CombinatorialPurgedCV",
    "calculate_pbo",
]

_MIN_GROUPS = 3
_MIN_TEST_GROUPS = 1
_DEFAULT_EMBARGO_SESSIONS = 5
_MIN_SESSIONS_FOR_SHARPE = 10
_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class CPCVFold:
    """One train/test split within Combinatorial Purged Cross-Validation."""

    fold_index: int
    train_indices: npt.NDArray[np.int64]
    test_indices: npt.NDArray[np.int64]
    purged_indices: npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class CPCVPathResult:
    """Backtest outcome across one CPCV path."""

    path_id: int
    train_sharpe: float
    test_sharpe: float
    train_return: float
    test_return: float
    overfit: bool


@dataclass(frozen=True, slots=True)
class CPCVResult:
    """Institutional evaluation across all Combinatorial Purged CV paths."""

    total_paths: int
    pbo: float  # Probability of Backtest Overfitting in [0.0, 1.0]
    mean_oos_sharpe: float
    median_oos_sharpe: float
    std_oos_sharpe: float
    paths: list[CPCVPathResult]


def calculate_pbo(
    train_sharpes: Sequence[float],
    test_sharpes: Sequence[float],
) -> float:
    """Compute Probability of Backtest Overfitting (PBO).

    PBO is the probability that the strategy that performs best in-sample
    ranks below the median out-of-sample (or fraction of paths where OOS
    underperforms the median IS benchmark).
    """
    n = len(train_sharpes)
    if n == 0 or len(test_sharpes) != n:
        return 0.0

    is_median = float(np.median(train_sharpes))
    overfit_count = sum(1 for oos in test_sharpes if oos < is_median)
    return float(overfit_count / n)


@dataclass(frozen=True, slots=True)
class CombinatorialPurgedCV:
    """Combinatorial Purged Cross-Validation partitioner."""

    n_groups: int = 6
    k_test_groups: int = 2
    embargo_sessions: int = _DEFAULT_EMBARGO_SESSIONS

    def __post_init__(self) -> None:
        if self.n_groups < _MIN_GROUPS:
            raise ValueError(f"n_groups must be >= {_MIN_GROUPS}, got {self.n_groups}")
        if self.k_test_groups < _MIN_TEST_GROUPS or self.k_test_groups >= self.n_groups:
            raise ValueError(
                f"k_test_groups must be between 1 and {self.n_groups - 1}, got {self.k_test_groups}"
            )

    def generate_folds(
        self,
        total_sessions: int,
        holding_period: int = 1,
    ) -> list[CPCVFold]:
        """Generate purged and embargoed train/test folds."""
        if total_sessions < self.n_groups:
            return []

        # Divide indices into n contiguous chronological groups
        group_bounds = np.linspace(0, total_sessions, self.n_groups + 1, dtype=int)
        groups = [np.arange(group_bounds[i], group_bounds[i + 1]) for i in range(self.n_groups)]

        combos = list(itertools.combinations(range(self.n_groups), self.k_test_groups))
        folds: list[CPCVFold] = []

        for fold_idx, test_grp_indices in enumerate(combos):
            test_mask = np.zeros(total_sessions, dtype=bool)
            for g_idx in test_grp_indices:
                test_mask[groups[g_idx]] = True

            train_mask = ~test_mask
            purged_mask = np.zeros(total_sessions, dtype=bool)

            for g_idx in test_grp_indices:
                test_start = int(groups[g_idx][0])
                test_end = int(groups[g_idx][-1])

                if holding_period > 1:
                    purge_start = max(0, test_start - holding_period + 1)
                    idx_range = np.arange(purge_start, test_start)
                    # Purge only training samples (not samples already in test set)
                    train_to_purge = idx_range[~test_mask[idx_range]]
                    purged_mask[train_to_purge] = True
                    train_mask[train_to_purge] = False

                if self.embargo_sessions > 0:
                    embargo_end = min(total_sessions, test_end + self.embargo_sessions + 1)
                    idx_range = np.arange(test_end + 1, embargo_end)
                    train_to_embargo = idx_range[~test_mask[idx_range]]
                    purged_mask[train_to_embargo] = True
                    train_mask[train_to_embargo] = False

            folds.append(
                CPCVFold(
                    fold_index=fold_idx,
                    train_indices=np.where(train_mask)[0],
                    test_indices=np.where(test_mask)[0],
                    purged_indices=np.where(purged_mask)[0],
                )
            )

        return folds

    def evaluate_paths(
        self,
        daily_returns: pd.Series,
        holding_period: int = 1,
    ) -> CPCVResult:
        """Run CPCV evaluation on an equity curve's daily returns."""
        clean = daily_returns.dropna().to_numpy(dtype=float)
        total_sessions = len(clean)
        folds = self.generate_folds(total_sessions, holding_period=holding_period)

        if not folds:
            return CPCVResult(
                total_paths=0,
                pbo=1.0,
                mean_oos_sharpe=0.0,
                median_oos_sharpe=0.0,
                std_oos_sharpe=0.0,
                paths=[],
            )

        paths: list[CPCVPathResult] = []
        is_sharpes: list[float] = []
        oos_sharpes: list[float] = []

        for fold in folds:
            train_ret = clean[fold.train_indices]
            test_ret = clean[fold.test_indices]

            s_train = self._compute_sharpe(train_ret)
            s_test = self._compute_sharpe(test_ret)

            cum_train = float(np.prod(1.0 + train_ret) - 1.0) if len(train_ret) > 0 else 0.0
            cum_test = float(np.prod(1.0 + test_ret) - 1.0) if len(test_ret) > 0 else 0.0

            is_sharpes.append(s_train)
            oos_sharpes.append(s_test)

            paths.append(
                CPCVPathResult(
                    path_id=fold.fold_index,
                    train_sharpe=s_train,
                    test_sharpe=s_test,
                    train_return=cum_train,
                    test_return=cum_test,
                    overfit=s_test < s_train,
                )
            )

        pbo = calculate_pbo(is_sharpes, oos_sharpes)
        oos_arr = np.array(oos_sharpes)

        return CPCVResult(
            total_paths=len(paths),
            pbo=pbo,
            mean_oos_sharpe=float(np.mean(oos_arr)),
            median_oos_sharpe=float(np.median(oos_arr)),
            std_oos_sharpe=float(np.std(oos_arr, ddof=1)) if len(oos_arr) > 1 else 0.0,
            paths=paths,
        )

    @staticmethod
    def _compute_sharpe(returns: npt.NDArray[np.float64]) -> float:
        """Annualized Sharpe ratio assuming zero risk-free rate for normalized comparison."""
        if len(returns) < _MIN_SESSIONS_FOR_SHARPE:
            return 0.0
        mean = float(np.mean(returns))
        std = float(np.std(returns, ddof=1))
        if std < _EPSILON:
            return 0.0
        return (mean / std) * float(np.sqrt(TRADING_DAYS_PER_YEAR))
