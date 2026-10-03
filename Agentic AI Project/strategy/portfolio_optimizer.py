"""Institutional portfolio optimization: Ledoit-Wolf shrinkage & convex QP.

Implements:
1. Ledoit & Wolf (2004) "A well-conditioned estimator for large-dimensional
   covariance matrices" with constant correlation shrinkage target.
2. Convex quadratic programming portfolio optimizer under institutional constraints:
   - Dollar neutrality: sum(w_i) = 0 (for L/S EMN)
   - Beta neutrality: sum(w_i * beta_i) = 0
   - Gross leverage bounds: sum(|w_i|) <= L_max
   - Sector exposure bounds: |sum_{i in Sector_k} w_i| <= sector_cap
   - Single-name position concentration caps
   - Rebalance turnover friction penalty
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import minimize

__all__ = [
    "LedoitWolfOptimizer",
    "OptimizationResult",
    "PortfolioMandate",
    "compute_ledoit_wolf_covariance",
]

_MIN_ASSETS = 2
_ANNUAL_SESSIONS = 252.0
_EPSILON = 1e-12


class PortfolioMandate(StrEnum):
    """Institutional portfolio construction mandate."""

    MARKET_NEUTRAL_LS = "market_neutral_ls"
    LONG_EXTENSION_130_30 = "long_extension_130_30"
    LONG_ONLY = "long_only"


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """Outcome of institutional portfolio optimization."""

    weights: dict[str, float]
    expected_return: float
    portfolio_volatility: float
    gross_leverage: float
    net_exposure: float
    shrinkage_intensity: float
    converged: bool
    status_message: str


def compute_ledoit_wolf_covariance(
    returns: pd.DataFrame,
) -> tuple[npt.NDArray[np.float64], float]:
    """Compute the Ledoit-Wolf (2004) constant-correlation shrinkage covariance."""
    arr = returns.to_numpy(dtype=float)
    t_sessions, n_assets = arr.shape
    if t_sessions < _MIN_ASSETS or n_assets < _MIN_ASSETS:
        cov = np.cov(arr, rowvar=False)
        return np.atleast_2d(cov), 0.0

    mean_x = np.mean(arr, axis=0)
    demeaned = arr - mean_x

    sample_cov = np.dot(demeaned.T, demeaned) / t_sessions
    var = np.diag(sample_cov)
    std = np.sqrt(np.maximum(var, _EPSILON))

    std_outer = np.outer(std, std)
    r_bar = (np.sum(sample_cov / std_outer) - n_assets) / (n_assets * (n_assets - 1))

    prior = r_bar * std_outer
    np.fill_diagonal(prior, var)

    y2 = demeaned**2
    phi_mat = (
        np.dot(y2.T, y2) / t_sessions
        - 2 * sample_cov * np.dot(demeaned.T, demeaned) / t_sessions
        + sample_cov**2
    )
    phi = float(np.sum(phi_mat))
    gamma = float(np.sum((sample_cov - prior) ** 2))

    term1 = np.dot((demeaned**3).T, demeaned) / t_sessions
    rho_diag = np.diag(term1)
    rho_mat = np.outer(std, 1.0 / std) * np.outer(np.ones(n_assets), rho_diag)
    rho_mat = 0.5 * (rho_mat + rho_mat.T)
    rho = float(
        np.sum(np.diag(phi_mat))
        + r_bar * np.sum(sample_cov / std_outer * (phi_mat - np.diag(np.diag(phi_mat))))
    )

    kappa = (phi - rho) / gamma if gamma > _EPSILON else 0.0
    delta = max(0.0, min(1.0, kappa / t_sessions))
    shrunk = delta * prior + (1.0 - delta) * sample_cov
    return shrunk, float(delta)


@dataclass(frozen=True, slots=True)
class LedoitWolfOptimizer:
    """Convex Quadratic Programming optimizer with Ledoit-Wolf shrinkage."""

    mandate: PortfolioMandate = PortfolioMandate.MARKET_NEUTRAL_LS
    risk_aversion: float = 1.0
    turnover_penalty: float = 0.05
    max_gross_leverage: float = 2.0
    max_single_weight: float = 0.05
    sector_cap: float = 0.03

    def _initial_and_bounds(
        self,
        alpha_vec: npt.NDArray[np.float64],
        n_assets: int,
    ) -> tuple[npt.NDArray[np.float64], list[tuple[float, float]], float]:
        """Compute initial weight vector x0, bounds, and target net exposure."""
        if self.mandate == PortfolioMandate.LONG_ONLY:
            x0 = np.ones(n_assets) / n_assets
            bounds = [(0.0, self.max_single_weight) for _ in range(n_assets)]
            return x0, bounds, 1.0

        if self.mandate == PortfolioMandate.LONG_EXTENSION_130_30:
            x0 = np.ones(n_assets) / n_assets
            bounds = [(-self.max_single_weight, self.max_single_weight) for _ in range(n_assets)]
            return x0, bounds, 1.0

        # PortfolioMandate.MARKET_NEUTRAL_LS
        pos = alpha_vec > np.median(alpha_vec)
        x0 = np.zeros(n_assets)
        n_pos = int(np.sum(pos))
        n_neg = n_assets - n_pos
        if n_pos > 0:
            x0[pos] = 1.0 / n_pos
        if n_neg > 0:
            x0[~pos] = -1.0 / n_neg
        bounds = [(-self.max_single_weight, self.max_single_weight) for _ in range(n_assets)]
        return x0, bounds, 0.0

    def _build_constraints(
        self,
        common_symbols: list[str],
        target_net: float,
        betas: Mapping[str, float] | None,
        sectors: Mapping[str, str] | None,
    ) -> list[dict[str, object]]:
        """Construct equality and inequality constraints for SLSQP."""
        n_assets = len(common_symbols)
        constraints: list[dict[str, object]] = [
            {
                "type": "eq",
                "fun": lambda w: float(np.sum(w) - target_net),
                "jac": lambda _w: np.ones(n_assets),
            },
            {
                "type": "ineq",
                "fun": lambda w: float(self.max_gross_leverage - np.sum(np.abs(w))),
            },
        ]

        if self.mandate == PortfolioMandate.MARKET_NEUTRAL_LS and betas is not None:
            b_vec = np.array([betas.get(s, 1.0) for s in common_symbols], dtype=float)
            constraints.append(
                {
                    "type": "eq",
                    "fun": lambda w: float(np.dot(w, b_vec)),
                    "jac": lambda _w: b_vec,
                }
            )

        if sectors is not None:
            for sec in set(sectors.values()):
                sec_mask = np.array([1.0 if sectors.get(s) == sec else 0.0 for s in common_symbols])
                if np.sum(sec_mask) > 0:
                    constraints.extend(
                        [
                            {
                                "type": "ineq",
                                "fun": lambda w, m=sec_mask: float(self.sector_cap - np.dot(w, m)),
                            },
                            {
                                "type": "ineq",
                                "fun": lambda w, m=sec_mask: float(np.dot(w, m) + self.sector_cap),
                            },
                        ]
                    )

        return constraints

    def optimize(
        self,
        alphas: Mapping[str, float],
        returns_history: pd.DataFrame,
        betas: Mapping[str, float] | None = None,
        sectors: Mapping[str, str] | None = None,
        current_weights: Mapping[str, float] | None = None,
    ) -> OptimizationResult:
        """Run quadratic programming portfolio optimization."""
        common_symbols = sorted(set(alphas.keys()) & set(returns_history.columns))
        n_assets = len(common_symbols)
        if n_assets < _MIN_ASSETS:
            return OptimizationResult(
                weights=dict.fromkeys(alphas, 0.0),
                expected_return=0.0,
                portfolio_volatility=0.0,
                gross_leverage=0.0,
                net_exposure=0.0,
                shrinkage_intensity=0.0,
                converged=False,
                status_message="Insufficient assets for optimization",
            )

        alpha_vec = np.array([alphas[s] for s in common_symbols], dtype=float)
        hist = returns_history[common_symbols].dropna()
        cov_matrix, shrinkage_delta = compute_ledoit_wolf_covariance(hist)
        cov_annual = cov_matrix * _ANNUAL_SESSIONS

        w0 = np.zeros(n_assets, dtype=float)
        if current_weights:
            for i, s in enumerate(common_symbols):
                w0[i] = current_weights.get(s, 0.0)

        x0, bounds, target_net = self._initial_and_bounds(alpha_vec, n_assets)
        constraints = self._build_constraints(common_symbols, target_net, betas, sectors)

        def objective(w: npt.NDArray[np.float64]) -> float:
            risk = 0.5 * self.risk_aversion * float(np.dot(w.T, np.dot(cov_annual, w)))
            ret = float(np.dot(w, alpha_vec))
            turnover = 0.5 * self.turnover_penalty * float(np.sum((w - w0) ** 2))
            return risk - ret + turnover

        def jacobian(w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            gradient = (
                self.risk_aversion * np.dot(cov_annual, w)
                - alpha_vec
                + self.turnover_penalty * (w - w0)
            )
            return np.asarray(gradient, dtype=np.float64)

        res = minimize(
            objective,
            x0,
            jac=jacobian,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            options={"maxiter": 300, "ftol": 1e-7},
        )

        w_opt = res.x if res.success else x0
        weights_dict = {s: float(w_opt[i]) for i, s in enumerate(common_symbols)}

        exp_ret = float(np.dot(w_opt, alpha_vec))
        var_p = float(np.dot(w_opt.T, np.dot(cov_annual, w_opt)))
        vol_p = float(np.sqrt(max(1e-8, var_p)))

        return OptimizationResult(
            weights=weights_dict,
            expected_return=exp_ret,
            portfolio_volatility=vol_p,
            gross_leverage=float(np.sum(np.abs(w_opt))),
            net_exposure=float(np.sum(w_opt)),
            shrinkage_intensity=shrinkage_delta,
            converged=bool(res.success),
            status_message=str(res.message),
        )
