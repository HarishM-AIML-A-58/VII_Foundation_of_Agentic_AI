"""The Strategy interfaces (Phase 2).

Two shapes, because two genuinely different things are being described.

:class:`Strategy` proposes **entries with geometry** -- an entry price, a stop
and a target per name -- and the engine holds each position until its stop or
target is hit. That is a trade-by-trade model.

:class:`PortfolioStrategy` proposes **target weights on a schedule**. It ranks a
universe, holds the top slice, and replaces it at the next rebalance. There is
no per-name stop, because the exit rule is "you fell out of the ranking", not
"you moved against me by 2 ATR".

Trying to express the second as the first is what produced the churn this
project measured: forcing a monthly-rebalanced ranking into a stop/target
engine turned roughly twelve decisions a year into 465 round trips and
₹2.3 lakh of charges on ₹10 lakh of capital. Varsity's momentum-portfolio
chapter (M10 Ch16) is explicit that there is no per-stock stop; the two
protocols exist so that can be honoured rather than approximated.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Protocol, runtime_checkable

import pandas as pd

from trading_agent.domain.signal import TradeSignal


@runtime_checkable
class Strategy(Protocol):
    """Turns features into signals.

    Deliberately narrow: a strategy sees data and proposes trades. It does not
    size them, does not know the account balance, and cannot place orders --
    those belong to the risk and execution layers, which can then be tested
    independently of any strategy.
    """

    @property
    def name(self) -> str: ...

    def generate(self, features: pd.DataFrame, *, on: date) -> list[TradeSignal]:
        """Signals for session `on`, using only data up to and including it."""
        ...


@runtime_checkable
class PortfolioStrategy(Protocol):
    """Ranks a universe and asks for target weights on rebalance dates.

    The contract has three parts, and each exists to close a specific way this
    kind of strategy lies about itself:

    ``is_rebalance_date`` -- the schedule is the strategy's own, not the
    engine's. A monthly strategy that is asked for weights daily has silently
    become a daily strategy with a hundred times the turnover.

    ``target_weights`` -- returns symbol -> fraction of equity, using data up to
    and including ``on`` and nothing after. Weights are ``Decimal`` for the same
    reason money is: a portfolio of floats does not sum to one.

    ``rank`` -- the ordering behind the weights, exposed so a report can show
    *why* a name was held. A portfolio strategy whose ranking cannot be
    inspected is a black box, and under SEBI's retail algo framework that is a
    materially heavier regulatory lane than a disclosed, replicable rule.
    """

    @property
    def name(self) -> str: ...

    def is_rebalance_date(self, on: date, *, previous: date | None) -> bool:
        """True when the portfolio should be rebuilt at the close of ``on``."""
        ...

    def rank(self, panel: dict[str, pd.DataFrame], *, on: date) -> list[tuple[str, float]]:
        """Every eligible symbol with its score, best first."""
        ...

    def target_weights(self, panel: dict[str, pd.DataFrame], *, on: date) -> dict[str, Decimal]:
        """Symbol -> fraction of equity to hold after this rebalance."""
        ...
