"""Equal-weight the whole eligible universe. No ranking at all.

**This exists as a control, and it is the most important strategy in the
package.** Any portfolio strategy that ranks a universe and holds a slice of it
is making two bets at once: that equal-weighting beats the index's own
weighting, and that its ranking beats not ranking. Benchmarking such a strategy
against a cap-weighted index cannot separate the two, so a strategy can look
skilful while its ranking is actively destroying value.

Measured on the NIFTY 50 over 2015-01-01 to 2026-08-07, monthly rebalance, after
Zerodha delivery charges, on the **point-in-time panel of 82 symbols** -- every
name that was in the index during the window, including the 33 since dropped.
Walk-forward is 24-month train / 6-month test, 20 windows:

====================================  =========  ========  =====  ==========
configuration                          in-samp       OOS     eff  OOS Sharpe
====================================  =========  ========  =====  ==========
NIFTY 50 buy-and-hold (same span)             -  +200.39%      -           -
12-1 momentum, vol-adjusted, top 15    +79.80%   +88.41%   1.11       +0.05
equal-weight all 82 names (this)       +146.66%  +76.76%   0.52       -0.02
12-1 momentum, top 20                  +70.01%   +71.73%   1.02       -0.04
12-1 momentum, top 12                  +72.64%   +64.13%   0.88       -0.08
12-1 momentum, top 12, 200DMA filter   +86.47%   +56.58%   0.65       -0.13
====================================  =========  ========  =====  ==========

**Nothing beat the index.** The best out-of-sample figure is 88% against 200%
for holding NIFTY 50 and doing nothing, and only one configuration reached a
positive out-of-sample Sharpe at all.

**A correction worth reading, because it was mine.** On the earlier
survivorship-biased panel -- today's 49 constituents only -- this control
returned +237.48% and beat every momentum variant, and the conclusion recorded
here was that the ranking subtracted value. Adding the 33 dropped constituents
cut the control to +146.66% in-sample, and out-of-sample the vol-adjusted
ranking now edges it. Most of what looked like proof that ranking was worthless
was the control quietly enjoying the same survivorship bias, only more of it:
holding *everything* benefits most from the failures being absent.

What survives both panels: the equal-weight *tilt* is worth a great deal in
apparent return and almost none of it is robust, this control's walk-forward
efficiency is the worst in the table at 0.52, and turnover is expensive. It
buys once and holds, because a name only leaves when it leaves the index, and
₹0 of round-trip charges against ₹1.9 lakh for the signal engine on the same
capital is the clearest statement of what churn costs.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pandas as pd

from trading_agent.observability import get_logger

__all__ = ["EqualWeightPortfolio"]

log = get_logger(__name__)

#: Sessions of history a name needs before it is held. Matched to the momentum
#: strategies' requirement so the control and the thing it controls for see the
#: same investable set -- otherwise the control gets a head start on names the
#: ranked strategy cannot score yet.
DEFAULT_MIN_HISTORY = 253


class EqualWeightPortfolio:
    """Hold every eligible symbol at equal weight, rebalanced on a schedule.

    ``max_names`` caps how many are held. It exists because the uncapped
    control silently stops being a control on a wide universe: the engine
    refuses any target below a tradeable floor (0.5% of equity), so 450 names at
    0.22% each leaves the book flat, the run reports +0.00%, and the verdict
    then compares the ranked strategy against a portfolio that never traded --
    a *false* comparison, and the most misleading output this package can
    produce.

    The cap keeps the control a control. It selects in symbol order, which is
    arbitrary but has nothing to do with momentum, price or size -- and that is
    the requirement: a no-ranking control must not smuggle in a ranking. The
    alternative to capping is more capital, and the caller is told so.
    """

    __slots__ = ("_max_names", "_min_history")

    def __init__(
        self,
        *,
        min_history_sessions: int = DEFAULT_MIN_HISTORY,
        max_names: int | None = None,
    ) -> None:
        if min_history_sessions < 1:
            raise ValueError(f"min_history_sessions must be at least 1, got {min_history_sessions}")
        if max_names is not None and max_names < 1:
            raise ValueError(f"max_names must be at least 1, got {max_names}")
        self._min_history = min_history_sessions
        self._max_names = max_names

    @property
    def name(self) -> str:
        if self._max_names is None:
            return "equal_weight_all_monthly"
        return f"equal_weight_top{self._max_names}_monthly"

    def is_rebalance_date(self, on: date, *, previous: date | None) -> bool:
        """First trading session of a new calendar month, matching the momentum
        strategies so the comparison is not confounded by schedule."""
        if previous is None:
            return True
        return (on.year, on.month) != (previous.year, previous.month)

    def _eligible(self, panel: dict[str, pd.DataFrame], on: date) -> list[str]:
        usable: list[str] = []
        for symbol, frame in panel.items():
            index = pd.DatetimeIndex(frame.index)
            if len(frame[index <= pd.Timestamp(on, tz=index.tz)]) >= self._min_history:
                usable.append(symbol)
        usable.sort()
        if self._max_names is not None and len(usable) > self._max_names:
            dropped = len(usable) - self._max_names
            log.info(
                "equal_weight_control_capped",
                on=on.isoformat(),
                held=self._max_names,
                dropped=dropped,
                detail=(
                    "the universe is wider than the account can equal-weight above the "
                    "tradeable floor; the cap selects in symbol order, which is unrelated "
                    "to any ranking signal"
                ),
            )
            usable = usable[: self._max_names]
        return usable

    def rank(self, panel: dict[str, pd.DataFrame], *, on: date) -> list[tuple[str, float]]:
        """Every eligible name at the same score.

        Not an ordering, and deliberately not faked into one: the point of this
        strategy is the absence of a ranking, so returning an arbitrary order
        with distinct scores would misrepresent it in any report that shows the
        top names.
        """
        return [(symbol, 0.0) for symbol in self._eligible(panel, on)]

    def target_weights(self, panel: dict[str, pd.DataFrame], *, on: date) -> dict[str, Decimal]:
        eligible = self._eligible(panel, on)
        if not eligible:
            return {}
        weight = (Decimal(1) / Decimal(len(eligible))).quantize(Decimal("0.000001"))
        weights = dict.fromkeys(eligible, weight)
        # Remainder to the first name so the weights sum to exactly one.
        weights[eligible[0]] += Decimal(1) - weight * len(eligible)
        return weights
