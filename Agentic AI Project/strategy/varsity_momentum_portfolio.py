"""Varsity's momentum portfolio, implemented as written.

Source: Zerodha Varsity Module 10 (Trading Systems), Chapter 16 -- "Momentum
Portfolios". The chapter gives six steps, and this is a literal reading of them
rather than an interpretation:

1. **Define the tracking universe.** Varsity suggests BSE 500, and 150-200
   names minimum for a 12-15 stock portfolio. The curated store holds the
   NIFTY 50, which is thinner than the chapter recommends -- so a top-12 slice
   of it is a top *quartile*, not a top decile. That is a real weakness of this
   configuration and is reported rather than hidden.
2. **Set up the data.** Corporate-action adjusted closes, one year of lookback.
   The engine's panel is already back-adjusted before features are computed.
3. **Calculate returns.** ``[ending value / starting value] - 1`` over the
   lookback. The chapter uses twelve months.
4. **Rank** them, highest return first.
5. **Create the portfolio** from the top N, **equal weighted**: "if the capital
   available is Rs.200,000 and there are 12 stocks, buy Rs.16,666 of each".
6. **Rebalance monthly**, on the first trading day of the month, computing the
   ranking after the previous month's close.

Three variants, because the chapter explicitly invites them and the differences
are worth measuring separately:

``PLAIN`` is step 3 as written -- a twelve-month total return.

``SKIP_MONTH`` is the 12-1 form: twelve-month return excluding the most recent
month. A reader asked Varsity about this (citing SSRN 3247865) and Karthik's
answer was that he was unsure why one would skip. The reason is short-term
reversal: last month's biggest gainers mean-revert, so including that month
pollutes a twelve-month trend signal with a one-month contrarian one.

``VOL_ADJUSTED`` divides the return by realised volatility, which is the
Sharpe-like ranking a Varsity commenter (Subodh Taigor) proposed. Raw momentum
systematically selects the most volatile names; dividing by volatility turns
"biggest move" into "most consistent move".

**What is deliberately NOT here.** No per-stock stop loss. The chapter says
there is none, and the exit rule is falling out of the ranking. Varsity's answer
on stops is a 2% *portfolio-level* stop, which belongs to the engine's regime
and drawdown controls, not to the ranking.

**Measured, and it does not clear the bar.** On the point-in-time panel of 82
NIFTY 50 symbols over 2015-2026, walk-forward 24/6 across 20 windows, after
Zerodha delivery charges:

====================================  =========  ========  =====  ==========
configuration                          in-samp       OOS     eff  OOS Sharpe
====================================  =========  ========  =====  ==========
NIFTY 50 buy-and-hold (same span)             -  +200.39%      -           -
vol-adjusted, top 15                   +79.80%   +88.41%   1.11       +0.05
plain 12-1, top 20                     +70.01%   +71.73%   1.02       -0.04
plain 12-1, top 12                     +72.64%   +64.13%   0.88       -0.08
====================================  =========  ========  =====  ==========

Efficiency above 1.0 says the configuration generalised -- it did not decay out
of sample, which is more than the signal-engine strategies manage. It generalised
a return of 88% against 200% for holding the index and doing nothing. Read the
benchmark row before the efficiency column.

**Varsity's own warning, which the numbers bear out.** "A price-based momentum
strategy works well only when the market is trending up. When the markets turn
choppy, the momentum strategy performs poorly, and when the markets go down, the
momentum portfolio bleeds heavier than the markets itself." That is what the
200-DMA regime filter in the engine exists to answer.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import StrEnum

import pandas as pd

from trading_agent.observability import get_logger

__all__ = [
    "DEFAULT_LOOKBACK_SESSIONS",
    "DEFAULT_PORTFOLIO_SIZE",
    "DEFAULT_SKIP_SESSIONS",
    "MIN_TRACKING_UNIVERSE",
    "RankingMethod",
    "RebalanceFrequency",
    "VarsityMomentumPortfolio",
]

log = get_logger(__name__)

#: Twelve months of trading sessions. NSE runs roughly 250 a year; 252 is the
#: conventional figure and the one the feature registry already uses.
DEFAULT_LOOKBACK_SESSIONS = 252

#: One month, for the 12-1 skip.
DEFAULT_SKIP_SESSIONS = 21

#: "A good momentum portfolio contains about 10-12 stocks. I'm comfortable with
#: up to 15." -- M10 Ch16. Twelve is the chapter's worked example.
DEFAULT_PORTFOLIO_SIZE = 12

#: A return needs two prices and a standard deviation needs two returns.
_MIN_WINDOW = 2

#: "I would suggest you have at least 150-200 stocks in your tracking universe
#: if you wish to build a momentum portfolio of 12-15 stocks." -- M10 Ch16.
#: The NIFTY 50 is well below this, so the strategy says so out loud.
MIN_TRACKING_UNIVERSE = 150


class RankingMethod(StrEnum):
    """How step 3 of the chapter is computed."""

    #: Twelve-month total return, exactly as the chapter writes it.
    PLAIN = "plain"
    #: Twelve-month return skipping the most recent month (12-1).
    SKIP_MONTH = "skip_month"
    #: Twelve-month return divided by realised volatility.
    VOL_ADJUSTED = "vol_adjusted"


class RebalanceFrequency(StrEnum):
    """How frequently the portfolio rebalances."""

    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMI_ANNUAL = "semi_annual"


class VarsityMomentumPortfolio:
    """Rank on trailing return, hold top N equally weighted; rebalance on schedule."""

    __slots__ = (
        "_frequency",
        "_lookback",
        "_method",
        "_min_history",
        "_size",
        "_skip",
        "_vol_window",
        "_warned_thin_universe",
    )

    def __init__(
        self,
        *,
        method: RankingMethod = RankingMethod.PLAIN,
        size: int = DEFAULT_PORTFOLIO_SIZE,
        lookback_sessions: int = DEFAULT_LOOKBACK_SESSIONS,
        skip_sessions: int = DEFAULT_SKIP_SESSIONS,
        vol_window: int = 60,
        frequency: RebalanceFrequency | str = RebalanceFrequency.MONTHLY,
    ) -> None:
        if size < 1:
            raise ValueError(f"size must be at least 1, got {size}")
        if lookback_sessions < _MIN_WINDOW:
            raise ValueError(
                f"lookback_sessions must be at least {_MIN_WINDOW}, got {lookback_sessions}"
            )
        if skip_sessions < 0:
            raise ValueError(f"skip_sessions must be non-negative, got {skip_sessions}")
        if skip_sessions >= lookback_sessions:
            raise ValueError(
                f"skip_sessions {skip_sessions} must be below lookback_sessions "
                f"{lookback_sessions}, or the ranking window is empty"
            )
        if vol_window < _MIN_WINDOW:
            raise ValueError(f"vol_window must be at least {_MIN_WINDOW}, got {vol_window}")

        self._frequency = (
            frequency
            if isinstance(frequency, RebalanceFrequency)
            else RebalanceFrequency(str(frequency).lower())
        )
        self._method = method
        self._size = size
        self._lookback = lookback_sessions
        self._skip = skip_sessions if method is RankingMethod.SKIP_MONTH else 0
        self._vol_window = vol_window
        # A name needs the whole ranking window plus the skip before it can be
        # scored at all. Ranking on a partial window sorts the newly listed to
        # the top, because a short window is a low bar.
        self._min_history = lookback_sessions + 1
        # Annotated: without it mypy narrows the attribute to Literal[False]
        # from this assignment and calls every "it did warn" assertion dead code.
        self._warned_thin_universe: bool = False

    @property
    def frequency(self) -> RebalanceFrequency:
        return self._frequency

    @property
    def name(self) -> str:
        months = round(self._lookback / 21)
        label = {
            RankingMethod.PLAIN: f"varsity_momentum_{months}m",
            RankingMethod.SKIP_MONTH: f"varsity_momentum_{months}m_skip1",
            RankingMethod.VOL_ADJUSTED: f"varsity_momentum_{months}m_voladj",
        }[self._method]
        freq_suffix = {
            RebalanceFrequency.MONTHLY: "",
            RebalanceFrequency.QUARTERLY: "_quarterly",
            RebalanceFrequency.SEMI_ANNUAL: "_semiannual",
        }[self._frequency]
        return f"{label}_top{self._size}{freq_suffix}"

    @property
    def size(self) -> int:
        return self._size

    @property
    def has_warned_thin_universe(self) -> bool:
        """True once the thin-universe warning has been emitted by this instance.

        Exposed so the warn-once contract is testable as a fact about the object
        rather than by scraping a log stream. Where structlog writes depends on
        whether logging has been configured in the process, so a test that reads
        stdout passes alone and fails in the full suite -- which it did.
        """
        return self._warned_thin_universe

    # ------------------------------------------------------------- schedule

    def is_rebalance_date(self, on: date, *, previous: date | None) -> bool:
        """True on the first trading session of a new rebalance cycle.

        ``previous`` is the last session seen, so this is derived from the
        trading calendar the panel actually contains rather than from a holiday
        table -- the first *trading* day of the period is checked.
        """
        if previous is None:
            return True
        if self._frequency is RebalanceFrequency.MONTHLY:
            return (on.year, on.month) != (previous.year, previous.month)
        if self._frequency is RebalanceFrequency.QUARTERLY:
            return (on.year, (on.month - 1) // 3) != (previous.year, (previous.month - 1) // 3)
        # RebalanceFrequency.SEMI_ANNUAL
        return (on.year, (on.month - 1) // 6) != (previous.year, (previous.month - 1) // 6)

    # -------------------------------------------------------------- ranking

    def _score(self, frame: pd.DataFrame, on: date) -> float | None:
        """Step 3 of the chapter, for one symbol. ``None`` when unscoreable."""
        index = pd.DatetimeIndex(frame.index)
        usable = frame[index <= pd.Timestamp(on, tz=index.tz)]
        if len(usable) < self._min_history:
            return None

        closes = usable["close"]
        # The window ends `skip` sessions before the decision, so the most
        # recent month is excluded in the 12-1 variant.
        end_position = len(closes) - 1 - self._skip
        start_position = end_position - (self._lookback - self._skip)
        if start_position < 0:
            return None

        start = float(closes.iloc[start_position])
        end = float(closes.iloc[end_position])
        if start <= 0 or end <= 0:
            return None

        # Return = [ending value / starting value] - 1
        total_return = (end / start) - 1.0

        if self._method is not RankingMethod.VOL_ADJUSTED:
            return total_return

        recent = closes.iloc[-(self._vol_window + 1) :]
        volatility = float(recent.pct_change().dropna().std())
        if not volatility > 0:
            return None
        return total_return / volatility

    def rank(self, panel: dict[str, pd.DataFrame], *, on: date) -> list[tuple[str, float]]:
        """Every scoreable symbol, best first. Step 4 of the chapter."""
        if len(panel) < MIN_TRACKING_UNIVERSE and not self._warned_thin_universe:
            # Once per instance: the same warning on every rebalance date for
            # eleven years is noise that trains everyone to ignore it.
            self._warned_thin_universe = True
            log.warning(
                "tracking_universe_below_varsity_guidance",
                tracking_universe=len(panel),
                recommended_minimum=MIN_TRACKING_UNIVERSE,
                portfolio_size=self._size,
                detail=(
                    "Varsity M10 Ch16 suggests 150-200 names for a 12-15 stock "
                    "portfolio; a top-N slice of a smaller universe is a much less "
                    "selective screen"
                ),
            )

        scored: list[tuple[str, float]] = []
        for symbol, frame in panel.items():
            score = self._score(frame, on)
            if score is not None:
                scored.append((symbol, score))
        # Symbol as the tiebreaker so an equal score is resolved
        # deterministically rather than by dict ordering.
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored

    # -------------------------------------------------------------- weights

    def target_weights(self, panel: dict[str, pd.DataFrame], *, on: date) -> dict[str, Decimal]:
        """The top N, equally weighted. Step 5 of the chapter.

        Momentum is a long-only ranking here: a name whose trailing return is
        negative is not a momentum stock, and buying the "best of a bad lot"
        because the ranking still has slots is how this strategy bleeds in a
        downtrend. The chapter's worked example ranks a universe where most
        returns are negative and takes the top 5 regardless; that is the single
        place this implementation deviates, and it deviates toward holding cash.
        """
        ranked = self.rank(panel, on=on)
        chosen = [symbol for symbol, score in ranked[: self._size] if score > 0]
        if not chosen:
            return {}

        # Equal weight, with the remainder given to the highest-ranked name so
        # the weights sum to exactly 1 rather than to 0.99999.
        weight = (Decimal(1) / Decimal(len(chosen))).quantize(Decimal("0.000001"))
        weights = dict.fromkeys(chosen, weight)
        weights[chosen[0]] += Decimal(1) - weight * len(chosen)
        return weights
