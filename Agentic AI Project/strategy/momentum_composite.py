"""Cross-sectional momentum composite.

**Where this came from.** A reverse-engineering exercise on a retail screener
(see the audit in ``Superinvesting/``) established that its five "superinvestor
AI personas" are, arithmetically, linear blends of four sub-scores --
fundamental, business, valuation, technical -- fitted at R² 0.32 to 0.57. The
personas are a presentation layer over one composite factor model. That is the
only part worth taking, and it is a public, decades-old idea: rank a universe
on several factors, blend the ranks, hold the top.

Nothing here uses that screener's data. It could not, honestly: its scores are
a single snapshot with no history, so backtesting them would mean scoring the
past with numbers computed today. This strategy computes every input from our
own bhavcopy archive, bar by bar.

**Why momentum and nothing else.** Of that model's four pillars, three
(fundamental, business, valuation) need earnings, returns on capital and
balance-sheet history. The curated store is OHLCV. Point-in-time fundamentals
we do not have cannot be faked from today's values without look-ahead of the
worst kind, so those pillars are simply absent rather than approximated. What
survives is the technical pillar -- and notably, in that fitted model the only
persona with a materially positive technical weight was CANSLIM (+0.36); the
rest sat near zero or slightly negative.

**The composite**, each term percentile-ranked across the universe on the day:

- ``momentum_12_1`` -- twelve-month return, skipping the last month. The
  canonical factor; the skip avoids folding short-term reversal into it.
- ``momentum_6_1`` -- the same over six months, so a name has to be trending
  on two horizons rather than carrying one lucky quarter.
- ``distance_from_high_252`` -- proximity to the 52-week high. A stock can sit
  above its 200-day mean while 30% off its high; that is a broken uptrend.
- ``realised_vol_60`` -- **inverted**. Raw momentum systematically selects the
  most volatile names; dividing by volatility is what turns "biggest move"
  into "most consistent move".

Ranks, not values: a percentile is scale-free, so a 300-rupee stock and a
3,000-rupee one compare honestly, and one outlier cannot dominate the blend.

**Measured result, and it is not good enough to trade.** Over 2015-2026 on
the NIFTY 50, walk-forward with 24-month train / 6-month test windows and the
stop/target grid fitted *inside each training window* (no selection leakage):

===========================  ==========  ==================
metric                       baseline    momentum composite
===========================  ==========  ==================
out-of-sample return           +62.77%             +170.00%
walk-forward efficiency           1.04                 0.49
consistency                        65%                  50%
out-of-sample Sharpe             -0.06                 0.32
out-of-sample max drawdown      -24.79%              -36.96%
===========================  ==========  ==================

Read that carefully before being encouraged by the return column. Efficiency
below 0.5 is the project's own threshold for "the in-sample result did not
generalise". Consistency of 50% is a coin flip. The training windows picked
nine different stop/target pairs across twenty-one windows, which is what a
parameter that is fitting noise looks like. And buy-and-hold on the index
returned +210% over the same period with a 25% shallower drawdown.

The one honest positive: Sharpe 0.32 is the first figure in this repository
that clears the 6.5% risk-free rate. That is a reason to keep the factor
machinery, not a reason to trade this.

**The breadth regime gate** (``min_breadth``) is the one improvement that
survived honest testing. Over 2023-2026 it lifts the strategy's return and
trims its drawdown in every capital and parameter cell -- but only from bad to
less bad: the strategy still trails both the baseline crossover and the index
over that window. It ships on because it is the literature-standard fix for the
momentum crash and it is not fitted to this data; the shipped 0.40 threshold is
a neutral "more of the market up than down", not the best-performing value.

**Small accounts are punished.** At Rs 1 lakh the fixed components of Indian
charges -- the DP debit, stamp-duty and STT minimums -- consume 6-13% of
capital versus 8-10% at Rs 10 lakh, and integer share sizing quantises
positions so coarsely that the risk model can barely express itself. This
strategy needs several lakh to trade as designed; below that, friction is the
strategy.

**A hard trend gate sits in front of the ranking.** Being the best of a
falling universe is not a buy signal, and a purely relative screen has no way
to say "nothing qualifies today". Price must be above its 200-day mean and
within ``max_drawdown_from_high`` of the 52-week high, or the name is not
eligible at any rank.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal

import pandas as pd

from trading_agent.domain.enums import Action, ProductType
from trading_agent.domain.signal import TradeSignal, build_idempotency_key

__all__ = ["FACTOR_WEIGHTS", "MomentumComposite"]

#: Factor -> (weight, higher_is_better). Weights sum to 1.0 so the composite
#: reads as a percentile itself. Deliberately round numbers: this model has not
#: been fitted to this universe, and a weight of 0.37 would imply a precision
#: that does not exist. Tuning them on the same decade used to evaluate them is
#: how a backtest becomes a curve fit.
FACTOR_WEIGHTS: dict[str, tuple[float, bool]] = {
    "momentum_12_1": (0.40, True),
    "momentum_6_1": (0.25, True),
    "distance_from_high_252": (0.20, True),
    "realised_vol_60": (0.15, False),
}

#: Columns the strategy cannot proceed without.
_REQUIRED = (*FACTOR_WEIGHTS, "sma_200", "atr_14", "close")


def _as_float(value: object) -> float:
    """Narrow a pandas scalar to a float for the type checker."""
    return float(value)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class MomentumComposite:
    """Long the strongest trends in the universe, on a blended factor rank."""

    #: How many names may be signalled on one session. The engine's own
    #: position cap still applies; this bounds the *proposals*, so a single
    #: session cannot flood the risk layer with forty candidates.
    top_n: int = 5

    #: A name must reach this composite percentile to be signalled at all.
    #: Without a floor, a universe in which nothing is working still produces
    #: five "best" names every day.
    min_percentile: float = 0.80

    #: Hard gate: price this far below the 52-week high is disqualified
    #: regardless of rank.
    max_drawdown_from_high: float = 0.25

    #: Stop and target in ATRs, matching the baseline so the two strategies
    #: differ in *selection* only and the comparison isolates that.
    stop_atr_multiple: Decimal = Decimal("2.0")
    target_atr_multiple: Decimal = Decimal("4.0")

    #: Minimum names with usable factors before any signal is produced. A
    #: percentile over four stocks is not a percentile.
    min_universe: int = 10

    #: Market-regime gate: signal nothing unless at least this fraction of the
    #: whole universe is above its 200-day mean. Momentum's defining weakness
    #: is the crash -- it works for years, then gives back a decade in one
    #: bear market, because a relative screen keeps buying "the strongest
    #: falling stock" all the way down. Breadth is a regime proxy computed
    #: from the panel itself (no index feed needed) and leads the index, since
    #: the average stock rolls over before the cap-weighted average does.
    #: 0.0 disables the gate, which is how the pre-regime behaviour is
    #: reproduced for the A/B comparison.
    min_breadth: float = 0.40

    weights: dict[str, tuple[float, bool]] = field(default_factory=lambda: dict(FACTOR_WEIGHTS))

    @property
    def name(self) -> str:
        return f"momentum_composite_top{self.top_n}"

    def generate(self, panel: dict[str, pd.DataFrame], *, on: date) -> list[TradeSignal]:
        """Signals for session ``on``, using only bars up to and including it."""
        session = on if isinstance(on, pd.Timestamp) else pd.Timestamp(on)

        # Regime gate first: in a broad downtrend, the best relative momentum
        # is still a falling knife, and holding through it is the crash this
        # filter exists to sit out. Computed over the whole universe, before
        # the per-name trend gate narrows it.
        if self.min_breadth > 0.0 and self._breadth(panel, session) < self.min_breadth:
            return []

        rows = self._eligible(panel, session)
        if len(rows) < self.min_universe:
            # Too thin to rank. Silence is the correct output, not a guess.
            return []

        frame = pd.DataFrame.from_records(rows).set_index("symbol")
        frame["composite"] = self._composite(frame)

        qualified = frame[frame["composite"] >= self.min_percentile]
        chosen = qualified.sort_values("composite", ascending=False).head(self.top_n)

        signals = [
            signal
            for symbol, row in chosen.iterrows()
            if (signal := self._signal(str(symbol), row, session)) is not None
        ]
        # Deterministic order by conviction, breaking ties by symbol: prioritises highest
        # composite momentum score when capacity/cash is constrained.
        return sorted(signals, key=lambda s: (-s.confidence, s.symbol))

    # ------------------------------------------------------------- internals

    def _breadth(self, panel: dict[str, pd.DataFrame], session: pd.Timestamp) -> float:
        """Fraction of the universe trading above its 200-day mean today.

        Over every name with a bar this session, not only the eligible ones:
        breadth is a measure of the whole market's health, so excluding the
        weak names -- the ones that define a weak regime -- would defeat it.
        A session with no readable bars returns 0.0, which the caller reads as
        "no regime, no trades", the safe default.
        """
        above = 0
        counted = 0
        for frame in panel.values():
            if session not in frame.index:
                continue
            position = frame.index.get_loc(session)
            if not isinstance(position, int):
                continue
            bar = frame.iloc[position]
            close, sma_200 = bar.get("close"), bar.get("sma_200")
            if pd.isna(close) or pd.isna(sma_200):
                continue
            counted += 1
            if _as_float(close) > _as_float(sma_200):
                above += 1
        return above / counted if counted else 0.0

    def _eligible(
        self, panel: dict[str, pd.DataFrame], session: pd.Timestamp
    ) -> list[dict[str, object]]:
        """Factor values for every name that passes the trend gate today."""
        rows: list[dict[str, object]] = []

        for symbol, frame in panel.items():
            if session not in frame.index:
                continue
            position = frame.index.get_loc(session)
            if not isinstance(position, int):
                continue
            bar = frame.iloc[position]
            if any(pd.isna(bar.get(column)) for column in _REQUIRED):
                continue

            close = _as_float(bar["close"])
            atr = _as_float(bar["atr_14"])
            if close <= 0 or atr <= 0:
                continue

            # -- the trend gate, before any ranking ------------------------
            if close <= _as_float(bar["sma_200"]):
                continue
            if _as_float(bar["distance_from_high_252"]) < -self.max_drawdown_from_high:
                continue

            row: dict[str, object] = {"symbol": symbol, "close": close, "atr": atr}
            for column in self.weights:
                row[column] = _as_float(bar[column])
            rows.append(row)
        return rows

    def _composite(self, frame: pd.DataFrame) -> pd.Series:
        """Weighted blend of percentile ranks, in [0, 1]."""
        total = pd.Series(0.0, index=frame.index)
        for column, (weight, higher_is_better) in self.weights.items():
            # `pct=True` gives the percentile directly; ascending=False flips
            # the factors where a smaller number is the better one.
            ranked = frame[column].rank(pct=True, ascending=higher_is_better)
            total = total + ranked * weight
        return total

    def _signal(self, symbol: str, row: pd.Series, session: pd.Timestamp) -> TradeSignal | None:
        entry = Decimal(str(round(_as_float(row["close"]), 2)))
        atr = Decimal(str(round(_as_float(row["atr"]), 2)))
        stop = entry - atr * self.stop_atr_multiple
        if stop <= 0:
            return None
        target = entry + atr * self.target_atr_multiple

        return TradeSignal(
            symbol=symbol,
            action=Action.BUY,
            product=ProductType.CNC,
            entry=entry,
            stop_loss=stop,
            target=target,
            generated_at=(
                session.to_pydatetime()
                if isinstance(session, pd.Timestamp)
                else datetime.combine(session, datetime.min.time())
            ),
            session_date=session.date(),
            strategy=self.name,
            idempotency_key=build_idempotency_key(
                strategy=self.name, symbol=symbol, session_date=session.date()
            ),
            # Confidence is the composite percentile, which is a genuine
            # cross-sectional rank rather than a number invented to look
            # like one.
            confidence=round(_as_float(row["composite"]), 3),
            rationale=(
                f"composite {_as_float(row['composite']):.2f} "
                f"(12-1 mom {_as_float(row['momentum_12_1']):+.1%}, "
                f"6-1 {_as_float(row['momentum_6_1']):+.1%}, "
                f"{_as_float(row['distance_from_high_252']):+.1%} from 52w high, "
                f"vol {_as_float(row['realised_vol_60']):.1%}); "
                f"ATR({atr}) stop {stop}, target {target}"
            ),
        )
