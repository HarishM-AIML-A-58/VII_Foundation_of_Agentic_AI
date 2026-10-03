"""Trading the formulaic-alpha composite.

The signal is a blend of published alphas from :mod:`trading_agent.features.alphas`,
selected by the information-coefficient screen and combined by averaging their
cross-sectional ranks. What makes this different from
:class:`~trading_agent.strategy.momentum_composite.MomentumComposite` is not the
mechanism -- both rank the universe and buy the top -- but the provenance of the
inputs: these formulas were published in 2016 and measured here without a single
parameter fitted to this data.

**How the four were chosen.** All 29 implemented alphas were scored by IC over
2015-2022 only, with 2023 onward held back untouched. Twenty-five cleared |t|>2.
The top eight were then checked for pairwise correlation, which found three
clusters -- alpha006/014/040 correlate at 0.72-0.87 and are one bet wearing
three names. One representative was taken from each cluster:

============  ==========================================  ===============
alpha         formula                                      cluster
============  ==========================================  ===============
alpha023      high vs its own 20-day mean                  independent
alpha044      -corr(high, rank(volume), 5)                 volume/price
alpha006      -corr(open, volume, 10)                      volume/price
alpha038      -rank(ts_rank(close,10)) * rank(close/open)  reversal
============  ==========================================  ===============

**Measured, in-sample then out.** Horizon 21 sessions:

=====================  ===========  ===========
signal                 IS mean IC   OOS mean IC
=====================  ===========  ===========
alpha023                   +0.0225      +0.0124
alpha044                   +0.0257      +0.0096
alpha006                   +0.0267      +0.0079
alpha038                   +0.0194      +0.0074
**composite (equal)**    **+0.0368**  **+0.0143**
=====================  ===========  ===========

Two things to take from that table. The composite beats every component in
*both* periods, which is diversification behaving as advertised. And every
component decays by roughly two thirds out of sample -- the expected fate of a
published alpha a decade after publication, on a market it was not written for.
The composite survives at t=2.47; the individual alphas mostly do not.

**This is a weak signal and should be treated as one.** An IC of 0.014 means the
ranking is right slightly more often than not. It is real -- eight of eight
components held their sign on genuinely unseen data -- but it is not a licence
to concentrate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal

import pandas as pd

from trading_agent.domain.enums import Action, ProductType
from trading_agent.domain.signal import TradeSignal, build_idempotency_key

__all__ = ["SELECTED_ALPHAS", "AlphaComposite", "attach_alpha_composite"]

#: The decorrelated survivors of the IC screen. See the module docstring for
#: how they were chosen and what they scored.
SELECTED_ALPHAS = ("alpha023", "alpha044", "alpha006", "alpha038")

#: Column the strategy reads. Written by :func:`attach_alpha_composite`.
COMPOSITE_COLUMN = "alpha_composite"

_REQUIRED = (COMPOSITE_COLUMN, "sma_200", "atr_14", "close")


def attach_alpha_composite(
    bars: dict[str, pd.DataFrame], *, names: tuple[str, ...] = SELECTED_ALPHAS
) -> dict[str, pd.DataFrame]:
    """Compute the alpha composite over the panel and attach it per symbol.

    Computed **once over the whole panel** rather than per session. The alphas
    are cross-sectional, so evaluating them inside a per-session loop would be
    quadratic in sessions and identical in result -- every operator is
    backward-looking, so a value at *t* does not depend on what comes after it.

    Ranks are averaged rather than raw values: the alphas have wildly different
    numeric ranges (a correlation lives in [-1,1], a price delta does not), and
    averaging raw values would hand the blend to whichever has the widest scale.
    """
    from trading_agent.features.alphas import Panel, compute_alphas

    panel = Panel.from_bars(bars)
    computed = compute_alphas(panel, list(names))
    if not computed:
        raise ValueError(f"no alpha in {names} could be computed over this panel")

    ranked = [values.rank(axis=1, pct=True) for values in computed.values()]
    total = ranked[0]
    for frame in ranked[1:]:
        total = total + frame
    composite: pd.DataFrame = total / len(ranked)

    attached: dict[str, pd.DataFrame] = {}
    for symbol, frame in bars.items():
        if symbol not in composite.columns:
            continue
        enriched = frame.copy()
        enriched[COMPOSITE_COLUMN] = composite[symbol].reindex(frame.index)
        attached[symbol] = enriched
    return attached


@dataclass(frozen=True, slots=True)
class AlphaComposite:
    """Long the top of the formulaic-alpha composite."""

    top_n: int = 5
    #: Composite percentile a name must reach. The composite is itself an
    #: average of percentiles, so this reads directly as "top quintile".
    min_percentile: float = 0.80
    #: Price must be above its 200-day mean. The alphas are direction-agnostic
    #: rankers; this keeps a long-only book out of names in a downtrend.
    require_uptrend: bool = True
    #: Signal nothing unless this fraction of the universe is above its 200-day
    #: mean. Same regime gate as the momentum strategy, same reasoning.
    min_breadth: float = 0.40
    min_universe: int = 10

    stop_atr_multiple: Decimal = Decimal("2.0")
    target_atr_multiple: Decimal = Decimal("4.0")

    weights: dict[str, float] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"alpha_composite_top{self.top_n}"

    def generate(self, panel: dict[str, pd.DataFrame], *, on: date) -> list[TradeSignal]:
        """Signals for session ``on``, from the precomputed composite column."""
        session = on if isinstance(on, pd.Timestamp) else pd.Timestamp(on)

        if self.min_breadth > 0.0 and self._breadth(panel, session) < self.min_breadth:
            return []

        rows = self._eligible(panel, session)
        if len(rows) < self.min_universe:
            return []

        frame = pd.DataFrame.from_records(rows).set_index("symbol")
        # Re-rank within the eligible set: the stored composite was ranked over
        # the whole universe, and the trend gate has since removed part of it.
        frame["score"] = frame["composite"].rank(pct=True)

        chosen = (
            frame[frame["score"] >= self.min_percentile]
            .sort_values("score", ascending=False)
            .head(self.top_n)
        )

        signals = [
            signal
            for symbol, row in chosen.iterrows()
            if (signal := self._signal(str(symbol), row, session)) is not None
        ]
        # Deterministic order by conviction, breaking ties by symbol: prioritises highest
        # alpha score when capacity or cash is constrained.
        return sorted(signals, key=lambda s: (-s.confidence, s.symbol))

    # ------------------------------------------------------------- internals

    def _breadth(self, panel: dict[str, pd.DataFrame], session: pd.Timestamp) -> float:
        above = counted = 0
        for frame in panel.values():
            if session not in frame.index:
                continue
            position = frame.index.get_loc(session)
            if not isinstance(position, int):
                continue
            bar = frame.iloc[position]
            close, mean = bar.get("close"), bar.get("sma_200")
            if pd.isna(close) or pd.isna(mean):
                continue
            counted += 1
            if float(close) > float(mean):
                above += 1
        return above / counted if counted else 0.0

    def _eligible(
        self, panel: dict[str, pd.DataFrame], session: pd.Timestamp
    ) -> list[dict[str, object]]:
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

            close, atr = float(bar["close"]), float(bar["atr_14"])
            if close <= 0 or atr <= 0:
                continue
            if self.require_uptrend and close <= float(bar["sma_200"]):
                continue

            rows.append(
                {
                    "symbol": symbol,
                    "close": close,
                    "atr": atr,
                    "composite": float(bar[COMPOSITE_COLUMN]),
                }
            )
        return rows

    def _signal(self, symbol: str, row: pd.Series, session: pd.Timestamp) -> TradeSignal | None:
        entry = Decimal(str(round(float(row["close"]), 2)))
        atr = Decimal(str(round(float(row["atr"]), 2)))
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
            confidence=round(float(row["score"]), 3),
            rationale=(
                f"alpha composite {float(row['composite']):.3f} "
                f"(rank {float(row['score']):.2f} of eligible); "
                f"ATR({atr}) stop {stop}, target {target}"
            ),
        )
