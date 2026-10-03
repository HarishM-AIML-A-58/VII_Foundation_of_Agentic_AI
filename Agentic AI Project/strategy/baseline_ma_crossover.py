"""20/50 moving-average crossover -- the honest benchmark.

Deliberately unsophisticated. Its job is not to make money; it is to answer
the only question worth asking before any agent exists:

    Does the machinery -- data, costs, fills, metrics -- produce a believable
    result, and does a trivial rule beat buying the index after charges?

Every later strategy is measured against this, not against zero. A signal
that cannot beat a 20/50 crossover is not an edge, however sophisticated the
reasoning that produced it.

Stops are ATR-based rather than a fixed percentage, because a fixed percentage
stop is far too tight on a volatile midcap and far too loose on a stable
large-cap, and that mismatch shows up as a stop-out rate that has nothing to
do with the signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal

import pandas as pd

from trading_agent.domain.enums import Action, ProductType
from trading_agent.domain.signal import TradeSignal, build_idempotency_key

__all__ = ["BaselineMaCrossover"]


@dataclass(frozen=True, slots=True)
class BaselineMaCrossover:
    """Long on a fast-over-slow SMA cross; ATR stop and target."""

    fast_window: int = 20
    slow_window: int = 50
    #: Stop distance in ATRs below the entry.
    stop_atr_multiple: Decimal = Decimal("2.0")
    #: Target distance in ATRs. 2.0 stop / 4.0 target gives R:R 2.0, which
    #: clears the 1.8 floor the risk manager will enforce in Phase 4.
    target_atr_multiple: Decimal = Decimal("4.0")

    @property
    def name(self) -> str:
        return f"baseline_ma_{self.fast_window}_{self.slow_window}"

    def generate(self, panel: dict[str, pd.DataFrame], *, on: date) -> list[TradeSignal]:
        """Signals for session ``on``, using only bars up to and including it."""
        session = pd.Timestamp(on) if not isinstance(on, pd.Timestamp) else on
        signals: list[TradeSignal] = []

        for symbol, frame in panel.items():
            if session not in frame.index:
                continue

            position = frame.index.get_loc(session)
            if not isinstance(position, int) or position < 1:
                continue

            bar = frame.iloc[position]
            previous = frame.iloc[position - 1]

            required = ("sma_fast", "sma_slow", "atr_14")
            if any(pd.isna(bar.get(column)) for column in required):
                continue
            if pd.isna(previous.get("sma_fast")) or pd.isna(previous.get("sma_slow")):
                continue

            # A CROSS, not merely "fast above slow": entering on every bar of an
            # established trend produces a different, much-traded strategy whose
            # charges swamp the edge.
            crossed_up = (
                previous["sma_fast"] <= previous["sma_slow"] and bar["sma_fast"] > bar["sma_slow"]
            )
            if not crossed_up:
                continue

            entry = Decimal(str(round(float(bar["close"]), 2)))
            atr = Decimal(str(round(float(bar["atr_14"]), 2)))
            if atr <= 0:
                continue

            stop = entry - atr * self.stop_atr_multiple
            if stop <= 0:
                continue
            target = entry + atr * self.target_atr_multiple

            signals.append(
                TradeSignal(
                    symbol=symbol,
                    action=Action.BUY,
                    product=ProductType.CNC,
                    entry=entry,
                    stop_loss=stop,
                    target=target,
                    generated_at=(
                        session.to_pydatetime()
                        if isinstance(session, pd.Timestamp)
                        else datetime.combine(on, datetime.min.time())
                    ),
                    session_date=session.date(),
                    strategy=self.name,
                    idempotency_key=build_idempotency_key(
                        strategy=self.name, symbol=symbol, session_date=session.date()
                    ),
                    confidence=0.5,
                    rationale=(
                        f"{self.fast_window}/{self.slow_window} SMA crossed up; "
                        f"ATR({atr}) stop at {stop}, target {target}"
                    ),
                )
            )

        # Deterministic order: without it, which signals get funded depends on
        # dict iteration order and the backtest stops being reproducible.
        return sorted(signals, key=lambda s: s.symbol)
