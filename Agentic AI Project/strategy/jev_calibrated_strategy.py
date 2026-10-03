"""JEV calibrated volatility-regime strategy conforming to the Strategy protocol.

The deterministic form of the Jev v3 questions (jev_opus_quant/engine/jev_questions.py):
long when both the market and the stock are in a calm volatility regime and the
latest session was not a shock. In the 2026-10-03 calibration over NSE history
this rule agreed with Jev's own GO calls on 94% of samples, so it serves as the
offline / backtest stand-in where calling Jev per bar is not wanted.

Required feature columns (rows missing any are skipped, never defaulted):
``date``, ``symbol``, ``close``, ``atr_14``, ``atr_pct_vs_100d_median``,
``return_1d_pct``, ``market_atr_pct_vs_100d_median``, ``market_return_20d_pct``.
"""

from __future__ import annotations

from datetime import date, datetime, time
from decimal import Decimal
from zoneinfo import ZoneInfo

import pandas as pd

from trading_agent.domain.enums import Action, Exchange, ProductType, SignalStatus
from trading_agent.domain.signal import TradeSignal, build_idempotency_key
from trading_agent.strategy.protocols import Strategy

REQUIRED_COLUMNS = (
    "date",
    "symbol",
    "close",
    "atr_14",
    "atr_pct_vs_100d_median",
    "return_1d_pct",
    "market_atr_pct_vs_100d_median",
    "market_return_20d_pct",
)

#: Signals are stamped at the NSE close of the session they were computed on.
_NSE_CLOSE = time(15, 30, tzinfo=ZoneInfo("Asia/Kolkata"))


class JevCalibratedStrategy(Strategy):
    """Calm-market, calm-stock entries with a 1.5 ATR stop and 3 ATR target."""

    def __init__(
        self,
        name: str = "jev_calibrated_v1",
        max_stock_atr_ratio: float = 0.95,
        max_market_atr_ratio: float = 0.95,
        min_market_return_20d_pct: float = -6.0,
        atr_stop_multiple: float = 1.5,
        atr_target_multiple: float = 3.0,
    ) -> None:
        self._name = name
        self.max_stock_atr_ratio = max_stock_atr_ratio
        self.max_market_atr_ratio = max_market_atr_ratio
        self.min_market_return_20d_pct = min_market_return_20d_pct
        self.atr_stop_multiple = atr_stop_multiple
        self.atr_target_multiple = atr_target_multiple

    @property
    def name(self) -> str:
        return self._name

    def generate(self, features: pd.DataFrame, *, on: date) -> list[TradeSignal]:
        """Signals for session ``on``, using only rows dated on or before it."""
        if features.empty or any(c not in features.columns for c in REQUIRED_COLUMNS):
            return []

        session = features[pd.to_datetime(features["date"]).dt.date <= on]
        signals: list[TradeSignal] = []

        for sym, df in session.groupby("symbol", sort=True):
            latest = df.sort_values("date").iloc[-1]
            # Only act on a bar from this session: a stale last row means the
            # symbol did not trade, not that yesterday's setup still holds.
            if pd.Timestamp(latest["date"]).date() != on:
                continue
            if latest[list(REQUIRED_COLUMNS)].isna().any():
                continue

            close, atr = float(latest["close"]), float(latest["atr_14"])
            if close <= 0 or atr <= 0:
                continue
            atr_pct = atr / close * 100.0

            stock_calm = (
                float(latest["atr_pct_vs_100d_median"]) <= self.max_stock_atr_ratio
                and abs(float(latest["return_1d_pct"])) < atr_pct
            )
            market_calm = (
                float(latest["market_atr_pct_vs_100d_median"]) <= self.max_market_atr_ratio
                and float(latest["market_return_20d_pct"]) > self.min_market_return_20d_pct
            )
            if not (stock_calm and market_calm):
                continue

            stock_ratio = float(latest["atr_pct_vs_100d_median"])
            market_ratio = float(latest["market_atr_pct_vs_100d_median"])
            entry = Decimal(str(round(close, 2)))
            stop = entry - Decimal(str(round(atr * self.atr_stop_multiple, 2)))
            target = entry + Decimal(str(round(atr * self.atr_target_multiple, 2)))
            if stop <= 0:
                continue

            signals.append(
                TradeSignal(
                    symbol=str(sym),
                    action=Action.BUY,
                    # Held up to ~20 sessions until stop or target: delivery, not intraday.
                    product=ProductType.CNC,
                    entry=entry,
                    stop_loss=stop,
                    target=target,
                    generated_at=datetime.combine(on, _NSE_CLOSE),
                    session_date=on,
                    strategy=self._name,
                    idempotency_key=build_idempotency_key(
                        strategy=self._name, symbol=str(sym), session_date=on
                    ),
                    exchange=Exchange.NSE,
                    confidence=0.5,
                    rationale=(
                        f"calm regime: stock ATR {stock_ratio:.2f}x median, "
                        f"market ATR {market_ratio:.2f}x median; stop {stop}, target {target}"
                    ),
                    status=SignalStatus.GENERATED,
                )
            )

        return signals
