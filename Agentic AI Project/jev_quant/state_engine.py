"""
Deterministic Market State Engine for the Jev Quantitative Trading System.
Packs the live market into a compact numeric snapshot strictly using information
timestamped before the decision timestamp (eliminating lookahead bias).
"""
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import math


@dataclass(frozen=True)
class Candle:
    timestamp: int  # Epoch ms
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class OrderBookLevel:
    price: float
    size: float


@dataclass(frozen=True)
class MarketSnapshot:
    timestamp: int
    symbol: str
    price: float
    spread_bps: float
    order_book_imbalance: float  # -1.0 to 1.0
    realized_vol_annualized: float
    trend_strength: float  # normalized fast vs slow EMA slope
    order_flow_delta: float  # buy volume minus sell volume


class StateEngine:
    def __init__(self, symbol: str):
        self.symbol = symbol

    def compute_snapshot(
        self,
        candles: List[Candle],
        bids: List[OrderBookLevel],
        asks: List[OrderBookLevel],
        decision_timestamp: int,
    ) -> MarketSnapshot:
        """
        Builds a compact numeric snapshot.
        Enforces strict timestamp validation: any candle at or after decision_timestamp is excluded.
        """
        valid_candles = [c for c in candles if c.timestamp < decision_timestamp]
        if not valid_candles:
            raise ValueError("No historical candles strictly prior to decision timestamp.")

        latest_candle = valid_candles[-1]
        price = latest_candle.close

        # Compute spread and order book imbalance
        if bids and asks:
            best_bid = bids[0].price
            best_ask = asks[0].price
            mid = (best_bid + best_ask) / 2.0
            spread_bps = ((best_ask - best_bid) / mid) * 10000.0 if mid > 0 else 0.0

            bid_vol = sum(b.size for b in bids[:5])
            ask_vol = sum(a.size for a in asks[:5])
            total_depth = bid_vol + ask_vol
            order_book_imbalance = (bid_vol - ask_vol) / total_depth if total_depth > 0 else 0.0
        else:
            spread_bps = 0.0
            order_book_imbalance = 0.0

        # Realized volatility (rolling log returns on close)
        if len(valid_candles) >= 5:
            returns = [
                math.log(valid_candles[i].close / valid_candles[i - 1].close)
                for i in range(1, len(valid_candles))
            ]
            mean_ret = sum(returns) / len(returns)
            variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
            # Annualized 1-minute volatility approx sqrt(252 * 375)
            realized_vol = math.sqrt(variance) * math.sqrt(252 * 375)
        else:
            realized_vol = 0.0

        # Trend strength: simple ratio of current price to 10-period moving average
        if len(valid_candles) >= 10:
            ma10 = sum(c.close for c in valid_candles[-10:]) / 10.0
            trend_strength = (price - ma10) / ma10 if ma10 > 0 else 0.0
        else:
            trend_strength = 0.0

        # Order flow delta (volume proxy: high-low direction)
        flow_delta = 0.0
        for c in valid_candles[-5:]:
            if c.close >= c.open:
                flow_delta += c.volume
            else:
                flow_delta -= c.volume

        return MarketSnapshot(
            timestamp=decision_timestamp,
            symbol=self.symbol,
            price=price,
            spread_bps=spread_bps,
            order_book_imbalance=order_book_imbalance,
            realized_vol_annualized=realized_vol,
            trend_strength=trend_strength,
            order_flow_delta=flow_delta,
        )
