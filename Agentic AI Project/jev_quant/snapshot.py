"""
Point-in-time daily market snapshot for Jev scoring.

Every field is computed from bars dated on or before `as_of`, so the same code
serves live scans and historical evaluation without lookahead. No field is a
placeholder: daily bars carry no bid/ask, so spread is not reported at all and
liquidity is described by traded value instead.
"""
from __future__ import annotations

from datetime import date, datetime, time as dtime
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

IST = ZoneInfo("Asia/Kolkata")
NSE_CLOSE = dtime(15, 30)

#: Trade geometry shared by sizing and evaluation labels.
STOP_ATR = 1.5
TARGET_ATR = 3.0

#: Bars needed before a snapshot is trusted (SMA-50 + slope + ATR median window).
MIN_BARS = 120

ETF_MARKERS = ("BEES", "ETF")
ETF_SYMBOLS = {"SPY", "QQQ", "IWM", "GLD"}


def market_index_for(symbol: str) -> str:
    """Benchmark whose regime is reported alongside the symbol."""
    return "^NSEI" if symbol.upper().endswith((".NS", ".BO")) else "^GSPC"


def is_etf(symbol: str) -> bool:
    s = symbol.upper()
    return s in ETF_SYMBOLS or any(m in s for m in ETF_MARKERS)


def clean_bars(df: pd.DataFrame, *, now: Optional[datetime] = None) -> pd.DataFrame:
    """Drop bars that are not real completed sessions.

    Yahoo emits a flat zero-volume bar for exchange holidays on some NSE
    instruments (e.g. NIFTYBEES on 2026-10-02), which would read as a 0% day
    with no volume. A bar for today before the NSE close is still forming.
    """
    out = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out[out["Volume"] > 0]
    now = now or datetime.now(IST)
    if len(out) and out.index[-1].date() == now.date() and now.time() < NSE_CLOSE:
        out = out.iloc[:-1]
    return out


def _wilder(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def _streak(changes: pd.Series, sign: int) -> int:
    count = 0
    for v in reversed(changes.to_numpy()):
        if np.sign(v) == sign:
            count += 1
        else:
            break
    return count


def _pct(a: float, b: float) -> float:
    return round((a - b) / b * 100.0, 2) if b else 0.0


def build_snapshot(
    bars: pd.DataFrame,
    symbol: str,
    as_of: Optional[date] = None,
    *,
    market_bars: Optional[pd.DataFrame] = None,
    anonymize: bool = False,
) -> Dict[str, Any]:
    """Numeric snapshot of `symbol` at the close of `as_of` (default: last bar).

    `bars` must already be cleaned. `market_bars` (the benchmark from
    `market_index_for`) adds a `market` section, cut at the same date. With
    `anonymize`, the symbol and dates are withheld so a model cannot recall
    what happened next (used in backtests). Returns {} when there is not
    enough history.
    """
    df = bars if as_of is None else bars[bars.index.date <= as_of]
    if len(df) < MIN_BARS:
        return {}
    market = None
    if market_bars is not None:
        cut = df.index[-1].date()
        market = build_snapshot(market_bars[market_bars.index.date <= cut], "market")

    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]
    c = float(close.iloc[-1])
    chg = close.diff()

    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr_series = _wilder(tr, 14)
    atr = float(atr_series.iloc[-1])
    atr_pct_series = atr_series / close * 100.0
    atr_pct = float(atr_pct_series.iloc[-1])
    atr_pct_median = float(atr_pct_series.tail(100).median())

    gain = _wilder(chg.clip(lower=0), 14)
    loss = _wilder(-chg.clip(upper=0), 14)
    rsi_series = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    rsi = float(rsi_series.fillna(100.0).iloc[-1])

    sma20, sma50 = close.rolling(20).mean(), close.rolling(50).mean()
    hi_3m, lo_3m = float(high.tail(63).max()), float(low.tail(63).min())
    hi_1y = float(high.tail(252).max())

    vol_20 = float(vol.iloc[-21:-1].mean())  # excludes today
    last10 = df.tail(10)
    up10 = last10["Close"].diff() > 0
    up_vol, down_vol = float(last10["Volume"][up10].sum()), float(last10["Volume"][~up10].sum())
    rng = float(high.iloc[-1] - low.iloc[-1])

    snap: Dict[str, Any] = {
        "symbol": symbol,
        "as_of": df.index[-1].date().isoformat(),
        "asset_class": "index_or_commodity_etf" if is_etf(symbol) else "equity",
        "close": round(c, 2),
        "returns_pct": {
            "1d": _pct(c, float(close.iloc[-2])),
            "5d": _pct(c, float(close.iloc[-6])),
            "20d": _pct(c, float(close.iloc[-21])),
            "60d": _pct(c, float(close.iloc[-61])),
        },
        "trend": {
            "price_vs_sma20_pct": _pct(c, float(sma20.iloc[-1])),
            "price_vs_sma50_pct": _pct(c, float(sma50.iloc[-1])),
            "sma20_slope_5d_pct": _pct(float(sma20.iloc[-1]), float(sma20.iloc[-6])),
            "sma50_slope_10d_pct": _pct(float(sma50.iloc[-1]), float(sma50.iloc[-11])),
            "sma20_above_sma50": bool(sma20.iloc[-1] > sma50.iloc[-1]),
            "up_days_last_10": int(up10.iloc[1:].sum()),
        },
        "momentum": {
            "rsi_14": round(rsi, 1),
            "rsi_14_5d_ago": round(float(rsi_series.fillna(100.0).iloc[-6]), 1),
            "consecutive_down_days": _streak(chg, -1),
            "consecutive_up_days": _streak(chg, 1),
        },
        "levels": {
            "discount_from_3m_high_pct": _pct(c, hi_3m),
            "discount_from_52w_high_pct": _pct(c, hi_1y),
            "support_3m_low": round(lo_3m, 2),
            "distance_to_support_atr": round((c - lo_3m) / atr, 2) if atr else None,
            "upside_to_sma50_pct": _pct(float(sma50.iloc[-1]), c),
        },
        "volume": {
            "volume_vs_20d_avg": round(float(vol.iloc[-1]) / vol_20, 2) if vol_20 else None,
            "volume_5d_vs_20d_avg": round(float(vol.tail(5).mean()) / vol_20, 2) if vol_20 else None,
            "up_vs_down_volume_10d": round(up_vol / down_vol, 2) if down_vol else None,
            "close_location_in_day_range": round((c - float(low.iloc[-1])) / rng, 2) if rng else 0.5,
            "avg_traded_value_cr": round(float((close * vol).tail(20).mean()) / 1e7, 1),
        },
        "volatility": {
            "atr_14": round(atr, 2),
            "atr_pct": round(atr_pct, 2),
            "atr_pct_vs_100d_median": round(atr_pct / atr_pct_median, 2) if atr_pct_median else None,
        },
        "plan": {
            "stop": round(c - STOP_ATR * atr, 2),
            "target": round(c + TARGET_ATR * atr, 2),
            "stop_distance_pct": round(STOP_ATR * atr_pct, 2),
            "reward_risk": round(TARGET_ATR / STOP_ATR, 2),
        },
    }
    if market:
        snap["market"] = {
            "atr_pct_vs_100d_median": market["volatility"]["atr_pct_vs_100d_median"],
            "returns_20d_pct": market["returns_pct"]["20d"],
            "returns_60d_pct": market["returns_pct"]["60d"],
            "price_vs_sma50_pct": market["trend"]["price_vs_sma50_pct"],
            "sma50_slope_10d_pct": market["trend"]["sma50_slope_10d_pct"],
        }
    if anonymize:
        snap.pop("symbol")
        snap.pop("as_of")
    return snap


def fetch_bars(symbol: str, period: str = "2y") -> pd.DataFrame:
    import yfinance as yf

    return clean_bars(yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=True))
