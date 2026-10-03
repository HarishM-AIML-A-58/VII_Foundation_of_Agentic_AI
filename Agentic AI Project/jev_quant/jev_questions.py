"""
Versioned Jev question sets, gates and the GO / NO-GO decision.

A version pins question wording and gates together, because a gate is only
meaningful for the wording it was tuned on. `scripts/jev_calibrate.py`
measures each version against forward outcomes on NSE history.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

from typesafe_sdk import Noul, NoulCriteria

MOMENTUM, ETF_DIP = "momentum", "etf_dip"


def _q(instructions: str, true: str, false: str) -> Noul:
    return Noul(instructions=instructions, criteria=NoulCriteria(true=true, false=false))


# v1: the original scanner questions, pointed at the real snapshot fields.
_V1 = {
    MOMENTUM: {
        "regime_trend": _q(
            "returns_pct and trend show an active directional trend rather than mean reversion.",
            "Returns keep one consistent sign and the trend fields are visibly away from zero.",
            "Returns alternate sign, choppy sideways movement, or trend fields are near zero."),
        "direction_long": _q(
            "The snapshot supports a momentum long position rather than a short position.",
            "Price is above its 5-day level and recent returns are positive.",
            "Price is below its 5-day level or recent returns are negative."),
        "buying_pressure_real": _q(
            "volume and price movement confirm genuine institutional buying pressure.",
            "Volume is significantly above average with strong positive price closes.",
            "Volume is average or declining, or buying pressure appears exhausted."),
        "setup_quality": _q(
            "Volatility, momentum, and volume combine to provide a high-probability clean setup.",
            "Trend momentum aligns with volume surge and volatility is within normal trading bounds.",
            "Setup is noisy, overextended, low-liquidity, or conflicting."),
        "risk_state_favorable": _q(
            "volatility and liquidity leave adequate room for a favorable risk-reward with a tight stop.",
            "ATR is ordinary for this stock and allows a structured 1.5x ATR stop distance.",
            "Volatility is excessively wide or erratic, risking whipsaw."),
    },
    ETF_DIP: {
        "oversold_exhaustion": _q(
            "rsi_14 and consecutive_down_days indicate seller exhaustion in this ETF.",
            "RSI is below 30 and there have been multiple consecutive down sessions indicating panic exhaustion.",
            "RSI is above 35 or price action shows orderly continuation rather than panic exhaustion."),
        "mean_reversion_accumulation": _q(
            "The discount from the recent high near support is favorable for staged accumulation.",
            "The ETF is down 5% to 10% from its 3-month high and within 1 ATR of the support zone.",
            "The discount is trivial (<3%) or support has broken into a prolonged structural downtrend."),
        "asymmetric_risk_reward": _q(
            "Risk-reward favors an accumulation swing entry given ATR relative to recovery upside.",
            "The stop distance is tight relative to the recovery toward the 50-day average.",
            "Downside risk is undefined or potential reward does not compensate for current volatility."),
    },
}

_V1_GATES = {
    MOMENTUM: {"regime_trend": 0.70, "direction_long": 0.65, "buying_pressure_real": 0.60,
               "setup_quality": 0.75, "risk_state_favorable": 0.75},
    ETF_DIP: {"oversold_exhaustion": 0.65, "mean_reversion_accumulation": 0.70, "asymmetric_risk_reward": 0.55},
}


# v2: rebuilt from the v1 calibration (output/jev_calib, 1200 NSE samples).
# v1 asked for volume-surge breakouts and panic exhaustion; on a 1.5/3 ATR plan
# both were neutral or negative. What held up in both date splits was calm
# volatility and orderly entries inside an established uptrend, so each
# question now targets one of those conditions, in snapshot field names.
_V2 = {
    MOMENTUM: {
        "trend_intact": _q(
            "Is this stock in an established, still-rising uptrend? Use trend.price_vs_sma50_pct, "
            "trend.sma20_above_sma50, trend.sma50_slope_10d_pct and returns_pct.60d.",
            "Price is above its 50-day average, the 20-day average is above the 50-day, the 50-day "
            "average is rising, and the 60-day return is positive.",
            "Price is below its 50-day average, or the 50-day average is flat or falling, or the "
            "60-day return is negative."),
        "volatility_calm": _q(
            "Is volatility calm relative to this stock's own recent norm? Use "
            "volatility.atr_pct_vs_100d_median and returns_pct.1d against volatility.atr_pct.",
            "atr_pct_vs_100d_median is at or below about 1.0 and the latest day moved less than one ATR.",
            "atr_pct_vs_100d_median is above about 1.15 (volatility expanding), or the latest day "
            "moved more than one ATR, suggesting a news shock."),
        "orderly_entry": _q(
            "Is this an orderly entry rather than chasing a spike or catching a breakdown? Use "
            "momentum.rsi_14, volume.volume_vs_20d_avg, returns_pct.5d and levels.discount_from_3m_high_pct.",
            "RSI is between about 40 and 65, volume is ordinary (volume_vs_20d_avg below about 1.6), "
            "and price is either a modest pullback or a quiet advance within about 6% of its 3-month high.",
            "RSI is above 70 or below 35, volume is a climax spike (above about 1.8x average), or the "
            "5-day move is a sharp run-up or collapse."),
    },
    ETF_DIP: {
        "uptrend_dip": _q(
            "Is this ETF pulling back inside a longer uptrend? Use returns_pct.60d, "
            "trend.sma50_slope_10d_pct, trend.price_vs_sma20_pct and momentum.rsi_14.",
            "The 60-day return is positive and the 50-day average is rising, while price has dipped "
            "below its 20-day average with RSI between about 35 and 50.",
            "The 60-day return is negative or the 50-day average is falling, or there is no dip "
            "(price above its 20-day average, RSI above 55)."),
        "volatility_calm": _q(
            "Is volatility calm relative to this ETF's own recent norm? Use volatility.atr_pct_vs_100d_median.",
            "atr_pct_vs_100d_median is at or below about 1.0.",
            "atr_pct_vs_100d_median is above about 1.15, volatility is expanding."),
        "support_holding": _q(
            "Is the 3-month support holding with room for the stop? Use levels.distance_to_support_atr "
            "and momentum.consecutive_down_days.",
            "Price is more than about 1.5 ATR above the 3-month low and the selling streak is short "
            "(3 days or fewer).",
            "Price is within about 1 ATR of, or below, the 3-month low, or selling has run 4+ days."),
    },
}

# Placeholder gates until the v2 calibration run sets them.
_V2_GATES = {
    MOMENTUM: {"trend_intact": 0.5, "volatility_calm": 0.5, "orderly_entry": 0.5},
    ETF_DIP: {"uptrend_dip": 0.5, "volatility_calm": 0.5, "support_holding": 0.5},
}


# v3: v2 showed only volatility_calm separating outcomes (AUC ~0.60 in both
# splits); trend and entry-shape questions were noise. Adding the benchmark's
# own volatility regime (snapshot.market) gave the most stable edge found, for
# stocks and ETFs alike, so v3 asks only about the regime plus a shock filter
# for stocks and a pullback condition for ETFs.
_MARKET_CALM = _q(
    "Is the broad market (snapshot.market) in a calm volatility regime? Use "
    "market.atr_pct_vs_100d_median, and market.returns_20d_pct for a crash check.",
    "market.atr_pct_vs_100d_median is at or below about 0.95 and the market has not fallen more "
    "than about 6% over 20 days.",
    "market.atr_pct_vs_100d_median is above about 1.05 (market volatility expanding), or the market "
    "is in a sharp 20-day selloff, or the market section is missing.")
_V3 = {
    MOMENTUM: {
        "market_calm": _MARKET_CALM,
        "stock_calm": _q(
            "Is this stock's volatility calm relative to its own norm, with no shock in the latest "
            "session? Use volatility.atr_pct_vs_100d_median, volatility.atr_pct and returns_pct.1d.",
            "atr_pct_vs_100d_median is at or below about 0.95 and the absolute 1-day return is smaller "
            "than atr_pct.",
            "atr_pct_vs_100d_median is above about 1.1, or the absolute 1-day return exceeds atr_pct "
            "(a gap or news shock)."),
    },
    ETF_DIP: {
        "market_calm": _MARKET_CALM,
        "pullback": _q(
            "Is this ETF pulling back rather than extended? Use trend.price_vs_sma20_pct and returns_pct.5d.",
            "Price is below its 20-day average, or the 5-day return is negative.",
            "Price is above its 20-day average and the 5-day return is positive (extended, no pullback)."),
    },
}

_V3_GATES = {
    MOMENTUM: {"market_calm": 0.5, "stock_calm": 0.5},
    ETF_DIP: {"market_calm": 0.5, "pullback": 0.5},
}


@dataclass(frozen=True)
class Edge:
    """Measured result of this version's GO trades, in R (multiples of the stop)."""

    win_rate: float  # share of GO trades closing with positive R
    payoff: float  # average winning R / average losing R
    trades: int


@dataclass(frozen=True)
class QuestionSet:
    version: str
    questions: Mapping[str, Mapping[str, Noul]]
    gates: Mapping[str, Mapping[str, float]]
    #: Out-of-sample edge per strategy; None until a version has been measured.
    edge: Optional[Mapping[str, Edge]] = None


VERSIONS: Dict[str, QuestionSet] = {
    "v1": QuestionSet("v1", _V1, _V1_GATES),
    "v2": QuestionSet("v2", _V2, _V2_GATES),
    # Edge from every sample not used to write the questions: the main-universe
    # holdout, the 12-symbol validation set and the 5-stock test basket
    # (2026-10-03, jev-1.13.0). Pooled GO edge +0.10R over base, 90% date-
    # bootstrap CI [-0.06, +0.26]: positive but not established. Gates stay at
    # 0.5 because tuning them moved holdout results less than that noise.
    "v3": QuestionSet("v3", _V3, _V3_GATES, {
        MOMENTUM: Edge(win_rate=0.408, payoff=1.76, trades=365),
        ETF_DIP: Edge(win_rate=0.514, payoff=1.39, trades=37),
    }),
}
CURRENT = "v3"

#: Ceiling on capital risked per trade. Quarter-Kelly on the measured edge is
#: 1.8% (stocks) and 4.1% (ETFs, only 37 trades), but GO signals cluster on the
#: same dates (correlated, not independent bets) and the edge's confidence
#: interval includes zero, so the cap is what actually binds.
MAX_RISK_PCT = 1.0
MAX_POSITION_PCT = 25.0


def strategy_for(snapshot: Mapping) -> str:
    return ETF_DIP if snapshot.get("asset_class", "").endswith("etf") else MOMENTUM


def decide(scores: Mapping[str, float], gates: Mapping[str, float]) -> Dict:
    """GO only when every gate clears. Failing gates are named for the log."""
    failed = {k: round(scores.get(k, 0.0), 3) for k, g in gates.items() if scores.get(k, 0.0) < g}
    return {"go": not failed, "failed_gates": failed}


def risk_pct(edge: Optional[Edge]) -> float:
    """Percent of capital to risk: 1/4 Kelly on the measured edge, capped."""
    if edge is None:
        return 0.0
    full = edge.win_rate - (1.0 - edge.win_rate) / edge.payoff
    return round(min(MAX_RISK_PCT, max(0.0, full) * 25.0), 2)


def position_pct(edge: Optional[Edge], stop_distance_pct: float) -> float:
    """Position as percent of capital, so a stop-out loses `risk_pct`."""
    if stop_distance_pct <= 0:
        return 0.0
    return round(min(MAX_POSITION_PCT, risk_pct(edge) / stop_distance_pct * 100.0), 2)
