"""Institutional AI Trade Autopsy and Forensic Post-Mortem Engine.

Performs automated forensic investigation of completed trades to categorize failure
mechanisms (thesis failure, execution slippage, macro beta drag, or volatility whipsaw)
and extract structured lessons for semantic vector memory ingestion.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Final

from trading_agent.domain.money import Rupees

__all__ = [
    "AutopsyFailureReason",
    "AutopsyResult",
    "TradeExecutionRecord",
    "conduct_trade_autopsy",
    "format_autopsy_for_memory",
    "query_autopsies",
    "record_autopsy",
]

_WHIPSAW_ATR_MULTIPLIER: Final[float] = 1.25
_PROFIT_THRESHOLD_PCT: Final[float] = 0.0
_MACRO_DOMINANCE_RATIO: Final[float] = 0.70
_EXCESSIVE_SLIPPAGE_BPS_THRESHOLD: Final[float] = 45.0


class AutopsyFailureReason(StrEnum):
    """Forensic classification of trade outcome."""

    WINNING_TRADE = "WINNING_TRADE"
    THESIS_INVALIDATION = "THESIS_INVALIDATION"
    EXECUTION_SLIPPAGE = "EXECUTION_SLIPPAGE"
    MACRO_BETA_DRAG = "MACRO_BETA_DRAG"
    VOLATILITY_WHIPSAW = "VOLATILITY_WHIPSAW"


@dataclass(frozen=True, slots=True)
class TradeExecutionRecord:
    """Historical trade data and price excursions over holding window."""

    symbol: str
    action: str  # BUY or SELL
    entry_price: Rupees
    exit_price: Rupees
    stop_loss: Rupees
    target_price: Rupees
    highest_price: Rupees
    lowest_price: Rupees
    holding_period_days: int
    benchmark_return_pct: float  # e.g., NIFTY 50 return in percent over same horizon
    beta: float = 1.0
    atr_at_entry: Rupees | None = None
    slippage_bps: float = 0.0


@dataclass(frozen=True, slots=True)
class AutopsyResult:
    """Diagnostic forensic autopsy report."""

    symbol: str
    realized_pnl_rupees: Rupees
    return_pct: float
    benchmark_relative_alpha_pct: float
    max_favorable_excursion_pct: float
    max_adverse_excursion_pct: float
    failure_reason: AutopsyFailureReason
    root_cause_analysis: str
    preventative_lesson: str


def conduct_trade_autopsy(record: TradeExecutionRecord) -> AutopsyResult:
    """Diagnose a completed trade and identify forensic root cause.

    Calculates MFE/MAE excursions, benchmark-adjusted alpha, and determines if
    the trade was stopped out due to broad market beta drag, premature stop noise
    (whipsaw), poor execution, or fundamental catalyst failure.
    """
    entry = record.entry_price.value
    exit_p = record.exit_price.value
    high = record.highest_price.value
    low = record.lowest_price.value

    is_buy = record.action.upper() == "BUY"

    # 1. P&L and Percentage Return
    if is_buy:
        pnl_dec = exit_p - entry
        return_pct = float((pnl_dec / entry) * Decimal(100))
        mfe_pct = float(((high - entry) / entry) * Decimal(100))
        mae_pct = float(((entry - low) / entry) * Decimal(100))
    else:
        pnl_dec = entry - exit_p
        return_pct = float((pnl_dec / entry) * Decimal(100))
        mfe_pct = float(((entry - low) / entry) * Decimal(100))
        mae_pct = float(((high - entry) / entry) * Decimal(100))

    realized_pnl = Rupees(pnl_dec)

    # 2. Benchmark-adjusted Alpha: Alpha = R_trade - Beta * R_benchmark
    expected_market_return = record.beta * record.benchmark_return_pct
    alpha_pct = return_pct - expected_market_return

    # 3. Winning trade
    if return_pct > _PROFIT_THRESHOLD_PCT:
        return AutopsyResult(
            symbol=record.symbol,
            realized_pnl_rupees=realized_pnl,
            return_pct=round(return_pct, 2),
            benchmark_relative_alpha_pct=round(alpha_pct, 2),
            max_favorable_excursion_pct=round(mfe_pct, 2),
            max_adverse_excursion_pct=round(mae_pct, 2),
            failure_reason=AutopsyFailureReason.WINNING_TRADE,
            root_cause_analysis=(
                f"Trade closed with positive edge ({return_pct:+.2f}%). Alpha over "
                f"NIFTY: {alpha_pct:+.2f}%."
            ),
            preventative_lesson="Maintain target discipline and positive risk-reward asymmetry.",
        )

    # 4. Forensic Diagnosis for Losing Trade
    # Case A: Excessive Slippage (> 45 bps execution drag)
    if record.slippage_bps >= _EXCESSIVE_SLIPPAGE_BPS_THRESHOLD:
        return AutopsyResult(
            symbol=record.symbol,
            realized_pnl_rupees=realized_pnl,
            return_pct=round(return_pct, 2),
            benchmark_relative_alpha_pct=round(alpha_pct, 2),
            max_favorable_excursion_pct=round(mfe_pct, 2),
            max_adverse_excursion_pct=round(mae_pct, 2),
            failure_reason=AutopsyFailureReason.EXECUTION_SLIPPAGE,
            root_cause_analysis=(
                f"Severe execution slippage ({record.slippage_bps:.1f} bps) eroded expected "
                f"alpha. Fill price deviated substantially from model signal price."
            ),
            preventative_lesson=(
                "Mandate Almgren-Chriss or VWAP child slicing with strict POV participation caps."
            ),
        )

    # Case B: Volatility Whipsaw (Stop was hit, but MAE was within 1.25x ATR of entry)
    if record.atr_at_entry is not None:
        atr_dec = record.atr_at_entry.value
        atr_distance = abs(entry - record.stop_loss.value)
        if atr_distance <= (atr_dec * Decimal(str(_WHIPSAW_ATR_MULTIPLIER))):
            return AutopsyResult(
                symbol=record.symbol,
                realized_pnl_rupees=realized_pnl,
                return_pct=round(return_pct, 2),
                benchmark_relative_alpha_pct=round(alpha_pct, 2),
                max_favorable_excursion_pct=round(mfe_pct, 2),
                max_adverse_excursion_pct=round(mae_pct, 2),
                failure_reason=AutopsyFailureReason.VOLATILITY_WHIPSAW,
                root_cause_analysis=(
                    f"Stop-loss was placed too tight within ordinary ATR noise band "
                    f"({float(atr_distance):.2f} vs ATR {float(atr_dec):.2f}). Stock was "
                    "whipsawed by normal intraday noise."
                ),
                preventative_lesson=(
                    "Enforce minimum stop-loss distance of 1.75x ATR to avoid premature exit."
                ),
            )

    # Case C: Macro Beta Drag (Loss was primarily market shock rather than stock error)
    if expected_market_return < 0.0 and abs(expected_market_return) >= (
        abs(return_pct) * _MACRO_DOMINANCE_RATIO
    ):
        return AutopsyResult(
            symbol=record.symbol,
            realized_pnl_rupees=realized_pnl,
            return_pct=round(return_pct, 2),
            benchmark_relative_alpha_pct=round(alpha_pct, 2),
            max_favorable_excursion_pct=round(mfe_pct, 2),
            max_adverse_excursion_pct=round(mae_pct, 2),
            failure_reason=AutopsyFailureReason.MACRO_BETA_DRAG,
            root_cause_analysis=(
                f"Loss was predominantly driven by systematic market shock (NIFTY move "
                f"{record.benchmark_return_pct:+.2f}% * Beta {record.beta:.2f}) rather than "
                f"idiosyncratic thesis failure."
            ),
            preventative_lesson=(
                "Reduce gross exposure or hedge beta using NIFTY index futures "
                "during high-vol regimes."
            ),
        )

    # Case D: Idiosyncratic Thesis Invalidation
    return AutopsyResult(
        symbol=record.symbol,
        realized_pnl_rupees=realized_pnl,
        return_pct=round(return_pct, 2),
        benchmark_relative_alpha_pct=round(alpha_pct, 2),
        max_favorable_excursion_pct=round(mfe_pct, 2),
        max_adverse_excursion_pct=round(mae_pct, 2),
        failure_reason=AutopsyFailureReason.THESIS_INVALIDATION,
        root_cause_analysis=(
            f"Idiosyncratic thesis failure with negative alpha ({alpha_pct:+.2f}%). "
            "Underlying company catalysts failed to materialize."
        ),
        preventative_lesson=(
            "Verify fundamental valuation multiples and confirm sector "
            "confirmation before re-entry."
        ),
    )


def format_autopsy_for_memory(autopsy: AutopsyResult) -> str:
    """Format forensic autopsy result for pgvector semantic memory ingestion."""
    return (
        f"[TRADE AUTOPSY: {autopsy.symbol}]\n"
        f"Outcome: {autopsy.failure_reason.value} | Return: {autopsy.return_pct:+.2f}% | "
        f"Alpha: {autopsy.benchmark_relative_alpha_pct:+.2f}%\n"
        f"Excursions: MFE {autopsy.max_favorable_excursion_pct:.2f}%, "
        f"MAE {autopsy.max_adverse_excursion_pct:.2f}%\n"
        f"Forensic Root Cause: {autopsy.root_cause_analysis}\n"
        f"Institutional Lesson: {autopsy.preventative_lesson}"
    )


# Pure dynamic in-memory store; populates solely from live trade execution post-mortems.
_AUTOPSY_STORE: list[AutopsyResult] = []


def record_autopsy(autopsy: AutopsyResult) -> None:
    """Register autopsy in persistent memory store."""
    _AUTOPSY_STORE.append(autopsy)


def query_autopsies(query: str = "", limit: int = 8) -> list[AutopsyResult]:
    """Retrieve indexed autopsies matching query string."""
    q = query.strip().lower()
    if not q:
        return _AUTOPSY_STORE[-limit:]

    matches = [
        a
        for a in reversed(_AUTOPSY_STORE)
        if q in a.symbol.lower()
        or q in a.failure_reason.value.lower()
        or q in a.root_cause_analysis.lower()
        or q in a.preventative_lesson.lower()
    ]
    return matches[:limit]
