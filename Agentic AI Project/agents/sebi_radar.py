"""SEBI Regulation 30 Event-Driven Radar.

Monitors corporate disclosures filed under Regulation 30 of SEBI (Listing Obligations
and Disclosure Requirements) Regulations, 2015. Automatically scores regulatory severity,
computes materiality ratios against Net Worth / Turnover, and recommends immediate
portfolio de-risking actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Final

from trading_agent.domain.money import Rupees

__all__ = [
    "ActionRecommendation",
    "SebiAnnouncement",
    "SebiEventType",
    "SebiRadarAnalysis",
    "SebiSeverityLevel",
    "classify_sebi_headline",
    "evaluate_sebi_disclosure",
]

_SEBI_MATERIALITY_THRESHOLD_PCT: Final[float] = 2.0
_CRITICAL_LIABILITY_RATIO_PCT: Final[float] = 15.0
_HIGH_LIABILITY_RATIO_PCT: Final[float] = 5.0
_HIGH_PLEDGE_BPS_THRESHOLD: Final[int] = 500  # 5.00% additional equity pledge


class SebiEventType(StrEnum):
    """Categorization of SEBI Regulation 30 corporate events."""

    AUDITOR_RESIGNATION = "AUDITOR_RESIGNATION"
    PROMOTER_PLEDGE_INCREASE = "PROMOTER_PLEDGE_INCREASE"
    TAX_DEMAND_RAID = "TAX_DEMAND_RAID"
    KMP_RESIGNATION = "KMP_RESIGNATION"
    DEBT_DEFAULT = "DEBT_DEFAULT"
    ORDER_WIN_EXPANSION = "ORDER_WIN_EXPANSION"
    GENERIC_DISCLOSURE = "GENERIC_DISCLOSURE"


class SebiSeverityLevel(StrEnum):
    """Graded regulatory severity."""

    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ActionRecommendation(StrEnum):
    """Immediate portfolio manager action."""

    EMERGENCY_EXIT = "EMERGENCY_EXIT"
    REDUCE_EXPOSURE_50 = "REDUCE_EXPOSURE_50"
    HOLD_MONITOR = "HOLD_MONITOR"
    OPPORTUNISTIC_BUY = "OPPORTUNISTIC_BUY"


_KEYWORD_EVENT_MAPPINGS: Final[tuple[tuple[tuple[str, ...], SebiEventType], ...]] = (
    (
        ("auditor resign", "resignation of statutory auditor", "resignation of auditor"),
        SebiEventType.AUDITOR_RESIGNATION,
    ),
    (
        ("default in payment", "default on debt", "default on loan", "npa"),
        SebiEventType.DEBT_DEFAULT,
    ),
    (
        ("pledge", "encumbrance"),
        SebiEventType.PROMOTER_PLEDGE_INCREASE,
    ),
    (
        ("search", "seizure", "income tax raid", "gst demand", "tax demand"),
        SebiEventType.TAX_DEMAND_RAID,
    ),
    (
        ("resignation of cfo", "resignation of managing director", "resignation of ceo"),
        SebiEventType.KMP_RESIGNATION,
    ),
    (
        ("order received", "contract award", "awarded contract", "bagged order"),
        SebiEventType.ORDER_WIN_EXPANSION,
    ),
)


@dataclass(frozen=True, slots=True)
class SebiAnnouncement:
    """Standardized representation of a corporate filing under SEBI LODR."""

    announcement_id: str
    symbol: str
    timestamp: datetime
    headline: str
    details: str
    event_type: SebiEventType | None = None
    liability_amount: Rupees | None = None
    net_worth: Rupees | None = None
    annual_turnover: Rupees | None = None
    pledge_increase_bps: int | None = None


@dataclass(frozen=True, slots=True)
class SebiRadarAnalysis:
    """Forensic radar evaluation result."""

    announcement_id: str
    symbol: str
    event_type: SebiEventType
    severity: SebiSeverityLevel
    materiality_ratio_pct: float
    recommendation: ActionRecommendation
    risk_summary: str
    deleveraging_pct: float


def classify_sebi_headline(headline: str, details: str = "") -> SebiEventType:
    """Heuristic NLP classification of corporate announcements."""
    combined = f"{headline} {details}".lower()
    for keywords, event_type in _KEYWORD_EVENT_MAPPINGS:
        if any(k in combined for k in keywords):
            return event_type
    return SebiEventType.GENERIC_DISCLOSURE


def _calculate_materiality_ratio(announcement: SebiAnnouncement) -> float:
    """Compute liability percentage against net worth or turnover."""
    if announcement.liability_amount is None:
        return 0.0
    liability = announcement.liability_amount.value
    if announcement.net_worth is not None and announcement.net_worth.value > Decimal(0):
        ratio = float((liability / announcement.net_worth.value) * Decimal(100))
        return round(ratio, 2)
    if announcement.annual_turnover is not None and announcement.annual_turnover.value > Decimal(0):
        ratio = float((liability / announcement.annual_turnover.value) * Decimal(100))
        return round(ratio, 2)
    return 0.0


def _resolve_promoter_pledge_impact(
    announcement: SebiAnnouncement,
) -> tuple[SebiSeverityLevel, ActionRecommendation, str, float]:
    bps = announcement.pledge_increase_bps or 0
    if bps >= _HIGH_PLEDGE_BPS_THRESHOLD:
        return (
            SebiSeverityLevel.HIGH,
            ActionRecommendation.REDUCE_EXPOSURE_50,
            (
                f"Promoter pledge increased by {bps / 100:.2f}%. High risk of "
                "forced lender invocation during market drawdown."
            ),
            50.0,
        )
    return (
        SebiSeverityLevel.MEDIUM,
        ActionRecommendation.HOLD_MONITOR,
        f"Minor pledge change of {bps / 100:.2f}%. Monitor collateral buffer.",
        0.0,
    )


def _resolve_tax_raid_impact(
    materiality_ratio: float,
) -> tuple[SebiSeverityLevel, ActionRecommendation, str, float]:
    if materiality_ratio >= _CRITICAL_LIABILITY_RATIO_PCT:
        sev, rec, del_pct = (
            SebiSeverityLevel.CRITICAL,
            ActionRecommendation.EMERGENCY_EXIT,
            100.0,
        )
    elif materiality_ratio >= _HIGH_LIABILITY_RATIO_PCT:
        sev, rec, del_pct = (
            SebiSeverityLevel.HIGH,
            ActionRecommendation.REDUCE_EXPOSURE_50,
            50.0,
        )
    elif materiality_ratio >= _SEBI_MATERIALITY_THRESHOLD_PCT:
        sev, rec, del_pct = (
            SebiSeverityLevel.MEDIUM,
            ActionRecommendation.HOLD_MONITOR,
            0.0,
        )
    else:
        sev, rec, del_pct = (
            SebiSeverityLevel.LOW,
            ActionRecommendation.HOLD_MONITOR,
            0.0,
        )

    summary = (
        f"Tax/Regulatory demand represents {materiality_ratio:.2f}% of net worth. "
        f"Severity graded as {sev.value}."
    )
    return sev, rec, summary, del_pct


def evaluate_sebi_disclosure(announcement: SebiAnnouncement) -> SebiRadarAnalysis:
    """Evaluate SEBI filing for quantitative materiality and portfolio risk."""
    event_type = announcement.event_type or classify_sebi_headline(
        announcement.headline, announcement.details
    )
    materiality_ratio = _calculate_materiality_ratio(announcement)

    if event_type == SebiEventType.AUDITOR_RESIGNATION:
        sev = SebiSeverityLevel.CRITICAL
        rec = ActionRecommendation.EMERGENCY_EXIT
        summary = (
            "Statutory auditor resigned prematurely. Extreme forensic risk of "
            "accounting irregularities, adverse audit opinions, or dispute."
        )
        del_pct = 100.0
    elif event_type == SebiEventType.DEBT_DEFAULT:
        sev = SebiSeverityLevel.CRITICAL
        rec = ActionRecommendation.EMERGENCY_EXIT
        summary = (
            "Default on debt obligations reported. Impending rating downgrade to D. "
            "Immediate liquidity insolvency risk."
        )
        del_pct = 100.0
    elif event_type == SebiEventType.PROMOTER_PLEDGE_INCREASE:
        sev, rec, summary, del_pct = _resolve_promoter_pledge_impact(announcement)
    elif event_type == SebiEventType.TAX_DEMAND_RAID:
        sev, rec, summary, del_pct = _resolve_tax_raid_impact(materiality_ratio)
    elif event_type == SebiEventType.KMP_RESIGNATION:
        sev = SebiSeverityLevel.MEDIUM
        rec = ActionRecommendation.HOLD_MONITOR
        summary = "Abrupt departure of key leadership. Potential execution disruption."
        del_pct = 25.0
    elif event_type == SebiEventType.ORDER_WIN_EXPANSION:
        sev = SebiSeverityLevel.INFO
        rec = ActionRecommendation.OPPORTUNISTIC_BUY
        summary = "Material order win or capacity expansion filing. Positive catalyst."
        del_pct = 0.0
    else:
        sev = SebiSeverityLevel.LOW
        rec = ActionRecommendation.HOLD_MONITOR
        summary = "Standard compliance disclosure with minimal quantifiable tail risk."
        del_pct = 0.0

    return SebiRadarAnalysis(
        announcement_id=announcement.announcement_id,
        symbol=announcement.symbol,
        event_type=event_type,
        severity=sev,
        materiality_ratio_pct=materiality_ratio,
        recommendation=rec,
        risk_summary=summary,
        deleveraging_pct=del_pct,
    )
