"""Indian Concall NLP & Management Evasion Index Engine.

Performs forensic textual analysis of Indian quarterly earnings conference call transcripts.
Computes Management Evasion Index (MEI), Analyst Aggression Index (AAI), and Guidance
Sentiment Drift to uncover hidden corporate stress and impending earnings downgrades.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

__all__ = [
    "ConcallAnalysisResult",
    "ConcallSpeaker",
    "ConcallUtterance",
    "TransparencyGrade",
    "analyze_concall_transcript",
    "parse_raw_transcript_text",
]

_MIN_TEXT_LENGTH: Final[int] = 10
_MAX_EXCERPTS: Final[int] = 5
_GRADE_BULLISH_THRESHOLD: Final[float] = 75.0
_GRADE_NEUTRAL_THRESHOLD: Final[float] = 50.0
_GRADE_WARNING_THRESHOLD: Final[float] = 30.0

_EVASION_PATTERNS: Final[tuple[str, ...]] = (
    r"macro environment",
    r"at the appropriate time",
    r"difficult to give (a |any )?(specific )?(guidance|number|target)",
    r"premature to (comment|speculate)",
    r"take (this |it )?offline",
    r"do not comment on (individual|specific)",
    r"check with the (finance )?team",
    r"cannot share (that|specific)",
    r"lumpy nature of",
    r"wait and watch",
    r"too early to say",
    r"broadly in line with what we said earlier",
)

_AGGRESSION_PATTERNS: Final[tuple[str, ...]] = (
    r"you mentioned last quarter",
    r"why has (the )?working capital",
    r"discrepancy between",
    r"margin declined despite",
    r"debt covenant",
    r"inventory days rising",
    r"cash flow from operations is negative",
    r"why was there a delay",
    r"auditor raised",
    r"promoter loan",
    r"pledged shares",
    r"growth has slowed down",
)

_POSITIVE_GUIDANCE_WORDS: Final[tuple[str, ...]] = (
    "robust",
    "accelerating",
    "record order",
    "tailwinds",
    "margin expansion",
    "strong pipeline",
    "outperforming",
    "market share gain",
)

_NEGATIVE_GUIDANCE_WORDS: Final[tuple[str, ...]] = (
    "headwinds",
    "subdued",
    "tapering",
    "cautious",
    "delay",
    "margin pressure",
    "inflationary drag",
    "decelerating",
)


class ConcallSpeaker(StrEnum):
    """Dialogue participant role."""

    MANAGEMENT = "MANAGEMENT"
    ANALYST = "ANALYST"
    MODERATOR = "MODERATOR"


class TransparencyGrade(StrEnum):
    """Institutional transparency assessment."""

    TRANSPARENT_BULLISH = "TRANSPARENT_BULLISH"
    NEUTRAL_BALANCED = "NEUTRAL_BALANCED"
    HIGH_EVASION_WARNING = "HIGH_EVASION_WARNING"
    CRITICAL_FORENSIC_CONCERN = "CRITICAL_FORENSIC_CONCERN"

    #: No management dialogue survived parsing, so nothing was measured. This
    #: is not a grade on the call; it is the absence of one, and it exists so
    #: an unparsed transcript cannot be reported as a transparent call.
    UNMEASURABLE = "UNMEASURABLE"


_PREFIX_SPEAKER_MAP: Final[tuple[tuple[tuple[str, ...], ConcallSpeaker, str], ...]] = (
    (("analyst:", "question:"), ConcallSpeaker.ANALYST, "Analyst"),
    (("management:", "answer:"), ConcallSpeaker.MANAGEMENT, "Management"),
    (("moderator:",), ConcallSpeaker.MODERATOR, "Moderator"),
)


@dataclass(frozen=True, slots=True)
class ConcallUtterance:
    """Individual statement or query in conference call."""

    speaker: ConcallSpeaker
    speaker_name: str
    text: str


@dataclass(frozen=True, slots=True)
class ConcallAnalysisResult:
    """Diagnostic concall report."""

    symbol: str
    quarter: str
    #: ``None`` when there was no management dialogue to score against. A
    #: transcript that the speaker-prefix parser could not attribute yields no
    #: utterances, and the ratio these indices are built from has no
    #: denominator. It previously fell back to 0.0 -- read downstream as "zero
    #: evasion", which graded an unreadable transcript TRANSPARENT_BULLISH and
    #: put a 100/100 transparency score on a call nobody analysed.
    evasion_index_pct: float | None
    analyst_aggression_index_pct: float | None
    guidance_drift_pct: float
    transparency_score: float | None
    grade: TransparencyGrade
    evasive_excerpts: list[str]
    aggressive_queries: list[str]
    forensic_verdict: str


def _flush_buffer(
    utterances: list[ConcallUtterance],
    speaker: ConcallSpeaker,
    name: str,
    buffer: list[str],
) -> None:
    if buffer:
        utterances.append(ConcallUtterance(speaker, name, " ".join(buffer)))
        buffer.clear()


def parse_raw_transcript_text(raw_text: str) -> list[ConcallUtterance]:
    """Parse raw transcript format into structured utterance sequence."""
    utterances: list[ConcallUtterance] = []
    current_speaker = ConcallSpeaker.MODERATOR
    current_name = "Moderator"
    buffer: list[str] = []

    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        lower = stripped.lower()
        matched = False
        for prefixes, speaker, default_name in _PREFIX_SPEAKER_MAP:
            for prefix in prefixes:
                if lower.startswith(prefix):
                    _flush_buffer(utterances, current_speaker, current_name, buffer)
                    current_speaker = speaker
                    current_name = default_name
                    content = stripped.split(":", 1)[1].strip()
                    if content:
                        buffer.append(content)
                    matched = True
                    break
            if matched:
                break

        if not matched:
            buffer.append(stripped)

    _flush_buffer(utterances, current_speaker, current_name, buffer)
    return utterances


def _score_guidance_drift(management_utterances: list[ConcallUtterance]) -> float:
    """Calculate guidance sentiment balance between [-100.0, +100.0]."""
    full_mgmt_text = " ".join(u.text.lower() for u in management_utterances)
    if not full_mgmt_text:
        return 0.0

    pos_count = sum(full_mgmt_text.count(word) for word in _POSITIVE_GUIDANCE_WORDS)
    neg_count = sum(full_mgmt_text.count(word) for word in _NEGATIVE_GUIDANCE_WORDS)
    total = pos_count + neg_count
    if total == 0:
        return 0.0

    drift = ((pos_count - neg_count) / total) * 100.0
    return round(drift, 2)


def _determine_transparency_grade(score: float) -> TransparencyGrade:
    if score >= _GRADE_BULLISH_THRESHOLD:
        return TransparencyGrade.TRANSPARENT_BULLISH
    if score >= _GRADE_NEUTRAL_THRESHOLD:
        return TransparencyGrade.NEUTRAL_BALANCED
    if score >= _GRADE_WARNING_THRESHOLD:
        return TransparencyGrade.HIGH_EVASION_WARNING
    return TransparencyGrade.CRITICAL_FORENSIC_CONCERN


def analyze_concall_transcript(
    symbol: str, quarter: str, transcript_dialogues: list[ConcallUtterance]
) -> ConcallAnalysisResult:
    """Analyze concall dialogue to compute MEI, AAI, and composite transparency."""
    mgmt_utterances = [
        u
        for u in transcript_dialogues
        if u.speaker == ConcallSpeaker.MANAGEMENT and len(u.text) >= _MIN_TEXT_LENGTH
    ]
    analyst_utterances = [
        u
        for u in transcript_dialogues
        if u.speaker == ConcallSpeaker.ANALYST and len(u.text) >= _MIN_TEXT_LENGTH
    ]

    evasive_excerpts: list[str] = []
    evasion_count = 0
    for u in mgmt_utterances:
        lower_text = u.text.lower()
        if any(re.search(pat, lower_text) for pat in _EVASION_PATTERNS):
            evasion_count += 1
            if len(evasive_excerpts) < _MAX_EXCERPTS:
                evasive_excerpts.append(u.text)

    mei = (evasion_count / len(mgmt_utterances)) * 100.0 if mgmt_utterances else None

    aggressive_queries: list[str] = []
    aggression_count = 0
    for u in analyst_utterances:
        lower_text = u.text.lower()
        if any(re.search(pat, lower_text) for pat in _AGGRESSION_PATTERNS):
            aggression_count += 1
            if len(aggressive_queries) < _MAX_EXCERPTS:
                aggressive_queries.append(u.text)

    # Analyst questions are genuinely optional -- a prepared-remarks-only
    # transcript has none, and that is a fact about the call rather than a
    # parse failure. Zero pushback is the correct reading; None is not.
    aai = (aggression_count / len(analyst_utterances)) * 100.0 if analyst_utterances else 0.0

    gsd = _score_guidance_drift(mgmt_utterances)

    if mei is None:
        # Without management dialogue there is no evasion ratio, so there is
        # no transparency score either -- the score is 50% weighted on it.
        # Returning nulls and a verdict that names the cause is the only
        # honest output; the caller renders an em dash and the reason.
        return ConcallAnalysisResult(
            symbol=symbol,
            quarter=quarter,
            evasion_index_pct=None,
            analyst_aggression_index_pct=None,
            guidance_drift_pct=gsd,
            transparency_score=None,
            grade=TransparencyGrade.UNMEASURABLE,
            evasive_excerpts=[],
            aggressive_queries=[],
            forensic_verdict=(
                "No management dialogue could be attributed in this transcript, so "
                "evasion and transparency were not measured. Check that speaker lines "
                "are prefixed (for example 'Management:' or 'Analyst:') and that "
                f"responses exceed {_MIN_TEXT_LENGTH} characters."
            ),
        )

    negative_drift_penalty = max(0.0, -gsd)
    score_penalty = (mei * 0.50) + (aai * 0.35) + (negative_drift_penalty * 0.15)
    transparency = max(0.0, min(100.0, 100.0 - score_penalty))

    grade = _determine_transparency_grade(transparency)

    if grade == TransparencyGrade.CRITICAL_FORENSIC_CONCERN:
        verdict = (
            f"Severe management deflection ({mei:.1f}%) combined with high analyst skepticism "
            f"({aai:.1f}%). High probability of negative guidance reset."
        )
    elif grade == TransparencyGrade.HIGH_EVASION_WARNING:
        verdict = (
            f"Elevated evasion index ({mei:.1f}%). Management avoided direct margin "
            "and working capital quantification."
        )
    elif grade == TransparencyGrade.TRANSPARENT_BULLISH:
        verdict = (
            "Clear forward guidance with high direct answer ratio and positive business momentum."
        )
    else:
        verdict = (
            "Neutral call tone with balanced management responses and manageable analyst pushback."
        )

    return ConcallAnalysisResult(
        symbol=symbol,
        quarter=quarter,
        evasion_index_pct=round(mei, 2),
        analyst_aggression_index_pct=round(aai, 2),
        guidance_drift_pct=gsd,
        transparency_score=round(transparency, 2),
        grade=grade,
        evasive_excerpts=evasive_excerpts,
        aggressive_queries=aggressive_queries,
        forensic_verdict=verdict,
    )
