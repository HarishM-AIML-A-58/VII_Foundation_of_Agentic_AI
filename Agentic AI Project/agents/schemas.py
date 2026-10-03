"""Structured outputs for every agent node.

**The single most important deviation from the reference implementations.**

TradingAgents produces prose, then spends a *second* LLM call asking another
model to extract "BUY, SELL, or HOLD" from it. That is fragile (the extractor
can misread), expensive (an extra call per decision), and untyped (the result
is a string that may be anything).

Here every node returns a validated Pydantic object via structured output. The
decision is typed at the source, so there is nothing to parse and nothing to
misparse. A malformed response fails loudly at the node instead of silently
becoming a trade.

Confidence is deliberately on every schema: the risk layer weights by it, and
an agent that cannot say how sure it is has not finished thinking.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "AnalystOpinion",
    "DebateArgument",
    "ForensicSkepticVerdict",
    "IPOResearchNote",
    "ModeratorVerdict",
    "RiskVerdict",
    "Stance",
]

# StrEnum would serialise fine, but a plain Literal keeps the JSON schema that
# reaches the model as small and unambiguous as possible.
Stance = str


class _Strict(BaseModel):
    """Reject unknown keys so a hallucinated field fails rather than vanishes."""

    model_config = ConfigDict(extra="forbid")


class AnalystOpinion(_Strict):
    """One analyst's read of a symbol."""

    stance: str = Field(
        description="Exactly one of BULLISH, BEARISH or NEUTRAL.",
        pattern="^(BULLISH|BEARISH|NEUTRAL)$",
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="How strongly the evidence supports this stance, 0 to 1."
    )
    summary: str = Field(
        min_length=1,
        max_length=1200,
        description="Two or three sentences of reasoning, citing the specific numbers used.",
    )
    key_evidence: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Up to five short factual observations that drove the stance.",
    )


class DebateArgument(_Strict):
    """One turn from the bull or bear researcher."""

    argument: str = Field(
        min_length=1,
        max_length=2000,
        description="The case being made this turn, engaging with the opponent's last point.",
    )
    strongest_point: str = Field(
        min_length=1, max_length=300, description="The single most persuasive point made."
    )
    conceded: list[str] = Field(
        default_factory=list,
        max_length=3,
        description=(
            "Opponent points genuinely accepted this turn. An empty list every round "
            "usually means the debate is theatre rather than analysis."
        ),
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this side of the case.")


class ModeratorVerdict(_Strict):
    """The moderator's synthesis of the bull/bear exchange."""

    action: str = Field(
        description="Exactly one of BUY, SELL or HOLD.", pattern="^(BUY|SELL|HOLD)$"
    )
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=1500)
    winning_side: str = Field(
        description="Which case was stronger: BULL, BEAR or NEITHER.",
        pattern="^(BULL|BEAR|NEITHER)$",
    )
    # Entry geometry is proposed here and re-checked in code. The model's
    # arithmetic is never trusted -- `risk.veto` recomputes reward-to-risk
    # from these numbers and rejects on the recomputed value.
    entry: float | None = Field(default=None, gt=0, description="Proposed entry price.")
    stop_loss: float | None = Field(default=None, gt=0, description="Proposed protective stop.")
    target: float | None = Field(default=None, gt=0, description="Proposed profit target.")


class RiskVerdict(_Strict):
    """The risk manager's judgement.

    Advisory only. The binding checks -- reward-to-risk floor, position and
    sector limits, the kill switch -- live in :mod:`trading_agent.risk` and
    run in code afterwards. A model cannot be prompted out of arithmetic.
    """

    approve: bool = Field(description="Whether the trade is acceptable on risk grounds.")
    confidence: float = Field(ge=0.0, le=1.0)
    concerns: list[str] = Field(
        default_factory=list, max_length=5, description="Specific risks identified."
    )
    rationale: str = Field(min_length=1, max_length=1500)
    suggested_stop_loss: float | None = Field(
        default=None, gt=0, description="A tighter stop, if the proposed one is too loose."
    )


class ForensicSkepticVerdict(_Strict):
    """The forensic skeptic's adversarial audit verdict."""

    manipulation_risk: str = Field(
        description="Beneish M-Score classification: LOW or HIGH.",
        pattern="^(LOW|HIGH)$",
    )
    solvency_zone: str = Field(
        description="Altman Z-Score classification: SAFE, GREY, or DISTRESS.",
        pattern="^(SAFE|GREY|DISTRESS)$",
    )
    forensic_score: float = Field(
        ge=0.0, le=100.0, description="Composite forensic health score (0-100)."
    )
    red_flags: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Key forensic accounting or balance sheet warnings.",
    )
    invalidation_triggers: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="Concrete financial metric breaches that invalidate the investment thesis.",
    )
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str = Field(min_length=1, max_length=1500)


class IPOResearchNote(_Strict):
    """One LLM read over an IPO's scraped disclosures.

    Deliberately has no verdict, score, or recommendation field. This is
    commentary that helps a reader understand what the scraped data already
    says -- ``features.ipo_gonogo`` never reads it, and it is rendered
    everywhere labelled as AI-generated, not as a measured input. ``confidence``
    here means "how complete was the disclosed data", not "should you apply".
    """

    fundamentals_read: str = Field(
        min_length=1,
        max_length=1200,
        description=(
            "Plain-language read of the revenue/profit/assets trend actually given -- "
            "growth rate, margin direction, anomalies. Say so if none was given."
        ),
    )
    quarterly_trend: str = Field(
        min_length=1,
        max_length=600,
        description=(
            "Comment on quarter-over-quarter figures ONLY if quarterly data was in the "
            "input. If none was given, say plainly that no interim quarter data was "
            "disclosed -- never infer a quarterly trend from annual figures."
        ),
    )
    issue_structure_read: str = Field(
        min_length=1,
        max_length=600,
        description=(
            "Whether this is a fresh issue, an offer for sale, or both, and what that "
            "means for where the raised money goes. Say so if not given."
        ),
    )
    promoter_and_anchor_read: str = Field(
        min_length=1,
        max_length=600,
        description=(
            "Comment on anchor allocation size only if given. Never invent a promoter "
            "shareholding percentage that was not in the input."
        ),
    )
    key_risks: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Up to 5 risks drawn only from the pros/cons and financials given.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "How complete the disclosed data given was for this read -- not a buy/sell confidence."
        ),
    )
