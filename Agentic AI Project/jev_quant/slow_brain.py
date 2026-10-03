"""
Slow Brain & Self-Improvement Layer, backed by Microsoft Foundry.

Uses the same Foundry deployment as the rest of the trading agent
(TRADING_AGENT__FOUNDRY__DEPLOYMENT, gpt-5-mini today), so no model name is
hardcoded here. Called after a session to:
1. Review the day's trade fills and misses.
2. Calculate calibration metrics (Brier score).
3. Identify root causes of losses.
4. Propose strategy rule revisions for backtesting against the performance gates.

It can also audit a Jev scan: read the snapshots Jev was given next to the
probabilities it returned, and flag scores the inputs do not support.

Everything the model returns is advisory. Metrics are computed here, never by
the model, and RiskGuardian limits live in code the model cannot reach.
"""
from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

#: Foundry role name. Resolves to the shared deployment unless an override is
#: set under `foundry.deployments.quant_reviewer`.
REVIEW_ROLE = "quant_reviewer"

#: A reasoning model auditing a multi-asset scan takes longer than the 120 s
#: default that suits the debate roles.
REVIEW_TIMEOUT_SECONDS = 300.0


@dataclass
class TradeRecord:
    trade_id: str
    symbol: str
    entry_price: float
    exit_price: float
    pnl_usd: float
    calibrated_prob: float
    actual_win: int  # 1 for win, 0 for loss
    jev_latency_ms: float


class RuleProposal(BaseModel):
    rule: str = Field(description="One concrete, backtestable change to a gate, filter or exit.")
    rationale: str = Field(description="Why the session evidence supports it.")
    evidence: str = Field(description="The specific metrics or trades it rests on.")


class SessionReview(BaseModel):
    summary: str = Field(description="Two or three sentences on what drove the session result.")
    loss_root_causes: List[str] = Field(default_factory=list)
    proposals: List[RuleProposal] = Field(default_factory=list)


class ScoreFlag(BaseModel):
    symbol: str
    question: str = Field(description="The Jev question whose score looks wrong.")
    score: float
    concern: str = Field(description="What in the snapshot contradicts the score.")


class ScanReview(BaseModel):
    verdict: Literal["consistent", "questionable", "inconsistent"]
    summary: str
    flags: List[ScoreFlag] = Field(default_factory=list)
    data_quality_issues: List[str] = Field(
        default_factory=list,
        description="Snapshot inputs that look stale, constant or impossible.",
    )


_SESSION_PROMPT = """You review a day of trades from a systematic quant engine.
Jev scores five fixed-outcome questions per setup; a deterministic risk guardian
sizes with capped 1/4 Kelly and enforces position, daily-loss and drawdown limits.
The metrics you are given were computed exactly; do not recompute or restate them
as your own. Explain what drove the result, name root causes of the losing trades,
and propose specific rule changes that can be backtested. Ground every claim in
the trades provided. If the sample is too small to support a change, say so and
propose nothing."""

_SCAN_PROMPT = """You audit one scan from a quant engine. For each asset you get the
numeric snapshot that was sent to Jev and the probability Jev returned for each
question; the questions and the gate each must clear are given once per strategy. Check whether each score is
consistent with the snapshot (for example, a high oversold score with RSI above 50,
or high buying pressure with a volume surge ratio below 1). Also flag snapshot
inputs that look hardcoded, stale or impossible. Only flag what the numbers show."""


def session_metrics(trades: List[TradeRecord]) -> Dict[str, Any]:
    """Deterministic session statistics. Never delegated to the model."""
    total = len(trades)
    wins = sum(1 for t in trades if t.pnl_usd > 0)
    return {
        "total_trades": total,
        "win_rate_pct": (wins / total) * 100.0,
        "total_pnl_usd": sum(t.pnl_usd for t in trades),
        "brier_score": sum((t.calibrated_prob - t.actual_win) ** 2 for t in trades) / total,
        "avg_jev_latency_ms": sum(t.jev_latency_ms for t in trades) / total,
        "largest_loss_usd": min((t.pnl_usd for t in trades), default=0.0),
    }


def baseline_rules(metrics: Dict[str, Any]) -> List[str]:
    """Fixed heuristics, used on their own when Foundry is unavailable."""
    rules = []
    if metrics["brier_score"] > 0.20:
        rules.append("Recalibrate Jev probability scaling curve; overconfidence observed.")
    if metrics["largest_loss_usd"] < -1000.0:
        rules.append("Tighten pre-market volatility filter before scheduled economic releases.")
    return rules


class SlowBrain:
    def __init__(self, client: Optional[Any] = None, role: str = REVIEW_ROLE):
        # `client` is a trading_agent FoundryClient (anything with the same
        # `structured` / `deployment_name` methods works). None runs the
        # deterministic review only.
        self.client = client
        self.role = role

    @classmethod
    def from_settings(cls) -> "SlowBrain":
        """Build against the Foundry deployment configured for the trading agent."""
        from trading_agent.agents.llm import FoundryClient
        from trading_agent.settings import get_settings

        foundry = get_settings().foundry
        return cls(FoundryClient(foundry.model_copy(update={"timeout_seconds": max(foundry.timeout_seconds, REVIEW_TIMEOUT_SECONDS)})))

    @property
    def deployment(self) -> Optional[str]:
        return self.client.deployment_name(self.role) if self.client is not None else None

    async def _ask(self, system: str, payload: Dict[str, Any], schema: type) -> tuple[Any, tuple[int, int]]:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=system),
            HumanMessage(content=json.dumps(payload, indent=2, default=str)),
        ]
        return await self.client.structured(self.role, messages, schema)

    async def areview_session(self, trades: List[TradeRecord]) -> Dict[str, Any]:
        """Reviews daily performance and generates an improvement report."""
        if not trades:
            return {"status": "no_trades", "recommendations": "Maintain existing parameters."}

        metrics = session_metrics(trades)
        report: Dict[str, Any] = {**metrics, "proposed_rules": baseline_rules(metrics), "reviewer": "deterministic"}
        if self.client is None:
            return report

        try:
            review, tokens = await self._ask(
                _SESSION_PROMPT,
                {"metrics": metrics, "trades": [asdict(t) for t in trades]},
                SessionReview,
            )
        except Exception as exc:  # noqa: BLE001 -- a failed review must not lose the metrics
            report["reviewer_error"] = f"{type(exc).__name__}: {exc}"
            return report

        report.update(
            reviewer=self.deployment,
            summary=review.summary,
            loss_root_causes=review.loss_root_causes,
            model_proposals=[p.model_dump() for p in review.proposals],
            tokens={"input": tokens[0], "output": tokens[1]},
        )
        return report

    def review_session(self, trades: List[TradeRecord]) -> Dict[str, Any]:
        return asyncio.run(self.areview_session(trades))

    async def areview_scan(self, scan_state: Dict[str, Any]) -> Dict[str, Any]:
        """Audits a scanner state file (logs/intensive_scanner_state.json)."""
        if self.client is None:
            raise RuntimeError("Scan review needs a Foundry client; use SlowBrain.from_settings().")
        assets = scan_state.get("assets", [])
        if not assets:
            return {"status": "empty_scan"}

        # Questions and gates are per strategy, so send each set once.
        strategies: Dict[str, Any] = {}
        compact = []
        for a in assets:
            strat = a.get("strategy", a.get("type", "default"))
            strategies.setdefault(strat, {"questions": a.get("questions"), "gates": a.get("gates")})
            compact.append({
                "symbol": a["symbol"], "strategy": strat, "snapshot": a.get("snapshot"),
                "scores": a.get("scores"), "decision": a.get("action"),
            })
        payload = {"scanned_at": scan_state.get("timestamp"), "strategies": strategies, "assets": compact}
        review, tokens = await self._ask(_SCAN_PROMPT, payload, ScanReview)
        return {
            "reviewer": self.deployment,
            **review.model_dump(),
            "tokens": {"input": tokens[0], "output": tokens[1]},
        }

    def review_scan(self, scan_state: Dict[str, Any]) -> Dict[str, Any]:
        return asyncio.run(self.areview_scan(scan_state))


if __name__ == "__main__":
    # Audit the latest Jev scan:  python jev_opus_quant/engine/slow_brain.py [state.json]
    state_path = Path(sys.argv[1] if len(sys.argv) > 1 else "logs/intensive_scanner_state.json")
    result = SlowBrain.from_settings().review_scan(json.loads(state_path.read_text()))
    print(json.dumps(result, indent=2))
