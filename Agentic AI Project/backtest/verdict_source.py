"""Loading recorded agent verdicts for backtest replay.

Verdicts arrive as a JSON file rather than a live database read, for two
reasons. It keeps the replay reproducible -- the same file gives the same answer
on every run, which a query against a mutating table does not. And it keeps the
measurement runnable without Postgres, so the agent layer can be evaluated on a
laptop instead of only where the operational store happens to be up.

Export shape, one object per recorded debate::

    [
        {
            "symbol": "RELIANCE",
            "session_date": "2026-08-06",
            "action": "BUY",
            "confidence": 0.72,
            "abstentions": ["sentiment_analyst"],
        }
    ]

``action`` is the moderator's verdict, and HOLD is a first-class value: a
recorded HOLD is the evidence that the agent declined, which is exactly what the
filter needs. Dropping HOLDs from the export would make the agent look like it
only ever agreed.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

from trading_agent.backtest.agent_filter import RecordedVerdict
from trading_agent.domain.enums import Action
from trading_agent.observability import get_logger

__all__ = ["VerdictLoadError", "load_recorded_verdicts"]

log = get_logger(__name__)


class VerdictLoadError(ValueError):
    """The verdict file could not be read as recorded debates."""


def _parse_session_date(value: str) -> date:
    """Accept a date or a full ISO timestamp, reject anything else.

    A database export naturally produces ``2015-01-01T00:00:00``, and refusing
    it would push a reformatting step onto every caller for no gain -- a
    timestamp's date part is unambiguous. Anything that is neither still raises,
    because a silently skipped row makes the agent look better covered than it
    was, and coverage decides whether a filter result means anything.
    """
    text = value.strip()
    try:
        return date.fromisoformat(text)
    except ValueError:
        return datetime.fromisoformat(text).date()


def load_recorded_verdicts(path: Path | str) -> list[RecordedVerdict]:
    """Read recorded verdicts, refusing anything malformed rather than skipping it.

    A silently dropped row makes the agent look better covered than it was, and
    coverage is the number that decides whether a filter result means anything.
    """
    source = Path(path)
    if not source.is_file():
        raise VerdictLoadError(f"no verdict file at {source}")

    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise VerdictLoadError(f"{source} is not valid JSON: {exc}") from exc

    if not isinstance(raw, list):
        raise VerdictLoadError(
            f"{source} must hold a JSON array of verdicts, got {type(raw).__name__}"
        )

    verdicts: list[RecordedVerdict] = []
    for position, row in enumerate(raw):
        if not isinstance(row, dict):
            raise VerdictLoadError(f"{source}[{position}] is not an object")
        try:
            verdicts.append(
                RecordedVerdict(
                    symbol=str(row["symbol"]).strip().upper(),
                    session_date=_parse_session_date(str(row["session_date"])),
                    action=Action(str(row["action"]).upper()),
                    confidence=float(row.get("confidence", 0.0)),
                    abstentions=tuple(str(a) for a in row.get("abstentions", ())),
                )
            )
        except (KeyError, ValueError) as exc:
            raise VerdictLoadError(f"{source}[{position}] is not a usable verdict: {exc}") from exc

    log.info(
        "agent_verdicts_loaded",
        path=str(source),
        verdicts=len(verdicts),
        symbols=len({v.symbol for v in verdicts}),
        sessions=len({v.session_date for v in verdicts}),
    )
    return verdicts
