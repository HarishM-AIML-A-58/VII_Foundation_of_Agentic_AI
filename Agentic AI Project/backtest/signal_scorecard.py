"""Does the Superinvesting data predict anything? Measured against NSE bars.

The research desk quotes third-party news tags, broker calls, persona verdicts
and trend signals. This module asks, for each, the only question that decides
how much weight it deserves: **after the signal was public, did the stock beat
the Nifty, by more than it costs to act?**

Four families, each tested only in the way its data honestly allows:

* **News impact tags** -- every item has a timestamp, so each is an event:
  enter at the first NSE open after publication, measure excess return over
  the Nifty at 1, 5 and 20 sessions.
* **Broker calls** -- the snapshot keeps each stock's latest call with its
  date; measured forward from the first open after that date.
* **Persona verdicts** -- only cards whose scoring date is in the past can be
  tested. The export carries 280 older cards (June to August) alongside the
  current ones; those are point-in-time opinions with a measurable future.
* **Trend signals** -- NOT scored on their past. The sweep lists signals that
  are still standing, so a Bullish call that failed has already flipped and
  vanished: its "return since signal" is a survivor by construction. What *can*
  be checked is integrity -- whether the stated signal date and price exist in
  our bars -- and the ledger records each export so the signals are scored
  forward from the day they were seen.

Rules that keep the numbers honest:

1. **No look-ahead.** Entry is the first session *open* strictly after the
   information was public (a pre-market story enters that day's open).
2. **Two benchmarks.** Excess over the Nifty says whether acting beat an index
   fund; excess over the *covered universe* (equal-weighted, same window) says
   whether the signal picked better than the stocks it was choosing among.
   Verdicts use the second: from February to August 2026 almost every covered
   stock beat the Nifty, and a test against the index alone credited broker
   Buy, Hold *and* Sell calls with the same outperformance.
3. **Date-clustered t-statistics.** Twenty defence stocks tagged Bullish on the
   same morning are one observation about that morning, not twenty. Events are
   averaged per entry date first, and the t-statistic is over dates.
4. **A cost hurdle.** The delivery round-trip cost from the verified charge
   schedule is reported beside every mean, because an edge smaller than STT is
   not an edge.
5. **Too few is a result.** Groups with fewer than :data:`MIN_DATES` distinct
   dates get no verdict.
"""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any, Final, cast

import pandas as pd

from trading_agent.costs.calculator import ChargeCalculator, UnverifiedScheduleWarning
from trading_agent.domain.enums import ProductType, Side
from trading_agent.market_data.corporate_actions import CorporateAction, adjust_for_actions
from trading_agent.market_data.store import BarStore
from trading_agent.market_data.superinvesting_lake import SuperinvestingLake, signal_ledger
from trading_agent.observability import get_logger

__all__ = [
    "MIN_DATES",
    "GroupResult",
    "ScorecardReport",
    "SectionResult",
    "build_scorecard",
    "load_scorecard",
    "write_scorecard",
]

log = get_logger(__name__)

IST: Final = "Asia/Kolkata"
MARKET_OPEN: Final = time(9, 15)
BENCHMARK: Final = "NIFTY50"

#: Distinct entry dates below which a group gets numbers but no verdict.
MIN_DATES: Final = 20
#: |t| at which an average is treated as distinguishable from zero.
T_THRESHOLD: Final = 2.0

SCORECARD_FILE: Final = "scorecard.json"

_RECO_LABELS: Final = {1: "BUY", 0: "HOLD", -1: "SELL"}
#: Sign a signal is claiming. None for tags that predict no direction.
_EXPECTED_SIGN: Final[dict[str, int | None]] = {
    "BULLISH": 1,
    "BEARISH": -1,
    "NEUTRAL": None,
    "BUY": 1,
    "SELL": -1,
    "HOLD": None,
    "STRONG CONSIDER": 1,
    "WATCH": None,
    "AVOID": -1,
    "Bullish": 1,
    "Bearish": -1,
    "Neutral": None,
}


# ------------------------------------------------------------------- results


@dataclass(frozen=True, slots=True)
class GroupResult:
    """One signal group at one horizon.

    Means are averaged per entry date first and then across dates, the same
    basis as the t-statistics, so a mean and its t never disagree in sign.
    """

    group: str
    horizon: str
    events: int
    dates: int
    #: Stock minus Nifty.
    mean_vs_nifty_pct: float | None
    t_vs_nifty: float | None
    #: Stock minus the equal-weighted covered universe. What the verdict uses.
    mean_vs_universe_pct: float | None
    t_vs_universe: float | None
    median_vs_universe_pct: float | None
    #: Share of events that beat the universe in the direction claimed.
    hit_rate: float | None
    verdict: str


@dataclass(slots=True)
class SectionResult:
    key: str
    title: str
    question: str
    method: str
    groups: list[GroupResult] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)
    summary: str = ""


@dataclass(slots=True)
class ScorecardReport:
    generated_at: str
    bars_through: str
    snapshot_as_of: dict[str, Any]
    round_trip_cost_pct: float | None
    sections: list[SectionResult]

    def to_json(self) -> dict[str, Any]:
        return cast("dict[str, Any]", _clean(asdict(self)))

    def headline(self) -> dict[str, str]:
        """Section key -> one-sentence summary, for prompts and evidence blocks."""
        return {s.key: s.summary for s in self.sections}


def _clean(value: Any) -> Any:
    if isinstance(value, float):
        return None if not math.isfinite(value) else round(value, 4)
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    if isinstance(value, date | datetime):
        return value.isoformat()
    return value


def load_scorecard(root: Path | str) -> dict[str, Any] | None:
    path = Path(root) / SCORECARD_FILE
    if not path.is_file():
        return None
    return cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))


# -------------------------------------------------------------------- prices


def _cell(frame: pd.DataFrame, day: date, column: str) -> float:
    """One price. pandas-stubs types a label lookup as any scalar at all."""
    return float(cast("float", frame.at[day, column]))  # noqa: PD008 -- scalar lookup


class _Prices:
    """Adjusted daily opens/highs/closes keyed by session date, read once."""

    def __init__(self, store: BarStore, actions: dict[str, list[CorporateAction]]) -> None:
        self._store = store
        self._actions = actions
        self._frames: dict[str, pd.DataFrame | None] = {}
        benchmark = self.frame(BENCHMARK)
        if benchmark is None or benchmark.empty:
            raise FileNotFoundError(f"no {BENCHMARK} bars; run `trading-agent data sync-index`")
        self.benchmark: pd.DataFrame = benchmark
        self.sessions: list[date] = list(benchmark.index)

    def frame(self, symbol: str) -> pd.DataFrame | None:
        if symbol not in self._frames:
            if not self._store.has(symbol):
                self._frames[symbol] = None
            else:
                bars = self._store.read(symbol)
                if symbol in self._actions:
                    bars = adjust_for_actions(bars, self._actions[symbol])
                bars = bars[["open", "high", "close"]].astype(float)
                bars.index = pd.DatetimeIndex(bars.index).date
                self._frames[symbol] = bars[~bars.index.duplicated(keep="last")]
        return self._frames[symbol]

    @property
    def last_session(self) -> date:
        return self.sessions[-1]

    def entry_index(self, public_at: pd.Timestamp) -> int | None:
        """Session index of the first open at or after ``public_at`` became known."""
        local = public_at.tz_convert(IST)
        day = local.date()
        strictly_after = local.time() >= MARKET_OPEN
        for i in range(self._first_on_or_after(day), len(self.sessions)):
            if self.sessions[i] == day and strictly_after:
                continue
            return i
        return None

    def entry_after_day(self, day: date) -> int | None:
        """First session strictly after ``day`` (information dated by day only)."""
        index = self._first_on_or_after(day)
        if index < len(self.sessions) and self.sessions[index] == day:
            index += 1
        return index if index < len(self.sessions) else None

    def _first_on_or_after(self, day: date) -> int:
        lo, hi = 0, len(self.sessions)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.sessions[mid] < day:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def window_return(self, frame: pd.DataFrame, start: int, sessions: int) -> float | None:
        """Open of session ``start`` to close ``sessions`` later (inclusive), in %."""
        end = start + sessions - 1
        if end >= len(self.sessions):
            return None
        first, last = self.sessions[start], self.sessions[end]
        if first not in frame.index or last not in frame.index:
            return None
        entry = _cell(frame, first, "open")
        exit_ = _cell(frame, last, "close")
        if entry <= 0:
            return None
        return (exit_ / entry - 1) * 100

    def set_universe(self, symbols: Iterable[str], *, min_members: int = 20) -> None:
        """The peer set. A window with fewer measurable members has no peer return:
        an average of three stocks is an anecdote, not a benchmark."""
        self._universe = [s for s in symbols if self.frame(s) is not None]
        self._min_members = min_members
        self._universe_cache: dict[tuple[int, int], float | None] = {}

    def universe_return(self, start: int, sessions: int) -> float | None:
        """Equal-weighted return of the covered universe over the same window."""
        key = (start, sessions)
        if key not in self._universe_cache:
            legs = [
                r
                for s in self._universe
                if (frame := self.frame(s)) is not None
                and (r := self.window_return(frame, start, sessions)) is not None
            ]
            self._universe_cache[key] = (
                sum(legs) / len(legs) if len(legs) >= self._min_members else None
            )
        return self._universe_cache[key]

    def excess(self, symbol: str, start: int, sessions: int) -> tuple[float, float] | None:
        """(stock minus Nifty %, stock minus universe %), or None if unmeasurable."""
        frame = self.frame(symbol)
        if frame is None:
            return None
        stock = self.window_return(frame, start, sessions)
        index = self.window_return(self.benchmark, start, sessions)
        universe = self.universe_return(start, sessions)
        if stock is None or index is None or universe is None:
            return None
        return stock - index, stock - universe


# ----------------------------------------------------------------- statistics


@dataclass(frozen=True, slots=True)
class _Event:
    group: str
    entry: date
    vs_nifty: float
    vs_universe: float


def _clustered(values: list[float], dates: list[date]) -> tuple[float | None, float | None, int]:
    """(mean over dates, t over dates, number of dates)."""
    daily = pd.Series(values, index=dates).groupby(level=0).mean()
    n = len(daily)
    if n == 0:
        return None, None, 0
    mean = float(daily.mean())
    if n < 3:  # noqa: PLR2004 -- a standard deviation needs a few points
        return mean, None, n
    spread = float(daily.std(ddof=1))
    return mean, (mean / (spread / math.sqrt(n)) if spread > 0 else None), n


def _label(group: str) -> str:
    """The signal label a group is named after: ``BULLISH · HIGH`` and
    ``canslim-1:AVOID`` are claims by ``BULLISH`` and ``AVOID``."""
    return group.split(" · ", maxsplit=1)[0].rsplit(":", maxsplit=1)[-1].strip()


def _verdict(
    group: str,
    mean: float | None,
    t: float | None,
    dates: int,
    cost_pct: float | None,
    *,
    before: bool = False,
) -> str:
    if mean is None or dates < MIN_DATES:
        return "too few dates to judge"
    if before:
        # A diagnostic, not a signal: nobody can trade the days before a story.
        expected = _EXPECTED_SIGN.get(_label(group))
        if t is not None and abs(t) >= T_THRESHOLD and expected is not None:
            if math.copysign(1, mean) == expected:
                return "price had already moved the tagged way"
            return "price had moved against the tag"
        return "no drift before publication"
    expected = _EXPECTED_SIGN.get(_label(group))
    if t is None or abs(t) < T_THRESHOLD:
        return "no measurable effect"
    if expected is not None and math.copysign(1, mean) != expected:
        return "works in reverse"
    if cost_pct is not None and abs(mean) < cost_pct:
        return "real but smaller than trading costs"
    return "earns weight" if expected is not None else "moves prices (direction untagged)"


def _summarise_groups(
    events: Iterable[_Event], horizon: str, cost_pct: float | None
) -> list[GroupResult]:
    by_group: dict[str, list[_Event]] = {}
    for event in events:
        by_group.setdefault(event.group, []).append(event)

    results: list[GroupResult] = []
    for group, rows in sorted(by_group.items()):
        dates = [r.entry for r in rows]
        nifty_mean, nifty_t, _ = _clustered([r.vs_nifty for r in rows], dates)
        mean, t, n_dates = _clustered([r.vs_universe for r in rows], dates)
        relative = pd.Series([r.vs_universe for r in rows])
        expected = _EXPECTED_SIGN.get(_label(group))
        hits = float((relative * expected > 0).mean()) if expected is not None else None
        results.append(
            GroupResult(
                group=group,
                horizon=horizon,
                events=len(rows),
                dates=n_dates,
                mean_vs_nifty_pct=nifty_mean,
                t_vs_nifty=nifty_t,
                mean_vs_universe_pct=mean,
                t_vs_universe=t,
                median_vs_universe_pct=float(relative.median()),
                hit_rate=hits,
                verdict=_verdict(
                    group, mean, t, n_dates, cost_pct, before=horizon.endswith("before")
                ),
            )
        )
    return results


def _headline(groups: list[GroupResult], preferred_horizon: str) -> str:
    """The strongest judged group at the preferred horizon, in one sentence."""
    judged = [
        g
        for g in groups
        if g.horizon == preferred_horizon and g.verdict != "too few dates to judge"
    ]
    if not judged:
        return f"Not enough matured observations at {preferred_horizon} to judge yet."
    parts = []
    for g in sorted(judged, key=lambda g: -(abs(g.t_vs_universe or 0))):
        parts.append(
            f"{g.group}: {g.mean_vs_universe_pct:+.2f}% vs peers over {g.horizon} "
            f"(t={g.t_vs_universe:.1f}; {g.events} events on {g.dates} dates) — {g.verdict}"
            if g.t_vs_universe is not None and g.mean_vs_universe_pct is not None
            else f"{g.group}: {g.verdict}"
        )
    return "; ".join(parts[:4])


# ------------------------------------------------------------------ sections


def _news_section(lake: SuperinvestingLake, prices: _Prices, cost: float | None) -> SectionResult:
    news = lake.table("news")
    section = SectionResult(
        key="news",
        title="News impact tags",
        question="After a story is tagged Bullish or Bearish, does the stock beat the Nifty?",
        method=(
            "Each (stock, entry session, direction) is one event, entered at the first NSE open "
            "after publication; excess return over the Nifty at 1, 5 and 20 sessions; "
            "t-statistic over entry dates."
        ),
    )
    seen: set[tuple[str, date, str]] = set()
    events: dict[str, list[_Event]] = {
        "5 sessions before": [],
        "1 session": [],
        "5 sessions": [],
        "20 sessions": [],
    }
    skipped = {"no_bars": 0, "not_matured": 0}
    rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    ordered = news.assign(_rank=news["impact_scale"].map(rank).fillna(0)).sort_values(
        "_rank", ascending=False
    )
    for item in ordered.to_dict(orient="records"):
        direction = str(item["impact_direction"] or "")
        if not direction:
            continue
        start = prices.entry_index(pd.Timestamp(item["published_at"]))
        if start is None:
            skipped["not_matured"] += 1
            continue
        key = (str(item["ticker"]), prices.sessions[start], direction)
        if key in seen:
            continue
        seen.add(key)
        if prices.frame(key[0]) is None:
            skipped["no_bars"] += 1
            continue
        windows = (
            ("5 sessions before", start - 5, 5),
            ("1 session", start, 1),
            ("5 sessions", start, 5),
            ("20 sessions", start, 20),
        )
        for label, first, length in windows:
            measured = prices.excess(key[0], first, length) if first >= 0 else None
            if measured is None:
                continue
            entry = prices.sessions[start]
            events[label].append(_Event(direction, entry, *measured))
            scale = str(item["impact_scale"] or "")
            if scale:
                events[label].append(_Event(f"{direction} · {scale}", entry, *measured))

    for label, rows in events.items():
        section.groups.extend(_summarise_groups(rows, label, cost))
    section.facts = {
        "items": len(news),
        "distinct_events": len(seen),
        "skipped_no_bars": skipped["no_bars"],
        "published_after_last_bar": skipped["not_matured"],
    }
    section.caveats = [
        "'5 sessions before' is a diagnostic of whether stories follow price moves; it is "
        "not tradeable and carries no verdict about the tag's value.",
        "Impact tags were assigned by the source's model when the story was ingested; the "
        "tagging process itself is not observable.",
        "Windows of adjacent dates overlap, so 20-session t-statistics overstate certainty.",
        f"Coverage is limited to stocks in the platform bar store "
        f"({skipped['no_bars']} events had no bars).",
    ]
    section.summary = _news_summary(section.groups)
    return section


def _news_summary(groups: list[GroupResult]) -> str:
    """Before-and-after in one sentence per direction: the story news tells."""
    lookup = {(g.group, g.horizon): g for g in groups}
    parts = []
    for direction in ("BULLISH", "BEARISH"):
        before = lookup.get((direction, "5 sessions before"))
        after = lookup.get((direction, "5 sessions"))
        if before is None or after is None:
            continue
        if before.mean_vs_universe_pct is None or after.mean_vs_universe_pct is None:
            continue
        parts.append(
            f"{direction.title()} stories arrive after {before.mean_vs_universe_pct:+.2f}% vs "
            f"peers in the prior 5 sessions (t={before.t_vs_universe or 0:.1f}) and are followed "
            f"by {after.mean_vs_universe_pct:+.2f}% over the next 5 "
            f"(t={after.t_vs_universe or 0:.1f}; {after.events} events on {after.dates} dates): "
            f"{after.verdict}"
        )
    return "; ".join(parts) or "Not enough matured news events to judge yet."


def _broker_section(lake: SuperinvestingLake, prices: _Prices, cost: float | None) -> SectionResult:
    companies = lake.table("companies")
    section = SectionResult(
        key="broker_calls",
        title="Broker calls",
        question="Did stocks with a broker Buy call beat the Nifty after the call?",
        method=(
            "Entry at the first open after the call date; excess over the Nifty at 20, 60 and "
            "120 sessions; target reached when any later high touched the broker's target."
        ),
    )
    events: dict[str, list[_Event]] = {"20 sessions": [], "60 sessions": [], "120 sessions": []}
    upsides: list[float] = []
    reached = 0
    measured_calls = 0
    for row in companies.dropna(subset=["analyst_date", "analyst_recommendation"]).to_dict(
        orient="records"
    ):
        label = _RECO_LABELS.get(int(row["analyst_recommendation"]))
        frame = prices.frame(str(row["ticker"]))
        if label is None or frame is None:
            continue
        start = prices.entry_after_day(pd.Timestamp(str(row["analyst_date"])[:10]).date())
        if start is None:
            continue
        entry_day = prices.sessions[start]
        if entry_day not in frame.index:
            continue
        measured_calls += 1
        entry_price = _cell(frame, entry_day, "open")
        target = row.get("analyst_target")
        if label == "BUY" and target and entry_price > 0:
            upsides.append((float(target) / entry_price - 1) * 100)
            if float(frame.loc[frame.index >= entry_day, "high"].max()) >= float(target):
                reached += 1
        for horizon, length in (("20 sessions", 20), ("60 sessions", 60), ("120 sessions", 120)):
            measured = prices.excess(str(row["ticker"]), start, length)
            if measured is not None:
                events[horizon].append(_Event(label, entry_day, *measured))

    for label, rows in events.items():
        section.groups.extend(_summarise_groups(rows, label, cost))
    buys = len(upsides)
    section.facts = {
        "calls_measured": measured_calls,
        "buy_calls_with_target": buys,
        "median_upside_at_entry_pct": float(pd.Series(upsides).median()) if upsides else None,
        "buy_targets_reached": reached,
        "buy_target_hit_rate": reached / buys if buys else None,
    }
    section.caveats = [
        "Only each stock's latest call survives in the snapshot; earlier calls it replaced "
        "cannot be tested.",
        "Calls cluster in February 2026, so most 'dates' share one market regime.",
    ]
    section.summary = _headline(section.groups, "60 sessions")
    return section


def _persona_section(
    lake: SuperinvestingLake, prices: _Prices, cost: float | None
) -> SectionResult:
    history = lake.table("persona_history")
    section = SectionResult(
        key="personas",
        title="Persona verdicts",
        question="Did STRONG CONSIDER cards beat AVOID cards after they were scored?",
        method=(
            "Every dated card is a point-in-time opinion. Entry at the first open after the "
            "scoring date; excess over the Nifty at 5, 20 and 40 sessions, by verdict."
        ),
    )
    events: dict[str, list[_Event]] = {"5 sessions": [], "20 sessions": [], "40 sessions": []}
    pending = 0
    for card in history.dropna(subset=["updated_at", "verdict"]).to_dict(orient="records"):
        start = prices.entry_index(pd.Timestamp(card["updated_at"]))
        measured_any = False
        for horizon, length in (("5 sessions", 5), ("20 sessions", 20), ("40 sessions", 40)):
            measured = (
                prices.excess(str(card["ticker"]), start, length) if start is not None else None
            )
            if measured is None or start is None:
                continue
            measured_any = True
            entry = prices.sessions[start]
            verdict = str(card["verdict"])
            events[horizon].append(_Event(verdict, entry, *measured))
            events[horizon].append(_Event(f"{card['persona_id']}:{verdict}", entry, *measured))
        pending += not measured_any

    for label, rows in events.items():
        section.groups.extend(_summarise_groups(rows, label, cost))
    for horizon in events:
        strong = next(
            (g for g in section.groups if g.group == "STRONG CONSIDER" and g.horizon == horizon),
            None,
        )
        avoid = next(
            (g for g in section.groups if g.group == "AVOID" and g.horizon == horizon), None
        )
        if (
            strong
            and avoid
            and strong.mean_vs_universe_pct is not None
            and avoid.mean_vs_universe_pct is not None
        ):
            section.facts[f"strong_minus_avoid_{horizon.split()[0]}_sessions_pct"] = (
                strong.mean_vs_universe_pct - avoid.mean_vs_universe_pct
            )
    dates = sorted({e.entry for rows in events.values() for e in rows})
    section.facts.update(
        {
            "cards": len(history),
            "cards_with_no_matured_horizon": pending,
            "scoring_dates": len(dates),
        }
    )
    section.caveats = [
        "Nearly all testable cards come from a handful of scoring days (mostly 23 July), so "
        "the date-clustered sample is tiny even when the card count is not.",
        "Cards scored in September have no future in the bar store yet; re-run as bars arrive.",
    ]
    section.summary = _headline([g for g in section.groups if ":" not in g.group], "20 sessions")
    return section


def _trend_section(lake: SuperinvestingLake, prices: _Prices, cost: float | None) -> SectionResult:
    indicators = lake.table("indicators")
    section = SectionResult(
        key="trend_signals",
        title="Trend signals",
        question="Are the signal dates real, and do signals recorded at export beat the Nifty?",
        method=(
            "Integrity: rebuild each signal's date and price from 'days since signal' and "
            "'price change since signal' and look for that close in NSE bars. Performance: "
            "score each export's recorded signals forward from the export date."
        ),
    )
    # Which calendar convention and sweep date explain the stated signal prices best.
    candidates: list[tuple[float, int, date]] = []
    last = prices.last_session
    for offset in range(0, 6):
        sweep = last + pd.Timedelta(days=offset).to_pytimedelta()
        checked = matched = 0
        for row in indicators.to_dict(orient="records"):
            frame = prices.frame(str(row["ticker"]))
            if frame is None or row.get("price_change_since_signal_pct") is None:
                continue
            implied = float(row["close"]) / (1 + float(row["price_change_since_signal_pct"]) / 100)
            signal_day = sweep - pd.Timedelta(days=int(row["days_since_signal"])).to_pytimedelta()
            before = frame.loc[frame.index <= signal_day, "close"]
            if before.empty or signal_day > last:
                continue
            checked += 1
            matched += abs(float(before.iloc[-1]) / implied - 1) < 0.015  # noqa: PLR2004
        if checked:
            candidates.append((matched / checked, checked, sweep))
    if candidates:
        rate, checked, sweep = max(candidates)
        section.facts = {
            "signals": len(indicators),
            "checked_against_bars": checked,
            "inferred_sweep_date": sweep,
            "signal_price_matches_bars": rate,
            "convention": "calendar days before the sweep date",
        }
        integrity = (
            f"{rate:.0%} of {checked} signal prices match NSE closes (±1.5%) when dated in "
            f"calendar days before {sweep.isoformat()}"
        )
    else:
        integrity = "no signal could be checked against bars"

    ledger = signal_ledger(lake.root)
    events: dict[str, list[_Event]] = {"5 sessions": [], "20 sessions": []}
    pending = 0
    for row in ledger.to_dict(orient="records") if not ledger.empty else []:
        start = prices.entry_after_day(date.fromisoformat(str(row["recorded_on"])))
        if start is None:
            pending += 1
            continue
        for horizon, length in (("5 sessions", 5), ("20 sessions", 20)):
            measured = prices.excess(str(row["ticker"]), start, length)
            if measured is not None:
                events[horizon].append(
                    _Event(str(row["sentiment"]), prices.sessions[start], *measured)
                )
    for label, rows in events.items():
        section.groups.extend(_summarise_groups(rows, label, cost))
    section.facts["ledger_signals_recorded"] = len(ledger)
    section.facts["ledger_signals_pending"] = pending
    section.caveats = [
        "Returns since signal are NOT used as evidence: the sweep only lists signals still "
        "standing, so failed signals have already disappeared from it.",
        "Forward scoring needs repeated exports; one export is one date, and one date is not "
        "a sample.",
    ]
    section.summary = integrity + (
        f"; forward record: {pending} recorded signals awaiting bars"
        if not any(section.groups)
        else "; " + _headline(section.groups, "20 sessions")
    )
    return section


# --------------------------------------------------------------------- entry


def _round_trip_cost_pct(calculator: ChargeCalculator | None, on: date) -> float | None:
    """Delivery round-trip charges on a ₹1 lakh position, in percent."""
    if calculator is None:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnverifiedScheduleWarning)
        charges = calculator.round_trip(
            entry_price=Decimal(1000),
            exit_price=Decimal(1000),
            quantity=100,
            side=Side.BUY,
            product=ProductType.CNC,
            entry_date=on,
        )
    return float(charges.total.value) / 100_000 * 100


def build_scorecard(
    lake: SuperinvestingLake,
    store: BarStore,
    *,
    calculator: ChargeCalculator | None = None,
    actions: dict[str, list[CorporateAction]] | None = None,
    min_universe: int = 20,
) -> ScorecardReport:
    """Measure every signal family against the bar store and return the report."""
    prices = _Prices(store, actions or {})
    prices.set_universe(lake.table("companies")["ticker"].tolist(), min_members=min_universe)
    cost = _round_trip_cost_pct(calculator, prices.last_session)
    sections = [
        _news_section(lake, prices, cost),
        _broker_section(lake, prices, cost),
        _persona_section(lake, prices, cost),
        _trend_section(lake, prices, cost),
    ]
    log.info("signal_scorecard_built", bars_through=str(prices.last_session))
    return ScorecardReport(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        bars_through=prices.last_session.isoformat(),
        snapshot_as_of=lake.manifest.get("as_of", {}),
        round_trip_cost_pct=cost,
        sections=sections,
    )


def write_scorecard(report: ScorecardReport, root: Path | str) -> Path:
    path = Path(root) / SCORECARD_FILE
    path.write_text(json.dumps(report.to_json(), indent=2), encoding="utf-8")
    return path
