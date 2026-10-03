"""The research desk's tools: seven specialist agents and a follow-up generator.

Modelled on the superinvesting.ai tool set (``Superinvesting/
SUPERINVESTING_AI_ARCHITECTURE.md``): ``screener``, ``browser``,
``comparison``, ``persona``, ``research``, ``sector``, ``portfolio`` and
``suggestionTool``. The reverse-engineered reference stubs every tool with a
hard-coded sentence ("Forensic audit shows clean balance sheet..."). Here each
one is arithmetic over two kinds of data, and says which it used:

* the **Superinvesting lake** -- a dated snapshot of scores, baselines, persona
  cards, news and baskets (:class:`SuperinvestingLake`);
* the **platform's own stores** -- curated NSE bars (:class:`BarStore`), dated
  fundamentals (:class:`FundamentalStore`) and, when a broker session is live,
  real holdings.

Every result carries ``as_of`` and ``source`` so the model can tell a
September snapshot from yesterday's close, and every value the data does not
hold comes back as ``null`` with a reason. The model is told to say "not in the
data"; it can only do that if the tools never paper over a gap.
"""

from __future__ import annotations

import math
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any, Final, Literal, cast

import pandas as pd
from pydantic import BaseModel, Field

from trading_agent.features.screener import score_symbol
from trading_agent.market_data.fundamentals import FundamentalStore, Metric
from trading_agent.market_data.store import BarStore
from trading_agent.market_data.superinvesting_lake import PERSONAS, SuperinvestingLake
from trading_agent.observability import get_logger

__all__ = [
    "AGENT_LABELS",
    "TOOL_ARGS",
    "HoldingRow",
    "HoldingsProvider",
    "HoldingsSnapshot",
    "ResearchToolkit",
]

log = get_logger(__name__)

#: Tool name -> label the UI shows beside the step.
AGENT_LABELS: Final[dict[str, str]] = {
    "screener": "Screener Agent",
    "browser": "Browser Agent",
    "comparison": "Comparison Agent",
    "persona": "Persona Agent",
    "research": "Research Agent",
    "sector": "Sector Agent",
    "portfolio": "Portfolio Agent",
    "memory": "Desk Ledger",
    "suggestionTool": "Suggestions",
}

_STEP = "User-facing progress label, e.g. 'Fetching ROCE and valuation for TCS...'"

PersonaId = Literal[
    "warren-buffett-1", "peter-lynch-1", "rakesh-jhunjhunwala-1", "100-bagger", "canslim-1"
]

# ------------------------------------------------------------------ schemas


class ScreenFilter(BaseModel):
    field: str = Field(description="A screenable field name; see the tool description.")
    op: Literal[">", ">=", "<", "<=", "==", "!="]
    value: float | str


class ScreenerArgs(BaseModel):
    """Fundamentals, valuation, scores and technicals for named stocks, or a
    filter screen across the universe.

    Profile mode: pass ``tickers`` (NSE symbols or company names).
    Screen mode: pass ``filters`` and optionally ``sectors`` / ``capitalization``.

    Screenable fields -- valuation: pe, industry_pe, peg, pb, intrinsic_value,
    price, market_cap_cr; quality: roce, roe, opm_latest_quarter,
    npm_latest_quarter, debt_to_equity, financial_leverage, interest_coverage,
    current_ratio; growth: sales_growth_1y, sales_cagr_3y, sales_cagr_5y,
    profit_growth_1y, profit_cagr_3y, profit_cagr_5y, eps_growth_3y_pct;
    scores (0-100): fundamental_score, business_score, valuation_score,
    technical_score, forensic_score, superinvesting_score; persona scores:
    warren_buffett, peter_lynch, rakesh_jhunjhunwala, hundred_bagger, canslim;
    signal: signal_sentiment (Bullish/Neutral/Bearish), days_since_signal,
    price_change_since_signal_pct; returns: one_year_return_pct,
    three_year_return_pct. Percentages are in percent (15 means 15%).
    Market cap is in INR crore.
    """

    message: str = Field(description=_STEP)
    tickers: list[str] = Field(default_factory=list, max_length=10)
    filters: list[ScreenFilter] = Field(default_factory=list)
    sectors: list[str] = Field(
        default_factory=list,
        description=(
            "Sector names or themes, any-match (e.g. ['energy'], ['Pharmaceuticals', 'Hospital'])"
        ),
    )
    capitalization: Literal["LARGECAP", "MIDCAP", "SMALLCAP"] | None = None
    sort_by: str | None = Field(default=None, description="Field to rank by")
    descending: bool = True
    limit: int = Field(default=15, ge=1, le=40)


class BrowserArgs(BaseModel):
    """Search the ingested market-news feed (company news with impact tags,
    April-September 2026) and optionally fetch live quarterly and annual
    financials from screener.in. It cannot open arbitrary web pages, concall
    transcripts or exchange filings."""

    message: str = Field(description=_STEP)
    query: str = Field(description="Keywords, e.g. 'order book guidance margin'")
    tickers: list[str] = Field(default_factory=list, max_length=5)
    days: int = Field(default=90, ge=1, le=365, description="Look-back window in days")
    direction: Literal["BULLISH", "BEARISH", "NEUTRAL"] | None = None
    live_financials: bool = Field(
        default=False,
        description="Also fetch the latest reported financials from screener.in for tickers",
    )
    limit: int = Field(default=10, ge=1, le=25)


class ComparisonArgs(BaseModel):
    """Benchmark 2+ stocks side by side: each metric with sector-peer median,
    percentile within the sector and within the whole covered universe."""

    message: str = Field(description=_STEP)
    tickers: list[str] = Field(min_length=1, max_length=8)
    metrics: list[str] = Field(
        default_factory=list,
        description="Fields as in screener; empty uses a standard valuation/quality/growth set",
    )


class PersonaArgs(BaseModel):
    """How a stock scores under legendary-investor frameworks: score (0-100),
    verdict (STRONG CONSIDER / WATCH / AVOID) and the written rationale."""

    message: str = Field(description=_STEP)
    ticker: str
    persona_id: PersonaId | None = Field(default=None, description="Omit for all five")


class ResearchArgs(BaseModel):
    """Forensic checks: growth quality, leverage, liquidity, return
    composition, promoter-holding and ROCE history, valuation gap and adverse
    news. Reports which checks the data cannot support."""

    message: str = Field(description=_STEP)
    ticker: str


class SectorArgs(BaseModel):
    """Sector aggregates: peer medians, leaders, signal breadth, news tone,
    and trailing price returns measured from NSE bars."""

    message: str = Field(description=_STEP)
    sector_name: str = Field(description="e.g. 'Pharmaceuticals', 'IT', 'Banks', 'Defence'")


class HoldingInput(BaseModel):
    ticker: str
    weight: float | None = Field(default=None, description="Portfolio weight in percent")


class PortfolioArgs(BaseModel):
    """Portfolio analysis. source='broker' reads the live demat holdings;
    'basket' analyses a Superinvesting model basket (omit basket to list all);
    'custom' analyses holdings the user typed. Reports sector concentration,
    weighted quality scores, flags, and for baskets the return series against
    the Nifty and a recomputation from NSE bars, plus which constituents the
    basket's own rebalancing rules would review."""

    message: str = Field(description=_STEP)
    source: Literal["broker", "basket", "custom"] = "broker"
    basket: str | None = Field(default=None, description="Basket name or id")
    holdings: list[HoldingInput] = Field(default_factory=list, max_length=40)


class SuggestionArgs(BaseModel):
    """Offer exactly three follow-up questions the user is likely to ask next.
    Call once, after the answer is written."""

    suggestions: list[str] = Field(min_length=3, max_length=3)


class MemoryArgs(BaseModel):
    """Desk ledger: prior debates, autopsies, scorecard family verdicts, holding
    flags and events. Call this before screening when the question is about a
    name this desk has already judged, or when the user asks what we remember."""

    message: str = Field(description=_STEP)
    query: str = Field(description="Free-text query, e.g. 'TCS forensic' or 'Bullish tags'")
    symbol: str | None = Field(default=None, description="Restrict to one NSE ticker")
    kind: str | None = Field(
        default=None,
        description="Optional: debate, trade_autopsy, scorecard_family, holding_flag, event",
    )


TOOL_ARGS: Final[dict[str, type[BaseModel]]] = {
    "screener": ScreenerArgs,
    "browser": BrowserArgs,
    "comparison": ComparisonArgs,
    "persona": PersonaArgs,
    "research": ResearchArgs,
    "sector": SectorArgs,
    "portfolio": PortfolioArgs,
    "memory": MemoryArgs,
    "suggestionTool": SuggestionArgs,
}

# --------------------------------------------------------------- holdings port


@dataclass(frozen=True, slots=True)
class HoldingRow:
    symbol: str
    quantity: int
    market_value: float


@dataclass(frozen=True, slots=True)
class HoldingsSnapshot:
    connected: bool
    detail: str = ""
    holdings: list[HoldingRow] = field(default_factory=list)


#: Injected by the caller. The broker lives above this layer, so the toolkit
#: is handed a way to read holdings rather than importing the execution stack.
HoldingsProvider = Callable[[], Awaitable[HoldingsSnapshot]]

#: Ledger lookup injected by the API. Agents cannot import persistence.
MemoryProvider = Callable[[str, str | None, str | None, int], Awaitable[list[dict[str, Any]]]]

# ------------------------------------------------------------------ helpers

_PERSONA_COLUMNS: Final[dict[str, str]] = {
    "warren-buffett-1": "warren_buffett",
    "peter-lynch-1": "peter_lynch",
    "rakesh-jhunjhunwala-1": "rakesh_jhunjhunwala",
    "100-bagger": "hundred_bagger",
    "canslim-1": "canslim",
}

_DEFAULT_COMPARE: Final[tuple[str, ...]] = (
    "market_cap_cr",
    "pe",
    "industry_pe",
    "peg",
    "pb",
    "roce",
    "roe",
    "opm_latest_quarter",
    "debt_to_equity",
    "sales_cagr_3y",
    "profit_cagr_3y",
    "fundamental_score",
    "valuation_score",
    "one_year_return_pct",
)

#: Lower is better for these; percentiles are flipped so 100 always means "best".
_LOWER_IS_BETTER: Final = {"pe", "peg", "pb", "debt_to_equity", "financial_leverage"}

#: Colloquial sector names -> the exchange classifications the data uses. The
#: literal match alone fails both ways: "IT" is a substring of "Hospital", and
#: "Energy" appears in no classification at all.
_SECTOR_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "it": ("Computers - Software & Consulting", "Software Products", "IT Enabled Services"),
    "technology": ("Computers - Software & Consulting", "Software Products", "IT Enabled Services"),
    "bank": ("Private Sector Bank", "Public Sector Bank", "Other Bank"),
    "nbfc": ("Non Banking Financial Company (NBFC)", "Finance", "Housing Finance Company"),
    "financial": (
        "Private Sector Bank",
        "Public Sector Bank",
        "Other Bank",
        "Non Banking Financial Company (NBFC)",
        "Finance",
        "Housing Finance Company",
        "Life Insurance",
        "Asset Management Company",
        "Stockbroking & Allied",
        "Exchange and Data Platform",
        "Financial Technology (Fintech)",
    ),
    "energy": (
        "Oil Exploration & Production",
        "Refineries & Marketing",
        "Gas Transmission/Marketing",
        "LPG/CNG/PNG/LNG Supplier",
        "Power Generation",
        "Integrated Power Utilities",
        "Power - Transmission",
        "Coal",
        "Lubricants",
    ),
    "oil": ("Oil Exploration & Production", "Refineries & Marketing", "Gas Transmission/Marketing"),
    "power": ("Power Generation", "Integrated Power Utilities", "Power - Transmission"),
    "renewable": ("Power Generation", "Integrated Power Utilities", "Other Electrical Equipment"),
    "pharma": ("Pharmaceuticals",),
    "healthcare": (
        "Pharmaceuticals",
        "Hospital",
        "Healthcare Service Provider",
        "Medical Equipment & Supplies",
        "Healthcare Research, Analytics & Technology",
    ),
    "hospital": ("Hospital", "Healthcare Service Provider"),
    "fmcg": (
        "Diversified FMCG",
        "Packaged Foods",
        "Personal Care",
        "Tea & Coffee",
        "Edible Oil",
        "Other Beverages",
        "Breweries & Distilleries",
    ),
    "auto": (
        "2/3 Wheelers",
        "Passenger Cars & Utility Vehicles",
        "Commercial Vehicles",
        "Auto Components & Equipments",
        "Tyres & Rubber Products",
        "Construction Vehicles",
    ),
    "defence": ("Aerospace & Defense", "Ship Building & Allied Services", "Explosives"),
    "defense": ("Aerospace & Defense", "Ship Building & Allied Services", "Explosives"),
    "consumer durables": (
        "Consumer Electronics",
        "Household Appliances",
        "Cables - Electricals",
        "Houseware",
    ),
    "industrials": (
        "Heavy Electrical Equipment",
        "Industrial Products",
        "Other Industrial Products",
        "Compressors, Pumps & Diesel Engines",
        "Civil Construction",
        "Castings & Forgings",
        "Other Electrical Equipment",
        "Electrodes & Refractories",
    ),
    "capital goods": (
        "Heavy Electrical Equipment",
        "Industrial Products",
        "Compressors, Pumps & Diesel Engines",
        "Other Electrical Equipment",
        "Castings & Forgings",
    ),
    "metals": ("Aluminium", "Iron & Steel", "Iron & Steel Products", "Diversified Metals", "Zinc"),
    "realty": ("Residential, Commercial Projects",),
    "real estate": ("Residential, Commercial Projects",),
    "chemicals": (
        "Specialty Chemicals",
        "Commodity Chemicals",
        "Pesticides & Agrochemicals",
        "Fertilizers",
    ),
    "retail": (
        "Diversified Retail",
        "Speciality Retail",
        "E-Retail/ E-Commerce",
        "Internet & Catalogue Retail",
    ),
    "qsr": ("Restaurants",),
    "telecom": ("Telecom - Cellular & Fixed line services",),
    "cement": ("Cement & Cement Products",),
    "textiles": ("Garments & Apparels", "Other Textile Products"),
    "insurance": ("Life Insurance",),
}


def match_sectors(query: str, known: list[str]) -> list[str]:
    """Classifications matching ``query``: alias, whole-word, then any-word."""
    lowered = query.strip().lower()
    if not lowered:
        return []
    chosen: list[str] = []
    for alias, targets in _SECTOR_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}s?\b", lowered):
            chosen.extend(t for t in targets if t in known)
    pattern = re.compile(rf"\b{re.escape(lowered)}", re.I)
    chosen.extend(s for s in known if pattern.search(s))
    if not chosen:
        words = [w for w in re.findall(r"[a-z]+", lowered) if len(w) > 3]
        chosen = [s for s in known if any(re.search(rf"\b{w}", s, re.I) for w in words)]
    return list(dict.fromkeys(chosen))


_FINANCIAL_SECTOR = re.compile(r"bank|financ|nbfc|insurance|lending|housing finance", re.I)


def _jsonable(value: Any) -> Any:
    """Plain JSON types, floats rounded, NaN/inf as ``None``."""
    if value is None:
        return None
    if isinstance(value, bool | str | int):
        return value
    if isinstance(value, float):
        return None if not math.isfinite(value) else round(value, 2)
    if isinstance(value, pd.Timestamp | datetime):
        # Minute precision is all a reader needs; microseconds and offsets
        # just get copied verbatim into answers.
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is not None:
            stamp = stamp.tz_convert(UTC)
        if (stamp.hour, stamp.minute, stamp.second) == (0, 0, 0):
            return stamp.date().isoformat()
        return stamp.strftime("%Y-%m-%d %H:%M UTC")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):  # numpy scalar
        return _jsonable(value.item())
    if pd.isna(value):
        return None
    return str(value)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Rows as plain dicts. ``itertuples`` loses column types under pandas-stubs."""
    return cast("list[dict[str, Any]]", frame.to_dict(orient="records"))


def _record(frame: pd.DataFrame) -> dict[str, Any] | None:
    return None if frame.empty else _jsonable(frame.iloc[0].to_dict())


def _median(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.median()) if len(clean) else None


def _percentile(series: pd.Series, value: float | None, *, lower_is_better: bool) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if value is None or len(clean) < 3:
        return None
    share = float((clean < value).mean() + 0.5 * (clean == value).mean())
    return round(100 * (1 - share if lower_is_better else share), 1)


class ResearchToolkit:
    """Executes tool calls against the lake and the platform stores."""

    def __init__(
        self,
        lake: SuperinvestingLake,
        bars: BarStore | None = None,
        fundamentals: FundamentalStore | None = None,
        holdings: HoldingsProvider | None = None,
        memory: MemoryProvider | None = None,
        *,
        today: Callable[[], date] | None = None,
    ) -> None:
        self._lake = lake
        self._bars = bars
        self._fundamentals = fundamentals
        self._holdings = holdings
        self._memory = memory
        self._today = today or (lambda: datetime.now(UTC).date())
        self._universe = self._build_universe()

    # ---------------------------------------------------------- dispatch

    async def run(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Validate ``arguments`` and execute ``name``. Never raises for bad input."""
        import asyncio

        schema = TOOL_ARGS.get(name)
        if schema is None:
            return {"error": f"unknown tool {name!r}"}
        try:
            args = schema.model_validate(arguments)
        except ValueError as exc:
            return {"error": f"invalid arguments: {exc}"[:600]}

        try:
            if isinstance(args, PortfolioArgs):
                result = await self.portfolio(args)
            elif isinstance(args, MemoryArgs):
                result = await self.memory(args)
            elif isinstance(args, SuggestionArgs):
                result = {"suggestions": args.suggestions}
            else:
                handlers: dict[str, Callable[[Any], dict[str, Any]]] = {
                    "screener": self.screener,
                    "browser": self.browser,
                    "comparison": self.comparison,
                    "persona": self.persona,
                    "research": self.research,
                    "sector": self.sector,
                }
                result = await asyncio.to_thread(handlers[name], args)
            return cast("dict[str, Any]", _jsonable(result))
        except Exception as exc:  # a tool failure is reported to the model, never raised
            log.exception("research_tool_failed", tool=name)
            return {"error": f"{type(exc).__name__}: {exc}"[:400]}

    # ---------------------------------------------------------- universe

    def _build_universe(self) -> pd.DataFrame:
        """One row per ticker: indicators (1,418) left-joined to deep coverage (284)."""
        indicators = self._lake.table("indicators").copy()
        companies = self._lake.table("companies")
        baselines = self._lake.table("baselines")
        personas = self._lake.table("personas")

        wide = companies.merge(baselines, on="ticker", how="left", suffixes=("", "_baseline"))
        if not personas.empty:
            pivot = personas.pivot_table(
                index="ticker", columns="persona_id", values="score", aggfunc="last"
            ).rename(columns=_PERSONA_COLUMNS)
            wide = wide.merge(pivot, left_on="ticker", right_index=True, how="left")

        indicators = indicators.rename(
            columns={
                "name": "indicator_name",
                "close": "indicator_close",
                "score": "trend_score",
                "sentiment": "signal_sentiment_sweep",
                "market_cap_cr": "market_cap_cr_sweep",
            }
        )
        merged = indicators.merge(wide, on="ticker", how="outer")
        merged["name"] = (
            merged["full_name"].fillna(merged.get("name")).fillna(merged["indicator_name"])
        )
        merged["market_cap_cr"] = merged["market_cap_cr"].fillna(merged["market_cap_cr_sweep"])
        merged["signal_sentiment"] = merged["signal_sentiment"].fillna(
            merged["signal_sentiment_sweep"]
        )
        merged["deep_coverage"] = merged["ticker"].isin(set(companies["ticker"]))
        return merged

    def _known_sectors(self) -> list[str]:
        return sorted(self._universe["sector"].dropna().unique().tolist())

    def _resolve(self, raw: str) -> str | None:
        ticker = self._lake.resolve(raw)
        if ticker is not None:
            return ticker
        upper = raw.strip().upper()
        if self._bars is not None and self._bars.has(upper):
            return upper
        return None

    # ---------------------------------------------------------- platform data

    def _technicals(self, ticker: str) -> dict[str, Any]:
        """Indicators computed from the platform's own NSE bars."""
        base: dict[str, Any] = {"source": "platform NSE bars (curated store)"}
        if self._bars is None or not self._bars.has(ticker):
            return {**base, "available": False, "reason": "no curated bars for this symbol"}
        bars = self._bars.read(ticker)
        if bars.empty:
            return {**base, "available": False, "reason": "bar file is empty"}
        close = bars["close"].astype(float)
        last_day = pd.Timestamp(bars.index[-1]).date()

        def trailing(sessions: int) -> float | None:
            if len(close) <= sessions:
                return None
            return (close.iloc[-1] / close.iloc[-1 - sessions] - 1) * 100

        row = score_symbol(ticker, bars)
        window = close.iloc[-252:]
        return {
            **base,
            "available": True,
            "as_of": last_day,
            "close": float(close.iloc[-1]),
            "return_1m_pct": trailing(21),
            "return_3m_pct": trailing(63),
            "return_6m_pct": trailing(126),
            "return_1y_pct": trailing(252),
            "high_52w": float(window.max()),
            "low_52w": float(window.min()),
            "rsi_14": row.rsi_14 if row else None,
            "sma_50": row.sma_50 if row else None,
            "sma_200": row.sma_200 if row else None,
            "above_sma_200": (row.price > row.sma_200 if row and row.sma_200 is not None else None),
            "momentum_20d_pct": row.momentum_20d_pct if row else None,
            "volume_ratio_20d": row.volume_ratio if row else None,
            "technical_scan_score": row.score if row else None,
            "technical_scan_band": row.band if row else None,
            "stale_days": (self._today() - last_day).days,
        }

    def _history(self, ticker: str, metric: Metric, years: int = 12) -> list[dict[str, Any]]:
        """The last ``years`` rows from the dated fundamentals store, oldest first.

        Rows are whatever periodicity the source filed -- the screener.in
        repair wrote quarterly shareholding -- so callers window by date, never
        by row count.
        """
        if self._fundamentals is None or not self._fundamentals.has(ticker):
            return []
        frame = self._fundamentals.read(ticker)
        frame = frame[frame["metric"] == str(metric)]
        if frame.empty:
            return []
        frame = frame.sort_values(["period_end", "published_at"]).drop_duplicates(
            "period_end", keep="last"
        )
        return [
            {
                "period_end": pd.Timestamp(r["period_end"]).date(),
                "value": float(r["value"]),
                "periodicity": r["periodicity"],
                "source": r["source"],
            }
            for r in _records(frame.tail(years))
        ]

    @staticmethod
    def _within_years(history: list[dict[str, Any]], years: int) -> list[dict[str, Any]]:
        """Rows whose period ends within ``years`` of the latest one."""
        if not history:
            return []
        start = history[-1]["period_end"] - timedelta(days=round(365.25 * years))
        return [row for row in history if row["period_end"] > start]

    def _lake_as_of(self) -> dict[str, Any]:
        header: dict[str, Any] = {
            "source": "Superinvesting snapshot",
            "as_of": self._lake.manifest.get("as_of", {}),
            "as_of_note": (
                "fundamentals_exported dates baselines, ratios, scores and snapshot prices; "
                "personas dates the persona cards; news and indicators date those feeds"
            ),
        }
        # Every result carries the measured record of each signal family, so
        # the model can weigh a news tag or a broker call whichever tool
        # surfaced it -- not only when it happened to call the right one.
        header["signal_track_record"] = self._lake.track_record() or (
            "not yet measured; run `trading-agent research scorecard`"
        )
        return header

    # ------------------------------------------------------------- screener

    def screener(self, args: ScreenerArgs) -> dict[str, Any]:
        if args.tickers:
            return {"mode": "profile", "profiles": [self._profile(t) for t in args.tickers]}
        return self._screen(args)

    def _profile(self, raw: str) -> dict[str, Any]:
        ticker = self._resolve(raw)
        if ticker is None:
            return {"query": raw, "error": "not found in the Superinvesting universe or NSE bars"}
        row = self._universe[self._universe["ticker"] == ticker]
        profile: dict[str, Any] = {"ticker": ticker}
        if not row.empty:
            record = row.iloc[0]
            deep = bool(record["deep_coverage"])
            keep = [
                "name",
                "sector",
                "capitalization",
                "market_cap_cr",
                "price",
                "pe",
                "industry_pe",
                "peg",
                "pb",
                "intrinsic_value",
                "roce",
                "roe",
                "opm_latest_quarter",
                "npm_latest_quarter",
                "debt_to_equity",
                "financial_leverage",
                "interest_coverage",
                "current_ratio",
                "sales_growth_1y",
                "sales_cagr_3y",
                "sales_cagr_5y",
                "profit_growth_1y",
                "profit_cagr_3y",
                "profit_cagr_5y",
                "eps_growth_3y_pct",
                "one_year_return_pct",
                "three_year_return_pct",
                "fundamental_score",
                "business_score",
                "valuation_score",
                "technical_score",
                "forensic_score",
                "forensic_positive",
                "forensic_negative",
                "analyst_target",
                "analyst_organization",
                "analyst_date",
                "signal_sentiment",
                "trend_score",
                "superinvesting_score",
                "days_since_signal",
                "price_change_since_signal_pct",
                *_PERSONA_COLUMNS.values(),
            ]
            profile["superinvesting"] = {
                **self._lake_as_of(),
                "deep_coverage": deep,
                **{k: record.get(k) for k in keep if k in record},
            }
            if not deep:
                profile["superinvesting"]["note"] = (
                    "Only the technical-indicator sweep covers this stock; no fundamentals, "
                    "baselines or persona scores in the snapshot."
                )
            companies = self._lake.table("companies")
            description = companies[companies["ticker"] == ticker]
            if not description.empty and description.iloc[0]["description"]:
                profile["business_description"] = str(description.iloc[0]["description"])[:900]
        profile["technicals"] = self._technicals(ticker)
        profile["roce_history"] = self._history(ticker, Metric.RETURN_ON_CAPITAL)
        profile["promoter_holding_history"] = self._history(ticker, Metric.PROMOTER_HOLDING)
        return profile

    def _screen(self, args: ScreenerArgs) -> dict[str, Any]:
        frame = self._universe
        applied: list[dict[str, Any]] = []
        if args.sectors:
            known = self._known_sectors()
            chosen = list(dict.fromkeys(m for q in args.sectors for m in match_sectors(q, known)))
            if not chosen:
                return {"error": f"no sector matches {args.sectors}", "known_sectors": known}
            frame = frame[frame["sector"].isin(chosen)]
            applied.append({"sectors": chosen, "remaining": len(frame)})
        if args.capitalization:
            frame = frame[frame["capitalization"] == args.capitalization]
            applied.append({"capitalization": args.capitalization, "remaining": len(frame)})

        unknown = [f.field for f in args.filters if f.field not in frame.columns]
        if unknown:
            return {"error": f"unknown screen fields: {unknown}"}

        for spec in args.filters:
            column = frame[spec.field]
            present = column.notna()
            if isinstance(spec.value, str):
                values = column.astype(str).str.lower()
                target = spec.value.lower()
                mask = values == target if spec.op == "==" else values != target
            else:
                numeric = pd.to_numeric(column, errors="coerce")
                mask = {
                    ">": numeric > spec.value,
                    ">=": numeric >= spec.value,
                    "<": numeric < spec.value,
                    "<=": numeric <= spec.value,
                    "==": numeric == spec.value,
                    "!=": numeric != spec.value,
                }[spec.op]
            missing = int((~present).sum())
            frame = frame[mask & present]
            applied.append(
                {
                    "filter": f"{spec.field} {spec.op} {spec.value}",
                    "excluded_for_missing_data": missing,
                    "remaining": len(frame),
                }
            )

        sort_by = args.sort_by if args.sort_by in frame.columns else None
        if sort_by is None:
            sort_by = (
                "fundamental_score" if frame["fundamental_score"].notna().any() else "market_cap_cr"
            )
        frame = frame.sort_values(sort_by, ascending=not args.descending, na_position="last")

        columns = [
            "ticker",
            "name",
            "sector",
            "capitalization",
            "market_cap_cr",
            "pe",
            "industry_pe",
            "peg",
            "roce",
            "roe",
            "debt_to_equity",
            "opm_latest_quarter",
            "sales_cagr_3y",
            "profit_cagr_3y",
            "fundamental_score",
            "valuation_score",
            "signal_sentiment",
            "superinvesting_score",
        ]
        extra = [f.field for f in args.filters if f.field not in columns]
        if sort_by not in columns:
            extra.append(sort_by)
        rows = frame[[c for c in [*columns, *extra] if c in frame.columns]].head(args.limit)
        return {
            "mode": "screen",
            **self._lake_as_of(),
            "universe_size": len(self._universe),
            "deep_coverage_size": int(self._universe["deep_coverage"].sum()),
            "steps": applied,
            "matches": len(frame),
            "sorted_by": sort_by,
            "results": rows.to_dict(orient="records"),
            "caveat": (
                "Fundamentals exist for the 284 deep-coverage names only; the other ~1,130 "
                "carry indicator fields. Multi-year consistency criteria (e.g. 'ROCE > 15% "
                "for 5 years') can be checked only against roce_history via profile mode."
            ),
        }

    # -------------------------------------------------------------- browser

    def browser(self, args: BrowserArgs) -> dict[str, Any]:
        news = self._lake.table("news")
        tickers = [t for t in (self._resolve(t) for t in args.tickers) if t]
        cutoff = pd.Timestamp(self._today() - timedelta(days=args.days), tz=UTC)
        frame = news[news["published_at"] >= cutoff]
        if tickers:
            frame = frame[frame["ticker"].isin(tickers)]
        if args.direction:
            frame = frame[frame["impact_direction"] == args.direction]

        terms = [t for t in re.findall(r"[a-z0-9&]+", args.query.lower()) if len(t) > 2]
        if terms and not frame.empty:
            haystack = (
                frame["title"].fillna("")
                + " "
                + frame["summary"].fillna("")
                + " "
                + frame["highlights"].fillna("")
            ).str.lower()
            score = sum(haystack.str.count(re.escape(term)) for term in terms)
            frame = frame.assign(_relevance=score)
            if not tickers:
                frame = frame[frame["_relevance"] > 0]
            frame = frame.sort_values(["_relevance", "published_at"], ascending=[False, False])

        deduped = frame.drop_duplicates(subset=["title"])
        items = deduped.head(args.limit)[
            [
                "ticker",
                "published_at",
                "title",
                "impact_direction",
                "impact_scale",
                "summary",
                "source",
                "link",
            ]
        ]
        result: dict[str, Any] = {
            "source": "Superinvesting news feed (ingested)",
            "feed_covers": {
                "from": news["published_at"].min(),
                "to": news["published_at"].max(),
            },
            "matched": len(deduped),
            "tone": deduped["impact_direction"].value_counts().to_dict(),
            "items": items.to_dict(orient="records"),
        }
        if args.live_financials and tickers:
            result["live_financials"] = [self._live_financials(t) for t in tickers[:3]]
        return result

    @staticmethod
    def _live_financials(ticker: str) -> dict[str, Any]:
        from trading_agent.market_data.screener_source import ScreenerSource

        try:
            fetched = ScreenerSource(timeout=12.0).fetch(ticker)
        except Exception as exc:
            return {"ticker": ticker, "error": f"screener.in fetch failed: {exc}"[:200]}
        by_metric: dict[str, list[dict[str, Any]]] = {}
        for fact in sorted(fetched.facts, key=lambda f: f.period_end):
            by_metric.setdefault(f"{fact.metric}_{fact.periodicity}", []).append(
                {"period_end": fact.period_end, "value": float(fact.value)}
            )
        return {
            "ticker": ticker,
            "source": "screener.in (live fetch)",
            "company": fetched.company_name,
            "fiscal_year_end_month": fetched.fiscal_year_end_month,
            "series": {k: v[-8:] for k, v in by_metric.items()},
        }

    # ----------------------------------------------------------- comparison

    def comparison(self, args: ComparisonArgs) -> dict[str, Any]:
        metrics = args.metrics or list(_DEFAULT_COMPARE)
        unknown = [m for m in metrics if m not in self._universe.columns]
        metrics = [m for m in metrics if m in self._universe.columns]
        deep = self._universe[self._universe["deep_coverage"]]

        rows: list[dict[str, Any]] = []
        for raw in args.tickers:
            ticker = self._resolve(raw)
            match = self._universe[self._universe["ticker"] == ticker] if ticker else pd.DataFrame()
            if match.empty:
                rows.append({"query": raw, "error": "not in the covered universe"})
                continue
            record = match.iloc[0]
            peers = deep[deep["sector"] == record["sector"]]
            entry: dict[str, Any] = {
                "ticker": ticker,
                "name": record["name"],
                "sector": record["sector"],
                "sector_peer_count": len(peers),
                "metrics": {},
            }
            for metric in metrics:
                value = record.get(metric)
                numeric = None if value is None or pd.isna(value) else value
                if isinstance(numeric, str):
                    entry["metrics"][metric] = {"value": numeric}
                    continue
                lower = metric in _LOWER_IS_BETTER
                entry["metrics"][metric] = {
                    "value": numeric,
                    "sector_median": _median(peers[metric]),
                    "sector_percentile": _percentile(peers[metric], numeric, lower_is_better=lower),
                    "universe_percentile": _percentile(
                        deep[metric], numeric, lower_is_better=lower
                    ),
                }
            rows.append(entry)

        return {
            **self._lake_as_of(),
            "percentile_note": (
                "0-100 where 100 is best; for pe, peg, pb, debt_to_equity and "
                "financial_leverage lower counts as better"
            ),
            "unknown_metrics": unknown,
            "companies": rows,
        }

    # -------------------------------------------------------------- persona

    def persona(self, args: PersonaArgs) -> dict[str, Any]:
        ticker = self._resolve(args.ticker)
        if ticker is None:
            return {"error": f"{args.ticker!r} not found"}
        cards = self._lake.table("personas")
        cards = cards[cards["ticker"] == ticker]
        if args.persona_id:
            cards = cards[cards["persona_id"] == args.persona_id]
        if cards.empty:
            return {"ticker": ticker, "error": "no persona scores in the snapshot for this stock"}

        companies = self._lake.table("companies")
        company = companies[companies["ticker"] == ticker]
        constituents = self._lake.table("basket_constituents")
        baskets = self._lake.table("baskets")
        held_in = baskets[
            baskets["basket_id"].isin(constituents[constituents["ticker"] == ticker]["basket_id"])
        ]
        return {
            **self._lake_as_of(),
            "ticker": ticker,
            "pillars": _record(
                company[
                    ["fundamental_score", "business_score", "valuation_score", "technical_score"]
                ]
            ),
            "cards": [
                {
                    "persona": PERSONAS.get(card["persona_id"], (card["persona_id"], ""))[0],
                    "philosophy": PERSONAS.get(card["persona_id"], ("", ""))[1],
                    "persona_id": card["persona_id"],
                    "score": card["score"],
                    "verdict": card["verdict"],
                    "scored_at": card["updated_at"],
                    "rationale": (card["narrative"] or "")[:1800],
                }
                for card in _records(cards.sort_values("score", ascending=False))
            ],
            "in_model_baskets": held_in[["name", "persona_ids"]].to_dict(orient="records"),
        }

    # ------------------------------------------------------------- research

    def research(self, args: ResearchArgs) -> dict[str, Any]:
        ticker = self._resolve(args.ticker)
        if ticker is None:
            return {"error": f"{args.ticker!r} not found"}
        row = self._universe[self._universe["ticker"] == ticker]
        if row.empty or not bool(row.iloc[0]["deep_coverage"]):
            return {
                "ticker": ticker,
                "error": (
                    "no fundamentals in the snapshot for this stock; forensic checks need them"
                ),
                "technicals": self._technicals(ticker),
            }
        r = row.iloc[0]
        financial = bool(_FINANCIAL_SECTOR.search(str(r["sector"] or "")))
        checks: list[dict[str, Any]] = []

        def check(name: str, value: Any, flag: bool | None, rule: str, note: str) -> None:
            present = value is not None and not (isinstance(value, float) and pd.isna(value))
            checks.append(
                {
                    "check": name,
                    "value": value if present else None,
                    "rule": rule,
                    "status": "unavailable"
                    if not present or flag is None
                    else ("flag" if flag else "pass"),
                    "concern_if_flagged": note,
                }
            )

        def val(key: str) -> float | None:
            v = r.get(key)
            return None if v is None or pd.isna(v) else float(v)

        for horizon in ("3y", "5y"):
            profit, sales = val(f"profit_cagr_{horizon}"), val(f"sales_cagr_{horizon}")
            gap = profit - sales if profit is not None and sales is not None else None
            check(
                f"profit_vs_sales_growth_{horizon}",
                gap,
                None if gap is None else gap > 15,
                "profit CAGR exceeds sales CAGR by more than 15 points",
                "Profit outrunning revenue for years needs a margin, tax or other-income "
                "explanation; it is where aggressive accounting usually shows first.",
            )
        roe, roce = val("roe"), val("roce")
        spread = roe - roce if roe is not None and roce is not None else None
        check(
            "roe_minus_roce",
            spread,
            None if spread is None or financial else spread > 5,
            "ROE above ROCE by more than 5 points (non-financials)",
            "Returns flattered by leverage rather than by the business itself.",
        )
        de = val("debt_to_equity")
        check(
            "debt_to_equity",
            de,
            None if de is None or financial else de > 1.0,
            "D/E above 1.0 (non-financials)",
            "Balance-sheet leverage.",
        )
        coverage = val("interest_coverage")
        check(
            "interest_coverage",
            coverage,
            None if coverage is None or financial else coverage < 3,
            "EBIT covers interest fewer than 3 times",
            "Debt-service strain.",
        )
        current = val("current_ratio")
        check(
            "current_ratio",
            current,
            None if current is None or financial else current < 1.0,
            "current ratio below 1.0",
            "Short-term liabilities exceed short-term assets.",
        )
        opm, npm = val("opm_latest_quarter"), val("npm_latest_quarter")
        check(
            "net_margin_above_operating_margin",
            None if opm is None or npm is None else npm - opm,
            None if opm is None or npm is None or financial else npm > opm,
            "net margin above operating margin in the latest quarter",
            "Profit arriving from below the operating line: other income, one-offs or tax credits.",
        )
        price, intrinsic = val("price"), val("intrinsic_value")
        premium = (price / intrinsic - 1) * 100 if price and intrinsic else None
        check(
            "price_vs_intrinsic_value_pct",
            premium,
            None if premium is None else premium > 50,
            "price more than 50% above Superinvesting's intrinsic value",
            "Valuation gap; the intrinsic-value method is the source's, not verified here.",
        )

        promoter = self._within_years(self._history(ticker, Metric.PROMOTER_HOLDING, years=40), 5)
        if len(promoter) >= 2:
            change = promoter[-1]["value"] - promoter[0]["value"]
            span = (promoter[-1]["period_end"] - promoter[0]["period_end"]).days / 365.25
            check(
                "promoter_holding_change_pts",
                change,
                change < -5,
                "promoter stake down more than 5 points over up to 5 years",
                f"{promoter[0]['value']:g}% on {promoter[0]['period_end']} to "
                f"{promoter[-1]['value']:g}% on {promoter[-1]['period_end']} "
                f"({span:.1f} years of data).",
            )
        else:
            check(
                "promoter_holding_change_pts",
                None,
                None,
                "needs 2+ values",
                "No history in the fundamentals store.",
            )

        roce_hist = self._within_years(self._history(ticker, Metric.RETURN_ON_CAPITAL, years=40), 5)
        recent = [p["value"] for p in roce_hist]
        if len(recent) >= 3:
            check(
                "roce_min_last_5y",
                min(recent),
                min(recent) < 15,
                "ROCE fell below 15% in any of the last 5 years",
                f"Values oldest→newest: {', '.join(f'{v:g}' for v in recent)}.",
            )
        else:
            check(
                "roce_min_last_5y",
                None,
                None,
                "needs 3+ annual values",
                "No history in the fundamentals store.",
            )

        news = self._lake.table("news")
        cutoff = pd.Timestamp(self._today() - timedelta(days=90), tz=UTC)
        adverse = news[
            (news["ticker"] == ticker)
            & (news["published_at"] >= cutoff)
            & (news["impact_direction"] == "BEARISH")
        ]
        flagged = [c["check"] for c in checks if c["status"] == "flag"]
        computed = None
        try:
            from trading_agent.features.statement_snapshot import auto_reports, load_snapshot

            snap = load_snapshot(
                ticker,
                self._today(),
                store=self._fundamentals,
                bars=self._bars,
                lake=self._lake,
            )
            forensic = next((rep for rep in auto_reports(snap) if rep.engine == "forensics"), None)
            if forensic is not None:
                computed = {
                    "available": forensic.available,
                    "missing": forensic.missing,
                    "reason": forensic.reason,
                    "result": forensic.result,
                }
        except Exception as exc:
            computed = {"available": False, "reason": str(exc)[:200]}
        return {
            **self._lake_as_of(),
            "ticker": ticker,
            "sector": r["sector"],
            "financial_sector_rules": financial,
            "source_forensic_card": {
                "score": r.get("forensic_score"),
                "positive": r.get("forensic_positive"),
                "negative": r.get("forensic_negative"),
            },
            "checks": checks,
            "flags": flagged,
            "bearish_news_90d": adverse.head(5)[["published_at", "title", "impact_scale"]].to_dict(
                orient="records"
            ),
            "computed_forensics": computed,
            "not_in_data": [
                "promoter share pledging",
                "auditor qualifications or resignations",
                "contingent liabilities",
                "related-party transactions",
                "receivable and inventory day trends",
            ],
        }

    # --------------------------------------------------------------- sector

    def sector(self, args: SectorArgs) -> dict[str, Any]:
        deep = self._universe[self._universe["deep_coverage"]]
        sectors = self._known_sectors()
        chosen = match_sectors(args.sector_name, sectors)
        if not chosen:
            return {"error": f"no sector matches {args.sector_name!r}", "known_sectors": sectors}

        members = deep[deep["sector"].isin(chosen)]
        tickers = members["ticker"].tolist()
        medians = {
            metric: _median(members[metric])
            for metric in (
                "pe",
                "industry_pe",
                "peg",
                "pb",
                "roce",
                "roe",
                "opm_latest_quarter",
                "debt_to_equity",
                "sales_cagr_3y",
                "profit_cagr_3y",
                "one_year_return_pct",
                "fundamental_score",
                "valuation_score",
            )
        }

        returns: list[float] = []
        for ticker in tickers:
            tech = self._technicals(ticker)
            if tech.get("available") and tech.get("return_3m_pct") is not None:
                returns.append(tech["return_3m_pct"])

        news = self._lake.table("news")
        cutoff = pd.Timestamp(self._today() - timedelta(days=30), tz=UTC)
        recent = news[(news["ticker"].isin(tickers)) & (news["published_at"] >= cutoff)]
        lead = [
            "ticker",
            "name",
            "market_cap_cr",
            "pe",
            "roce",
            "sales_cagr_3y",
            "fundamental_score",
        ]
        return {
            **self._lake_as_of(),
            "matched_sectors": chosen,
            "companies": len(members),
            "total_market_cap_cr": float(members["market_cap_cr"].sum()),
            "medians": medians,
            "largest": members.sort_values("market_cap_cr", ascending=False)[lead]
            .head(6)
            .to_dict(orient="records"),
            "highest_fundamental_score": members.sort_values("fundamental_score", ascending=False)[
                lead
            ]
            .head(6)
            .to_dict(orient="records"),
            "signal_breadth": members["signal_sentiment"].value_counts().to_dict(),
            "news_tone_30d": recent["impact_direction"].value_counts().to_dict(),
            "recent_headlines": recent.drop_duplicates("title")
            .head(5)[["ticker", "published_at", "title", "impact_direction"]]
            .to_dict(orient="records"),
            "price_return_3m_from_nse_bars": {
                "source": "platform NSE bars",
                "measured": len(returns),
                "median_pct": float(pd.Series(returns).median()) if returns else None,
                "share_positive": round(sum(r > 0 for r in returns) / len(returns), 2)
                if returns
                else None,
            },
            "coverage_note": (
                f"Aggregates cover the {len(members)} deep-coverage names in the snapshot, "
                "not every listed company in the sector. No TAM, policy or supply-chain data "
                "is ingested."
            ),
        }

    # ------------------------------------------------------------ portfolio

    async def portfolio(self, args: PortfolioArgs) -> dict[str, Any]:
        import asyncio

        if args.source == "basket":
            return await asyncio.to_thread(self._basket, args.basket)

        if args.source == "custom":
            if not args.holdings:
                return {"error": "source 'custom' needs holdings"}
            weights = {h.ticker: h.weight for h in args.holdings}
            if any(w is None for w in weights.values()):
                weights = dict.fromkeys(weights, 100 / len(weights))
            return await asyncio.to_thread(
                self._analyse_holdings, weights, "user-supplied holdings"
            )

        if self._holdings is None:
            return {"error": "broker holdings are not wired into this session"}
        snapshot = await self._holdings()
        if not snapshot.connected:
            return {"connected": False, "detail": snapshot.detail or "broker session is not live"}
        total = sum(h.market_value for h in snapshot.holdings)
        if not snapshot.holdings or total <= 0:
            return {"connected": True, "holdings": [], "note": "no open holdings"}
        weights = {h.symbol: 100 * h.market_value / total for h in snapshot.holdings}
        result = await asyncio.to_thread(self._analyse_holdings, weights, "live broker holdings")
        result["invested_value"] = total
        return result

    async def memory(self, args: MemoryArgs) -> dict[str, Any]:
        if self._memory is None:
            return {
                "available": False,
                "reason": "Desk ledger is not wired into this session.",
                "hits": [],
            }
        hits = await self._memory(args.query, args.symbol, args.kind, 8)
        return {
            "available": True,
            "query": args.query,
            "symbol": args.symbol,
            "hits": hits,
            "note": (
                "These are dated ledger rows this desk wrote (debates, autopsies, "
                "scorecard families, holding flags, events). Not a language-model memory."
            ),
        }

    def _analyse_holdings(self, weights: dict[str, float | None], source: str) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        unresolved: list[str] = []
        for raw, weight in weights.items():
            ticker = self._resolve(raw)
            if ticker is None:
                unresolved.append(raw)
                continue
            match = self._universe[self._universe["ticker"] == ticker]
            record = match.iloc[0].to_dict() if not match.empty else {}
            tech = self._technicals(ticker)
            rows.append(
                {
                    "ticker": ticker,
                    "weight_pct": weight,
                    "sector": record.get("sector"),
                    "fundamental_score": record.get("fundamental_score"),
                    "valuation_score": record.get("valuation_score"),
                    "forensic_score": record.get("forensic_score"),
                    "signal_sentiment": record.get("signal_sentiment"),
                    "return_3m_pct": tech.get("return_3m_pct"),
                    "above_sma_200": tech.get("above_sma_200"),
                }
            )
        frame = pd.DataFrame(rows)
        if frame.empty:
            return {
                "source": source,
                "error": "no holding could be matched",
                "unresolved": unresolved,
            }

        by_sector = (
            frame.assign(sector=frame["sector"].fillna("Unclassified"))
            .groupby("sector")["weight_pct"]
            .sum()
            .sort_values(ascending=False)
        )

        def weighted(column: str) -> dict[str, Any]:
            present = frame[frame[column].notna()]
            covered = float(present["weight_pct"].sum())
            if present.empty or covered <= 0:
                return {"value": None, "weight_covered_pct": 0.0}
            value = float((present[column] * present["weight_pct"]).sum() / covered)
            return {"value": value, "weight_covered_pct": covered}

        flags: list[str] = []
        if len(by_sector) and by_sector.iloc[0] > 30:
            flags.append(f"{by_sector.index[0]} is {by_sector.iloc[0]:.1f}% of the book")
        top = frame.sort_values("weight_pct", ascending=False).iloc[0]
        if top["weight_pct"] > 15:
            flags.append(f"{top['ticker']} alone is {top['weight_pct']:.1f}%")
        for row in _records(frame):
            if row["signal_sentiment"] == "Bearish" and row["above_sma_200"] is False:
                flags.append(f"{row['ticker']}: bearish trend signal and below its 200-day average")
            forensic = row["forensic_score"]
            if forensic is not None and not pd.isna(forensic) and forensic < 50:
                flags.append(f"{row['ticker']}: forensic score {forensic:g} (< 50)")
        return {
            "source": source,
            **self._lake_as_of(),
            "positions": len(frame),
            "unresolved": unresolved,
            "sector_weights_pct": by_sector.to_dict(),
            "weighted_fundamental_score": weighted("fundamental_score"),
            "weighted_valuation_score": weighted("valuation_score"),
            "weighted_forensic_score": weighted("forensic_score"),
            "flags": flags,
            "holdings": frame.to_dict(orient="records"),
        }

    def _basket(self, query: str | None) -> dict[str, Any]:
        baskets = self._lake.table("baskets")
        if not query:
            listing = baskets[
                [
                    "name",
                    "category",
                    "persona_ids",
                    "created_at",
                    "rebalanced_at",
                    "reported_return_pct",
                ]
            ].assign(
                created_at=baskets["created_at"].dt.date,
                rebalanced_at=baskets["rebalanced_at"].dt.date,
            )
            return {
                **self._lake_as_of(),
                "baskets": listing.sort_values("reported_return_pct", ascending=False).to_dict(
                    orient="records"
                ),
            }

        lowered = query.lower()
        match = baskets[(baskets["basket_id"] == query) | (baskets["name"].str.lower() == lowered)]
        if match.empty:
            match = baskets[baskets["name"].str.lower().str.contains(lowered, regex=False)]
        if match.empty:
            return {"error": f"no basket matches {query!r}", "baskets": baskets["name"].tolist()}
        basket = match.iloc[0]
        basket_id = basket["basket_id"]

        returns = self._lake.table("basket_returns")
        constituents = self._lake.table("basket_constituents")
        members = constituents[constituents["basket_id"] == basket_id]
        series = returns[returns["basket_id"] == basket_id].sort_values("date")
        nifty = self._lake.table("benchmark_nifty")

        compounded = (
            float(((1 + series["daily_return_pct"] / 100).prod() - 1) * 100)
            if len(series)
            else None
        )
        vs_nifty: dict[str, Any] = {"value": None}
        if len(series):
            window = nifty[
                (nifty["date"] >= series["date"].min()) & (nifty["date"] <= series["date"].max())
            ]
            if len(window) >= 0.8 * len(series):
                nifty_return = float(((1 + window["daily_return_pct"] / 100).prod() - 1) * 100)
                vs_nifty = {
                    "window": [series["date"].min(), series["date"].max()],
                    "basket_pct": compounded,
                    "nifty_pct": nifty_return,
                    "excess_pct": compounded - nifty_return if compounded is not None else None,
                }
            else:
                vs_nifty["reason"] = "Nifty series does not cover the basket's window"

        replication = self._lake.basket_replication(str(basket_id))
        recomputed = (
            {
                "source": "basket replication (rebalance engine, NSE bars, delivery charges)",
                "findings": replication["findings"],
                "windows": replication["windows"],
                "series_definition": replication["series_definition"],
            }
            if replication is not None
            else self._recompute_from_bars(members["ticker"].tolist(), series)
        )

        universe = self._universe.set_index("ticker")
        review: list[dict[str, Any]] = []
        for member in members.itertuples():
            info = universe.loc[member.ticker] if member.ticker in universe.index else None
            sentiment = None if info is None else info.get("signal_sentiment")
            composite = None if info is None else info.get("superinvesting_score")
            reasons = []
            if sentiment == "Bearish":
                reasons.append("trend signal flipped Bearish")
            if composite is not None and not pd.isna(composite) and composite < 60:
                reasons.append(f"composite score {composite:g} below 60")
            review.append(
                {
                    "ticker": member.ticker,
                    "basket_score": member.score,
                    "signal_sentiment": sentiment,
                    "composite_score": composite,
                    "exit_rule_triggered": len(reasons) == 2,
                    "reasons": reasons,
                }
            )
        return {
            **self._lake_as_of(),
            "basket": basket[
                [
                    "name",
                    "category",
                    "description",
                    "persona_ids",
                    "created_at",
                    "rebalanced_at",
                    "reported_return_pct",
                ]
            ].to_dict(),
            "constituents": len(members),
            "series_sessions": len(series),
            "compounded_daily_series_pct": compounded,
            "vs_nifty": vs_nifty,
            "replication": recomputed,
            "rebalance_review": {
                "rule": (
                    "From the basket engine spec: a constituent is replaced when its trend "
                    "signal is Bearish AND its composite score is below 60."
                ),
                "constituents": review,
            },
        }

    def _recompute_from_bars(self, tickers: list[str], series: pd.DataFrame) -> dict[str, Any]:
        """Equal-weight return of today's constituents over the series window.

        A check on the reported number, not a replacement: it uses current
        members for the whole window, so a basket that rebalanced will differ.
        """
        if self._bars is None or series.empty:
            return {"value": None, "reason": "no bars or no series"}
        start, end = series["date"].min(), series["date"].max()
        legs: dict[str, float] = {}
        for ticker in tickers:
            if not self._bars.has(ticker):
                continue
            close = self._bars.read(ticker)["close"].astype(float)
            days = pd.DatetimeIndex(close.index).date
            before = close[days < start]
            within = close[days <= end]
            if before.empty or within.empty:
                continue
            legs[ticker] = (within.iloc[-1] / before.iloc[-1] - 1) * 100
        if not legs:
            return {"value": None, "reason": "no constituent has bars covering the window"}
        return {
            "source": "platform NSE bars, equal weight, current constituents",
            "window": [start, end],
            "value_pct": sum(legs.values()) / len(legs),
            "constituents_measured": len(legs),
            "constituents_total": len(tickers),
            "per_stock_pct": legs,
            "caveat": (
                "Ignores rebalances inside the window, so it will not match a rebalanced basket."
            ),
        }
