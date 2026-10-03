"""Replicate a Superinvesting model basket from NSE bars, with real costs.

A basket page shows one number -- "+21.02% overall" -- beside a Nifty line.
Three things sit behind that number that it does not disclose, and this module
measures each:

1. **What the series is.** Where our bars cover every constituent, the
   reported daily returns are reproduced exactly by an equal-weighted
   close-to-close average, i.e. a portfolio re-weighted to equal every day, at
   no cost. The Nifty comparison is a *different* definition: its daily figure
   is the index's open-to-close move, which omits every overnight gap. Both
   facts are checked per basket rather than assumed.
2. **What an investor would have earned.** The rebalance engine buys the
   constituents at the next open, holds them, and pays delivery charges on
   every fill -- against a Nifty measured close-to-close, the way an index fund
   would have moved.
3. **How much of it is hindsight.** Only today's constituents are known. Since
   the last rebalance they are exactly what the basket held, so that window is
   an honest replication. Over the full history they are the names that
   survived selection, which flatters any backtest; that window is reported,
   and labelled as such.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass, field
from datetime import date
from decimal import ROUND_DOWN, Decimal
from functools import partial
from typing import Any, Final, cast

import pandas as pd

from trading_agent.backtest.rebalance import run_rebalance_backtest
from trading_agent.costs.calculator import ChargeCalculator, UnverifiedScheduleWarning
from trading_agent.domain.money import Rupees
from trading_agent.market_data.corporate_actions import CorporateAction, adjust_for_actions
from trading_agent.market_data.store import BarStore
from trading_agent.market_data.superinvesting_lake import SuperinvestingLake
from trading_agent.observability import get_logger

__all__ = [
    "REPLICATIONS_FILE",
    "BasketReplication",
    "WindowResult",
    "replicate_all",
    "replicate_basket",
]

log = get_logger(__name__)

BENCHMARK: Final = "NIFTY50"
#: Written beside the lake so the research desk can quote replications
#: without importing the backtest layer.
REPLICATIONS_FILE: Final = "basket_replications.json"
#: Sessions a window needs before its numbers mean anything.
MIN_SESSIONS: Final = 3
#: Correlation at which a recomputed series counts as reproducing the source.
REPRODUCED_CORRELATION: Final = 0.99


@dataclass(frozen=True, slots=True)
class CurvePoint:
    date: date
    reported_pct: float | None
    costless_pct: float | None
    investable_pct: float | None
    nifty_pct: float | None


@dataclass(slots=True)
class WindowResult:
    key: str
    label: str
    start: date
    end: date
    sessions: int
    hindsight: bool
    reported_pct: float | None
    #: The source's Nifty figure over the window (open-to-close compounded).
    source_nifty_pct: float | None
    #: The Nifty close-to-close from our bars, what an index fund tracked.
    nifty_pct: float | None
    #: Equal weight, re-weighted daily, no costs: the source's own method.
    costless_pct: float | None
    #: Bought at the next open and held, net of buy charges and the estimated
    #: charges to sell at the end -- a reported return has no exit to pay for.
    investable_pct: float | None
    charges_rupees: float | None
    #: Of which estimated for the final sale.
    exit_charges_rupees: float | None
    constituents_measured: int
    constituents_total: int
    charges_pct_of_capital: float | None
    max_drawdown_pct: float | None
    note: str
    curve: list[CurvePoint] = field(default_factory=list)


@dataclass(slots=True)
class BasketReplication:
    basket_id: str
    name: str
    category: str | None
    description: str | None
    persona_ids: list[str]
    created_at: str | None
    rebalanced_at: str | None
    reported_overall_pct: float | None
    constituents: list[dict[str, Any]]
    missing_bars: list[str]
    capital_rupees: float
    series_definition: dict[str, Any]
    windows: list[WindowResult]
    findings: list[str]

    def to_json(self) -> dict[str, Any]:
        return cast("dict[str, Any]", _clean(asdict(self)))

    def summary(self) -> dict[str, Any]:
        """Everything but the daily curves: small enough for a list or a prompt."""
        payload = self.to_json()
        for window in payload["windows"]:
            window.pop("curve", None)
        return payload


def _clean(value: Any) -> Any:
    if isinstance(value, float):
        return None if not math.isfinite(value) else round(value, 4)
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    if isinstance(value, date):
        return value.isoformat()
    return value


# ------------------------------------------------------------------- helpers


def _compound(daily_pct: pd.Series) -> float | None:
    clean = daily_pct.dropna()
    if clean.empty:
        return None
    growth = float(cast("float", (1 + clean / 100).prod()))
    return (growth - 1) * 100


def _cumulative(daily_pct: pd.Series) -> pd.Series:
    return ((1 + daily_pct.fillna(0) / 100).cumprod() - 1) * 100


def _daily_bars(
    store: BarStore, symbol: str, actions: dict[str, list[CorporateAction]]
) -> pd.DataFrame | None:
    if not store.has(symbol):
        return None
    bars = store.read(symbol)
    if symbol in actions:
        bars = adjust_for_actions(bars, actions[symbol])
    return bars


def _by_date(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.index = pd.DatetimeIndex(out.index).date
    return out[~out.index.duplicated(keep="last")]


@dataclass(slots=True)
class _HoldBasket:
    """Buy the constituents in equal weight once, then hold."""

    symbols: list[str]
    name: str = "basket_replica"

    def is_rebalance_date(self, on: date, *, previous: date | None) -> bool:  # noqa: ARG002
        return previous is None

    def rank(self, panel: dict[str, pd.DataFrame], *, on: date) -> list[tuple[str, float]]:  # noqa: ARG002
        return [(s, 1.0) for s in self.symbols if s in panel]

    def target_weights(self, panel: dict[str, pd.DataFrame], *, on: date) -> dict[str, Decimal]:  # noqa: ARG002
        live = [s for s in self.symbols if s in panel]
        if not live:
            return {}
        # Rounded down so the weights never ask for more cash than exists.
        weight = (Decimal(1) / len(live)).quantize(Decimal("0.000001"), rounding=ROUND_DOWN)
        return dict.fromkeys(live, weight)


def _series_definition(
    reported: pd.Series,
    constituents: list[str],
    closes: dict[str, pd.DataFrame],
    nifty: pd.DataFrame,
    source_nifty: pd.Series,
    since: date | None,
) -> dict[str, Any]:
    """Which daily-return definition reproduces the source's two series."""
    result: dict[str, Any] = {}

    def fit(a: pd.Series, b: pd.Series) -> dict[str, Any] | None:
        joined = pd.concat([a, b], axis=1, join="inner").dropna()
        if len(joined) < MIN_SESSIONS or joined.iloc[:, 1].std() == 0:
            return None
        return {
            "sessions": len(joined),
            "correlation": float(joined.iloc[:, 0].corr(joined.iloc[:, 1])),
            "mean_abs_diff_pct": float((joined.iloc[:, 0] - joined.iloc[:, 1]).abs().mean()),
        }

    nifty_c2c = nifty["close"].pct_change() * 100
    nifty_o2c = (nifty["close"] / nifty["open"] - 1) * 100
    result["benchmark"] = {
        "open_to_close": fit(source_nifty, nifty_o2c),
        "close_to_close": fit(source_nifty, nifty_c2c),
    }

    covered = [s for s in constituents if s in closes]
    if since is not None and covered:
        after = reported[reported.index > since]
        c2c = pd.DataFrame({s: closes[s]["close"].pct_change() * 100 for s in covered}).mean(axis=1)
        o2c = pd.DataFrame(
            {s: (closes[s]["close"] / closes[s]["open"] - 1) * 100 for s in covered}
        ).mean(axis=1)
        result["basket_since_rebalance"] = {
            "coverage": f"{len(covered)}/{len(constituents)}",
            "equal_weight_close_to_close": fit(after, c2c),
            "equal_weight_open_to_close": fit(after, o2c),
        }
    return result


# ------------------------------------------------------------------ windows


def _window(
    *,
    key: str,
    label: str,
    start: date,
    end: date,
    hindsight: bool,
    reported: pd.Series,
    source_nifty: pd.Series,
    nifty: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    calculator: ChargeCalculator,
    capital: Rupees,
    note: str,
    total_members: int,
) -> WindowResult | None:
    sessions = [d for d in nifty.index if start <= d <= end]
    if len(sessions) < MIN_SESSIONS:
        return None
    before = [d for d in nifty.index if d < start]
    if not before:
        return None
    anchor = before[-1]

    in_window = reported[(reported.index >= start) & (reported.index <= end)]
    source_window = source_nifty[(source_nifty.index >= start) & (source_nifty.index <= end)]
    nifty_closes = nifty["close"]
    nifty_pct = (float(nifty_closes[sessions[-1]]) / float(nifty_closes[anchor]) - 1) * 100

    daily = (
        pd.DataFrame({s: _by_date(f)["close"].pct_change() * 100 for s, f in frames.items()}).mean(
            axis=1
        )
        if frames
        else pd.Series(dtype=float)
    )
    costless_daily = daily[(daily.index >= start) & (daily.index <= end)]

    investable_pct = charges = charges_pct = drawdown = exit_charges = None
    equity_by_day = pd.Series(dtype=float)
    panel: dict[str, pd.DataFrame] = {}
    if frames:
        for symbol, frame in frames.items():
            dated = pd.DatetimeIndex(frame.index).date
            clipped = frame[(dated >= anchor) & (dated <= end)]
            if len(clipped) >= MIN_SESSIONS:
                panel[symbol] = clipped
        if panel:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UnverifiedScheduleWarning)
                result = run_rebalance_backtest(
                    strategy=_HoldBasket(sorted(panel)),
                    panel=panel,
                    calculator=calculator,
                    initial_capital=capital,
                )
            equity_by_day = pd.Series(
                result.equity_curve.to_numpy(),
                index=pd.DatetimeIndex(result.equity_curve.index).date,
            )
            initial = float(capital.value)
            exit_charges = _exit_charges(
                calculator, float(equity_by_day.iloc[-1]), len(panel), sessions[-1]
            )
            investable_pct = ((float(equity_by_day.iloc[-1]) - exit_charges) / initial - 1) * 100
            charges = float(result.total_charges.value) + exit_charges
            charges_pct = charges / initial * 100
            drawdown = result.metrics.max_drawdown * 100

    if exit_charges and not equity_by_day.empty:
        # The last mark is where the book would be sold; charge it there so the
        # curve ends on the same number as the headline.
        equity_by_day.iloc[-1] = float(equity_by_day.iloc[-1]) - exit_charges
    reported_curve = _cumulative(in_window)
    costless_curve = _cumulative(costless_daily)
    curve = [
        CurvePoint(
            date=day,
            reported_pct=float(reported_curve[day]) if day in reported_curve.index else None,
            costless_pct=float(costless_curve[day]) if day in costless_curve.index else None,
            investable_pct=(
                (float(equity_by_day[day]) / float(capital.value) - 1) * 100
                if day in equity_by_day.index
                else None
            ),
            nifty_pct=(float(nifty_closes[day]) / float(nifty_closes[anchor]) - 1) * 100,
        )
        for day in sessions
    ]
    return WindowResult(
        key=key,
        label=label,
        start=start,
        end=sessions[-1],
        sessions=len(sessions),
        hindsight=hindsight,
        reported_pct=_compound(in_window),
        source_nifty_pct=_compound(source_window)
        if len(source_window) >= 0.8 * len(sessions)
        else None,
        nifty_pct=nifty_pct,
        costless_pct=_compound(costless_daily),
        investable_pct=investable_pct,
        charges_rupees=charges,
        exit_charges_rupees=exit_charges,
        constituents_measured=len(panel),
        constituents_total=total_members,
        charges_pct_of_capital=charges_pct,
        max_drawdown_pct=drawdown,
        note=note,
        curve=curve,
    )


def _exit_charges(calculator: ChargeCalculator, equity: float, holdings: int, on: date) -> float:
    """Delivery sell charges on the final book, split evenly across holdings.

    An estimate: the engine marks positions rather than selling them, and the
    exact split depends on how weights drifted. STT dominates and is
    proportional, so the error is confined to the per-scrip DP charge.
    """
    if holdings <= 0 or equity <= 0:
        return 0.0
    from trading_agent.domain.enums import ProductType, Side

    per_holding = Decimal(str(round(equity / holdings, 2)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnverifiedScheduleWarning)
        leg = calculator.for_notional(
            price=per_holding, quantity=1, side=Side.SELL, product=ProductType.CNC, on=on
        )
    return float(leg.total.value) * holdings


# --------------------------------------------------------------------- entry


def replicate_basket(
    lake: SuperinvestingLake,
    store: BarStore,
    calculator: ChargeCalculator,
    basket: str,
    *,
    capital: Rupees,
    actions: dict[str, list[CorporateAction]] | None = None,
) -> BasketReplication:
    """Replicate the basket whose id or name is ``basket``."""
    baskets = lake.table("baskets")
    match = baskets[
        (baskets["basket_id"] == basket) | (baskets["name"].str.lower() == basket.lower())
    ]
    if match.empty:
        raise KeyError(f"no basket {basket!r}")
    meta = match.iloc[0]
    basket_id = str(meta["basket_id"])
    actions = actions or {}

    members = lake.table("basket_constituents")
    members = members[members["basket_id"] == basket_id].drop_duplicates("ticker")
    returns = lake.table("basket_returns")
    reported = (
        returns[returns["basket_id"] == basket_id]
        .set_index("date")["daily_return_pct"]
        .sort_index()
    )
    source_nifty = lake.table("benchmark_nifty").set_index("date")["daily_return_pct"].sort_index()

    nifty_bars = _daily_bars(store, BENCHMARK, actions)
    if nifty_bars is None:
        raise FileNotFoundError(f"no {BENCHMARK} bars; run `trading-agent data sync-index`")
    nifty = _by_date(nifty_bars)

    frames: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for ticker in members["ticker"]:
        bars = _daily_bars(store, str(ticker), actions)
        if bars is None:
            missing.append(str(ticker))
        else:
            frames[str(ticker)] = bars
    dated = {s: _by_date(f) for s, f in frames.items()}

    rebalanced = (
        pd.Timestamp(meta["rebalanced_at"]).date() if pd.notna(meta["rebalanced_at"]) else None
    )
    last_bar = nifty.index[-1]
    windows: list[WindowResult] = []
    window = partial(
        _window,
        total_members=len(members),
        reported=reported,
        source_nifty=source_nifty,
        nifty=nifty,
        frames=frames,
        calculator=calculator,
        capital=capital,
    )
    if rebalanced is not None:
        after = [d for d in reported.index if d > rebalanced]
        if after:
            since = window(
                key="since_rebalance",
                label=f"Since last rebalance ({rebalanced.isoformat()})",
                start=after[0],
                end=min(after[-1], last_bar),
                hindsight=False,
                note=(
                    "Today's constituents are exactly what the basket held: an honest replication."
                ),
            )
            if since is not None:
                windows.append(since)
    if len(reported):
        full = window(
            key="full_history",
            label="Full reported history, today's constituents",
            start=reported.index[0],
            end=min(reported.index[-1], last_bar),
            hindsight=rebalanced is not None and rebalanced > reported.index[0],
            note=(
                "Uses today's members for the whole history. They were selected knowing how "
                "they had done, so this flatters the basket."
            ),
        )
        # When the last rebalance predates the history the two windows are the
        # same run; report it once.
        if full is not None and not any(w.start == full.start for w in windows):
            windows.append(full)

    definition = _series_definition(
        reported, members["ticker"].tolist(), dated, nifty, source_nifty, rebalanced
    )
    persona_ids = [p for p in str(meta["persona_ids"] or "").split(",") if p]
    replication = BasketReplication(
        basket_id=basket_id,
        name=str(meta["name"]),
        category=meta["category"],
        description=meta["description"],
        persona_ids=persona_ids,
        created_at=str(meta["created_at"])[:10] if pd.notna(meta["created_at"]) else None,
        rebalanced_at=rebalanced.isoformat() if rebalanced else None,
        reported_overall_pct=float(meta["reported_return_pct"])
        if pd.notna(meta["reported_return_pct"])
        else None,
        constituents=[
            {
                "ticker": r["ticker"],
                "name": r["name"],
                "score": r["score"],
                "has_bars": r["ticker"] in frames,
            }
            for r in members.to_dict(orient="records")
        ],
        missing_bars=missing,
        capital_rupees=float(capital.value),
        series_definition=definition,
        windows=windows,
        findings=[],
    )
    replication.findings = _findings(replication)
    log.info(
        "basket_replicated", basket=replication.name, windows=len(windows), missing=len(missing)
    )
    return replication


def _findings(rep: BasketReplication) -> list[str]:
    """Plain sentences for the page and the chat. Only what the numbers support."""
    out: list[str] = []
    bench = rep.series_definition.get("benchmark", {})
    o2c, c2c = bench.get("open_to_close"), bench.get("close_to_close")
    if o2c and c2c and o2c["correlation"] > REPRODUCED_CORRELATION > c2c["correlation"]:
        out.append(
            "The Nifty line the source compares against is compounded open-to-close returns, "
            "which leave out every overnight move; an index fund earned the close-to-close figure."
        )
    since = rep.series_definition.get("basket_since_rebalance") or {}
    fit = since.get("equal_weight_close_to_close")
    if fit and fit["correlation"] > REPRODUCED_CORRELATION:
        out.append(
            f"Since the last rebalance the reported series is reproduced by an equal-weighted, "
            f"daily re-weighted, cost-free portfolio of the current members "
            f"(correlation {fit['correlation']:.3f}, {fit['sessions']} sessions)."
        )
    elif fit:
        out.append(
            f"Since the last rebalance the reported series only partly matches an equal-weighted "
            f"portfolio of the members we have bars for (correlation {fit['correlation']:.2f}, "
            f"coverage {since.get('coverage')}); weights or members differ from what is published."
        )
    if rep.missing_bars:
        out.append(
            f"{len(rep.missing_bars)} of {len(rep.constituents)} constituents have no NSE bars in "
            f"the platform store ({', '.join(rep.missing_bars)}), so replicated figures cover "
            "the rest."
        )
    for window in rep.windows:
        if window.reported_pct is None or window.investable_pct is None or window.nifty_pct is None:
            continue
        partial = window.constituents_measured < window.constituents_total
        out.append(
            f"{window.label}: reported {window.reported_pct:+.2f}%; bought and held net of "
            f"₹{window.charges_rupees or 0:,.0f} charges {window.investable_pct:+.2f}%"
            + (
                f" ({window.constituents_measured} of {window.constituents_total} members)"
                if partial
                else ""
            )
            + f"; Nifty {window.nifty_pct:+.2f}%"
            + (" — hindsight-flattered" if window.hindsight else "")
            + "."
        )
    return out


def replicate_all(
    lake: SuperinvestingLake,
    store: BarStore,
    calculator: ChargeCalculator,
    *,
    capital: Rupees,
    actions: dict[str, list[CorporateAction]] | None = None,
) -> list[BasketReplication]:
    """Every basket, and the summaries persisted for the research desk."""
    import json

    results = [
        replicate_basket(lake, store, calculator, str(b), capital=capital, actions=actions)
        for b in lake.table("baskets")["basket_id"]
    ]
    (lake.root / REPLICATIONS_FILE).write_text(
        json.dumps([r.summary() for r in results], indent=2), encoding="utf-8"
    )
    return results
