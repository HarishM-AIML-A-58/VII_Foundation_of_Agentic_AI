"""Backtest reporting: equity curve versus NIFTY buy-and-hold.

One HTML file, no external assets. Charts are inline SVG built from the
series itself rather than a plotting library, because a report that needs a
CDN is a report that renders blank on the laptop of whoever you sent it to,
six months later, when it matters.

What the report leads with is deliberate. The headline is **net return versus
buy-and-hold**, not the equity curve: a rising curve is not a result if the
index rose faster, and a report that opens with a pretty line encourages
exactly that confusion. Charges are shown next to it for the same reason --
on Indian equities they are frequently the whole difference.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from trading_agent.observability import get_logger

if TYPE_CHECKING:
    from trading_agent.backtest.engine import BacktestResult

__all__ = ["render_report", "write_report"]

log = get_logger(__name__)

#: Points plotted per chart. A five-year daily curve is ~1250 points, which
#: draws fine; anything much larger is downsampled so the SVG stays small
#: enough to email.
_MAX_POINTS = 1500

_CHART_WIDTH = 960
_CHART_HEIGHT = 320
_DRAWDOWN_HEIGHT = 160
_PADDING = 40


@dataclass(frozen=True, slots=True)
class _Series:
    """A named, normalised series ready to plot."""

    label: str
    colour: str
    values: pd.Series


def _downsample(series: pd.Series) -> pd.Series:
    if len(series) <= _MAX_POINTS:
        return series
    step = len(series) // _MAX_POINTS + 1
    # Always keep the final point: the last value is the result, and dropping
    # it to a stride boundary would misstate the total return by a day.
    sampled = series.iloc[::step]
    if sampled.index[-1] != series.index[-1]:
        sampled = pd.concat([sampled, series.iloc[-1:]])
    return sampled


def _rebase(series: pd.Series) -> pd.Series:
    """Index to 100 at the start so two curves are comparable by shape."""
    first = float(series.iloc[0])
    if first == 0:
        return series
    return series / first * 100.0


def _path(series: pd.Series, *, low: float, high: float, height: int) -> str:
    """An SVG polyline path for ``series`` scaled into ``height``."""
    if len(series) < 2:  # noqa: PLR2004 -- a line needs two points
        return ""
    span = high - low or 1.0
    width = _CHART_WIDTH - 2 * _PADDING
    plot_height = height - 2 * _PADDING
    points = []
    for position, value in enumerate(series.to_numpy()):
        x = _PADDING + position / (len(series) - 1) * width
        y = _PADDING + (1.0 - (float(value) - low) / span) * plot_height
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def _line_chart(series: list[_Series], *, title: str, height: int = _CHART_HEIGHT) -> str:
    plotted = [_Series(s.label, s.colour, _downsample(s.values)) for s in series if len(s.values)]
    if not plotted:
        return f'<p class="empty">{html.escape(title)}: no data</p>'

    low = min(float(s.values.min()) for s in plotted)
    high = max(float(s.values.max()) for s in plotted)

    polylines = "\n".join(
        f'<polyline points="{_path(s.values, low=low, high=high, height=height)}" '
        f'fill="none" stroke="{s.colour}" stroke-width="2" />'
        for s in plotted
    )
    legend = " ".join(
        f'<span class="key"><i style="background:{s.colour}"></i>{html.escape(s.label)}</span>'
        for s in plotted
    )
    axis = (
        f'<text x="{_PADDING}" y="{_PADDING - 12}" class="axis">{high:,.1f}</text>'
        f'<text x="{_PADDING}" y="{height - _PADDING + 18}" class="axis">{low:,.1f}</text>'
    )
    return (
        f'<section><h2>{html.escape(title)}</h2><div class="legend">{legend}</div>'
        f'<svg viewBox="0 0 {_CHART_WIDTH} {height}" role="img" '
        f'aria-label="{html.escape(title)}">{axis}{polylines}</svg></section>'
    )


def _drawdown_series(equity: pd.Series) -> pd.Series:
    return (equity / equity.cummax() - 1.0) * 100.0


def _summary_table(result: BacktestResult) -> str:
    rows = "".join(
        f"<tr><th>{html.escape(label)}</th><td>{html.escape(value)}</td></tr>"
        for label, value in result.metrics.summary_rows()
    )
    return f"<table class='summary'>{rows}</table>"


def _verdict(result: BacktestResult) -> str:
    """The headline. Everything else in the report is supporting evidence."""
    if result.benchmark_metrics is None:
        return (
            '<p class="verdict neutral">No benchmark was supplied, so this run '
            "says nothing about whether the strategy was worth running.</p>"
        )
    strategy = result.metrics.total_return
    benchmark = result.benchmark_metrics.total_return
    edge = strategy - benchmark
    verdict = "beat" if result.beat_benchmark else "did not beat"
    tone = "good" if result.beat_benchmark else "bad"
    return (
        f'<p class="verdict {tone}">Net of every charge, the strategy '
        f"<strong>{verdict}</strong> buy-and-hold: "
        f"{strategy:+.2%} versus {benchmark:+.2%} ({edge:+.2%}).</p>"
    )


def _trade_table(result: BacktestResult, *, limit: int = 200) -> str:
    trips = result.round_trips
    if not trips:
        return "<section><h2>Trades</h2><p class='empty'>No round trips.</p></section>"

    shown = trips[:limit]
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(t.symbol)}</td>"
        f"<td>{t.side.value}</td>"
        f"<td class='num'>{t.quantity}</td>"
        f"<td>{t.entry_at:%Y-%m-%d}</td>"
        f"<td>{t.exit_at:%Y-%m-%d}</td>"
        f"<td class='num'>{t.entry_price}</td>"
        f"<td class='num'>{t.exit_price}</td>"
        f"<td class='num'>{t.total_charges}</td>"
        f"<td class='num {'win' if t.is_winner else 'loss'}'>{t.net_pnl}</td>"
        f"<td>{html.escape(t.exit_reason or '')}</td>"
        "</tr>"
        for t in shown
    )
    truncated = (
        f"<p class='note'>Showing {limit} of {len(trips)} trades.</p>" if len(trips) > limit else ""
    )
    return (
        "<section><h2>Trades</h2><table class='trades'><thead><tr>"
        "<th>symbol</th><th>side</th><th>qty</th><th>entry</th><th>exit</th>"
        "<th>entry px</th><th>exit px</th><th>charges</th><th>net P&amp;L</th>"
        "<th>why closed</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>{truncated}</section>"
    )


def _skipped_table(result: BacktestResult) -> str:
    """What the engine declined to trade, and why.

    Reported prominently because a backtest that silently skipped 80% of its
    signals is not the strategy anybody thinks they measured.
    """
    if not result.skipped:
        return ""
    rows = "".join(
        f"<tr><th>{html.escape(reason)}</th><td class='num'>{count}</td></tr>"
        for reason, count in sorted(result.skipped.items(), key=lambda kv: -kv[1])
    )
    return f"<section><h2>Signals not taken</h2><table class='summary'>{rows}</table></section>"


_STYLE = """
:root { color-scheme: light dark; --fg:#111; --bg:#fff; --muted:#666; --line:#e3e3e3;
        --good:#0a7d33; --bad:#b3261e; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e8e8e8; --bg:#131313; --muted:#9a9a9a; --line:#2c2c2c;
          --good:#5fd07f; --bad:#ef8a82; }
}
* { box-sizing: border-box; }
body { margin:0; padding:2rem 1.25rem 4rem; background:var(--bg); color:var(--fg);
       font:15px/1.55 ui-sans-serif,-apple-system,Segoe UI,Roboto,sans-serif;
       max-width:1040px; margin-inline:auto; }
h1 { font-size:1.5rem; margin:0 0 .25rem; }
h2 { font-size:1.05rem; margin:2.25rem 0 .6rem; font-weight:600; }
.sub { color:var(--muted); margin:0 0 1.5rem; }
.verdict { font-size:1.05rem; padding:.85rem 1rem; border-radius:8px;
           border:1px solid var(--line); }
.verdict.good { border-left:4px solid var(--good); }
.verdict.bad  { border-left:4px solid var(--bad); }
.verdict.neutral { border-left:4px solid var(--muted); }
table { border-collapse:collapse; width:100%; font-size:14px; }
th, td { text-align:left; padding:.4rem .6rem; border-bottom:1px solid var(--line); }
.summary th { width:45%; font-weight:500; color:var(--muted); }
.num { text-align:right; font-variant-numeric:tabular-nums; }
.win { color:var(--good); } .loss { color:var(--bad); }
svg { width:100%; height:auto; border:1px solid var(--line); border-radius:8px; }
.axis { fill:var(--muted); font-size:11px; }
.legend { margin:.35rem 0 .5rem; color:var(--muted); font-size:13px; }
.key { margin-right:1rem; } .key i { display:inline-block; width:10px; height:10px;
       border-radius:2px; margin-right:.35rem; vertical-align:middle; }
.empty, .note { color:var(--muted); font-size:13px; }
.scroll { overflow-x:auto; }
footer { margin-top:3rem; color:var(--muted); font-size:12px;
         border-top:1px solid var(--line); padding-top:1rem; }
"""


def render_report(result: BacktestResult, *, title: str = "Backtest report") -> str:
    """Render the report as a self-contained HTML document."""
    equity = result.equity_curve
    curves = [_Series("strategy", "#2f6fed", _rebase(equity))]
    if result.benchmark_curve is not None and len(result.benchmark_curve):
        curves.append(_Series("buy & hold", "#8a8a8a", _rebase(result.benchmark_curve)))

    body = "".join(
        [
            f"<h1>{html.escape(title)}</h1>",
            f'<p class="sub">{result.metrics.start:%Y-%m-%d} to '
            f"{result.metrics.end:%Y-%m-%d} &middot; "
            f"{result.metrics.trades} trades &middot; net of all charges</p>",
            _verdict(result),
            _line_chart(curves, title="Equity, rebased to 100"),
            _line_chart(
                [_Series("drawdown %", "#b3261e", _drawdown_series(equity))],
                title="Drawdown",
                height=_DRAWDOWN_HEIGHT,
            ),
            "<section><h2>Performance</h2>",
            _summary_table(result),
            "</section>",
            _skipped_table(result),
            f'<div class="scroll">{_trade_table(result)}</div>',
            "<footer>Every figure is net of brokerage, STT, stamp duty, GST, "
            "SEBI and exchange charges, and modelled slippage. "
            "Past performance in a backtest is not a prediction.</footer>",
        ]
    )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{_STYLE}</style></head>"
        f"<body>{body}</body></html>"
    )


def write_report(result: BacktestResult, *, output: Path, title: str = "Backtest report") -> Path:
    """Write an HTML report with the equity curve, drawdowns and trade list."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(result, title=title), encoding="utf-8")
    log.info("report_written", path=str(output), trades=result.metrics.trades)
    return output
