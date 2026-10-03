"""Screening alphas by information coefficient, before any backtest.

A backtest answers "what would this have earned", which is the wrong first
question for fifty candidate signals: it conflates the signal with position
sizing, the stop rule, the position cap and the charge model, and it takes a
run per candidate. The information coefficient asks the narrower question --
**does this number rank tomorrow's winners above tomorrow's losers?** -- and
answers it for every alpha at once.

IC here is the *rank* (Spearman) correlation, computed cross-sectionally within
each session, between an alpha's value and the forward return over the next
``horizon`` sessions. One number per session; the summary is their mean.

How to read the output:

``mean_ic``
    Above roughly +0.02 is a real signal in equities; 0.05 is strong. The sign
    matters as much as the size -- a consistent -0.04 is the same edge traded
    the other way round, which several of these alphas are written to expect.
``ic_t_stat``
    ``mean_ic / stderr(ic)``. |t| > 2 is the usual bar for "not chance". This
    is the number that matters: a high mean IC over 30 sessions is noise, and
    a modest one over 900 is not.
``ic_ir``
    Mean IC over its standard deviation -- consistency rather than magnitude.
``hit_rate``
    Fraction of sessions with an IC of the same sign as the mean.

**No parameter is fitted here.** The alphas arrive as published formulas and
are measured as-is, which is what makes the screen honest: nothing is tuned on
the data it is scored against. The screen ranks candidates; a walk-forward
backtest is still required before believing any of them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_agent.features.alphas import Panel, compute_alphas
from trading_agent.observability import get_logger

__all__ = ["AlphaScore", "forward_returns", "score_alpha", "screen_alphas"]

log = get_logger(__name__)

#: Sessions with fewer than this many ranked names produce a meaningless
#: cross-sectional correlation and are dropped rather than averaged in.
MIN_NAMES = 10

#: |t| above this is the conventional bar for "unlikely to be chance". Note it
#: is uncorrected for multiple testing: screening 30 alphas at t=2 expects
#: roughly one false positive by luck alone.
SIGNIFICANT_T = 2.0


@dataclass(frozen=True, slots=True)
class AlphaScore:
    """What one alpha's information coefficient looks like."""

    name: str
    horizon: int
    mean_ic: float
    ic_std: float
    ic_t_stat: float
    hit_rate: float
    sessions: int
    coverage: float

    @property
    def ic_ir(self) -> float:
        """Information ratio of the IC series: consistency, not magnitude."""
        return self.mean_ic / self.ic_std if self.ic_std > 0 else 0.0

    @property
    def is_significant(self) -> bool:
        """|t| > 2, the conventional bar for "unlikely to be chance"."""
        return abs(self.ic_t_stat) > SIGNIFICANT_T

    def summary_row(self) -> tuple[str, str, str, str, str, str]:
        return (
            self.name,
            f"{self.mean_ic:+.4f}",
            f"{self.ic_t_stat:+.2f}",
            f"{self.ic_ir:+.3f}",
            f"{self.hit_rate:.0%}",
            f"{self.coverage:.0%}",
        )


def _spearman(left: pd.Series, right: pd.Series) -> float | None:
    """Rank correlation, computed as Pearson over ranks.

    Spelled out rather than delegated to ``Series.corr(method="spearman")``
    so that the screen stays exact about how ties and constant columns are
    handled -- both are ordinary in a 50-name universe, and the branch below
    is the whole reason this is not a one-liner. Rank correlation is the right
    choice here because the alphas are used to *order* the universe: only the
    ordering is traded, and a single outlier must not dominate the score.
    """
    ranked_left = left.rank().to_numpy(dtype=float)
    ranked_right = right.rank().to_numpy(dtype=float)
    left_centred = ranked_left - ranked_left.mean()
    right_centred = ranked_right - ranked_right.mean()
    denominator = np.sqrt((left_centred @ left_centred) * (right_centred @ right_centred))
    if denominator <= 0:
        # One side is constant -- every name tied. No ordering, no information.
        return None
    return float((left_centred @ right_centred) / denominator)


def forward_returns(panel: Panel, horizon: int) -> pd.DataFrame:
    """Return from the next session's open to the open ``horizon`` sessions later.

    Open-to-open, not close-to-close. A signal computed from session *t*'s
    close can only be acted on at *t+1*'s open, so scoring it against a return
    that starts at *t*'s close credits it with a move it could never have
    captured -- the same one-bar lookahead the backtest engine avoids with
    ``next_bar_open`` fills.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")
    entry = panel.open.shift(-1)
    exit_ = panel.open.shift(-(1 + horizon))
    return exit_ / entry - 1.0


def score_alpha(
    name: str, values: pd.DataFrame, ahead: pd.DataFrame, *, horizon: int
) -> AlphaScore | None:
    """Information coefficient of one alpha against forward returns."""
    aligned = values.reindex_like(ahead)
    coverage = float(aligned.notna().to_numpy().mean())

    daily_ic: list[float] = []
    for session in aligned.index:
        signal = aligned.loc[session]
        future = ahead.loc[session]
        usable = signal.notna() & future.notna()
        if int(usable.sum()) < MIN_NAMES:
            continue
        ic = _spearman(signal[usable], future[usable])
        if ic is not None:
            daily_ic.append(ic)

    if len(daily_ic) < MIN_NAMES:
        log.debug("alpha_scored_too_few_sessions", alpha=name, sessions=len(daily_ic))
        return None

    series = np.asarray(daily_ic, dtype=float)
    mean_ic = float(series.mean())
    ic_std = float(series.std(ddof=1)) if len(series) > 1 else 0.0
    hits = float((np.sign(series) == np.sign(mean_ic)).mean()) if mean_ic != 0 else 0.0

    # A zero-variance IC series means the signal was *perfectly* consistent, so
    # the confidence in its mean is unbounded -- infinity, not zero. Returning
    # 0.0 here (the naive guard against dividing by zero) would mark a flawless
    # signal as insignificant, which is exactly backwards.
    if ic_std > 0:
        t_stat = mean_ic / (ic_std / np.sqrt(len(series)))
    else:
        t_stat = 0.0 if mean_ic == 0 else float(np.inf) * np.sign(mean_ic)

    return AlphaScore(
        name=name,
        horizon=horizon,
        mean_ic=mean_ic,
        ic_std=ic_std,
        ic_t_stat=float(t_stat),
        hit_rate=hits,
        sessions=len(series),
        coverage=coverage,
    )


def screen_alphas(
    panel: Panel, *, horizon: int = 5, names: list[str] | None = None
) -> list[AlphaScore]:
    """Score every alpha over ``panel``, strongest absolute IC first.

    Sorted on |mean IC| rather than the signed value: an alpha that reliably
    ranks losers first is exactly as useful as one that ranks winners first,
    and is traded by flipping its sign.
    """
    ahead = forward_returns(panel, horizon)
    computed = compute_alphas(panel, names)

    scored = [
        score
        for name, values in computed.items()
        if (score := score_alpha(name, values, ahead, horizon=horizon)) is not None
    ]
    scored.sort(key=lambda s: abs(s.mean_ic), reverse=True)

    log.info(
        "alpha_screen_complete",
        horizon=horizon,
        scored=len(scored),
        significant=sum(1 for s in scored if s.is_significant),
    )
    return scored
