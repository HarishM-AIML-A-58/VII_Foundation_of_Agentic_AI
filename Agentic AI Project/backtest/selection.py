"""Which symbols a backtest is allowed to trade.

Separated from the CLI because the two mistakes it prevents are both silent,
and a silent mistake needs a test rather than a code review.

**The benchmark is not an instrument.** A curated store that holds NIFTY50
alongside its constituents will, if handed wholesale to a strategy, have that
strategy buy and sell the index -- paying STT and DP charges on something no
retail account can hold, and then be compared against a buy-and-hold of the
very series it just traded. Nothing errors. The equity curve simply includes
returns that were never available.

**Membership has a date.** Testing today's index on ten-year-old data is
survivorship bias: the constituents that failed are already gone, so the
backtest measures a portfolio chosen with hindsight. This module cannot
manufacture the membership history that would fix it, but it does report
exactly how much history exists, so a caller can say so out loud instead of
quoting the number as though it were clean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from trading_agent.market_data.universe import Universe
from trading_agent.observability import get_logger

__all__ = ["SymbolSelection", "select_tradeable"]

#: How many symbol names a warning lists before summarising the rest. Enough to
#: recognise the gap, short enough to stay one readable line.
_NAMES_SHOWN = 8

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SymbolSelection:
    """The tradeable set, and what was dropped to get there."""

    symbols: list[str]
    #: symbol -> why it is not tradeable. Rendered by the caller.
    excluded: dict[str, str] = field(default_factory=dict)
    #: Members whose index membership is dated, over the requested window.
    dated_members: int = 0
    #: Members in the current snapshot, dated or not.
    total_members: int = 0
    #: True when the universe carries dated membership at all.
    has_history: bool = False
    #: Index members on the requested date for which the store holds no bars.
    #: These are former constituents: the names whose absence is what inflates
    #: a backtest, and the reason a complete membership file is necessary but
    #: not sufficient.
    absent_from_store: tuple[str, ...] = ()
    #: Index members over the requested window, from the membership history.
    #: Larger than 50 for a multi-year window: the index turns over.
    members_in_window: int = 0

    #: Current-snapshot members the history file never mentions. These cannot
    #: be gated on membership at all, so they trade over the whole window.
    undated_members: tuple[str, ...] = ()
    #: Symbols the snapshot calls current while the history says they left. One
    #: of the two files is stale.
    snapshot_conflicts: tuple[str, ...] = ()

    @property
    def is_survivorship_biased(self) -> bool:
        """True while the run cannot see the whole index as it stood.

        Three distinct deficiencies, and any one is enough:

        1. **No membership history at all.** Today's members are tested over
           periods they may not have been in the index.
        2. **Some members are undated.** The file omits them, so they cannot be
           gated and trade over the whole window regardless of when they joined.
           Note this is *not* the same as a symbol being absent from the index
           on a given date -- that case is correctly handled and must not warn,
           or the warning fires forever on a complete file.
        3. **The two universe files disagree.** The snapshot calls a symbol a
           current member while the history gives it only closed windows, so
           gating it cannot be trusted in either direction.
        4. **Former constituents have no bars.** Membership is known and
           entries are gated on it, but the store holds no prices for names
           since dropped, so the panel is still the survivors. This is the
           residual case, and it is the one that completing the membership file
           would otherwise silence while leaving the bias in place.
        """
        return not self.has_history or bool(
            self.undated_members or self.snapshot_conflicts or self.absent_from_store
        )

    @property
    def survivorship_note(self) -> str:
        """One line stating the coverage, for a caller to print."""
        if not self.is_survivorship_biased:
            return ""
        if not self.has_history:
            return (
                f"survivorship bias: no dated membership for {self.total_members} "
                f"constituents, so all of them are tested over periods they may not "
                f"have been in the index. Returns are inflated by an unknown amount."
            )
        parts: list[str] = []
        if self.snapshot_conflicts:
            listed = ", ".join(self.snapshot_conflicts[:_NAMES_SHOWN])
            parts.append(
                f"{len(self.snapshot_conflicts)} symbol(s) are listed as current members "
                f"but the history says they left ({listed}); one of the two universe "
                f"files is stale"
            )
        if self.undated_members:
            listed = ", ".join(self.undated_members[:_NAMES_SHOWN])
            parts.append(
                f"{len(self.undated_members)} of {self.total_members} constituents have "
                f"no membership record ({listed}), so they trade over the whole window "
                f"regardless of when they joined the index"
            )
        if self.absent_from_store:
            listed = ", ".join(self.absent_from_store[:_NAMES_SHOWN])
            hidden = len(self.absent_from_store) - _NAMES_SHOWN
            more = f" and {hidden} more" if hidden > 0 else ""
            parts.append(
                f"{len(self.absent_from_store)} of the {self.members_in_window} symbols "
                f"that were in the index during this window have no bars in the store "
                f"({listed}{more}), so the panel is the "
                f"{self.members_in_window - len(self.absent_from_store)} survivors"
            )
        return (
            "survivorship bias: " + "; and ".join(parts) + ". Returns are inflated by an "
            "unknown amount."
        )


def select_tradeable(
    stored: list[str],
    *,
    universe: Universe,
    on: date,
    benchmark: str | None = None,
    until: date | None = None,
) -> SymbolSelection:
    """Restrict ``stored`` to instruments the strategy may actually hold.

    Membership in the universe is the test, not absence from the benchmark
    slot: an index left out of ``benchmark`` is still an index, and a store
    that grows a second one later must not silently become tradeable. The
    benchmark is checked too, so that using a *constituent* as the benchmark
    still removes it from the panel -- holding it on both sides of the
    comparison would net most of its contribution to zero.

    Membership is judged over the WHOLE window ``[on, until]``, not on one date.
    A former constituent belongs in the panel: it was tradeable then, and
    excluding it is survivorship bias no matter how carefully entries are gated
    afterwards. :meth:`Universe.was_member` decides *when* each name may be
    entered; this decides *whether* it is in the panel at all.

    Getting that distinction wrong is silent and self-reinforcing. With the panel
    limited to current members, syncing the 33 dropped NIFTY 50 constituents
    changed no result whatsoever, because every one of them was excluded here as
    "not a member".
    """
    window_start, window_end = min(on, until or on), max(on, until or on)
    members = (
        set(universe.members_between(window_start, window_end))
        if universe.has_history
        else set(universe.current_symbols())
    )
    excluded: dict[str, str] = {}
    symbols: list[str] = []

    normalised_benchmark = (benchmark or "").strip().upper()

    for symbol in stored:
        if symbol.upper() == normalised_benchmark:
            excluded[symbol] = "benchmark; held on the comparison side, not traded"
            continue
        if symbol not in members:
            excluded[symbol] = (
                f"never a member of {universe.name} in this window; index or stale listing"
            )
            continue
        symbols.append(symbol)

    # `total_members` stays the CURRENT snapshot count, which is what the
    # undated-members message is a fraction of. It is a different quantity from
    # the window membership used to build the panel above, and conflating the
    # two makes "3 of 50 constituents have no membership record" report a
    # denominator that grows with the window length.
    snapshot = set(universe.current_symbols())
    dated = {s for s in universe.symbols_on(on) if s in members} if universe.has_history else set()

    # Members on `on` that the store cannot price. These are former
    # constituents, and their absence -- not the membership file -- is what is
    # left of the survivorship problem once membership is dated.
    # Coverage is judged over the whole window, not on one date. A run from
    # 2015 to 2026 whose panel is complete *today* is still missing the twenty
    # names that were in the index in 2015 and have since been dropped.
    stored_upper = {s.upper() for s in stored}
    window_members = sorted(members) if universe.has_history else []
    absent = tuple(sorted(s for s in window_members if s not in stored_upper))
    undated = tuple(universe.undated_members()) if universe.has_history else ()
    conflicts = tuple(universe.snapshot_conflicts()) if universe.has_history else ()

    selection = SymbolSelection(
        symbols=sorted(symbols),
        excluded=excluded,
        dated_members=len(dated),
        total_members=len(snapshot),
        has_history=universe.has_history,
        absent_from_store=absent,
        members_in_window=len(window_members),
        undated_members=undated,
        snapshot_conflicts=conflicts,
    )

    if excluded:
        # A count and the distinct reasons, not the whole map. The rebalance
        # path feeds this the entire store, so the map is ~450 entries saying
        # the same thing, and it buried the result it was meant to qualify.
        # Callers that need the detail read `selection.excluded`.
        reasons: dict[str, int] = {}
        for reason in excluded.values():
            reasons[reason] = reasons.get(reason, 0) + 1
        log.info("backtest_symbols_excluded", count=len(excluded), reasons=reasons)
    if selection.is_survivorship_biased:
        log.warning(
            "backtest_survivorship_bias",
            has_history=selection.has_history,
            members_in_window=selection.members_in_window,
            absent_from_store=len(selection.absent_from_store),
            undated_members=len(selection.undated_members),
            snapshot_conflicts=len(selection.snapshot_conflicts),
            dated_members=selection.dated_members,
            total_members=selection.total_members,
        )
    return selection
