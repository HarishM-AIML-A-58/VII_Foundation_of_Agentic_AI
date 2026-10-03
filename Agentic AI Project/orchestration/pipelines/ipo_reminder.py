"""IPO reminders and Go/No-Go verdicts, refreshed once a session.

Six steps, in this order:

1. **Refresh from NSE.** Pull the official upcoming-issues and active-
   subscription feeds, upsert them into :class:`IPORepository`, then merge in
   NSE's own per-symbol issue-detail page (``api/ipo-detail``) -- an
   immediate QIB/HNI/retail split plus issue structure (fresh issue vs.
   offer-for-sale, parsed from NSE's own disclosed text), anchor share
   count, lead managers, and RHP/anchor-report links.
2. **Refresh from Groww.** NSE's feed has no lot size, no listing date, no
   sector/registrar, and no financials at all -- Groww's IPO pages
   (``market_data.groww_ipo_source``) carry all of that, plus pros/cons and
   a company blurb whose "use of proceeds" paragraph is the closest thing to
   a structured objects-of-issue field either source has. Every listing
   (open, closed, upcoming) is scraped, including its own detail sub-page,
   and merged in: Groww wins on fields it and NSE both report, since it is
   the richer source, but a field neither source has published yet never
   erases what the other already gave (see ``IPORepository.upsert_from_groww``
   / ``update_issue_detail``).
3. **Apply curated fundamentals** (``market_data.ipo_fundamentals``) on top
   -- neither feed carries a P/E figure, so a symbol with no curated file
   stays exactly as unmeasured as it was.
4. **Research.** For issues with something substantive scraped (financials,
   a company blurb, or disclosed pros/cons), ask an LLM for a plain-language
   read of it -- fundamentals trend, issue structure, risks -- via
   ``agents.nodes.ipo_researcher``. This is commentary, never a scored input:
   ``features.ipo_gonogo`` never reads it, and it is only regenerated once
   the existing note is older than ``research_refresh_days`` so an unchanged
   IPO is not re-billed every single day. Skipped entirely when Foundry
   isn't configured.
5. **Evaluate.** Score every issue currently bidding with
   :class:`~trading_agent.features.ipo_gonogo.IPOGoNoGoAnalyzer`, reading
   whatever fundamentals are on file for it (possibly none).
6. **Remind.** Journal one entry per issue that has hit a reminder date today
   (N days before bidding opens, bidding-opens day, last day to apply,
   listing day). This is the audit trail an operator reads to know an IPO
   deadline is near -- there is no separate notification channel yet, so it
   reuses the journal the rest of the system already writes to.

Network I/O runs on a worker thread, matching the daily data-sync job, so it
never stalls the event loop the intraday monitor also ticks on. Groww's
per-issue detail pages are fetched with a polite pause between requests and
cached once an issue's outcome is final (see ``GrowwIPOSource.fetch_detail``),
so only currently-bidding or upcoming issues cost a live request on a normal
day; the daily tick does not re-fetch every closed issue's detail -- that is
what ``trading-agent ipo groww-backfill`` is for.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Final

from trading_agent.agents.llm import FoundryClient
from trading_agent.agents.nodes.ipo_researcher import IPOResearchInput, research_ipo
from trading_agent.features.ipo_gonogo import IPOGoNoGoAnalyzer, IPOGoNoGoReport
from trading_agent.market_data.groww_ipo_source import GrowwIPOSource, GrowwListingRow
from trading_agent.market_data.ipo_fundamentals import load_ipo_fundamentals
from trading_agent.market_data.ipo_source import NSEIPOSource, NSEIssueDetail
from trading_agent.observability import get_logger
from trading_agent.persistence import (
    Database,
    IPOItem,
    IPORepository,
    JournalEntry,
    JournalRepository,
)
from trading_agent.settings import Settings

__all__ = ["IPOReminderOutcome", "IPOReminderPipeline", "run_ipo_reminder"]

log = get_logger(__name__)

#: Days before bidding opens to raise the first reminder.
_ADVANCE_REMINDER_DAYS: Final = 3

#: An AI research note is only regenerated after this many days -- a
#: company's disclosures don't change day to day, so re-billing an unchanged
#: read every session would spend money for no new information.
_DEFAULT_RESEARCH_REFRESH_DAYS: Final = 7


@dataclass(frozen=True, slots=True)
class IPOReminderOutcome:
    """What one reminder pass did, for the scheduler's job log and tests."""

    session_date: date
    refreshed: int
    reminders_raised: int
    go_no_go: list[IPOGoNoGoReport] = field(default_factory=list)


class IPOReminderPipeline:
    """Refreshes NSE's + Groww's IPO feeds, scores active issues, journals reminders."""

    __slots__ = (
        "_database",
        "_fundamentals_dir",
        "_groww_source",
        "_research_client",
        "_research_refresh_days",
        "_source",
    )

    def __init__(
        self,
        *,
        source: NSEIPOSource,
        groww_source: GrowwIPOSource,
        database: Database,
        fundamentals_dir: Path,
        research_client: FoundryClient | None = None,
        research_refresh_days: int = _DEFAULT_RESEARCH_REFRESH_DAYS,
    ) -> None:
        self._source = source
        self._groww_source = groww_source
        self._database = database
        self._fundamentals_dir = fundamentals_dir
        self._research_client = research_client
        self._research_refresh_days = research_refresh_days

    async def tick(self, session_date: date, *, include_closed: bool = False) -> IPOReminderOutcome:
        refreshed = await self._refresh(session_date)
        refreshed += await self._refresh_groww(include_closed=include_closed)
        await self._apply_fundamentals()
        await self._recompute_statuses(session_date)
        await self._run_research(session_date)
        items = await self._list_all()
        reports = [
            IPOGoNoGoAnalyzer.analyze(
                symbol=item.symbol,
                qib_times=item.qib_times,
                hni_times=item.hni_times,
                retail_times=item.retail_times,
                total_times=item.total_times,
                company_pe=item.company_pe,
                sector_pe=item.sector_pe,
                debt_to_equity=item.debt_to_equity,
            )
            for item in items
            if item.status == "bidding"
        ]
        raised = await self._remind(items, session_date=session_date, reports=reports)
        return IPOReminderOutcome(
            session_date=session_date,
            refreshed=refreshed,
            reminders_raised=raised,
            go_no_go=reports,
        )

    # ------------------------------------------------------------ internals

    async def _refresh(self, session_date: date) -> int:
        upcoming, subscriptions = await asyncio.gather(
            asyncio.to_thread(self._source.fetch_upcoming),
            asyncio.to_thread(self._source.fetch_active_subscription),
        )
        subscription_by_symbol = {s.symbol: s for s in subscriptions}
        issue_details = await asyncio.to_thread(
            self._fetch_issue_details, [i.symbol for i in upcoming]
        )

        async with self._database.session() as session:
            repository = IPORepository(session)
            for issue in upcoming:
                status = _status_for(
                    issue.bidding_opens, issue.bidding_closes, issue.listing_date, session_date
                )
                await repository.upsert_from_source(
                    symbol=issue.symbol,
                    company_name=issue.company_name,
                    category=issue.category,
                    status=status,
                    price_band_low=issue.price_band_low,
                    price_band_high=issue.price_band_high,
                    lot_size=issue.lot_size,
                    issue_size_cr=issue.issue_size_cr,
                    bidding_opens=issue.bidding_opens,
                    bidding_closes=issue.bidding_closes,
                    listing_date=issue.listing_date,
                )
                snapshot = subscription_by_symbol.get(issue.symbol)
                if snapshot is not None:
                    await repository.update_subscription(
                        issue.symbol,
                        qib_times=snapshot.qib_times,
                        hni_times=snapshot.hni_times,
                        retail_times=snapshot.retail_times,
                        total_times=snapshot.total_times,
                        as_of=snapshot.as_of,
                    )
                detail = issue_details.get(issue.symbol)
                if detail is not None:
                    await self._merge_issue_detail(repository, detail)
        return len(upcoming)

    def _fetch_issue_details(self, symbols: list[str]) -> dict[str, NSEIssueDetail]:
        details: dict[str, NSEIssueDetail] = {}
        for symbol in symbols:
            detail = self._source.fetch_issue_detail(symbol)
            if detail is not None:
                details[symbol] = detail
        return details

    @staticmethod
    async def _merge_issue_detail(repository: IPORepository, detail: NSEIssueDetail) -> None:
        subscription = detail.subscription
        await repository.update_issue_detail(
            detail.symbol,
            issue_structure=detail.issue_structure,
            anchor_shares=detail.anchor_shares,
            lead_managers=detail.lead_managers,
            registrar=detail.registrar,
            rhp_url=detail.rhp_url,
            anchor_report_url=detail.anchor_report_url,
            qib_times=subscription.qib_times if subscription else None,
            hni_times=subscription.hni_times if subscription else None,
            retail_times=subscription.retail_times if subscription else None,
            total_times=subscription.total_times if subscription else None,
            as_of=subscription.as_of if subscription else None,
        )

    async def _refresh_groww(self, *, include_closed: bool) -> int:
        listing = await asyncio.to_thread(self._groww_source.fetch_listing)
        as_of = datetime.now().astimezone()

        async with self._database.session() as session:
            repository = IPORepository(session)
            for row in listing.all_rows():
                await self._upsert_groww_listing_row(repository, row, as_of=as_of)

        detail_rows = [*listing.open, *listing.upcoming]
        if include_closed:
            detail_rows += listing.closed
        search_ids = [r.search_id for r in detail_rows if r.search_id]
        details = await asyncio.to_thread(self._groww_source.fetch_all_details, search_ids)

        async with self._database.session() as session:
            repository = IPORepository(session)
            for detail in details:
                await repository.upsert_from_groww(
                    symbol=detail.symbol,
                    company_name=detail.company_name,
                    is_sme=detail.is_sme,
                    search_id=detail.search_id,
                    logo_url=detail.logo_url,
                    sector=detail.sector,
                    registrar=detail.registrar,
                    lot_size=detail.lot_size,
                    price_band_low=detail.price_band_low,
                    price_band_high=detail.price_band_high,
                    bidding_opens=detail.bidding_opens,
                    bidding_closes=detail.bidding_closes,
                    allotment_date=detail.allotment_date,
                    listing_date=detail.listing_date,
                    listing_price=detail.listing_price,
                    qib_times=detail.qib_times,
                    hni_times=detail.hni_times,
                    retail_times=detail.retail_times,
                    total_times=detail.total_times,
                    financials=[{"title": f.title, "yearly": f.yearly} for f in detail.financials],
                    pros=detail.pros,
                    cons=detail.cons,
                    about_company=detail.about_company,
                    as_of=as_of,
                )
        return len(listing.all_rows())

    @staticmethod
    async def _upsert_groww_listing_row(
        repository: IPORepository, row: GrowwListingRow, *, as_of: datetime
    ) -> None:
        await repository.upsert_from_groww(
            symbol=row.symbol,
            company_name=row.company_name,
            is_sme=row.is_sme,
            search_id=row.search_id,
            logo_url=row.logo_url,
            lot_size=row.lot_size,
            price_band_low=row.price_band_low,
            price_band_high=row.price_band_high,
            bidding_opens=row.bidding_opens,
            bidding_closes=row.bidding_closes,
            listing_date=row.listing_date,
            listing_price=row.listing_price,
            total_times=row.overall_subscription,
            as_of=as_of,
        )

    async def _apply_fundamentals(self) -> None:
        """Re-apply the curated fundamentals file for every tracked symbol.

        Runs after both feed refreshes so it covers symbols either source
        introduced, not only the ones NSE's feed happens to list.
        """
        items = await self._list_all()
        async with self._database.session() as session:
            repository = IPORepository(session)
            for item in items:
                fundamentals = load_ipo_fundamentals(self._fundamentals_dir, item.symbol)
                if fundamentals is None:
                    continue
                as_of = datetime.combine(fundamentals.as_of, datetime.min.time())
                await repository.update_fundamentals(
                    item.symbol,
                    company_pe=fundamentals.company_pe,
                    sector_pe=fundamentals.sector_pe,
                    debt_to_equity=fundamentals.debt_to_equity,
                    source=fundamentals.source,
                    as_of=as_of.astimezone(),
                )

    async def _recompute_statuses(self, session_date: date) -> None:
        items = await self._list_all()
        async with self._database.session() as session:
            repository = IPORepository(session)
            for item in items:
                status = _status_for(
                    item.bidding_opens, item.bidding_closes, item.listing_date, session_date
                )
                if status != item.status:
                    await repository.set_status(item.symbol, status)

    async def _run_research(self, session_date: date) -> None:
        if self._research_client is None:
            return

        items = await self._list_all()
        stale_before = session_date - timedelta(days=self._research_refresh_days)
        for item in items:
            if item.status not in ("upcoming", "bidding"):
                continue
            if not (item.financials or item.about_company or item.pros or item.cons):
                # Nothing substantive scraped yet (the common case for a
                # DRHP-stage placeholder symbol) -- a note here would be five
                # sentences of "not disclosed" for a real LLM bill.
                continue
            if (
                item.ai_research_generated_at is not None
                and item.ai_research_generated_at.date() >= stale_before
            ):
                continue

            data = IPOResearchInput(
                symbol=item.symbol,
                company_name=item.company_name,
                sector=item.sector,
                category=item.category,
                price_band_low=item.price_band_low,
                price_band_high=item.price_band_high,
                issue_size_cr=item.issue_size_cr,
                issue_structure=item.issue_structure,
                anchor_shares=item.anchor_shares,
                lead_managers=item.lead_managers,
                registrar=item.registrar,
                about_company=item.about_company,
                financials=item.financials,
                pros=item.pros,
                cons=item.cons,
            )
            result = await research_ipo(self._research_client, data)
            if result is None:
                continue
            note, usage = result
            async with self._database.session() as session:
                await IPORepository(session).update_ai_research(
                    item.symbol,
                    note=note.model_dump(),
                    deployment=usage.deployment,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    as_of=datetime.now().astimezone(),
                )

    async def _list_all(self) -> list[IPOItem]:
        async with self._database.session() as session:
            return list(await IPORepository(session).list_all())

    async def _remind(
        self,
        items: list[IPOItem],
        *,
        session_date: date,
        reports: list[IPOGoNoGoReport],
    ) -> int:
        reports_by_symbol = {r.symbol: r for r in reports}
        raised = 0
        async with self._database.session() as session:
            journal = JournalRepository(session)
            for item in items:
                for event, detail in _due_reminders(item, session_date):
                    report = reports_by_symbol.get(item.symbol)
                    await journal.add(
                        JournalEntry(
                            event=event,
                            summary=f"{item.symbol}: {detail}",
                            occurred_at=datetime.now().astimezone(),
                            session_date=session_date,
                            symbol=item.symbol,
                            payload={
                                "company_name": item.company_name,
                                "go_no_go": report.verdict if report is not None else None,
                                "go_no_go_score": report.score if report is not None else None,
                            },
                        )
                    )
                    raised += 1
        return raised


def _status_for(
    bidding_opens: date | None,
    bidding_closes: date | None,
    listing_date: date | None,
    today: date,
) -> str:
    if listing_date is not None and today >= listing_date:
        return "listed"
    if bidding_opens is None:
        return "upcoming"
    if today < bidding_opens:
        return "upcoming"
    if bidding_closes is not None and today > bidding_closes:
        return "closed"
    return "bidding"


def _due_reminders(item: IPOItem, session_date: date) -> list[tuple[str, str]]:
    """Reminder events whose trigger date is exactly ``session_date``.

    Exact-day matching, not "on or after": a job that runs daily raises each
    reminder once, on the day it is due, rather than repeating it every
    session until the window closes.
    """
    due: list[tuple[str, str]] = []
    if item.bidding_opens is not None:
        advance_date = item.bidding_opens - timedelta(days=_ADVANCE_REMINDER_DAYS)
        if advance_date == session_date:
            detail = f"bidding opens {item.bidding_opens.isoformat()}"
            due.append(("ipo_bidding_opens_soon", detail))
        if item.bidding_opens == session_date:
            due.append(("ipo_bidding_opens_today", "bidding opens today"))
    if item.bidding_closes == session_date:
        due.append(("ipo_last_day_to_apply", "last day to apply"))
    if item.listing_date == session_date:
        due.append(("ipo_listing_today", "lists today"))
    return due


async def run_ipo_reminder(
    *,
    on: date,
    settings: Settings,
    database: Database | None = None,
    include_closed: bool = False,
) -> IPOReminderOutcome:
    """One reminder pass. The scheduler calls this once daily, pre-open.

    ``include_closed=True`` also scrapes every closed Groww issue's detail
    page rather than just the ones currently upcoming or bidding -- a much
    bigger one-off pull, meant for ``trading-agent ipo groww-backfill``,
    not the daily job.
    """
    owns_database = database is None
    database = database or Database.from_settings(settings)

    research_client: FoundryClient | None = None
    if settings.foundry.is_configured:
        from trading_agent.agents import cache_from_settings

        research_client = FoundryClient(settings.foundry, cache=cache_from_settings(settings))
    else:
        log.info("ipo_research_skipped", reason="foundry not configured")

    try:
        pipeline = IPOReminderPipeline(
            source=NSEIPOSource(),
            groww_source=GrowwIPOSource(settings.raw_dir),
            database=database,
            fundamentals_dir=settings.ipo_fundamentals_dir,
            research_client=research_client,
        )
        return await pipeline.tick(on, include_closed=include_closed)
    finally:
        if owns_database:
            await database.dispose()
