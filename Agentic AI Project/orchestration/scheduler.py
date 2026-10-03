"""APScheduler wiring.

Jobs are pinned to the NSE calendar rather than to wall-clock cron, so they do
not fire on exchange holidays. APScheduler has no notion of an exchange
calendar, and its ``day_of_week='mon-fri'`` gets Diwali wrong every year -- so
the cron trigger decides *when to consider running* and a calendar guard
decides *whether to actually run*. Keeping those separate is what makes the
guard testable without starting a scheduler.

Three failure modes this module exists to prevent, all of which cost money:

**Firing on a holiday.** A research run on a closed exchange debates stale
bars and produces confident nonsense, billed at full price.

**Overlapping runs.** A pre-open job that takes longer than expected must not
be started again on top of itself. ``max_instances=1`` stops that within one
process; the optional Redis lock stops it across processes, which is what
actually happens when a deployment leaves two containers briefly alive.

**Silent misfires.** A process that was down at 08:15 and starts at 08:20
should still run the pre-open research -- but one that starts at 11:00 should
not, because the session is already underway. That is a grace window, not a
retry, and it is configured per job.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from trading_agent.keyvalue import KeyValueStore
from trading_agent.market_data import IST, TradingCalendar
from trading_agent.observability import get_logger
from trading_agent.settings import Settings

__all__ = ["JobRun", "Scheduler", "SessionJob", "build_default_scheduler"]

log = get_logger(__name__)

#: IPO reminder refresh. Ahead of the bar sync -- it hits NSE's own site
#: rather than the archive host, so it is kept off the data sync's critical
#: path in case NSE's anti-bot gate is having a bad morning.
IPO_REMINDER = (7, 30)

#: Pre-open bar sync, before research reads the store. Everything downstream
#: is arithmetic over bars, so a session that starts with a stale store
#: produces a confidently empty dashboard rather than a visible failure.
DATA_SYNC = (7, 45)

#: Pre-open research. An hour before the 09:15 bell: long enough for a
#: ten-symbol debate to finish and be read, late enough that the previous
#: session's bhavcopy has landed.
PRE_OPEN = (8, 15)

#: Entry routing, just after the open. Separate from research on purpose:
#: research decides *what* to trade before the bell and costs money, routing
#: decides *how much* and needs the account snapshot, which is only meaningful
#: once the session is live. Running them as one job would size positions
#: against an equity figure taken an hour before the market existed.
ENTRY = (9, 20)

#: Inclusive hour range the intraday monitor ticks over, IST. 09:00 to 15:00
#: brackets the 09:15-15:30 session with a margin either side; the monitor's
#: own clock check makes the out-of-hours ticks no-ops rather than errors.
MONITOR_HOURS = (9, 15)

#: Post-close bar sync, ahead of the review that grades against those bars.
POST_CLOSE_SYNC = (18, 30)

#: Post-close review. Late enough that the session's bhavcopy has landed --
#: a review that runs at 15:31 has nothing to grade itself against.
POST_CLOSE = (19, 0)

#: How long after its scheduled time a job may still start. Beyond this the
#: run is skipped rather than queued: a pre-open job that begins after the
#: open is not late, it is wrong.
DEFAULT_GRACE_SECONDS = 1800

#: The distributed lock's lifetime. Longer than any job should take, short
#: enough that a crashed process does not block tomorrow's run.
LOCK_TTL_SECONDS = 6 * 3600


@dataclass(frozen=True, slots=True)
class JobRun:
    """The outcome of one attempt to run a job. Returned so it can be asserted on."""

    job: str
    session_date: date
    ran: bool
    reason: str = ""


@dataclass(frozen=True, slots=True)
class SessionJob:
    """A job pinned to the trading calendar.

    ``func`` takes the session date, so a job never has to work out "which day
    am I running for" from the clock -- a distinction that matters the moment
    a job runs a minute after midnight or is replayed for a past session.
    """

    name: str
    hour: int
    minute: int
    func: Callable[[date], Awaitable[Any]]
    #: When False the job runs every day, holidays included. Housekeeping that
    #: does not touch the market wants this; anything reading prices does not.
    trading_days_only: bool = True
    grace_seconds: int = DEFAULT_GRACE_SECONDS
    #: Extra cron fields, e.g. ``{"day_of_week": "fri"}`` for a weekly job.
    cron: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 <= self.hour <= 23:  # noqa: PLR2004 -- hours of a day
            raise ValueError(f"hour must be 0-23, got {self.hour}")
        if not 0 <= self.minute <= 59:  # noqa: PLR2004 -- minutes of an hour
            raise ValueError(f"minute must be 0-59, got {self.minute}")
        if self.grace_seconds <= 0:
            raise ValueError(f"grace_seconds must be positive, got {self.grace_seconds}")


class Scheduler:
    """Runs :class:`SessionJob` s against the NSE calendar.

    The APScheduler instance is built lazily in :meth:`start`, so constructing
    a Scheduler -- and testing every guard on it -- needs no event loop.
    """

    __slots__ = ("_calendar", "_jobs", "_lock_store", "_scheduler")

    def __init__(
        self,
        calendar: TradingCalendar,
        *,
        lock_store: KeyValueStore | None = None,
    ) -> None:
        self._calendar = calendar
        self._lock_store = lock_store
        self._jobs: list[SessionJob] = []
        self._scheduler: Any | None = None

    @property
    def jobs(self) -> list[SessionJob]:
        return list(self._jobs)

    def add(self, job: SessionJob) -> None:
        """Register a job. Names are unique -- a duplicate is a wiring bug."""
        if any(existing.name == job.name for existing in self._jobs):
            raise ValueError(f"job {job.name!r} is already registered")
        self._jobs.append(job)

    # ------------------------------------------------------------- the guard

    async def run_job(self, job: SessionJob, *, on: date | None = None) -> JobRun:
        """Run ``job`` for ``on``, subject to every guard.

        This is what the trigger calls, and what tests call directly. An
        exception inside the job is logged and reported, never raised: a
        scheduler that dies on its first bad session stops running the jobs
        that would have told you why.
        """
        session_date = on or datetime.now(IST).date()
        session = session_date.isoformat()

        if job.trading_days_only and not self._calendar.is_trading_day(session_date):
            log.info("job_skipped", job=job.name, session=session, reason="holiday")
            return JobRun(job.name, session_date, ran=False, reason="not a trading day")

        if not await self._claim(job, session_date):
            # Another process already has this session. Not an error -- it is
            # the lock doing exactly what it exists to do.
            log.info("job_skipped", job=job.name, session=session, reason="locked")
            return JobRun(
                job.name, session_date, ran=False, reason="another instance holds the lock"
            )

        log.info("job_started", job=job.name, session=session)
        try:
            await job.func(session_date)
        except Exception as exc:
            log.exception("job_failed", job=job.name, session=session)
            return JobRun(job.name, session_date, ran=False, reason=f"{type(exc).__name__}: {exc}")

        log.info("job_finished", job=job.name, session=session)
        return JobRun(job.name, session_date, ran=True)

    async def _claim(self, job: SessionJob, session_date: date) -> bool:
        """Claim this (job, session) across processes. True when we won.

        Without a store there is nothing to claim and the answer is always
        yes: a single-process deployment does not need Redis to be running.
        A store that is unreachable also answers yes -- refusing to trade
        because the lock is down is its own kind of outage, and
        ``max_instances=1`` still covers the common case.
        """
        if self._lock_store is None:
            return True
        key = f"trading_agent:job:{job.name}:{session_date.isoformat()}"
        try:
            return await self._lock_store.set_if_absent(key, "held", ttl_seconds=LOCK_TTL_SECONDS)
        except Exception:  # noqa: BLE001 -- an unreachable lock must not block the session
            log.warning("job_lock_unavailable", job=job.name, session=session_date.isoformat())
            return True

    # ------------------------------------------------------------- lifecycle

    def start(self) -> None:
        """Build the APScheduler instance, register every trigger, and start.

        Must be called from inside a running event loop: ``AsyncIOScheduler``
        binds to the loop it is started on.
        """
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.cron import CronTrigger

        if self._scheduler is not None:
            raise RuntimeError("scheduler is already running")
        if not self._jobs:
            raise ValueError("no jobs registered; a scheduler with nothing to run is a bug")

        scheduler = AsyncIOScheduler(timezone=IST)
        for job in self._jobs:
            # `cron` OVERRIDES hour/minute rather than joining them. Passing
            # both as keywords raises TypeError, and a job that wants a range
            # ("every minute from 09:00 to 15:00") has to be able to say so.
            fields = {"hour": job.hour, "minute": job.minute, **job.cron}
            scheduler.add_job(
                self._trigger_for(job),
                CronTrigger(timezone=IST, **fields),
                id=job.name,
                name=job.name,
                # One at a time. A pre-open run that overruns must not have a
                # second copy started on top of it.
                max_instances=1,
                # A backlog collapses to one run, not one per missed tick.
                coalesce=True,
                misfire_grace_time=job.grace_seconds,
            )

        scheduler.start()
        self._scheduler = scheduler
        log.info(
            "scheduler_started",
            jobs=[j.name for j in self._jobs],
            timezone=str(IST),
        )

    def _trigger_for(self, job: SessionJob) -> Callable[[], Awaitable[JobRun]]:
        async def fire() -> JobRun:
            return await self.run_job(job)

        fire.__name__ = f"fire_{job.name}"
        return fire

    def next_run_times(self) -> dict[str, datetime | None]:
        """When each job fires next. For the CLI and the health endpoint."""
        if self._scheduler is None:
            return {job.name: None for job in self._jobs}
        return {job.id: getattr(job, "next_run_time", None) for job in self._scheduler.get_jobs()}

    def shutdown(self, *, wait: bool = True) -> None:
        if self._scheduler is None:
            return
        self._scheduler.shutdown(wait=wait)
        self._scheduler = None
        log.info("scheduler_stopped")

    async def serve_forever(self) -> None:
        """Start and block until cancelled. The container entry point."""
        self.start()
        try:
            await asyncio.Event().wait()
        finally:
            self.shutdown()


def _register_ipo_reminder(
    scheduler: Scheduler,
    settings: Settings,
    run_ipo_reminder: Callable[..., Awaitable[Any]],
) -> None:
    """Register the IPO reminder job. Unconditional -- it needs no broker.

    A free function rather than inlined in :func:`build_default_scheduler`
    only to keep that function under the statement-count lint.
    """

    async def ipo_reminder(on: date) -> None:
        await run_ipo_reminder(on=on, settings=settings)

    scheduler.add(
        SessionJob(
            name="ipo_reminder",
            hour=IPO_REMINDER[0],
            minute=IPO_REMINDER[1],
            func=ipo_reminder,
        )
    )


def build_default_scheduler(
    settings: Settings,
    *,
    lock_store: KeyValueStore | None = None,
    broker: Any | None = None,
    router: Any | None = None,
    live_acknowledged: bool = False,
) -> Scheduler:
    """The daily jobs, in the order the session runs them.

    The intraday monitor is registered only when a ``broker`` is supplied.
    Without one there is no book to watch, and a job that fails every minute
    of every session teaches everyone to ignore the alert.

    The monitor here is **observe-only** unless a ``router`` is supplied: it
    reports exits and journals them rather than placing them. Arming it is a
    deliberate act, not a default.

    ``router`` also registers the entry job. Without one the system researches
    and observes but never trades -- which was the state before Phase 5: every
    piece present, nothing joining signals to orders.
    """
    from trading_agent.market_data import BarStore
    from trading_agent.orchestration.pipelines.daily_research import run_daily_research
    from trading_agent.orchestration.pipelines.data_sync import run_data_sync
    from trading_agent.orchestration.pipelines.intraday_monitor import run_intraday_monitor
    from trading_agent.orchestration.pipelines.ipo_reminder import run_ipo_reminder
    from trading_agent.orchestration.pipelines.paper_session import run_paper_session
    from trading_agent.orchestration.pipelines.post_close_review import run_post_close_review
    from trading_agent.persistence import Database
    from trading_agent.risk import KillSwitch

    calendar = TradingCalendar.from_csv(
        settings.holidays_path,
        session_open=settings.market.session_open,
        session_close=settings.market.session_close,
    )
    scheduler = Scheduler(calendar, lock_store=lock_store)
    _register_ipo_reminder(scheduler, settings, run_ipo_reminder)

    async def sync(on: date) -> None:
        # Blocking HTTP and Parquet work, so it runs on a worker thread rather
        # than stalling the event loop the monitor also ticks on.
        await asyncio.to_thread(run_data_sync, on=on, settings=settings)

    # Twice a session. The pre-open run is the catch-up (a weekend, a redeploy,
    # a VM that was off); the post-close run picks up the session that just
    # ended, once the archive has published it.
    scheduler.add(
        SessionJob(
            name="data_sync_pre_open",
            hour=DATA_SYNC[0],
            minute=DATA_SYNC[1],
            func=sync,
            # Worth replaying late: research an hour later still benefits, and
            # a store left stale is the failure this job exists to prevent.
            grace_seconds=3600,
        )
    )
    scheduler.add(
        SessionJob(
            name="data_sync_post_close",
            hour=POST_CLOSE_SYNC[0],
            minute=POST_CLOSE_SYNC[1],
            func=sync,
            grace_seconds=3 * 3600,
        )
    )

    async def research(on: date) -> None:
        await run_daily_research(on=on, settings=settings)

    scheduler.add(
        SessionJob(name="daily_research", hour=PRE_OPEN[0], minute=PRE_OPEN[1], func=research)
    )

    if broker is not None:
        if lock_store is None:
            raise ValueError(
                "the intraday monitor needs a lock_store: the kill switch is "
                "Redis-backed, and a monitor that cannot read it would trade "
                "through a stop that had already been pulled"
            )
        kill_switch = KillSwitch(lock_store, key=settings.risk.kill_switch_key)

        if router is not None:

            async def entries(on: date) -> None:
                database = Database.from_settings(settings)
                try:
                    await run_paper_session(
                        on=on,
                        broker=broker,
                        router=router,
                        database=database,
                        settings=settings,
                        live_acknowledged=live_acknowledged,
                    )
                finally:
                    await database.dispose()

            scheduler.add(
                SessionJob(name="entry_routing", hour=ENTRY[0], minute=ENTRY[1], func=entries)
            )

        async def monitor(on: date) -> None:  # noqa: ARG001 -- the tick reads its own clock
            database = Database.from_settings(settings)
            try:
                await run_intraday_monitor(
                    broker=broker,
                    calendar=calendar,
                    kill_switch=kill_switch,
                    database=database,
                    settings=settings,
                    router=router,
                    live_acknowledged=live_acknowledged,
                )
            finally:
                await database.dispose()

        # Every minute of the session. The cron fields do the windowing; the
        # monitor's own clock check is what makes an out-of-hours tick a no-op.
        scheduler.add(
            SessionJob(
                name="intraday_monitor",
                hour=MONITOR_HOURS[0],
                minute=0,
                func=monitor,
                # A missed tick is not worth replaying: the next one is a
                # minute away and carries fresher marks.
                grace_seconds=60,
                cron={"hour": f"{MONITOR_HOURS[0]}-{MONITOR_HOURS[1]}", "minute": "*"},
            )
        )

    async def review(on: date) -> None:
        database = Database.from_settings(settings)
        store = BarStore(settings.curated_dir, interval=settings.data.bar_interval)
        try:
            await run_post_close_review(on=on, store=store, database=database, settings=settings)
        finally:
            await database.dispose()

    scheduler.add(
        SessionJob(
            name="post_close_review",
            hour=POST_CLOSE[0],
            minute=POST_CLOSE[1],
            func=review,
            # The bhavcopy for the session lands well after the close, and a
            # review that runs before it has nothing to grade against.
            grace_seconds=4 * 3600,
        )
    )
    return scheduler
