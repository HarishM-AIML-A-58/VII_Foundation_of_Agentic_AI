"""Scheduling and the daily pipelines.

The scheduler decides *when*; the pipelines decide *what*. Keeping them apart
means a pipeline can be run by hand for any past session -- which is how you
debug one -- without going anywhere near a trigger.
"""

from trading_agent.orchestration.pipelines.daily_research import (
    DailyResearch,
    ResearchOutcome,
    run_daily_research,
)
from trading_agent.orchestration.scheduler import (
    JobRun,
    Scheduler,
    SessionJob,
    build_default_scheduler,
)

__all__ = [
    "DailyResearch",
    "JobRun",
    "ResearchOutcome",
    "Scheduler",
    "SessionJob",
    "build_default_scheduler",
    "run_daily_research",
]
