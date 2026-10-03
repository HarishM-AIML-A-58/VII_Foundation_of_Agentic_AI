"""In-process backtest job registry.

Completed metrics are persisted to ``backtest_runs`` by the CLI path this
invokes. Job *status* is process-local: a restart loses queued/running
handles, not recorded results.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID, uuid4

from trading_agent.observability import get_logger

__all__ = ["BacktestJob", "get_job", "submit_job"]

log = get_logger(__name__)

Status = Literal["queued", "running", "completed", "failed"]

_LOCK = threading.Lock()
_JOBS: dict[UUID, BacktestJob] = {}


@dataclass
class BacktestJob:
    job_id: UUID
    status: Status
    strategy: str
    walk_forward: bool
    start: str
    end: str | None
    created_at: datetime
    finished_at: datetime | None = None
    run_id: str | None = None
    error: str | None = None
    detail: str = ""


def get_job(job_id: UUID) -> BacktestJob | None:
    with _LOCK:
        return _JOBS.get(job_id)


def submit_job(
    *,
    strategy: str,
    walk_forward: bool,
    start: str,
    end: str | None,
) -> BacktestJob:
    job = BacktestJob(
        job_id=uuid4(),
        status="queued",
        strategy=strategy,
        walk_forward=walk_forward,
        start=start,
        end=end,
        created_at=datetime.now(UTC),
    )
    with _LOCK:
        _JOBS[job.job_id] = job
    thread = threading.Thread(target=_run, args=(job.job_id,), daemon=True)
    thread.start()
    return job


def _run(job_id: UUID) -> None:
    with _LOCK:
        job = _JOBS[job_id]
        job.status = "running"
        strategy = job.strategy
        walk = job.walk_forward
        start = job.start
        end = job.end

    cmd = [
        sys.executable,
        "-m",
        "trading_agent",
        "backtest",
        "run",
        "--strategy",
        strategy,
        "--from",
        start,
    ]
    if end:
        cmd.extend(["--to", end])
    if walk:
        cmd.append("--walk-forward")

    log.info("backtest_job_start", job_id=str(job_id), cmd=cmd)
    try:
        completed = subprocess.run(  # noqa: S603 -- argv list, no shell
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=60 * 30,
        )
    except subprocess.TimeoutExpired:
        _finish(job_id, "failed", error="backtest exceeded 30 minute timeout")
        return
    except Exception as exc:  # noqa: BLE001 -- any failure must land on the job record, not vanish
        _finish(job_id, "failed", error=f"{type(exc).__name__}: {exc}"[:400])
        return

    output = (completed.stdout or "") + "\n" + (completed.stderr or "")
    run_id = _parse_run_id(output)
    if completed.returncode != 0:
        _finish(
            job_id,
            "failed",
            error=(completed.stderr or completed.stdout or f"exit {completed.returncode}")[:800],
        )
        return
    _finish(job_id, "completed", run_id=run_id, detail=output[-500:])


def _parse_run_id(output: str) -> str | None:
    for line in reversed(output.splitlines()):
        if "recorded as " in line:
            return line.rsplit("recorded as ", 1)[-1].strip()
    return None


def _finish(
    job_id: UUID,
    status: Status,
    *,
    run_id: str | None = None,
    error: str | None = None,
    detail: str = "",
) -> None:
    with _LOCK:
        job = _JOBS[job_id]
        job.status = status
        job.finished_at = datetime.now(UTC)
        job.run_id = run_id
        job.error = error
        job.detail = detail
    log.info("backtest_job_done", job_id=str(job_id), status=status, run_id=run_id)


def job_view(job: BacktestJob) -> dict[str, Any]:
    return {
        "job_id": str(job.job_id),
        "status": job.status,
        "strategy": job.strategy,
        "walk_forward": job.walk_forward,
        "start": job.start,
        "end": job.end,
        "created_at": job.created_at.isoformat(),
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "run_id": job.run_id,
        "error": job.error,
    }
