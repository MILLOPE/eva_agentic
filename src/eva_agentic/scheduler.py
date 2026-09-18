"""Local, resource-capped scheduling for a single experiment run."""

from __future__ import annotations

import fcntl
import inspect
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from eva_agentic.schema import AttemptStatus, EpisodeResult, Job
from eva_agentic.store import attempt_directory, commit_attempt, prepare_attempt
from eva_agentic.resources import ResourceAllocator


AttemptRunner = Callable[[Job, Path], "AttemptExecution"]


@dataclass(frozen=True)
class AttemptExecution:
    status: AttemptStatus
    process: Mapping[str, Any] = field(default_factory=dict)
    results: tuple[EpisodeResult, ...] = ()


class RunLock:
    """A non-reentrant, host-local advisory lock for one run directory."""

    def __init__(self, run_dir: str | Path) -> None:
        self.path = Path(run_dir) / ".eva" / "scheduler.lock"
        self._fd: int | None = None

    def __enter__(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(self.path, os.O_CREAT | os.O_RDWR, mode=0o600)
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as error:
            os.close(self._fd)
            self._fd = None
            raise RuntimeError(f"run is already being scheduled: {self.path}") from error
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None


def run_jobs(
    run_dir: str | Path,
    jobs: Sequence[Job],
    runner: AttemptRunner,
    max_workers: int,
    resource_allocator: ResourceAllocator | None = None,
    resource_requirements: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, AttemptStatus]:
    """Run unfinished jobs in parallel; completed attempts are never rerun."""
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")

    statuses: dict[str, AttemptStatus] = {}
    with RunLock(run_dir):
        pending = [job for job in jobs if not _job_is_complete(run_dir, job)]
        for job in jobs:
            if _job_is_complete(run_dir, job):
                statuses[job.job_id] = AttemptStatus.COMPLETED

        workers = min(max_workers, len(pending))
        if workers == 0:
            return statuses

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="eva-job") as pool:
            futures = {
                pool.submit(
                    _run_one,
                    run_dir,
                    job,
                    runner,
                    resource_allocator,
                    resource_requirements,
                ): job
                for job in pending
            }
            for future in as_completed(futures):
                job = futures[future]
                statuses[job.job_id] = future.result()

    return statuses


def _run_one(
    run_dir: str | Path,
    job: Job,
    runner: AttemptRunner,
    resource_allocator: ResourceAllocator | None,
    resource_requirements: Mapping[str, Mapping[str, int]] | None,
) -> AttemptStatus:
    requirements = (resource_requirements or {}).get(job.job_id, {})
    lease = (
        resource_allocator.allocate(requirements, timeout_s=3600)
        if resource_allocator and requirements
        else None
    )
    allocated_resources = [
        grant.to_dict() for grant in (lease.grants if lease else ())
    ]
    try:
        attempt_dir = _prepare_next_attempt(
            run_dir,
            job,
            request={"allocated_resources": allocated_resources},
        )
        execution = _invoke_runner(
            runner, job, attempt_dir, lease.grants if lease is not None else ()
        )
        commit_attempt(
            attempt_dir,
            status=execution.status,
            process=execution.process,
            results=execution.results,
        )
    except Exception as error:
        if "attempt_dir" in locals():
            commit_attempt(
                attempt_dir,
                status=AttemptStatus.FAILED,
                process={
                    "exception": type(error).__name__,
                    "message": str(error),
                },
            )
        return AttemptStatus.FAILED
    finally:
        if lease is not None:
            lease.release()
    return execution.status


def _prepare_next_attempt(
    run_dir: str | Path,
    job: Job,
    request: Mapping[str, Any] | None = None,
) -> Path:
    root = attempt_directory(run_dir, job.participant, job.job_id, attempt_id=1).parent
    for _ in range(1000):
        attempt_id = _next_attempt_id(root)
        try:
            return prepare_attempt(run_dir, job, attempt_id, request)
        except FileExistsError:
            continue
    raise RuntimeError(f"too many attempts for job: {job.job_id}")


def _job_is_complete(run_dir: str | Path, job: Job) -> bool:
    attempts_root = attempt_directory(
        run_dir, job.participant, job.job_id, attempt_id=1
    ).parent
    if not attempts_root.exists():
        return False
    for attempt_dir in attempts_root.iterdir():
        result_path = attempt_dir / "result.json"
        if not attempt_dir.is_dir() or not result_path.is_file():
            continue
        try:
            with result_path.open("r", encoding="utf-8") as handle:
                result = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return False
        if result.get("status") != AttemptStatus.COMPLETED.value:
            continue
        return True
    return False


def _next_attempt_id(attempts_root: Path) -> int:
    highest = 0
    if attempts_root.exists():
        for path in attempts_root.iterdir():
            if path.is_dir() and path.name.isdigit():
                highest = max(highest, int(path.name))
    return highest + 1


def _invoke_runner(
    runner: AttemptRunner,
    job: Job,
    attempt_dir: Path,
    grants: tuple[object, ...],
) -> AttemptExecution:
    """Support legacy two-argument runners and resource-aware runners."""
    try:
        inspect.signature(runner).bind(job, attempt_dir, grants)
    except TypeError:
        return runner(job, attempt_dir)
    return runner(job, attempt_dir, grants)
