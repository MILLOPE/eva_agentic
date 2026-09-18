"""Bridge a framework adapter to the generic scheduler."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from eva_agentic.adapters.base import FrameworkAdapter, UnsupportedCondition
from eva_agentic.frameworks import FrameworkProfile
from eva_agentic.process import NativeRunStatus, run_native_launch
from eva_agentic.resources import ResourceGrant
from eva_agentic.scheduler import AttemptExecution
from eva_agentic.schema import AttemptStatus, EpisodeResult, Job, OutcomeStatus


def run_adapter_job(
    adapter: FrameworkAdapter,
    profile: FrameworkProfile,
    timeout_s: float,
    job: Job,
    attempt_dir: Path,
    grants: Sequence[ResourceGrant] = (),
) -> AttemptExecution:
    """Launch and parse one native framework evaluation attempt."""
    try:
        launch = adapter.build_launch(job, attempt_dir, profile, grants)
    except UnsupportedCondition as error:
        if len(job.cases) != 1:
            raise
        case = job.cases[0]
        return AttemptExecution(
            status=AttemptStatus.COMPLETED,
            process={"native_status": "unsupported_condition", "message": str(error)},
            results=(
                EpisodeResult(
                    case_id=case.case_id,
                    status=OutcomeStatus.INVALID,
                    task_success=None,
                    error_source="adapter.conditions",
                    termination_reason=str(error),
                    requested_conditions={"seed": case.seed},
                ),
            ),
        )
    run = run_native_launch(launch, timeout_s=timeout_s, resource_grants=grants)
    process = {
        "native_status": run.status.value,
        "exit_code": run.exit_code,
        "duration_s": run.duration_s,
        "pid": run.pid,
        "terminated": run.terminated,
        "launch": str(run.launch_path),
        "runtime": str(run.runtime_path),
        "stdout": str(run.stdout_path),
        "stderr": str(run.stderr_path),
        "allocated_resources": [grant.to_dict() for grant in grants],
    }
    if run.status is not NativeRunStatus.COMPLETED:
        return AttemptExecution(status=AttemptStatus.FAILED, process=process)
    return AttemptExecution(
        status=AttemptStatus.COMPLETED,
        process=process,
        results=adapter.parse(job, attempt_dir, run),
    )
