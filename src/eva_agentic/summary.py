"""Read-only summaries of normalized evaluation attempts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eva_agentic.schema import AttemptStatus, OutcomeStatus
from eva_agentic.store import attempt_directory, load_run


def summarize_run(run_dir: str | Path) -> dict[str, object]:
    """Summarize one frozen run without altering its attempts."""
    inputs = load_run(run_dir)
    counts = {
        "planned_jobs": len(inputs.plan.jobs),
        "completed_jobs": 0,
        "successes": 0,
        "task_failures": 0,
        "invalid": 0,
        "infrastructure_failures": 0,
        "unstarted": 0,
    }
    for job in inputs.plan.jobs:
        attempt = terminal_attempt(inputs.run_dir, job.participant, job.job_id)
        if attempt is None:
            counts["unstarted"] += 1
            continue
        if attempt.get("status") != AttemptStatus.COMPLETED.value:
            counts["infrastructure_failures"] += 1
            continue
        counts["completed_jobs"] += 1
        results = attempt.get("results")
        if not isinstance(results, list) or not results:
            counts["invalid"] += 1
            continue
        status = results[0].get("status") if isinstance(results[0], dict) else None
        if status == OutcomeStatus.SUCCESS.value:
            counts["successes"] += 1
        elif status == OutcomeStatus.TASK_FAILURE.value:
            counts["task_failures"] += 1
        else:
            counts["invalid"] += 1
    denominator = counts["successes"] + counts["task_failures"]
    return {
        **counts,
        "success_rate": counts["successes"] / denominator if denominator else None,
        "success_rate_denominator": denominator,
    }


def terminal_attempt(run_dir: str | Path, participant: str, job_id: str) -> dict[str, Any] | None:
    """Return the terminal attempt result for one job, prioritizing completed attempts."""
    run_dir = Path(run_dir)
    root = attempt_directory(run_dir, participant, job_id, 1).parent
    if not root.is_dir():
        return None
    attempts: list[tuple[int, dict[str, Any]]] = []
    for directory in root.iterdir():
        result_path = directory / "result.json"
        if not directory.is_dir() or not directory.name.isdigit() or not result_path.is_file():
            continue
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            attempts.append((int(directory.name), data))
    if not attempts:
        return None
    completed = [item for item in attempts if item[1].get("status") == AttemptStatus.COMPLETED.value]
    return max(completed or attempts, key=lambda item: item[0])[1]
