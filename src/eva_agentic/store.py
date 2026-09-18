"""Atomic storage helpers and run-directory initialization."""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from eva_agentic.experiment import build_plan, load_cases, resolve_experiment
from eva_agentic.snapshot import (
    MANIFEST_NAME,
    create_input_manifest,
    verify_input_manifest,
)
from eva_agentic.schema import Case, Experiment, Plan
from eva_agentic.schema import (
    Attempt,
    AttemptStatus,
    EpisodeResult,
    Job,
    SCHEMA_VERSION,
)


RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SAFE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class RunInputs:
    run_dir: Path
    experiment: Experiment
    cases: tuple[Case, ...]
    plan: Plan


def atomic_write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    """Atomically write a JSON object, replacing any existing file."""
    _atomic_write(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def atomic_write_jsonl(path: str | Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Atomically write JSON Lines, replacing any existing file."""
    content = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    _atomic_write(path, content)


def generate_run_id(now: datetime | None = None) -> str:
    """Generate a sortable local-time run identifier."""
    timestamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def initialize_run(
    runs_root: str | Path,
    experiment: Experiment,
    run_id: str | None = None,
) -> Path:
    """Create an immutable input area for one experiment run."""
    final_run_id = run_id or generate_run_id()
    if not RUN_ID_PATTERN.fullmatch(final_run_id):
        raise ValueError("run_id contains unsupported characters")

    run_dir = Path(runs_root) / final_run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")

    cases = load_cases(experiment.cases_file)
    plan = build_plan(experiment, cases)

    inputs_dir = run_dir / "inputs"
    inputs_dir.mkdir(parents=True, mode=0o755)
    try:
        atomic_write_json(inputs_dir / "experiment.resolved.json", experiment.to_dict())
        atomic_write_jsonl(inputs_dir / "cases.jsonl", [case.to_dict() for case in cases])
        atomic_write_json(inputs_dir / "plan.json", plan.to_dict())
        atomic_write_json(inputs_dir / MANIFEST_NAME, create_input_manifest(inputs_dir))
    except Exception:
        run_dir.rmdir()
        raise
    return run_dir


def load_run(run_dir: str | Path) -> RunInputs:
    """Load and verify the frozen inputs of an existing run."""
    directory = Path(run_dir)
    inputs_dir = directory / "inputs"
    manifest_path = inputs_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run input manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"invalid input manifest: {manifest_path}")
    verify_input_manifest(inputs_dir, manifest)

    with (inputs_dir / "experiment.resolved.json").open("r", encoding="utf-8") as handle:
        experiment_data = json.load(handle)
    experiment = resolve_experiment(experiment_data)
    cases = load_cases(inputs_dir / "cases.jsonl")

    rebuilt_plan = build_plan(experiment, cases)
    with (inputs_dir / "plan.json").open("r", encoding="utf-8") as handle:
        stored_plan = json.load(handle)
    if rebuilt_plan.to_dict() != stored_plan:
        raise ValueError("stored execution plan does not match resolved inputs")

    return RunInputs(
        run_dir=directory,
        experiment=experiment,
        cases=cases,
        plan=rebuilt_plan,
    )


def prepare_attempt(
    run_dir: str | Path,
    job: Job,
    attempt_id: int,
    request: Mapping[str, Any] | None = None,
) -> Path:
    """Create a unique attempt directory and freeze its request context."""
    attempt_dir = attempt_directory(
        run_dir=run_dir,
        participant=job.participant,
        job_id=job.job_id,
        attempt_id=attempt_id,
    )
    attempt_dir.mkdir(parents=True, exist_ok=False)
    request_data = {
        "schema_version": SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "job": job.to_dict(),
        "request": dict(request or {}),
    }
    atomic_write_json(attempt_dir / "request.json", request_data)
    return attempt_dir


def commit_attempt(
    attempt_dir: str | Path,
    status: AttemptStatus | str,
    process: Mapping[str, Any],
    results: Sequence[EpisodeResult] = (),
) -> Path:
    """Commit attempt evidence once; the result file is never overwritten."""
    directory = Path(attempt_dir)
    request_path = directory / "request.json"
    with request_path.open("r", encoding="utf-8") as handle:
        request_data = json.load(handle)

    job_data = request_data["job"]
    requested_case_ids = {
        case["case_id"] for case in job_data.get("cases", [])
    }
    unknown_cases = [
        result.case_id for result in results if result.case_id not in requested_case_ids
    ]
    if unknown_cases:
        raise ValueError(f"results contain unknown case_ids: {sorted(unknown_cases)}")

    attempt = Attempt(
        participant=job_data["participant"],
        job_id=job_data["job_id"],
        attempt_id=int(request_data["attempt_id"]),
        status=AttemptStatus(status),
        process=process,
        results=tuple(results),
    )
    result_path = directory / "result.json"
    write_json_no_replace(result_path, attempt.to_dict())
    return result_path


def attempt_directory(
    run_dir: str | Path,
    participant: str,
    job_id: str,
    attempt_id: int,
) -> Path:
    if isinstance(attempt_id, bool) or not isinstance(attempt_id, int) or attempt_id <= 0:
        raise ValueError("attempt_id must be positive")
    for name, value in (("participant", participant), ("job_id", job_id)):
        if not SAFE_NAME_PATTERN.fullmatch(value):
            raise ValueError(f"{name} contains unsupported path characters")
    return (
        Path(run_dir)
        / "participants"
        / participant
        / "jobs"
        / job_id
        / "attempts"
        / f"{attempt_id:04d}"
    )


def _atomic_write(path: str | Path, content: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def write_json_no_replace(path: str | Path, data: Mapping[str, Any]) -> None:
    """Atomically create a JSON file, raising FileExistsError if it exists."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, destination)
        except FileExistsError:
            raise FileExistsError(f"result file already exists: {destination}") from None
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
