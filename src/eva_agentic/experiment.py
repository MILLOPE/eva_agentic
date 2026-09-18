"""Experiment loading and fixed task planning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from eva_agentic.schema import (
    ArtifactPolicy,
    Budget,
    Case,
    Execution,
    Experiment,
    Job,
    Plan,
    Protocol,
    RunMode,
    SCHEMA_VERSION,
)

from eva_agentic.schema import Case


def load_cases(path: str | Path) -> tuple[Case, ...]:
    """Load an ordered case list from JSON Lines and reject duplicate IDs."""
    cases: list[Case] = []
    seen_ids: set[str] = set()
    source = Path(path)

    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{source}:{line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(item, dict):
                raise ValueError(f"{source}:{line_number}: case must be a JSON object")

            try:
                case = Case(**item)
            except TypeError as error:
                raise ValueError(f"{source}:{line_number}: invalid case fields: {error}") from error
            except ValueError as error:
                raise ValueError(f"{source}:{line_number}: {error}") from error

            if case.case_id in seen_ids:
                raise ValueError(f"{source}:{line_number}: duplicate case_id: {case.case_id}")
            seen_ids.add(case.case_id)
            cases.append(case)

    if not cases:
        raise ValueError(f"{source}: case list must not be empty")
    return tuple(cases)


def resolve_experiment(data: Mapping[str, Any]) -> Experiment:
    """Validate an already-expanded experiment mapping."""
    if not isinstance(data, Mapping):
        raise ValueError("experiment must be a mapping")

    protocol = data.get("protocol")
    execution = data.get("execution")
    artifacts = data.get("artifacts")
    if not isinstance(protocol, Mapping):
        raise ValueError("protocol must be a mapping")
    if not isinstance(execution, Mapping):
        raise ValueError("execution must be a mapping")
    if not isinstance(artifacts, Mapping):
        raise ValueError("artifacts must be a mapping")

    budget_data = protocol.get("budget", {})
    if not isinstance(budget_data, Mapping):
        raise ValueError("protocol.budget must be a mapping")

    try:
        mode = RunMode(execution["mode"])
    except (KeyError, ValueError) as error:
        raise ValueError("execution.mode must be 'debug' or 'frozen'") from error

    try:
        budget = Budget(
            max_control_steps=budget_data.get("max_control_steps"),
            episode_timeout_s=budget_data.get("episode_timeout_s"),
        )
        return Experiment(
            schema_version=data["schema_version"],
            name=data["name"],
            benchmark=data["benchmark"],
            cases_file=data["cases_file"],
            participants=tuple(data["participants"]),
            protocol=Protocol(
                track=protocol["track"],
                phase=protocol["phase"],
                memory_policy=protocol["memory_policy"],
                scoring=protocol["scoring"],
                budget=budget,
            ),
            execution=Execution(
                mode=mode,
                max_jobs=execution["max_jobs"],
                max_infrastructure_retries=execution["max_infrastructure_retries"],
                job_timeout_s=execution["job_timeout_s"],
            ),
            artifacts=ArtifactPolicy(
                video=artifacts["video"],
                trace=artifacts["trace"],
            ),
        )
    except KeyError as error:
        raise ValueError(f"missing experiment field: {error.args[0]}") from error
    except (TypeError, ValueError) as error:
        if str(error).startswith("missing experiment field:"):
            raise
        raise ValueError(f"invalid experiment: {error}") from error


def load_experiment(path: str | Path) -> Experiment:
    """Load an experiment configuration from JSON or YAML."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        if source.suffix.lower() == ".json":
            data = json.load(handle)
        elif source.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as error:
                raise RuntimeError(
                    "PyYAML is required to load YAML experiment configurations"
                ) from error
            data = yaml.safe_load(handle)
        else:
            raise ValueError(f"unsupported experiment config format: {source}")
    # A config should be runnable from any working directory. Keep the
    # experiment file as the anchor for its case list instead of depending on
    # the caller's current directory.
    if isinstance(data, dict) and isinstance(data.get("cases_file"), str):
        cases_file = Path(data["cases_file"])
        if not cases_file.is_absolute():
            data = dict(data)
            data["cases_file"] = str((source.parent / cases_file).resolve())
    try:
        return resolve_experiment(data)
    except ValueError as error:
        raise ValueError(f"{source}: {error}") from error


def build_plan(experiment: Experiment, cases: Sequence[Case]) -> Plan:
    """Build the first-version plan: one participant/case job per planned trial."""
    if not cases:
        raise ValueError("cannot build a plan without cases")

    jobs: list[Job] = []
    job_number = 1
    known_participants = set(experiment.participants)
    for participant in experiment.participants:
        if participant not in known_participants:
            raise ValueError(f"unknown participant: {participant}")
        for case in cases:
            jobs.append(
                Job(
                    job_id=f"job_{job_number:04d}",
                    participant=participant,
                    cases=(case,),
                )
            )
            job_number += 1

    return Plan(
        schema_version=SCHEMA_VERSION,
        experiment_name=experiment.name,
        jobs=tuple(jobs),
    )
