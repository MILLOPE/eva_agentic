"""Native RPent adapter for one shared-LIBERO evaluation case."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from eva_agentic.frameworks import FrameworkProfile, FrameworkSpec, NativeLaunch, resolve_native_launch
from eva_agentic.process import NativeRunResult
from eva_agentic.resources import ResourceGrant
from eva_agentic.schema import EpisodeResult, Job, OutcomeStatus


class RpentLiberoAdapter:
    """Connect RPent to external VLA and SAM3 services for one LIBERO case."""

    name = "rpent_libero"

    def __init__(self, spec: FrameworkSpec) -> None:
        self.spec = spec

    def build_launch(
        self,
        job: Job,
        attempt_dir: Path,
        profile: FrameworkProfile,
        grants: Sequence[ResourceGrant],
    ) -> NativeLaunch:
        case = _one_case(job)
        suite, task_id, max_episode_steps = _conditions(case.initialization)
        local = profile.frameworks.get(self.name)
        if local is None:
            raise ValueError(f"local profile has no framework: {self.name}")
        vla_endpoint = _endpoint(local.env, "RPENT_VLA_ENDPOINT")
        sam3_endpoint = _endpoint(local.env, "RPENT_SAM3_ENDPOINT")
        command = (
            *self.spec.command,
            "--robot",
            "libero",
            "--suite",
            suite,
            "--task",
            str(task_id),
            "--seed",
            "{seed}",
            "--max-episode-steps",
            str(max_episode_steps),
            "--output-dir",
            "{output_dir}",
            "--vla-endpoint",
            vla_endpoint,
            "--sam3-endpoint",
            sam3_endpoint,
        )
        return resolve_native_launch(
            replace(self.spec, command=command), profile, job, attempt_dir, grants
        )

    def parse(
        self, job: Job, attempt_dir: Path, run: NativeRunResult
    ) -> tuple[EpisodeResult, ...]:
        case = _one_case(job)
        suite, task_id, max_episode_steps = _conditions(case.initialization)
        audit_path = Path(attempt_dir) / "native" / _recipe_tag(suite, task_id, case.seed)
        audit_path = audit_path.with_suffix(".json")
        evidence = [str(audit_path)]
        recipe_path = audit_path.with_name(f"{audit_path.stem}_recipe.jsonl")
        if recipe_path.is_file():
            evidence.append(str(recipe_path))
        conditions = {
            "suite": suite,
            "task_id": task_id,
            "seed": case.seed,
            "max_episode_steps": max_episode_steps,
            "vla_service": "external",
            "sam3_service": "external",
        }
        if run.exit_code != 0:
            return (
                EpisodeResult(
                    case_id=case.case_id,
                    status=OutcomeStatus.INFRASTRUCTURE_FAILURE,
                    task_success=None,
                    error_source="rpent.process",
                    termination_reason=f"native RPent command exited {run.exit_code}",
                    requested_conditions=conditions,
                    effective_conditions=conditions,
                    metrics={"native_exit_code": run.exit_code},
                    evidence_paths=tuple(evidence),
                ),
            )
        try:
            audit = _load_audit(audit_path, suite, task_id, case.seed)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            return (
                EpisodeResult(
                    case_id=case.case_id,
                    status=OutcomeStatus.INVALID,
                    task_success=None,
                    error_source="rpent.audit",
                    termination_reason=str(error),
                    requested_conditions=conditions,
                    effective_conditions=conditions,
                    metrics={"native_exit_code": run.exit_code},
                    evidence_paths=tuple(evidence),
                ),
            )
        success = audit["terminated"]
        effective = dict(conditions)
        if isinstance(audit.get("regime"), str):
            effective["regime"] = audit["regime"]
        return (
            EpisodeResult(
                case_id=case.case_id,
                status=OutcomeStatus.SUCCESS if success else OutcomeStatus.TASK_FAILURE,
                task_success=success,
                success_source="rpent.audit.terminated",
                requested_conditions=conditions,
                effective_conditions=effective,
                metrics={
                    "native_exit_code": run.exit_code,
                    "native_truncated": _bool_as_int(audit.get("truncated")),
                },
                evidence_paths=tuple(evidence),
            ),
        )


def _one_case(job: Job):
    if len(job.cases) != 1:
        raise ValueError("RPent LIBERO adapter requires one case per job")
    return job.cases[0]


def _conditions(initialization: dict[str, Any] | Any) -> tuple[str, int, int]:
    suite = initialization.get("suite")
    task_id = initialization.get("task_id")
    max_episode_steps = initialization.get("max_episode_steps")
    if not isinstance(suite, str) or not suite:
        raise ValueError("RPent LIBERO case initialization requires suite")
    if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id < 0:
        raise ValueError("RPent LIBERO case initialization requires non-negative integer task_id")
    if (
        isinstance(max_episode_steps, bool)
        or not isinstance(max_episode_steps, int)
        or max_episode_steps <= 0
    ):
        raise ValueError("RPent LIBERO case initialization requires positive max_episode_steps")
    return suite, task_id, max_episode_steps


def _endpoint(env: dict[str, str] | Any, key: str) -> str:
    value = env.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"RPent LIBERO profile requires {key}")
    return value


def _recipe_tag(suite: str, task_id: int, seed: int) -> str:
    return f"{suite.removeprefix('libero_')}_t{task_id}_s{seed}"


def _load_audit(path: Path, suite: str, task_id: int, seed: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        audit = json.load(handle)
    if not isinstance(audit, dict):
        raise ValueError("RPent audit must be a JSON object")
    expected = {"suite": suite, "task_id": task_id, "seed": seed}
    for key, value in expected.items():
        if audit.get(key) != value:
            raise ValueError(f"RPent audit {key} does not match requested condition")
    if not isinstance(audit.get("terminated"), bool):
        raise ValueError("RPent audit must contain boolean terminated")
    return audit


def _bool_as_int(value: object) -> int | None:
    return int(value) if isinstance(value, bool) else None
