"""Native RATs LIBERO-Pro evaluation adapter."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from eva_agentic.adapters.base import UnsupportedCondition
from eva_agentic.frameworks import FrameworkProfile, FrameworkSpec, NativeLaunch, resolve_native_launch
from eva_agentic.process import NativeRunResult
from eva_agentic.resources import ResourceGrant
from eva_agentic.schema import EpisodeResult, Job, OutcomeStatus


class RatsLiberoAdapter:
    """Run one native RATs LIBERO-Pro evaluation at native iteration zero.

    RATs currently exposes no CLI seed override and resets LIBERO with its
    iteration index. This adapter therefore accepts only ``Case.seed == 0``.
    """

    name = "rats_libero"

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
        if case.seed != 0:
            raise UnsupportedCondition(
                "RATs LIBERO adapter supports only seed 0: native CLI has no seed override"
            )
        suite = case.initialization.get("suite")
        task_id = case.initialization.get("task_id")
        if not isinstance(suite, str) or not suite:
            raise ValueError("RATs LIBERO case initialization requires suite")
        if isinstance(task_id, bool) or not isinstance(task_id, int):
            raise ValueError("RATs LIBERO case initialization requires integer task_id")
        command = (
            *self.spec.command,
            "--env-type",
            "libero",
            "--libero-suite",
            suite,
            "--libero-task",
            str(task_id),
            "--iterations",
            "1",
            "--fixed-task",
            "--no-skill-reuse",
            "--no-failure-memory",
            "--output-dir",
            "{output_dir}",
        )
        spec = replace(
            self.spec,
            command=command,
            env={**self.spec.env, "RATS_VERIFIER_STRICT_BENCHMARK": "1"},
        )
        return resolve_native_launch(spec, profile, job, attempt_dir, grants)

    def parse(
        self, job: Job, attempt_dir: Path, run: NativeRunResult
    ) -> tuple[EpisodeResult, ...]:
        case = _one_case(job)
        summary_path = Path(attempt_dir) / "native" / "final_summary.json"
        try:
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            if not isinstance(summary, dict):
                raise ValueError("summary must be a JSON object")
            total = _integer(summary, "total_iterations")
            successes = _integer(summary, "successful_iterations")
            failures = _integer(summary, "failed_iterations")
            if total != 1 or successes + failures != 1:
                raise ValueError("one-trial summary must contain exactly one outcome")
            status = OutcomeStatus.SUCCESS if successes == 1 else OutcomeStatus.TASK_FAILURE
            return (
                EpisodeResult(
                    case_id=case.case_id,
                    status=status,
                    task_success=successes == 1,
                    success_source="rats.final_summary.successful_iterations",
                    requested_conditions={"seed": case.seed},
                    effective_conditions={
                        "seed": 0,
                        "native_iteration": 0,
                        "strict_benchmark": True,
                    },
                    metrics={
                        "native_success_rate": _number(summary.get("success_rate")),
                        "native_exit_code": run.exit_code,
                    },
                    evidence_paths=(str(summary_path),),
                ),
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            return (
                EpisodeResult(
                    case_id=case.case_id,
                    status=OutcomeStatus.INVALID,
                    task_success=None,
                    error_source="rats.final_summary",
                    termination_reason=str(error),
                    requested_conditions={"seed": case.seed},
                    effective_conditions={"seed": 0, "native_iteration": 0},
                    evidence_paths=(str(summary_path),),
                ),
            )


def _one_case(job: Job):
    if len(job.cases) != 1:
        raise ValueError("RATs LIBERO adapter requires one case per job")
    return job.cases[0]


def _integer(summary: dict[str, object], name: str) -> int:
    value = summary.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"invalid {name}")
    return value


def _number(value: object) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    return value
