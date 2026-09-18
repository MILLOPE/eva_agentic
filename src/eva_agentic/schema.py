"""Core data structures for experiment inputs, execution plans, and results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1


class RunMode(str, Enum):
    DEBUG = "debug"
    FROZEN = "frozen"


class OutcomeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    TASK_FAILURE = "task_failure"
    TIMEOUT = "timeout"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"
    INVALID = "invalid"


class AttemptStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class Budget:
    max_control_steps: int | None = None
    episode_timeout_s: float | None = None

    def __post_init__(self) -> None:
        _positive("max_control_steps", self.max_control_steps, allow_none=True)
        _positive("episode_timeout_s", self.episode_timeout_s, allow_none=True)


@dataclass(frozen=True)
class Protocol:
    track: str
    phase: str
    memory_policy: str
    scoring: str
    budget: Budget = field(default_factory=Budget)


@dataclass(frozen=True)
class Execution:
    mode: RunMode
    max_jobs: int
    max_infrastructure_retries: int
    job_timeout_s: float

    def __post_init__(self) -> None:
        _positive("max_jobs", self.max_jobs)
        _non_negative("max_infrastructure_retries", self.max_infrastructure_retries)
        _positive("job_timeout_s", self.job_timeout_s)


@dataclass(frozen=True)
class ArtifactPolicy:
    video: str
    trace: str


@dataclass(frozen=True)
class Experiment:
    schema_version: int
    name: str
    benchmark: str
    cases_file: str
    participants: tuple[str, ...]
    protocol: Protocol
    execution: Execution
    artifacts: ArtifactPolicy

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        if not self.name:
            raise ValueError("name must not be empty")
        if not self.benchmark:
            raise ValueError("benchmark must not be empty")
        if not self.cases_file:
            raise ValueError("cases_file must not be empty")
        if not self.participants:
            raise ValueError("participants must not be empty")
        if len(set(self.participants)) != len(self.participants):
            raise ValueError("participants must be unique")

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)


@dataclass(frozen=True)
class Case:
    case_id: str
    task_id: str
    seed: int
    initialization: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must not be empty")
        if not self.task_id:
            raise ValueError("task_id must not be empty")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        object.__setattr__(
            self, "initialization", MappingProxyType(dict(self.initialization))
        )

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)


@dataclass(frozen=True)
class PlannedCase:
    participant: str
    case: Case


@dataclass(frozen=True)
class Job:
    job_id: str
    participant: str
    cases: tuple[Case, ...]

    def __post_init__(self) -> None:
        if not self.job_id:
            raise ValueError("job_id must not be empty")
        if not self.participant:
            raise ValueError("participant must not be empty")
        if not self.cases:
            raise ValueError("job must contain at least one case")

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)


@dataclass(frozen=True)
class Plan:
    schema_version: int
    experiment_name: str
    jobs: tuple[Job, ...]

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported plan schema_version: {self.schema_version}")
        if not self.experiment_name:
            raise ValueError("experiment_name must not be empty")
        if not self.jobs:
            raise ValueError("plan must contain at least one job")
        job_ids = [job.job_id for job in self.jobs]
        if len(set(job_ids)) != len(job_ids):
            raise ValueError("job_ids must be unique")
        planned_pairs = [
            (job.participant, case.case_id) for job in self.jobs for case in job.cases
        ]
        if len(set(planned_pairs)) != len(planned_pairs):
            raise ValueError("participant and case pairs must be unique")

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)


@dataclass(frozen=True)
class EpisodeResult:
    case_id: str
    status: OutcomeStatus
    task_success: bool | None
    termination_reason: str | None = None
    error_source: str | None = None
    success_source: str | None = None
    requested_conditions: Mapping[str, Any] = field(default_factory=dict)
    effective_conditions: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, float | int | None] = field(default_factory=dict)
    evidence_paths: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must not be empty")
        if self.status is OutcomeStatus.SUCCESS and self.task_success is not True:
            raise ValueError("SUCCESS requires task_success=True")
        if self.status is OutcomeStatus.TASK_FAILURE and self.task_success is not False:
            raise ValueError("TASK_FAILURE requires task_success=False")
        if self.status in {
            OutcomeStatus.TIMEOUT,
            OutcomeStatus.INFRASTRUCTURE_FAILURE,
            OutcomeStatus.INVALID,
        } and self.task_success is not None:
            raise ValueError(f"{self.status.value} requires task_success=None")
        object.__setattr__(
            self, "requested_conditions", MappingProxyType(dict(self.requested_conditions))
        )
        object.__setattr__(
            self, "effective_conditions", MappingProxyType(dict(self.effective_conditions))
        )
        object.__setattr__(
            self, "metrics", MappingProxyType(dict(self.metrics))
        )

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)


@dataclass(frozen=True)
class Attempt:
    participant: str
    job_id: str
    attempt_id: int
    status: AttemptStatus
    process: Mapping[str, Any] = field(default_factory=dict)
    results: tuple[EpisodeResult, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _positive("attempt_id", self.attempt_id)
        object.__setattr__(self, "process", MappingProxyType(dict(self.process)))
        _unique_case_ids(self.results)
        if self.status is AttemptStatus.RUNNING and self.results:
            raise ValueError("RUNNING attempt must not contain results")
        if self.status is AttemptStatus.COMPLETED and not self.results:
            raise ValueError("COMPLETED attempt must contain results")

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)


def _positive(name: str, value: float | int | None, allow_none: bool = False) -> None:
    if value is None and allow_none:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be positive")


def _non_negative(name: str, value: float | int) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{name} must be non-negative")


def _unique_case_ids(results: Sequence[EpisodeResult]) -> None:
    case_ids = [result.case_id for result in results]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("result case_ids must be unique")


def _to_plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_to_plain(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return {
            key: _to_plain(getattr(value, key)) for key in value.__dataclass_fields__
        }
    return value
