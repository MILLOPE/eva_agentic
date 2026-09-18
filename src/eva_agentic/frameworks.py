"""Portable framework declarations and local native-runtime resolution."""

from __future__ import annotations

import json
import string
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from eva_agentic.resources import ResourceGrant
from eva_agentic.schema import Job


_ALLOWED_PLACEHOLDERS = {
    "task_id",
    "seed",
    "output_dir",
    "attempt_dir",
    "workspace",
}


class RuntimeBackend(str, Enum):
    CONDA = "conda"
    UV = "uv"
    PYTHON = "python"


@dataclass(frozen=True)
class FrameworkSpec:
    name: str
    backend: RuntimeBackend
    workdir: Path
    command: tuple[str, ...]
    resources: Mapping[str, int] = field(default_factory=dict)
    env: Mapping[str, str] = field(default_factory=dict)
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("framework name must not be empty")
        if not self.command:
            raise ValueError(f"framework {self.name} requires a command")
        _validate_templates(self.command)
        _validate_resources(self.resources)
        _validate_string_mapping(self.provenance, "framework provenance")
        object.__setattr__(self, "command", tuple(self.command))
        object.__setattr__(self, "resources", MappingProxyType(dict(self.resources)))
        object.__setattr__(self, "env", MappingProxyType(dict(self.env)))
        object.__setattr__(
            self, "provenance", MappingProxyType(dict(self.provenance))
        )


@dataclass(frozen=True)
class FrameworkLocal:
    conda_executable: Path | None = None
    environment: str | None = None
    uv_executable: Path | None = None
    interpreter: Path | None = None
    env: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "env", MappingProxyType(dict(self.env)))


@dataclass(frozen=True)
class FrameworkProfile:
    frameworks: Mapping[str, FrameworkLocal]
    resource_slots: Mapping[str, tuple[int, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "frameworks", MappingProxyType(dict(self.frameworks)))
        slots = {kind: tuple(values) for kind, values in self.resource_slots.items()}
        for kind, values in slots.items():
            if not values or len(set(values)) != len(values):
                raise ValueError(f"resource slots for {kind} must be non-empty and unique")
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
                raise ValueError(f"resource slots for {kind} must be non-negative integers")
        object.__setattr__(self, "resource_slots", MappingProxyType(slots))


@dataclass(frozen=True)
class NativeLaunch:
    argv: tuple[str, ...]
    cwd: Path
    env: Mapping[str, str]
    attempt_dir: Path
    output_dir: Path
    requested_resources: Mapping[str, int]
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.argv:
            raise ValueError("native launch requires argv")
        object.__setattr__(self, "argv", tuple(self.argv))
        object.__setattr__(self, "env", MappingProxyType(dict(self.env)))
        object.__setattr__(
            self, "requested_resources", MappingProxyType(dict(self.requested_resources))
        )
        _validate_string_mapping(self.provenance, "launch provenance")
        object.__setattr__(
            self, "provenance", MappingProxyType(dict(self.provenance))
        )


def load_framework_specs(path: str | Path) -> dict[str, FrameworkSpec]:
    source = Path(path)
    data = _load_mapping(source)
    raw_specs = data.get("frameworks")
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError(f"{source}: frameworks must be a non-empty list")
    specs = [parse_framework_spec(item, source.parent) for item in raw_specs]
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"{source}: framework names must be unique")
    return {spec.name: spec for spec in specs}


def parse_framework_spec(data: Mapping[str, Any], root: Path) -> FrameworkSpec:
    try:
        provenance = data.get("provenance", {})
        if not isinstance(provenance, Mapping):
            raise ValueError("framework provenance must be a mapping")
        return FrameworkSpec(
            name=str(data["name"]),
            backend=RuntimeBackend(data["backend"]),
            workdir=(root / str(data["workdir"])).resolve(),
            command=tuple(str(item) for item in data["command"]),
            resources=dict(data.get("resources", {})),
            env={str(key): str(value) for key, value in data.get("env", {}).items()},
            provenance=dict(provenance),
        )
    except KeyError as error:
        raise ValueError(f"missing framework field: {error.args[0]}") from error
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid framework declaration: {error}") from error


def load_framework_profile(path: str | Path) -> FrameworkProfile:
    source = Path(path)
    data = _load_mapping(source)
    raw_frameworks = data.get("frameworks")
    if not isinstance(raw_frameworks, Mapping):
        raise ValueError(f"{source}: frameworks must be a mapping")
    frameworks: dict[str, FrameworkLocal] = {}
    for name, item in raw_frameworks.items():
        if not isinstance(item, Mapping):
            raise ValueError(f"{source}: framework profile {name} must be a mapping")
        frameworks[str(name)] = FrameworkLocal(
            conda_executable=_optional_path(item.get("conda_executable")),
            environment=_optional_string(item.get("environment")),
            uv_executable=_optional_path(item.get("uv_executable")),
            interpreter=_optional_path(item.get("interpreter")),
            env={str(key): str(value) for key, value in item.get("env", {}).items()},
        )
    raw_slots = data.get("resource_slots", {})
    if not isinstance(raw_slots, Mapping):
        raise ValueError(f"{source}: resource_slots must be a mapping")
    return FrameworkProfile(
        frameworks=frameworks,
        resource_slots={str(kind): tuple(values) for kind, values in raw_slots.items()},
    )


def resolve_native_launch(
    spec: FrameworkSpec,
    profile: FrameworkProfile,
    job: Job,
    attempt_dir: str | Path,
    grants: Sequence[ResourceGrant] = (),
) -> NativeLaunch:
    """Render and wrap one framework-native command for a single-case job."""
    if len(job.cases) != 1:
        raise ValueError("native framework launch requires exactly one case per job")
    local = profile.frameworks.get(spec.name)
    if local is None:
        raise ValueError(f"local profile has no framework: {spec.name}")
    case = job.cases[0]
    attempt = Path(attempt_dir).resolve()
    output_dir = attempt / "native"
    values = {
        "task_id": case.task_id,
        "seed": str(case.seed),
        "output_dir": str(output_dir),
        "attempt_dir": str(attempt),
        "workspace": str(attempt),
    }
    command = tuple(token.format(**values) for token in spec.command)
    argv = _wrap_command(spec.backend, local, spec.workdir, command)
    env = dict(spec.env)
    env.update(local.env)
    env["EVA_ATTEMPT_DIR"] = str(attempt)
    env["EVA_OUTPUT_DIR"] = str(output_dir)
    gpu_slots = [str(grant.slot) for grant in grants if grant.kind == "gpu"]
    if gpu_slots:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_slots)
    return NativeLaunch(
        argv=argv,
        cwd=spec.workdir,
        env=env,
        attempt_dir=attempt,
        output_dir=output_dir,
        requested_resources=spec.resources,
        provenance=spec.provenance,
    )


def _wrap_command(
    backend: RuntimeBackend,
    local: FrameworkLocal,
    workdir: Path,
    command: tuple[str, ...],
) -> tuple[str, ...]:
    if backend is RuntimeBackend.CONDA:
        if local.conda_executable is None or not local.environment:
            raise ValueError("conda framework profile requires conda_executable and environment")
        return (
            str(local.conda_executable),
            "run",
            "--name",
            local.environment,
            "--no-capture-output",
            *command,
        )
    if backend is RuntimeBackend.UV:
        if local.uv_executable is None:
            raise ValueError("uv framework profile requires uv_executable")
        return (
            str(local.uv_executable),
            "run",
            "--project",
            str(workdir),
            "--frozen",
            *command,
        )
    if local.interpreter is None:
        raise ValueError("python framework profile requires interpreter")
    return (str(local.interpreter), *command)


def _validate_templates(command: Sequence[str]) -> None:
    formatter = string.Formatter()
    for token in command:
        for _, field_name, _, _ in formatter.parse(token):
            if field_name is not None and field_name not in _ALLOWED_PLACEHOLDERS:
                raise ValueError(f"unsupported command placeholder: {field_name}")


def _validate_resources(resources: Mapping[str, int]) -> None:
    for kind, count in resources.items():
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"resource {kind} must be a positive integer")


def _validate_string_mapping(mapping: Mapping[str, str], name: str) -> None:
    for key, value in mapping.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(f"{name} keys and values must be strings")


def _optional_path(value: Any) -> Path | None:
    return Path(value).expanduser() if value else None


def _optional_string(value: Any) -> str | None:
    return str(value) if value else None


def _load_mapping(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            data = json.load(handle)
        else:
            try:
                import yaml
            except ImportError as error:
                raise RuntimeError(
                    "PyYAML is required to load YAML framework configurations"
                ) from error
            data = yaml.safe_load(handle)
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: expected a mapping")
    return data
