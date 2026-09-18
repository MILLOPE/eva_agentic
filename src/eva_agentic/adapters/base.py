"""Adapter contract for framework-native commands."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

from eva_agentic.frameworks import FrameworkProfile, FrameworkSpec, NativeLaunch, resolve_native_launch
from eva_agentic.process import NativeRunResult
from eva_agentic.resources import ResourceGrant
from eva_agentic.schema import EpisodeResult, Job


class UnsupportedCondition(ValueError):
    """A requested evaluation condition lacks a native framework mapping."""


class FrameworkAdapter(Protocol):
    name: str

    def build_launch(
        self,
        job: Job,
        attempt_dir: Path,
        profile: FrameworkProfile,
        grants: Sequence[ResourceGrant],
    ) -> NativeLaunch: ...

    def parse(
        self, job: Job, attempt_dir: Path, run: NativeRunResult
    ) -> tuple[EpisodeResult, ...]: ...


class DeclaredAdapter:
    """A base adapter for frameworks that only need command templating."""

    def __init__(self, spec: FrameworkSpec) -> None:
        self.name = spec.name
        self.spec = spec

    def build_launch(
        self,
        job: Job,
        attempt_dir: Path,
        profile: FrameworkProfile,
        grants: Sequence[ResourceGrant],
    ) -> NativeLaunch:
        return resolve_native_launch(self.spec, profile, job, attempt_dir, grants)

    def parse(
        self, job: Job, attempt_dir: Path, run: NativeRunResult
    ) -> tuple[EpisodeResult, ...]:
        raise NotImplementedError("framework requires a result parser adapter")
