"""Execution and evidence collection for one native framework command."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

from eva_agentic.frameworks import NativeLaunch
from eva_agentic.resources import ResourceGrant
from eva_agentic.schema import SCHEMA_VERSION
from eva_agentic.store import write_json_no_replace


class NativeRunStatus(str, Enum):
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    LAUNCH_FAILED = "launch_failed"


@dataclass(frozen=True)
class NativeRunResult:
    status: NativeRunStatus
    launch_path: Path
    runtime_path: Path
    stdout_path: Path
    stderr_path: Path
    pid: int | None = None
    exit_code: int | None = None
    terminated: bool = False
    duration_s: float = 0.0


def run_native_launch(
    launch: NativeLaunch,
    *,
    timeout_s: float,
    resource_grants: Sequence[ResourceGrant] = (),
) -> NativeRunResult:
    """Execute a framework-native command and preserve launch evidence."""
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    launch.attempt_dir.mkdir(parents=True, exist_ok=True)
    launch.output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = launch.attempt_dir / "stdout.log"
    stderr_path = launch.attempt_dir / "stderr.log"
    launch_path = launch.attempt_dir / "launch.json"
    runtime_path = launch.attempt_dir / "runtime.json"
    _write_launch_record(launch_path, launch, resource_grants)
    started_at = time.monotonic()

    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        try:
            child_env = {**os.environ, **launch.env}
            child_env.setdefault("EVA_ATTEMPT_DIR", str(launch.attempt_dir))
            child_env.setdefault("EVA_OUTPUT_DIR", str(launch.output_dir))
            process = subprocess.Popen(
                list(launch.argv),
                cwd=str(launch.cwd),
                env=child_env,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                start_new_session=True,
            )
        except OSError as error:
            result = NativeRunResult(
                status=NativeRunStatus.LAUNCH_FAILED,
                launch_path=launch_path,
                runtime_path=runtime_path,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                duration_s=time.monotonic() - started_at,
            )
            _write_runtime(runtime_path, result, resource_grants, error=str(error))
            return result

        try:
            exit_code = process.wait(timeout=timeout_s)
            result = NativeRunResult(
                status=NativeRunStatus.COMPLETED,
                launch_path=launch_path,
                runtime_path=runtime_path,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                pid=process.pid,
                exit_code=exit_code,
                duration_s=time.monotonic() - started_at,
            )
        except subprocess.TimeoutExpired:
            terminated = _terminate_process_tree(process)
            result = NativeRunResult(
                status=NativeRunStatus.TIMEOUT,
                launch_path=launch_path,
                runtime_path=runtime_path,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                pid=process.pid,
                exit_code=process.wait(),
                terminated=terminated,
                duration_s=time.monotonic() - started_at,
            )
    _write_runtime(runtime_path, result, resource_grants)
    return result


def _write_launch_record(
    path: Path, launch: NativeLaunch, resource_grants: Sequence[ResourceGrant]
) -> None:
    write_json_no_replace(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "command": list(launch.argv),
            "cwd": str(launch.cwd),
            "attempt_dir": str(launch.attempt_dir),
            "output_dir": str(launch.output_dir),
            "env_keys": sorted(launch.env),
            "requested_resources": dict(launch.requested_resources),
            "allocated_resources": [grant.to_dict() for grant in resource_grants],
            "provenance": dict(launch.provenance),
        },
    )


def _write_runtime(
    path: Path,
    result: NativeRunResult,
    resource_grants: Sequence[ResourceGrant],
    **extra: Any,
) -> None:
    write_json_no_replace(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": result.status.value,
            "pid": result.pid,
            "exit_code": result.exit_code,
            "terminated": result.terminated,
            "duration_s": result.duration_s,
            "stdout": str(result.stdout_path),
            "stderr": str(result.stderr_path),
            "allocated_resources": [grant.to_dict() for grant in resource_grants],
            "extra": extra,
        },
    )


def _terminate_process_tree(process: subprocess.Popen[Any]) -> bool:
    if process.poll() is not None:
        return False
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        process.terminate()
    try:
        process.wait(timeout=2)
        return True
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            process.kill()
        process.wait()
        return True
