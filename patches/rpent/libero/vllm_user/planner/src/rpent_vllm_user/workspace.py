# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Optional, read-only repository tools for the signed vLLM planner.

The model server never executes these tools.  Calls are dispatched beside the
RPent process against one fixed repository root.  The disabled path returns the
original Toolkit unchanged, so enabling this module does not create a second
tool registry or lifecycle owner.
"""

from __future__ import annotations

import fnmatch
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .compat import (
    CapabilityCatalogEntry,
    DashboardEventSink,
    ToolResult,
    dashboard_catalog_for_toolkit,
    execute_observed_tool,
)

_WORKSPACE_TOOL_NAMES = frozenset({"workspace_search", "workspace_shell"})
_SHELL_COMMANDS = frozenset({"head", "ls", "pwd", "tail", "wc"})
_SKIP_DIRS = frozenset(
    {".git", ".pytest_cache", ".ruff_cache", ".venv", "__pycache__", "node_modules"}
)
_MAX_FILE_BYTES = 2 * 1024 * 1024
_MAX_LINE_CHARS = 500
_MAX_OUTPUT_BYTES = 60_000
_COMMAND_TIMEOUT_S = 10.0


@dataclass(frozen=True, slots=True)
class WorkspaceProfile:
    """Configuration for the default-off repository observation surface."""

    mode: str = "disabled"
    root: Path = Path(".")

    def __post_init__(self) -> None:
        if self.mode not in {"disabled", "readonly"}:
            raise ValueError("workspace mode must be 'disabled' or 'readonly'")
        root = self.root.expanduser().resolve()
        # Keep the rollback path inert: a disabled optional module must not
        # depend on workspace availability or fail planner construction.
        if self.enabled and not root.is_dir():
            raise ValueError(f"workspace root is not a directory: {root}")
        object.__setattr__(self, "root", root)

    @property
    def enabled(self) -> bool:
        return self.mode == "readonly"


def workspace_profile_from_env(
    *,
    repo_root: str | Path,
    environ: Mapping[str, str] | None = None,
) -> WorkspaceProfile:
    """Build the repository profile from the documented environment switch."""
    env = os.environ if environ is None else environ
    mode = env.get("RPENT_VLLM_WORKSPACE_MODE", "disabled").strip().lower()
    return WorkspaceProfile(mode=mode, root=Path(repo_root))


class CompositeToolSurface:
    """Add one non-owning tool module to an existing Toolkit-like surface."""

    def __init__(
        self,
        base: Any,
        workspace: "ReadonlyWorkspaceTools",
        *,
        dashboard_events: DashboardEventSink,
    ) -> None:
        self._base = base
        self._workspace = workspace
        self._dashboard_events = dashboard_events
        base_names = {
            str(spec.get("name"))
            for spec in base.get_tools_spec()
            if isinstance(spec, Mapping)
        }
        duplicates = base_names & _WORKSPACE_TOOL_NAMES
        if duplicates:
            raise ValueError(
                "workspace tools conflict with existing Toolkit tools: "
                + ", ".join(sorted(duplicates))
            )

    def get_tools_spec(self) -> list[dict[str, Any]]:
        return [*self._base.get_tools_spec(), *self._workspace.get_tools_spec()]

    def dashboard_catalog(self) -> tuple[CapabilityCatalogEntry, ...]:
        return (
            *dashboard_catalog_for_toolkit(self._base),
            *self._workspace.dashboard_catalog(),
        )

    def execute_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if name in _WORKSPACE_TOOL_NAMES:
            return execute_observed_tool(
                dashboard_events=self._dashboard_events,
                name=name,
                input_dict=arguments,
                category="context",
                handler=self._workspace.execute_tool,
            )
        return self._base.execute_tool(name, arguments)

    def is_tool_multi_call_safe(self, name: str) -> bool:
        if name in _WORKSPACE_TOOL_NAMES:
            return True
        checker = getattr(self._base, "is_tool_multi_call_safe", None)
        return bool(callable(checker) and checker(name))

    def validate_planner_proposal(self, calls):
        """Preserve the shared runtime preflight through this tool projection."""
        validator = getattr(self._base, "validate_planner_proposal", None)
        return validator(calls) if callable(validator) else None


def compose_tool_surface(
    base: Any,
    profile: WorkspaceProfile,
    *,
    dashboard_events: DashboardEventSink,
) -> Any:
    """Return ``base`` unchanged unless repository observation is enabled."""
    if not profile.enabled:
        return base
    return CompositeToolSurface(
        base,
        ReadonlyWorkspaceTools(profile),
        dashboard_events=dashboard_events,
    )


class ReadonlyWorkspaceTools:
    """Search and run a small non-executable argv command set in one repo."""

    def __init__(self, profile: WorkspaceProfile) -> None:
        if not profile.enabled:
            raise ValueError("readonly workspace tools require an enabled profile")
        self._root = profile.root

    def get_tools_spec(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "workspace_search",
                "description": (
                    "Search UTF-8 source files below the RPent repository root. "
                    "Paths are repo-relative; symlink escapes are rejected."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "path": {
                            "type": "string",
                            "description": "Repo-relative file or directory; default '.'",
                        },
                        "glob": {
                            "type": "string",
                            "description": "Filename glob; default '*'",
                        },
                        "regex": {"type": "boolean", "default": False},
                        "case_sensitive": {"type": "boolean", "default": False},
                        "max_results": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 200,
                            "default": 100,
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "workspace_shell",
                "description": (
                    "Run one read-only argv command in the RPent repository. "
                    "Arguments that name files are confined to repo-relative paths. "
                    "No shell parsing, pipes, redirects, network tools, or writes. "
                    "Allowed commands: head, ls, pwd, tail, wc."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "argv": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "maxItems": 32,
                        }
                    },
                    "required": ["argv"],
                    "additionalProperties": False,
                },
            },
        ]

    def dashboard_catalog(self) -> tuple[CapabilityCatalogEntry, ...]:
        entries = []
        for spec in self.get_tools_spec():
            schema = spec["input_schema"]
            entries.append(
                CapabilityCatalogEntry(
                    tool_name=spec["name"],
                    summary=" ".join(spec["description"].split()),
                    category="context",
                    parameter_names=tuple(schema.get("properties", {})),
                    source="toolkit",
                )
            )
        return tuple(entries)

    def execute_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        try:
            if name == "workspace_search":
                result = self._search(**arguments)
            elif name == "workspace_shell":
                result = self._shell(**arguments)
            else:
                result = {"error": f"unknown workspace tool: {name}"}
        except (TypeError, ValueError, re.error) as exc:
            result = {"error": f"bad arguments for {name}: {exc}"}
        return ToolResult(name=name, result=result)

    def _resolve(self, value: str) -> Path:
        if not isinstance(value, str):
            raise TypeError("path must be a string")
        path = Path(value or ".")
        if path.is_absolute():
            raise ValueError("path must be repo-relative")
        resolved = (self._root / path).resolve()
        try:
            resolved.relative_to(self._root)
        except ValueError as exc:
            raise ValueError("path escapes the repository root") from exc
        return resolved

    def _search(
        self,
        query: str,
        path: str = ".",
        glob: str = "*",
        regex: bool = False,
        case_sensitive: bool = False,
        max_results: int = 100,
    ) -> dict[str, Any]:
        if not isinstance(query, str) or not query:
            raise ValueError("query must be a non-empty string")
        if not isinstance(glob, str) or not glob:
            raise ValueError("glob must be a non-empty string")
        if not isinstance(regex, bool) or not isinstance(case_sensitive, bool):
            raise TypeError("regex and case_sensitive must be booleans")
        if (
            isinstance(max_results, bool)
            or not isinstance(max_results, int)
            or not 1 <= max_results <= 200
        ):
            raise ValueError("max_results must be an integer from 1 to 200")
        target = self._resolve(path)
        if not target.exists():
            return {"error": f"path not found: {path}"}

        flags = 0 if case_sensitive else re.IGNORECASE
        pattern = re.compile(query if regex else re.escape(query), flags)
        matches: list[dict[str, Any]] = []
        skipped_files = 0
        for candidate in self._files(target, glob):
            try:
                if candidate.stat().st_size > _MAX_FILE_BYTES:
                    skipped_files += 1
                    continue
                data = candidate.read_bytes()
            except OSError:
                skipped_files += 1
                continue
            if b"\x00" in data:
                skipped_files += 1
                continue
            text = data.decode("utf-8", errors="replace")
            for line_number, line in enumerate(text.splitlines(), start=1):
                if pattern.search(line) is None:
                    continue
                matches.append(
                    {
                        "path": str(candidate.relative_to(self._root)),
                        "line": line_number,
                        "text": line[:_MAX_LINE_CHARS],
                    }
                )
                if len(matches) >= max_results:
                    return {
                        "query": query,
                        "matches": matches,
                        "truncated": True,
                        "skipped_files": skipped_files,
                    }
        return {
            "query": query,
            "matches": matches,
            "truncated": False,
            "skipped_files": skipped_files,
        }

    def _files(self, target: Path, glob: str):
        if target.is_file():
            if fnmatch.fnmatch(target.name, glob):
                yield target
            return
        for directory, dirs, files in os.walk(target, followlinks=False):
            dirs[:] = sorted(name for name in dirs if name not in _SKIP_DIRS)
            for name in sorted(files):
                if not fnmatch.fnmatch(name, glob):
                    continue
                candidate = Path(directory) / name
                try:
                    resolved = candidate.resolve()
                    resolved.relative_to(self._root)
                except (OSError, ValueError):
                    continue
                if resolved.is_file():
                    yield resolved

    def _shell(self, argv: Sequence[str]) -> dict[str, Any]:
        if isinstance(argv, (str, bytes)) or not isinstance(argv, Sequence):
            raise TypeError("argv must be an array of strings")
        command = list(argv)
        if not command or len(command) > 32:
            raise ValueError("argv must contain 1 to 32 items")
        if any(
            not isinstance(item, str) or not item or len(item) > 1000
            for item in command
        ):
            raise ValueError(
                "each argv item must be a non-empty string up to 1000 chars"
            )
        executable = Path(command[0]).name
        if command[0] != executable or executable not in _SHELL_COMMANDS:
            raise ValueError(
                "command must be one of: " + ", ".join(sorted(_SHELL_COMMANDS))
            )
        resolved_executable = shutil.which(executable, path="/usr/bin:/bin")
        if resolved_executable is None:
            return {"error": f"allowed command is unavailable: {executable}"}
        validated_arguments = self._validated_command_arguments(executable, command[1:])
        scoped_command = [resolved_executable, *validated_arguments]
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            process = subprocess.Popen(
                scoped_command,
                cwd=self._root,
                env={"HOME": "/tmp", "LANG": "C.UTF-8", "PATH": "/usr/bin:/bin"},
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                close_fds=True,
                start_new_session=True,
            )
            try:
                return_code = process.wait(timeout=_COMMAND_TIMEOUT_S)
                timed_out = False
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                return_code = process.returncode
                timed_out = True
            stdout_text, stdout_truncated = _read_bounded(stdout)
            stderr_text, stderr_truncated = _read_bounded(stderr)
        result: dict[str, Any] = {
            "argv": command,
            "returncode": return_code,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
            "sandbox": "repo_scoped_readonly_argv",
        }
        if timed_out:
            result["error"] = f"command timed out after {_COMMAND_TIMEOUT_S:g}s"
        elif return_code != 0:
            result["error"] = f"command exited with status {return_code}"
        return result

    def _validated_command_arguments(
        self, executable: str, arguments: list[str]
    ) -> list[str]:
        if executable == "pwd":
            if arguments:
                raise ValueError("pwd does not accept arguments")
            return []
        if executable == "ls":
            return self._validate_ls_arguments(arguments)
        if executable in {"head", "tail"}:
            return self._validate_head_tail_arguments(arguments)
        if executable == "wc":
            return self._validate_wc_arguments(arguments)
        raise ValueError(f"unsupported command: {executable}")

    def _validate_ls_arguments(self, arguments: list[str]) -> list[str]:
        validated: list[str] = []
        paths_started = False
        path_count = 0
        for argument in arguments:
            if argument == "--" and not paths_started:
                paths_started = True
                validated.append(argument)
                continue
            if not paths_started and argument.startswith("-"):
                flags = argument[1:]
                if not flags or any(flag not in "alhd1" for flag in flags):
                    raise ValueError("ls only accepts combined -a -l -h -d -1 options")
                validated.append(argument)
                continue
            paths_started = True
            validated.append(str(self._resolve(argument)))
            path_count += 1
        if path_count == 0:
            validated.append(str(self._root))
        return validated

    def _validate_head_tail_arguments(self, arguments: list[str]) -> list[str]:
        validated: list[str] = []
        index = 0
        path_count = 0
        while index < len(arguments):
            argument = arguments[index]
            if argument in {"-n", "-c"}:
                if index + 1 >= len(arguments):
                    raise ValueError(f"{argument} requires a positive integer")
                value = _bounded_count(arguments[index + 1])
                validated.extend((argument, str(value)))
                index += 2
                continue
            if argument.startswith("--lines=") or argument.startswith("--bytes="):
                option, raw_value = argument.split("=", 1)
                validated.append(f"{option}={_bounded_count(raw_value)}")
                index += 1
                continue
            if argument.startswith("-"):
                raise ValueError("head/tail only accept -n, -c, --lines, or --bytes")
            validated.append(str(self._resolve(argument)))
            path_count += 1
            index += 1
        if path_count == 0:
            raise ValueError("head/tail require at least one repo-relative file")
        return validated

    def _validate_wc_arguments(self, arguments: list[str]) -> list[str]:
        validated: list[str] = []
        path_count = 0
        for argument in arguments:
            if argument.startswith("-"):
                flags = argument[1:]
                if not flags or any(flag not in "cmlwL" for flag in flags):
                    raise ValueError("wc only accepts combined -c -m -l -w -L options")
                validated.append(argument)
                continue
            validated.append(str(self._resolve(argument)))
            path_count += 1
        if path_count == 0:
            raise ValueError("wc requires at least one repo-relative file")
        return validated


def _read_bounded(file: Any) -> tuple[str, bool]:
    file.seek(0)
    data = file.read(_MAX_OUTPUT_BYTES + 1)
    truncated = len(data) > _MAX_OUTPUT_BYTES
    if truncated:
        data = data[:_MAX_OUTPUT_BYTES]
    return data.decode("utf-8", errors="replace"), truncated


def _bounded_count(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("line/byte count must be an integer") from exc
    if not 1 <= value <= 100_000:
        raise ValueError("line/byte count must be from 1 to 100000")
    return value


__all__ = [
    "CompositeToolSurface",
    "ReadonlyWorkspaceTools",
    "WorkspaceProfile",
    "compose_tool_surface",
    "workspace_profile_from_env",
]
