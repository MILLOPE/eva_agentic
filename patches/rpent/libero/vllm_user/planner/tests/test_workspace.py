# Copyright 2026 The RPent Authors.

from __future__ import annotations

from rpent.dashboard.events import NullDashboardEventSink
from rpent.tools.toolkit import ToolResult
from rpent_vllm_user.compat import CapabilityCatalogEntry, ToolCallEvent
from rpent_vllm_user.workspace import (
    ReadonlyWorkspaceTools,
    WorkspaceProfile,
    compose_tool_surface,
    workspace_profile_from_env,
)


class _RecordingDashboardEvents:
    def __init__(self) -> None:
        self.events = []

    @property
    def enabled(self) -> bool:
        return True

    def emit(self, event) -> None:
        self.events.append(event)


class _BaseSurface:
    def get_tools_spec(self):
        return [
            {
                "name": "observe",
                "description": "observe",
                "input_schema": {"type": "object"},
            }
        ]

    def dashboard_catalog(self):
        return (
            CapabilityCatalogEntry(
                tool_name="observe",
                summary="observe",
                category="generic",
            ),
        )

    def execute_tool(self, name, arguments):
        return ToolResult(name=name, result={"arguments": arguments})

    def is_tool_multi_call_safe(self, name):
        return name == "observe"


def test_workspace_is_default_off_and_keeps_base_identity(tmp_path) -> None:
    profile = workspace_profile_from_env(repo_root=tmp_path, environ={})
    base = _BaseSurface()

    assert profile.mode == "disabled"
    assert (
        compose_tool_surface(
            base,
            profile,
            dashboard_events=NullDashboardEventSink(),
        )
        is base
    )


def test_disabled_workspace_does_not_require_existing_root(tmp_path) -> None:
    profile = WorkspaceProfile(mode="disabled", root=tmp_path / "missing")

    assert profile.enabled is False


def test_workspace_search_is_scoped_and_bounded(tmp_path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "one.py").write_text("Alpha\nneedle here\n")
    (tmp_path / "src" / "two.md").write_text("NEEDLE again\n")
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("needle secret\n")
    tools = ReadonlyWorkspaceTools(WorkspaceProfile(mode="readonly", root=tmp_path))

    result = tools.execute_tool(
        "workspace_search",
        {"query": "needle", "path": "src", "glob": "*", "max_results": 1},
    ).result
    escaped = tools.execute_tool(
        "workspace_search", {"query": "needle", "path": "../outside.txt"}
    ).result

    assert result["truncated"] is True
    assert result["matches"] == [
        {"path": "src/one.py", "line": 2, "text": "needle here"}
    ]
    assert "escapes the repository root" in escaped["error"]


def test_workspace_shell_rejects_shells_and_network_clients(tmp_path) -> None:
    tools = ReadonlyWorkspaceTools(WorkspaceProfile(mode="readonly", root=tmp_path))

    for argv in (["bash", "-lc", "id"], ["curl", "https://example.com"]):
        result = tools.execute_tool("workspace_shell", {"argv": argv}).result
        assert "command must be one of" in result["error"]


def test_composed_surface_routes_workspace_and_base_tools(tmp_path) -> None:
    (tmp_path / "guide.md").write_text("use back_project\n")
    base = _BaseSurface()
    dashboard_events = _RecordingDashboardEvents()
    surface = compose_tool_surface(
        base,
        WorkspaceProfile(mode="readonly", root=tmp_path),
        dashboard_events=dashboard_events,
    )

    assert {spec["name"] for spec in surface.get_tools_spec()} == {
        "observe",
        "workspace_search",
        "workspace_shell",
    }
    assert {entry.tool_name for entry in surface.dashboard_catalog()} == {
        "observe",
        "workspace_search",
        "workspace_shell",
    }
    assert surface.execute_tool("observe", {"step": 1}).result == {
        "arguments": {"step": 1}
    }
    assert dashboard_events.events == []

    search = surface.execute_tool("workspace_search", {"query": "back_project"})
    shell = surface.execute_tool("workspace_shell", {"argv": ["pwd"]})
    assert search.result["matches"][0]["path"] == "guide.md"
    assert search.call_id
    assert shell.call_id
    lifecycle = [
        event
        for event in dashboard_events.events
        if isinstance(event, ToolCallEvent)
    ]
    assert [(event.name, event.phase) for event in lifecycle] == [
        ("workspace_search", "started"),
        ("workspace_search", "returned"),
        ("workspace_shell", "started"),
        ("workspace_shell", "returned"),
    ]
    assert lifecycle[0].call_id == lifecycle[1].call_id == search.call_id
    assert lifecycle[2].call_id == lifecycle[3].call_id == shell.call_id
    assert all(event.category == "context" for event in lifecycle)
    assert surface.is_tool_multi_call_safe("workspace_search") is True
    assert surface.is_tool_multi_call_safe("observe") is True


def test_workspace_shell_scopes_paths_and_runs_without_inheriting_env(tmp_path) -> None:
    (tmp_path / "note.txt").write_text("first\nsecond\n")
    tools = ReadonlyWorkspaceTools(WorkspaceProfile(mode="readonly", root=tmp_path))

    result = tools.execute_tool(
        "workspace_shell", {"argv": ["head", "-n", "1", "note.txt"]}
    ).result
    escaped = tools.execute_tool(
        "workspace_shell", {"argv": ["head", "../outside.txt"]}
    ).result

    assert result["stdout"] == "first\n"
    assert result["returncode"] == 0
    assert result["sandbox"] == "repo_scoped_readonly_argv"
    assert "escapes the repository root" in escaped["error"]
