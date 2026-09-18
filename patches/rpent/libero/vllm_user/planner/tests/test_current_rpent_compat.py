from __future__ import annotations

from rpent.dashboard.events import NullDashboardEventSink
from rpent.tools.toolkit import ToolResult
from rpent_vllm_user import create_planner
from rpent_vllm_user.compat import (
    CapabilityCatalogEntry,
    CapabilityCatalogEvent,
    CompatibleDashboardEventSink,
    PlannerRequestUsageEvent,
)
from rpent_vllm_user.workspace import WorkspaceProfile, compose_tool_surface


def test_factory_constructs_without_correction_rpent(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class Config:
        model = "test-model"
        expected_user_id = None

    class Client:
        def __init__(self, config):
            captured["config"] = config

        def chat(self, **kwargs):
            raise AssertionError(f"unexpected model request: {kwargs}")

    def from_env(**kwargs):
        captured["config_kwargs"] = kwargs
        return Config()

    monkeypatch.setattr(
        "rpent_vllm_user.SignedVllmConfig.from_env",
        from_env,
    )
    monkeypatch.setattr("rpent_vllm_user.SignedVllmClient", Client)

    planner = create_planner(
        base_url="http://vllm.test",
        model="test-model",
        max_tokens=32,
        timeout_s=7,
        reasoning_effort="none",
        dashboard_events=NullDashboardEventSink(),
        no_images=True,
    )

    assert planner.execution_surface == "in_process_tool_surface"
    assert captured["config_kwargs"] == {
        "base_url_override": "http://vllm.test",
        "model_override": "test-model",
        "timeout_s": 7,
    }


def test_current_toolkit_surface_is_accepted_when_workspace_disabled() -> None:
    class CurrentToolkit:
        def get_tools_spec(self):
            return [
                {
                    "name": "observe",
                    "description": "observe",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ]

        def execute_tool(self, name, input_dict):
            return ToolResult(name=name, result={"ok": True})

    toolkit = CurrentToolkit()
    surface = compose_tool_surface(
        toolkit,
        WorkspaceProfile(mode="disabled"),
        dashboard_events=NullDashboardEventSink(),
    )
    assert surface is toolkit
    assert surface.execute_tool("observe", {}).result == {"ok": True}


def test_dashboard_adapter_downgrades_optional_events_to_current_usage() -> None:
    class CurrentSink:
        enabled = True

        def __init__(self):
            self.events = []

        def emit(self, event):
            from rpent.dashboard.events import UsageEvent

            if isinstance(event, UsageEvent):
                self.events.append(event)
                return
            raise TypeError("current RPent does not know optional event")

    sink = CurrentSink()
    adapted = CompatibleDashboardEventSink(sink)
    adapted.emit(
        PlannerRequestUsageEvent(
            turn=1,
            input_tokens=2,
            output_tokens=3,
        )
    )
    adapted.emit(CapabilityCatalogEvent(entries=(CapabilityCatalogEntry("x", "x"),)))
    assert len(sink.events) == 1
    assert sink.events[0].inp == 2
    assert sink.events[0].out == 3
