# Copyright 2026 The RPent Authors.

from __future__ import annotations

import copy
import json
import threading
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from jsonschema.validators import validator_for
from robots.libero import tools as libero_tools

import rpent_vllm_user.planner as planner_module
from rpent.dashboard.events import (
    NullDashboardEventSink,
    TranscriptEvent,
    UsageEvent,
)
from rpent_vllm_user.compat import (
    CapabilityCatalogEntry,
    CapabilityCatalogEvent,
    PlannerRequestUsageEvent,
    ToolCallEvent,
)
from rpent.dashboard.interaction import DashboardMessage
from rpent.memory import MemoryManager
from rpent.planner.runtime import PlannerRuntime
from rpent.session import EnvState
from rpent.tools.toolkit import Toolkit, ToolResult
from rpent.utils import templates
from rpent_vllm_user.client import (
    _build_chat_payload,
    _chat_payload_inline_image_bytes,
    _chat_payload_text_bytes,
    _chat_payload_wire_bytes,
)
from rpent_vllm_user.history import ToolHistoryPolicy
from rpent_vllm_user.liveness import ToolLoopGuard, tool_loop_guard_from_env
from rpent_vllm_user.multi_tool import MultiToolPolicy, multi_tool_policy_from_env
from rpent_vllm_user.planner import ModelApiUtilsPlanner
from rpent_vllm_user.result_projection import bounded_tool_result_text
from rpent_vllm_user.workspace import WorkspaceProfile


class _RecordingDashboardEvents:
    def __init__(self) -> None:
        self.events = []

    @property
    def enabled(self) -> bool:
        return True

    def emit(self, event) -> None:
        self.events.append(event)


class _DashboardInteraction:
    def __init__(self) -> None:
        self.activity = "starting"
        self.accepting_input = False
        self.messages: list[DashboardMessage] = []
        self.version = 0
        self.interrupt_requested = False
        self.interrupt_in_flight = False
        self.replacement_requested = False
        self.interrupt_error = None
        self.replacement_error = None
        self.condition = threading.Condition()

    @property
    def planner_activity(self):
        with self.condition:
            return self.activity

    @property
    def interaction_version(self):
        with self.condition:
            return self.version

    @property
    def task_replacement_requested(self):
        with self.condition:
            return self.replacement_requested

    def submit(self, text):
        with self.condition:
            assert self.accepting_input
            message = DashboardMessage(
                message_id=f"message-{len(self.messages) + 1}",
                text=text,
                status="pending",
            )
            self.messages.append(message)
            self._changed()
            return message

    def request_interrupt(self):
        with self.condition:
            self.interrupt_requested = True
            self._changed()

    def request_replacement(self):
        with self.condition:
            self.replacement_requested = True
            self.accepting_input = False
            self._changed()

    def claim_next_pending_message(self):
        with self.condition:
            if self.interrupt_requested or self.replacement_requested:
                return None
            for message in self.messages:
                if message.status == "pending":
                    message.status = "sending"
                    self._changed()
                    return message
            return None

    def mark_message_sent(self, message_id):
        return self._mark(message_id, "sent")

    def mark_message_failed(self, message_id, error):
        message = self._mark(message_id, "failed")
        message.error = error
        return message

    def mark_message_unsent(self, message_id):
        return self._mark(message_id, "unsent")

    def claim_interrupt_request(self):
        with self.condition:
            if not self.interrupt_requested:
                return False
            self.interrupt_requested = False
            self.interrupt_in_flight = True
            self._changed()
            return True

    def complete_interrupt(self, error=None):
        with self.condition:
            assert self.interrupt_in_flight
            self.interrupt_in_flight = False
            self.interrupt_error = error
            self._changed()

    def complete_task_replacement(self, error=None):
        with self.condition:
            self.replacement_error = error
            self.activity = "ended"
            self.accepting_input = False
            self._changed()

    def set_planner_activity(self, activity, *, accepting_input=None):
        with self.condition:
            self.activity = activity
            if accepting_input is not None:
                self.accepting_input = accepting_input
            self._changed()

    def seal_interaction(self):
        with self.condition:
            self.activity = "ended"
            self.accepting_input = False
            for message in self.messages:
                if message.status in {"pending", "sending"}:
                    message.status = "unsent"
            self._changed()

    def wait_for_interaction_change(self, since, timeout=None):
        with self.condition:
            self.condition.wait_for(lambda: self.version != since, timeout=timeout)
            return self.version

    def _mark(self, message_id, status):
        with self.condition:
            message = next(
                message for message in self.messages if message.message_id == message_id
            )
            assert message.status == "sending"
            message.status = status
            self._changed()
            return message

    def _changed(self):
        self.version += 1
        self.condition.notify_all()


class _Client:
    def __init__(self, *messages, usage=None):
        self.messages = list(messages)
        self.calls = []
        self.usage = {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        } if usage is None else usage

    def chat(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        message = self.messages.pop(0)
        if isinstance(message, Exception):
            raise message
        return SimpleNamespace(
            raw={"choices": [{"message": message}]},
            usage=copy.deepcopy(self.usage),
        )


class _Toolkit:
    def __init__(
        self,
        results,
        *,
        multi_call_safe: bool = True,
        dashboard_events=None,
    ):
        self.results = list(results)
        self.calls = []
        self.multi_call_safe = multi_call_safe
        self.cancelled = threading.Event()
        self.dashboard_events = dashboard_events or NullDashboardEventSink()

    def get_tools_spec(self):
        return [
            {
                "name": "observe",
                "description": "Observe and optionally finish",
                "input_schema": {"type": "object", "properties": {}},
            }
        ]

    def dashboard_catalog(self):
        return (
            CapabilityCatalogEntry(
                tool_name="observe",
                summary="Observe and optionally finish",
                category="generic",
            ),
        )

    def execute_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return ToolResult(name, self.results.pop(0))

    def is_tool_multi_call_safe(self, name):
        return name == "observe" and self.multi_call_safe

    def cancel_active_and_wait(self):
        self.cancelled.set()


class _BlockingToolkit(_Toolkit):
    def execute_tool(self, name, arguments):
        self.calls.append((name, arguments))
        self.cancelled.wait(timeout=1)
        return ToolResult(name, self.results.pop(0))


def _call(arguments="{}", *, call_id="call-1", name="observe"):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


def _with_shared_runtime(
    planner: ModelApiUtilsPlanner,
    *,
    guard: ToolLoopGuard | None = None,
) -> PlannerRuntime:
    """Exercise adapter behavior through the production shared boundary."""
    selected = guard or ToolLoopGuard()
    return PlannerRuntime(inner=planner, guard_factory=lambda: selected)


@pytest.mark.parametrize("thinking", [None, False, True])
def test_planner_enables_thinking_by_default_and_preserves_explicit_override(
    thinking: bool | None,
) -> None:
    client = _Client(_call())
    toolkit = _Toolkit([{"_finish": True, "status": "success"}])
    planner = ModelApiUtilsPlanner(
        client=client,
        model="default",
        **({"enable_thinking": thinking} if thinking is not None else {}),
    )

    result = planner.solve(
        system_prompt="", user_message="task", toolkit=toolkit, max_turns=1
    )

    assert result.error is None
    assert client.calls[0]["extra_body"]["enable_thinking"] is (
        False if thinking is None else thinking
    )


@pytest.mark.parametrize("shared_runtime", [False, True])
def test_dashboard_message_replans_before_stale_tool_call(shared_runtime: bool) -> None:
    interaction = _DashboardInteraction()
    dashboard_events = _RecordingDashboardEvents()

    class InteractiveClient(_Client):
        def chat(self, **kwargs):
            if not self.calls:
                assert interaction.accepting_input is True
                interaction.submit("Use the handle instead.")
            return super().chat(**kwargs)

    stale = _call('{"target":"stale"}', call_id="stale-call")
    stale["content"] = "I found a possible target."
    stale["reasoning_content"] = "Inspect the scene first."
    client = InteractiveClient(stale, _call(call_id="fresh-call"))
    toolkit = _Toolkit([{"_finish": True, "status": "success"}])
    planner = ModelApiUtilsPlanner(
        client=client,
        model="default",
        dashboard_events=dashboard_events,
    )
    if shared_runtime:
        planner = _with_shared_runtime(planner)

    result = planner.solve(
        system_prompt="rules",
        user_message="task",
        toolkit=toolkit,
        max_turns=2,
        dashboard_interaction=interaction,
    )

    assert result.error is None
    assert toolkit.calls == [("observe", {})]
    assert interaction.messages[0].status == "sent"
    assert interaction.activity == "ended"
    assert interaction.accepting_input is False
    assert [
        message["content"]
        for message in client.calls[1]["messages"]
        if message["role"] == "user"
    ] == [
        "task",
        "Use the handle instead.",
    ]
    dashboard_system = [
        message["content"]
        for message in client.calls[0]["messages"]
        if message["role"] == "system"
    ]
    assert len(dashboard_system) == 1
    assert dashboard_system[0].startswith("rules\n\n")
    assert "Tool-call protocol:" in dashboard_system[0]
    assert "only for these batch-safe tools: observe" in dashboard_system[0]
    assert "concise user-facing progress update" in dashboard_system[0]
    assert client.calls[0]["extra_body"]["enable_thinking"] is False
    if shared_runtime:
        assert result.stats["planner_runtime"] == "shared"
    payloads = [
        event.payload
        for event in dashboard_events.events
        if isinstance(event, TranscriptEvent)
    ]
    assert {"type": "initial_prompt"} in payloads
    assert {
        "type": "thinking",
        "text": "Inspect the scene first.",
        "request_index": 1,
        "turn": 1,
    } in payloads
    assert {
        "type": "text",
        "text": "I found a possible target.",
        "request_index": 1,
        "turn": 1,
    } in payloads
    assert {"type": "user", "text": "Use the handle instead."} in payloads


def test_dashboard_message_is_failed_when_followup_request_fails() -> None:
    interaction = _DashboardInteraction()

    class InteractiveClient(_Client):
        def chat(self, **kwargs):
            if not self.calls:
                interaction.submit("Try the other side.")
            return super().chat(**kwargs)

    client = InteractiveClient(_call(call_id="stale-call"), RuntimeError("offline"))
    planner = ModelApiUtilsPlanner(client=client, model="default")

    result = planner.solve(
        system_prompt="",
        user_message="task",
        toolkit=_Toolkit([{"_finish": True}]),
        max_turns=2,
        dashboard_interaction=interaction,
    )

    assert result.error == "RuntimeError: offline"
    assert interaction.messages[0].status == "failed"
    assert interaction.messages[0].error == "RuntimeError: offline"
    assert interaction.activity == "ended"


def test_dashboard_interrupt_cancels_before_vllm_tool_execution() -> None:
    interaction = _DashboardInteraction()
    toolkit = _Toolkit([{"_finish": True}])

    class InterruptingClient(_Client):
        def chat(self, **kwargs):
            interaction.request_interrupt()
            assert toolkit.cancelled.wait(timeout=1)
            return super().chat(**kwargs)

    planner = ModelApiUtilsPlanner(
        client=InterruptingClient(_call()),
        model="default",
    )

    result = planner.solve(
        system_prompt="",
        user_message="task",
        toolkit=toolkit,
        max_turns=1,
        dashboard_interaction=interaction,
    )

    assert result.error == "vLLM planner interrupted from the Dashboard"
    assert toolkit.calls == []
    assert interaction.interrupt_in_flight is False
    assert interaction.interrupt_error is None
    assert interaction.activity == "ended"


def test_dashboard_final_message_blocks_superseded_tool_call() -> None:
    interaction = _DashboardInteraction()

    class InteractiveClient(_Client):
        def chat(self, **kwargs):
            interaction.submit("Do not execute that action.")
            return super().chat(**kwargs)

    planner = ModelApiUtilsPlanner(
        client=InteractiveClient(_call(call_id="stale-call")),
        model="default",
    )
    toolkit = _Toolkit([{"_finish": True}])

    result = planner.solve(
        system_prompt="",
        user_message="task",
        toolkit=toolkit,
        max_turns=1,
        dashboard_interaction=interaction,
    )

    assert result.error == (
        "Dashboard input superseded the final vLLM response, "
        "but max_turns=1 is exhausted"
    )
    assert toolkit.calls == []
    assert interaction.messages[0].status == "unsent"


def test_dashboard_replacement_and_request_failure_preserve_message_state() -> None:
    interaction = _DashboardInteraction()
    toolkit = _Toolkit([{"_finish": True}])

    class ReplacingClient(_Client):
        def chat(self, **kwargs):
            if not self.calls:
                interaction.submit("Use the replacement task.")
            else:
                interaction.request_replacement()
                assert toolkit.cancelled.wait(timeout=1)
            return super().chat(**kwargs)

    planner = ModelApiUtilsPlanner(
        client=ReplacingClient(_call(call_id="stale-call"), RuntimeError("closed")),
        model="default",
    )

    result = planner.solve(
        system_prompt="",
        user_message="task",
        toolkit=toolkit,
        max_turns=2,
        dashboard_interaction=interaction,
    )

    assert result.error == "vLLM planner replaced by a newer Dashboard task"
    assert interaction.messages[0].status == "failed"
    assert interaction.messages[0].error == "RuntimeError: closed"
    assert interaction.replacement_error is None
    assert interaction.activity == "ended"


def test_dashboard_timeout_invalidates_late_model_response(monkeypatch) -> None:
    interaction = _DashboardInteraction()
    release = threading.Event()
    second_request_started = threading.Event()

    class LateClient(_Client):
        def chat(self, **kwargs):
            if not self.calls:
                interaction.submit("Wait for my correction.")
            else:
                second_request_started.set()
                release.wait(timeout=1)
            return super().chat(**kwargs)

    monkeypatch.setattr(planner_module, "_WORKER_CANCEL_GRACE_S", 0.02)
    planner = ModelApiUtilsPlanner(
        client=LateClient(_call(call_id="stale-call"), _call(call_id="late-call")),
        model="default",
        timeout_s=0.02,
    )

    result = planner.solve(
        system_prompt="",
        user_message="task",
        toolkit=_Toolkit([{"_finish": True}]),
        max_turns=2,
        dashboard_interaction=interaction,
    )

    assert second_request_started.is_set()
    assert result.error == "vLLM planner timed out after 0.02s"
    assert interaction.messages[0].status == "unsent"
    assert interaction.activity == "ended"

    release.set()
    time.sleep(0.05)
    assert interaction.messages[0].status == "unsent"
    assert interaction.activity == "ended"


def test_dashboard_timeout_does_not_reopen_after_late_toolkit_setup(
    monkeypatch,
) -> None:
    interaction = _DashboardInteraction()
    dashboard_events = _RecordingDashboardEvents()
    release = threading.Event()
    setup_started = threading.Event()

    class LateToolkit(_Toolkit):
        def get_tools_spec(self):
            setup_started.set()
            release.wait(timeout=1)
            return super().get_tools_spec()

    monkeypatch.setattr(planner_module, "_WORKER_CANCEL_GRACE_S", 0.02)
    toolkit = LateToolkit([{"_finish": True}])
    planner = ModelApiUtilsPlanner(
        client=_Client(_call()),
        model="default",
        dashboard_events=dashboard_events,
        timeout_s=0.02,
    )

    result = planner.solve(
        system_prompt="",
        user_message="task",
        toolkit=toolkit,
        max_turns=1,
        dashboard_interaction=interaction,
    )

    assert setup_started.is_set()
    assert result.error == "vLLM planner timed out after 0.02s"
    assert interaction.activity == "ended"
    assert interaction.accepting_input is False

    release.set()
    time.sleep(0.05)
    assert interaction.activity == "ended"
    assert interaction.accepting_input is False
    assert not any(
        isinstance(event, TranscriptEvent)
        and event.payload.get("type") == "initial_prompt"
        for event in dashboard_events.events
    )


@pytest.mark.parametrize("shared_runtime", [False, True])
def test_dashboard_interrupt_cancels_an_active_tool(shared_runtime: bool) -> None:
    interaction = _DashboardInteraction()
    toolkit = _BlockingToolkit([{"step": 0}])
    planner = ModelApiUtilsPlanner(
        client=_Client(_call()),
        model="default",
        timeout_s=2,
    )
    if shared_runtime:
        planner = _with_shared_runtime(planner)
    outcome = {}

    worker = threading.Thread(
        target=lambda: outcome.setdefault(
            "result",
            planner.solve(
                system_prompt="",
                user_message="task",
                toolkit=toolkit,
                max_turns=2,
                dashboard_interaction=interaction,
            ),
        )
    )
    worker.start()
    deadline = time.monotonic() + 1
    while not toolkit.calls and time.monotonic() < deadline:
        time.sleep(0.005)
    assert toolkit.calls == [("observe", {})]

    interaction.request_interrupt()
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert outcome["result"].error == "vLLM planner interrupted from the Dashboard"
    assert toolkit.cancelled.is_set()
    assert interaction.activity == "ended"


def test_input_queue_and_dashboard_interaction_are_mutually_exclusive() -> None:
    planner = ModelApiUtilsPlanner(client=_Client(_call()), model="default")

    try:
        planner.solve(
            system_prompt="",
            user_message="task",
            toolkit=_Toolkit([{"_finish": True}]),
            max_turns=1,
            input_queue=object(),
            dashboard_interaction=_DashboardInteraction(),
        )
    except ValueError as exc:
        assert str(exc) == (
            "input_queue and dashboard_interaction cannot be used together"
        )
    else:
        raise AssertionError("expected mutually exclusive interaction inputs")


def test_transcript_preserves_actual_provider_call_identity() -> None:
    sink = _RecordingDashboardEvents()
    toolkit = _Toolkit([{"_finish": True, "status": "success"}])
    result = ModelApiUtilsPlanner(client=_Client(_call(call_id="actual-provider-call")), model="default", dashboard_events=sink).solve(
        system_prompt="rules", user_message="task", toolkit=toolkit, max_turns=1,
    )
    assert result.error is None
    tool_events = [event.payload for event in sink.events if isinstance(event, TranscriptEvent) and event.payload.get("type") in {"tool_call", "tool_result"}]
    assert [event["type"] for event in tool_events] == ["tool_call", "tool_result"]
    assert [event["tool_call_id"] for event in tool_events] == ["actual-provider-call", "actual-provider-call"]


def test_missing_provider_id_does_not_invent_tool_transcript_identity() -> None:
    sink = _RecordingDashboardEvents()
    toolkit = _Toolkit([{"_finish": True, "status": "success"}])
    proposal = _call()
    del proposal["tool_calls"][0]["id"]
    result = ModelApiUtilsPlanner(client=_Client(proposal), model="default", dashboard_events=sink).solve(
        system_prompt="rules", user_message="task", toolkit=toolkit, max_turns=1,
    )
    assert result.error is None
    assert toolkit.calls == [("observe", {})]
    tool_events = [event.payload for event in sink.events if isinstance(event, TranscriptEvent) and event.payload.get("type") in {"tool_call", "tool_result"}]
    assert [event["type"] for event in tool_events] == ["tool_call", "tool_result"]
    assert all("tool_call_id" not in event for event in tool_events)


def test_planner_executes_one_call_and_finishes_with_usage() -> None:
    client = _Client(_call())
    toolkit = _Toolkit([{"_finish": True, "status": "success"}])
    planner = ModelApiUtilsPlanner(
        client=client,
        model="default",
        dashboard_events=NullDashboardEventSink(),
    )

    result = planner.solve(
        system_prompt="rules", user_message="task", toolkit=toolkit, max_turns=2
    )

    assert result.error is None
    assert result.finish_result["status"] == "success"
    stats = dict(result.stats)
    assert stats.pop("model_requests") == 1
    for key in (
        "model_elapsed_s",
        "max_model_request_elapsed_s",
        "tool_elapsed_s",
        "max_tool_elapsed_s",
    ):
        assert stats.pop(key) >= 0
    max_request_chars = stats.pop("max_request_chars")
    max_request_text_bytes = stats.pop("max_request_text_bytes")
    max_request_wire_bytes = stats.pop("max_request_wire_bytes")
    max_unprepared_request_text_bytes = stats.pop("max_unprepared_request_text_bytes")
    max_unprepared_request_wire_bytes = stats.pop("max_unprepared_request_wire_bytes")
    assert stats == {
        "turns_used": 1,
        "tool_calls": 1,
        "total_input_tokens": 3,
        "total_output_tokens": 2,
        "total_tokens": 5,
        "history_compactions": 0,
        "request_budget_compactions": 0,
        "request_text_budget_bytes": 131072,
        "request_image_budget_bytes": 4194304,
        "request_wire_budget_bytes": 4325376,
        "max_request_messages": 2,
        "max_request_image_bytes": 0,
        "max_unprepared_request_image_bytes": 0,
        "tool_execution_errors": 0,
        "loop_feedbacks": 0,
        "proposal_repairs": 0,
        "multi_tool_batches": 0,
        "stateful_batches_deferred": 0,
        "max_tool_calls_per_response": 1,
        "max_observations_without_action": 0,
        "stagnant_actions": 0,
        "state_changes": 0,
        "semantic_progress_feedbacks": 0,
        "max_equivalent_capped_actions_seen": 0,
        "workspace_tools_enabled": 0,
        "multi_tool_calls_enabled": 1,
        "context_overflow_retries": 0,
        "no_thinking_recovery_requests": 0,
    }
    assert max_request_chars > 70
    assert max_request_text_bytes > 375
    assert max_request_text_bytes == max_request_wire_bytes
    assert max_unprepared_request_text_bytes == max_request_text_bytes
    assert max_unprepared_request_wire_bytes == max_request_wire_bytes
    assert client.calls[0]["extra_body"]["parallel_tool_calls"] is True
    assert client.calls[0]["extra_body"]["tool_choice"] == "required"
    assert client.calls[0]["extra_body"]["tools"][0]["function"]["strict"] is True


def test_workspace_catalog_and_lifecycle_are_published_before_use(tmp_path) -> None:
    (tmp_path / "guide.md").write_text("needle here\n")
    dashboard_events = _RecordingDashboardEvents()
    client = _Client(
        _call('{"query":"needle"}', name="workspace_search"),
        _call(call_id="call-2"),
    )
    toolkit = _Toolkit(
        [{"_finish": True, "status": "success"}],
        dashboard_events=dashboard_events,
    )
    planner = ModelApiUtilsPlanner(
        client=client,
        model="default",
        dashboard_events=dashboard_events,
        workspace_profile=WorkspaceProfile(mode="readonly", root=tmp_path),
    )

    result = planner.solve(
        system_prompt="rules", user_message="task", toolkit=toolkit, max_turns=2
    )

    assert result.error is None
    catalog_events = [
        event
        for event in dashboard_events.events
        if isinstance(event, CapabilityCatalogEvent)
    ]
    assert len(catalog_events) == 1
    assert {entry.tool_name for entry in catalog_events[0].entries} == {
        "observe",
        "workspace_search",
        "workspace_shell",
    }
    lifecycle = [
        event for event in dashboard_events.events if isinstance(event, ToolCallEvent)
    ]
    assert [(event.name, event.phase) for event in lifecycle] == [
        ("workspace_search", "started"),
        ("workspace_search", "returned"),
    ]
    assert lifecycle[0].call_id == lifecycle[1].call_id
    assert lifecycle[0].argument_keys == ("query",)
    assert dashboard_events.events.index(catalog_events[0]) < dashboard_events.events.index(
        lifecycle[0]
    )

    disabled_events = _RecordingDashboardEvents()
    disabled = ModelApiUtilsPlanner(
        client=_Client(_call()),
        model="default",
        dashboard_events=disabled_events,
    )
    disabled.solve(
        system_prompt="rules",
        user_message="task",
        toolkit=_Toolkit([{"_finish": True}], dashboard_events=disabled_events),
        max_turns=1,
    )
    assert not any(
        isinstance(event, CapabilityCatalogEvent) for event in disabled_events.events
    )


def test_planner_projects_oversized_tool_result_without_losing_safety_facts() -> None:
    client = _Client(_call(), _call(call_id="call-2"))
    toolkit = _Toolkit(
        [
            {
                "status": "HOLD",
                "safe": False,
                "hold_reason": "collision envelope violated",
                "hold_confirmed": True,
                "lower_hold_confirmed": True,
                "proves_motion_or_hold": True,
                "authority": "lower_runtime",
                "execution_id": "exec-17",
                "world_bounds_xyz": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                "content": "begin-" + "x" * 100_000 + "-end",
            },
            {"_finish": True, "status": "failure"},
        ]
    )
    planner = ModelApiUtilsPlanner(client=client, model="default")

    result = planner.solve(
        system_prompt="rules", user_message="task", toolkit=toolkit, max_turns=2
    )

    assert result.error is None
    tool_message = next(
        message
        for message in client.calls[1]["messages"]
        if message.get("role") == "tool"
    )
    assert len(tool_message["content"].encode("utf-8")) <= 60_000
    projected = json.loads(tool_message["content"])
    assert projected["_rpent_projection"]["reason"] == "tool_result_limit"
    assert projected["status"] == "HOLD"
    assert projected["safe"] is False
    assert projected["hold_reason"] == "collision envelope violated"
    assert projected["hold_confirmed"] is True
    assert projected["lower_hold_confirmed"] is True
    assert projected["proves_motion_or_hold"] is True
    assert projected["authority"] == "lower_runtime"
    assert projected["execution_id"] == "exec-17"
    assert projected["world_bounds_xyz"] == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert projected["content_excerpt"].startswith("begin-")
    assert projected["content_excerpt"].endswith("-end")


def test_projection_preserves_deep_nested_authoritative_scalars() -> None:
    safety = {
        **{f"detail_{index}": "optional" for index in range(40)},
        "accepted": False,
        "hold_confirmed": True,
        "lower_hold_confirmed": True,
        "proves_motion_or_hold": True,
        "reason_code": "collision_envelope",
        "reconciliation_required": True,
        "retry_allowed": False,
        "safe": False,
    }
    rendered = bounded_tool_result_text(
        tool_name="act",
        result={
            "result": {"execution": {"receipt": {"safety": safety}}},
            "content": "x" * 100_000,
        },
        max_bytes=512,
        reason="test",
    )

    projected = json.loads(rendered)
    projected_safety = projected["result"]["execution"]["receipt"]["safety"]
    for key in (
        "accepted",
        "hold_confirmed",
        "lower_hold_confirmed",
        "proves_motion_or_hold",
        "reconciliation_required",
        "retry_allowed",
        "safe",
    ):
        assert projected_safety[key] is safety[key]
    assert projected_safety["reason_code"] == "collision_envelope"
    assert projected_safety["_omitted_key_count"] > 0


def test_projection_accepts_nested_mappings_with_mixed_key_types() -> None:
    rendered = bounded_tool_result_text(
        tool_name="observe",
        result={
            "status": "ok",
            "payload": {1: "one", "two": 2},
            "content": "x" * 100_000,
        },
        max_bytes=2_000,
        reason="test",
    )

    projected = json.loads(rendered)
    assert projected["payload"] == {"1": "one", "two": 2}


def test_projection_preserves_selected_candidate_geometry_at_minimum_budget() -> None:
    critical_candidate = {
        "ordinal": 2,
        "T_reference_from_tool_tcp": [
            1.0,
            0.0,
            0.0,
            0.4,
            0.0,
            1.0,
            0.0,
            0.1,
            0.0,
            0.0,
            1.0,
            0.2,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        "tool_tcp_xyz_reference_m": [0.4, 0.1, 0.2],
        "visible_scene_collision_free": True,
        "classification": "standard_reachable",
        "path_planning_eligible": True,
        "fits_with_clearance": True,
        "clearance_m": 0.01,
    }
    candidate = {
        **critical_candidate,
        "score": 0.91,
        "model_score": 0.88,
        "click_distance_m": 0.004,
        "gripper_width_m": 0.04,
    }
    rendered = bounded_tool_result_text(
        tool_name="select_point_grasp",
        result={
            "selected_candidate": candidate,
            "content": "x" * 100_000,
        },
        max_bytes=512,
        reason="test",
    )

    projected = json.loads(rendered)
    selected = projected["selected_candidate"]
    assert {key: selected[key] for key in critical_candidate} == critical_candidate
    assert selected["_omitted_key_count"] == 4
    assert not set(candidate).difference(critical_candidate) & set(selected)


def test_planner_reports_projection_failure_after_tool_execution(monkeypatch) -> None:
    monkeypatch.setattr(ToolResult, "MAX_TEXT_BYTES_IN_RESULT", 512)
    client = _Client(_call())
    result_fields = {f"field_{index}_id": "x" * 100 for index in range(100)}
    toolkit = _Toolkit([{"status": "HOLD", **result_fields}])
    planner = ModelApiUtilsPlanner(client=client, model="default")

    result = planner.solve(
        system_prompt="rules", user_message="task", toolkit=toolkit, max_turns=1
    )

    assert "executed but its result could not be represented safely" in result.error
    assert toolkit.calls == [("observe", {})]
    assert result.stats["tool_calls"] == 1
    audit = json.loads(result.messages[-1]["content"])
    assert audit == {
        "schema": "rpent.executed_tool_result_unavailable.v1",
        "status": "executed_result_unavailable",
        "error": (
            "authoritative tool result fields exceed compacted result budget; "
            "tool=observe; max_bytes=512"
        ),
        "tool": "observe",
        "tool_call_id": "call-1",
        "executed": True,
        "retry_allowed": False,
        "reconciliation_required": True,
    }


def test_planner_enforces_exact_outbound_text_budget_on_every_request() -> None:
    client = _Client(_call(), _call(call_id="call-2"), _call(call_id="call-3"))
    toolkit = _Toolkit(
        [
            {"step": 1, "status": "ok", "content": "a" * 20_000},
            {"step": 2, "status": "ok", "content": "b" * 20_000},
            {"_finish": True, "status": "success"},
        ]
    )
    policy = ToolHistoryPolicy(
        max_request_bytes=4_500,
        compacted_tool_result_bytes=1_200,
        max_summary_chars=512,
    )
    planner = ModelApiUtilsPlanner(
        client=client,
        model="default",
        history_policy=policy,
    )

    result = planner.solve(
        system_prompt="rules", user_message="task", toolkit=toolkit, max_turns=3
    )

    assert result.error is None
    assert result.stats["request_budget_compactions"] == 2
    assert result.stats["max_request_text_bytes"] <= policy.max_request_bytes
    assert result.stats["max_unprepared_request_text_bytes"] > policy.max_request_bytes
    for call in client.calls:
        payload = _build_chat_payload(
            messages=call["messages"],
            model=call["model"],
            temperature=call["temperature"],
            max_tokens=call["max_tokens"],
            extra_body=call["extra_body"],
        )
        assert _chat_payload_text_bytes(payload) <= policy.max_request_bytes
        assert (
            _chat_payload_inline_image_bytes(payload) <= policy.max_request_image_bytes
        )
        assert _chat_payload_wire_bytes(payload) <= policy.max_request_wire_bytes


def test_planner_does_not_call_model_when_required_context_exceeds_budget() -> None:
    client = _Client(_call())
    toolkit = _Toolkit([{"_finish": True}])
    policy = ToolHistoryPolicy(
        max_request_bytes=1_024,
        compacted_tool_result_bytes=512,
        max_summary_chars=512,
    )
    planner = ModelApiUtilsPlanner(
        client=client,
        model="default",
        history_policy=policy,
    )

    result = planner.solve(
        system_prompt="required rules " + "x" * 2_000,
        user_message="task",
        toolkit=toolkit,
        max_turns=1,
    )

    assert result.error.startswith("RequestBudgetExceeded:")
    assert client.calls == []
    assert toolkit.calls == []
    assert result.stats["request_budget_compactions"] == 1
    assert result.stats["max_unprepared_request_text_bytes"] > 1_024


def test_planner_accepts_bounded_batch_of_safe_observations() -> None:
    first = _call('{"step":1}', call_id="call-1")
    first["tool_calls"].extend(_call('{"step":2}', call_id="call-2")["tool_calls"])
    client = _Client(first, _call(call_id="call-3"))
    toolkit = _Toolkit(
        [{"step": 1}, {"step": 2}, {"_finish": True, "status": "success"}]
    )
    planner = ModelApiUtilsPlanner(client=client, model="default")

    result = planner.solve(
        system_prompt="", user_message="task", toolkit=toolkit, max_turns=2
    )

    assert result.error is None
    assert toolkit.calls == [
        ("observe", {"step": 1}),
        ("observe", {"step": 2}),
        ("observe", {}),
    ]
    assert result.stats["tool_calls"] == 3
    assert result.stats["multi_tool_batches"] == 1
    assert result.stats["max_tool_calls_per_response"] == 2
    second_request = client.calls[1]["messages"]
    tool_results = [
        message for message in second_request if message.get("role") == "tool"
    ]
    assert [message["tool_call_id"] for message in tool_results] == [
        "call-1",
        "call-2",
    ]


def test_planner_repairs_duplicate_calls_inside_one_batch() -> None:
    duplicate = _call('{"step":0}', call_id="call-1")
    duplicate["tool_calls"].extend(_call('{"step":0}', call_id="call-2")["tool_calls"])
    client = _Client(duplicate, _call('{"step":1}', call_id="call-3"))
    toolkit = _Toolkit([{"_finish": True, "status": "success"}])
    planner = ModelApiUtilsPlanner(client=client, model="default")

    result = planner.solve(
        system_prompt="", user_message="task", toolkit=toolkit, max_turns=2
    )

    assert result.error is None
    assert toolkit.calls == [("observe", {"step": 1})]
    assert result.stats["proposal_repairs"] == 1
    repair = json.loads(client.calls[1]["messages"][-1]["content"])
    assert repair["code"] == "duplicate_tool_call"
    assert repair["executed"] is False
    rejected = json.loads(repair["rejected_proposal"])
    assert rejected["tool_calls"] == duplicate["tool_calls"]
    assert not any(
        message["role"] in {"assistant", "tool"}
        for message in client.calls[1]["messages"]
    )


def test_planner_appends_batch_images_after_every_tool_result() -> None:
    first = _call('{"step":1}', call_id="call-1")
    first["tool_calls"].extend(_call('{"step":2}', call_id="call-2")["tool_calls"])
    client = _Client(first, _call(call_id="call-3"))
    toolkit = _Toolkit(
        [
            {"step": 1, "_image_bytes": b"one"},
            {"step": 2, "_image_bytes": b"two"},
            {"_finish": True},
        ]
    )
    planner = ModelApiUtilsPlanner(client=client, model="default")

    result = planner.solve(
        system_prompt="", user_message="task", toolkit=toolkit, max_turns=2
    )

    assert result.error is None
    roles = [message["role"] for message in client.calls[1]["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "tool", "user"]
    image_message = client.calls[1]["messages"][-1]["content"]
    assert [item["type"] for item in image_message].count("image_url") == 2


def test_planner_defers_batch_after_first_non_batchable_tool() -> None:
    first = _call(call_id="call-1", name="set_gripper")
    first["tool_calls"].extend(_call(call_id="call-2", name="move_to")["tool_calls"])
    client = _Client(first, _call(call_id="call-3"))
    toolkit = _Toolkit(
        [{"gripper": "closed"}, {"_finish": True, "status": "success"}],
        multi_call_safe=False,
    )
    planner = ModelApiUtilsPlanner(client=client, model="default")

    result = planner.solve(
        system_prompt="", user_message="task", toolkit=toolkit, max_turns=2
    )

    assert result.error is None
    assert toolkit.calls == [("set_gripper", {}), ("observe", {})]
    assert result.stats["proposal_repairs"] == 0
    assert result.stats["stateful_batches_deferred"] == 1
    assert result.stats["max_tool_calls_per_response"] == 2
    second_request = client.calls[1]["messages"]
    assistant = next(
        message for message in second_request if message["role"] == "assistant"
    )
    assert [call["id"] for call in assistant["tool_calls"]] == ["call-1"]
    feedback = json.loads(second_request[-1]["content"])
    assert feedback == {
        "schema": "rpent.planner_execution_feedback.v1",
        "stage": "planner_call",
        "code": "stateful_batch_deferred",
        "executed_tool": "set_gripper",
        "deferred_tools": ["move_to"],
        "instruction": (
            "Only the first call was executed. The remaining calls were not "
            "executed because they may depend on pre-action state; inspect the "
            "tool result and re-plan."
        ),
    }
    transcript_feedback = next(
        message
        for message in result.messages
        if message["role"] == "planner_execution_feedback"
    )
    assert transcript_feedback["content"] == feedback
    assert result.messages.index(transcript_feedback) > next(
        index
        for index, message in enumerate(result.messages)
        if message.get("tool_call_id") == "call-1"
    )


def test_multi_tool_mode_can_restore_one_call_behavior() -> None:
    first = _call(call_id="call-1")
    first["tool_calls"].extend(_call(call_id="call-2")["tool_calls"])
    client = _Client(first, _call(call_id="call-3"))
    planner = ModelApiUtilsPlanner(
        client=client,
        model="default",
        multi_tool_policy=MultiToolPolicy(enabled=False),
    )

    result = planner.solve(
        system_prompt="",
        user_message="task",
        toolkit=_Toolkit([{"_finish": True}]),
        max_turns=2,
    )

    assert result.error is None
    assert client.calls[0]["extra_body"]["parallel_tool_calls"] is False
    assert "multi_tool_calls_disabled" in client.calls[1]["messages"][-1]["content"]


def test_multi_tool_policy_reads_bounded_environment() -> None:
    policy = multi_tool_policy_from_env(
        {
            "RPENT_VLLM_MULTI_TOOL_MODE": "read_only",
            "RPENT_VLLM_MAX_MULTI_TOOL_CALLS": "3",
        }
    )

    assert policy.enabled is True
    assert policy.max_calls == 3


def test_multi_tool_policy_defaults_to_documented_maximum() -> None:
    assert multi_tool_policy_from_env({}).max_calls == 8


def test_planner_exposes_batch_limit_and_safe_tools_to_model() -> None:
    client = _Client(_call())
    planner = ModelApiUtilsPlanner(client=client, model="default")

    result = planner.solve(
        system_prompt="robot rules",
        user_message="task",
        toolkit=_Toolkit([{"_finish": True}]),
        max_turns=1,
    )

    assert result.error is None
    system = client.calls[0]["messages"][0]
    assert system["role"] == "system"
    assert "robot rules" in system["content"]
    assert "at most 8" in system["content"]
    assert "batch-safe tools: observe" in system["content"]


def test_planner_preserves_reasoning_content_for_diagnostics() -> None:
    message = _call()
    message["reasoning_content"] = "inspect, then act"
    client = _Client(message)
    planner = ModelApiUtilsPlanner(client=client, model="default")

    result = planner.solve(
        system_prompt="",
        user_message="task",
        toolkit=_Toolkit([{"_finish": True}]),
        max_turns=1,
    )

    assert result.messages[1]["reasoning_content"] == "inspect, then act"
    assert all(
        "concise user-facing progress update" not in str(item.get("content", ""))
        for item in client.calls[0]["messages"]
    )


def test_planner_preserves_service_reasoning_field_in_history_and_transcript() -> None:
    message = _call()
    message["reasoning"] = "inspect current state"
    client = _Client(message, _call(call_id="call-2"))
    planner = ModelApiUtilsPlanner(client=client, model="default")

    result = planner.solve(
        system_prompt="",
        user_message="task",
        toolkit=_Toolkit([{"step": 1}, {"_finish": True}]),
        max_turns=2,
    )

    assert result.messages[1]["reasoning_content"] == "inspect current state"
    assistant = next(
        message
        for message in client.calls[1]["messages"]
        if message.get("role") == "assistant"
    )
    assert assistant["reasoning"] == "inspect current state"


def test_planner_repairs_invalid_arguments_with_a_bound() -> None:
    client = _Client(_call("not-json"), _call("still-not-json"))
    planner = ModelApiUtilsPlanner(
        client=client,
        model="default",
        max_proposal_repairs=1,
    )

    result = planner.solve(
        system_prompt="", user_message="task", toolkit=_Toolkit([]), max_turns=3
    )

    assert result.error == "Planner proposal repair budget exhausted"
    assert len(client.calls) == 2


@pytest.mark.parametrize(
    "arguments", ["not-json", "not-json" + "错" * 20000], ids=["small", "oversized"]
)
def test_planner_keeps_bounded_rejected_arguments_without_dispatch(arguments) -> None:
    client = _Client(_call(arguments), _call(call_id="safe-call"))
    toolkit = _Toolkit([{"_finish": True}])
    planner = ModelApiUtilsPlanner(client=client, model="default")

    result = planner.solve(
        system_prompt="", user_message="task", toolkit=toolkit, max_turns=2
    )

    assert result.error is None
    assert toolkit.calls == [("observe", {})]
    feedback = json.loads(client.calls[1]["messages"][-1]["content"])
    assert feedback["code"] == "tool_arguments_json_invalid"
    assert feedback["executed"] is False
    rejected = feedback["rejected_proposal"]
    assert len(rejected.encode("utf-8")) <= 4096
    if arguments == "not-json":
        proposal = json.loads(rejected)
        assert proposal["tool_calls"][0]["function"]["name"] == "observe"
        assert proposal["tool_calls"][0]["function"]["arguments"] == arguments
    else:
        assert '"name":"observe"' in rejected
        assert rejected.endswith("[truncated]")
    assert result.messages[1]["content"] == feedback


def test_planner_missing_tool_feedback_names_safe_finish_path() -> None:
    client = _Client(
        {"role": "assistant", "content": "The task looks complete."},
        _call(
            '{"status":"stuck","summary":"native completion was not confirmed"}',
            name="finish",
        ),
    )
    toolkit = _Toolkit([{"_finish": True, "status": "stuck"}])
    planner = ModelApiUtilsPlanner(client=client, model="default")

    result = planner.solve(
        system_prompt="", user_message="task", toolkit=toolkit, max_turns=2
    )

    assert result.error is None
    assert result.stats["proposal_repairs"] == 1
    repair = json.loads(client.calls[1]["messages"][-1]["content"])
    assert repair["code"] == "planner_tool_calls_missing"
    assert repair["actual"] == (
        "assistant prose without tool call: The task looks complete."
    )
    assert "finish with status='failure' or status='stuck'" in repair["expected"]
    assert "do not respond with prose alone" in repair["expected"]


def test_planner_retries_missing_tool_call_once_without_thinking_then_restores() -> (
    None
):
    client = _Client(
        {"role": "assistant", "content": None},
        _call(call_id="call-2"),
        _call(call_id="call-3"),
    )
    toolkit = _Toolkit([{"step": 1}, {"_finish": True, "status": "success"}])
    planner = ModelApiUtilsPlanner(
        client=client,
        model="default",
        enable_thinking=True,
    )

    result = planner.solve(
        system_prompt="", user_message="task", toolkit=toolkit, max_turns=3
    )

    assert result.error is None
    assert [call["extra_body"]["enable_thinking"] for call in client.calls] == [
        True,
        False,
        True,
    ]
    assert result.stats["proposal_repairs"] == 1
    assert result.stats["no_thinking_recovery_requests"] == 1


def test_planner_reports_empty_reasoning_only_response_metadata() -> None:
    class ReasoningOnlyClient(_Client):
        def chat(self, **kwargs):
            self.calls.append(copy.deepcopy(kwargs))
            return SimpleNamespace(
                raw={
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "reasoning_content": "inspect forever",
                            },
                        }
                    ]
                },
                usage={},
            )

    planner = ModelApiUtilsPlanner(
        client=ReasoningOnlyClient(),
        model="default",
        enable_thinking=True,
        max_proposal_repairs=0,
    )

    result = planner.solve(
        system_prompt="", user_message="task", toolkit=_Toolkit([]), max_turns=1
    )

    repair = result.messages[-1]["content"]
    assert repair["code"] == "planner_tool_calls_missing"
    assert repair["actual"] == (
        "empty assistant message; finish_reason='length'; reasoning_chars=15"
    )


def test_planner_repairs_missing_top_level_required_argument() -> None:
    class RequiredToolkit(_Toolkit):
        def get_tools_spec(self):
            specs = super().get_tools_spec()
            specs[0]["input_schema"] = {
                "type": "object",
                "properties": {
                    "camera": {
                        "type": "string",
                        "enum": ["agentview", "wrist"],
                        "description": "Choose the source camera explicitly.",
                    },
                    "prompt": {"type": ["string", "null"]},
                    "point": {"type": ["array", "null"]},
                },
                "required": ["camera"],
                "oneOf": [
                    {"required": ["prompt"]},
                    {"required": ["point"]},
                ],
            }
            return specs

    client = _Client(
        _call(),
        _call('{"camera":"wrist","prompt":"object","point":null}'),
        _call('{"camera":"agentview","prompt":"target","point":null}'),
    )
    toolkit = RequiredToolkit(
        [{"status": "observed"}, {"_finish": True, "status": "success"}]
    )
    planner = ModelApiUtilsPlanner(client=client, model="default")

    result = planner.solve(
        system_prompt="", user_message="task", toolkit=toolkit, max_turns=3
    )

    assert result.error is None
    assert result.stats["proposal_repairs"] == 1
    assert toolkit.calls == [
        ("observe", {"camera": "wrist", "prompt": "object", "point": None}),
        ("observe", {"camera": "agentview", "prompt": "target", "point": None}),
    ]
    repair = json.loads(client.calls[1]["messages"][-1]["content"])
    assert repair["code"] == "tool_argument_required"
    assert repair["path"].endswith("/camera")
    assert repair["retry_tool"] == "observe"
    assert "'observe'" in repair["expected"]
    assert '["agentview", "wrist"]' in repair["expected"]
    assert "Choose the source camera explicitly." in repair["expected"]
    assert client.calls[1]["extra_body"]["tool_choice"] == {
        "type": "function",
        "function": {"name": "observe"},
    }
    assert client.calls[2]["extra_body"]["tool_choice"] == "required"
    parameters = client.calls[0]["extra_body"]["tools"][0]["function"]["parameters"]
    assert "oneOf" not in parameters
    assert parameters["required"] == ["camera", "prompt", "point"]


@pytest.mark.skip(
    reason="back_project schema ownership belongs to the current RPent LIBERO Toolkit"
)
def test_planner_keeps_back_project_owner_contract_after_schema_flattening(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        templates, "default_variables", lambda: {"output_dir": str(tmp_path)}
    )
    state = EnvState(tmp_path / "state")
    world = np.ones((4, 4, 3), dtype=np.float32)
    world[..., 2] = 0.4255
    with state.record_step(state={}):
        state.save("agentview_world_high.npz", world)
    surface = Toolkit(
        dashboard_events=NullDashboardEventSink(),
        state=state,
        memory=MemoryManager(tmp_path / "memory"),
    )
    spec = copy.deepcopy(
        next(item for item in libero_tools.TOOLS_SPEC if item["name"] == "back_project")
    )
    original_schema = copy.deepcopy(spec["input_schema"])
    surface.add_tool(
        "back_project", spec, partial(libero_tools.back_project, state=state)
    )
    region = {"row_range": [0, 4], "col_range": [0, 4], "z_min": 0.5, "z_max": 0.7}
    client = _Client(
        _call(
            json.dumps({"row": 1, "col": 1, **region}),
            name="back_project",
            call_id="mixed",
        ),
        _call(
            json.dumps({"row": None, "col": None, **region}),
            name="back_project",
            call_id="filtered",
        ),
        _call(
            json.dumps({"row": 1, "col": 1, "row_range": None, "col_range": None}),
            name="back_project",
            call_id="pixel",
        ),
        _call(
            '{"status":"stuck","summary":"read-only probe complete"}',
            name="finish",
            call_id="done",
        ),
    )
    planner = _with_shared_runtime(ModelApiUtilsPlanner(client=client, model="default"))

    result = planner.solve(
        system_prompt="", user_message="probe", toolkit=surface, max_turns=4
    )

    assert result.error is None
    assert result.finish_result["status"] == "stuck"
    tool_results = [
        json.loads(message["content"])
        for message in result.messages
        if message.get("role") == "tool" and message.get("name") == "back_project"
    ]
    assert isinstance(tool_results[0].get("error"), str)
    assert tool_results[1]["n_valid_before_zfilter"] == 16
    assert isinstance(tool_results[1].get("error"), str)
    assert tool_results[2]["world_xyz"] == [1.0, 1.0, 0.4255]
    next_request_results = [
        json.loads(message["content"])
        for message in client.calls[2]["messages"]
        if message.get("role") == "tool"
    ]
    assert isinstance(next_request_results[-1].get("error"), str)
    parameters = next(
        item["function"]["parameters"]
        for item in client.calls[0]["extra_body"]["tools"]
        if item["function"]["name"] == "back_project"
    )
    assert "oneOf" not in parameters
    assert set(parameters.get("required", ())) == set()
    assert parameters["properties"]["row_range"]["maxItems"] == 2
    mixed_arguments = {"row": 1, "col": 1, **region}
    assert not validator_for(original_schema)(original_schema).is_valid(mixed_arguments)
    assert validator_for(parameters)(parameters).is_valid(mixed_arguments)
    assert not validator_for(parameters)(parameters).is_valid(
        {**mixed_arguments, "row_range": [0, 4, 99]}
    )
    assert spec["input_schema"] == original_schema
    assert len(state.records()) == 1


def test_planner_keeps_only_two_newest_tool_images() -> None:
    client = _Client(_call(), _call(), _call())
    toolkit = _Toolkit(
        [
            {"step": 1, "_image_bytes": b"one"},
            {"step": 2, "_image_bytes": b"two"},
            {"_finish": True, "status": "success", "_image_bytes": b"three"},
        ]
    )
    planner = ModelApiUtilsPlanner(client=client, model="default")

    result = planner.solve(
        system_prompt="", user_message="task", toolkit=toolkit, max_turns=3
    )

    assert result.error is None
    third_request = client.calls[2]["messages"]
    image_items = [
        item
        for message in third_request
        if isinstance(message.get("content"), list)
        for item in message["content"]
        if item.get("type") == "image_url"
    ]
    assert len(image_items) == 2


def test_planner_prunes_recent_images_to_cumulative_byte_budget() -> None:
    client = _Client(_call(), _call(), _call())
    toolkit = _Toolkit(
        [
            {"step": 1, "_image_bytes": b"a" * 400},
            {"step": 2, "_image_bytes": b"b" * 400},
            {"_finish": True, "status": "success"},
        ]
    )
    policy = ToolHistoryPolicy(max_request_image_bytes=700)
    planner = ModelApiUtilsPlanner(
        client=client,
        model="default",
        history_policy=policy,
    )

    result = planner.solve(
        system_prompt="", user_message="task", toolkit=toolkit, max_turns=3
    )

    assert result.error is None
    payload = _build_chat_payload(
        messages=client.calls[2]["messages"],
        model=client.calls[2]["model"],
        temperature=client.calls[2]["temperature"],
        max_tokens=client.calls[2]["max_tokens"],
        extra_body=client.calls[2]["extra_body"],
    )
    assert 0 < _chat_payload_inline_image_bytes(payload) <= 700
    assert _chat_payload_wire_bytes(payload) <= policy.max_request_wire_bytes
    assert result.stats["request_budget_compactions"] == 1


def test_planner_fails_before_resending_an_oversized_newest_image() -> None:
    client = _Client(_call(), _call(call_id="unused"))
    toolkit = _Toolkit([{"step": 1, "_image_bytes": b"x" * 1_000}])
    policy = ToolHistoryPolicy(max_request_image_bytes=512)
    planner = ModelApiUtilsPlanner(
        client=client,
        model="default",
        history_policy=policy,
    )

    result = planner.solve(
        system_prompt="", user_message="task", toolkit=toolkit, max_turns=2
    )

    assert result.error.startswith("RequestBudgetExceeded:")
    assert "newest inline image exceeds" in result.error
    assert len(client.calls) == 1
    assert toolkit.calls == [("observe", {})]
    assert result.stats["max_unprepared_request_image_bytes"] > 512


def test_planner_compacts_old_tool_exchanges_deterministically() -> None:
    calls = [_call() for _ in range(5)]
    client = _Client(*calls)
    toolkit = _Toolkit(
        [
            {"step": 1, "large": "x" * 2000},
            {"step": 2, "large": "x" * 2000},
            {"step": 3, "large": "x" * 2000},
            {"step": 4, "large": "x" * 2000},
            {"_finish": True, "status": "success"},
        ]
    )
    planner = ModelApiUtilsPlanner(
        client=client,
        model="default",
        history_policy=ToolHistoryPolicy(
            recent_tool_exchanges=2, max_summary_chars=1000
        ),
    )

    result = planner.solve(
        system_prompt="rules", user_message="task", toolkit=toolkit, max_turns=5
    )

    assert result.error is None
    assert result.stats["history_compactions"] == 2
    final_messages = client.calls[-1]["messages"]
    summaries = [
        message["content"]
        for message in final_messages
        if message.get("role") == "user"
        and isinstance(message.get("content"), str)
        and message["content"].startswith("[deterministic earlier tool history")
    ]
    assert len(summaries) == 1
    assert 'tool_counts={"observe":2}' in summaries[0]
    assert "x" * 100 not in summaries[0]


def test_history_never_splits_a_multi_tool_response() -> None:
    messages = [
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "old", "function": {"name": "observe", "arguments": {}}}
            ],
        },
        {"role": "tool", "tool_call_id": "old", "name": "observe", "content": "{}"},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "a", "function": {"name": "observe", "arguments": {}}},
                {"id": "b", "function": {"name": "observe", "arguments": {}}},
            ],
        },
        {"role": "tool", "tool_call_id": "a", "name": "observe", "content": "{}"},
        {"role": "tool", "tool_call_id": "b", "name": "observe", "content": "{}"},
    ]

    prepared, compacted = ToolHistoryPolicy(recent_tool_exchanges=1).prepare(messages)

    assert compacted is True
    recent_assistant = next(
        message
        for message in prepared
        if message.get("role") == "assistant"
        and len(message.get("tool_calls", [])) == 2
    )
    assert [call["id"] for call in recent_assistant["tool_calls"]] == ["a", "b"]
    assert [
        message["tool_call_id"] for message in prepared if message.get("role") == "tool"
    ] == ["a", "b"]


def test_history_compacts_on_request_size_before_exchange_limit() -> None:
    messages = [
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "old", "function": {"name": "observe", "arguments": {}}}
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "old",
            "name": "observe",
            "content": json.dumps({"step": 0, "raw": "x" * 20000}),
        },
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "recent", "function": {"name": "observe", "arguments": {}}}
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "recent",
            "name": "observe",
            "content": json.dumps({"step": 1}),
        },
    ]

    prepared, compacted = ToolHistoryPolicy(
        recent_tool_exchanges=12,
        max_summary_chars=1000,
        max_request_bytes=16384,
    ).prepare(messages)

    assert compacted is True
    assert len(json.dumps(prepared, separators=(",", ":")).encode("utf-8")) <= 16384
    assert "x" * 100 not in json.dumps(prepared)
    assert any(
        message.get("tool_call_id") == "recent"
        for message in prepared
        if message.get("role") == "tool"
    )


def test_planner_retries_one_context_overflow_with_one_latest_image() -> None:
    overflow = RuntimeError(
        "vLLM rejected the request: maximum context length is 101376 tokens; "
        "prompt contains 93185 input tokens"
    )
    client = _Client(
        _call(call_id="call-1"),
        overflow,
        _call(call_id="call-2"),
    )
    toolkit = _Toolkit(
        [
            {
                "step": 1,
                "_image_bytes": b"one",
                "_image_cam_bytes": b"two",
            },
            {"_finish": True, "status": "success"},
        ]
    )
    sink = _RecordingDashboardEvents()
    planner = ModelApiUtilsPlanner(
        client=client,
        model="default",
        dashboard_events=sink,
    )

    result = planner.solve(
        system_prompt="rules", user_message="task", toolkit=toolkit, max_turns=2
    )

    assert result.error is None
    assert result.stats["context_overflow_retries"] == 1
    assert result.stats["model_requests"] == 3
    regular_images = [
        item
        for message in client.calls[1]["messages"]
        if isinstance(message.get("content"), list)
        for item in message["content"]
        if item.get("type") == "image_url"
    ]
    retry_images = [
        item
        for message in client.calls[2]["messages"]
        if isinstance(message.get("content"), list)
        for item in message["content"]
        if item.get("type") == "image_url"
    ]
    assert len(regular_images) == 2
    assert len(retry_images) == 1
    request_events = [
        event for event in sink.events if isinstance(event, PlannerRequestUsageEvent)
    ]
    assert [event.request_index for event in request_events] == [1, 3]
    assert request_events[-1].context_overflow_retry is True
    assert request_events[-1].compacted is False
    retry_call = client.calls[2]
    retry_payload = _build_chat_payload(
        messages=retry_call["messages"],
        model=retry_call["model"],
        temperature=retry_call["temperature"],
        max_tokens=retry_call["max_tokens"],
        extra_body=retry_call["extra_body"],
    )
    assert request_events[-1].request_image_bytes == _chat_payload_inline_image_bytes(
        retry_payload
    )
    assert request_events[-1].request_text_bytes == _chat_payload_text_bytes(
        retry_payload
    )
    assert request_events[-1].request_wire_bytes == _chat_payload_wire_bytes(
        retry_payload
    )


def test_history_summary_retains_geometry_and_robot_state_facts() -> None:
    messages = [
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "segment-1",
                    "function": {
                        "name": "segment",
                        "arguments": {"text": "bowl"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "segment-1",
            "name": "segment",
            "content": json.dumps(
                {
                    "step": 0,
                    "score": 0.97,
                    "world_xyz": [0.1, 0.2, 0.9],
                    "world_center_xyz": [0.11, 0.21, 0.9],
                    "eef_to_mask_center_xyz": [-0.02, 0.03, -0.04],
                    "target_eef_xy": [0.13, 0.18],
                    "minimum_lateral_eef_z": 0.95,
                    "fits_with_clearance": True,
                    "transition_evidence_accepted": False,
                    "transition_evidence_reason": "segment_predates_checkpoint",
                    "transition_checkpoint": {
                        "phase": "placement_plan_required",
                        "reference_step": 2,
                        "reason": "fresh_placement_plan_required",
                    },
                    "image_artifact": "x" * 2000,
                }
            ),
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "state-1",
                    "function": {"name": "view_env_state", "arguments": {}},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "state-1",
            "name": "view_env_state",
            "content": json.dumps(
                {
                    "step": 0,
                    "state": {
                        "robot0_eef_pos": [0.3, 0.4, 1.0],
                        "robot0_gripper_qpos": [0.03, -0.03],
                    },
                }
            ),
        },
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "recent", "function": {"name": "observe", "arguments": {}}}
            ],
        },
        {"role": "tool", "tool_call_id": "recent", "name": "observe", "content": "{}"},
    ]

    prepared, compacted = ToolHistoryPolicy(recent_tool_exchanges=1).prepare(messages)

    assert compacted is True
    summary = next(
        message["content"]
        for message in prepared
        if isinstance(message.get("content"), str)
        and message["content"].startswith("[deterministic earlier tool history")
    )
    assert '"world_xyz":[0.1,0.2,0.9]' in summary
    assert '"eef_to_mask_center_xyz":[-0.02,0.03,-0.04]' in summary
    assert '"target_eef_xy":[0.13,0.18]' in summary
    assert '"minimum_lateral_eef_z":0.95' in summary
    assert '"fits_with_clearance":true' in summary
    assert '"transition_evidence_accepted":false' in summary
    assert '"transition_evidence_reason":"segment_predates_checkpoint"' in summary
    assert '"phase":"placement_plan_required"' in summary
    assert '"reference_step":2' in summary
    assert '"robot0_eef_pos":[0.3,0.4,1.0]' in summary
    assert "x" * 100 not in summary


def test_history_summary_pins_authoritative_task_memory_before_pruned_tail() -> None:
    memory_path = "/memory/suite_task.md"
    messages = [
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "initial",
                    "function": {"name": "view_env_state", "arguments": {}},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "initial",
            "name": "view_env_state",
            "content": json.dumps(
                {
                    "step": 0,
                    "task_language": "put the target in the container",
                    "state": {"object_names": ["target_1", "container_1"]},
                    "task_memory_hints": {
                        "routing_status": "matched",
                        "suite_memories": [{"path": memory_path}],
                    },
                }
            ),
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "memory",
                    "function": {
                        "name": "read_text_file",
                        "arguments": {"path": memory_path},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "memory",
            "name": "read_text_file",
            "content": json.dumps(
                {
                    "path": memory_path,
                    "size": 1600,
                    "content": "context " * 80
                    + "The blue and yellow can is the target, not the red can.",
                }
            ),
        },
    ]
    for index in range(20):
        messages.extend(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": f"recent-{index}",
                            "function": {
                                "name": "move_to",
                                "arguments": {"xyz": [index, 0, 0]},
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": f"recent-{index}",
                    "name": "move_to",
                    "content": json.dumps({"step": index + 1}),
                },
            ]
        )

    prepared, compacted = ToolHistoryPolicy(
        recent_tool_exchanges=2, max_summary_chars=4000
    ).prepare(messages)

    assert compacted is True
    summary = next(
        message["content"]
        for message in prepared
        if isinstance(message.get("content"), str)
        and message["content"].startswith("[deterministic earlier tool history")
    )
    assert '"task_language":"put the target in the container"' in summary
    assert memory_path in summary
    assert "The blue and yellow can is the target, not the red can." in summary


def test_history_summary_pins_all_shared_task_reference_paths() -> None:
    paths = {
        "audit_path": "/memory/task_only/blue_s0.json",
        "recipe_path": "/memory/task_only/blue_s0_recipe.jsonl",
        "markdown_path": "/memory/task_only/blue.md",
    }
    messages = [
        {"role": "user", "content": "task"},
        {
            "role": "tool",
            "name": "view_env_state",
            "content": json.dumps(
                {
                    "task_language": "pick the blue block",
                    "task_memory_hints": {
                        "routing_status": "matched",
                        "task_references": [paths],
                    },
                }
            ),
        },
    ]

    from rpent_vllm_user.history import _history_anchors

    anchor = _history_anchors(messages, max_chars=4000)

    for path in paths.values():
        assert path in anchor


def test_history_summary_retains_provider_neutral_context_anchors() -> None:
    audit_path = "/memory/task_only/OpenDrawer_s0.json"
    recipe_path = "/memory/task_only/OpenDrawer_s0_recipe.jsonl"
    messages = [
        {"role": "user", "content": "open the left drawer"},
        {
            "role": "tool",
            "name": "read_task_memory",
            "content": json.dumps(
                {
                    "schema": "rpent.task_memory_bundle.v1",
                    "context_anchors": [
                        {
                            "kind": "audit",
                            "path": audit_path,
                            "present": True,
                            "content": "already parked; do not navigate",
                        },
                        {
                            "kind": "recipe",
                            "path": recipe_path,
                            "present": True,
                            "content": '{"action":"rldx_arm"}',
                        },
                    ],
                }
            ),
        },
    ]

    from rpent_vllm_user.history import _history_anchors

    anchor = _history_anchors(messages, max_chars=4000)

    assert audit_path in anchor
    assert recipe_path in anchor
    assert "already parked; do not navigate" in anchor
    assert "rldx_arm" in anchor


def test_planner_stops_reproposing_a_known_failed_call() -> None:
    client = _Client(_call(), _call(), _call(), _call())
    toolkit = _Toolkit([{"error": "bad args", "code": "invalid_arguments"}] * 4)
    planner = _with_shared_runtime(
        ModelApiUtilsPlanner(
            client=client,
            model="default",
            max_proposal_repairs=2,
        ),
        guard=ToolLoopGuard(max_identical_errors=3, max_consecutive_errors=10),
    )

    result = planner.solve(
        system_prompt="", user_message="task", toolkit=toolkit, max_turns=5
    )

    assert result.error == "Planner proposal repair budget exhausted"
    assert result.stats["tool_execution_errors"] == 1
    assert result.stats["loop_feedbacks"] == 1
    assert result.stats["proposal_repairs"] == 3
    assert toolkit.calls == [("observe", {})]
    assert result.messages[-1]["role"] == "repair_feedback"
    assert result.messages[-1]["content"]["code"] == "repeated_failed_tool_call"


def test_planner_accepts_changed_call_after_known_failure_repair() -> None:
    client = _Client(
        _call('{"step":0}', call_id="call-1"),
        _call('{"step":0}', call_id="call-2"),
        _call('{"step":1}', call_id="call-3"),
    )
    toolkit = _Toolkit(
        [
            {"error": "step not available", "code": "invalid_step"},
            {"_finish": True, "status": "success"},
        ]
    )
    planner = _with_shared_runtime(ModelApiUtilsPlanner(client=client, model="default"))

    result = planner.solve(
        system_prompt="", user_message="task", toolkit=toolkit, max_turns=3
    )

    assert result.error is None
    assert toolkit.calls == [
        ("observe", {"step": 0}),
        ("observe", {"step": 1}),
    ]
    assert result.stats["proposal_repairs"] == 1
    assert result.messages[-3]["role"] == "repair_feedback"
    assert result.messages[-3]["content"]["code"] == "repeated_failed_tool_call"


def test_planner_repair_budget_is_consecutive_not_lifetime() -> None:
    invalid = {"role": "assistant", "content": "reasoning", "tool_calls": []}
    client = _Client(invalid, _call(), invalid, _call())
    toolkit = _Toolkit([{"step": 1}, {"_finish": True, "status": "success"}])
    planner = ModelApiUtilsPlanner(
        client=client,
        model="default",
        max_proposal_repairs=1,
    )

    result = planner.solve(
        system_prompt="", user_message="task", toolkit=toolkit, max_turns=4
    )

    assert result.error is None
    assert result.stats["proposal_repairs"] == 2
    assert result.stats["tool_calls"] == 2


def test_planner_stops_repeated_unchanged_calls_before_more_execution() -> None:
    client = _Client(_call(), _call(), _call(), _call())
    toolkit = _Toolkit([{"step": 0}] * 4)
    planner = _with_shared_runtime(
        ModelApiUtilsPlanner(
            client=client,
            model="default",
            max_proposal_repairs=1,
        ),
        guard=ToolLoopGuard(
            max_identical_errors=3,
            max_consecutive_errors=10,
            max_identical_outcomes=4,
        ),
    )

    result = planner.solve(
        system_prompt="", user_message="task", toolkit=toolkit, max_turns=5
    )

    assert result.error == "Planner proposal repair budget exhausted"
    assert result.stats["tool_execution_errors"] == 0
    assert result.stats["loop_feedbacks"] == 1
    assert result.stats["proposal_repairs"] == 2
    assert result.stats["tool_calls"] == 2


def test_planner_repairs_third_unchanged_call_before_execution() -> None:
    client = _Client(
        _call('{"step":0}', call_id="call-1"),
        _call('{"step":0}', call_id="call-2"),
        _call('{"step":0}', call_id="call-3"),
        _call('{"step":1}', call_id="call-4"),
    )
    toolkit = _Toolkit(
        [{"step": 0}, {"step": 0}, {"_finish": True, "status": "success"}]
    )
    planner = _with_shared_runtime(ModelApiUtilsPlanner(client=client, model="default"))

    result = planner.solve(
        system_prompt="", user_message="task", toolkit=toolkit, max_turns=4
    )

    assert result.error is None
    assert toolkit.calls == [
        ("observe", {"step": 0}),
        ("observe", {"step": 0}),
        ("observe", {"step": 1}),
    ]
    assert result.stats["proposal_repairs"] == 1
    repair = next(
        message["content"]
        for message in result.messages
        if message["role"] == "repair_feedback"
    )
    assert repair["code"] == "repeated_tool_call_no_progress"


def test_planner_repairs_third_identical_action_despite_state_changes() -> None:
    class ActionToolkit(_Toolkit):
        def get_tools_spec(self):
            return [
                {
                    "name": "move",
                    "description": "Move to one target.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"target": {"type": "string"}},
                        "required": ["target"],
                    },
                }
            ]

        def is_tool_multi_call_safe(self, name):
            return False

    def action_result(step: int) -> dict[str, object]:
        return {
            "step": step,
            "agent_elapsed_s": 0.1,
            "state": {"position": [float(step), 0.0, 0.0]},
            "planner_progress": {
                "schema": "rpent.planner_progress.v1",
                "source_tool": "move",
                "outcome": "advanced",
                "native_success": False,
            },
        }

    client = _Client(
        _call('{"target":"a"}', call_id="call-1", name="move"),
        _call('{"target":"a"}', call_id="call-2", name="move"),
        _call('{"target":"a"}', call_id="call-3", name="move"),
        _call('{"target":"b"}', call_id="call-4", name="move"),
    )
    toolkit = ActionToolkit(
        [
            action_result(1),
            action_result(2),
            {"_finish": True, "status": "success"},
        ]
    )
    planner = _with_shared_runtime(ModelApiUtilsPlanner(client=client, model="default"))

    result = planner.solve(
        system_prompt="", user_message="task", toolkit=toolkit, max_turns=4
    )

    assert result.error is None
    assert toolkit.calls == [
        ("move", {"target": "a"}),
        ("move", {"target": "a"}),
        ("move", {"target": "b"}),
    ]
    assert result.stats["proposal_repairs"] == 1
    assert result.stats["max_consecutive_identical_actions_seen"] == 2
    repair = next(
        message["content"]
        for message in result.messages
        if message["role"] == "repair_feedback"
    )
    assert repair["code"] == "repeated_action_without_replan"


def test_planner_does_not_collapse_distinct_visual_outcomes() -> None:
    client = _Client(_call(), _call(), _call(), _call())
    toolkit = _Toolkit(
        [{"step": 0, "_image_bytes": value} for value in (b"one", b"two", b"three")]
        + [{"_finish": True, "status": "success", "_image_bytes": b"four"}]
    )
    planner = _with_shared_runtime(
        ModelApiUtilsPlanner(client=client, model="default"),
        guard=ToolLoopGuard(max_identical_outcomes=2),
    )

    result = planner.solve(
        system_prompt="", user_message="task", toolkit=toolkit, max_turns=4
    )

    assert result.error is None
    assert result.stats["loop_feedbacks"] == 0


def test_planner_ignores_segment_artifact_counters_when_detecting_repeats() -> None:
    client = _Client(
        _call('{"step":0}', call_id="call-1"),
        _call('{"step":0}', call_id="call-2"),
        _call('{"step":0}', call_id="call-3"),
        _call('{"step":1}', call_id="call-4"),
    )
    toolkit = _Toolkit(
        [
            {
                "step": 0,
                "found": True,
                "score": 0.9,
                "world_xyz": [0.1, 0.2, 0.3],
                "segment_artifact": "segment_00.json",
                "overlay_artifact": "segment_overlay_00.png",
            },
            {
                "step": 0,
                "found": True,
                "score": 0.9,
                "world_xyz": [0.1, 0.2, 0.3],
                "segment_artifact": "segment_01.json",
                "overlay_artifact": "segment_overlay_01.png",
            },
            {"_finish": True, "status": "success"},
        ]
    )
    planner = _with_shared_runtime(ModelApiUtilsPlanner(client=client, model="default"))

    result = planner.solve(
        system_prompt="", user_message="task", toolkit=toolkit, max_turns=4
    )

    assert result.error is None
    assert toolkit.calls == [
        ("observe", {"step": 0}),
        ("observe", {"step": 0}),
        ("observe", {"step": 1}),
    ]
    assert result.stats["proposal_repairs"] == 1
    assert any(
        message.get("role") == "repair_feedback"
        and message["content"].get("code") == "repeated_tool_call_no_progress"
        for message in result.messages
    )


def test_planner_stops_release_noops_despite_image_drift() -> None:
    client = _Client(
        _call(name="release"),
        _call('{"max_steps":20}', name="release"),
        _call(name="release"),
    )
    results = []
    state = {"position": [0.1, 0.2, 0.3], "gripper_opening": 0.0799}
    for index, opening in enumerate((0.0792, 0.0797, 0.0799)):
        results.append(
            {
                "step": index + 1,
                "agent_elapsed_s": 0.0,
                "state": state,
                "command": {"action": "release"},
                "result": {
                    "name": "release",
                    "start_gripper_opening": opening,
                    "final_gripper_opening": 0.0799,
                    "terminated": False,
                    "truncated": False,
                },
                "_image_bytes": f"frame-{index}".encode(),
            }
        )
    planner = _with_shared_runtime(
        ModelApiUtilsPlanner(client=client, model="default"),
        guard=ToolLoopGuard(max_identical_outcomes=3, max_stagnant_actions=2),
    )

    result = planner.solve(
        system_prompt="", user_message="task", toolkit=_Toolkit(results), max_turns=4
    )

    assert result.error == "tool loop guard stopped after 2 state-unchanged actions"
    assert result.stats["loop_feedbacks"] == 2
    assert result.stats["semantic_progress_feedbacks"] == 2


def test_planner_feedbacks_after_many_distinct_observations() -> None:
    client = _Client(_call(), _call(), _call(), _call())
    planner = _with_shared_runtime(
        ModelApiUtilsPlanner(client=client, model="default"),
        guard=ToolLoopGuard(max_observations_without_action=3),
    )

    result = planner.solve(
        system_prompt="",
        user_message="task",
        toolkit=_Toolkit(
            [
                {"step": 0, "value": 1},
                {"step": 0, "value": 2},
                {"step": 0, "value": 3},
                {"_finish": True, "status": "success"},
            ]
        ),
        max_turns=4,
    )

    assert result.error is None
    assert result.stats["max_observations_without_action"] == 3
    assert result.stats["semantic_progress_feedbacks"] == 1
    assert any(
        message.get("role") == "tool"
        and "observations_without_action" in str(message.get("content"))
        for message in result.messages
    )
    assert any(
        "rpent.semantic_progress_feedback.v1" in str(message.get("content"))
        for message in client.calls[-1]["messages"]
    )


def test_planner_wall_clock_timeout_cancels_active_tool() -> None:
    toolkit = _BlockingToolkit([{"step": 0}])
    planner = ModelApiUtilsPlanner(
        client=_Client(_call()), model="default", timeout_s=0.05
    )
    started = time.monotonic()

    result = planner.solve(
        system_prompt="",
        user_message="task",
        toolkit=toolkit,
        max_turns=2,
    )

    assert result.error == "vLLM planner timed out after 0.05s"
    assert result.stats["tool_calls"] == 1
    assert toolkit.cancelled.is_set()
    assert time.monotonic() - started < 0.5


def test_loop_guard_can_disable_enforcement_without_hiding_telemetry() -> None:
    guard = tool_loop_guard_from_env({"RPENT_VLLM_LOOP_GUARD_MODE": "disabled"})

    assert guard.observe(tool="observe", arguments={}, result={"error": "bad"}) == (
        None,
        None,
    )
    assert guard.total_errors == 1


def test_loop_guard_progress_thresholds_load_from_environment() -> None:
    guard = tool_loop_guard_from_env(
        {
            "RPENT_VLLM_MAX_OBSERVATIONS_WITHOUT_ACTION": "9",
            "RPENT_VLLM_MAX_STAGNANT_ACTIONS": "3",
            "RPENT_VLLM_MAX_EQUIVALENT_CAPPED_ACTIONS": "6",
            "RPENT_VLLM_MAX_REPEATED_FAILURE_FAMILIES": "5",
            "RPENT_VLLM_STATE_TOLERANCE": "0.01",
        }
    )

    assert guard.max_observations_without_action == 9
    assert guard.max_stagnant_actions == 3
    assert guard.max_equivalent_capped_actions == 6
    assert guard.max_repeated_failure_families == 5
    assert guard.state_tolerance == 0.01


def test_loop_guard_loads_consecutive_error_limit_with_legacy_fallback() -> None:
    configured = tool_loop_guard_from_env(
        {"RPENT_VLLM_MAX_CONSECUTIVE_TOOL_ERRORS": "7"}
    )
    legacy = tool_loop_guard_from_env({"RPENT_VLLM_MAX_TOTAL_TOOL_ERRORS": "8"})

    assert configured.max_consecutive_errors == 7
    assert legacy.max_consecutive_errors == 8


def test_loop_guard_error_budget_resets_after_successful_tool_progress() -> None:
    guard = ToolLoopGuard(max_consecutive_errors=2)

    first_feedback, first_error = guard.observe(
        tool="segment",
        arguments={"prompt": "first"},
        result={"error": "first failure", "code": "invalid_first"},
    )
    assert first_feedback is not None
    assert first_error is None
    assert guard.total_errors == 1
    assert guard.consecutive_errors == 1

    assert guard.observe(
        tool="view_env_state",
        arguments={"step": 1},
        result={"step": 1, "state": {"robot0_eef_pos": [0.0, 0.0, 1.0]}},
    ) == (None, None)
    assert guard.consecutive_errors == 0

    _, second_error = guard.observe(
        tool="move_to",
        arguments={"xyz": [0.1, 0.0, 1.0]},
        result={"error": "second failure", "code": "invalid_second"},
    )
    _, third_error = guard.observe(
        tool="move_to",
        arguments={"xyz": [0.2, 0.0, 1.0]},
        result={"error": "third failure", "code": "invalid_third"},
    )

    assert second_error is None
    assert third_error == (
        "tool loop guard stopped after 2 consecutive execution errors"
    )
    assert guard.total_errors == 3
    assert guard.consecutive_errors == 2


def test_disabled_loop_guard_does_not_hide_execution_error_stats() -> None:
    client = _Client(_call(), _call())
    planner = _with_shared_runtime(
        ModelApiUtilsPlanner(client=client, model="default"),
        guard=ToolLoopGuard(enabled=False),
    )

    result = planner.solve(
        system_prompt="",
        user_message="task",
        toolkit=_Toolkit([{"error": "bad"}, {"_finish": True}]),
        max_turns=2,
    )

    assert result.error is None
    assert result.stats["tool_execution_errors"] == 1
    assert result.stats["loop_feedbacks"] == 0


def test_loop_guard_counts_identical_outcomes_consecutively() -> None:
    guard = ToolLoopGuard(max_identical_outcomes=2)

    assert guard.observe(tool="observe", arguments={}, result={"step": 0}) == (
        None,
        None,
    )
    assert guard.observe(tool="observe", arguments={}, result={"step": 1}) == (
        None,
        None,
    )
    assert guard.observe(tool="observe", arguments={}, result={"step": 0}) == (
        None,
        None,
    )


def test_loop_guard_tracks_repeated_calls_inside_interleaved_batches() -> None:
    guard = ToolLoopGuard(max_identical_outcomes=2)

    assert guard.observe(tool="camera", arguments={"id": 1}, result={"step": 0}) == (
        None,
        None,
    )
    assert guard.observe(tool="camera", arguments={"id": 2}, result={"step": 0}) == (
        None,
        None,
    )
    feedback, error = guard.observe(
        tool="camera", arguments={"id": 1}, result={"step": 0}
    )

    assert feedback["identical_outcome_count"] == 2
    assert error == "tool loop guard stopped 2 identical outcomes for camera"


def test_loop_guard_preserves_owner_supplied_failure_recovery() -> None:
    guard = ToolLoopGuard(max_identical_errors=3)
    arguments = {
        "prompt": "the held object",
        "camera": "wrist",
        "step": 3,
        "point": None,
        "min_score": 0.2,
    }
    first_result = {
        "found": False,
        "step": 3,
        "camera": "wrist",
        "score": 0.136,
        "box": [970.0, 416.0, 1028.0, 543.0],
        "segment_artifact": "segment_00.json",
        "error": "top score 0.136 is below min_score 0.200",
        "code": "segment_below_min_score",
        "instruction": "A successful segment is still required.",
        "suggested_call": {
            "tool": "segment",
            "arguments": {
                "prompt": None,
                "camera": "wrist",
                "step": 3,
                "point": [480, 999],
                "min_score": 0.2,
            },
            "precondition": "the candidate box visually matches the named target",
        },
    }

    feedback, error = guard.observe(
        tool="segment", arguments=arguments, result=first_result
    )

    assert error is None
    assert feedback["identical_failure_count"] == 1
    assert feedback["suggested_call"] == {
        "tool": "segment",
        "arguments": {
            "prompt": None,
            "camera": "wrist",
            "step": 3,
            "point": [480, 999],
            "min_score": 0.2,
        },
        "precondition": "the candidate box visually matches the named target",
    }
    assert "successful segment is still required" in feedback["instruction"]
    proposal_feedback = guard.validate_proposal([("retry", "segment", arguments)])
    assert proposal_feedback["code"] == "repeated_failed_tool_call"
    assert proposal_feedback["suggested_call"] == feedback["suggested_call"]

    second_result = dict(first_result, segment_artifact="segment_01.json")
    repeated, repeated_error = guard.observe(
        tool="segment", arguments=arguments, result=second_result
    )
    assert repeated["identical_failure_count"] == 2
    assert repeated_error is None


def test_loop_guard_allows_a_failed_call_after_environment_action() -> None:
    guard = ToolLoopGuard(max_identical_errors=2)
    failed_arguments = {"xyz": [0.1, 0.2, 0.3]}
    first_feedback, first_error = guard.observe(
        tool="move_to",
        arguments=failed_arguments,
        result={"error": "blocked", "code": "needs_new_state"},
    )
    assert first_feedback["identical_failure_count"] == 1
    assert first_error is None
    assert guard.validate_proposal([("retry", "move_to", failed_arguments)]) is not None

    guard.observe(
        tool="set_gripper",
        arguments={"gripper": 1},
        result={
            "step": 1,
            "agent_elapsed_s": 0.1,
            "state": {
                "robot0_eef_pos": [0.0, 0.0, 0.2],
                "robot0_gripper_qpos": [0.03, -0.03],
            },
        },
    )

    assert guard.validate_proposal([("retry", "move_to", failed_arguments)]) is None
    retried_feedback, retried_error = guard.observe(
        tool="move_to",
        arguments=failed_arguments,
        result={"error": "blocked", "code": "needs_new_state"},
    )
    assert retried_feedback["identical_failure_count"] == 1
    assert retried_error is None


def test_loop_guard_keeps_bounded_retry_for_transient_failure() -> None:
    guard = ToolLoopGuard()
    arguments = {"prompt": "object"}

    guard.observe(
        tool="segment",
        arguments=arguments,
        result={
            "error": "segmentation service call failed: connection reset",
            "fallback": "Use manual visual localization and back_project.",
        },
    )

    assert guard.validate_proposal([("retry", "segment", arguments)]) is None


def test_loop_guard_groups_changed_arguments_by_failure_family() -> None:
    guard = ToolLoopGuard(max_repeated_failure_families=2)
    result = {
        "error": "transition review required",
        "code": "destination_segment_clipped",
        "instruction": "obtain a complete destination observation",
    }

    first, first_error = guard.observe(
        tool="plan_placement",
        arguments={"destination_point": [100, 100]},
        result=result,
    )
    assert first_error is None
    assert first is not None
    assert first.get("code") is None

    second, second_error = guard.observe(
        tool="plan_placement",
        arguments={"destination_point": [200, 200]},
        result=result,
    )
    assert second_error is None
    assert second["code"] == "repeated_failure_family"
    assert second["failure_family_count"] == 2
    assert "materially different action" in second["instruction"]

    third, third_error = guard.observe(
        tool="plan_placement",
        arguments={"destination_point": [300, 300]},
        result=result,
    )
    assert third["code"] == "repeated_failure_family"
    assert third["failure_family_count"] == 3
    assert third_error == (
        "tool loop guard stopped after 3 repeated failures in one semantic "
        "failure family for plan_placement"
    )


def test_loop_guard_counts_one_failure_family_attempt_per_parallel_scope() -> None:
    guard = ToolLoopGuard(max_repeated_failure_families=2)
    failure = {"error": "reading this memory path is denied"}

    for path in ("guide-a.md", "guide-b.md", "guide-c.md"):
        feedback, error = guard.observe(
            tool="read_text_file",
            arguments={"path": path},
            result=failure,
            failure_scope=1,
        )
        assert error is None
        assert feedback is not None
        assert feedback.get("failure_family_count", 1) == 1

    second, second_error = guard.observe(
        tool="read_text_file",
        arguments={"path": "guide-d.md"},
        result=failure,
        failure_scope=2,
    )
    assert second_error is None
    assert second["failure_family_count"] == 2

    third, third_error = guard.observe(
        tool="read_text_file",
        arguments={"path": "guide-e.md"},
        result=failure,
        failure_scope=3,
    )
    assert third["failure_family_count"] == 3
    assert third_error == (
        "tool loop guard stopped after 3 repeated failures in one semantic "
        "failure family for read_text_file"
    )


def test_loop_guard_clears_failure_family_after_success_or_reset() -> None:
    guard = ToolLoopGuard(max_repeated_failure_families=2)
    failure = {
        "error": "transition review required",
        "code": "destination_segment_clipped",
    }
    guard.observe(tool="plan_placement", arguments={"point": [1, 1]}, result=failure)
    guard.observe(tool="plan_placement", arguments={"point": [2, 2]}, result=failure)

    assert guard.observe(
        tool="move_to",
        arguments={"xyz": [0.1, 0.0, 1.0]},
        result={
            "step": 1,
            "agent_elapsed_s": 0.1,
            "state": {"robot0_eef_pos": [0.1, 0.0, 1.0]},
        },
    ) == (None, None)

    feedback, error = guard.observe(
        tool="plan_placement",
        arguments={"point": [3, 3]},
        result=failure,
    )
    assert error == (
        "tool loop guard stopped after 3 repeated failures in one semantic "
        "failure family for plan_placement"
    )

    guard.observe(
        tool="reset",
        arguments={},
        result={
            "step": 0,
            "command": {"action": "reset"},
            "state": {"robot0_eef_pos": [0.0, 0.0, 1.0]},
        },
    )
    feedback, error = guard.observe(
        tool="plan_placement",
        arguments={"point": [4, 4]},
        result=failure,
    )
    assert error is None
    assert feedback.get("code") is None

    successful_tool = ToolLoopGuard(max_repeated_failure_families=2)
    successful_tool.observe(
        tool="plan_placement", arguments={"point": [1, 1]}, result=failure
    )
    successful_tool.observe(
        tool="plan_placement", arguments={"point": [2, 2]}, result=failure
    )
    successful_tool.observe(
        tool="plan_placement",
        arguments={"point": [3, 3]},
        result={"fits_with_clearance": True},
    )
    feedback, error = successful_tool.observe(
        tool="plan_placement",
        arguments={"point": [4, 4]},
        result=failure,
    )
    assert error is None
    assert feedback.get("code") is None


def test_loop_guard_tracks_state_unchanged_actions_across_tool_names() -> None:
    guard = ToolLoopGuard(max_stagnant_actions=2)
    state = {"robot0_eef_pos": [0.1, 0.2, 0.9], "robot0_gripper_qpos": [0.04]}

    assert guard.observe(
        tool="view_env_state", arguments={}, result={"step": 0, "state": state}
    ) == (None, None)
    first_feedback, first_error = guard.observe(
        tool="move_to",
        arguments={"xyz": [0.2, 0.2, 0.9]},
        result={"step": 1, "agent_elapsed_s": 0.1, "state": state},
    )
    assert first_feedback["unchanged_action_count"] == 1
    assert first_error is None
    repeated = guard.validate_proposal([("retry", "move_to", {"xyz": [0.2, 0.2, 0.9]})])
    assert repeated["code"] == "repeated_failed_tool_call"
    assert "did not measurably change" in repeated["expected"]

    second_feedback, second_error = guard.observe(
        tool="set_gripper",
        arguments={"gripper": 1},
        result={"step": 2, "agent_elapsed_s": 0.1, "state": state},
    )
    assert second_feedback["unchanged_action_count"] == 2
    assert second_error == "tool loop guard stopped after 2 state-unchanged actions"


def test_loop_guard_bounds_runtime_equivalent_capped_actions() -> None:
    guard = ToolLoopGuard(max_equivalent_capped_actions=3)

    def result(step: int, requested_prompt: str, action: str) -> dict[str, object]:
        del requested_prompt
        return {
            "step": step,
            "agent_elapsed_s": 1.0,
            "state": {"robot0_eef_pos": [step / 10, 0.0, 1.0]},
            "success": False,
            "planner_progress": {
                "schema": "rpent.planner_progress.v1",
                "source_tool": action,
                "outcome": "budget_exhausted",
                "native_success": False,
                "action_family": "open the cabinet door.",
            },
        }

    assert guard.observe(
        tool="rldx_skill",
        arguments={"prompt": "open cabinet"},
        result=result(1, "open cabinet", "rldx_skill"),
    ) == (None, None)
    feedback, error = guard.observe(
        tool="rldx_arm",
        arguments={"prompt": "pull the red handle"},
        result=result(2, "pull the red handle", "rldx_arm"),
    )
    assert error is None
    assert feedback["code"] == "equivalent_capped_action"
    assert feedback["equivalent_capped_action_count"] == 2
    assert "rewording the same family" in feedback["instruction"].lower()

    feedback, error = guard.observe(
        tool="rldx_skill",
        arguments={"prompt": "open it wider"},
        result=result(3, "open it wider", "rldx_skill"),
    )
    assert feedback["equivalent_capped_action_count"] == 3
    assert error == (
        "tool loop guard stopped after 3 capped executions for one equivalent "
        "learned action"
    )


def test_loop_guard_does_not_count_successful_or_uncapped_actions() -> None:
    base = {
        "step": 1,
        "agent_elapsed_s": 1.0,
        "state": {"robot0_eef_pos": [0.1, 0.0, 1.0]},
        "planner_progress": {
            "schema": "rpent.planner_progress.v1",
            "source_tool": "rldx_skill",
            "outcome": "advanced",
            "native_success": False,
            "action_family": "open the cabinet door.",
        },
    }

    assert ToolLoopGuard(max_equivalent_capped_actions=2).observe(
        tool="rldx_skill",
        arguments={},
        result={**base, "success": False},
    ) == (None, None)
    assert ToolLoopGuard(max_equivalent_capped_actions=2).observe(
        tool="rldx_skill",
        arguments={},
        result={
            **base,
            "success": True,
            "planner_progress": {
                "schema": "rpent.planner_progress.v1",
                "source_tool": "rldx_skill",
                "outcome": "native_success",
                "native_success": True,
                "action_family": "open the cabinet door.",
            },
        },
    ) == (None, None)


def test_loop_guard_does_not_recount_capped_action_replayed_by_observation() -> None:
    guard = ToolLoopGuard(max_equivalent_capped_actions=2)
    result = {
        "step": 1,
        "state": {"robot0_eef_pos": [0.1, 0.0, 1.0]},
        "success": False,
        "planner_progress": {
            "schema": "rpent.planner_progress.v1",
            "source_tool": "rldx_skill",
            "outcome": "budget_exhausted",
            "native_success": False,
            "action_family": "open the cabinet door.",
        },
    }

    assert guard.observe(tool="rldx_skill", arguments={}, result=result) == (
        None,
        None,
    )
    assert guard.observe(tool="view_env_state", arguments={}, result=result) == (
        None,
        None,
    )
    assert guard.stats()["max_equivalent_capped_actions_seen"] == 1


def test_loop_guard_preserves_owner_recovery_for_stagnant_action() -> None:
    guard = ToolLoopGuard()
    closed_state = {
        "robot0_eef_pos": [-0.18, 0.25, 1.06],
        "robot0_gripper_qpos": [0.004, -0.004],
    }
    guard.observe(
        tool="view_env_state",
        arguments={},
        result={"step": 4, "state": closed_state},
    )
    arguments = {
        "xyz": [-0.18, 0.25, 1.06],
        "gripper": -1,
        "max_steps": 120,
    }
    feedback, error = guard.observe(
        tool="move_to",
        arguments=arguments,
        result={
            "step": 5,
            "agent_elapsed_s": 0.0,
            "state": closed_state,
            "planner_progress": {
                "schema": "rpent.planner_progress.v1",
                "source_tool": "move_to",
                "outcome": "unchanged",
                "native_success": False,
                "instruction": (
                    "Use the dedicated set_gripper tool to open it; do not repeat "
                    "this move call."
                ),
                "suggested_call": {
                    "tool": "set_gripper",
                    "arguments": {"gripper": -1, "steps": 10},
                },
            },
        },
    )

    assert error is None
    assert "dedicated set_gripper" in feedback["instruction"]
    assert feedback["suggested_call"] == {
        "tool": "set_gripper",
        "arguments": {"gripper": -1, "steps": 10},
    }
    repeated = guard.validate_proposal([("retry", "move_to", arguments)])
    assert repeated["code"] == "repeated_failed_tool_call"
    assert repeated["suggested_call"] == feedback["suggested_call"]


def test_loop_guard_prefers_state_over_action_progress_description() -> None:
    guard = ToolLoopGuard(max_stagnant_actions=2)
    state = {"robot0_eef_pos": [0.1, 0.2, 0.9], "robot0_gripper_qpos": [0.04]}

    guard.observe(
        tool="view_env_state",
        arguments={},
        result={"step": 0, "terminated": False, "state": state},
    )
    feedback, error = guard.observe(
        tool="move_to",
        arguments={"xyz": [0.2, 0.2, 0.9]},
        result={
            "step": 1,
            "terminated": False,
            "agent_elapsed_s": 0.1,
            "state": state,
            "planner_progress": {
                "schema": "rpent.planner_progress.v1",
                "source_tool": "move_to",
                "outcome": "unchanged",
                "native_success": False,
                "instruction": "latest-action-specific description",
            },
        },
    )

    assert feedback["unchanged_action_count"] == 1
    assert error is None


def test_loop_guard_resets_stagnation_after_state_change() -> None:
    guard = ToolLoopGuard(max_stagnant_actions=2)

    guard.observe(
        tool="view_env_state",
        arguments={},
        result={"step": 0, "state": {"robot0_eef_pos": [0.0, 0.0, 0.0]}},
    )
    feedback, error = guard.observe(
        tool="move_to",
        arguments={"xyz": [0.0, 0.0, 0.0]},
        result={
            "step": 1,
            "agent_elapsed_s": 0.1,
            "state": {"robot0_eef_pos": [0.001, 0.0, 0.0]},
        },
    )
    assert feedback["unchanged_action_count"] == 1
    assert error is None

    assert guard.observe(
        tool="move_to",
        arguments={"xyz": [0.1, 0.0, 0.0]},
        result={
            "step": 2,
            "agent_elapsed_s": 0.1,
            "state": {"robot0_eef_pos": [0.1, 0.0, 0.0]},
        },
    ) == (None, None)
    assert guard.stats()["stagnant_actions"] == 0
    assert guard.stats()["state_changes"] == 1


def test_incomplete_usage_does_not_publish_an_exact_request_measurement() -> None:
    sink = _RecordingDashboardEvents()
    planner = ModelApiUtilsPlanner(
        client=_Client(
            _call(),
            usage={"prompt_tokens": 3, "total_tokens": 3},
        ),
        model="default",
        dashboard_events=sink,
    )

    result = planner.solve(
        system_prompt="",
        user_message="task",
        toolkit=_Toolkit([{"_finish": True}]),
        max_turns=1,
    )

    assert result.error is None
    assert not any(
        isinstance(event, PlannerRequestUsageEvent) for event in sink.events
    )
    usage_events = [event for event in sink.events if isinstance(event, UsageEvent)]
    assert usage_events[-1].inp == 3
    assert usage_events[-1].out == 0


@pytest.mark.parametrize("thinking", [None, False, True])
def test_repair_responses_publish_request_context_and_fresh_cumulative_usage(
    thinking: bool | None,
) -> None:
    client = _Client(
        _call("not-json"),
        _call("still-not-json"),
        usage={
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 1},
            "completion_tokens_details": {"reasoning_tokens": 1},
        },
    )
    sink = _RecordingDashboardEvents()
    planner = ModelApiUtilsPlanner(
        client=client,
        model="default",
        max_proposal_repairs=1,
        dashboard_events=sink,
        **({"enable_thinking": thinking} if thinking is not None else {}),
    )

    result = planner.solve(
        system_prompt="",
        user_message="task",
        toolkit=_Toolkit([]),
        max_turns=3,
    )

    assert result.error == "Planner proposal repair budget exhausted"
    request_events = [
        event for event in sink.events if isinstance(event, PlannerRequestUsageEvent)
    ]
    assert [
        (event.turn, event.input_tokens, event.output_tokens)
        for event in request_events
    ] == [(1, 3, 2), (2, 3, 2)]
    assert [event.request_index for event in request_events] == [1, 2]
    assert request_events[0].model == "default"
    assert request_events[0].temperature == 0.0
    assert request_events[0].max_tokens == 8192
    expected_thinking = False if thinking is None else thinking
    assert [event.enable_thinking for event in request_events] == [
        expected_thinking,
        expected_thinking,
    ]
    assert all(
        event.enable_thinking is call["extra_body"]["enable_thinking"]
        for event, call in zip(request_events, client.calls, strict=True)
    )
    assert request_events[0].parallel_tool_calls is True
    assert request_events[0].tool_choice == "required"
    assert request_events[0].message_count == len(client.calls[0]["messages"]) == 2
    assert request_events[0].message_chars == planner_module._chat_messages_chars(
        client.calls[0]["messages"]
    )
    assert request_events[0].tool_count == 1
    assert request_events[0].user_message_chars == 4
    assert request_events[0].assistant_message_chars == 0
    assert request_events[0].tool_result_chars == 0
    assert request_events[0].image_count == 0
    assert request_events[0].tool_schema_bytes == len(
        json.dumps(
            [planner_module._openai_tool(spec) for spec in _Toolkit([]).get_tools_spec()], ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    )
    assert request_events[0].system_message_chars == len(
        client.calls[0]["messages"][0]["content"]
    )
    assert request_events[0].system_prompt == client.calls[0]["messages"][0]["content"]
    assert request_events[0].elapsed_s is not None
    assert request_events[0].request_text_bytes > 0
    assert request_events[0].request_image_bytes == 0
    assert request_events[0].request_wire_bytes >= request_events[0].request_text_bytes
    assert request_events[0].compacted is False
    assert request_events[0].context_overflow_retry is False
    assert request_events[0].cached_input_tokens == 1
    assert request_events[0].reasoning_output_tokens == 1
    usage_events = [event for event in sink.events if isinstance(event, UsageEvent)]
    assert usage_events[-1].inp == 6
    assert usage_events[-1].out == 4


def test_context_parts_count_text_without_encoded_image_payloads():
    measured = planner_module._request_context_parts(
        [
            {"role": "system", "content": "rules"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "你好"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,SECRET"},
                    },
                ],
            },
            {
                "role": "assistant",
                "content": "reply",
                "tool_calls": [{"raw": "SECRET"}],
            },
            {"role": "tool", "content": "result"},
        ],
        [],
    )
    assert measured == {
        "system_message_chars": 5,
        "user_message_chars": 2,
        "assistant_message_chars": 5,
        "tool_result_chars": 6,
        "image_count": 1,
        "tool_schema_bytes": 2,
        "system_prompt": "rules",
        "context_previews": {
            "user": {"text": "你好"}, "assistant": {"text": "reply"},
            "tool": {"text": "result"}, "tools": {"text": "[]"},
        },
    }
    assert "SECRET" not in repr(measured)
