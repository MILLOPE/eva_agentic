# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Small OpenAI-compatible tool loop used by the signed vLLM planner.

The adapter keeps the official RPent planner contract intact while avoiding a
bearer-token/OpenAI SDK path that the deployed Chat API does not support.  Its
default execution surface is the caller's Toolkit; an optional read-only
workspace module can be composed without creating another Toolkit owner.
"""

from __future__ import annotations

import copy
import json
import logging
import threading
import time
from collections.abc import Mapping
from typing import Any

from .compat import (
    CapabilityCatalogEvent,
    DashboardEventSink,
    NullDashboardEventSink,
    PlannerRequestUsageEvent,
    PlannerSystemContextEvent,
    TranscriptEvent,
    UsageEvent,
)
from rpent.dashboard.interaction import DashboardInteractionPort, DashboardMessage
from rpent.planner.base import PlannerResult
from .compat import DEFAULT_REASONING_EFFORT
from .result_projection import (
    ResultProjectionError,
    bounded_tool_result_text,
    executed_result_unavailable,
)
from rpent.tools.toolkit import Toolkit, ToolResult

from .client import (
    _build_chat_payload,
    _chat_messages_chars,
    _chat_payload_inline_image_bytes,
    _chat_payload_text_bytes,
    _chat_payload_wire_bytes,
)
from .compat import adapt_dashboard_events
from .history import RequestBudgetExceeded, ToolHistoryPolicy
from .multi_tool import MultiToolPolicy
from .workspace import WorkspaceProfile, compose_tool_surface

logger = logging.getLogger(__name__)

_WORKER_CANCEL_GRACE_S = 15.0

_DASHBOARD_PUBLIC_PROGRESS_INSTRUCTION = (
    "For this Dashboard session, include one concise user-facing progress update "
    "in assistant content whenever you call tools. In one or two sentences and "
    "in the user's language, say what you are checking or doing next. Keep the "
    "required tool calls unchanged. Do not put private chain-of-thought in the "
    "public update."
)


class _VllmDashboardConversation:
    """Bridge one synchronous vLLM tool loop to the Dashboard Session owner."""

    def __init__(
        self,
        *,
        interaction: DashboardInteractionPort,
        toolkit: Toolkit,
        dashboard_events: DashboardEventSink,
    ) -> None:
        self._interaction = interaction
        self._toolkit = toolkit
        self._dashboard_events = dashboard_events
        self._claimed: list[DashboardMessage] = []
        self._lifecycle_lock = threading.RLock()
        self._done = threading.Event()
        self._stop_requested = threading.Event()
        self._cancel_done = threading.Event()
        self._stop_observed = threading.Event()
        self._stop_kind: str | None = None
        self._cancel_error: str | None = None
        self._monitor: threading.Thread | None = None
        self._started = False

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested.is_set()

    def start(self) -> None:
        """Expose input once the local tool surface is ready for model work."""
        with self._lifecycle_lock:
            if self._started or self._done.is_set():
                return
            self._started = True
            self._interaction.set_planner_activity(
                "busy",
                accepting_input=not self._interaction.task_replacement_requested,
            )
            self._dashboard_events.emit(TranscriptEvent({"type": "initial_prompt"}))
            self._monitor = threading.Thread(
                target=self._monitor_commands,
                name="qwen-dashboard-control",
                daemon=True,
            )
            self._monitor.start()

    def claim_messages(self, request_messages: list[dict[str, Any]]) -> int:
        """Move all currently queued messages into the next model request."""
        with self._lifecycle_lock:
            if self._done.is_set() or self._stop_requested.is_set():
                return 0
            claimed = 0
            message = self._interaction.claim_next_pending_message()
            while message is not None:
                request_messages.append({"role": "user", "content": message.text})
                self._claimed.append(message)
                claimed += 1
                message = self._interaction.claim_next_pending_message()
            return claimed

    def acknowledge_request(self, transcript: list[dict[str, Any]]) -> None:
        """Commit messages only after their containing model request succeeds."""
        with self._lifecycle_lock:
            claimed, self._claimed = self._claimed, []
            if self._done.is_set():
                return
            for message in claimed:
                self._interaction.mark_message_sent(message.message_id)
                transcript.append({"role": "user", "content": message.text})
                self._dashboard_events.emit(
                    TranscriptEvent({"type": "user", "text": message.text})
                )

    def fail_request(self, error: BaseException) -> None:
        """Report messages whose containing model request was rejected."""
        detail = f"{type(error).__name__}: {error}"
        with self._lifecycle_lock:
            claimed, self._claimed = self._claimed, []
            if self._done.is_set():
                return
            for message in claimed:
                self._interaction.mark_message_failed(message.message_id, detail)

    def stopped_result(
        self,
        *,
        transcript: list[dict[str, Any]],
        stats: dict[str, object],
    ) -> PlannerResult:
        """Finish a claimed interrupt or task replacement at a safe boundary."""
        self._stop_observed.set()
        self._cancel_done.wait()
        if self._stop_kind == "replacement":
            detail = "vLLM planner replaced by a newer Dashboard task"
        else:
            detail = "vLLM planner interrupted from the Dashboard"
        if self._cancel_error is not None:
            detail = f"{detail}: {self._cancel_error}"
        with self._lifecycle_lock:
            if not self._done.is_set():
                if self._stop_kind == "replacement":
                    self._interaction.complete_task_replacement(
                        error=self._cancel_error
                    )
                else:
                    self._interaction.complete_interrupt(error=self._cancel_error)
        return PlannerResult(messages=transcript, stats=stats, error=detail)

    def end(self) -> None:
        """Seal input and restore any claimed message that was never submitted."""
        with self._lifecycle_lock:
            if self._done.is_set():
                return
            self._done.set()
            self._stop_observed.set()
            self._interaction.seal_interaction()
            monitor = self._monitor
        if monitor is not None and monitor is not threading.current_thread():
            monitor.join(timeout=0.5)

    def _monitor_commands(self) -> None:
        version = self._interaction.interaction_version
        while True:
            with self._lifecycle_lock:
                if self._done.is_set() or self._stop_requested.is_set():
                    return
                replacement_requested = (
                    self._interaction.task_replacement_requested
                )
                interrupt_requested = (
                    False
                    if replacement_requested
                    else self._interaction.claim_interrupt_request()
                )
            if replacement_requested:
                self._request_stop("replacement")
                return
            if interrupt_requested:
                self._request_stop("interrupt")
                return
            version = self._interaction.wait_for_interaction_change(
                version, timeout=0.2
            )

    def _request_stop(self, kind: str) -> None:
        with self._lifecycle_lock:
            if self._done.is_set():
                return
            self._stop_kind = kind
            self._stop_requested.set()
        try:
            while not self._stop_observed.is_set():
                self._toolkit.cancel_active_and_wait()
                self._stop_observed.wait(timeout=0.01)
        except Exception as exc:  # noqa: BLE001 - Dashboard control boundary
            self._cancel_error = f"{type(exc).__name__}: {exc}"
        finally:
            self._cancel_done.set()


def _publish_workspace_catalog(
    dashboard_events: DashboardEventSink,
    tool_surface: Any,
) -> None:
    """Replace the preliminary catalog with the planner's full tool surface."""

    try:
        if dashboard_events.enabled:
            dashboard_events.emit(
                CapabilityCatalogEvent(entries=tool_surface.dashboard_catalog())
            )
    except Exception as error:
        logger.warning("Dashboard workspace catalog publication failed: %s", error)


class _RepairableModelProposalError(ValueError):
    """An invalid model response that can be corrected on the next turn."""

    def __init__(self, feedback: dict[str, object]) -> None:
        super().__init__(str(feedback.get("code", "model_proposal_invalid")))
        self.feedback = feedback


class ModelApiUtilsPlanner:
    """Run a bounded OpenAI-compatible structured-tool agent loop."""

    execution_surface = "in_process_tool_surface"
    supports_visual_input = True

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        enable_thinking: bool = DEFAULT_REASONING_EFFORT != "none",
        max_proposal_repairs: int = 3,
        dashboard_events: DashboardEventSink | None = None,
        no_images: bool = False,
        history_policy: ToolHistoryPolicy | None = None,
        multi_tool_policy: MultiToolPolicy | None = None,
        workspace_profile: WorkspaceProfile | None = None,
        timeout_s: float = 900.0,
    ) -> None:
        if not callable(getattr(client, "chat", None)):
            raise TypeError("client must provide a callable chat method")
        if not isinstance(model, str) or not model:
            raise ValueError("model must be a non-empty string")
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens <= 0
        ):
            raise ValueError("max_tokens must be a positive integer")
        if isinstance(max_proposal_repairs, bool) or not isinstance(
            max_proposal_repairs, int
        ):
            raise TypeError("max_proposal_repairs must be an integer")
        if max_proposal_repairs < 0:
            raise ValueError("max_proposal_repairs must be nonnegative")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = float(temperature)
        self._enable_thinking = bool(enable_thinking)
        self._max_proposal_repairs = max_proposal_repairs
        self._dashboard_events = adapt_dashboard_events(dashboard_events)
        self._no_images = no_images
        self._history_policy = history_policy or ToolHistoryPolicy()
        self._multi_tool_policy = multi_tool_policy or MultiToolPolicy()
        self._workspace_profile = workspace_profile
        self._timeout_s = float(timeout_s)

    def solve(
        self,
        *,
        system_prompt: str,
        user_message: str,
        toolkit: Toolkit,
        max_turns: int,
        input_queue=None,
        dashboard_interaction=None,
    ) -> PlannerResult:
        """Run one Qwen agent turn under one wall-clock deadline."""
        if input_queue is not None and dashboard_interaction is not None:
            raise ValueError(
                "input_queue and dashboard_interaction cannot be used together"
            )
        if (
            isinstance(max_turns, bool)
            or not isinstance(max_turns, int)
            or max_turns <= 0
        ):
            raise ValueError("max_turns must be a positive integer")
        deadline = time.monotonic() + self._timeout_s
        stop_event = threading.Event()
        dashboard_conversation = (
            None
            if dashboard_interaction is None
            else _VllmDashboardConversation(
                interaction=dashboard_interaction,
                toolkit=toolkit,
                dashboard_events=self._dashboard_events,
            )
        )
        state: dict[str, object] = {}

        def run() -> None:
            try:
                state["result"] = self._solve(
                    system_prompt=system_prompt,
                    user_message=user_message,
                    toolkit=toolkit,
                    max_turns=max_turns,
                    input_queue=input_queue,
                    dashboard_conversation=dashboard_conversation,
                    deadline=deadline,
                    stop_event=stop_event,
                )
            except BaseException as exc:  # noqa: BLE001 - cross-thread boundary
                state["error"] = exc

        worker = threading.Thread(target=run, name="qwen-planner", daemon=True)
        worker.start()
        worker.join(timeout=self._timeout_s)
        if worker.is_alive():
            stop_event.set()
            cancel_error: str | None = None
            cancel = getattr(toolkit, "cancel_active_and_wait", None)
            if callable(cancel):
                try:
                    cancel()
                except Exception as exc:  # noqa: BLE001 - report boundary failure
                    cancel_error = f"{type(exc).__name__}: {exc}"
            worker.join(timeout=_WORKER_CANCEL_GRACE_S)
            if worker.is_alive() and dashboard_conversation is not None:
                dashboard_conversation.end()
            result = state.get("result")
            if isinstance(result, PlannerResult):
                if cancel_error is not None:
                    result.stats["timeout_cancel_error"] = cancel_error
                return result
            stats: dict[str, object] = {
                "planner_elapsed_s": round(
                    time.monotonic() - (deadline - self._timeout_s), 6
                )
            }
            if cancel_error is not None:
                stats["timeout_cancel_error"] = cancel_error
            return PlannerResult(
                error=f"vLLM planner timed out after {self._timeout_s:g}s",
                stats=stats,
            )
        if "error" in state:
            exc = state["error"]
            assert isinstance(exc, BaseException)
            return PlannerResult(error=f"{type(exc).__name__}: {exc}")
        result = state.get("result")
        assert isinstance(result, PlannerResult)
        return result

    def _solve(
        self,
        *,
        system_prompt: str,
        user_message: str,
        toolkit: Toolkit,
        max_turns: int,
        input_queue,
        dashboard_conversation: _VllmDashboardConversation | None,
        deadline: float,
        stop_event: threading.Event,
    ) -> PlannerResult:
        """Request and execute bounded structured Toolkit calls."""
        del input_queue
        try:
            return self._solve_loop(
                system_prompt=system_prompt,
                user_message=user_message,
                toolkit=toolkit,
                max_turns=max_turns,
                deadline=deadline,
                stop_event=stop_event,
                dashboard_conversation=dashboard_conversation,
            )
        finally:
            if dashboard_conversation is not None:
                dashboard_conversation.end()

    def _solve_loop(
        self,
        *,
        system_prompt: str,
        user_message: str,
        toolkit: Toolkit,
        max_turns: int,
        deadline: float,
        stop_event: threading.Event,
        dashboard_conversation: _VllmDashboardConversation | None,
    ) -> PlannerResult:
        """Run the model/tool loop, optionally attached to Dashboard input."""
        try:
            tool_surface = (
                toolkit
                if self._workspace_profile is None
                else compose_tool_surface(
                    toolkit,
                    self._workspace_profile,
                    dashboard_events=self._dashboard_events,
                )
            )
            tool_specs = tool_surface.get_tools_spec()
            tools = [_openai_tool(spec) for spec in tool_specs]
            tool_specs_by_name = {str(spec["name"]): spec for spec in tool_specs}
            if self._workspace_profile is not None and self._workspace_profile.enabled:
                _publish_workspace_catalog(self._dashboard_events, tool_surface)
        except Exception as exc:  # noqa: BLE001 - fail before any tool executes
            return PlannerResult(
                error=f"{type(exc).__name__}: {exc}",
                stats={"turns_used": 0, "tool_calls": 0},
            )
        if not tools:
            return PlannerResult(
                error="ModelApiUtilsPlanner requires a tool surface",
                stats={"turns_used": 0, "tool_calls": 0},
            )
        if dashboard_conversation is not None:
            dashboard_conversation.start()
        tool_choice = "required"
        request_messages: list[dict[str, Any]] = []
        protocol_prompt = _multi_tool_protocol_prompt(
            tool_surface=tool_surface,
            tool_specs=tool_specs,
            policy=self._multi_tool_policy,
        )
        dashboard_instruction = (
            _DASHBOARD_PUBLIC_PROGRESS_INSTRUCTION
            if dashboard_conversation is not None
            else ""
        )
        effective_system_prompt = "\n\n".join(
            part
            for part in (system_prompt.strip(), protocol_prompt, dashboard_instruction)
            if part
        )
        self._dashboard_events.emit(PlannerSystemContextEvent(
            effective_system_prompt, source="prepared_request",
        ))
        request_messages.append({"role": "system", "content": effective_system_prompt})
        request_messages.append({"role": "user", "content": user_message})
        transcript: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
        usage_totals = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
        }
        tool_calls_used = 0
        proposal_repairs = 0
        consecutive_repairs = 0
        last_feedback = None
        retry_without_thinking = False
        history_stats = {
            "history_compactions": 0,
            "request_budget_compactions": 0,
            "request_text_budget_bytes": self._history_policy.max_request_bytes,
            "request_image_budget_bytes": (
                self._history_policy.max_request_image_bytes
            ),
            "request_wire_budget_bytes": (self._history_policy.max_request_wire_bytes),
            "max_request_messages": 0,
            "max_request_chars": 0,
            "max_request_text_bytes": 0,
            "max_request_image_bytes": 0,
            "max_request_wire_bytes": 0,
            "max_unprepared_request_text_bytes": 0,
            "max_unprepared_request_image_bytes": 0,
            "max_unprepared_request_wire_bytes": 0,
            "tool_execution_errors": 0,
            "loop_feedbacks": 0,
            "proposal_repairs": 0,
            "model_requests": 0,
            "model_elapsed_s": 0.0,
            "max_model_request_elapsed_s": 0.0,
            "tool_elapsed_s": 0.0,
            "max_tool_elapsed_s": 0.0,
            "multi_tool_batches": 0,
            "stateful_batches_deferred": 0,
            "max_tool_calls_per_response": 0,
            "max_observations_without_action": 0,
            "stagnant_actions": 0,
            "state_changes": 0,
            "semantic_progress_feedbacks": 0,
            "max_equivalent_capped_actions_seen": 0,
            "workspace_tools_enabled": int(
                bool(self._workspace_profile and self._workspace_profile.enabled)
            ),
            "multi_tool_calls_enabled": int(self._multi_tool_policy.enabled),
            "context_overflow_retries": 0,
            "no_thinking_recovery_requests": 0,
        }
        planner_started = deadline - self._timeout_s

        def stopped_result(turns: int) -> PlannerResult | None:
            if (
                dashboard_conversation is not None
                and dashboard_conversation.stop_requested
            ):
                return dashboard_conversation.stopped_result(
                    transcript=transcript,
                    stats=_stats(
                        turns,
                        tool_calls_used,
                        usage_totals,
                        history_stats,
                    ),
                )
            if stop_event.is_set() or time.monotonic() >= deadline:
                return _timeout_result(
                    timeout_s=self._timeout_s,
                    started=planner_started,
                    transcript=transcript,
                    turns=turns,
                    tool_calls=tool_calls_used,
                    usage=usage_totals,
                    history=history_stats,
                )
            return None

        for turn in range(1, max_turns + 1):
            if result := stopped_result(turn - 1):
                return result
            if dashboard_conversation is not None:
                dashboard_conversation.claim_messages(request_messages)
            request_enable_thinking = (
                self._enable_thinking and not retry_without_thinking
            )
            if self._enable_thinking and retry_without_thinking:
                history_stats["no_thinking_recovery_requests"] += 1
            retry_without_thinking = False
            context_overflow_retry = False
            try:
                request_extra_body = {
                    "enable_thinking": request_enable_thinking,
                    "tools": tools,
                    "tool_choice": tool_choice,
                    "parallel_tool_calls": self._multi_tool_policy.enabled,
                }

                def request_payload(
                    messages: list[dict[str, Any]],
                ) -> dict[str, Any]:
                    return _build_chat_payload(
                        messages=messages,
                        model=self._model,
                        temperature=self._temperature,
                        max_tokens=self._max_tokens,
                        extra_body=request_extra_body,
                    )

                unpruned_payload = request_payload(request_messages)
                unpruned_image_bytes = _chat_payload_inline_image_bytes(
                    unpruned_payload
                )
                history_stats["max_unprepared_request_image_bytes"] = max(
                    history_stats["max_unprepared_request_image_bytes"],
                    unpruned_image_bytes,
                )
                history_stats["max_unprepared_request_wire_bytes"] = max(
                    history_stats["max_unprepared_request_wire_bytes"],
                    _chat_payload_wire_bytes(unpruned_payload),
                )
                image_budget_exceeded = (
                    unpruned_image_bytes > self._history_policy.max_request_image_bytes
                )
                if image_budget_exceeded:
                    history_stats["request_budget_compactions"] += 1
                images_pruned = _prune_request_images(
                    request_messages,
                    max_bytes=self._history_policy.max_request_image_bytes,
                )

                def request_text_bytes(messages: list[dict[str, Any]]) -> int:
                    return _chat_payload_text_bytes(request_payload(messages))

                unprepared_request_bytes = request_text_bytes(request_messages)
                if not image_budget_exceeded and (
                    unprepared_request_bytes > self._history_policy.max_request_bytes
                    or images_pruned > 0
                ):
                    history_stats["request_budget_compactions"] += 1
                history_stats["max_unprepared_request_text_bytes"] = max(
                    history_stats["max_unprepared_request_text_bytes"],
                    unprepared_request_bytes,
                )
                prepared_messages, compacted = self._history_policy.prepare(
                    request_messages,
                    request_size=request_text_bytes,
                )
                prepared_payload = request_payload(prepared_messages)
                prepared_request_bytes = _chat_payload_text_bytes(prepared_payload)
                prepared_image_bytes = _chat_payload_inline_image_bytes(
                    prepared_payload
                )
                prepared_wire_bytes = _chat_payload_wire_bytes(prepared_payload)
                if prepared_image_bytes > self._history_policy.max_request_image_bytes:
                    raise RequestBudgetExceeded(
                        "prepared inline images exceed "
                        "max_request_image_bytes="
                        f"{self._history_policy.max_request_image_bytes}; "
                        f"image_bytes={prepared_image_bytes}"
                    )
                if prepared_wire_bytes > self._history_policy.max_request_wire_bytes:
                    raise RequestBudgetExceeded(
                        "prepared request body exceeds "
                        "max_request_wire_bytes="
                        f"{self._history_policy.max_request_wire_bytes}; "
                        f"wire_bytes={prepared_wire_bytes}"
                    )
                history_stats["history_compactions"] += int(compacted)
                history_stats["max_request_messages"] = max(
                    history_stats["max_request_messages"], len(prepared_messages)
                )
                history_stats["max_request_chars"] = max(
                    history_stats["max_request_chars"],
                    _chat_messages_chars(prepared_messages),
                )
                history_stats["max_request_text_bytes"] = max(
                    history_stats["max_request_text_bytes"], prepared_request_bytes
                )
                history_stats["max_request_image_bytes"] = max(
                    history_stats["max_request_image_bytes"], prepared_image_bytes
                )
                history_stats["max_request_wire_bytes"] = max(
                    history_stats["max_request_wire_bytes"], prepared_wire_bytes
                )
            except Exception as exc:  # noqa: BLE001 - surfaced in PlannerResult
                if dashboard_conversation is not None:
                    dashboard_conversation.fail_request(exc)
                return PlannerResult(
                    messages=transcript,
                    stats=_stats(
                        turn - 1, tool_calls_used, usage_totals, history_stats
                    ),
                    error=f"{type(exc).__name__}: {exc}",
                )
            request_started = time.perf_counter()
            try:
                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0:
                    return _timeout_result(
                        timeout_s=self._timeout_s,
                        started=planner_started,
                        transcript=transcript,
                        turns=turn - 1,
                        tool_calls=tool_calls_used,
                        usage=usage_totals,
                        history=history_stats,
                    )
                response = self._client.chat(
                    model=self._model,
                    messages=prepared_messages,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    extra_body=request_extra_body,
                    timeout_s=remaining_s,
                )
            except Exception as exc:  # noqa: BLE001 - surfaced in PlannerResult
                _record_elapsed(
                    history_stats,
                    prefix="model",
                    elapsed_s=time.perf_counter() - request_started,
                )
                if (
                    dashboard_conversation is not None
                    and dashboard_conversation.stop_requested
                ):
                    dashboard_conversation.fail_request(exc)
                    if result := stopped_result(turn - 1):
                        return result
                if stop_event.is_set() or time.monotonic() >= deadline:
                    return stopped_result(turn - 1)
                if not (
                    self._history_policy.enabled and _is_context_overflow_error(exc)
                ):
                    if dashboard_conversation is not None:
                        dashboard_conversation.fail_request(exc)
                    return PlannerResult(
                        messages=transcript,
                        stats=_stats(
                            turn - 1, tool_calls_used, usage_totals, history_stats
                        ),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                context_overflow_retry = True
                retry_source = copy.deepcopy(request_messages)
                _prune_request_images(
                    retry_source,
                    max_bytes=self._history_policy.max_request_image_bytes,
                    max_images=1,
                )
                try:
                    prepared_messages, compacted = self._history_policy.prepare(
                        retry_source,
                        request_size=request_text_bytes,
                        recent_tool_exchanges_override=1,
                    )
                    retry_payload = request_payload(prepared_messages)
                    prepared_request_bytes = _chat_payload_text_bytes(retry_payload)
                    retry_image_bytes = _chat_payload_inline_image_bytes(retry_payload)
                    retry_wire_bytes = _chat_payload_wire_bytes(retry_payload)
                    prepared_image_bytes = retry_image_bytes
                    prepared_wire_bytes = retry_wire_bytes
                    if retry_image_bytes > self._history_policy.max_request_image_bytes:
                        raise RequestBudgetExceeded(
                            "retry inline images exceed "
                            "max_request_image_bytes="
                            f"{self._history_policy.max_request_image_bytes}; "
                            f"image_bytes={retry_image_bytes}"
                        )
                    if retry_wire_bytes > self._history_policy.max_request_wire_bytes:
                        raise RequestBudgetExceeded(
                            "retry request body exceeds "
                            "max_request_wire_bytes="
                            f"{self._history_policy.max_request_wire_bytes}; "
                            f"wire_bytes={retry_wire_bytes}"
                        )
                    history_stats["history_compactions"] += int(compacted)
                    history_stats["context_overflow_retries"] += 1
                    history_stats["max_request_messages"] = max(
                        history_stats["max_request_messages"], len(prepared_messages)
                    )
                    request_started = time.perf_counter()
                    remaining_s = deadline - time.monotonic()
                    if remaining_s <= 0:
                        return _timeout_result(
                            timeout_s=self._timeout_s,
                            started=planner_started,
                            transcript=transcript,
                            turns=turn - 1,
                            tool_calls=tool_calls_used,
                            usage=usage_totals,
                            history=history_stats,
                        )
                    response = self._client.chat(
                        model=self._model,
                        messages=prepared_messages,
                        temperature=self._temperature,
                        max_tokens=self._max_tokens,
                        extra_body=request_extra_body,
                        timeout_s=remaining_s,
                    )
                except Exception as retry_exc:  # noqa: BLE001
                    _record_elapsed(
                        history_stats,
                        prefix="model",
                        elapsed_s=time.perf_counter() - request_started,
                    )
                    if dashboard_conversation is not None:
                        dashboard_conversation.fail_request(retry_exc)
                    if result := stopped_result(turn - 1):
                        return result
                    return PlannerResult(
                        messages=transcript,
                        stats=_stats(
                            turn - 1, tool_calls_used, usage_totals, history_stats
                        ),
                        error=f"{type(retry_exc).__name__}: {retry_exc}",
                    )
            request_elapsed_s = time.perf_counter() - request_started
            _record_elapsed(
                history_stats,
                prefix="model",
                elapsed_s=request_elapsed_s,
            )
            if dashboard_conversation is not None:
                dashboard_conversation.acknowledge_request(transcript)
            if result := stopped_result(turn - 1):
                return result
            _accumulate_usage(usage_totals, getattr(response, "usage", {}))
            request_usage = _request_usage(getattr(response, "usage", {}))
            if request_usage is not None:
                self._dashboard_events.emit(
                    PlannerRequestUsageEvent(
                        turn=turn,
                        input_tokens=request_usage[0],
                        output_tokens=request_usage[1],
                        request_index=int(history_stats["model_requests"]),
                        model=self._model,
                        temperature=self._temperature,
                        max_tokens=self._max_tokens,
                        enable_thinking=request_enable_thinking,
                        parallel_tool_calls=self._multi_tool_policy.enabled,
                        tool_choice=_tool_choice_label(tool_choice),
                        message_count=len(prepared_messages),
                        message_chars=_chat_messages_chars(prepared_messages),
                        tool_count=len(tools),
                        elapsed_s=request_elapsed_s,
                        request_text_bytes=prepared_request_bytes,
                        request_image_bytes=prepared_image_bytes,
                        request_wire_bytes=prepared_wire_bytes,
                        **_request_context_parts(prepared_messages, tools),
                        compacted=bool(compacted),
                        context_overflow_retry=context_overflow_retry,
                        cached_input_tokens=_nested_usage_tokens(
                            getattr(response, "usage", {}),
                            "prompt_tokens_details",
                            "cached_tokens",
                        ),
                        reasoning_output_tokens=_nested_usage_tokens(
                            getattr(response, "usage", {}),
                            "completion_tokens_details",
                            "reasoning_tokens",
                        ),
                    )
                )
            self._dashboard_events.emit(
                UsageEvent(
                    inp=usage_totals["total_input_tokens"],
                    out=usage_totals["total_output_tokens"],
                    tool_calls=tool_calls_used,
                )
            )
            execution_feedback: dict[str, object] | None = None
            try:
                assistant = _assistant_message(response)
                _emit_assistant_events(
                    self._dashboard_events,
                    assistant,
                    request_index=int(history_stats["model_requests"]),
                    turn=turn,
                )
                if dashboard_conversation is not None:
                    claimed_messages = dashboard_conversation.claim_messages(
                        request_messages
                    )
                    if claimed_messages and turn < max_turns:
                        continue
                    if claimed_messages:
                        return PlannerResult(
                            messages=transcript,
                            stats=_stats(
                                turn,
                                tool_calls_used,
                                usage_totals,
                                history_stats,
                            ),
                            error=(
                                "Dashboard input superseded the final vLLM response, "
                                f"but max_turns={max_turns} is exhausted"
                            ),
                        )
                calls = [
                    _parse_tool_call(raw_call, index=index)
                    for index, raw_call in enumerate(assistant["tool_calls"])
                ]
                _validate_top_level_required_arguments(calls, tool_specs_by_name)
                batch_feedback = self._multi_tool_policy.validate(
                    calls, tool_surface=tool_surface
                )
                proposed_call_count = len(calls)
                if (
                    batch_feedback is not None
                    and batch_feedback.get("code") == "multi_tool_call_not_safe"
                ):
                    execution_feedback = {
                        "schema": "rpent.planner_execution_feedback.v1",
                        "stage": "planner_call",
                        "code": "stateful_batch_deferred",
                        "executed_tool": calls[0][1],
                        "deferred_tools": [name for _id, name, _args in calls[1:]],
                        "instruction": (
                            "Only the first call was executed. The remaining calls "
                            "were not executed because they may depend on pre-action "
                            "state; inspect the tool result and re-plan."
                        ),
                    }
                    calls = calls[:1]
                    assistant = dict(assistant)
                    assistant["tool_calls"] = list(assistant["tool_calls"][:1])
                elif batch_feedback is not None:
                    raise _RepairableModelProposalError(batch_feedback)
                validate_proposal = getattr(
                    tool_surface, "validate_planner_proposal", None
                )
                loop_feedback = (
                    validate_proposal(calls) if callable(validate_proposal) else None
                )
                if loop_feedback is not None:
                    raise _RepairableModelProposalError(loop_feedback)
            except _RepairableModelProposalError as exc:
                proposal_repairs += 1
                consecutive_repairs += 1
                history_stats["proposal_repairs"] = proposal_repairs
                # Rejected proposals are not added as assistant tool exchanges.
                # Include their bounded context so the model can identify what
                # failed instead of interpreting the previous accepted call as
                # the rejected proposal. No tool in this response has executed.
                feedback = {
                    **exc.feedback,
                    "executed": False,
                    "rejected_proposal": _rejected_proposal_text(response),
                }
                last_feedback = feedback
                retry_without_thinking = bool(
                    self._enable_thinking
                    and request_enable_thinking
                    and exc.feedback.get("code") == "planner_tool_calls_missing"
                )
                retry_tool = exc.feedback.get("retry_tool")
                tool_choice = (
                    {"type": "function", "function": {"name": retry_tool}}
                    if isinstance(retry_tool, str) and retry_tool in tool_specs_by_name
                    else "required"
                )
                transcript.append({"role": "repair_feedback", "content": feedback})
                if consecutive_repairs > self._max_proposal_repairs:
                    return PlannerResult(
                        messages=transcript,
                        stats=_stats(
                            turn, tool_calls_used, usage_totals, history_stats
                        ),
                        error="Planner proposal repair budget exhausted",
                    )
                request_messages.append(
                    {
                        "role": "user",
                        "content": json.dumps(feedback, separators=(",", ":")),
                    }
                )
                continue
            except Exception as exc:  # noqa: BLE001
                return PlannerResult(
                    messages=transcript,
                    stats=_stats(turn, tool_calls_used, usage_totals, history_stats),
                    error=f"{type(exc).__name__}: {exc}",
                )
            consecutive_repairs = 0
            last_feedback = None
            tool_choice = "required"
            history_stats["max_tool_calls_per_response"] = max(
                history_stats["max_tool_calls_per_response"], proposed_call_count
            )
            history_stats["multi_tool_batches"] += int(
                proposed_call_count > 1 and execution_feedback is None
            )
            history_stats["stateful_batches_deferred"] += int(
                execution_feedback is not None
            )

            request_messages.append(assistant)
            transcript_message = {
                "role": "assistant",
                "content": assistant.get("content"),
                "tool_calls": assistant["tool_calls"],
            }
            reasoning = assistant.get("reasoning_content")
            if not isinstance(reasoning, str):
                reasoning = assistant.get("reasoning")
            if isinstance(reasoning, str):
                transcript_message["reasoning_content"] = reasoning
            transcript.append(transcript_message)
            pending_images: list[tuple[str, list[dict[str, object]]]] = []
            pending_feedback: list[dict[str, object]] = []
            if execution_feedback is not None:
                pending_feedback.append(execution_feedback)
            terminal_loop_error: str | None = None
            finish_result: dict[str, Any] | None = None
            for call_index, (call_id, name, arguments) in enumerate(calls):
                if result := stopped_result(turn):
                    return result
                tool_calls_used += 1
                # Internal response bookkeeping may synthesize an ID. Only the
                # actual provider field is public association evidence.
                reported_id = assistant["tool_calls"][call_index].get("id")
                identity = (
                    {"tool_call_id": reported_id}
                    if isinstance(reported_id, str) and reported_id
                    else {}
                )
                self._dashboard_events.emit(
                    TranscriptEvent(
                        {"type": "tool_call", "tool": name, "args": arguments, **identity}
                    )
                )
                tool_started = time.perf_counter()
                try:
                    result = tool_surface.execute_tool(name, arguments)
                    result_dict = result.result
                    result_blocks = getattr(result, "content_blocks", ())
                except Exception as exc:  # noqa: BLE001 - adapter boundary
                    result_dict = {"error": f"tool {name} failed: {exc}"}
                    result_blocks = ()
                _record_elapsed(
                    history_stats,
                    prefix="tool",
                    elapsed_s=time.perf_counter() - tool_started,
                )
                history_stats["tool_execution_errors"] += int(
                    bool(result_dict.get("error"))
                )
                try:
                    result_text, image_content = _tool_content(
                        name,
                        result_dict,
                        result_blocks,
                        no_images=self._no_images,
                    )
                except ResultProjectionError as exc:
                    detail = (
                        f"tool {name!r} executed but its result could not be "
                        f"represented safely: {exc}"
                    )
                    audit_result = executed_result_unavailable(
                        tool_name=name,
                        tool_call_id=call_id,
                        error=exc,
                    )
                    transcript.append(
                        {
                            "role": "tool",
                            "name": name,
                            "tool_call_id": call_id,
                            "content": json.dumps(
                                audit_result,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        }
                    )
                    return PlannerResult(
                        messages=transcript,
                        stats=_stats(
                            turn,
                            tool_calls_used,
                            usage_totals,
                            history_stats,
                        ),
                        error=detail,
                    )
                request_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        # The deployed backend accepts text-only tool messages;
                        # images are appended after every result in this batch.
                        "content": result_text,
                    }
                )
                if image_content:
                    pending_images.append((name, image_content))
                transcript.append(
                    {
                        "role": "tool",
                        "name": name,
                        "tool_call_id": call_id,
                        "content": result_text,
                    }
                )
                loop_error = result_dict.get("planner_loop_error")
                if isinstance(loop_error, str) and terminal_loop_error is None:
                    terminal_loop_error = loop_error
                if result_dict.get("_finish") is True:
                    finish_result = result_dict
                self._dashboard_events.emit(
                    TranscriptEvent(
                        {
                            "type": "tool_result",
                            "tool": name,
                            **identity,
                            "result": {
                                "is_error": bool(result_dict.get("error")),
                                "size": len(result_text),
                            },
                        }
                    )
                )
                if result := stopped_result(turn):
                    return result
                self._dashboard_events.emit(
                    UsageEvent(
                        inp=usage_totals["total_input_tokens"],
                        out=usage_totals["total_output_tokens"],
                        tool_calls=tool_calls_used,
                    )
                )
            if execution_feedback is not None:
                transcript.append(
                    {
                        "role": "planner_execution_feedback",
                        "content": execution_feedback,
                    }
                )
            if pending_images:
                request_messages.append(
                    {
                        "role": "user",
                        "content": _batch_image_content(pending_images),
                    }
                )
            for feedback in pending_feedback:
                request_messages.append(
                    {
                        "role": "user",
                        "content": json.dumps(feedback, separators=(",", ":")),
                    }
                )
            if terminal_loop_error is not None:
                return PlannerResult(
                    messages=transcript,
                    stats=_stats(turn, tool_calls_used, usage_totals, history_stats),
                    error=terminal_loop_error,
                )
            if finish_result is not None:
                return PlannerResult(
                    finish_result=finish_result,
                    messages=transcript,
                    stats=_stats(turn, tool_calls_used, usage_totals, history_stats),
                )

        suffix = " after a repairable response" if last_feedback else ""
        return PlannerResult(
            messages=transcript,
            stats=_stats(max_turns, tool_calls_used, usage_totals, history_stats),
            error=f"model exhausted max_turns={max_turns} without finish{suffix}",
        )


def _emit_assistant_events(
    dashboard_events: DashboardEventSink,
    assistant: Mapping[str, Any],
    *,
    request_index: int,
    turn: int,
) -> None:
    """Publish vLLM text and reasoning while the TaskRun is still active."""
    reasoning = assistant.get("reasoning_content")
    if not isinstance(reasoning, str):
        reasoning = assistant.get("reasoning")
    association = {"request_index": request_index, "turn": turn}
    if isinstance(reasoning, str) and reasoning:
        dashboard_events.emit(
            TranscriptEvent({"type": "thinking", "text": reasoning, **association})
        )
    content = assistant.get("content")
    if isinstance(content, str) and content:
        dashboard_events.emit(
            TranscriptEvent({"type": "text", "text": content, **association})
        )


def _openai_tool(spec: Mapping[str, object]) -> dict[str, object]:
    name = spec.get("name")
    schema = spec.get("input_schema")
    if not isinstance(name, str) or not name:
        raise ValueError("Toolkit tool name must be a non-empty string")
    if not isinstance(schema, Mapping):
        raise ValueError(f"Toolkit tool {name!r} has no input_schema")
    description = spec.get("description", "")
    if not isinstance(description, str):
        raise ValueError(f"Toolkit tool {name!r} description must be a string")
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": _vllm_compatible_parameters(schema),
            "strict": True,
        },
    }


def _multi_tool_protocol_prompt(
    *,
    tool_surface: Any,
    tool_specs: list[dict[str, Any]],
    policy: MultiToolPolicy,
) -> str:
    """Describe the adapter's real batching contract to the model."""
    if not policy.enabled:
        return (
            "Tool-call protocol: make exactly one tool call in each response. "
            "Wait for its result before proposing the next call."
        )
    checker = getattr(tool_surface, "is_tool_multi_call_safe", None)
    safe_names = sorted(
        str(spec["name"])
        for spec in tool_specs
        if callable(checker) and checker(str(spec["name"]))
    )
    allowed = ", ".join(safe_names) if safe_names else "none"
    return (
        "Tool-call protocol: you may put at most "
        f"{policy.max_calls} distinct independent read-only calls in one response, "
        f"and only for these batch-safe tools: {allowed}. Every other tool must "
        "be the only call in its response. Wait for action results before planning "
        "another action, and do not repeat an unchanged observation whose result "
        "is already present in the conversation."
    )


def _vllm_compatible_parameters(
    schema: Mapping[str, object],
) -> dict[str, object]:
    """Flatten nullable top-level oneOf branches for vLLM tool decoding."""
    parameters = dict(schema)
    branches = parameters.get("oneOf")
    properties = parameters.get("properties")
    if not isinstance(branches, list) or not isinstance(properties, Mapping):
        return parameters

    branch_fields: list[str] = []
    for branch in branches:
        required = branch.get("required") if isinstance(branch, Mapping) else None
        if not isinstance(required, list):
            return parameters
        for field in required:
            field_schema = properties.get(field) if isinstance(field, str) else None
            field_types = (
                field_schema.get("type") if isinstance(field_schema, Mapping) else None
            )
            if not isinstance(field, str) or not (
                field_types == "null"
                or isinstance(field_types, list)
                and "null" in field_types
            ):
                return parameters
            if field not in branch_fields:
                branch_fields.append(field)

    required = list(parameters.get("required", ()))
    parameters["required"] = required + [
        field for field in branch_fields if field not in required
    ]
    parameters.pop("oneOf")
    return parameters


def _assistant_message(response: object) -> dict[str, object]:
    raw = getattr(response, "raw", None)
    choices = raw.get("choices") if isinstance(raw, Mapping) else None
    if not isinstance(choices, list) or len(choices) != 1:
        raise _repair(
            "/choices", "choice_count_invalid", "exactly one assistant choice"
        )
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, Mapping) else None
    if not isinstance(message, Mapping):
        raise _repair(
            "/choices/0/message", "assistant_message_missing", "one assistant message"
        )
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        content = message.get("content")
        actual_parts = [
            f"assistant prose without tool call: {content[:500]}"
            if isinstance(content, str) and content.strip()
            else "empty assistant message"
        ]
        finish_reason = choice.get("finish_reason")
        if isinstance(finish_reason, str) and finish_reason:
            actual_parts.append(f"finish_reason={finish_reason!r}")
        reasoning = message.get("reasoning_content")
        if not isinstance(reasoning, str):
            reasoning = message.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            actual_parts.append(f"reasoning_chars={len(reasoning)}")
        raise _repair(
            "/choices/0/message/tool_calls",
            "planner_tool_calls_missing",
            (
                "at least one structured tool call; continue with an observation "
                "or action tool when more evidence or work is needed, or call "
                "finish with status='failure' or status='stuck' and a summary "
                "when no safe recovery remains; do not respond with prose alone"
            ),
            actual="; ".join(actual_parts),
        )
    result = {
        "role": "assistant",
        "content": message.get("content"),
        "tool_calls": tool_calls,
    }
    for field in ("reasoning", "reasoning_content"):
        if isinstance(message.get(field), str):
            result[field] = message[field]
    return result


def _rejected_proposal_text(response: object) -> str:
    """Describe a rejected provider message without replaying invalid calls."""
    raw = getattr(response, "raw", None)
    choices = raw.get("choices") if isinstance(raw, Mapping) else None
    choice = choices[0] if isinstance(choices, list) and len(choices) == 1 else None
    message = choice.get("message") if isinstance(choice, Mapping) else None
    if not isinstance(message, Mapping):
        return "assistant message unavailable"
    text = json.dumps(
        {"tool_calls": message.get("tool_calls"), "content": message.get("content")},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    encoded = text.encode("utf-8")
    if len(encoded) <= 4096:
        return text
    marker = "...[truncated]"
    return encoded[: 4096 - len(marker)].decode("utf-8", errors="ignore") + marker


def _parse_tool_call(
    raw_call: object, *, index: int
) -> tuple[str, str, dict[str, Any]]:
    path = f"/tool_calls/{index}"
    if not isinstance(raw_call, Mapping):
        raise _repair(path, "tool_call_not_object", "one tool call object")
    call_id = raw_call.get("id") or f"rpent-call-{index}"
    function = raw_call.get("function")
    if not isinstance(function, Mapping):
        raise _repair(
            f"{path}/function", "tool_function_missing", "one function object"
        )
    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise _repair(
            f"{path}/function/name", "tool_function_name_missing", "a tool name"
        )
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise _repair(
                f"{path}/function/arguments",
                "tool_arguments_json_invalid",
                "JSON object arguments",
                actual=f"line {exc.lineno} column {exc.colno}",
            ) from exc
    if not isinstance(arguments, Mapping):
        raise _repair(
            f"{path}/function/arguments",
            "tool_arguments_not_object",
            "JSON object arguments",
        )
    return str(call_id), name, dict(arguments)


def _validate_top_level_required_arguments(
    calls: list[tuple[str, str, dict[str, Any]]],
    specs_by_name: Mapping[str, Mapping[str, object]],
) -> None:
    """Reject proposals missing fields declared required by a tool schema."""
    for index, (_call_id, name, arguments) in enumerate(calls):
        spec = specs_by_name.get(name)
        schema = spec.get("input_schema") if isinstance(spec, Mapping) else None
        required = schema.get("required") if isinstance(schema, Mapping) else None
        if not isinstance(required, list):
            continue
        for field in required:
            if isinstance(field, str) and field not in arguments:
                expected = f"call {name!r} again with required argument {field!r}"
                properties = schema.get("properties")
                field_schema = (
                    properties.get(field) if isinstance(properties, Mapping) else None
                )
                if isinstance(field_schema, Mapping):
                    allowed = field_schema.get("enum")
                    if isinstance(allowed, list):
                        expected += f" set to one of {json.dumps(allowed)}"
                    description = field_schema.get("description")
                    if isinstance(description, str) and description:
                        expected += f". {description}"
                _repair(
                    f"/tool_calls/{index}/function/arguments/{field}",
                    "tool_argument_required",
                    expected,
                    retry_tool=name,
                )


def _repair(
    path: str,
    code: str,
    expected: str,
    *,
    actual: str = "missing",
    retry_tool: str | None = None,
):
    feedback: dict[str, object] = {
        "schema": "rpent.planner_repair_feedback.v1",
        "stage": "planner_call",
        "path": path,
        "code": code,
        "expected": expected,
        "actual": actual,
        "repairable": True,
    }
    if retry_tool is not None:
        feedback["retry_tool"] = retry_tool
    raise _RepairableModelProposalError(feedback)


def _tool_content(
    tool_name: str,
    result: Mapping[str, object],
    blocks,
    *,
    no_images: bool,
) -> tuple[str, list[dict[str, object]]]:
    """Keep textual state in the tool message and attach bounded PNG images."""
    text = bounded_tool_result_text(
        tool_name=tool_name,
        result=result,
        max_bytes=ToolResult.MAX_TEXT_BYTES_IN_RESULT,
        reason="tool_result_limit",
    )
    if no_images:
        return text, []
    content: list[dict[str, object]] = []
    for block in blocks or ():
        if not isinstance(block, Mapping) or block.get("type") != "image":
            continue
        source = block.get("source")
        if not isinstance(source, Mapping) or source.get("type") != "base64":
            continue
        data = source.get("data")
        if isinstance(data, str) and data:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{source.get('media_type', 'image/png')};base64,{data}"
                    },
                }
            )
    return text, content


def _batch_image_content(
    pending: list[tuple[str, list[dict[str, object]]]],
) -> list[dict[str, object]]:
    """Attach images only after every tool call has a matching result."""
    if len(pending) == 1:
        _name, images = pending[0]
        return [
            {
                "type": "text",
                "text": "Current visual observation from the tool result:",
            },
            *images,
        ]
    content: list[dict[str, object]] = [
        {
            "type": "text",
            "text": "Current visual observations from the completed tool batch:",
        }
    ]
    for name, images in pending:
        content.append({"type": "text", "text": f"Observation from {name}:"})
        content.extend(images)
    return content


def _timeout_result(
    *,
    timeout_s: float,
    started: float,
    transcript: list[dict[str, Any]],
    turns: int,
    tool_calls: int,
    usage: Mapping[str, int],
    history: Mapping[str, int | float],
) -> PlannerResult:
    stats = _stats(turns, tool_calls, usage, history)
    stats["planner_elapsed_s"] = round(time.monotonic() - started, 6)
    return PlannerResult(
        messages=transcript,
        stats=stats,
        error=f"vLLM planner timed out after {timeout_s:g}s",
    )


def _prune_request_images(
    messages: list[dict[str, Any]],
    *,
    max_bytes: int,
    max_images: int = 2,
) -> int:
    """Keep the newest image suffix within both a count and byte bound."""
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    if (
        isinstance(max_images, bool)
        or not isinstance(max_images, int)
        or max_images < 1
    ):
        raise ValueError("max_images must be a positive integer")
    located: list[tuple[int, int, int]] = []
    for message_index, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item_index, item in enumerate(content):
            if not isinstance(item, Mapping) or item.get("type") != "image_url":
                continue
            image_url = item.get("image_url")
            url = image_url.get("url") if isinstance(image_url, Mapping) else None
            if isinstance(url, str) and url.startswith("data:"):
                located.append((message_index, item_index, len(url.encode("utf-8"))))
    if not located:
        return 0
    retained: set[tuple[int, int]] = set()
    total = 0
    for message_index, item_index, size in reversed(located):
        if len(retained) >= max_images:
            break
        if total + size > max_bytes:
            if not retained:
                raise RequestBudgetExceeded(
                    "newest inline image exceeds "
                    f"max_request_image_bytes={max_bytes}; image_bytes={size}"
                )
            break
        retained.add((message_index, item_index))
        total += size
    for message_index, item_index, _ in located:
        if (message_index, item_index) in retained:
            continue
        content = messages[message_index].get("content")
        if isinstance(content, list):
            content[item_index] = {
                "type": "text",
                "text": (
                    "[camera image omitted to satisfy request image/count budget]"
                ),
            }
    return len(located) - len(retained)


def _is_context_overflow_error(error: Exception) -> bool:
    detail = str(error).lower()
    return "maximum context length" in detail and "prompt contains" in detail


def _accumulate_usage(totals: dict[str, int], usage: object) -> None:
    if not isinstance(usage, Mapping):
        return
    for source, destination in (
        ("prompt_tokens", "total_input_tokens"),
        ("completion_tokens", "total_output_tokens"),
        ("total_tokens", "total_tokens"),
    ):
        value = usage.get(source)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            totals[destination] += value


def _tool_choice_label(tool_choice: object) -> str:
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, Mapping):
        function = tool_choice.get("function")
        if isinstance(function, Mapping):
            name = function.get("name")
            if isinstance(name, str) and name:
                return f"function:{name}"
    return "custom"


def _nested_usage_tokens(
    usage: object,
    details_key: str,
    token_key: str,
) -> int | None:
    if not isinstance(usage, Mapping):
        return None
    details = usage.get(details_key)
    if not isinstance(details, Mapping):
        return None
    value = details.get(token_key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _request_context_parts(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> dict[str, Any]:
    """Measure text content by role after history/image preparation.

    Image payloads and tool-call JSON are not counted as natural-language text.
    These character counts are not provider token estimates.
    """
    fields = {
        "system": "system_message_chars",
        "user": "user_message_chars",
        "assistant": "assistant_message_chars",
        "tool": "tool_result_chars",
    }
    measured: dict[str, Any] = dict.fromkeys(fields.values(), 0)
    measured["image_count"] = 0
    role_texts: dict[str, list[str]] = {role: [] for role in fields}
    for message in messages:
        content = message.get("content")
        texts = [content] if isinstance(content, str) else []
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") in {"text", "input_text"} and isinstance(
                    part.get("text"), str
                ):
                    texts.append(part["text"])
                elif part.get("type") in {"image_url", "input_image"}:
                    measured["image_count"] += 1
        field = fields.get(message.get("role"))
        if field:
            measured[field] += sum(len(text) for text in texts)
            role_texts[message["role"]].extend(texts)
    measured["tool_schema_bytes"] = len(
        json.dumps(tools, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    measured["system_prompt"] = "\n\n".join(role_texts["system"])
    measured["context_previews"] = {
        role: {"text": "\n\n".join(role_texts[role])}
        for role in ("user", "assistant", "tool")
    }
    measured["context_previews"]["tools"] = {
        "text": json.dumps(tools, ensure_ascii=False, indent=2),
    }
    return measured


def _request_usage(usage: object) -> tuple[int, int] | None:
    if not isinstance(usage, Mapping):
        return None
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if isinstance(prompt, bool) or not isinstance(prompt, int) or prompt < 0:
        return None
    if (
        isinstance(completion, bool)
        or not isinstance(completion, int)
        or completion < 0
    ):
        return None
    return prompt, completion


def _record_elapsed(
    stats: dict[str, int | float], *, prefix: str, elapsed_s: float
) -> None:
    if prefix == "model":
        stats["model_requests"] += 1
        total_key = "model_elapsed_s"
        maximum_key = "max_model_request_elapsed_s"
    elif prefix == "tool":
        total_key = "tool_elapsed_s"
        maximum_key = "max_tool_elapsed_s"
    else:
        raise ValueError(f"unsupported elapsed prefix: {prefix}")
    stats[total_key] += elapsed_s
    stats[maximum_key] = max(stats[maximum_key], elapsed_s)


def _stats(
    turns: int,
    tool_calls: int,
    usage: Mapping[str, int],
    history: Mapping[str, int | float],
) -> dict[str, int | float]:
    result: dict[str, int | float] = {
        "turns_used": turns,
        "tool_calls": tool_calls,
        **dict(usage),
        **dict(history),
    }
    for key in (
        "model_elapsed_s",
        "max_model_request_elapsed_s",
        "tool_elapsed_s",
        "max_tool_elapsed_s",
    ):
        result[key] = round(float(result[key]), 6)
    return result


__all__ = ["ModelApiUtilsPlanner"]
