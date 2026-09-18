# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Shared planner runtime applied above every model backend.

Backends retain ownership of provider transport, streaming, message encoding,
and SDK-specific cancellation. This wrapper owns provider-neutral observation
of RPent Toolkit calls and feeds structured liveness feedback through the same
ToolResult channel used by every backend.
"""

from __future__ import annotations

import copy
import queue
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from rpent.dashboard.interaction import DashboardInteractionPort
from rpent.planner.base import Planner, PlannerResult
from rpent.planner.liveness import (
    ParsedToolCall,
    ToolLoopGuard,
    tool_loop_guard_from_env,
    tool_result_fingerprint,
)
from rpent.tools.toolkit import ToolResult

GuardFactory = Callable[[], ToolLoopGuard]


class PlannerRuntimeToolkit:
    """Apply one run-local liveness policy around a Toolkit-like object."""

    def __init__(self, *, inner: Any, guard: ToolLoopGuard) -> None:
        if not callable(getattr(inner, "get_tools_spec", None)):
            raise TypeError("inner must provide get_tools_spec")
        if not callable(getattr(inner, "execute_tool", None)):
            raise TypeError("inner must provide execute_tool")
        if not isinstance(guard, ToolLoopGuard):
            raise TypeError("guard must be a ToolLoopGuard")
        self._inner = inner
        self._guard = guard
        self._terminal_error: str | None = None
        self._preflight_rejections = 0
        self._consecutive_preflight_rejections = 0
        self._next_proposal_scope = 0
        self._active_failure_scope: int | None = None
        self._remaining_proposal_calls = 0
        self._active_recovery: dict[str, object] | None = None
        self._recovery_results = 0

    @property
    def memory(self):
        return self._inner.memory

    @property
    def state(self):
        return self._inner.state

    @property
    def terminal_error(self) -> str | None:
        return self._terminal_error

    def get_tools_spec(self) -> list[dict[str, Any]]:
        return self._inner.get_tools_spec()

    def is_tool_multi_call_safe(self, name: str) -> bool:
        checker = getattr(self._inner, "is_tool_multi_call_safe", None)
        return bool(callable(checker) and checker(name))

    def validate_planner_proposal(
        self,
        calls: Sequence[ParsedToolCall],
    ) -> dict[str, object] | None:
        """Expose the shared preflight to backends that see a whole proposal."""
        feedback = self._guard.validate_proposal(calls)
        if feedback is not None:
            self._active_failure_scope = None
            self._remaining_proposal_calls = 0
            return feedback
        self._next_proposal_scope += 1
        self._active_failure_scope = self._next_proposal_scope
        self._remaining_proposal_calls = len(calls)
        return None

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> ToolResult:
        """Dispatch one call and project shared policy feedback into its result."""
        failure_scope = self._take_failure_scope()
        if self._terminal_error is not None and name != "finish":
            return self._result_with_recovery(
                ToolResult(
                    name=name,
                    result={
                        "error": self._terminal_error,
                        "code": "planner_loop_stopped",
                        "planner_loop_error": self._terminal_error,
                        "instruction": (
                            "The shared planner runtime has stopped further tool "
                            "execution. Call finish with status='stuck' or 'failure'."
                        ),
                    },
                )
            )

        feedback = self._guard.validate_proposal([("runtime-call", name, input_dict)])
        if feedback is not None:
            self._preflight_rejections += 1
            self._consecutive_preflight_rejections += 1
            payload: dict[str, Any] = {
                "error": "planner tool call rejected before execution",
                "code": str(feedback.get("code") or "planner_call_rejected"),
                "instruction": str(
                    feedback.get("expected")
                    or feedback.get("instruction")
                    or "Choose a materially different tool call."
                ),
                "planner_feedback": copy.deepcopy(feedback),
                "executed": False,
            }
            if (
                self._consecutive_preflight_rejections
                >= self._guard.max_identical_errors
            ):
                self._terminal_error = (
                    "shared planner runtime stopped after "
                    f"{self._consecutive_preflight_rejections} consecutive "
                    "rejected tool proposals"
                )
                payload["planner_loop_error"] = self._terminal_error
            return self._result_with_recovery(ToolResult(name=name, result=payload))

        self._consecutive_preflight_rejections = 0
        result = self._inner.execute_tool(name, input_dict)
        if not isinstance(result, ToolResult):
            raise TypeError("Toolkit.execute_tool must return ToolResult")
        self._refresh_recovery()
        result_dict = result.result
        if not isinstance(result_dict, Mapping):
            result_dict = {"value": result_dict}
        feedback, loop_error = self._guard.observe(
            tool=name,
            arguments=input_dict,
            result=result_dict,
            result_fingerprint=tool_result_fingerprint(
                result_dict,
                result.content_blocks,
            ),
            failure_scope=failure_scope,
        )
        if feedback is None and loop_error is None:
            return self._result_with_recovery(result)

        payload = dict(result_dict)
        if feedback is not None:
            _append_feedback(payload, feedback)
        if loop_error is not None:
            self._terminal_error = loop_error
            payload["planner_loop_error"] = loop_error
            payload.setdefault("error", loop_error)
            payload.setdefault("code", "planner_loop_stopped")
        return self._result_with_recovery(
            ToolResult(name=name, result=payload, call_id=result.call_id)
        )

    def _refresh_recovery(self) -> None:
        """Read current owner state without interpreting environment semantics."""
        getter = getattr(self._inner, "get_planner_recovery", None)
        if not callable(getter):
            return
        recovery = getter()
        if recovery is None:
            self._active_recovery = None
            return
        if not isinstance(recovery, Mapping):
            raise TypeError(
                "Toolkit.get_planner_recovery must return a mapping or None"
            )
        if recovery.get("schema") != "rpent.planner_recovery.v1":
            raise ValueError("unsupported planner recovery schema")
        recovery_id = recovery.get("recovery_id")
        instruction = recovery.get("instruction")
        if not isinstance(recovery_id, str) or not recovery_id.strip():
            raise ValueError("planner recovery_id must be non-empty")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("planner recovery instruction must be non-empty")
        self._active_recovery = copy.deepcopy(dict(recovery))

    def _result_with_recovery(self, result: ToolResult) -> ToolResult:
        """Project one active recovery obligation into an agent-visible result."""
        if self._active_recovery is None:
            return result
        payload = (
            dict(result.result)
            if isinstance(result.result, Mapping)
            else {"value": result.result}
        )
        payload["planner_recovery"] = copy.deepcopy(self._active_recovery)
        self._recovery_results += 1
        return ToolResult(name=result.name, result=payload, call_id=result.call_id)

    def _take_failure_scope(self) -> int | None:
        """Return the active proposal id once for each declared tool call."""
        if self._remaining_proposal_calls <= 0:
            return None
        scope = self._active_failure_scope
        self._remaining_proposal_calls -= 1
        if self._remaining_proposal_calls == 0:
            self._active_failure_scope = None
        return scope

    def stats(self) -> dict[str, int]:
        return {
            **self._guard.stats(),
            "planner_preflight_rejections": self._preflight_rejections,
            "planner_loop_stopped": int(self._terminal_error is not None),
            "planner_recovery_results": self._recovery_results,
        }

    def cancel_active_and_wait(self) -> None:
        self._inner.cancel_active_and_wait()

    def raise_if_cancelled(self) -> None:
        self._inner.raise_if_cancelled()

    def solved(self) -> bool:
        return self._inner.solved()

    def write_recipe(self, recipe_tag: str) -> str | None:
        return self._inner.write_recipe(recipe_tag)

    def close(self) -> None:
        self._inner.close()


class PlannerRuntime:
    """Wrap any Planner with the common Toolkit-call policy."""

    def __init__(
        self,
        *,
        inner: Planner,
        guard_factory: GuardFactory = tool_loop_guard_from_env,
    ) -> None:
        if not callable(getattr(inner, "solve", None)):
            raise TypeError("inner must implement Planner.solve")
        if not callable(guard_factory):
            raise TypeError("guard_factory must be callable")
        self._inner = inner
        self._guard_factory = guard_factory

    @property
    def inner(self) -> Planner:
        return self._inner

    def solve(
        self,
        *,
        system_prompt: str,
        user_message: str,
        toolkit: Any,
        max_turns: int,
        input_queue: queue.Queue[str | None] | None = None,
        dashboard_interaction: DashboardInteractionPort | None = None,
    ) -> PlannerResult:
        """Create one isolated policy state and delegate one planner run."""
        guard = self._guard_factory()
        guard.reset()
        runtime_toolkit = PlannerRuntimeToolkit(inner=toolkit, guard=guard)
        result = self._inner.solve(
            system_prompt=system_prompt,
            user_message=user_message,
            toolkit=runtime_toolkit,
            max_turns=max_turns,
            input_queue=input_queue,
            dashboard_interaction=dashboard_interaction,
        )
        if not isinstance(result, PlannerResult):
            raise TypeError("Planner.solve must return PlannerResult")
        _merge_runtime_stats(result.stats, runtime_toolkit.stats())
        result.stats["planner_runtime"] = "shared"
        if (
            runtime_toolkit.terminal_error is not None
            and result.finish_result is None
            and result.error is None
        ):
            result.error = runtime_toolkit.terminal_error
        return result


def _append_feedback(payload: dict[str, Any], feedback: Mapping[str, object]) -> None:
    existing = payload.get("planner_feedback")
    projected = copy.deepcopy(dict(feedback))
    if existing is None:
        payload["planner_feedback"] = projected
    elif isinstance(existing, list):
        payload["planner_feedback"] = [*existing, projected]
    else:
        payload["planner_feedback"] = [existing, projected]


def _merge_runtime_stats(
    backend_stats: dict[str, Any], runtime_stats: Mapping[str, int]
) -> None:
    """Merge overlapping observations without erasing backend telemetry."""
    for name, value in runtime_stats.items():
        existing = backend_stats.get(name)
        if (
            isinstance(existing, (int, float))
            and not isinstance(existing, bool)
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            backend_stats[name] = max(existing, value)
        else:
            backend_stats[name] = value


__all__ = ["PlannerRuntime", "PlannerRuntimeToolkit"]
