# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Policy for accepting several independent observations in one response."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

ParsedToolCall = tuple[str, str, dict[str, Any]]


@dataclass(frozen=True, slots=True)
class MultiToolPolicy:
    """Permit bounded batches only for explicitly marked read-only tools."""

    enabled: bool = True
    max_calls: int = 8

    def __post_init__(self) -> None:
        if isinstance(self.max_calls, bool) or not isinstance(self.max_calls, int):
            raise TypeError("max_calls must be an integer")
        if not 2 <= self.max_calls <= 8:
            raise ValueError("max_calls must be from 2 to 8")

    def validate(
        self,
        calls: Sequence[ParsedToolCall],
        *,
        tool_surface: Any,
    ) -> dict[str, object] | None:
        """Return repair feedback when a proposed batch is not safe."""
        if len(calls) <= 1:
            return None
        if not self.enabled:
            return _feedback(
                "multi_tool_calls_disabled",
                "exactly one tool call per response",
                actual=f"{len(calls)} calls",
            )
        if len(calls) > self.max_calls:
            return _feedback(
                "multi_tool_call_limit_exceeded",
                f"at most {self.max_calls} independent read-only calls",
                actual=f"{len(calls)} calls",
            )
        call_ids = [call_id for call_id, _name, _arguments in calls]
        if len(set(call_ids)) != len(call_ids):
            return _feedback(
                "duplicate_tool_call_id",
                "a unique id for every tool call",
                actual="duplicate ids",
            )
        call_signatures = [
            json.dumps(
                {"tool": name, "arguments": arguments},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            for _call_id, name, arguments in calls
        ]
        if len(set(call_signatures)) != len(call_signatures):
            return _feedback(
                "duplicate_tool_call",
                "distinct independent observations within one response",
                actual="duplicate tool name and arguments",
            )
        checker = getattr(tool_surface, "is_tool_multi_call_safe", None)
        unsafe = [
            name
            for _call_id, name, _arguments in calls
            if not callable(checker) or not checker(name)
        ]
        if unsafe:
            return _feedback(
                "multi_tool_call_not_safe",
                "one state-changing/control call, or only explicitly batch-safe observations",
                actual=", ".join(unsafe),
            )
        return None


def multi_tool_policy_from_env(
    environ: Mapping[str, str] | None = None,
) -> MultiToolPolicy:
    """Build a rollback-friendly policy from the documented environment."""
    env = os.environ if environ is None else environ
    mode = env.get("RPENT_VLLM_MULTI_TOOL_MODE", "read_only").strip().lower()
    if mode not in {"read_only", "disabled"}:
        raise ValueError("RPENT_VLLM_MULTI_TOOL_MODE must be 'read_only' or 'disabled'")
    raw_limit = env.get("RPENT_VLLM_MAX_MULTI_TOOL_CALLS", "8")
    try:
        limit = int(raw_limit)
    except ValueError as exc:
        raise ValueError("RPENT_VLLM_MAX_MULTI_TOOL_CALLS must be an integer") from exc
    return MultiToolPolicy(enabled=mode == "read_only", max_calls=limit)


def _feedback(code: str, expected: str, *, actual: str) -> dict[str, object]:
    return {
        "schema": "rpent.planner_repair_feedback.v1",
        "stage": "planner_call",
        "path": "/choices/0/message/tool_calls",
        "code": code,
        "expected": expected,
        "actual": actual,
        "repairable": True,
    }


__all__ = ["MultiToolPolicy", "ParsedToolCall", "multi_tool_policy_from_env"]
