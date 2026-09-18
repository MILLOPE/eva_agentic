# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Deterministic, bounded history views for repeated vLLM tool requests."""

from __future__ import annotations

import copy
import json
import os
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .result_projection import project_tool_messages

_MAX_POLICY_BYTES = 8_000_000
_MIN_POLICY_BYTES = 512
_DEFAULT_RECENT_TOOL_EXCHANGES = 12
_DEFAULT_SUMMARY_CHARS = 12_000
_DEFAULT_MAX_REQUEST_BYTES = 131_072
_DEFAULT_MAX_REQUEST_IMAGE_BYTES = 4 * 1024 * 1024
_DEFAULT_COMPACTED_TOOL_RESULT_BYTES = 12_000


class RequestBudgetExceeded(ValueError):
    """The required Planner context cannot fit its configured text budget."""


@dataclass(frozen=True, slots=True)
class ToolHistoryPolicy:
    """Bound tool history and every model-visible text request deterministically."""

    enabled: bool = True
    recent_tool_exchanges: int = _DEFAULT_RECENT_TOOL_EXCHANGES
    max_summary_chars: int = _DEFAULT_SUMMARY_CHARS
    max_request_bytes: int = _DEFAULT_MAX_REQUEST_BYTES
    compacted_tool_result_bytes: int = _DEFAULT_COMPACTED_TOOL_RESULT_BYTES
    max_request_image_bytes: int = _DEFAULT_MAX_REQUEST_IMAGE_BYTES

    @property
    def max_request_wire_bytes(self) -> int:
        """Upper bound for the exact JSON body after both component budgets."""
        return self.max_request_bytes + self.max_request_image_bytes

    def __post_init__(self) -> None:
        if isinstance(self.recent_tool_exchanges, bool) or not isinstance(
            self.recent_tool_exchanges, int
        ):
            raise TypeError("recent_tool_exchanges must be an integer")
        if self.recent_tool_exchanges < 1:
            raise ValueError("recent_tool_exchanges must be positive")
        if isinstance(self.max_summary_chars, bool) or not isinstance(
            self.max_summary_chars, int
        ):
            raise TypeError("max_summary_chars must be an integer")
        if self.max_summary_chars < 512:
            raise ValueError("max_summary_chars must be at least 512")
        for field_name in (
            "max_request_bytes",
            "max_request_image_bytes",
            "compacted_tool_result_bytes",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
            if not _MIN_POLICY_BYTES <= value <= _MAX_POLICY_BYTES:
                raise ValueError(
                    f"{field_name} must be from {_MIN_POLICY_BYTES} "
                    f"to {_MAX_POLICY_BYTES}"
                )
        if self.compacted_tool_result_bytes > self.max_request_bytes:
            raise ValueError(
                "compacted_tool_result_bytes cannot exceed max_request_bytes"
            )

    def prepare(
        self,
        messages: list[dict[str, Any]],
        *,
        request_size: Callable[[list[dict[str, Any]]], int] | None = None,
        recent_tool_exchanges_override: int | None = None,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Return an isolated request view within the hard text budget.

        The ordinary bounded mode keeps the configured number of recent tool
        results.  Independently of that mode, an oversized request retains the
        largest whole recent suffix that fits.  If the newest atomic exchange
        is itself too large, only its tool-result payloads receive a typed,
        deterministic projection.  System instructions, the initial user
        request, assistant tool calls, and exchange pairings are never silently
        truncated.
        """
        if recent_tool_exchanges_override is not None and (
            isinstance(recent_tool_exchanges_override, bool)
            or not isinstance(recent_tool_exchanges_override, int)
            or recent_tool_exchanges_override < 1
        ):
            raise ValueError("recent_tool_exchanges_override must be positive")
        request = copy.deepcopy(messages)
        tool_indexes = [
            index
            for index, message in enumerate(request)
            if message.get("role") == "tool"
        ]
        measure = request_size or _message_bytes
        recent_limit = (
            self.recent_tool_exchanges
            if recent_tool_exchanges_override is None
            else recent_tool_exchanges_override
        )
        configured_recent = (
            min(len(tool_indexes), recent_limit) if self.enabled else len(tool_indexes)
        )
        candidate, compacted = _compact_to_recent(
            request,
            recent_tool_exchanges=configured_recent,
            max_summary_chars=self.max_summary_chars,
        )
        if measure(candidate) <= self.max_request_bytes:
            return candidate, compacted
        if not tool_indexes:
            current = measure(request)
            raise RequestBudgetExceeded(
                "required planner context exceeds "
                f"max_request_bytes={self.max_request_bytes}; "
                f"request_bytes={current}; minimum_projected_bytes={current}"
            )

        for recent in range(configured_recent - 1, 0, -1):
            candidate, _ = _compact_to_recent(
                request,
                recent_tool_exchanges=recent,
                max_summary_chars=self.max_summary_chars,
            )
            if measure(candidate) <= self.max_request_bytes:
                return candidate, True

        # Keep the newest assistant/tool exchange structurally intact.  Reduce
        # only its result payloads, using progressively smaller deterministic
        # projections when fixed request overhead leaves less room than usual.
        summary_limits = (
            _descending_limits(self.max_summary_chars, minimum=512)
            if len(tool_indexes) > 1
            else (self.max_summary_chars,)
        )
        result_limits = _descending_limits(
            self.compacted_tool_result_bytes,
            minimum=512,
        )
        smallest: int | None = None
        for summary_limit in summary_limits:
            latest, _ = _compact_to_recent(
                request,
                recent_tool_exchanges=1,
                max_summary_chars=summary_limit,
            )
            for result_limit in result_limits:
                projected = project_tool_messages(
                    latest,
                    max_bytes=result_limit,
                    reason="request_budget",
                )
                size = measure(projected)
                smallest = size if smallest is None else min(smallest, size)
                if size <= self.max_request_bytes:
                    return projected, True

        # The earlier summary is useful but not required request structure.  A
        # final no-summary attempt avoids rejecting a request where the fixed
        # instructions plus newest atomic exchange do fit by themselves.
        latest, _ = _compact_to_recent(
            request,
            recent_tool_exchanges=1,
            max_summary_chars=self.max_summary_chars,
            include_summary=False,
        )
        for result_limit in result_limits:
            projected = project_tool_messages(
                latest,
                max_bytes=result_limit,
                reason="request_budget",
            )
            size = measure(projected)
            smallest = size if smallest is None else min(smallest, size)
            if size <= self.max_request_bytes:
                return projected, True

        current = measure(request)
        minimum = smallest if smallest is not None else current
        raise RequestBudgetExceeded(
            "required planner context exceeds "
            f"max_request_bytes={self.max_request_bytes}; "
            f"request_bytes={current}; minimum_projected_bytes={minimum}"
        )


def _compact_to_recent(
    request: list[dict[str, Any]],
    *,
    recent_tool_exchanges: int,
    max_summary_chars: int,
    include_summary: bool = True,
) -> tuple[list[dict[str, Any]], bool]:
    tool_indexes = [
        index for index, message in enumerate(request) if message.get("role") == "tool"
    ]
    if not tool_indexes or len(tool_indexes) <= recent_tool_exchanges:
        return request, False
    first_recent_tool = tool_indexes[-recent_tool_exchanges]
    cutoff = _exchange_start(request, first_recent_tool)
    base_end = next(
        (
            index
            for index, message in enumerate(request)
            if message.get("role") == "assistant"
        ),
        cutoff,
    )
    if cutoff <= base_end:
        return request, False
    prefix = list(request[:base_end])
    if include_summary:
        compacted = _history_summary(
            request[base_end:cutoff], max_chars=max_summary_chars
        )
        prefix.append({"role": "user", "content": compacted})
    return [*prefix, *request[cutoff:]], True


def history_policy_from_env(
    environ: Mapping[str, str] | None = None,
) -> ToolHistoryPolicy:
    """Build the history policy from the documented optional environment."""
    env = os.environ if environ is None else environ
    mode = env.get("RPENT_VLLM_HISTORY_MODE", "bounded").strip().lower()
    if mode not in {"bounded", "full"}:
        raise ValueError("RPENT_VLLM_HISTORY_MODE must be 'bounded' or 'full'")
    return ToolHistoryPolicy(
        enabled=mode == "bounded",
        recent_tool_exchanges=_env_int(
            env,
            "RPENT_VLLM_HISTORY_RECENT_TOOL_EXCHANGES",
            _DEFAULT_RECENT_TOOL_EXCHANGES,
            minimum=1,
        ),
        max_summary_chars=_env_int(
            env,
            "RPENT_VLLM_HISTORY_MAX_SUMMARY_CHARS",
            _DEFAULT_SUMMARY_CHARS,
            minimum=512,
        ),
        max_request_bytes=_env_int(
            env,
            "RPENT_VLLM_MAX_REQUEST_TEXT_BYTES",
            _DEFAULT_MAX_REQUEST_BYTES,
            minimum=_MIN_POLICY_BYTES,
            maximum=_MAX_POLICY_BYTES,
        ),
        max_request_image_bytes=_env_int(
            env,
            "RPENT_VLLM_MAX_REQUEST_IMAGE_BYTES",
            _DEFAULT_MAX_REQUEST_IMAGE_BYTES,
            minimum=_MIN_POLICY_BYTES,
            maximum=_MAX_POLICY_BYTES,
        ),
        compacted_tool_result_bytes=_env_int(
            env,
            "RPENT_VLLM_COMPACTED_TOOL_RESULT_MAX_BYTES",
            _DEFAULT_COMPACTED_TOOL_RESULT_BYTES,
            minimum=_MIN_POLICY_BYTES,
            maximum=_MAX_POLICY_BYTES,
        ),
    )


def _exchange_start(messages: list[dict[str, Any]], tool_index: int) -> int:
    for index in range(tool_index - 1, -1, -1):
        role = messages[index].get("role")
        if role == "assistant":
            return index
        # One assistant response may contain several independent read-only
        # calls.  Keep that response and all of its tool results atomic rather
        # than compacting from the middle of the batch.
    return tool_index


def _history_summary(messages: list[dict[str, Any]], *, max_chars: int) -> str:
    exchanges: list[str] = []
    counts: Counter[str] = Counter()
    pending_args: dict[str, object] = {}
    feedback = 0
    for message in messages:
        role = message.get("role")
        if role in {"repair_feedback", "tool_loop_feedback"}:
            feedback += 1
        if role == "assistant":
            calls = message.get("tool_calls")
            if isinstance(calls, list):
                for call in calls:
                    if not isinstance(call, Mapping):
                        continue
                    function = call.get("function")
                    if not isinstance(function, Mapping):
                        continue
                    call_id = str(call.get("id") or "")
                    pending_args[call_id] = function.get("arguments", {})
        if role != "tool":
            continue
        name = str(message.get("name") or "unknown")
        counts[name] += 1
        call_id = str(message.get("tool_call_id") or "")
        args = _compact_json(pending_args.pop(call_id, {}), max_chars=180)
        result = _compact_result(message.get("content"))
        exchanges.append(f"#{len(exchanges) + 1} {name} args={args} result={result}")
    header = (
        "[deterministic earlier tool history; no model-generated summary]\n"
        f"exchanges={len(exchanges)} feedback={feedback} "
        f"tool_counts={json.dumps(dict(sorted(counts.items())), separators=(',', ':'))}\n"
    )
    anchor_budget = min(4000, max(0, max_chars - len(header)) // 2)
    anchors = _history_anchors(messages, max_chars=anchor_budget)
    if anchors:
        header += anchors + "\n"
    budget = max_chars - len(header)
    kept: list[str] = []
    used = 0
    for line in reversed(exchanges):
        size = len(line) + 1
        if kept and used + size > budget:
            break
        if size > budget:
            line = line[: max(0, budget - 24)] + "...[truncated]"
            size = len(line) + 1
        kept.append(line)
        used += size
    kept.reverse()
    omitted = len(exchanges) - len(kept)
    if omitted:
        header += f"older_exchange_lines_omitted={omitted}\n"
    return header + "\n".join(kept)


def _history_anchors(messages: list[dict[str, Any]], *, max_chars: int) -> str:
    """Keep authoritative task context when the chronological tail is pruned."""
    if max_chars <= 0:
        return ""

    task_context: dict[str, object] | None = None
    hinted_paths: set[str] = set()
    decoded_tools: list[tuple[str, Mapping[str, object]]] = []
    for message in messages:
        if message.get("role") != "tool":
            continue
        result = _as_mapping(message.get("content"))
        if result is None:
            continue
        name = str(message.get("name") or "unknown")
        decoded_tools.append((name, result))
        language = result.get("task_language")
        if task_context is None and isinstance(language, str) and language.strip():
            task_context = {"task_language": language}
            state = result.get("state")
            if isinstance(state, Mapping) and isinstance(
                state.get("object_names"), list
            ):
                task_context["object_names"] = state["object_names"]
            hints = result.get("task_memory_hints")
            if isinstance(hints, Mapping):
                compact_hints: dict[str, object] = {}
                for key in ("routing_status", "match_policy", "instruction"):
                    if key in hints:
                        compact_hints[key] = hints[key]
                for key in ("suite_memories", "task_references"):
                    entries = hints.get(key)
                    if not isinstance(entries, list):
                        continue
                    fields = (
                        ("path",)
                        if key == "suite_memories"
                        else ("audit_path", "recipe_path", "markdown_path", "path")
                    )
                    paths: list[str] = []
                    for entry in entries:
                        if not isinstance(entry, Mapping):
                            continue
                        for field in fields:
                            value = entry.get(field)
                            if isinstance(value, str) and value:
                                paths.append(value)
                    if paths:
                        compact_hints[key] = paths
                        hinted_paths.update(paths)
                if compact_hints:
                    task_context["task_memory_hints"] = compact_hints

    lines: list[str] = []
    if task_context is not None:
        lines.append(
            "task_anchor=" + _compact_json(task_context, max_chars=min(1000, max_chars))
        )

    memories: list[Mapping[str, object]] = []
    anchored_paths: set[str] = set()
    for _name, result in decoded_tools:
        anchors = result.get("context_anchors")
        if not isinstance(anchors, list):
            continue
        for anchor in anchors:
            if not isinstance(anchor, Mapping):
                continue
            path = str(anchor.get("path") or "")
            if (
                path
                and path not in anchored_paths
                and anchor.get("present") is True
                and isinstance(anchor.get("content"), str)
            ):
                memories.append(anchor)
                anchored_paths.add(path)

    # Preserve compatibility with transcripts produced before the shared
    # atomic context protocol. New tools opt in through ``context_anchors``;
    # this transport does not need to understand their domain semantics.
    for name, result in decoded_tools:
        path = str(result.get("path") or "")
        if (
            name == "read_text_file"
            and not result.get("error")
            and path in hinted_paths
            and path not in anchored_paths
            and isinstance(result.get("content"), str)
        ):
            memories.append(result)
            anchored_paths.add(path)
    remaining = max_chars - sum(len(line) + 1 for line in lines)
    for index, result in enumerate(memories):
        if remaining <= 0:
            break
        slots = len(memories) - index
        line_budget = max(0, remaining // slots - 1)
        rendered = _compact_json(
            {
                "path": result["path"],
                "content": result["content"],
            },
            max_chars=line_budget,
        )
        line = "task_memory_anchor=" + rendered
        if len(line) > remaining:
            line = _shorten(line, remaining)
        lines.append(line)
        remaining -= len(line) + 1
    return "\n".join(lines)


def _as_mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, str):
        return None
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, Mapping) else None


def _descending_limits(value: int, *, minimum: int) -> tuple[int, ...]:
    limits: list[int] = []
    current = value
    while current > minimum:
        limits.append(current)
        current = max(minimum, current // 2)
    limits.append(minimum)
    return tuple(dict.fromkeys(limits))


def _message_bytes(messages: list[dict[str, Any]]) -> int:
    return len(
        json.dumps(
            messages,
            ensure_ascii=False,
            default=str,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )


def _compact_result(value: object) -> str:
    candidate = value
    if isinstance(value, str):
        try:
            candidate = json.loads(value)
        except json.JSONDecodeError:
            return _shorten(value, 300)
    if not isinstance(candidate, Mapping):
        return _compact_json(candidate, max_chars=300)
    selected: dict[str, object] = {}
    for key in (
        "error",
        "code",
        "status",
        "success",
        "terminated",
        "truncated",
        "eval_success",
        "stop_reason",
        "step",
        "step_idx",
        "name",
        "summary",
        "found",
        "score",
        "box",
        "world_xyz",
        "world_center_xyz",
        "world_bounds_xyz",
        "world_extent_xyz",
        "reference_eef_xyz",
        "eef_to_mask_center_xyz",
        "target_eef_xy",
        "minimum_lateral_eef_z",
        "held_eef_to_center_xy",
        "held_eef_to_center_z",
        "held_eef_to_center_xyz",
        "held_eef_offset_norm_m",
        "max_held_eef_offset_m",
        "held_extent_xy",
        "destination_center_xy",
        "destination_top_z",
        "destination_bounds_xy",
        "containment_margin_xy_m",
        "fits_with_clearance",
        "clearance_m",
        "held_gripper_opening",
        "semantic_caveat",
        "instruction",
        "world_error",
        "pixel",
        "center_xyz",
        "median_xyz",
        "n_valid",
        "camera",
        "resolution",
        "mode",
        "row_range",
        "col_range",
        "z_band",
        "task_progress",
        "transition_checkpoint",
        "transition_evidence_accepted",
        "transition_evidence_reason",
        "vla_desync",
        "robocasa_terminated",
    ):
        if key in candidate:
            selected[key] = candidate[key]
    for container_name in ("episode_status", "state", "result"):
        nested = candidate.get(container_name)
        if not isinstance(nested, Mapping):
            continue
        nested_selected = {
            key: nested[key]
            for key in (
                "error",
                "status",
                "success",
                "terminated",
                "eval_success",
                "stop_reason",
                "step_idx",
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
                "robot0_base_pos",
                "robot0_base_quat",
                "robot_state",
                "final_eef_pos",
                "target_xyz",
                "final_dist_m",
                "residual_xyz",
                "position_reached",
                "position_status",
                "step_budget_exhausted",
                "downward_contact_or_limit",
                "position_status_note",
                "final_gripper_opening",
                "min_gripper_opening",
                "peak_gripper_opening",
                "peak_lift_m",
                "diagnostics",
            )
            if key in nested
        }
        episode = nested.get("episode_status")
        if isinstance(episode, Mapping):
            nested_selected["episode_status"] = {
                key: episode[key]
                for key in ("eval_success", "take_action_cnt", "native_actions")
                if key in episode
            }
        if nested_selected:
            selected[container_name] = nested_selected
    log = candidate.get("log")
    if isinstance(log, Mapping):
        compact_log: dict[str, object] = {}
        if "command" in log:
            compact_log["command"] = log["command"]
        effect = log.get("result")
        if isinstance(effect, Mapping):
            compact_log["result"] = {
                key: effect[key]
                for key in (
                    "name",
                    "success",
                    "terminated",
                    "truncated",
                    "final_eef_pos",
                    "target_xyz",
                    "final_dist_m",
                    "residual_xyz",
                    "position_reached",
                    "position_status",
                    "step_budget_exhausted",
                    "downward_contact_or_limit",
                    "position_status_note",
                    "final_gripper_opening",
                    "min_gripper_opening",
                    "peak_gripper_opening",
                    "peak_lift_m",
                    "diagnostics",
                )
                if key in effect
            }
        if compact_log:
            selected["log"] = compact_log
    return _compact_json(selected or candidate, max_chars=600)


def _compact_json(value: object, *, max_chars: int) -> str:
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return _shorten(value, max_chars)
        value = decoded
    rendered = json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
    return _shorten(rendered, max_chars)


def _shorten(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max(0, max_chars - 14)] + "...[truncated]"


def _env_int(
    env: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return value


__all__ = [
    "RequestBudgetExceeded",
    "ToolHistoryPolicy",
    "history_policy_from_env",
]
