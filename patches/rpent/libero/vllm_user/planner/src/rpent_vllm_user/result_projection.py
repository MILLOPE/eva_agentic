# Copyright 2026 The RPent Authors.

"""One bounded, structured planner view of Toolkit results."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

_PROJECTION_SCHEMA = "rpent.tool_result_projection.v1"
_IMAGE_RESULT_KEYS = frozenset(
    {
        "_image_bytes",
        "_image_cam_bytes",
        "_image_nav_bytes",
        "_image_wrist_bytes",
    }
)
_LARGE_TEXT_KEYS = frozenset({"content", "output", "stdout", "stderr", "text"})
_CORE_PRIORITY_KEYS = (
    "_finish",
    "schema",
    "error",
    "code",
    "status",
    "success",
    "safe",
    "safe_action",
    "safety_result",
    "hold",
    "hold_state",
    "hold_reason",
    "hold_confirmed",
    "lower_hold_confirmed",
    "proves_motion_or_hold",
    "authority",
    "motion_invoked",
    "safety_assessed",
    "terminal",
    "terminated",
    "truncated",
    "task_success_asserted",
    "stop_reason",
    "step",
    "step_idx",
    "name",
    "path",
    "size",
    "count",
    "exit_code",
    "summary",
    "instruction",
    "task_language",
    "task_memory_hints",
    "context_anchors",
    "task_progress",
    "fits_with_clearance",
    "state",
    "result",
    "execution",
    "safety",
    "episode_status",
    "receipt",
    "log",
)
_PRIORITY_SUFFIXES = (
    "_id",
    "_status",
    "_outcome",
    "_reason",
    "_error",
    "_safe",
    "_success",
    "_xyz",
    "_xy",
    "_bounds",
    "_extent",
    "_pos",
    "_quat",
    "_qpos",
    "_opening",
    "_count",
    "_size",
    "_truncated",
    "_confirmed",
    "_asserted",
    "_terminated",
    "_m",
)
_AUTHORITATIVE_KEYS = frozenset(
    {
        "_finish",
        "accepted",
        "authority",
        "T_execution_from_tool_tcp",
        "T_reference_from_tool_tcp",
        "candidate_T_reference_from_tool_tcp",
        "classification",
        "clearance_m",
        "code",
        "containment_margin_xy_m",
        "error",
        "evaluation_scope",
        "executed",
        "fits_with_clearance",
        "hold",
        "hold_confirmed",
        "hold_reason",
        "hold_state",
        "lower_hold_confirmed",
        "motion_invoked",
        "ordinal",
        "proves_motion_or_hold",
        "path_planning_eligible",
        "orientation_residual_deg",
        "position_residual_m",
        "reason_code",
        "reconciliation_required",
        "retry_allowed",
        "safe",
        "safe_action",
        "safety_assessed",
        "safety_result",
        "selected_candidate",
        "selected_candidate_ordinal",
        "semantic_caveat",
        "status",
        "stop_reason",
        "success",
        "task_success_asserted",
        "terminal",
        "terminated",
        "tool_tcp_xyz_reference_m",
        "truncated",
        "visible_scene_collision_free",
    }
)
_AUTHORITATIVE_SUFFIXES = (
    "_asserted",
    "_code",
    "_confirmed",
    "_error",
    "_id",
    "_outcome",
    "_reason",
    "_safe",
    "_status",
    "_success",
    "_terminated",
    "_truncated",
)
_MAX_PROJECTION_DEPTH = 8
_PROFILES = (
    (1_024, 32, 16),
    (256, 16, 8),
    (64, 8, 4),
    (64, 0, 1),
)


class ResultProjectionError(ValueError):
    """Authoritative result facts cannot fit a projection budget."""


def executed_result_unavailable(
    *,
    tool_name: str,
    error: BaseException | str,
    tool_call_id: str | None = None,
) -> dict[str, object]:
    """Describe a completed tool whose authoritative result cannot be projected."""
    result: dict[str, object] = {
        "schema": "rpent.executed_tool_result_unavailable.v1",
        "status": "executed_result_unavailable",
        "error": str(error),
        "tool": tool_name,
        "executed": True,
        "retry_allowed": False,
        "reconciliation_required": True,
    }
    if tool_call_id is not None:
        result["tool_call_id"] = tool_call_id
    return result


def bounded_tool_result_text(
    *,
    tool_name: str,
    result: Mapping[str, object],
    max_bytes: int,
    reason: str,
) -> str:
    """Render one valid-JSON, model-facing result within an exact byte limit."""
    _validate_inputs(
        tool_name=tool_name,
        result=result,
        max_bytes=max_bytes,
        reason=reason,
    )
    candidate = {
        str(key): value
        for key, value in result.items()
        if str(key) not in _IMAGE_RESULT_KEYS
    }
    rendered = _json_text(candidate)
    original_bytes = len(rendered.encode("utf-8"))
    if original_bytes <= max_bytes:
        return rendered
    return _project_mapping(
        tool_name=tool_name,
        result=candidate,
        original_bytes=original_bytes,
        max_bytes=max_bytes,
        reason=reason,
    )


def project_tool_messages(
    messages: list[dict[str, Any]],
    *,
    max_bytes: int,
    reason: str,
) -> list[dict[str, Any]]:
    """Project only oversized tool contents without changing exchange structure."""
    projected = [
        dict(message) if message.get("role") == "tool" else message
        for message in messages
    ]
    for message in projected:
        if message.get("role") != "tool":
            continue
        content = message.get("content")
        rendered = content if isinstance(content, str) else _json_text(content)
        if len(rendered.encode("utf-8")) <= max_bytes:
            continue
        decoded = _as_mapping(content)
        tool_name = str(message.get("name") or "unknown")
        if decoded is None:
            message["content"] = _project_plain_text(
                tool_name=tool_name,
                content=rendered,
                max_bytes=max_bytes,
                reason=reason,
            )
        else:
            message["content"] = bounded_tool_result_text(
                tool_name=tool_name,
                result=decoded,
                max_bytes=max_bytes,
                reason=reason,
            )
    return projected


def _project_mapping(
    *,
    tool_name: str,
    result: Mapping[str, object],
    original_bytes: int,
    max_bytes: int,
    reason: str,
) -> str:
    priority_keys = _ordered_priority_keys(result)
    for max_string_bytes, max_keys, max_items in _PROFILES:
        metadata = _projection_metadata(
            tool_name=tool_name,
            result=result,
            original_bytes=original_bytes,
            reason=reason,
        )
        projection: dict[str, object] = {"_rpent_projection": metadata}
        for key in priority_keys:
            projection[key] = _bounded_value(
                result[key],
                max_string_bytes=max_string_bytes,
                max_keys=max_keys,
                max_items=max_items,
                authoritative=_is_authoritative_key(key),
            )
        if _json_bytes(projection) > max_bytes:
            continue

        included = set(projection)
        optional_keys = [
            key
            for key in sorted(result)
            if key not in included and key not in _IMAGE_RESULT_KEYS
        ]
        for key in optional_keys[:16]:
            value = result[key]
            if key in _LARGE_TEXT_KEYS or _json_bytes(value) > 512:
                continue
            tentative = {**projection, key: _bounded_value(value)}
            if _json_bytes(tentative) <= max_bytes:
                projection[key] = tentative[key]

        omitted = [
            key
            for key in optional_keys
            if key not in projection and key != "_rpent_projection"
        ]
        if not _add_omission_metadata(projection, omitted, max_bytes=max_bytes):
            continue
        _add_text_excerpts(
            projection,
            result=result,
            omitted=omitted,
            max_bytes=max_bytes,
        )
        rendered = _json_text(projection)
        if len(rendered.encode("utf-8")) <= max_bytes:
            return rendered
    raise ResultProjectionError(
        "authoritative tool result fields exceed compacted result budget; "
        f"tool={tool_name}; max_bytes={max_bytes}"
    )


def _projection_metadata(
    *,
    tool_name: str,
    result: Mapping[str, object],
    original_bytes: int,
    reason: str,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "schema": _PROJECTION_SCHEMA,
        "tool": tool_name,
        "reason": reason,
        "original_utf8_bytes": original_bytes,
    }
    source = result.get("_rpent_projection")
    if isinstance(source, Mapping):
        source_bytes = source.get("original_utf8_bytes")
        if isinstance(source_bytes, int) and source_bytes > original_bytes:
            metadata["source_original_utf8_bytes"] = source_bytes
    return metadata


def _ordered_priority_keys(result: Mapping[str, object]) -> list[str]:
    keys = [key for key in _CORE_PRIORITY_KEYS if key in result]
    keys.extend(
        sorted(
            key
            for key in result
            if key not in keys and key.endswith(_PRIORITY_SUFFIXES)
        )
    )
    keys.extend(
        sorted(
            key
            for key, value in result.items()
            if key not in keys and _contains_authoritative_field(value)
        )
    )
    return keys


def _bounded_value(
    value: object,
    *,
    max_string_bytes: int = 256,
    max_keys: int = 16,
    max_items: int = 8,
    depth: int = 0,
    authoritative: bool = False,
) -> object:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return (
            value
            if authoritative
            else _head_tail_utf8(value, max_bytes=max_string_bytes)
        )
    if isinstance(value, (bytes, bytearray, memoryview)):
        if authoritative:
            raise ResultProjectionError(
                "authoritative binary tool result field is not JSON-representable"
            )
        return _value_descriptor(value)
    if depth >= _MAX_PROJECTION_DEPTH:
        if _contains_authoritative_field(value):
            raise ResultProjectionError(
                "authoritative tool result fields exceed projection depth"
            )
        descriptor = _value_descriptor(value)
        descriptor["omitted_at_depth"] = depth
        return descriptor
    if isinstance(value, Mapping):
        normalized = {str(key): item for key, item in value.items()}
        ordered = _ordered_priority_keys(normalized)
        ordered.extend(sorted(key for key in normalized if key not in ordered))
        authoritative = [
            key
            for key in ordered
            if _is_authoritative_key(key)
            or _contains_authoritative_field(normalized[key])
        ]
        optional = [key for key in ordered if key not in authoritative]
        selected = [
            *authoritative,
            *optional[: max(0, max_keys - len(authoritative))],
        ]
        bounded = {
            key: _bounded_value(
                normalized[key],
                max_string_bytes=max_string_bytes,
                max_keys=max_keys,
                max_items=max_items,
                depth=depth + 1,
                authoritative=_is_authoritative_key(key),
            )
            for key in selected
        }
        if len(ordered) > len(selected):
            bounded["_omitted_key_count"] = len(ordered) - len(selected)
        return bounded
    if isinstance(value, (list, tuple)):
        values = list(value)
        if len(values) > max_items and not authoritative:
            head = max_items // 2
            selected_indexes = {
                *range(head),
                *range(len(values) - (max_items - head), len(values)),
                *(
                    index
                    for index, item in enumerate(values)
                    if _contains_authoritative_field(item)
                ),
            }
            selected: list[object] = []
            previous = -1
            for index in sorted(selected_indexes):
                if index > previous + 1:
                    selected.append({"_omitted_item_count": index - previous - 1})
                selected.append(values[index])
                previous = index
            if previous < len(values) - 1:
                selected.append({"_omitted_item_count": len(values) - previous - 1})
            values = selected
        return [
            _bounded_value(
                item,
                max_string_bytes=max_string_bytes,
                max_keys=max_keys,
                max_items=max_items,
                depth=depth + 1,
            )
            for item in values
        ]
    return value


def _add_omission_metadata(
    projection: dict[str, object],
    omitted: list[str],
    *,
    max_bytes: int,
) -> bool:
    if not omitted:
        return True
    metadata = projection["_rpent_projection"]
    assert isinstance(metadata, dict)
    metadata["omitted_field_count"] = len(omitted)
    for keep in dict.fromkeys((min(16, len(omitted)), 8, 4, 1)):
        if keep > len(omitted):
            continue
        metadata["omitted_fields"] = omitted[:keep]
        if _json_bytes(projection) <= max_bytes:
            return True
    metadata.pop("omitted_fields", None)
    return _json_bytes(projection) <= max_bytes


def _is_authoritative_key(key: str) -> bool:
    return key in _AUTHORITATIVE_KEYS or key.endswith(_AUTHORITATIVE_SUFFIXES)


def _contains_authoritative_field(
    value: object,
    *,
    depth: int = 0,
    seen: set[int] | None = None,
) -> bool:
    if depth > 32:
        return True
    if not isinstance(value, (Mapping, list, tuple)):
        return False
    active = set() if seen is None else seen
    identity = id(value)
    if identity in active:
        return True
    active.add(identity)
    try:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if _is_authoritative_key(str(key)) or _contains_authoritative_field(
                    item, depth=depth + 1, seen=active
                ):
                    return True
            return False
        return any(
            _contains_authoritative_field(item, depth=depth + 1, seen=active)
            for item in value
        )
    finally:
        active.remove(identity)


def _add_text_excerpts(
    projection: dict[str, object],
    *,
    result: Mapping[str, object],
    omitted: list[str],
    max_bytes: int,
) -> None:
    texts = [
        (key, result[key])
        for key in omitted
        if key in _LARGE_TEXT_KEYS and isinstance(result[key], str)
    ]
    for index, (key, value) in enumerate(texts):
        remaining = len(texts) - index
        available = max_bytes - _json_bytes(projection)
        target = max(0, available // remaining - len(key.encode("utf-8")) - 16)
        excerpt_key = f"{key}_excerpt"
        excerpt = _largest_fitting_excerpt(
            projection=projection,
            key=excerpt_key,
            value=value,
            target_bytes=target,
            max_bytes=max_bytes,
        )
        if excerpt:
            projection[excerpt_key] = excerpt


def _project_plain_text(
    *,
    tool_name: str,
    content: str,
    max_bytes: int,
    reason: str,
) -> str:
    projection: dict[str, object] = {
        "_rpent_projection": {
            "schema": _PROJECTION_SCHEMA,
            "tool": tool_name,
            "reason": reason,
            "original_utf8_bytes": len(content.encode("utf-8")),
        }
    }
    excerpt = _largest_fitting_excerpt(
        projection=projection,
        key="text_excerpt",
        value=content,
        target_bytes=max_bytes,
        max_bytes=max_bytes,
    )
    if excerpt:
        projection["text_excerpt"] = excerpt
    return _json_text(projection)


def _largest_fitting_excerpt(
    *,
    projection: Mapping[str, object],
    key: str,
    value: str,
    target_bytes: int,
    max_bytes: int,
) -> str:
    low = 0
    high = min(len(value.encode("utf-8")), target_bytes)
    best = ""
    while low <= high:
        target = (low + high) // 2
        excerpt = _head_tail_utf8(value, max_bytes=target)
        if _json_bytes({**projection, key: excerpt}) <= max_bytes:
            best = excerpt
            low = target + 1
        else:
            high = target - 1
    return best


def _head_tail_utf8(value: str, *, max_bytes: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    marker = (f"\n...[middle omitted; original_utf8_bytes={len(encoded)}]...\n").encode(
        "utf-8"
    )
    if max_bytes <= len(marker):
        return marker[:max_bytes].decode("utf-8", errors="ignore")
    body = max_bytes - len(marker)
    head_size = body // 2
    tail_size = body - head_size
    head = encoded[:head_size].decode("utf-8", errors="ignore")
    tail = encoded[-tail_size:].decode("utf-8", errors="ignore")
    return head + marker.decode("utf-8") + tail


def _value_descriptor(value: object) -> dict[str, object]:
    descriptor: dict[str, object] = {"type": type(value).__name__}
    if isinstance(value, str):
        descriptor["utf8_bytes"] = len(value.encode("utf-8"))
    elif isinstance(value, (bytes, bytearray, memoryview)):
        descriptor["bytes"] = len(value)
    elif isinstance(value, (Mapping, list, tuple)):
        descriptor["items"] = len(value)
    return descriptor


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


def _validate_inputs(
    *,
    tool_name: object,
    result: object,
    max_bytes: object,
    reason: object,
) -> None:
    if not isinstance(tool_name, str) or not tool_name:
        raise ValueError("tool_name must be a non-empty string")
    if not isinstance(result, Mapping):
        raise TypeError("result must be a mapping")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 512:
        raise ValueError("max_bytes must be an integer of at least 512")
    if not isinstance(reason, str) or not reason:
        raise ValueError("reason must be a non-empty string")


def _json_text(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        default=str,
        separators=(",", ":"),
    )


def _json_bytes(value: object) -> int:
    return len(_json_text(value).encode("utf-8"))


__all__ = [
    "ResultProjectionError",
    "bounded_tool_result_text",
    "executed_result_unavailable",
    "project_tool_messages",
]
