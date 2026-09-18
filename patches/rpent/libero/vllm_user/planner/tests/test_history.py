# Copyright 2026 The RPent Authors.

from __future__ import annotations

import json

import pytest

from rpent_vllm_user.history import (
    RequestBudgetExceeded,
    ToolHistoryPolicy,
    history_policy_from_env,
)


def _call(call_id: str) -> dict[str, object]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "observe", "arguments": "{}"},
            }
        ],
    }


def _message_bytes(messages: list[dict[str, object]]) -> int:
    return len(
        json.dumps(
            messages,
            ensure_ascii=False,
            default=str,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )


def test_request_budget_keeps_latest_exchange_and_projects_large_result() -> None:
    messages = [
        {"role": "system", "content": "safety rules"},
        {"role": "user", "content": "move only when safe"},
        _call("old"),
        {
            "role": "tool",
            "tool_call_id": "old",
            "name": "observe",
            "content": json.dumps({"step": 1, "content": "old" * 2_000}),
        },
        _call("latest"),
        {
            "role": "tool",
            "tool_call_id": "latest",
            "name": "observe",
            "content": json.dumps(
                {
                    "status": "HOLD",
                    "safe": False,
                    "safe_action": {"kind": "HOLD", "reason": "collision"},
                    "task_id": "task-1",
                    "execution_id": "exec-1",
                    "world_xyz": [0.1, 0.2, 0.3],
                    "content": "head-" + "测" * 8_000 + "-tail",
                },
                ensure_ascii=False,
            ),
        },
    ]
    policy = ToolHistoryPolicy(
        max_request_bytes=2_500,
        compacted_tool_result_bytes=1_200,
        max_summary_chars=512,
    )

    prepared, compacted = policy.prepare(messages)

    assert compacted is True
    assert _message_bytes(prepared) <= policy.max_request_bytes
    assert not any(message.get("tool_call_id") == "old" for message in prepared)
    latest_assistant = next(
        message for message in prepared if message.get("role") == "assistant"
    )
    assert latest_assistant["tool_calls"][0]["id"] == "latest"
    latest_tool = next(message for message in prepared if message.get("role") == "tool")
    projected = json.loads(latest_tool["content"])
    assert projected["_rpent_projection"]["reason"] == "request_budget"
    assert projected["status"] == "HOLD"
    assert projected["safe"] is False
    assert projected["safe_action"] == {"kind": "HOLD", "reason": "collision"}
    assert projected["task_id"] == "task-1"
    assert projected["execution_id"] == "exec-1"
    assert projected["world_xyz"] == [0.1, 0.2, 0.3]
    assert projected["content_excerpt"].startswith("head-")
    assert projected["content_excerpt"].endswith("-tail")


def test_full_history_mode_still_obeys_hard_request_budget() -> None:
    messages: list[dict[str, object]] = [{"role": "user", "content": "task"}]
    for index in range(3):
        call_id = f"call-{index}"
        messages.extend(
            [
                _call(call_id),
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": "observe",
                    "content": json.dumps(
                        {"step": index, "content": str(index) * 4_000}
                    ),
                },
            ]
        )
    policy = ToolHistoryPolicy(
        enabled=False,
        max_request_bytes=2_500,
        compacted_tool_result_bytes=1_000,
        max_summary_chars=512,
    )

    prepared, compacted = policy.prepare(messages)

    assert compacted is True
    assert _message_bytes(prepared) <= policy.max_request_bytes
    assert any(message.get("tool_call_id") == "call-2" for message in prepared)
    assert not any(message.get("tool_call_id") == "call-0" for message in prepared)


def test_request_budget_never_splits_latest_multi_tool_exchange() -> None:
    latest = _call("latest-a")
    latest["tool_calls"].append(_call("latest-b")["tool_calls"][0])
    messages = [
        {"role": "user", "content": "task"},
        _call("old"),
        {
            "role": "tool",
            "tool_call_id": "old",
            "name": "observe",
            "content": json.dumps({"content": "old" * 2_000}),
        },
        latest,
        {
            "role": "tool",
            "tool_call_id": "latest-a",
            "name": "observe",
            "content": json.dumps({"status": "ok", "content": "a" * 5_000}),
        },
        {
            "role": "tool",
            "tool_call_id": "latest-b",
            "name": "observe",
            "content": json.dumps({"status": "HOLD", "content": "b" * 5_000}),
        },
    ]
    policy = ToolHistoryPolicy(
        max_request_bytes=3_500,
        compacted_tool_result_bytes=800,
        max_summary_chars=512,
    )

    prepared, compacted = policy.prepare(messages)

    assert compacted is True
    assert _message_bytes(prepared) <= policy.max_request_bytes
    latest_call = next(
        message for message in prepared if message.get("role") == "assistant"
    )
    assert [call["id"] for call in latest_call["tool_calls"]] == [
        "latest-a",
        "latest-b",
    ]
    assert [
        message["tool_call_id"] for message in prepared if message.get("role") == "tool"
    ] == ["latest-a", "latest-b"]


def test_request_budget_fails_before_model_when_required_context_cannot_fit() -> None:
    policy = ToolHistoryPolicy(
        max_request_bytes=1_024,
        compacted_tool_result_bytes=512,
        max_summary_chars=512,
    )
    messages = [
        {"role": "system", "content": "x" * 2_000},
        {"role": "user", "content": "required task"},
    ]

    with pytest.raises(RequestBudgetExceeded, match="required planner context"):
        policy.prepare(messages)


def test_request_budget_drops_optional_old_summary_before_failing() -> None:
    messages = [
        {"role": "system", "content": "safety rules"},
        {"role": "user", "content": "task"},
        _call("old"),
        {
            "role": "tool",
            "tool_call_id": "old",
            "name": "observe",
            "content": json.dumps({"content": "old" * 2_000}),
        },
        _call("latest"),
        {
            "role": "tool",
            "tool_call_id": "latest",
            "name": "observe",
            "content": json.dumps({"status": "HOLD", "safe": False}),
        },
    ]
    policy = ToolHistoryPolicy(
        max_request_bytes=600,
        compacted_tool_result_bytes=512,
        max_summary_chars=512,
    )

    prepared, compacted = policy.prepare(messages)

    assert compacted is True
    assert _message_bytes(prepared) <= 600
    assert not any(
        isinstance(message.get("content"), str)
        and message["content"].startswith("[deterministic earlier tool history")
        for message in prepared
    )
    assert [
        message["tool_call_id"] for message in prepared if message.get("role") == "tool"
    ] == ["latest"]


def test_history_policy_environment_exposes_bounded_text_controls() -> None:
    policy = history_policy_from_env(
        {
            "RPENT_VLLM_HISTORY_MODE": "full",
            "RPENT_VLLM_MAX_REQUEST_TEXT_BYTES": "4096",
            "RPENT_VLLM_MAX_REQUEST_IMAGE_BYTES": "2048",
            "RPENT_VLLM_COMPACTED_TOOL_RESULT_MAX_BYTES": "1024",
        }
    )

    assert policy.enabled is False
    assert policy.max_request_bytes == 4_096
    assert policy.max_request_image_bytes == 2_048
    assert policy.max_request_wire_bytes == 6_144
    assert policy.compacted_tool_result_bytes == 1_024

    with pytest.raises(ValueError, match="at most"):
        history_policy_from_env({"RPENT_VLLM_MAX_REQUEST_TEXT_BYTES": "8000001"})


def test_history_policy_preserves_existing_positional_argument_order() -> None:
    policy = ToolHistoryPolicy(True, 3, 1_000, 4_096, 1_024)

    assert policy.recent_tool_exchanges == 3
    assert policy.max_summary_chars == 1_000
    assert policy.max_request_bytes == 4_096
    assert policy.compacted_tool_result_bytes == 1_024
    assert policy.max_request_image_bytes == 4 * 1024 * 1024
