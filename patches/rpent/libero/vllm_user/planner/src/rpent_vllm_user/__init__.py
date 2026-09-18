# Copyright 2026 The RPent Authors.
"""Optional signed-vLLM planner integration for RPent."""

from __future__ import annotations

from rpent.dashboard.events import DashboardEventSink
from rpent.utils.config import get_repo_root

from .client import SignedVllmClient, SignedVllmConfig, VllmChatResponse
from .history import ToolHistoryPolicy, history_policy_from_env
from .liveness import ToolLoopGuard, tool_loop_guard_from_env
from .multi_tool import MultiToolPolicy, multi_tool_policy_from_env
from .planner import ModelApiUtilsPlanner
from .workspace import WorkspaceProfile, workspace_profile_from_env


def create_planner(
    *,
    base_url: str | None,
    model: str | None,
    max_tokens: int,
    timeout_s: float,
    reasoning_effort: str,
    dashboard_events: DashboardEventSink,
    no_images: bool,
) -> ModelApiUtilsPlanner:
    """Build a planner from the documented ``RPENT_VLLM_*`` environment."""
    config = SignedVllmConfig.from_env(
        base_url_override=base_url,
        model_override=model,
        timeout_s=timeout_s,
    )
    client = SignedVllmClient(config)
    if config.expected_user_id is not None:
        client.validate_identity()
    return ModelApiUtilsPlanner(
        client=client,
        model=config.model,
        max_tokens=max_tokens,
        enable_thinking=reasoning_effort != "none",
        dashboard_events=dashboard_events,
        no_images=no_images,
        history_policy=history_policy_from_env(),
        multi_tool_policy=multi_tool_policy_from_env(),
        workspace_profile=workspace_profile_from_env(repo_root=get_repo_root()),
        timeout_s=timeout_s,
    )


__all__ = [
    "ModelApiUtilsPlanner",
    "MultiToolPolicy",
    "SignedVllmClient",
    "SignedVllmConfig",
    "ToolHistoryPolicy",
    "ToolLoopGuard",
    "VllmChatResponse",
    "WorkspaceProfile",
    "create_planner",
    "tool_loop_guard_from_env",
]
