# Copyright 2026 The RPent Authors.

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from rpent.dashboard.events import NullDashboardEventSink
from rpent.planner.base import build_planner
from rpent.planner.runtime import PlannerRuntime
from rpent_vllm_user.compat import DEFAULT_REASONING_EFFORT


@pytest.mark.parametrize("effort", [None, "none", "high"])
def test_core_lazily_builds_optional_vllm_planner(monkeypatch, tmp_path, effort) -> None:
    class MarkerPlanner:
        def solve(self, **kwargs):
            raise AssertionError(f"unexpected solve: {kwargs}")

    marker = MarkerPlanner()
    fake = ModuleType("rpent_vllm_user")
    captured = {}

    def create_planner(**kwargs):
        captured.update(kwargs)
        return marker

    fake.create_planner = create_planner
    monkeypatch.setitem(sys.modules, "rpent_vllm_user", fake)

    result = build_planner(
        "vllm_user",
        output_dir=tmp_path,
        recipe_tag="test",
        robot_name="libero",
        base_url="http://service:8100",
        model="default",
        max_tokens=123,
        planner_timeout_s=45,
        dashboard_events=NullDashboardEventSink(),
        no_images=True,
        **({"reasoning_effort": effort} if effort is not None else {}),
    )

    assert isinstance(result, PlannerRuntime)
    assert result.inner is marker
    assert captured["base_url"] == "http://service:8100"
    assert captured["max_tokens"] == 123
    assert captured["timeout_s"] == 45
    assert captured["no_images"] is True
    assert captured["reasoning_effort"] == (effort or DEFAULT_REASONING_EFFORT)
