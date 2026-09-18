"""Unit tests for eva_agentic.probing (capacity-probe input helpers)."""

from __future__ import annotations

import pytest

from eva_agentic.probing import (
    build_cases,
    parse_task_spec,
    render_experiment,
    render_profile,
    write_cases_jsonl,
)


class TestParseTaskSpec:
    def test_lists_and_ranges(self) -> None:
        assert parse_task_spec("1,2,4-6") == (1, 2, 4, 5, 6)

    def test_unsorted_and_duplicates_normalized(self) -> None:
        assert parse_task_spec("5,3,5,1-2") == (1, 2, 3, 5)

    def test_single_task(self) -> None:
        assert parse_task_spec("0") == (0,)

    @pytest.mark.parametrize("spec", ["", "1-2-3", "5-2", "a", ",1", "1,"])
    def test_invalid(self, spec: str) -> None:
        with pytest.raises(ValueError):
            parse_task_spec(spec)


class TestBuildCases:
    def test_order_and_fields(self) -> None:
        cases = build_cases("libero_object", [0, 2], [0, 1], max_episode_steps=500)
        assert [c.case_id for c in cases] == [
            "libero_object:t0:s0",
            "libero_object:t0:s1",
            "libero_object:t2:s0",
            "libero_object:t2:s1",
        ]
        assert cases[0].task_id == "libero_object:0"
        assert cases[0].initialization == {
            "suite": "libero_object",
            "task_id": 0,
            "max_episode_steps": 500,
        }

    def test_duplicate_ids_rejected(self) -> None:
        with pytest.raises(ValueError):
            build_cases("libero_object", [1, 1], [0, 0], max_episode_steps=500)

    @pytest.mark.parametrize("seeds", [[0, "1"], [True]])
    def test_invalid_seeds(self, seeds) -> None:
        with pytest.raises(ValueError):
            build_cases("libero_object", [0], seeds, max_episode_steps=500)

    @pytest.mark.parametrize("steps", [0, -1, 2.5])
    def test_invalid_steps(self, steps) -> None:
        with pytest.raises(ValueError):
            build_cases("libero_object", [0], [0], max_episode_steps=steps)

    def test_empty_tasks(self) -> None:
        with pytest.raises(ValueError):
            build_cases("libero_object", [], [0], max_episode_steps=500)


class TestRenderExperiment:
    def test_substitutes_and_returns_mapping(self) -> None:
        template = {
            "name": "probe-{TAG}",
            "cases_file": "{CASES_FILE}",
            "execution": {"max_jobs": "{MAX_JOBS}"},
            "keep": ["a", "{TAG}"],
        }
        rendered = render_experiment(
            template, tag="t1", cases_file="/tmp/c.jsonl", max_jobs=4
        )
        assert rendered["name"] == "probe-t1"
        assert rendered["cases_file"] == "/tmp/c.jsonl"
        assert rendered["execution"]["max_jobs"] == 4
        assert rendered["keep"] == ["a", "t1"]
        assert rendered["execution"] is not template["execution"]

    def test_unsubstituted_known_placeholder_rejected(self) -> None:
        template = {
            "cases_file": "{CASES_FILE}",
            "name": "missing-{TAG}",
            "execution": {"max_jobs": "{MAX_JOBS}"},
        }
        # Missing {TAG} value must be rejected even if not provided as a token.
        with pytest.raises(ValueError):
            render_experiment(template, tag="", cases_file="/x", max_jobs=1)

    def test_unknown_token_left_as_literal(self) -> None:
        template = {
            "cases_file": "{CASES_FILE}",
            "execution": {"max_jobs": "{MAX_JOBS}"},
            "missing": "{UNKNOWN}",
        }
        rendered = render_experiment(template, tag="t", cases_file="/x", max_jobs=1)
        assert rendered["missing"] == "{UNKNOWN}"

    @pytest.mark.parametrize("kwargs", [
        {"tag": "", "cases_file": "/x", "max_jobs": 2},
        {"tag": "t", "cases_file": "", "max_jobs": 2},
        {"tag": "t", "cases_file": "/x", "max_jobs": 0},
        {"tag": "t", "cases_file": "/x", "max_jobs": 1.5},
    ])
    def test_invalid_arguments(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            render_experiment({"name": "{TAG}"}, **kwargs)


class TestRenderProfile:
    def test_expands_slots_for_required_kinds(self) -> None:
        base = {"resource_slots": {"gpu": [0]}, "frameworks": {"k": {"x": 1}}}
        rendered = render_profile(base, {"gpu": 1}, max_jobs=3)
        assert rendered["resource_slots"]["gpu"] == [0, 1, 2]
        assert rendered["frameworks"] == {"k": {"x": 1}}

    def test_slots_scale_with_requirement_per_job(self) -> None:
        rendered = render_profile({"resource_slots": {}}, {"gpu": 2}, max_jobs=3)
        assert rendered["resource_slots"]["gpu"] == list(range(6))

    def test_preserves_unrelated_kinds(self) -> None:
        base = {"resource_slots": {"other": [5], "gpu": [0]}}
        rendered = render_profile(base, {"gpu": 1}, max_jobs=2)
        assert rendered["resource_slots"]["other"] == [5]

    def test_invalid(self) -> None:
        with pytest.raises(ValueError):
            render_profile({"resource_slots": {}}, {"gpu": 0}, max_jobs=2)
        with pytest.raises(ValueError):
            render_profile({"resource_slots": {}}, {"gpu": 1}, max_jobs=0)


class TestWriteCases:
    def test_round_trip(self, tmp_path) -> None:
        cases = build_cases("libero_object", [0, 1], [0], max_episode_steps=500)
        path = write_cases_jsonl(tmp_path / "cases.jsonl", cases)
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == len(cases)
        for case, line in zip(cases, lines):
            assert case.to_dict() == __import__("json").loads(line)
