"""Unit tests for the read-only visualization aggregation + SVG rendering."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eva_agentic.experiment import resolve_experiment
from eva_agentic.schema import AttemptStatus, EpisodeResult, OutcomeStatus
from eva_agentic.store import commit_attempt, initialize_run, load_run, prepare_attempt
from eva_agentic.viz import collect_rows, discover_runs, render_charts, render_report, write_summary


def _make_run(root: Path, run_name: str) -> Path:
    cases = root / "cases.jsonl"
    cases.write_text(
        "".join(
            json.dumps({
                "case_id": f"libero_object:t{task}:s{seed}",
                "task_id": f"libero_object:{task}",
                "seed": seed,
                "initialization": {"suite": "libero_object", "task_id": task, "max_episode_steps": 500},
            }) + "\n"
            for task in (0, 1) for seed in (0, 1)
        ),
        encoding="utf-8",
    )
    experiment = resolve_experiment({
        "schema_version": 1, "name": "viz", "benchmark": "fake", "cases_file": str(cases),
        "participants": ["fake"],
        "protocol": {"track": "test", "phase": "eval", "memory_policy": "frozen", "scoring": "native"},
        "execution": {"mode": "debug", "max_jobs": 1, "max_infrastructure_retries": 0, "job_timeout_s": 1},
        "artifacts": {"video": "off", "trace": "off"},
    })
    run_dir = initialize_run(root / "runs", experiment, run_name)
    plan = load_run(run_dir).plan
    outcomes = [OutcomeStatus.SUCCESS, OutcomeStatus.TASK_FAILURE, OutcomeStatus.SUCCESS, OutcomeStatus.TASK_FAILURE]
    for job, status in zip(plan.jobs, outcomes):
        attempt = prepare_attempt(run_dir, job, 1)
        success = True if status is OutcomeStatus.SUCCESS else False
        result = EpisodeResult(
            job.cases[0].case_id, status, success,
            success_source="rpent.audit.terminated" if success else None,
            error_source=None if success else "rpent.task",
        )
        commit_attempt(attempt, AttemptStatus.COMPLETED, {"duration_s": 55.0}, (result,))
    return run_dir


class VizTest(unittest.TestCase):
    def test_aggregation_and_charts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = _make_run(root, "viz-0001")
            runs_root = run_dir.parent

            self.assertEqual(discover_runs(runs_root), [run_dir])
            rows = collect_rows([run_dir])
            self.assertEqual(len(rows), 4)
            by_case = {row["case_id"]: row for row in rows}
            success = by_case["libero_object:t0:s0"]
            failure = by_case["libero_object:t0:s1"]
            self.assertEqual(success["status"], OutcomeStatus.SUCCESS.value)
            self.assertIs(success["task_success"], True)
            self.assertEqual(success["success_source"], "rpent.audit.terminated")
            self.assertEqual(failure["status"], OutcomeStatus.TASK_FAILURE.value)
            self.assertEqual(failure["error_source"], "rpent.task")
            self.assertEqual(success["duration_s"], 55.0)
            self.assertEqual(success["task_index"], 0)
            self.assertEqual(success["seed"], 0)

            out_dir = root / "viz-out"
            write_summary(rows, out_dir / "summary.csv", out_dir / "summary.json")
            self.assertTrue((out_dir / "summary.csv").is_file())
            self.assertTrue((out_dir / "summary.json").is_file())

            charts = render_charts(rows, out_dir)
            for name in ("success_rate.svg", "status_heatmap.svg", "duration.svg"):
                self.assertTrue(charts[name].is_file())
            self.assertTrue(render_report(rows, out_dir).is_file())


if __name__ == "__main__":
    unittest.main()
