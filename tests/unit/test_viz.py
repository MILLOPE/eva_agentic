"""Unit tests for the read-only visualization aggregation + SVG rendering."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eva_agentic.experiment import resolve_experiment
from eva_agentic.schema import AttemptStatus, EpisodeResult, OutcomeStatus
from eva_agentic.store import commit_attempt, initialize_run, load_run, prepare_attempt
from eva_agentic.viz import collect_rows, discover_runs, render_charts, render_report, visualize_runs, write_provenance, write_summary


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
            chart_names = list(charts)  # svg (no seaborn in eval venv) or png
            self.assertTrue(chart_names, "expected at least one chart")
            self.assertTrue(render_report(rows, out_dir).is_file())
            self.assertTrue(write_provenance(out_dir, "svg").is_file())

    def test_visualize_runs_writes_in_place(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs_root = root / "runs"
            run_a = _make_run(root, "viz-0001")
            run_b = _make_run(root, "viz-0002")

            aggregate = root / "agg"
            result = visualize_runs([run_a, run_b], aggregate_out_dir=aggregate)

            self.assertEqual(result["renderer"], "svg")  # eval venv has no seaborn
            self.assertEqual(len(result["per_run"]), 2)
            for entry in result["per_run"]:
                out = Path(entry["out_dir"])
                # in-place under the run dir
                self.assertEqual(out.parent, root / "runs" / entry["run_id"])
                for name in ("summary.csv", "summary.json", "report.html", "provenance.json"):
                    self.assertTrue((out / name).is_file(), f"{name} missing in {out}")
                self.assertGreaterEqual(len(entry["charts"]), 1)

            aggregate_result = result["aggregate"]
            self.assertEqual(sorted(aggregate_result["runs"]), ["viz-0001", "viz-0002"])
            for name in ("summary.csv", "summary.json", "report.html", "provenance.json"):
                self.assertTrue((aggregate / name).is_file(), f"aggregate {name} missing")
        assert runs_root  # keep reference


if __name__ == "__main__":
    unittest.main()
