"""Unit tests for the read-only visualization aggregation + SVG rendering."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from eva_agentic.experiment import resolve_experiment
from eva_agentic.schema import AttemptStatus, EpisodeResult, OutcomeStatus
from eva_agentic.store import commit_attempt, initialize_run, load_run, prepare_attempt
from eva_agentic.viz import (
    CASE_METRIC_FIELDS,
    SUMMARY_FIELDS,
    aggregate_metrics,
    collect_rows,
    discover_runs,
    pages_for,
    render_charts,
    render_report,
    suite_of,
    visualize_runs,
    write_case_metrics,
    write_provenance,
    write_summary,
)


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
            native = run_dir / "participants" / "fake" / "jobs" / "job_0001" / "attempts" / "0001" / "native"
            native.mkdir(parents=True)
            (native / "transcript_libero_object_t0_s0.json").write_text(json.dumps({
                "model": "fake-model",
                "finish": {"status": "success", "summary": "finished"},
                "messages": [{"tool_calls": [{"function": {"name": "pi0_pick"}}, {"function": {"name": "segment"}}]}],
                "stats": {
                    "model_requests": 7, "turns_used": 7, "tool_calls": 4,
                    "tool_execution_errors": 1, "loop_feedbacks": 2,
                    "proposal_repairs": 1, "planner_loop_stopped": 0,
                    "planner_runtime": "shared", "total_input_tokens": 120,
                    "total_output_tokens": 34, "model_elapsed_s": 3.5,
                },
            }), encoding="utf-8")
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
            write_case_metrics(rows, out_dir / "case_metrics.csv", out_dir / "case_metrics.json")
            self.assertTrue((out_dir / "summary.csv").is_file())
            self.assertTrue((out_dir / "summary.json").is_file())
            self.assertTrue((out_dir / "case_metrics.csv").is_file())
            self.assertTrue((out_dir / "case_metrics.json").is_file())

            with (out_dir / "summary.csv").open(newline="", encoding="utf-8") as handle:
                self.assertEqual(next(csv.reader(handle)), SUMMARY_FIELDS)
            with (out_dir / "case_metrics.csv").open(newline="", encoding="utf-8") as handle:
                self.assertEqual(next(csv.reader(handle)), CASE_METRIC_FIELDS)
            summary_json = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            case_json = json.loads((out_dir / "case_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(set(summary_json[0]), set(SUMMARY_FIELDS))
            self.assertEqual(set(case_json[0]), set(CASE_METRIC_FIELDS))
            self.assertEqual(case_json[0]["model_name"], "fake-model")
            self.assertEqual(case_json[0]["planner_rounds"], 7)
            self.assertEqual(case_json[0]["tool_calls"], 4)
            self.assertEqual(case_json[0]["vla_calls"], 1)
            self.assertEqual(case_json[0]["sam3_calls"], 1)
            self.assertEqual(case_json[0]["failure_reason"], None)

            charts = render_charts(rows, out_dir)
            chart_names = list(charts)  # svg (no seaborn in eval venv) or png
            self.assertTrue(chart_names, "expected at least one chart")
            report = render_report(rows, out_dir)
            self.assertTrue(report.is_file())
            report_text = report.read_text(encoding="utf-8")
            self.assertIn("Master case table", report_text)
            self.assertIn("Planner rounds", report_text)
            self.assertIn("Failure reason", report_text)
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

    def test_suite_of_extracts_prefix(self) -> None:
        self.assertEqual(suite_of({"task_id": "libero_object:2"}), "libero_object")
        self.assertEqual(suite_of({"task_id": "bare_task"}), "bare_task")
        self.assertEqual(suite_of({"task_id": None}), "suite")

    def test_aggregate_metrics_coverage_semantics(self) -> None:
        rows = [
            _row("libero_object:0", 0, "success", True),
            _row("libero_object:0", 1, "task_failure", False),
            _row("libero_object:1", 0, "timeout", False),
        ]
        metrics = aggregate_metrics(rows)
        suite = next(m for m in metrics["suites"] if m["suite_id"] == "libero_object")
        # valid = decisive outcomes; unknown = infra/invalid/unstarted
        self.assertEqual(suite["planned"], 3)
        self.assertEqual(suite["valid"], 3)
        self.assertEqual(suite["unknown"], 0)
        self.assertEqual(suite["successes"], 1)
        self.assertAlmostEqual(suite["yield_pct"], 100.0 / 3)
        self.assertAlmostEqual(suite["valid_pct"], 100.0)
        # task-level
        task0 = next(t for t in metrics["tasks"] if t["task_id"] == "libero_object:0")
        self.assertEqual(task0["planned"], 2)
        self.assertEqual(task0["valid"], 2)
        self.assertEqual(task0["successes"], 1)

    def test_aggregate_metrics_pending_and_unknown(self) -> None:
        rows = [
            _row("libero_object:0", 0, "infrastructure_failure", None),
            _row("libero_object:1", 0, "invalid", None),
        ]
        suite = aggregate_metrics(rows)["suites"][0]
        # no decisive outcome -> valid 0 -> pending, not 0%
        self.assertEqual(suite["valid"], 0)
        self.assertEqual(suite["unknown"], 2)
        self.assertEqual(suite["yield_pct"], 0.0)
        self.assertEqual(suite["conditional_success_pct"], None)
        self.assertEqual(suite["valid_pct"], 0.0)

    def test_duration_quantiles_exclude_unstarted(self) -> None:
        rows = [
            _row("libero_object:0", 0, "success", True, duration=20.0),
            _row("libero_object:0", 1, "success", True, duration=40.0),
            _row("libero_object:1", 0, "unstarted", None, duration=None),
        ]
        suite = aggregate_metrics(rows)["suites"][0]
        self.assertEqual(suite["duration_n"], 2)
        q = suite["duration_quantiles"]
        self.assertEqual(q[2], 30.0)  # median
        # linear interpolation between the two samples, matching numpy quantile defaults
        self.assertAlmostEqual(q[0], 22.0)  # p10
        self.assertAlmostEqual(q[4], 38.0)  # p90

    def test_pages_for_splits_and_validates(self) -> None:
        self.assertEqual(pages_for(list("abcdef"), page_size=2), [["a", "b"], ["c", "d"], ["e", "f"]])
        with self.assertRaises(ValueError):
            pages_for(["a"], page_size=0)


def _row(task_id, seed, status, success, duration=10.0):
    from eva_agentic.viz import GLYPHS  # noqa: F401 -- keep import local for parity
    return {
        "run_id": "r", "job_id": "j", "case_id": f"{task_id}:s{seed}",
        "task_id": task_id, "task_index": int(task_id.split(":")[1]),
        "seed": seed, "participant": "rpent_libero", "attempt": 1,
        "status": status, "task_success": success, "duration_s": duration,
        "termination_reason": None, "error_source": None, "success_source": None,
    }


if __name__ == "__main__":
    unittest.main()
