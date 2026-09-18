import json
import tempfile
import unittest
from pathlib import Path

from eva_agentic.experiment import resolve_experiment
from eva_agentic.schema import AttemptStatus, EpisodeResult, OutcomeStatus
from eva_agentic.store import commit_attempt, initialize_run, prepare_attempt
from eva_agentic.summary import summarize_run


class SummaryTest(unittest.TestCase):
    def test_counts_known_outcomes_separately(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = root / "cases.jsonl"
            cases.write_text(
                "".join(
                    json.dumps({"case_id": f"case-{index}", "task_id": f"task-{index}", "seed": 0, "initialization": {}}) + "\n"
                    for index in range(4)
                ),
                encoding="utf-8",
            )
            experiment = resolve_experiment({
                "schema_version": 1, "name": "summary", "benchmark": "fake", "cases_file": str(cases), "participants": ["fake"],
                "protocol": {"track": "test", "phase": "eval", "memory_policy": "frozen", "scoring": "native"},
                "execution": {"mode": "debug", "max_jobs": 1, "max_infrastructure_retries": 0, "job_timeout_s": 1},
                "artifacts": {"video": "off", "trace": "off"},
            })
            run_dir = initialize_run(root / "runs", experiment, "summary")
            statuses = [OutcomeStatus.SUCCESS, OutcomeStatus.TASK_FAILURE, OutcomeStatus.INVALID]
            from eva_agentic.store import load_run
            plan = load_run(run_dir).plan
            for job, status in zip(plan.jobs[:3], statuses):
                attempt = prepare_attempt(run_dir, job, 1)
                success = True if status is OutcomeStatus.SUCCESS else False if status is OutcomeStatus.TASK_FAILURE else None
                commit_attempt(attempt, AttemptStatus.COMPLETED, {}, (EpisodeResult(job.cases[0].case_id, status, success),))
            failed = prepare_attempt(run_dir, plan.jobs[3], 1)
            commit_attempt(failed, AttemptStatus.FAILED, {"native_status": "timeout"})
            summary = summarize_run(run_dir)
        self.assertEqual(summary["planned_jobs"], 4)
        self.assertEqual(summary["completed_jobs"], 3)
        self.assertEqual(summary["successes"], 1)
        self.assertEqual(summary["task_failures"], 1)
        self.assertEqual(summary["invalid"], 1)
        self.assertEqual(summary["infrastructure_failures"], 1)
        self.assertEqual(summary["success_rate"], 0.5)
        self.assertEqual(summary["success_rate_denominator"], 2)


if __name__ == "__main__":
    unittest.main()
