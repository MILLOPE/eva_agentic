import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from eva_agentic.store import (
    atomic_write_json,
    atomic_write_jsonl,
    commit_attempt,
    initialize_run,
    load_run,
    prepare_attempt,
)

from tests.unit.test_experiment_config import EXPERIMENT_DATA
from eva_agentic.experiment import resolve_experiment
from eva_agentic.schema import (
    AttemptStatus,
    Case,
    EpisodeResult,
    Job,
    OutcomeStatus,
)


class AtomicWriteTest(unittest.TestCase):
    def test_atomic_write_json_replaces_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "data.json"

            atomic_write_json(path, {"value": 1})
            atomic_write_json(path, {"value": 2})

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"value": 2})
            self.assertEqual(list(path.parent.glob(".data.json.*")), [])

    def test_atomic_write_jsonl_writes_one_record_per_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"

            atomic_write_jsonl(path, [{"id": 1}, {"id": 2}])

            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual([json.loads(line) for line in lines], [{"id": 1}, {"id": 2}])


class InitializeRunTest(unittest.TestCase):
    def test_creates_resolved_experiment_cases_and_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cases_file = self._write_case_file(Path(directory))
            experiment = replace(
                resolve_experiment(EXPERIMENT_DATA),
                cases_file=str(cases_file),
            )

            run_dir = initialize_run(Path(directory), experiment, run_id="run_001")

            self.assertEqual(run_dir, Path(directory) / "run_001")
            experiment_data = json.loads(
                (run_dir / "inputs/experiment.resolved.json").read_text(encoding="utf-8")
            )
            case_lines = (run_dir / "inputs/cases.jsonl").read_text(encoding="utf-8").splitlines()
            plan_data = json.loads((run_dir / "inputs/plan.json").read_text(encoding="utf-8"))
            self.assertEqual(experiment_data["name"], "libero_pro_compare")
            self.assertEqual(len(case_lines), 1)
            self.assertEqual(len(plan_data["jobs"]), 2)

    def test_rejects_existing_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cases_file = self._write_case_file(Path(directory))
            experiment = replace(
                resolve_experiment(EXPERIMENT_DATA),
                cases_file=str(cases_file),
            )
            initialize_run(Path(directory), experiment, run_id="run_001")

            with self.assertRaisesRegex(FileExistsError, "run directory already exists"):
                initialize_run(Path(directory), experiment, run_id="run_001")

    def test_load_run_verifies_frozen_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cases_file = self._write_case_file(Path(directory))
            experiment = replace(
                resolve_experiment(EXPERIMENT_DATA),
                cases_file=str(cases_file),
            )
            run_dir = initialize_run(Path(directory), experiment, run_id="run_001")

            run_inputs = load_run(run_dir)

            self.assertEqual(run_inputs.experiment.name, "libero_pro_compare")
            self.assertEqual(len(run_inputs.cases), 1)
            self.assertEqual(len(run_inputs.plan.jobs), 2)

    def test_load_run_rejects_modified_case_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cases_file = self._write_case_file(Path(directory))
            experiment = replace(
                resolve_experiment(EXPERIMENT_DATA),
                cases_file=str(cases_file),
            )
            run_dir = initialize_run(Path(directory), experiment, run_id="run_001")
            cases_path = run_dir / "inputs" / "cases.jsonl"
            modified = json.loads(cases_path.read_text(encoding="utf-8").splitlines()[0])
            modified["seed"] = 2
            cases_path.write_text(json.dumps(modified) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "input file digest changed"):
                load_run(run_dir)

    def _write_case_file(self, directory: Path) -> Path:
        cases_dir = directory / "configs" / "cases"
        cases_dir.mkdir(parents=True)
        cases_file = cases_dir / "libero_pro_smoke.jsonl"
        cases_file.write_text(
            json.dumps(
                {
                    "case_id": "case-001",
                    "task_id": "task-a",
                    "seed": 1,
                    "initialization": {},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return cases_file


class AttemptStoreTest(unittest.TestCase):
    def test_prepare_and_commit_attempt_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            case = Case(case_id="case-001", task_id="task-a", seed=1, initialization={})
            job = Job(job_id="job_0001", participant="fake", cases=(case,))
            attempt_dir = initialize_attempt(Path(directory), job)
            result = EpisodeResult(
                case_id="case-001",
                status=OutcomeStatus.SUCCESS,
                task_success=True,
            )

            result_path = commit_attempt(
                attempt_dir,
                status=AttemptStatus.COMPLETED,
                process={"exit_code": 0},
                results=(result,),
            )

            first_result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(first_result["status"], "completed")
            self.assertEqual(first_result["results"][0]["status"], "success")
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                commit_attempt(
                    attempt_dir,
                    status=AttemptStatus.FAILED,
                    process={"exit_code": 1},
                )
            self.assertEqual(
                json.loads(result_path.read_text(encoding="utf-8")),
                first_result,
            )

    def test_rejects_result_for_unknown_case(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            case = Case(case_id="case-001", task_id="task-a", seed=1, initialization={})
            job = Job(job_id="job_0001", participant="fake", cases=(case,))
            attempt_dir = initialize_attempt(Path(directory), job)
            result = EpisodeResult(
                case_id="case-002",
                status=OutcomeStatus.INVALID,
                task_success=None,
            )

            with self.assertRaisesRegex(ValueError, "unknown case_ids"):
                commit_attempt(
                    attempt_dir,
                    status=AttemptStatus.COMPLETED,
                    process={"exit_code": 0},
                    results=(result,),
                )


def initialize_attempt(runs_root: Path, job: Job) -> Path:
    from eva_agentic.store import prepare_attempt

    return prepare_attempt(runs_root, job, attempt_id=1)
