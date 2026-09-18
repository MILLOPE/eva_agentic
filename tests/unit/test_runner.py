import json
import sys
import tempfile
import unittest
from pathlib import Path

from eva_agentic.adapters.base import UnsupportedCondition
from eva_agentic.frameworks import FrameworkProfile, NativeLaunch
from eva_agentic.runner import run_adapter_job
from eva_agentic.schema import AttemptStatus, Case, EpisodeResult, Job, OutcomeStatus


class FakeAdapter:
    name = "fake"

    def __init__(self, worker: Path, malformed: bool = False) -> None:
        self.worker = worker
        self.malformed = malformed

    def build_launch(self, job, attempt_dir, profile, grants):
        return NativeLaunch((sys.executable, str(self.worker)), self.worker.parent, {}, attempt_dir, attempt_dir / "native", {})

    def parse(self, job, attempt_dir, run):
        result_path = attempt_dir / "native" / "result.json"
        try:
            data = json.loads(result_path.read_text())
            if data["success"] is not True:
                raise ValueError("not successful")
            return (EpisodeResult(job.cases[0].case_id, OutcomeStatus.SUCCESS, True, success_source="fake.result", evidence_paths=(str(result_path),)),)
        except (OSError, ValueError, json.JSONDecodeError, KeyError):
            return (EpisodeResult(job.cases[0].case_id, OutcomeStatus.INVALID, None, error_source="fake.result", evidence_paths=(str(result_path),)),)


class UnsupportedAdapter:
    name = "unsupported"

    def build_launch(self, job, attempt_dir, profile, grants):
        raise UnsupportedCondition("native framework cannot set this seed")

    def parse(self, job, attempt_dir, run):
        raise AssertionError("unsupported adapter must not launch")


class AdapterRunnerTest(unittest.TestCase):
    def make_job(self) -> Job:
        return Job("job-1", "fake", (Case("case-1", "task", 0, {}),))

    def test_parses_native_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root / "worker.py"
            worker.write_text("import json, os, pathlib\npathlib.Path(os.environ['EVA_OUTPUT_DIR']).joinpath('result.json').write_text(json.dumps({'success': True}))\n", encoding="utf-8")
            execution = run_adapter_job(FakeAdapter(worker), FrameworkProfile({}), 5, self.make_job(), root / "attempt")
            self.assertEqual(execution.status, AttemptStatus.COMPLETED)
            self.assertEqual(execution.results[0].status, OutcomeStatus.SUCCESS)
            self.assertEqual(execution.results[0].success_source, "fake.result")

    def test_missing_result_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root / "worker.py"
            worker.write_text("pass\n", encoding="utf-8")
            execution = run_adapter_job(FakeAdapter(worker), FrameworkProfile({}), 5, self.make_job(), root / "attempt")
            self.assertEqual(execution.status, AttemptStatus.COMPLETED)
            self.assertEqual(execution.results[0].status, OutcomeStatus.INVALID)

    def test_unsupported_condition_is_a_terminal_invalid_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            execution = run_adapter_job(UnsupportedAdapter(), FrameworkProfile({}), 5, self.make_job(), Path(directory) / "attempt")
        self.assertEqual(execution.status, AttemptStatus.COMPLETED)
        self.assertEqual(execution.results[0].status, OutcomeStatus.INVALID)
        self.assertEqual(execution.process["native_status"], "unsupported_condition")


if __name__ == "__main__":
    unittest.main()
