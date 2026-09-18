import json
import tempfile
import unittest
from pathlib import Path

from eva_agentic.adapters.rats_libero import RatsLiberoAdapter
from eva_agentic.frameworks import FrameworkLocal, FrameworkProfile, FrameworkSpec, RuntimeBackend
from eva_agentic.process import NativeRunResult, NativeRunStatus
from eva_agentic.schema import Case, Job, OutcomeStatus


class RatsLiberoAdapterTest(unittest.TestCase):
    def make_adapter(self, root: Path) -> RatsLiberoAdapter:
        return RatsLiberoAdapter(FrameworkSpec("rats_libero", RuntimeBackend.PYTHON, root, ("scripts/run_rats.py",)))

    def make_job(self, seed: int = 0) -> Job:
        return Job("job-1", "rats_libero", (Case("case-1", "libero", seed, {"suite": "libero_object_swap", "task_id": 0}),))

    def run_result(self, root: Path) -> NativeRunResult:
        return NativeRunResult(NativeRunStatus.COMPLETED, root / "launch.json", root / "runtime.json", root / "stdout.log", root / "stderr.log", exit_code=0)

    def test_builds_one_trial_native_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launch = self.make_adapter(root).build_launch(self.make_job(), root / "attempt", FrameworkProfile({"rats_libero": FrameworkLocal(interpreter=Path("/bin/python"))}), ())
            self.assertIn("--no-skill-reuse", launch.argv)
            self.assertIn("--iterations", launch.argv)
            self.assertEqual(launch.env["RATS_VERIFIER_STRICT_BENCHMARK"], "1")
            self.assertEqual(launch.env["EVA_OUTPUT_DIR"], str(root / "attempt" / "native"))

    def test_parses_success_and_invalid_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = root / "attempt"
            native = attempt / "native"
            native.mkdir(parents=True)
            (native / "final_summary.json").write_text(json.dumps({"total_iterations": 1, "successful_iterations": 1, "failed_iterations": 0, "success_rate": 1.0}), encoding="utf-8")
            result = self.make_adapter(root).parse(self.make_job(), attempt, self.run_result(root))[0]
            self.assertEqual(result.status, OutcomeStatus.SUCCESS)
            (native / "final_summary.json").write_text("{}", encoding="utf-8")
            invalid = self.make_adapter(root).parse(self.make_job(), attempt, self.run_result(root))[0]
            self.assertEqual(invalid.status, OutcomeStatus.INVALID)

    def test_rejects_unsupported_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "only seed 0"):
                self.make_adapter(root).build_launch(self.make_job(seed=1), root / "attempt", FrameworkProfile({"rats_libero": FrameworkLocal(interpreter=Path("/bin/python"))}), ())


if __name__ == "__main__":
    unittest.main()
