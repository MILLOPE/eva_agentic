import json
import tempfile
import unittest
from pathlib import Path

from eva_agentic.adapters.rpent_libero import RpentLiberoAdapter
from eva_agentic.frameworks import FrameworkLocal, FrameworkProfile, FrameworkSpec, RuntimeBackend
from eva_agentic.process import NativeRunResult, NativeRunStatus
from eva_agentic.schema import Case, Job, OutcomeStatus


class RpentLiberoAdapterTest(unittest.TestCase):
    def make_adapter(
        self,
        root: Path,
        command: tuple[str, ...] = ("-m", "rpent.cli.main"),
    ) -> RpentLiberoAdapter:
        return RpentLiberoAdapter(
            FrameworkSpec("rpent_libero", RuntimeBackend.PYTHON, root, command)
        )

    def make_job(self) -> Job:
        return Job(
            "job-1",
            "rpent_libero",
            (
                Case(
                    "case-1",
                    "libero_object:2",
                    7,
                    {"suite": "libero_object", "task_id": 2, "max_episode_steps": 500},
                ),
            ),
        )

    def profile(self) -> FrameworkProfile:
        return FrameworkProfile(
            {
                "rpent_libero": FrameworkLocal(
                    interpreter=Path("/opt/eva-rpent/bin/python"),
                    env={"RPENT_VLA_ENDPOINT": "127.0.0.1:8113", "RPENT_SAM3_ENDPOINT": "127.0.0.1:8114"},
                )
            }
        )

    def run_result(self, root: Path, exit_code: int = 0) -> NativeRunResult:
        return NativeRunResult(
            NativeRunStatus.COMPLETED,
            root / "launch.json",
            root / "runtime.json",
            root / "stdout.log",
            root / "stderr.log",
            exit_code=exit_code,
        )

    def test_builds_external_service_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launch = self.make_adapter(root).build_launch(
                self.make_job(), root / "attempt", self.profile(), ()
            )
        self.assertEqual(launch.argv[:3], ("/opt/eva-rpent/bin/python", "-m", "rpent.cli.main"))
        self.assertIn("--vla-endpoint", launch.argv)
        self.assertIn("127.0.0.1:8113", launch.argv)
        self.assertIn("--sam3-endpoint", launch.argv)
        self.assertIn("--max-episode-steps", launch.argv)
        self.assertIn("500", launch.argv)

    def test_preserves_vllm_user_native_planner_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = (
                "-m",
                "rpent.cli.main",
                "--planner",
                "vllm_user",
                "--model",
                "default",
                "--memory-profile",
                "hf",
                "--max-turns",
                "20",
                "--planner-timeout-s",
                "1200",
            )
            launch = self.make_adapter(root, command).build_launch(
                self.make_job(), root / "attempt", self.profile(), ()
            )
        argv = launch.argv
        self.assertIn("--planner", argv)
        self.assertEqual(argv[argv.index("--planner") + 1], "vllm_user")
        self.assertEqual(argv[argv.index("--model") + 1], "default")
        self.assertEqual(argv[argv.index("--max-turns") + 1], "20")
        self.assertEqual(argv[argv.index("--planner-timeout-s") + 1], "1200")
        self.assertIn("--suite", argv)
        self.assertIn("libero_object", argv)
        self.assertIn("--seed", argv)
        self.assertIn("7", argv)

    def test_parses_audit_and_rejects_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = root / "attempt"
            native = attempt / "native"
            native.mkdir(parents=True)
            audit = native / "object_t2_s7.json"
            audit.write_text(
                json.dumps(
                    {
                        "suite": "libero_object",
                        "task_id": 2,
                        "seed": 7,
                        "terminated": True,
                        "truncated": False,
                        "regime": "strict",
                    }
                ),
                encoding="utf-8",
            )
            adapter = self.make_adapter(root)
            success = adapter.parse(self.make_job(), attempt, self.run_result(root))[0]
            self.assertEqual(success.status, OutcomeStatus.SUCCESS)
            self.assertEqual(success.success_source, "rpent.audit.terminated")
            audit.write_text(json.dumps({"terminated": True}), encoding="utf-8")
            invalid = adapter.parse(self.make_job(), attempt, self.run_result(root))[0]
            self.assertEqual(invalid.status, OutcomeStatus.INVALID)

    def test_terminated_false_is_task_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = root / "attempt"
            native = attempt / "native"
            native.mkdir(parents=True)
            (native / "object_t2_s7.json").write_text(
                json.dumps(
                    {
                        "suite": "libero_object",
                        "task_id": 2,
                        "seed": 7,
                        "terminated": False,
                    }
                ),
                encoding="utf-8",
            )
            result = self.make_adapter(root).parse(
                self.make_job(), attempt, self.run_result(root)
            )[0]
        self.assertEqual(result.status, OutcomeStatus.TASK_FAILURE)
        self.assertIs(result.task_success, False)
        self.assertEqual(result.success_source, "rpent.audit.terminated")

    def test_nonzero_exit_is_infrastructure_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.make_adapter(root).parse(
                self.make_job(), root / "attempt", self.run_result(root, exit_code=2)
            )[0]
        self.assertEqual(result.status, OutcomeStatus.INFRASTRUCTURE_FAILURE)


if __name__ == "__main__":
    unittest.main()
