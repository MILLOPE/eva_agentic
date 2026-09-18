import tempfile
import unittest
from pathlib import Path

from eva_agentic.frameworks import (
    FrameworkLocal,
    FrameworkProfile,
    FrameworkSpec,
    RuntimeBackend,
    load_framework_specs,
    resolve_native_launch,
)
from eva_agentic.resources import ResourceGrant
from eva_agentic.schema import Case, Job


class FrameworkResolutionTest(unittest.TestCase):
    def make_job(self) -> Job:
        return Job(
            job_id="job_001",
            participant="fake",
            cases=(Case("case-001", "task-7", 42, {}),),
        )

    def test_conda_command_and_gpu_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = FrameworkSpec(
                "fake", RuntimeBackend.CONDA, root,
                ("python", "worker.py", "--task", "{task_id}", "--seed", "{seed}", "--output-dir", "{output_dir}"),
                {"gpu": 1},
            )
            profile = FrameworkProfile({"fake": FrameworkLocal(Path("/opt/conda/bin/conda"), "fake-env")})
            launch = resolve_native_launch(
                spec, profile, self.make_job(), root / "attempt",
                (ResourceGrant("gpu", 3, root / "gpu.lock"),),
            )
        self.assertEqual(launch.argv[:5], ("/opt/conda/bin/conda", "run", "--name", "fake-env", "--no-capture-output"))
        self.assertIn("task-7", launch.argv)
        self.assertEqual(launch.env["CUDA_VISIBLE_DEVICES"], "3")
        self.assertEqual(launch.env["EVA_OUTPUT_DIR"], str(root / "attempt" / "native"))

    def test_uv_and_python_wrappers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = self.make_job()
            uv_spec = FrameworkSpec("uv", RuntimeBackend.UV, root, ("worker.py",))
            uv = resolve_native_launch(uv_spec, FrameworkProfile({"uv": FrameworkLocal(uv_executable=Path("/share/bin/uv"))}), job, root / "uv")
            self.assertEqual(uv.argv[:4], ("/share/bin/uv", "run", "--project", str(root)))
            py_spec = FrameworkSpec("py", RuntimeBackend.PYTHON, root, ("worker.py",))
            py = resolve_native_launch(py_spec, FrameworkProfile({"py": FrameworkLocal(interpreter=Path("/opt/env/bin/python"))}), job, root / "py")
            self.assertEqual(py.argv[:2], ("/opt/env/bin/python", "worker.py"))

    def test_load_rejects_duplicate_name_and_unknown_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frameworks.json"
            path.write_text('{"frameworks": [{"name": "same", "backend": "python", "workdir": ".", "command": ["worker.py"]}, {"name": "same", "backend": "python", "workdir": ".", "command": ["worker.py"]}]}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unique"):
                load_framework_specs(path)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            FrameworkSpec("bad", RuntimeBackend.PYTHON, Path("."), ("{unknown}",))

    def test_provenance_is_loaded_and_must_be_string_valued(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "frameworks.json"
            path.write_text(
                '{"frameworks": [{"name": "rpent", "backend": "python", '
                '"workdir": ".", "command": ["worker"], "provenance": {'
                '"framework_source": "third_party/frameworks/rpent", '
                '"framework_revision": "849143b", '
                '"integration_patch": "patches/rpent/libero/vllm_user"}}]}',
                encoding="utf-8",
            )
            spec = load_framework_specs(path)["rpent"]
            self.assertEqual(spec.provenance["framework_revision"], "849143b")
            with self.assertRaisesRegex(ValueError, "strings"):
                FrameworkSpec(
                    "bad-provenance",
                    RuntimeBackend.PYTHON,
                    root,
                    ("worker",),
                    provenance={"revision": 849143},
                )


if __name__ == "__main__":
    unittest.main()
