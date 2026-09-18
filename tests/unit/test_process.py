import json
import sys
import tempfile
import unittest
from pathlib import Path

from eva_agentic.frameworks import NativeLaunch
from eva_agentic.process import NativeRunStatus, run_native_launch
from eva_agentic.resources import ResourceGrant


class NativeProcessTest(unittest.TestCase):
    def test_success_records_redacted_launch_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root / "worker.py"
            worker.write_text("import json, os, pathlib\npathlib.Path(os.environ['EVA_OUTPUT_DIR']).joinpath('result.json').write_text(json.dumps({'ok': True}))\n", encoding="utf-8")
            launch = NativeLaunch(
                (sys.executable, str(worker)),
                root,
                {"SECRET": "redact-me"},
                root / "attempt",
                root / "attempt" / "native",
                {"gpu": 1},
                {"framework_revision": "849143b"},
            )
            result = run_native_launch(launch, timeout_s=5, resource_grants=(ResourceGrant("gpu", 0, root / "gpu.lock"),))
            evidence = json.loads(result.launch_path.read_text())
            self.assertEqual(result.status, NativeRunStatus.COMPLETED)
            self.assertEqual(json.loads((launch.output_dir / "result.json").read_text()), {"ok": True})
            self.assertIn("SECRET", evidence["env_keys"])
            self.assertEqual(evidence["provenance"]["framework_revision"], "849143b")
            self.assertNotIn("redact-me", result.launch_path.read_text())

    def test_timeout_terminates_worker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launch = NativeLaunch((sys.executable, "-c", "import time; time.sleep(30)"), root, {}, root / "attempt", root / "attempt" / "native", {})
            result = run_native_launch(launch, timeout_s=0.05)
            self.assertEqual(result.status, NativeRunStatus.TIMEOUT)
            self.assertTrue(result.terminated)


if __name__ == "__main__":
    unittest.main()
