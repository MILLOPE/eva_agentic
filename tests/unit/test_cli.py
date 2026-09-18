import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from eva_agentic.cli import main


class CliTest(unittest.TestCase):
    def test_init_and_summary_emit_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = root / "cases.jsonl"
            cases.write_text(json.dumps({"case_id": "case-1", "task_id": "task-1", "seed": 0, "initialization": {}}) + "\n", encoding="utf-8")
            experiment = root / "experiment.json"
            experiment.write_text(json.dumps({
                "schema_version": 1, "name": "cli", "benchmark": "fake", "cases_file": str(cases), "participants": ["fake"],
                "protocol": {"track": "test", "phase": "eval", "memory_policy": "frozen", "scoring": "native"},
                "execution": {"mode": "debug", "max_jobs": 1, "max_infrastructure_retries": 0, "job_timeout_s": 1},
                "artifacts": {"video": "off", "trace": "off"},
            }), encoding="utf-8")
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["init", "--experiment", str(experiment), "--runs-root", str(root / "runs"), "--run-id", "cli"]), 0)
            self.assertEqual(json.loads(output.getvalue())["run_dir"], str(root / "runs" / "cli"))
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["summary", "--run-dir", str(root / "runs" / "cli")]), 0)
            self.assertEqual(json.loads(output.getvalue())["planned_jobs"], 1)


if __name__ == "__main__":
    unittest.main()
