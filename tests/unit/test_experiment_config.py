import json
import tempfile
import unittest
from pathlib import Path

from eva_agentic.experiment import load_experiment, resolve_experiment
from eva_agentic.schema import RunMode


EXPERIMENT_DATA = {
    "schema_version": 1,
    "name": "libero_pro_compare",
    "benchmark": "libero_pro",
    "cases_file": "configs/cases/libero_pro_smoke.jsonl",
    "participants": ["zetta_libero_pro", "rats_libero_pro"],
    "protocol": {
        "track": "native_system",
        "phase": "evaluation",
        "memory_policy": "frozen_per_case",
        "scoring": "benchmark",
        "budget": {
            "max_control_steps": 500,
            "episode_timeout_s": 600,
        },
    },
    "execution": {
        "mode": "frozen",
        "max_jobs": 2,
        "max_infrastructure_retries": 0,
        "job_timeout_s": 1800,
    },
    "artifacts": {
        "video": "all",
        "trace": "actions",
    },
}


class ResolveExperimentTest(unittest.TestCase):
    def test_resolves_valid_mapping(self) -> None:
        experiment = resolve_experiment(EXPERIMENT_DATA)

        self.assertEqual(experiment.name, "libero_pro_compare")
        self.assertEqual(experiment.execution.mode, RunMode.FROZEN)
        self.assertEqual(
            experiment.protocol.budget.max_control_steps,
            500,
        )

    def test_rejects_missing_execution(self) -> None:
        data = dict(EXPERIMENT_DATA)
        del data["execution"]

        with self.assertRaisesRegex(ValueError, "execution must be a mapping"):
            resolve_experiment(data)

    def test_rejects_missing_field(self) -> None:
        data = dict(EXPERIMENT_DATA)
        data["protocol"] = dict(EXPERIMENT_DATA["protocol"])
        del data["protocol"]["scoring"]

        with self.assertRaisesRegex(ValueError, "missing experiment field: scoring"):
            resolve_experiment(data)


class LoadExperimentTest(unittest.TestCase):
    def test_resolves_relative_cases_file_from_config_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = root / "cases.jsonl"
            cases.write_text(
                json.dumps(
                    {
                        "case_id": "case-1",
                        "task_id": "task-1",
                        "seed": 0,
                        "initialization": {},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            path = root / "experiment.json"
            path.write_text(json.dumps(EXPERIMENT_DATA | {"cases_file": "cases.jsonl"}), encoding="utf-8")

            experiment = load_experiment(path)

            self.assertEqual(experiment.cases_file, str(cases.resolve()))

    def test_loads_json_with_source_prefix_on_error(self) -> None:
        data = dict(EXPERIMENT_DATA)
        data["name"] = ""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.json"
            path.write_text(json.dumps(data), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, r"experiment.json: invalid experiment"):
                load_experiment(path)
