import json
import tempfile
import unittest
from pathlib import Path

from eva_agentic.experiment import load_cases
from eva_agentic.schema import Case


class LoadCasesTest(unittest.TestCase):
    def test_loads_ordered_cases_from_jsonl(self) -> None:
        items = [
            {
                "case_id": "case-002",
                "task_id": "task-b",
                "seed": 2,
                "initialization": {"layout": "b"},
            },
            {
                "case_id": "case-001",
                "task_id": "task-a",
                "seed": 1,
                "initialization": {},
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text(
                "".join(json.dumps(item) + "\n" for item in items), encoding="utf-8"
            )

            cases = load_cases(path)

        self.assertEqual(cases, tuple(Case(**item) for item in items))

    def test_rejects_duplicate_case_ids(self) -> None:
        item = {
            "case_id": "case-001",
            "task_id": "task-a",
            "seed": 1,
            "initialization": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text(
                json.dumps(item) + "\n" + json.dumps(item) + "\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "duplicate case_id: case-001"):
                load_cases(path)

    def test_rejects_invalid_json_with_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text("{\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "cases.jsonl:1: invalid JSON"):
                load_cases(path)

    def test_rejects_empty_case_list(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text("\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "case list must not be empty"):
                load_cases(path)
