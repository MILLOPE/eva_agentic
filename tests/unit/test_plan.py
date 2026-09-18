import unittest

from eva_agentic.experiment import build_plan
from eva_agentic.schema import Case

from tests.unit.test_experiment_config import EXPERIMENT_DATA
from eva_agentic.experiment import resolve_experiment


class BuildPlanTest(unittest.TestCase):
    def test_expands_participant_and_case_pairs(self) -> None:
        experiment = resolve_experiment(EXPERIMENT_DATA)
        cases = (
            Case(case_id="case-002", task_id="task-b", seed=2, initialization={}),
            Case(case_id="case-001", task_id="task-a", seed=1, initialization={}),
        )

        plan = build_plan(experiment, cases)

        self.assertEqual(len(plan.jobs), 4)
        self.assertEqual(
            [job.job_id for job in plan.jobs],
            ["job_0001", "job_0002", "job_0003", "job_0004"],
        )
        self.assertEqual(
            [(job.participant, job.cases[0].case_id) for job in plan.jobs],
            [
                ("zetta_libero_pro", "case-002"),
                ("zetta_libero_pro", "case-001"),
                ("rats_libero_pro", "case-002"),
                ("rats_libero_pro", "case-001"),
            ],
        )

    def test_plan_round_trips_to_plain_data(self) -> None:
        experiment = resolve_experiment(EXPERIMENT_DATA)
        case = Case(case_id="case-001", task_id="task-a", seed=1, initialization={})

        plan = build_plan(experiment, (case,))
        data = plan.to_dict()

        self.assertEqual(data["jobs"][0]["job_id"], "job_0001")
        self.assertEqual(data["jobs"][0]["cases"][0]["case_id"], "case-001")
