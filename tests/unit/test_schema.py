import unittest

from eva_agentic.schema import (
    ArtifactPolicy,
    Attempt,
    AttemptStatus,
    Budget,
    Case,
    EpisodeResult,
    Execution,
    Experiment,
    OutcomeStatus,
    Protocol,
    RunMode,
)


def make_experiment() -> Experiment:
    return Experiment(
        schema_version=1,
        name="libero_pro_compare",
        benchmark="libero_pro",
        cases_file="configs/cases/libero_pro_smoke.jsonl",
        participants=("zetta_libero_pro", "rats_libero_pro"),
        protocol=Protocol(
            track="native_system",
            phase="evaluation",
            memory_policy="frozen_per_case",
            scoring="benchmark",
            budget=Budget(max_control_steps=500, episode_timeout_s=600),
        ),
        execution=Execution(
            mode=RunMode.FROZEN,
            max_jobs=2,
            max_infrastructure_retries=0,
            job_timeout_s=1800,
        ),
        artifacts=ArtifactPolicy(video="all", trace="actions"),
    )


class SchemaTest(unittest.TestCase):
    def test_experiment_serializes_enums_and_tuples(self) -> None:
        experiment = make_experiment()

        data = experiment.to_dict()

        self.assertEqual(data["schema_version"], 1)
        self.assertEqual(data["participants"], ["zetta_libero_pro", "rats_libero_pro"])
        self.assertEqual(data["execution"]["mode"], "frozen")

    def test_experiment_rejects_duplicate_participants(self) -> None:
        experiment = make_experiment()

        with self.assertRaisesRegex(ValueError, "participants must be unique"):
            Experiment(
                schema_version=experiment.schema_version,
                name=experiment.name,
                benchmark=experiment.benchmark,
                cases_file=experiment.cases_file,
                participants=(experiment.participants[0], experiment.participants[0]),
                protocol=experiment.protocol,
                execution=experiment.execution,
                artifacts=experiment.artifacts,
            )

    def test_success_result_requires_benchmark_success(self) -> None:
        with self.assertRaisesRegex(ValueError, "SUCCESS requires"):
            EpisodeResult(
                case_id="case-001",
                status=OutcomeStatus.SUCCESS,
                task_success=False,
            )

    def test_runtime_failure_requires_unknown_task_success(self) -> None:
        with self.assertRaisesRegex(ValueError, "timeout requires"):
            EpisodeResult(
                case_id="case-001",
                status=OutcomeStatus.TIMEOUT,
                task_success=True,
            )

    def test_attempt_rejects_duplicate_case_results(self) -> None:
        result = EpisodeResult(
            case_id="case-001",
            status=OutcomeStatus.INVALID,
            task_success=None,
        )

        with self.assertRaisesRegex(ValueError, "case_ids must be unique"):
            Attempt(
                participant="rats_libero_pro",
                job_id="job_001",
                attempt_id=1,
                status=AttemptStatus.FAILED,
                results=(result, result),
            )
