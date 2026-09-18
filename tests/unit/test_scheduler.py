import threading
import tempfile
import time
import unittest
from pathlib import Path

from eva_agentic.scheduler import AttemptExecution, run_jobs
from eva_agentic.schema import (
    AttemptStatus,
    Case,
    EpisodeResult,
    Job,
    OutcomeStatus,
)


def make_job(job_id: str) -> Job:
    case = Case(case_id=job_id, task_id="task", seed=1, initialization={})
    return Job(job_id=job_id, participant="fake", cases=(case,))


def success_result(job: Job) -> EpisodeResult:
    return EpisodeResult(
        case_id=job.cases[0].case_id,
        status=OutcomeStatus.SUCCESS,
        task_success=True,
    )


class RunJobsTest(unittest.TestCase):
    def test_runs_jobs_in_parallel_with_resource_cap(self) -> None:
        jobs = [make_job(f"case-{number:03d}") for number in range(1, 6)]
        active = 0
        max_active = 0
        counter_lock = threading.Lock()

        def runner(job: Job, attempt_dir: Path) -> object:
            nonlocal active, max_active
            with counter_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.01)
            with counter_lock:
                active -= 1
            return AttemptExecution(
                status=AttemptStatus.COMPLETED,
                process={},
                results=(success_result(job),),
            )

        with tempfile.TemporaryDirectory() as directory:
            statuses = run_jobs(Path(directory), jobs, runner, max_workers=2)

        self.assertEqual(set(statuses.values()), {AttemptStatus.COMPLETED})
        self.assertLessEqual(max_active, 2)

    def test_preserves_failure_and_reruns_unfinished_job_with_new_attempt(self) -> None:
        jobs = [make_job("case-001")]
        runner_calls: list[int] = []

        def failing_runner(job: Job, attempt_dir: Path) -> object:
            runner_calls.append(int(attempt_dir.name))
            raise RuntimeError("simulated crash before result commit")

        with tempfile.TemporaryDirectory() as directory:
            run_jobs(Path(directory), jobs, failing_runner, max_workers=1)
            self.assertEqual(runner_calls, [1])

            def successful_runner(job: Job, attempt_dir: Path) -> object:
                runner_calls.append(int(attempt_dir.name))
                return AttemptExecution(
                    status=AttemptStatus.COMPLETED,
                    process={},
                    results=(success_result(job),),
                )

            statuses = run_jobs(Path(directory), jobs, successful_runner, max_workers=1)

        self.assertEqual(runner_calls, [1, 2])
        self.assertEqual(statuses["case-001"], AttemptStatus.COMPLETED)

    def test_completed_attempt_is_terminal_but_failed_attempt_can_resume(self) -> None:
        jobs = [make_job("case-001")]

        def unexpected_runner(job: Job, attempt_dir: Path) -> object:
            raise AssertionError("completed job must not rerun")

        with tempfile.TemporaryDirectory() as directory:
            run_jobs(
                Path(directory),
                jobs,
                lambda job, attempt_dir: AttemptExecution(
                    status=AttemptStatus.FAILED,
                    process={"exit_code": 1},
                    results=(),
                ),
                max_workers=1,
            )
            statuses = run_jobs(
                Path(directory), jobs, unexpected_runner, max_workers=1
            )

        self.assertEqual(statuses["case-001"], AttemptStatus.FAILED)

    def test_completed_task_failure_is_terminal(self) -> None:
        jobs = [make_job("case-001")]
        result = EpisodeResult(
            case_id="case-001",
            status=OutcomeStatus.TASK_FAILURE,
            task_success=False,
        )

        def unexpected_runner(job: Job, attempt_dir: Path) -> object:
            raise AssertionError("task failure is a valid completed attempt")

        with tempfile.TemporaryDirectory() as directory:
            run_jobs(
                Path(directory),
                jobs,
                lambda job, attempt_dir: AttemptExecution(
                    status=AttemptStatus.COMPLETED,
                    process={"exit_code": 0},
                    results=(result,),
                ),
                max_workers=1,
            )
            statuses = run_jobs(
                Path(directory), jobs, unexpected_runner, max_workers=1
            )

        self.assertEqual(statuses["case-001"], AttemptStatus.COMPLETED)
