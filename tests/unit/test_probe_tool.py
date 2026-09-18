"""End-to-end offline tests for the capacity-probe CLI commands.

These run the real ``eva-agentic`` entry point in subprocesses, but only with
``--dry-run`` (no services, no scheduling).  They guard template placeholder
rendering and case generation against regressions like YAML flow-mapping
surprises.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT_TEMPLATE = """\
schema_version: 1
name: "probe-{TAG}"
benchmark: libero
cases_file: "{CASES_FILE}"
participants: [rpent_libero]
protocol:
  track: native_system
  phase: exploration
  memory_policy: frozen
  scoring: benchmark
execution:
  mode: debug
  max_jobs: "{MAX_JOBS}"
  max_infrastructure_retries: 0
  job_timeout_s: 1800
artifacts:
  video: all
  trace: actions
"""

FRAMEWORKS = """\
frameworks:
  - name: rpent_libero
    backend: python
    workdir: /tmp/example-workdir
    command: ["-m", "rpent.cli.main"]
    resources:
      gpu: 1
"""

PROFILE = """\
resource_slots:
  gpu: [0]
frameworks:
  rpent_libero:
    env:
      RPENT_VLA_ENDPOINT: 127.0.0.1:18115
"""


def _write(root: Path, name: str, content: str) -> Path:
    path = root / name
    path.write_text(content, encoding="utf-8")
    return path


def _run_cli(argv: list[str]) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "eva_agentic", *argv],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, f"{argv[0]} failed: {result.stderr}\n{result.stdout}"
    return result.stdout


def test_gen_cases(tmp_path) -> None:
    cases = tmp_path / "cases.jsonl"
    _run_cli(["gen-cases", "--suite", "libero_object", "--tasks", "0,2",
               "--seed-base", "0", "--seed-count", "2",
               "--max-episode-steps", "500", "--out", str(cases)])
    lines = cases.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    assert "libero_object" in lines[0]


def test_probe_dry_run_renders_configs(tmp_path) -> None:
    exp = _write(tmp_path, "experiment.yaml", EXPERIMENT_TEMPLATE)
    frameworks = _write(tmp_path, "frameworks.yaml", FRAMEWORKS)
    profile = _write(tmp_path, "profile.yaml", PROFILE)
    cases = _write(tmp_path, "cases.jsonl", '{"case_id":"libero_object:t0:s0"}\n')

    out = _run_cli(
        ["probe", "--experiment", str(exp), "--cases", str(cases),
         "--frameworks", str(frameworks), "--profile", str(profile),
         "--concurrency", "1,2", "--tag", "t", "--dry-run",
         "--work-dir", str(tmp_path / ".probe"),
         "--runs-root", str(tmp_path / "runs")]
    )
    header, *rows = out.splitlines()
    assert "run_id,planned" in header
    assert len(rows) == 2

    exp2 = (tmp_path / ".probe" / "c2" / "experiment.yaml").read_text(encoding="utf-8")
    prof2 = (tmp_path / ".probe" / "c2" / "profile.yaml").read_text(encoding="utf-8")
    assert "max_jobs: 2" in exp2
    assert "cases_file:" in exp2 and str(cases) in exp2
    assert "probe-t" in exp2
    assert "  - 0\n  - 1" in prof2




def test_collect_metrics_counts_all_attempts(tmp_path) -> None:
    """Lock per-attempt result discovery against the runs/ directory layout."""
    import json

    from eva_agentic.experiment import resolve_experiment
    from eva_agentic.probe import _collect_metrics
    from eva_agentic.schema import AttemptStatus, EpisodeResult, OutcomeStatus
    from eva_agentic.store import commit_attempt, initialize_run, load_run, prepare_attempt

    root = Path(tmp_path)
    cases = root / "cases.jsonl"
    cases.write_text(
        "".join(
            json.dumps({"case_id": f"c{index}", "task_id": f"t{index}", "seed": 0, "initialization": {"task_id": index}}) + "\n"
            for index in range(2)
        ),
        encoding="utf-8",
    )
    experiment = resolve_experiment({
        "schema_version": 1, "name": "metrics", "benchmark": "fake", "cases_file": str(cases),
        "participants": ["fake"],
        "protocol": {"track": "test", "phase": "eval", "memory_policy": "frozen", "scoring": "native"},
        "execution": {"mode": "debug", "max_jobs": 1, "max_infrastructure_retries": 0, "job_timeout_s": 1},
        "artifacts": {"video": "off", "trace": "off"},
    })
    run_dir = initialize_run(root / "runs", experiment, "metrics")
    plan = load_run(run_dir).plan
    outcomes = [OutcomeStatus.SUCCESS, OutcomeStatus.TASK_FAILURE]
    for job, status in zip(plan.jobs, outcomes):
        attempt = prepare_attempt(run_dir, job, 1)
        success = status is OutcomeStatus.SUCCESS
        commit_attempt(attempt, AttemptStatus.COMPLETED, {"duration_s": 60.0},
                       (EpisodeResult(job.cases[0].case_id, status, success),))
    metrics = _collect_metrics(run_dir)

    assert metrics["completed"] == 2
    assert metrics["success"] == 1
    assert metrics["task_failure"] == 1
    assert metrics["infra"] == 0
    assert metrics["ep_obs"] == 2
