"""End-to-end offline tests for the capacity-probe CLI scripts.

These run the real ``scripts/`` executables but only in ``--dry-run`` (no
services, no scheduling).  They guard the template placeholder rendering and
case generation against regressions like YAML flow-mapping surprises.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GEN = REPO_ROOT / "scripts" / "gen_libero_cases.py"
PROBE = REPO_ROOT / "scripts" / "probe_concurrency.py"

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
        [sys.executable, *argv], cwd=REPO_ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, f"{argv[0]} failed: {result.stderr}\n{result.stdout}"
    return result.stdout


def test_gen_cases(tmp_path) -> None:
    cases = tmp_path / "cases.jsonl"
    _run_cli([str(GEN), "--suite", "libero_object", "--tasks", "0,2",
              "--seed-base", "0", "--seed-count", "2",
              "--max-episode-steps", "500", "--out", str(cases)])
    lines = cases.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    assert cases.read_text(encoding="utf-8").splitlines()[0].count("libero_object") >= 1


def test_probe_dry_run_renders_configs(tmp_path) -> None:
    exp = _write(tmp_path, "experiment.yaml", EXPERIMENT_TEMPLATE)
    frameworks = _write(tmp_path, "frameworks.yaml", FRAMEWORKS)
    profile = _write(tmp_path, "profile.yaml", PROFILE)
    cases = _write(tmp_path, "cases.jsonl", '{"case_id":"libero_object:t0:s0"}\n')

    out = _run_cli(
        [str(PROBE), "--experiment", str(exp), "--cases", str(cases),
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
