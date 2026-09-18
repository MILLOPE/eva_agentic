"""One entry for generating case matrices and running serial/concurrent probes.

Holders of this module keep the launch path single: ``eva-agentic`` renders an
immutable experiment/profile per concurrency level, runs each as its own run via
the public CLI, and reports metrics.  Configs and fixtures therefore live in one
place instead of being scattered across ad-hoc scripts.
"""

from __future__ import annotations

import csv
import json
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from eva_agentic.probing import build_cases, parse_task_spec, render_experiment, render_profile, write_cases_jsonl
from eva_agentic.schema import Case


def generate_cases(
    *,
    suite: str,
    tasks: str,
    seed_base: int,
    seed_count: int,
    max_episode_steps: int,
    out: str | Path,
) -> tuple[Case, ...]:
    """Generate a fixed case matrix and write it as JSON Lines."""
    task_ids = parse_task_spec(tasks)
    seeds = tuple(range(seed_base, seed_base + int(seed_count)))
    cases = build_cases(suite, task_ids, seeds, max_episode_steps=max_episode_steps)
    write_cases_jsonl(out, cases)
    return cases


def run_probe(
    *,
    experiment: str | Path,
    cases: str | Path,
    frameworks: str | Path,
    profile: str | Path,
    concurrency: str,
    tag: str,
    runs_root: str | Path,
    work_dir: str | Path,
    out: str | None,
    dry_run: bool,
) -> int:
    """Render and run one probe experiment per concurrency level."""
    experiment_template = _load_yaml(experiment)
    base_profile = _load_yaml(profile)
    frameworks_data = _load_yaml(frameworks)
    cases_file = str(Path(cases).resolve())

    levels = sorted({int(x) for x in concurrency.split(",")})
    if not levels:
        raise ValueError("concurrency must contain at least one level")

    requirement = _requirement(frameworks_data, experiment_template)
    cleaned_tag = _sanitize_tag(tag)
    runs_root_path = Path(runs_root)
    work_dir_path = Path(work_dir)
    work_dir_path.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    failure = False
    for level in levels:
        exp_dict = render_experiment(
            experiment_template, tag=cleaned_tag, cases_file=cases_file, max_jobs=level
        )
        prof_dict = render_profile(base_profile, requirement, max_jobs=level)
        exp_path = work_dir_path / f"c{level}" / "experiment.yaml"
        prof_path = work_dir_path / f"c{level}" / "profile.yaml"
        _dump_yaml(exp_path, exp_dict)
        _dump_yaml(prof_path, prof_dict)

        run_id = f"probe-{cleaned_tag}-c{level}-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
        planned = _n_cases(cases_file)
        if dry_run:
            rows.append(_row(run_id + " (DRY-RUN)", planned))
            continue

        run_dir = _init_run(exp_path, runs_root_path, run_id)
        _run_eva(run_dir, frameworks, prof_path)
        metrics = _collect_metrics(run_dir)
        rows.append(_row(run_id, planned, **metrics))
        failure = failure or bool(metrics["infra"])

    _emit_csv(rows, out)
    return 1 if failure else 0


def _load_yaml(path: str | Path) -> dict:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{source}: expected a YAML mapping")
    return data


def _dump_yaml(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(data), handle, sort_keys=False)


def _n_cases(cases_file: str) -> int:
    count = 0
    with open(cases_file, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _requirement(frameworks_data: dict, experiment_template: dict) -> dict[str, int]:
    participants = experiment_template.get("participants") or []
    wanted = str(participants[0]) if participants else "rpent_libero"
    for spec in frameworks_data.get("frameworks") or []:
        if str(spec.get("name")) == wanted:
            return {str(k): int(v) for k, v in (spec.get("resources") or {}).items()}
    raise ValueError(f"no framework declaration for participant: {wanted}")


def _sanitize_tag(tag: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]", "-", tag or "").strip("-_.")
    return clean or "probe"


def _init_run(exp_path: Path, runs_root: Path, run_id: str) -> Path:
    result = _run(
        ["-m", "eva_agentic", "init", "--experiment", str(exp_path),
         "--runs-root", str(runs_root), "--run-id", run_id]
    )
    try:
        return Path(json.loads(result.stdout)["run_dir"])
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeError(f"init did not return a run_dir: {result.stdout}") from error


def _run_eva(run_dir: Path, frameworks: str | Path, prof_path: Path) -> None:
    result = _run(
        ["-m", "eva_agentic", "run", "--run-dir", str(run_dir),
         "--frameworks", str(frameworks), "--profile", str(prof_path)]
    )
    if result.returncode != 0:
        print(result.stdout, end="", file=sys.stderr)
        print(result.stderr, end="", file=sys.stderr)
        raise RuntimeError(f"eva run failed with exit {result.returncode}")


def _run(argv: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *argv], capture_output=True, text=True)


def _collect_metrics(run_dir: Path) -> dict:
    counts = {"completed": 0, "success": 0, "task_failure": 0, "invalid": 0, "infra": 0}
    durations: list[float] = []
    terminal: dict[str, Path] = {}
    for path in run_dir.glob("participants/*/jobs/*/attempts/*/result.json"):
        terminal[path.parents[2].name] = path
    for path in terminal.values():
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("status") == "completed" and isinstance(data.get("results"), list) and data["results"]:
            counts["completed"] += 1
            status = data["results"][0].get("status")
            key = {"success": "success", "task_failure": "task_failure",
                   "invalid": "invalid", "infrastructure_failure": "infra"}.get(status)
            if key:
                counts[key] += 1
        elif data.get("status") != "completed":
            counts["infra"] += 1
        duration = (data.get("process") or {}).get("duration_s")
        if isinstance(duration, (int, float)):
            durations.append(float(duration))
    return {
        **counts,
        "ep_median_s": statistics.median(durations) if durations else None,
        "ep_max_s": max(durations) if durations else None,
        "ep_obs": len(durations),
    }


def _row(run_id: str, planned: int, **metrics: object) -> dict[str, object]:
    return {
        "run_id": run_id,
        "planned": planned,
        "completed": metrics.get("completed"),
        "success": metrics.get("success"),
        "task_failure": metrics.get("task_failure"),
        "invalid": metrics.get("invalid"),
        "infra": metrics.get("infra"),
        "ep_median_s": metrics.get("ep_median_s"),
        "ep_max_s": metrics.get("ep_max_s"),
        "ep_obs": metrics.get("ep_obs"),
    }


def _emit_csv(rows: list[dict], out: str | None) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    stream = None
    if out:
        stream = open(out, "w", encoding="utf-8", newline="")
    else:
        stream = sys.stdout
    try:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if out:
            stream.close()
