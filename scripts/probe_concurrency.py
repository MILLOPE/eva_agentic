#!/usr/bin/env python3
"""Run one fixed case list at several concurrency levels as separate eva runs.

Each concurrency level gets its own immutable run (fresh run id) and its own
rendered experiment/profile, so no attempt is shared or overwritten.  The GPU
and VLA/SAM3 services live where the base profile already points them; probes
must be launched on the host that can reach those endpoints (see README).

Usage (on the GPU-capable host that owns the services):
    scripts/with-rpent-vllm-user-env.sh ./.venv/bin/python scripts/probe_concurrency.py \
        --experiment examples/rpent-vllm-user-libero.probe.experiment.yaml \
        --cases /tmp/libero-matrix.jsonl \
        --frameworks examples/rpent-vllm-user-libero.frameworks.example.yaml \
        --profile profiles/rpent-vllm-user-libero.yaml \
        --concurrency 1,2,4,8 --tag capacity --out runs/probe-capacity.csv

Dry-run renders configs and prints the plan without touching services:
    ./.venv/bin/python scripts/probe_concurrency.py \
        --experiment examples/rpent-vllm-user-libero.probe.experiment.yaml \
        --cases /tmp/libero-matrix.jsonl \
        --frameworks examples/rpent-vllm-user-libero.frameworks.example.yaml \
        --profile profiles/rpent-vllm-user-libero.yaml \
        --concurrency 1,2,4,8 --tag dry --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

from eva_agentic.probing import render_experiment, render_profile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="probe_concurrency")
    parser.add_argument("--experiment", required=True, help="probe experiment template (YAML)")
    parser.add_argument("--cases", required=True, help="generated cases file (JSON-Lines)")
    parser.add_argument("--frameworks", required=True, help="framework declaration (YAML)")
    parser.add_argument("--profile", required=True, help="base local profile (YAML)")
    parser.add_argument("--concurrency", required=True, help="comma-separated levels, e.g. 1,2,4,8")
    parser.add_argument("--tag", default="probe", help="tag embedded in run ids")
    parser.add_argument("--runs-root", default="runs", help="runs root directory")
    parser.add_argument("--work-dir", default="runs/.probe", help="staging dir for rendered configs")
    parser.add_argument("--out", help="optional CSV output path")
    parser.add_argument("--dry-run", action="store_true", help="render configs only, do not run")
    args = parser.parse_args(argv)

    experiment_template = _load_yaml(args.experiment)
    base_profile = _load_yaml(args.profile)
    frameworks_data = _load_yaml(args.frameworks)
    cases_file = str(Path(args.cases).resolve())
    concurrency = sorted({int(x) for x in args.concurrency.split(",")})
    if not concurrency:
        parser.error("--concurrency must contain at least one level")

    prohibition = _requirement(frameworks_data, experiment_template)
    tag = _sanitize_tag(args.tag)
    runs_root = Path(args.runs_root)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    rows, failure = [], False
    for k in concurrency:
        exp_dict = render_experiment(
            experiment_template, tag=tag, cases_file=cases_file, max_jobs=k
        )
        prof_dict = render_profile(base_profile, prohibition, max_jobs=k)
        exp_path = work_dir / f"c{k}" / "experiment.yaml"
        prof_path = work_dir / f"c{k}" / "profile.yaml"
        exp_path.parent.mkdir(parents=True, exist_ok=True)
        with exp_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(exp_dict, handle, sort_keys=False)
        with prof_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(prof_dict, handle, sort_keys=False)

        run_id = f"probe-{tag}-c{k}-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
        if args.dry_run:
            rows.append(_row(run_id + " (DRY-RUN)", planned=_n_cases(cases_file)))
            continue

        run_dir = _init_run(exp_path, runs_root, run_id)
        _run_eva(run_dir, args.frameworks, prof_path)
        metrics = _collect_metrics(run_dir)
        rows.append(_row(run_id, planned=_n_cases(cases_file), **metrics))
        failure = failure or metrics["infra"] > 0

    _emit_csv(rows, args.out)
    return 1 if failure else 0


def _load_yaml(path: str) -> dict:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{source}: expected a YAML mapping")
    return data


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
        payload = json.loads(result.stdout)
        return Path(payload["run_dir"])
    except (json.JSONDecodeError, KeyError) as error:
        raise RuntimeError(f"init did not return a run_dir: {result.stdout}") from error


def _run_eva(run_dir: Path, frameworks: str, prof_path: Path) -> None:
    result = _run(
        ["-m", "eva_agentic", "run", "--run-dir", str(run_dir),
         "--frameworks", frameworks, "--profile", str(prof_path)]
    )
    if result.returncode != 0:
        print(result.stdout, end="", file=sys.stderr)
        print(result.stderr, end="", file=sys.stderr)
        raise RuntimeError(f"eva run failed with exit {result.returncode}")


def _run(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *argv], capture_output=True, text=True)


def _collect_metrics(run_dir: Path) -> dict:
    counts = {"completed": 0, "success": 0, "task_failure": 0,
              "invalid": 0, "infra": 0}
    durations: list[float] = []
    results = list(run_dir.glob("*/jobs/*/attempts/*/result.json"))
    terminal: dict[str, Path] = {}
    for path in results:
        job_dir = path.parents[2]
        terminal[job_dir.name] = path  # last attempt wins on ordering
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
    if out:
        with open(out, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
