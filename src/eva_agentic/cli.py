"""Small command-line interface for frozen native evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from eva_agentic.adapters import RatsLiberoAdapter, RpentLiberoAdapter
from eva_agentic.experiment import load_experiment
from eva_agentic.frameworks import FrameworkProfile, FrameworkSpec, load_framework_profile, load_framework_specs
from eva_agentic.resources import ResourceAllocator
from eva_agentic.runner import run_adapter_job
from eva_agentic.scheduler import run_jobs
from eva_agentic.store import initialize_run, load_run
from eva_agentic.summary import summarize_run


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eva-agentic")
    subcommands = parser.add_subparsers(dest="command", required=True)
    init = subcommands.add_parser("init")
    init.add_argument("--experiment", required=True)
    init.add_argument("--runs-root", required=True)
    init.add_argument("--run-id")
    run = subcommands.add_parser("run")
    run.add_argument("--run-dir", required=True)
    run.add_argument("--frameworks", required=True)
    run.add_argument("--profile", required=True)
    run.add_argument("--lock-root")
    summary = subcommands.add_parser("summary")
    summary.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    if args.command == "init":
        run_dir = initialize_run(args.runs_root, load_experiment(args.experiment), args.run_id)
        print(json.dumps({"run_dir": str(run_dir)}))
        return 0
    if args.command == "summary":
        print(json.dumps(summarize_run(args.run_dir), indent=2))
        return 0
    return _run(args.run_dir, args.frameworks, args.profile, args.lock_root)


def _run(run_dir: str, specs_path: str, profile_path: str, lock_root: str | None) -> int:
    inputs = load_run(run_dir)
    specs = load_framework_specs(specs_path)
    profile = load_framework_profile(profile_path)
    adapters = {name: _adapter_for(spec) for name, spec in specs.items()}
    missing = sorted(set(inputs.experiment.participants) - set(adapters))
    if missing:
        raise ValueError(f"framework declarations missing participants: {missing}")

    def runner(job, attempt_dir, grants=()):
        return run_adapter_job(
            adapters[job.participant],
            profile,
            inputs.experiment.execution.job_timeout_s,
            job,
            attempt_dir,
            grants,
        )

    allocator = (
        ResourceAllocator(profile.resource_slots, lock_root or str(Path(run_dir) / ".eva" / "resources"))
        if profile.resource_slots
        else None
    )
    requirements = {job.job_id: specs[job.participant].resources for job in inputs.plan.jobs}
    if allocator is None and any(requirements.values()):
        raise ValueError("framework resource requests require profile.resource_slots")
    statuses = run_jobs(
        inputs.run_dir,
        inputs.plan.jobs,
        runner,
        inputs.experiment.execution.max_jobs,
        resource_allocator=allocator,
        resource_requirements=requirements,
    )
    print(json.dumps({key: value.value for key, value in statuses.items()}, indent=2))
    return 0


def _adapter_for(spec: FrameworkSpec):
    if spec.name == "rats_libero":
        return RatsLiberoAdapter(spec)
    if spec.name == "rpent_libero":
        return RpentLiberoAdapter(spec)
    raise ValueError(f"no result parser adapter registered for framework: {spec.name}")
