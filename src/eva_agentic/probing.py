"""Pure helpers for building reproducible capacity-probe inputs.

These functions never touch services, GPUs, or scheduling.  They translate a
probe specification (suite, tasks, seeds, concurrency levels) into immutable
case lists and per-concurrency experiment/profile renderings that the thin
CLI wrappers in ``scripts/`` hand to the eva CLI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from eva_agentic.schema import Case

_KEYS = ("{CASES_FILE}", "{MAX_JOBS}", "{TAG}")


def parse_task_spec(text: str) -> tuple[int, ...]:
    """Parse a task spec like ``0,2-4,10`` into a sorted unique int tuple."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("task spec must be a non-empty string")
    seen: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"empty task token in spec: {text!r}")
        if "-" in part:
            left, _, right = part.partition("-")
            if not left.isdigit() or not right.isdigit():
                raise ValueError(f"invalid task range {part!r} in spec: {text!r}")
            start, end = int(left), int(right)
            if start > end:
                raise ValueError(f"task range is descending: {start}-{end}")
            seen.update(range(start, end + 1))
        else:
            if not part.isdigit():
                raise ValueError(f"invalid task token {part!r} in spec: {text!r}")
            seen.add(int(part))
    if not seen:
        raise ValueError(f"task spec produced no tasks: {text!r}")
    return tuple(sorted(seen))


def build_cases(
    suite: str,
    task_ids: Sequence[int],
    seeds: Sequence[int],
    max_episode_steps: int,
) -> tuple[Case, ...]:
    """Build an ordered, de-duplicated Case list for one LIBERO suite."""
    if not suite:
        raise ValueError("suite must not be empty")
    if not task_ids:
        raise ValueError("task_ids must not be empty")
    if isinstance(max_episode_steps, bool) or not isinstance(max_episode_steps, int):
        raise ValueError("max_episode_steps must be an integer")
    if max_episode_steps <= 0:
        raise ValueError("max_episode_steps must be positive")
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds):
        raise ValueError("seeds must be integers")
    if any(not isinstance(task, int) or isinstance(task, bool) or task < 0 for task in task_ids):
        raise ValueError("task ids must be non-negative integers")

    cases: list[Case] = []
    for task in task_ids:
        for seed in seeds:
            cases.append(
                Case(
                    case_id=f"{suite}:t{task}:s{seed}",
                    task_id=f"{suite}:{task}",
                    seed=seed,
                    initialization={
                        "suite": suite,
                        "task_id": task,
                        "max_episode_steps": max_episode_steps,
                    },
                )
            )
    # Case ids are unique by construction, but validate explicitly so schema
    # and plan uniqueness checks stay consistent.
    if len({c.case_id for c in cases}) != len(cases):
        raise ValueError("case_id collision after expansion")
    return tuple(cases)


def render_experiment(
    template: Mapping[str, Any],
    *,
    tag: str,
    cases_file: str | Path,
    max_jobs: int,
) -> dict[str, Any]:
    """Render a probe experiment mapping, substituting placeholder tokens."""
    if isinstance(max_jobs, bool) or not isinstance(max_jobs, int) or max_jobs <= 0:
        raise ValueError("max_jobs must be a positive integer")
    if not tag:
        raise ValueError("tag must not be empty")
    if not cases_file:
        raise ValueError("cases_file must not be empty")

    data = _clone(template)
    execution = data.setdefault("execution", {})
    if not isinstance(execution, dict):
        raise ValueError("experiment template execution must be a mapping")
    raw_max_jobs = execution.get("max_jobs")
    if raw_max_jobs != "{MAX_JOBS}" and (
        isinstance(raw_max_jobs, bool) or not isinstance(raw_max_jobs, int)
    ):
        raise ValueError("execution.max_jobs must be {MAX_JOBS} or an integer")
    execution["max_jobs"] = max_jobs

    data = _substitute(data, {"{CASES_FILE}": str(cases_file), "{TAG}": tag})
    _reject_leftover(data)
    return data


def render_profile(
    base: Mapping[str, Any],
    requirement: Mapping[str, int],
    *,
    max_jobs: int,
) -> dict[str, Any]:
    """Render a probe profile: expand slot lists so ``max_jobs`` jobs fit."""
    if isinstance(max_jobs, bool) or not isinstance(max_jobs, int) or max_jobs <= 0:
        raise ValueError("max_jobs must be a positive integer")
    data = _clone(base)
    slots = dict(data.get("resource_slots", {}))
    for kind, count in requirement.items():
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"resource requirement {kind} must be a positive integer")
        slots[str(kind)] = list(range(max_jobs * count))
    data["resource_slots"] = slots
    return data


def _substitute(value: Any, mapping: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        result = value
        for token, replacement in mapping.items():
            result = result.replace(token, replacement)
        return result
    if isinstance(value, Mapping):
        return {str(k): _substitute(v, mapping) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_substitute(item, mapping) for item in value]
    return value


def _reject_leftover(value: Any) -> None:
    if isinstance(value, str):
        remaining = [token for token in _KEYS if token in value]
        if remaining:
            raise ValueError(f"unsubstituted placeholder(s) {sorted(remaining)} in template")
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_leftover(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_leftover(item)


def _clone(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _clone(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone(item) for item in value]
    if isinstance(value, tuple):
        return [_clone(item) for item in value]
    return value


def write_cases_jsonl(path: str | Path, cases: Sequence[Case]) -> Path:
    """Write an ordered, de-duplicated case list as JSON Lines."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n")
    return destination
