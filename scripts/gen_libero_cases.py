#!/usr/bin/env python3
"""Generate a fixed, de-duplicated LIBERO case matrix for large-scale runs.

Usage:
    ./.venv/bin/python scripts/gen_libero_cases.py \
        --suite libero_object --tasks 0,1,4-6 --seed-base 0 --seed-count 3 \
        --max-episode-steps 500 --out runs/libero-matrix.jsonl
"""

from __future__ import annotations

import argparse

from eva_agentic.probing import build_cases, parse_task_spec, write_cases_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gen_libero_cases")
    parser.add_argument("--suite", required=True, help="LIBERO suite, e.g. libero_object")
    parser.add_argument(
        "--tasks", required=True, help="task ids as list/range, e.g. 0,1,4-6"
    )
    parser.add_argument("--seed-base", type=int, default=0, help="first seed (default 0)")
    parser.add_argument("--seed-count", type=int, default=3, help="number of seeds (default 3)")
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--out", required=True, help="output JSON-Lines path")
    args = parser.parse_args(argv)

    tasks = parse_task_spec(args.tasks)
    seeds = tuple(range(args.seed_base, args.seed_base + args.seed_count))
    cases = build_cases(
        suite=args.suite,
        task_ids=tasks,
        seeds=seeds,
        max_episode_steps=args.max_episode_steps,
    )
    write_cases_jsonl(args.out, cases)
    print(f"wrote {len(cases)} cases to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
