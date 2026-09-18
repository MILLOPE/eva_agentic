#!/usr/bin/env bash
# Run a command under the dedicated plotting environment (seaborn/matplotlib).
# Usage: scripts/with-eva-viz-env.sh ./.venv/bin/eva-agentic visualize --runs-root runs
set -eo pipefail

repo_root="$(cd -- "$(dirname -- "$0")/.." && pwd)"
env_dir="$repo_root/.envs/eva-viz"

if [[ ! -x "$env_dir/bin/python" ]]; then
  echo "missing eva-viz interpreter: $env_dir/bin/python" >&2
  echo "create it with:  uv venv --python 3.11 $env_dir && uv pip install --python $env_dir/bin/python numpy pandas matplotlib seaborn" >&2
  exit 2
fi

export VIRTUAL_ENV="$env_dir"
export PATH="$env_dir/bin:$PATH"

if [[ "$#" == 0 ]]; then
  echo "usage: $0 <command> [args ...]" >&2
  exit 2
fi

exec "$@"
