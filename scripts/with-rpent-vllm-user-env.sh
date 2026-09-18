#!/usr/bin/env bash

set -eo pipefail

# Use the eva-local runtime config: all connection and signing variables come
# from .envs/eva-rpent-vllm-user/.env (git-ignored), not from an external
# project.  This env file carries no PYTHONPATH or interpreter selectors, so
# nothing stale can decide which RPent eva launches.
repo_root="$(cd -- "$(dirname -- "$0")/.." && pwd)"
if [[ -n "$RPENT_SOURCE_ENV" ]]; then
  source_env="$RPENT_SOURCE_ENV"
else
  source_env="$repo_root/.envs/eva-rpent-vllm-user/.env"
fi

if [[ ! -f "$source_env" ]]; then
  echo "missing eva-local RPent env: $source_env" >&2
  echo "create it from examples/rpent-vllm-user-libero.env.example:" >&2
  echo "  mkdir -p \"$repo_root/.envs/eva-rpent-vllm-user\"" >&2
  echo "  cp examples/rpent-vllm-user-libero.env.example \"$source_env\"" >&2
  exit 2
fi

# Bring in only the connection/signing variables from the eva-local .env.
set -a
source "$source_env"
set +a

# Defensive: clear any interpreter/PYTHONPATH selectors a legacy override env
# might carry, so the eva-local venv always wins.  The vLLM credential
# variables sourced above are deliberately retained.
unset \
  PYTHONPATH \
  RPENT_MAIN_PYTHON \
  RPENT_CORE_PYTHON \
  RPENT_LIBERO_PYTHON \
  RPENT_SAM3_PYTHON \
  RPENT_UV \
  OPENPI_ROOT \
  OPENPI_PYTHON \
  PI05_CHECKPOINT_PATH \
  SAM3_CHECKPOINT_PATH \
  RPENT_LAUNCH_PLANS_DIR

export RPENT_REPO_ROOT="$repo_root/third_party/frameworks/rpent"
export LIBERO_CONFIG_PATH="$repo_root/.envs/eva-rpent-vllm-user/libero-config"
export LIBERO_ASSET_PATH="$repo_root/.assets/libero"
planner_src="$repo_root/patches/rpent/libero/vllm_user/planner/src"
if [[ ! -d "$planner_src/rpent_vllm_user" ]]; then
  echo "missing vllm_user planner source: $planner_src" >&2
  exit 2
fi
export PYTHONPATH="$planner_src"
mkdir -p "$LIBERO_CONFIG_PATH"

dedicated_env="$repo_root/.envs/eva-rpent-vllm-user"
if [[ ! -x "$dedicated_env/bin/python" ]]; then
  echo "missing dedicated RPent interpreter: $dedicated_env/bin/python" >&2
  exit 2
fi
export VIRTUAL_ENV="$dedicated_env"
export PATH="$dedicated_env/bin:$PATH"

if [[ -n "$RPENT_VLLM_PRIVATE_KEY" && -n "$RPENT_VLLM_SIGNER_URL" ]]; then
  echo "configure exactly one RPENT_VLLM_PRIVATE_KEY or RPENT_VLLM_SIGNER_URL" >&2
  exit 2
fi
if [[ -z "$RPENT_VLLM_PRIVATE_KEY" && -z "$RPENT_VLLM_SIGNER_URL" ]]; then
  echo "missing RPent vLLM signing configuration" >&2
  exit 2
fi
if [[ -n "$RPENT_VLLM_PRIVATE_KEY" && ! -f "$RPENT_VLLM_PRIVATE_KEY" ]]; then
  echo "RPENT_VLLM_PRIVATE_KEY does not point to a readable file" >&2
  exit 2
fi
if [[ "$#" == 0 ]]; then
  echo "usage: $0 <command> [args ...]" >&2
  exit 2
fi

exec "$@"
