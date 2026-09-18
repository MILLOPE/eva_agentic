# RPent vllm_user Planner Integration Design

## Status

Design approved in conversation; implementation has not started.

## Goal

Run one reproducible LIBERO smoke case through eva_agentic using the RPent
checkout already stored at `third_party/frameworks/rpent` and the custom
`vllm_user` planner migrated from the separate RPent correction workspace.
The task result is successful only when the RPent-native audit contains the
boolean field `terminated: true`.

## Context and fixed boundaries

- The runtime framework remains `third_party/frameworks/rpent`.
- The shared benchmark remains `third_party/benchmarks/libero`.
- `/share/repos/RPent-correction` is a source workspace for the planner only;
  eva does not use it as the RPent runtime and does not modify it.
- The planner is maintained in eva as a manually replaceable migration
  snapshot under `patches/rpent/libero/vllm_user/`.
- The third-party RPent tree is not rewritten into a new framework and its
  control loop, Toolkit, environment server, VLA/SAM3 lifecycle, transcript,
  audit, and recipe ownership remain native to RPent.
- No independent robot framework is connected or compared in this phase.
- No batch scheduler, automatic planner synchronization, or automatic code
  update is introduced.

The current RPent baseline does not list `vllm_user` in its CLI planner choices
and its `build_planner()` function has no `vllm_user` branch. Therefore a
configuration-only change cannot satisfy the requested command. The patch
must add the smallest registration and compatibility surface needed to expose
the migrated planner through the existing RPent `Planner`/`PlannerResult`
contract.

## Approaches considered

### Recommended: patch the current RPent planner seam

Store the migrated planner package and a small RPent registration patch under
`patches/rpent/libero/vllm_user/`. Apply that patch to the current RPent source
baseline, install the resulting RPent and planner package into a dedicated
eva-local framework environment, and let the existing eva RPent adapter launch
the native command with `--planner vllm_user`.

This preserves the current framework and keeps planner ownership separate from
eva core. Any compatibility changes stay inside the patch's planner migration
or registration layer; the RPent control loop is not reimplemented.

### Rejected: use the correction checkout as the framework runtime

This would make `/share/repos/RPent-correction` the actual RPent source and
would mix the user's unrelated development state into eva experiments. It also
violates the requirement that the current eva RPent framework remains the
framework under test.

### Rejected: bypass the planner choice

Calling the existing `api` planner with a custom endpoint, monkey-patching the
process at launch, or adding planner logic to eva would not exercise the
official `--planner vllm_user` interface and would make the result incomparable
to the requested planner run.

## Repository layout

The migration source of truth in eva will be:

```text
patches/rpent/libero/vllm_user/
├── README.md
├── planner/
│   ├── pyproject.toml
│   ├── src/rpent_vllm_user/
│   └── tests/
└── rpent-vllm-user.patch
```

`README.md` records the RPent baseline revision, the source checkout and
planner revision used for the migration, the reason for the patch, the
application procedure, dependency requirements, and known compatibility
limits. `planner/` is the manually replaceable planner snapshot. The unified
patch contains only the current RPent registration/check changes and any
minimal compatibility edits proven necessary by tests; it does not contain a
second RPent tree.

The top-level patch documentation will clarify that this path is a
framework-planner integration scoped to the RPent+LIBERO experiment. It is not
a copied benchmark variant.

## Runtime and configuration

Use a dedicated local RPent environment under `eva_agentic/.envs/` that is
based on `third_party/frameworks/rpent` and installs the migrated planner
package. It must not install or expose the correction workspace's RPent or its
planner integration.

The versioned framework declaration keeps the existing participant name
`rpent_libero` so the existing eva adapter is reused. Its native command
contains at least:

```text
-m rpent.cli.main
--planner vllm_user
--model <configured-model-id>
--memory-profile hf
--max-turns 20
--planner-timeout-s 1200
```

The adapter appends the fixed LIBERO task arguments and the configured VLA and
SAM3 endpoints. Local profiles provide the interpreter, resource slots,
endpoint values, LIBERO assets, and vLLM signing configuration without
committing secrets.

The initial smoke case is:

```text
suite: libero_object
task_id: 2
seed: 0
max_episode_steps: 500
```

## Execution and evidence flow

1. Prepare/apply the recorded RPent planner patch against the current local
   RPent baseline.
2. Install RPent and the planner snapshot into the dedicated framework
   environment.
3. Run static import/CLI checks and the signed vLLM connectivity check.
4. Check the VLA and SAM3 RPent JSON-RPC `healthz` methods through `/call`.
5. Initialize a new immutable eva run with one LIBERO case.
6. Let eva launch the native RPent process and preserve its actual command,
   cwd, resource allocation, stdout, stderr, and runtime status.
7. Preserve RPent's native output directory, including transcript, audit, and
   recipe when produced.
8. Parse the audit after the process exits. A non-zero process exit is an
   infrastructure failure; a missing, malformed, or mismatched audit is an
   invalid result; a valid audit with `terminated: false` is a task failure; a
   valid audit with `terminated: true` is success.

The run evidence must identify the task, seed, initialization conditions,
episode and planner budgets, model, RPent baseline/patch revision, planner
snapshot revision, actual launch command, and raw logs. Secrets are represented
only by variable names, paths, or redacted values.

## eva changes

Keep the existing RPent LIBERO adapter as the owner of native argument mapping
and audit parsing. Add only the generic evidence needed to snapshot the
framework declaration/profile and the applied source/patch metadata into the
run; do not add vLLM request logic, Toolkit calls, or RPent control behavior to
eva. The parser continues to use `rpent.audit.terminated` as its success
source.

## Verification

Before a real smoke run:

- the patch applies cleanly to the recorded RPent baseline;
- `rpent --help` lists `vllm_user`;
- the planner package imports in the selected interpreter;
- the planner's signed vLLM probe succeeds;
- VLA and SAM3 health checks succeed;
- unit tests cover planner registration, command mapping, audit success,
  audit failure, invalid/mismatched audit, and process failure paths.

The real smoke run is not considered verified until the native audit is read
and its `terminated` value is recorded. Process exit code alone is never a
success signal.

## Non-goals and limitations

- No comparison with the independent framework.
- No automatic copying or updating of planner versions.
- No migration of unrelated correction-workspace changes.
- No claim that the vLLM, VLA, SAM3, or LIBERO services are healthy until the
  checks are actually executed.
- No claim of benchmark fairness beyond this one fixed smoke configuration.
