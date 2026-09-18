# RPent vllm_user Planner Integration Implementation Plan

> **For agentic workers:** Use the available `subagent-driven-development` skill or execute the tasks inline with checkpoints. Each task has an independent verification gate.

**Goal:** Connect the migrated signed `vllm_user` planner to eva's existing RPent+LIBERO framework baseline, preserve auditable provenance, and complete one real LIBERO smoke run whose only task-success signal is RPent audit `terminated=true`.

**Architecture:** Keep `third_party/frameworks/rpent` as the framework baseline and keep the migrated planner plus its RPent registration changes under `patches/rpent/libero/vllm_user`. Install the patched local RPent and planner snapshot in a dedicated eva-local environment. Keep eva responsible for command construction, evidence, and audit parsing; keep planner, Toolkit, services, simulation, transcript, and audit ownership in RPent.

**Tech Stack:** Python 3.10, uv, setuptools editable installs, RPent `Planner`/`PlannerResult`, signed vLLM HTTP client, YAML/JSONL experiment declarations, pytest, and RPent JSON-RPC health checks.
**Tech Stack:** Python 3.11 dedicated env, uv, setuptools editable installs, RPent `Planner`/`PlannerResult`, signed vLLM HTTP client, YAML/JSONL experiment declarations, pytest, and RPent JSON-RPC health checks.

**Spec:** `docs/superpowers/specs/2026-09-17-rpent-vllm-user-integration-design.md`

## Global Constraints

- GPU-bound LIBERO, VLA, and SAM3 services run on the designated GPU host; a CPU-only eva host is an orchestration location, not a valid RPent execution host.
- Use `/share/repos/eva_agentic/third_party/frameworks/rpent` as the RPent runtime baseline; do not use `/share/repos/RPent-correction` as the runtime.
- Copy only `/share/repos/RPent-correction/integrations/vllm_user` as the planner migration source; do not copy unrelated correction-workspace changes.
- Store the migration and registration delta under `patches/rpent/libero/vllm_user/`; do not commit a second RPent tree.
- Invoke the official native command with `--planner vllm_user`.
- Keep planner, Toolkit, VLA/SAM3, simulation, control loop, transcript, audit, and recipe logic out of eva core.
- Fix one case: `libero_object`, task `2`, seed `0`, `max_episode_steps=500`.
- Fix planner inputs: `max_turns=20`, `planner_timeout_s=1200`, explicit model id, and `memory_profile=hf`.
- Treat non-zero process exit as infrastructure failure, missing/malformed/mismatched audit as invalid, `terminated=false` as task failure, and only `terminated=true` as success.
- Never put private key contents or signer tokens in source control or run evidence.
- Every real run gets a new run id and preserves all attempts.

---

### Task 1: Freeze the planner migration source and patch contract

**Files:**
- Create: `patches/rpent/libero/vllm_user/README.md`
- Create: `patches/rpent/libero/vllm_user/source-manifest.yaml`
- Create: `patches/rpent/libero/vllm_user/planner/pyproject.toml`
- Create: `patches/rpent/libero/vllm_user/planner/src/rpent_vllm_user/`
- Create: `patches/rpent/libero/vllm_user/planner/tests/`
- Modify: `patches/README.md`

**Interfaces:**
- Consumes: `/share/repos/RPent-correction/integrations/vllm_user` and the current RPent revision recorded in `third_party/sources.yaml`.
- Produces: A manually replaceable planner package snapshot and a manifest identifying the RPent baseline, planner source path, source revision, and migration scope.

- [ ] **Step 1: Record the source facts before copying.**

Run:

```bash
cd /share/repos/eva_agentic
git -C /share/repos/RPent-correction rev-parse HEAD
git -C /share/repos/RPent-correction status --short -- integrations/vllm_user
rg -n 'id: rpent|revision:' third_party/sources.yaml
```

Expected: the manifest records the exact correction checkout revision, whether the planner subdirectory is dirty, and the current eva RPent baseline revision `849143b`.

- [ ] **Step 2: Copy only the planner package.**

Copy the source package files from `/share/repos/RPent-correction/integrations/vllm_user` into `patches/rpent/libero/vllm_user/planner`, excluding `__pycache__`, `.pytest_cache`, `.ruff_cache`, and generated `*.egg-info` files. Do not copy `/share/repos/RPent-correction/rpent`, `robots`, `third_party`, or unrelated integration directories.

- [ ] **Step 3: Write the migration README.**

Document the exact copy source, RPent baseline, why the planner needs a registration patch, the manual replacement procedure, the editable-install command, the required `RPENT_VLLM_*` variables, the no-secret rule, and the fact that the planner is not runtime-loaded from the correction checkout.

- [ ] **Step 4: Update the patch catalog convention.**

Clarify in `patches/README.md` that `patches/<framework>/<benchmark>/` may contain a narrowly scoped framework-planner integration bundle, provided it names the framework baseline and does not copy a benchmark or framework tree.

- [ ] **Step 5: Verify the snapshot boundary.**

Run:

```bash
cd /share/repos/eva_agentic
rg --files patches/rpent/libero/vllm_user/planner | sort
rg -n 'Rpent-correction|third_party/frameworks/rpent|revision|private|secret' patches/rpent/libero/vllm_user/README.md patches/rpent/libero/vllm_user/source-manifest.yaml
```

Expected: only planner package/test files are present; the README and manifest identify sources without containing credentials.

---

### Task 2: Make the migrated planner compatible with the current RPent public contract

**Files:**
- Modify: `patches/rpent/libero/vllm_user/planner/src/rpent_vllm_user/__init__.py`
- Modify: `patches/rpent/libero/vllm_user/planner/src/rpent_vllm_user/planner.py`
- Modify: `patches/rpent/libero/vllm_user/planner/src/rpent_vllm_user/history.py`
- Modify: `patches/rpent/libero/vllm_user/planner/src/rpent_vllm_user/liveness.py`
- Modify: `patches/rpent/libero/vllm_user/planner/src/rpent_vllm_user/multi_tool.py`
- Modify: `patches/rpent/libero/vllm_user/planner/src/rpent_vllm_user/result_projection.py`
- Modify: `patches/rpent/libero/vllm_user/planner/src/rpent_vllm_user/workspace.py`
- Create: `patches/rpent/libero/vllm_user/planner/src/rpent_vllm_user/compat.py`
- Create: `patches/rpent/libero/vllm_user/planner/tests/test_current_rpent_compat.py`

**Interfaces:**
- Consumes: Current RPent `PlannerResult`, `Toolkit.get_tools_spec()`, `Toolkit.execute_tool()`, `DashboardInteractionPort`, `DashboardEventSink`, and `ToolResult`.
- Produces: `rpent_vllm_user.create_planner(...) -> ModelApiUtilsPlanner` whose `solve(...)` satisfies the current RPent `Planner` protocol.

- [ ] **Step 1: Write compatibility tests first.**

Test that the migrated package can import with the current RPent source path and that `create_planner` can be constructed using a fake signed-client configuration without starting a model request. Test that a fake Toolkit exposing only the current baseline methods can execute one parsed tool call and return a current `ToolResult`.

- [ ] **Step 2: Add a planner-local compatibility module.**

Provide local fallbacks for correction-only optional symbols used during import or Dashboard publication, including the default reasoning value and optional catalog/usage/system-context events. Fallback event objects must be inert data objects accepted by the current `DashboardEventSink`; they must not create a second event bus.

- [ ] **Step 3: Move result projection ownership into the migration package.**

Make `history.py` and `planner.py` use the migrated package's projection functions rather than requiring `rpent.planner.result_projection` from the correction RPent. Preserve the existing bounded JSON projection behavior and `ResultProjectionError` failure semantics.

- [ ] **Step 4: Make liveness and workspace features optional for the current baseline.**

Keep workspace mode disabled by default. When disabled, planner construction must not require correction-only Toolkit catalog helpers. When enabled, use only the current Toolkit's public methods or return a structured unsupported result; never add a second Toolkit owner. Keep the planner's signed client, history policy, multi-tool validation, and tool-loop behavior intact.

- [ ] **Step 5: Run the focused migration tests.**

Run:

```bash
cd /share/repos/eva_agentic/patches/rpent/libero/vllm_user/planner
python -m pytest -q tests/test_current_rpent_compat.py
```

Expected: PASS with `PYTHONPATH` pointing at the current RPent baseline and the package `src` directory; no vLLM network request is made.

---

### Task 3: Add the minimal RPent registration patch

**Files:**
- Modify: `third_party/frameworks/rpent/rpent/planner/base.py`
- Modify: `third_party/frameworks/rpent/rpent/cli/main.py`
- Modify: `third_party/frameworks/rpent/rpent/planner/check.py`
- Modify: `third_party/frameworks/rpent/rpent/cli/check_llm.py`
- Create: `third_party/frameworks/rpent/tests/unit_tests/rpent/planner/test_vllm_user_contracts.py`
- Create: `patches/rpent/libero/vllm_user/rpent-vllm-user.patch`

**Interfaces:**
- Consumes: `rpent_vllm_user.create_planner`, current CLI arguments, and current `Planner` protocol.
- Produces: `rpent --planner vllm_user ...` and `rpent-check-llm --planner vllm_user --json` using the current RPent checkout.

- [ ] **Step 1: Add a failing planner factory test.**

Monkeypatch a fake `rpent_vllm_user` module into `sys.modules`, call `build_planner("vllm_user", ...)`, and assert that the factory receives `base_url`, `model`, `max_tokens`, `timeout_s`, `reasoning_effort`, `dashboard_events`, and `no_images`. Assert that the returned object is the fake planner and implements `solve`.

- [ ] **Step 2: Add the lazy `vllm_user` factory branch.**

Add the branch to `build_planner` without importing the optional package at RPent module import time. Use the existing `planner_timeout_s` fallback. Report a clear installation error when `rpent_vllm_user` is absent.

- [ ] **Step 3: Extend the native CLI choices and help.**

Add `vllm_user` to the planner choices and update model/base-url/reasoning help text. Preserve the current `api`, `claude_code`, `codex`, and `task_card` behavior unchanged.

- [ ] **Step 4: Add the vLLM connectivity check.**

Extend the current check request classification and CLI choice with `vllm_user`. Reuse the migrated `SignedVllmClient` for the same identity and short chat request used by the planner. Redact key/token values and return structured status JSON; never make a check pass merely because the package imports.

- [ ] **Step 5: Run current RPent unit tests.**

Run:

```bash
cd /share/repos/eva_agentic/third_party/frameworks/rpent
python -m pytest -q tests/unit_tests/rpent/planner/test_vllm_user_contracts.py tests/unit_tests/rpent/cli/test_main_contracts.py tests/unit_tests/rpent/cli/test_check_llm_contracts.py
```

Expected: the new factory and CLI/check tests pass, and existing planner choices still pass.

- [ ] **Step 6: Generate and inspect the patch artifact.**

Generate a unified diff from the recorded clean RPent baseline to the minimal registration changes and save it as `patches/rpent/libero/vllm_user/rpent-vllm-user.patch`. The patch must not include unrelated RPent files, generated caches, model assets, or the correction workspace.

- [ ] **Step 7: Verify patch replay in a temporary copy.**

Create a temporary copy of `third_party/frameworks/rpent`, apply `rpent-vllm-user.patch`, and rerun the focused RPent tests there. Expected: the patch applies cleanly and produces the same test result without modifying the canonical source during this verification.

---

### Task 4: Add generic framework provenance to eva evidence

**Files:**
- Modify: `src/eva_agentic/frameworks.py`
- Modify: `src/eva_agentic/process.py`
- Modify: `tests/unit/test_frameworks.py`
- Modify: `tests/unit/test_process.py`

**Interfaces:**
- Consumes: optional non-secret `provenance` mapping in a framework declaration.
- Produces: `launch.json` evidence containing source/patch/planner provenance in addition to the actual command, cwd, env keys, resources, and output paths.

- [ ] **Step 1: Add a failing provenance parsing test.**

Load a declaration containing:

```yaml
provenance:
  framework_source: third_party/frameworks/rpent
  framework_revision: 849143b
  integration_patch: patches/rpent/libero/vllm_user
  planner_source_revision: source-manifest
```

Assert that the parsed `FrameworkSpec` preserves these strings and rejects non-string provenance values.

- [ ] **Step 2: Carry provenance into `NativeLaunch`.**

Add an immutable mapping to `FrameworkSpec` and `NativeLaunch`, pass it through `resolve_native_launch`, and keep the existing command-template/resource validation unchanged.

- [ ] **Step 3: Write provenance into `launch.json`.**

Include the non-secret provenance mapping in `_write_launch_record`. Keep secret environment values out of the record; retain the current `env_keys` behavior and any explicit redaction rules.

- [ ] **Step 4: Test launch evidence.**

Build a fake native launch, write its launch record through the existing process test path, and assert that command, cwd, output dir, resource allocation, and provenance all survive serialization.

- [ ] **Step 5: Run the eva unit tests.**

Run:

```bash
cd /share/repos/eva_agentic
./.venv/bin/python -m pytest -q tests/unit/test_frameworks.py tests/unit/test_process.py tests/unit/test_rpent_libero.py
```

Expected: PASS, including existing audit parsing tests.

---

### Task 5: Add the versioned RPent+vllm_user declarations and local profile template

**Files:**
- Create: `examples/rpent-vllm-user-libero.frameworks.example.yaml`
- Create: `examples/rpent-vllm-user-libero.experiment.example.yaml`
- Create: `examples/rpent-vllm-user-libero.cases.example.jsonl`
- Create: `examples/rpent-vllm-user-libero.profile.example.yaml`
- Modify: `README.md`

**Interfaces:**
- Consumes: the existing `rpent_libero` adapter, `FrameworkSpec.provenance`, and the current local RPent source.
- Produces: a one-case experiment declaration whose actual command includes `--planner vllm_user`, explicit model, max turns, and planner timeout.

- [ ] **Step 1: Add the framework declaration.**

Use `backend: python`, workdir `../third_party/frameworks/rpent`, participant name `rpent_libero`, one GPU resource, and command arguments `-m rpent.cli.main --planner vllm_user --model default --memory-profile hf --max-turns 20 --planner-timeout-s 1200`. Add provenance fields for the RPent path, baseline revision, patch path, and planner source manifest.

- [ ] **Step 2: Add the case and experiment files.**

Use `libero_object:2`, seed `0`, suite `libero_object`, task id `2`, and `max_episode_steps: 500`. Keep one participant, one job, debug mode, no infrastructure retries, and outer job timeout longer than the planner timeout.

- [ ] **Step 3: Add the untracked profile template.**

Document the dedicated interpreter, GPU slot, `RPENT_VLA_ENDPOINT`, `RPENT_SAM3_ENDPOINT`, `LIBERO_CONFIG_PATH`, `RPENT_VLLM_BASE_URL`, `RPENT_VLLM_KEY_ID`, `RPENT_VLLM_MODEL`, and exactly one of `RPENT_VLLM_PRIVATE_KEY` or loopback `RPENT_VLLM_SIGNER_URL`. Mark key/token values as local-only and do not put them in the versioned example.

- [ ] **Step 4: Document preparation and run commands.**

Add commands for applying the patch, installing the local RPent and planner packages in the dedicated environment, running import/help/check/health probes, initializing a fresh run id, running eva, and inspecting `launch.json`, stdout, stderr, transcript, audit, recipe, and summary.

- [ ] **Step 5: Validate declarations without services.**

Run:

```bash
cd /share/repos/eva_agentic
./.venv/bin/python -m pytest -q tests/unit/test_experiment_config.py tests/unit/test_frameworks.py
```

Expected: declarations parse, the case is frozen, and provenance remains present before any network or simulator dependency is used.

---

### Task 6: Prepare the dedicated environment and run preflight checks

**Files:**
- Modify only local ignored paths under `.envs/eva-rpent-vllm-user/`.
- Modify only the local ignored profile copied from `examples/rpent-vllm-user.profile.example.yaml`.

**Interfaces:**
- Consumes: patched `third_party/frameworks/rpent`, planner snapshot, local service credentials, VLA/SAM3 endpoints, and compatible LIBERO assets.
- Produces: a runtime in which RPent and `rpent_vllm_user` import from the intended eva-local sources and all required endpoints are independently checked.

- [ ] **Step 1: Create the environment without correction RPent paths.**

Use the repository's configured uv executable to create `.envs/eva-rpent-vllm-user` with Python 3.10. Install the local RPent checkout editable and install `patches/rpent/libero/vllm_user/planner` editable. Install only the planner's declared extra dependency `cryptography` plus the already required local RPent/LIBERO runtime dependencies.

- [ ] **Step 2: Verify import origins.**

Run:

```bash
cd /share/repos/eva_agentic
.envs/eva-rpent-vllm-user/bin/python -c 'import rpent, rpent_vllm_user; print(rpent.__file__); print(rpent_vllm_user.__file__)'
```

Expected: `rpent` resolves under eva's current patched `third_party/frameworks/rpent`, and `rpent_vllm_user` resolves under eva's `patches/rpent/libero/vllm_user/planner`; no `/share/repos/RPent-correction` path appears.

- [ ] **Step 3: Verify the planner CLI registration.**

Run:

```bash
.envs/eva-rpent-vllm-user/bin/python -m rpent.cli.main --help
.envs/eva-rpent-vllm-user/bin/python -m rpent.cli.check_llm --help
```

Expected: both help outputs list `vllm_user`.

- [ ] **Step 4: Run signed vLLM preflight.**

With local credential variables loaded but values kept out of logs, run:

```bash
.envs/eva-rpent-vllm-user/bin/rpent-check-llm --planner vllm_user --model default --json
```

Expected: JSON reports `ok: true`; if it fails, classify the failure and do not start LIBERO.

- [ ] **Step 5: Run VLA/SAM3 JSON-RPC health checks.**

For each configured endpoint, POST `{"method":"healthz","args":[],"kwargs":{}}` to `/call`. Save the redacted JSON responses as preflight evidence and do not treat an ordinary `/health` GET as a valid check.

---

### Task 7: Validate eva launch mapping and fake audit paths

**Files:**
- Modify: `tests/unit/test_rpent_libero.py`
- Create or modify: `tests/unit/test_rpent_vllm_user_integration.py`

**Interfaces:**
- Consumes: the versioned framework declaration, local profile shape, and current `RpentLiberoAdapter`.
- Produces: tests proving that eva launches the current RPent with `--planner vllm_user` and classifies native outcomes from audit evidence.

- [ ] **Step 1: Test the rendered command.**

Assert that the adapter-generated argv contains the current RPent module, `--planner vllm_user`, the explicit model, `--max-turns 20`, `--planner-timeout-s 1200`, LIBERO suite/task/seed/max steps, and configured VLA/SAM3 endpoints.

- [ ] **Step 2: Test successful audit parsing.**

Write a fake native audit with matching suite/task/seed and `terminated: true`; assert `OutcomeStatus.SUCCESS`, `task_success=True`, and `success_source == "rpent.audit.terminated"`.

- [ ] **Step 3: Test task failure and invalid paths.**

Use `terminated: false`, missing `terminated`, and mismatched suite/seed records; assert task failure for the valid false signal and invalid for missing or mismatched native evidence.

- [ ] **Step 4: Test process failure.**

Use a non-zero exit code and assert infrastructure failure with `task_success=None`, while preserving the audit evidence path and native exit code metric.

- [ ] **Step 5: Run the focused integration tests.**

Run:

```bash
cd /share/repos/eva_agentic
./.venv/bin/python -m pytest -q tests/unit/test_rpent_libero.py tests/unit/test_rpent_vllm_user_integration.py
```

---

### Task 8: Execute and audit the real LIBERO smoke run

**Files:**
- Create ignored runtime artifacts under `runs/<new-run-id>/`.
- Do not overwrite an existing run directory.

**Interfaces:**
- Consumes: passing unit tests, patched local RPent environment, successful planner/vLLM preflight, successful VLA/SAM3 health checks, and the fixed one-case experiment.
- Produces: a complete eva run with immutable inputs, actual launch evidence, raw logs, RPent native transcript/audit/recipe artifacts, and a summary whose task result is based on audit `terminated`.

- [ ] **Step 1: Create a fresh run id and freeze inputs.**

Run:

```bash
cd /share/repos/eva_agentic
RUN_ID="rpent-vllm-user-libero-$(date -u +%Y%m%d-%H%M%S)"
./.venv/bin/eva-agentic init \
  --experiment examples/rpent-vllm-user-libero.experiment.example.yaml \
  --runs-root runs \
  --run-id "$RUN_ID"
```

Expected: `inputs/experiment.resolved.json`, `inputs/cases.jsonl`, `inputs/plan.json`, and the input manifest identify exactly one fixed case.

- [ ] **Step 2: Run eva with the dedicated declaration and profile.**

Run:

```bash
./.venv/bin/eva-agentic run \
  --run-dir "runs/$RUN_ID" \
  --frameworks examples/rpent-vllm-user-libero.frameworks.example.yaml \
  --profile profiles/rpent-vllm-user-libero.yaml
```

Expected: eva starts the patched local RPent process, not the correction checkout, and creates one attempt with `launch.json`, `runtime.json`, `stdout.log`, and `stderr.log`.

- [ ] **Step 3: Inspect native evidence.**

Find the audit, transcript, and recipe under the attempt's `native/` directory. Verify that the audit suite/task/seed match the frozen case and that the audit contains a boolean `terminated`.

- [ ] **Step 4: Run the standardized summary.**

Run:

```bash
./.venv/bin/eva-agentic summary --run-dir "runs/$RUN_ID"
```

Confirm that the result status and success source are consistent with the audit, not merely with the process exit code or planner finish result.

- [ ] **Step 5: Record the final verification boundary.**

Report the run directory, exact command/cwd, RPent baseline and patch provenance, planner snapshot provenance, preflight results, preserved artifact paths, audit `terminated` value, and any remaining limitation. If `terminated` is not `true`, keep the run as evidence and report the correct failure/invalid status; do not rerun selectively and overwrite it.

---

## Plan self-review

- The source boundary is covered by Tasks 1 and 6; the correction checkout is never the runtime.
- The official `--planner vllm_user` path is covered by Task 3 and the command test in Task 7.
- Planner migration compatibility is covered by Task 2 rather than by silently copying newer RPent control code.
- Fixed task, seed, initialization, max episode steps, max turns, planner timeout, and model are covered by Tasks 5 and 8.
- Revision/provenance and actual command evidence are covered by Task 4 and Task 8.
- Import, signed vLLM, VLA, and SAM3 checks are covered by Task 6.
- Audit-only success semantics and all required failure classes are covered by Tasks 7 and 8.
- Transcript, stdout, stderr, audit, and recipe preservation are covered by Task 8.
- No step requires an automatic planner update, framework comparison, batch scheduling, or control-loop rewrite.
