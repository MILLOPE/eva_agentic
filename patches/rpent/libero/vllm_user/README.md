# RPent `vllm_user` planner integration

This bundle adapts the signed `vllm_user` planner from the separate
`/share/repos/Rpent-correction` development workspace to the RPent baseline
vendored by this repository. The runtime framework remains
`third_party/frameworks/rpent`; the correction checkout is never on the
runtime `PYTHONPATH`.

## Scope

The bundle contains:

- a snapshot of `integrations/vllm_user` as an independently installable
  `rpent_vllm_user` package;
- the two provider-neutral RPent planner runtime files required by that
  planner's official factory wrapper;
- a small RPent registration patch exposing `--planner vllm_user` and the
  native signed-vLLM connectivity check.

It does not contain a second RPent tree, LIBERO copy, robot assets, model
weights, or service code. eva owns only launch declarations and evidence
collection; RPent owns the planner, Toolkit, services, control loop,
transcript, audit, and recipe.

## Source and replacement policy

See `source-manifest.yaml` for the source revision and baseline. The planner
snapshot is intentionally manual-replacement friendly: to update it, compare
the desired planner files from the source workspace, replace only the files
under `planner/`, update the manifest, rerun the focused tests, and keep old
run artifacts unchanged. Do not automatically overwrite a prior experiment.

Apply the RPent delta to the local framework checkout only:

```bash
cd /share/repos/eva_agentic
patch -p1 < patches/rpent/libero/vllm_user/rpent-vllm-user.patch
patch -p1 < patches/rpent/libero/sam3-rpc-compat.patch
```

The second, separately recorded patch is a narrowly scoped client
compatibility shim for the already-running remote SAM3 service. That service
uses the legacy bare `segment` RPC method. RPent still tries its native
`sam3.segment` method first and falls back only when the server explicitly
reports an unknown method; inference and transport errors are not masked.

Install the local RPent and this planner snapshot into the dedicated eva
framework environment, using the same interpreter for both:

```bash
python -m pip install -e third_party/frameworks/rpent
python -m pip install -e patches/rpent/libero/vllm_user/planner
```

The `rpent_vllm_user` package must be imported from this snapshot, never from
`/share/repos/Rpent-correction`.

## Required local configuration

Use a local, untracked profile for the service and signed-vLLM settings. The
versioned example names the variables but contains no secret values. Required
names include `RPENT_VLA_ENDPOINT`, `RPENT_SAM3_ENDPOINT`,
`RPENT_VLLM_BASE_URL`, `RPENT_VLLM_KEY_ID`, `RPENT_VLLM_MODEL`, and exactly
one of `RPENT_VLLM_PRIVATE_KEY` or `RPENT_VLLM_SIGNER_URL`. Keep private keys,
signer tokens, machine paths, and service credentials out of source control
and out of launch evidence.

## Verification order

Before a real run, verify planner import origin and native CLI registration,
run `rpent-check-llm --planner vllm_user --json`, and send RPent's JSON-RPC
health probe to the configured VLA and SAM3 `/call` endpoints. Then run one
fixed LIBERO case through eva. The only task-success signal is the native
audit field `terminated == true`; process exit status and planner
`finish_result` are not substitutes.
