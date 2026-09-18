# rpent_vllm_user

This directory is a manually replaceable snapshot of the signed vllm_user
planner package. It is installed alongside the RPent checkout used by eva:

    python -m pip install -e third_party/frameworks/rpent
    python -m pip install -e patches/rpent/libero/vllm_user/planner

The package owns signed vLLM transport, planner history, tool proposal
handling, and optional read-only workspace helpers. RPent continues to own the
Toolkit, robot services, control loop, transcript, audit, and recipe.

The package is adapted to the RPent baseline recorded in the parent
source-manifest.yaml. The planner-local compatibility module handles
observability symbols that are absent from that baseline; the provider-neutral
PlannerRuntime and liveness seam is supplied by the parent RPent patch.

This package must not be imported from the separate correction checkout at
runtime. To update it, replace the planner snapshot manually, update the
source manifest, run the focused tests, and start a new run id.
