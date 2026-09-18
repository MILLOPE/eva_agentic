# Third-party source workspace

This directory contains clean source checkouts used by the evaluator. The
repository records their URL, requested ref, and checked-out commit in
[`sources.yaml`](sources.yaml). Large benchmark assets and model checkpoints do
not belong here.

- `benchmarks/` holds one canonical source tree for each benchmark. Frameworks
  share that source tree.
- `frameworks/` holds each framework's native implementation.
- `models/` holds inference-service implementations such as OpenPI and SAM3.

Do not copy a benchmark to make a framework-specific variant. Put a small,
reviewable patch under `../patches/<framework>/<benchmark>/` and record the
benchmark revision it applies to. Environments remain separate from source:
local profiles select interpreters, checkpoints, GPU slots, and endpoints.
