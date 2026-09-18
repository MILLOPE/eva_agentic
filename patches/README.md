# Framework patches

Use `patches/<framework>/<benchmark>/` for a small patch against the canonical
benchmark source in `third_party/benchmarks/`. Each patch must state the source
revision, its reason, and how it is applied. Do not add a copied benchmark tree.

The same directory may contain a narrowly scoped framework-planner integration
bundle when the framework remains under `third_party/frameworks/`. Such a
bundle must identify the framework baseline, keep the planner independently
replaceable, document any required framework runtime seam, and must not copy a
second framework tree, benchmark tree, model asset, or secret.
