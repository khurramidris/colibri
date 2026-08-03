# Lattice Runtime — Provenance Notice

This work is developed inside the `khurramidris/colibri` fork of Colibri. Colibri is distributed
under the Apache License 2.0. The repository's root license and notices continue to apply.

The Lattice sources under `lattice/` were added on the `lattice/runtime-v0.1` branch as a new,
self-contained implementation. They use the following published engineering ideas as design
inputs while preserving a clear implementation boundary:

- **Colibri:** expert streaming, heterogeneous residency, route telemetry and accelerator tiers.
- **FareedKhan-dev/kimi-k3-in-c:** the exact dense-trunk model of a pinned prefix plus a streaming
  ring, native packed expert execution, strict low-memory validation and explicit byte accounting.
- **Recent inference research:** tensor-level placement, request-local expert prediction,
  proactive/on-demand/speculative transfer and token-adaptive width pruning.

No third-party source file has been copied into `lattice/` by this milestone. The dense-trunk
planner and block-addressable expert format are independent implementations behind new APIs.
Any future direct code import must be recorded here with source repository, commit, files,
license and modifications.

Model weights remain governed by their own model licenses and are not redistributed by Lattice.
Generated manifests and packed files contain model-derived metadata or weight bytes and must be
handled under the corresponding model license.
