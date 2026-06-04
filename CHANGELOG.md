# Changelog

## 0.1.0 (2026-06-04)

First release.

Added:
- Two-stage cascading vector index: a 1-bit graph routes to candidates,
  3-bit TurboQuant codes rerank them, and an exact pass (fp32 or int8)
  settles the final order. Random-rotation preconditioning.
- Configurable routing-code length (`routing_bits`) with a per-dimension
  auto default that widens the code on low-dimensional data.
- Deletes (O(1) tombstones), allowlist-filtered search, and `.tq` save/load.
- Multi-threaded batch search and parallel index build.
- Allocation-free queries (the search path takes a preallocated context).
- C ABI (`include/quantal.h`).
- Python package with an ergonomic `Index`, plus LangChain, LlamaIndex, and
  LangGraph (BaseStore) integrations.
- Prebuilt wheels for common embedding dimensions on Linux (x86_64, aarch64),
  macOS (arm64), and Windows (x86_64); other dimensions build on demand from
  a source checkout.

Requires Zig 0.16.0 to build from source.
