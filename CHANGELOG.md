# Changelog

## 0.1.2 (2026-06-09)

- Single-query `Index.search` now reuses a cached search context and
  output buffers instead of going through the batch path; ~20% more QPS
  at 1M scale through the Python layer.
- `quantal_index_memory_bytes` in the C ABI and `Index.memory_bytes` in
  Python: exact heap accounting for the index (payloads, routing graph,
  rerank store, bookkeeping), as opposed to RSS sampling.

## 0.1.1 (2026-06-09)

- x86-64 wheels now bundle x86-64-v3 (AVX2) and x86-64-v4 (AVX-512)
  variants of each library alongside the portable baseline; the loader
  picks the best one the CPU supports. The baseline-only 0.1.0 wheels
  ran ~8x slower than native on modern x86-64.

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
